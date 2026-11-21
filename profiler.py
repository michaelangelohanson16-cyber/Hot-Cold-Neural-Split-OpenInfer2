"""
desdemona_profiler.py
Offline neuron activation profiler for hot/cold weight classification.

Registers forward hooks on every FFN intermediate activation in a
LLaMA-style model, runs inference over a text corpus, and records
per-neuron activation frequency.

Supports:
  - Dense models (LLaMA, Mistral, Qwen, Phi-3 style)
  - MoE models (Mixtral style) where each layer has N expert FFNs

Output: profile.json
  {
    "arch":          "llama" | "mixtral",
    "hidden_dim":    int,
    "num_layers":    int,
    "intermediate_dim": int,
    "num_experts":   int | null,
    "total_tokens":  int,
    "layers": {
      "<layer_idx>": {
        "expert_<eid>": [float, ...]   # activation freq per neuron
      }
    }
  }

Usage:
    python desdemona_profiler.py \
        --model  /path/to/model_or_hf_id \
        --corpus /path/to/corpus_dir \
        --output profile.json \
        --tokens 10_000_000 \
        --threshold 0.10
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from magnitude import MassAccumulator, save_thresholds, load_thresholds  # noqa: F401

# ─── Architecture detection ───────────────────────────────────────────────────

def detect_arch(model) -> str:
    cls = type(model).__name__.lower()
    if "bamboo" in cls:
        return "bamboo"     # TurboSparse/Bamboo dReLU family
    if "mixtral" in cls:
        return "mixtral"
    if any(k in cls for k in ("llama", "mistral", "qwen", "phi")):
        return "llama"
    # Fallback: inspect first layer
    try:
        layer = model.model.layers[0]
        if hasattr(layer.mlp, "experts"):
            return "mixtral"
    except AttributeError:
        pass
    return "llama"


# ─── Profiler ─────────────────────────────────────────────────────────────────

class NeuronProfiler:
    """
    Registers hooks on every FFN and accumulates per-neuron activation
    statistics across the profiling corpus.

    "Fired" is a MAGNITUDE criterion (magnitude.py), not the old sign test.
    The old hook counted silu(gate) > 0 as fired; on Hermes-3-Llama-3.1-8B
    that keeps 23.9% of neurons while the 99%-mass set needs 82.6%, so it
    dropped real work (desdemona-x3 docs/OPENINFER_STEP0_RESULTS.md s.2).
    Now, per (layer, expert):
      - |a| mass per neuron, with a = down_proj's input = act(g)*u
        (act(g)*act(u) for dReLU), and the per-token share of neurons that
        carries 90/95/99% of mass (k90/k95/k99);
      - a reservoir of |act(g)| from which per-layer CATS thresholds tau_L
        are calibrated (arXiv 2404.08763) for any target sparsity;
      - per-neuron active frequency under |act(g)| >= tau_L, which is what
        splitter.py ranks hot neurons by;
      - the legacy >0 share, reported only, so old numbers stay comparable.
    """

    def __init__(self, model: nn.Module, arch: str, thresholds: Optional[dict] = None):
        self.model = model
        self.arch = arch
        self.hooks: list = []
        # {layer_idx: {expert_idx: MassAccumulator}}
        self.acc: dict[int, dict[int, MassAccumulator]] = {}
        self._pending_act: dict[tuple[int, int], torch.Tensor] = {}
        # thresholds: {layer_idx: {"0.40": tau, ...}} from a previous calibration
        self.thresholds = thresholds or {}
        self._register_hooks()

    # Kept for callers of the old attribute names.
    @property
    def activation_counts(self):
        return {li: {ei: a.mass for ei, a in ex.items()} for li, ex in self.acc.items()}

    @property
    def token_counts(self):
        return {li: {ei: a.tokens for ei, a in ex.items()} for li, ex in self.acc.items()}

    # ── hook registration ──────────────────────────────────────────────────

    def _register_hooks(self):
        layers = self.model.model.layers
        for layer_idx, layer in enumerate(layers):
            if self.arch == "mixtral":
                for ei, expert in enumerate(layer.block_sparse_moe.experts):
                    self._register_ffn_hooks(layer_idx, ei, expert)
            else:
                self._register_ffn_hooks(layer_idx, 0, layer.mlp)

    def _accumulator(self, layer_idx: int, expert_idx: int, I: int) -> MassAccumulator:
        ex = self.acc.setdefault(layer_idx, {})
        if expert_idx not in ex:
            ex[expert_idx] = MassAccumulator(I, seed=layer_idx * 1000 + expert_idx)
            if layer_idx in self.thresholds:
                ex[expert_idx].set_thresholds(self.thresholds[layer_idx])
        return ex[expert_idx]

    def _make_act_hook(self, layer_idx: int, expert_idx: int):
        """Stores act(gate) until the matching down_proj input arrives."""
        def hook(module, inputs, output: torch.Tensor):
            self._pending_act[(layer_idx, expert_idx)] = output.detach()
        return hook

    def _make_down_pre_hook(self, layer_idx: int, expert_idx: int, gate_from_act: bool):
        """down_proj's INPUT is the true per-neuron contribution a. For dReLU/
        Bamboo the shared act_fn fires twice per forward (once per branch) and
        a built-in mask zeroes neurons afterwards, so a itself is used as the
        gate signal there (|a| > 0 iff the neuron contributes)."""
        def hook(module, inputs):
            a = inputs[0].detach()
            act_g = self._pending_act.pop((layer_idx, expert_idx), None) if gate_from_act else None
            if act_g is None or act_g.shape != a.shape:
                act_g = a
            self._accumulator(layer_idx, expert_idx, a.shape[-1]).feed(act_g, a)
        return hook

    def _register_ffn_hooks(self, layer_idx: int, expert_idx: int, mlp):
        target = getattr(mlp, "act_fn", None) or getattr(mlp, "activation_fn", None)
        gate_from_act = self.arch != "bamboo" and target is not None
        if gate_from_act:
            self.hooks.append(target.register_forward_hook(self._make_act_hook(layer_idx, expert_idx)))
        down = getattr(mlp, "down_proj", None) or getattr(mlp, "w2", None)
        if down is None:
            raise ValueError(f"layer {layer_idx}: FFN has no down_proj/w2 to read activations from")
        self.hooks.append(down.register_forward_pre_hook(
            self._make_down_pre_hook(layer_idx, expert_idx, gate_from_act)))

    # ── profiling loop ─────────────────────────────────────────────────────

    def profile_corpus(
        self,
        tokenizer,
        corpus_dir: str,
        max_tokens: int = 10_000_000,
        chunk_size: int = 512,
        device: str = "cuda",
    ):
        """
        Iterate over .txt files in corpus_dir, tokenize in chunks,
        run forward passes until max_tokens profiling tokens consumed.
        """
        total_seen = 0
        corpus_path = Path(corpus_dir)
        files = sorted(corpus_path.glob("**/*.txt"))

        if not files:
            raise FileNotFoundError(f"No .txt files found in {corpus_dir}")

        print(f"Profiling corpus: {len(files)} files, target {max_tokens:,} tokens")
        t0 = time.time()

        for filepath in files:
            if total_seen >= max_tokens:
                break

            text = filepath.read_text(encoding="utf-8", errors="replace")
            tokens = tokenizer.encode(text, add_special_tokens=False)

            for i in range(0, len(tokens), chunk_size):
                if total_seen >= max_tokens:
                    break

                chunk = tokens[i : i + chunk_size]
                input_ids = torch.tensor([chunk], dtype=torch.long).to(device)

                with torch.no_grad():
                    self.model(input_ids=input_ids)

                total_seen += len(chunk)

                if total_seen % 100_000 < chunk_size:
                    elapsed = time.time() - t0
                    rate = total_seen / elapsed
                    eta = (max_tokens - total_seen) / rate
                    print(f"  {total_seen:>10,} / {max_tokens:,} tokens  "
                          f"({rate:,.0f} tok/s, ETA {eta/60:.1f}m)")

        print(f"Profiling complete: {total_seen:,} tokens in "
              f"{(time.time()-t0)/60:.1f}m")
        return total_seen

    # ── output ─────────────────────────────────────────────────────────────

    def calibrate_thresholds(self, sparsities=(0.2, 0.3, 0.4, 0.5, 0.6)) -> dict:
        """Per-layer CATS tau for each target sparsity: {layer: {"0.40": tau}}."""
        return {li: ex[0].calibrate(sparsities) for li, ex in self.acc.items() if 0 in ex}

    def get_activation_frequencies(self, threshold_key: Optional[str] = None) -> dict:
        """Per-neuron active frequency under the magnitude criterion
        |act(g)| >= tau_L[threshold_key]; without thresholds, normalised |a|
        mass (a magnitude ranking, never the old sign count)."""
        return {li: {ei: a.frequencies(threshold_key) for ei, a in ex.items()}
                for li, ex in self.acc.items()}

    def save_profile(self, path: str, model_config=None, threshold_key: Optional[str] = None):
        cfg = model_config or {}
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        freqs = self.get_activation_frequencies(threshold_key)

        all_freqs = [f for ex in freqs.values() for fl in ex.values() for f in fl]
        t = torch.tensor(all_freqs) if all_freqs else torch.zeros(1)
        per_layer = {str(li): {f"expert_{ei}": a.summary() for ei, a in ex.items()}
                     for li, ex in self.acc.items()}
        summary = {
            "criterion":              f"|act(g)| >= tau_L[{threshold_key}]" if threshold_key
                                      else "normalised |act(g)*u| mass",
            "mean_activation_rate":   float(t.mean()),
            "median_activation_rate": float(t.median()),
            "p10":  float(t.quantile(0.10)) if t.numel() <= 16_000_000 else None,
            "p90":  float(t.quantile(0.90)) if t.numel() <= 16_000_000 else None,
        }
        for k in ("pos_frac_legacy", "k90", "k95", "k99"):
            vals = [v[k] for ex in per_layer.values() for v in ex.values() if k in v]
            if vals:
                summary[k] = sum(vals) / len(vals)

        total_tokens = max(sum(a.tokens for ex in self.acc.values() for a in ex.values()), 1)
        profile = {
            "arch":             self.arch,
            "hidden_dim":       cfg.get("hidden_size", None),
            "num_layers":       len(freqs),
            "intermediate_dim": cfg.get("intermediate_size", None),
            "num_experts":      cfg.get("num_local_experts", None),
            "total_tokens":     total_tokens,
            "summary":          summary,
            "layer_stats":      per_layer,
            "layers":           {
                str(li): {f"expert_{ei}": fl for ei, fl in ex.items()}
                for li, ex in freqs.items()
            },
        }
        with open(path, "w") as f:
            json.dump(profile, f, separators=(",", ":"))
        print(f"Profile saved → {path}  ({os.path.getsize(path)/1e6:.1f} MB)")

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─── Hot/cold classification (post-profiling) ─────────────────────────────────

def classify_neurons(
    profile_path: str,
    hot_count: int = 128,
) -> dict:
    """
    Read a saved profile and classify each neuron as hot (top-N by activation freq)
    or cold.  hot_count=128 keeps ~4.25 GB of hot weights within the 8 GB VRAM budget.

    Returns:
        {
          "layer_<i>": {
            "expert_<j>": {
              "hot":  [neuron_indices...],
              "cold": [neuron_indices...]
            }
          }
        }
    """
    with open(profile_path) as f:
        profile = json.load(f)

    result = {}
    total_neurons = 0
    total_hot = 0

    for layer_str, experts in profile["layers"].items():
        layer_key = f"layer_{layer_str}"
        result[layer_key] = {}
        for expert_str, freqs in experts.items():
            n = len(freqs)
            k = min(hot_count, n)
            # top-k indices by activation frequency
            hot_set = set(sorted(range(n), key=lambda i: freqs[i], reverse=True)[:k])
            hot  = [i for i in range(n) if i in hot_set]
            cold = [i for i in range(n) if i not in hot_set]
            result[layer_key][expert_str] = {"hot": hot, "cold": cold}
            total_neurons += n
            total_hot += len(hot)

    if total_neurons > 0:
        print(f"Classification @ hot_count={hot_count}:")
        print(f"  Hot  neurons: {total_hot:,}  ({100*total_hot/total_neurons:.2f}%)")
        print(f"  Cold neurons: {total_neurons - total_hot:,}  "
              f"({100*(total_neurons-total_hot)/total_neurons:.2f}%)")

    return result


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Desdemona neuron profiler")
    parser.add_argument("--model",     required=True,
                        help="Model path or HuggingFace model id")
    parser.add_argument("--corpus",    required=True,
                        help="Directory of .txt files to profile on")
    parser.add_argument("--output",    default="profile.json",
                        help="Output profile path (default: profile.json)")
    parser.add_argument("--tokens",    type=int, default=10_000_000,
                        help="Target profiling tokens (default: 10M)")
    parser.add_argument("--chunk",     type=int, default=512,
                        help="Tokens per forward pass (default: 512)")
    parser.add_argument("--hot-count", type=int, default=128,
                        help="Top-N neurons per (layer, expert) to classify as hot (default: 128)")
    parser.add_argument("--dtype",     default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model load dtype (default: float16)")
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--thresholds", default=None,
                        help="thresholds.json from an earlier run; enables per-neuron "
                             "active frequency under |act(g)| >= tau_L")
    parser.add_argument("--sparsity",  type=float, default=0.40,
                        help="Target sparsity whose tau_L ranks neurons (default 0.40)")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Needed for checkpoints with custom modeling code "
                             "(e.g. TurboSparse/Bamboo)")
    args = parser.parse_args()

    dtype_map = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    arch = detect_arch(model)
    print(f"Architecture detected: {arch}")

    sparsities = (0.2, 0.3, 0.4, 0.5, 0.6)
    key = f"{args.sparsity:.2f}"
    thr = None
    if args.thresholds:
        with open(args.thresholds) as f:
            thr = {int(k): v for k, v in json.load(f)["layers"].items()}
    profiler = NeuronProfiler(model, arch, thresholds=thr)

    try:
        profiler.profile_corpus(
            tokenizer=tokenizer,
            corpus_dir=args.corpus,
            max_tokens=args.tokens,
            chunk_size=args.chunk,
            device=args.device,
        )
    finally:
        profiler.remove_hooks()

    cfg = model.config.to_dict() if hasattr(model, "config") else {}
    taus = profiler.calibrate_thresholds(sparsities)
    thr_out = os.path.join(os.path.dirname(args.output) or ".", "thresholds.json")
    if not args.thresholds:
        save_thresholds(thr_out, taus, {"corpus": args.corpus, "criterion": "|act(g)| quantile (CATS)"})
        print(f"Thresholds saved → {thr_out}. Re-run with --thresholds {thr_out} to rank "
              f"neurons by active frequency at sparsity {key}; this profile ranks by |a| mass.")
    profiler.save_profile(args.output, cfg, threshold_key=key if thr else None)

    classify_neurons(args.output, hot_count=args.hot_count)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
