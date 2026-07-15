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
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    Registers hooks on FFN gate/activation outputs and accumulates
    per-neuron activation counts across the profiling corpus.
    """

    def __init__(self, model: nn.Module, arch: str):
        self.model = model
        self.arch = arch
        self.hooks: list = []

        # {layer_idx: {expert_idx: Tensor(intermediate_dim)}}
        self.activation_counts: dict[int, dict[int, torch.Tensor]] = {}
        self.token_counts: dict[int, dict[int, int]] = {}

        self._register_hooks()

    # ── hook registration ──────────────────────────────────────────────────

    def _register_hooks(self):
        layers = self.model.model.layers
        for layer_idx, layer in enumerate(layers):
            if self.arch == "mixtral":
                self._register_moe_hooks(layer_idx, layer)
            else:
                self._register_dense_hooks(layer_idx, layer)

    def _make_hook(self, layer_idx: int, expert_idx: int):
        """Returns a forward hook that accumulates activation counts."""
        def hook(module, input, output: torch.Tensor):
            # output: (batch, seq_len, intermediate_dim) after activation fn
            # Count neurons where output > 0 (fired)
            fired = (output.detach() > 0).float()          # (B, T, D)
            counts = fired.sum(dim=(0, 1)).cpu()            # (D,)
            tokens = output.shape[0] * output.shape[1]

            if layer_idx not in self.activation_counts:
                self.activation_counts[layer_idx] = {}
                self.token_counts[layer_idx] = {}

            if expert_idx not in self.activation_counts[layer_idx]:
                self.activation_counts[layer_idx][expert_idx] = \
                    torch.zeros(output.shape[-1])
                self.token_counts[layer_idx][expert_idx] = 0

            self.activation_counts[layer_idx][expert_idx] += counts
            self.token_counts[layer_idx][expert_idx] += tokens

        return hook

    def _make_down_pre_hook(self, layer_idx: int, expert_idx: int):
        """
        Counts nonzero entries of down_proj's INPUT — the actual per-neuron
        contribution pattern. Needed for dReLU/Bamboo models, whose shared
        act_fn module fires twice per forward (once per branch) and whose
        built-in predictor mask zeroes neurons after activation; hooking
        act_fn there would double-count and miss the mask.
        """
        def hook(module, inputs):
            h_in = inputs[0]                                  # (B, T, D)
            fired  = (h_in.detach() != 0).float()
            counts = fired.sum(dim=(0, 1)).cpu()
            tokens = h_in.shape[0] * h_in.shape[1]

            self.activation_counts.setdefault(layer_idx, {})
            self.token_counts.setdefault(layer_idx, {})
            if expert_idx not in self.activation_counts[layer_idx]:
                self.activation_counts[layer_idx][expert_idx] = \
                    torch.zeros(h_in.shape[-1])
                self.token_counts[layer_idx][expert_idx] = 0

            self.activation_counts[layer_idx][expert_idx] += counts
            self.token_counts[layer_idx][expert_idx] += tokens
        return hook

    def _register_dense_hooks(self, layer_idx: int, layer):
        mlp = layer.mlp

        if self.arch == "bamboo":
            h = mlp.down_proj.register_forward_pre_hook(
                self._make_down_pre_hook(layer_idx, 0)
            )
            self.hooks.append(h)
            return

        # LLaMA / Mistral uses SwiGLU: act_fn applied to gate_proj output
        # The intermediate activation lives in mlp.act_fn or we hook the
        # post-activation by wrapping forward.  Safest: hook act_fn directly.
        target = None
        if hasattr(mlp, "act_fn"):
            target = mlp.act_fn
        elif hasattr(mlp, "activation_fn"):
            target = mlp.activation_fn
        else:
            # Fallback: hook the whole MLP forward, capture gate * up product
            target = mlp

        h = target.register_forward_hook(self._make_hook(layer_idx, 0))
        self.hooks.append(h)

    def _register_moe_hooks(self, layer_idx: int, layer):
        for expert_idx, expert in enumerate(layer.block_sparse_moe.experts):
            mlp = expert
            target = None
            if hasattr(mlp, "act_fn"):
                target = mlp.act_fn
            elif hasattr(mlp, "activation_fn"):
                target = mlp.activation_fn
            else:
                target = mlp

            h = target.register_forward_hook(
                self._make_hook(layer_idx, expert_idx)
            )
            self.hooks.append(h)

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

    def get_activation_frequencies(self) -> dict:
        """
        Returns per-neuron activation frequency (fraction of tokens where
        neuron fired) for every (layer, expert) pair.
        """
        freqs: dict[int, dict[int, list[float]]] = {}
        for layer_idx, experts in self.activation_counts.items():
            freqs[layer_idx] = {}
            for expert_idx, counts in experts.items():
                n_tokens = self.token_counts[layer_idx][expert_idx]
                freq = (counts / n_tokens).tolist() if n_tokens > 0 else []
                freqs[layer_idx][expert_idx] = freq
        return freqs

    def save_profile(self, path: str, model_config=None):
        cfg = model_config or {}
        # Create the output directory if the caller pointed at a nested path
        # (e.g. out/profile.json) that doesn't exist yet.
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        freqs = self.get_activation_frequencies()

        # Compute summary stats
        all_freqs = []
        for layer_experts in freqs.values():
            for freq_list in layer_experts.values():
                all_freqs.extend(freq_list)

        t = torch.tensor(all_freqs)
        summary = {
            "mean_activation_rate":   float(t.mean()),
            "median_activation_rate": float(t.median()),
            "p10":  float(t.quantile(0.10)),
            "p25":  float(t.quantile(0.25)),
            "p75":  float(t.quantile(0.75)),
            "p90":  float(t.quantile(0.90)),
        }

        total_tokens = max(
            sum(v for layer in self.token_counts.values() for v in layer.values()),
            1
        )

        profile = {
            "arch":             self.arch,
            "hidden_dim":       cfg.get("hidden_size", None),
            "num_layers":       len(freqs),
            "intermediate_dim": cfg.get("intermediate_size", None),
            "num_experts":      cfg.get("num_local_experts", None),
            "total_tokens":     total_tokens,
            "summary":          summary,
            "layers":           {
                str(li): {
                    f"expert_{ei}": freq_list
                    for ei, freq_list in experts.items()
                }
                for li, experts in freqs.items()
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
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Needed for checkpoints with custom modeling code "
                             "(e.g. TurboSparse/Bamboo)")
    args = parser.parse_args()

    dtype_map = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }

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

    profiler = NeuronProfiler(model, arch)

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
    profiler.save_profile(args.output, cfg)

    classify_neurons(args.output, hot_count=args.hot_count)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
