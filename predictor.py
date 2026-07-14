"""
desdemona_predictor.py
Neuron activation predictor — the prefetch oracle.

Given the hidden state just before a FFN layer, predicts which cold neurons
will activate so the inference engine can issue NVMe prefetch requests before
computing.

Architecture:
    hidden_state (hidden_dim,)
        → down_proj (hidden_dim → rank)       [low-rank bottleneck]
        → up_proj   (rank → intermediate_dim) [per-neuron logit]
        → sigmoid                              [activation probability]

Why low-rank: at rank=64 the predictor is ~425K params per (layer, expert)
pair — measured at ~0.4 ms per batch-1 forward on an RTX 5070 Laptop GPU
and ~0.3 ms on CPU (kernel-launch overhead dominates at this size; see
BENCHMARKS.md), and small enough to keep all predictors resident in VRAM.

Training data: collected during profiling via ProfilingDataCollector, which
records (hidden_state, activation_mask) pairs for each (layer, expert).

Usage — collect training data during a profiling run:
    collector = ProfilingDataCollector(model, arch)
    collector.collect(tokenizer, corpus_dir, max_tokens=1_000_000)
    collector.save("predictor_data/")

Usage — train predictors:
    trainer = PredictorTrainer("predictor_data/", "predictors/")
    trainer.train_all(epochs=3, batch_size=256)

Usage — inference:
    predictor = NeuronPredictor.load("predictors/layer_0_expert_0.pt")
    probs = predictor(hidden_state)           # (intermediate_dim,)
    predicted_mask = probs > 0.5              # threshold
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─── Model ────────────────────────────────────────────────────────────────────

class NeuronPredictor(nn.Module):
    """
    Low-rank linear predictor: hidden_dim → rank → intermediate_dim.
    One instance per (layer, expert) pair.
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int, rank: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.rank = rank

        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up   = nn.Linear(rank, intermediate_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., hidden_dim)
        returns: (..., intermediate_dim) — probability each neuron activates
        """
        return torch.sigmoid(self.up(self.down(x)))

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Returns boolean mask of predicted-active neurons."""
        return self.forward(x) >= threshold

    # ── serialisation ──────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "hidden_dim":       self.hidden_dim,
            "intermediate_dim": self.intermediate_dim,
            "rank":             self.rank,
            "state_dict":       self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "NeuronPredictor":
        ckpt = torch.load(path, map_location=device)
        model = cls(ckpt["hidden_dim"], ckpt["intermediate_dim"], ckpt["rank"])
        model.load_state_dict(ckpt["state_dict"])
        return model.to(device).eval()


# ─── Training data collection ─────────────────────────────────────────────────

class ProfilingDataCollector:
    """
    Registers hooks that capture (hidden_state, activation_mask) pairs
    during inference.  Hidden state = input to the FFN layer.
    Activation mask = which neurons fired after the activation function.

    Saves collected tensors to disk as memory-mapped .pt files, one file
    per (layer, expert) pair.  Large datasets need not fit in RAM.
    """

    def __init__(self, model: nn.Module, arch: str, output_dir: str):
        self.model = model
        self.arch = arch
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # In-memory buffer before flushing to disk
        # {(layer_idx, expert_idx): {"hidden": [tensors], "mask": [tensors]}}
        self.buffers: dict = {}
        self.hooks: list = []
        self._flush_every = 5_000          # flush to disk every N samples
        self._total_samples = 0

        self._register_hooks()

    # ── hooks ──────────────────────────────────────────────────────────────

    def _make_hidden_hook(self, layer_idx: int, expert_idx: int):
        """Captures input to the FFN (hidden state)."""
        def hook(module, input, output):
            # input[0]: (batch, seq_len, hidden_dim)
            h = input[0].detach().cpu().reshape(-1, input[0].shape[-1])  # (B*T, H)
            key = (layer_idx, expert_idx)
            if key not in self.buffers:
                self.buffers[key] = {"hidden": [], "mask": []}
            self.buffers[key]["hidden"].append(h)
        return hook

    def _make_activation_hook(self, layer_idx: int, expert_idx: int):
        """Captures post-activation output (which neurons fired)."""
        def hook(module, input, output):
            # output: (batch, seq_len, intermediate_dim)
            mask = (output.detach().cpu() > 0).reshape(
                -1, output.shape[-1]
            )                                              # (B*T, D) bool
            key = (layer_idx, expert_idx)
            if key not in self.buffers:
                self.buffers[key] = {"hidden": [], "mask": []}
            self.buffers[key]["mask"].append(mask)

            self._total_samples += mask.shape[0]
            if self._total_samples % self._flush_every < mask.shape[0]:
                self._flush()
        return hook

    def _make_down_mask_hook(self, layer_idx: int, expert_idx: int):
        """
        dReLU/Bamboo variant: read the fired mask off down_proj's INPUT.
        Their shared act_fn module fires twice per forward (gate and up
        branches) and a built-in mask zeroes neurons afterwards — hooking
        act_fn would double-record and mislabel; down_proj's input is the
        ground-truth per-neuron contribution.
        """
        def hook(module, inputs):
            h_in = inputs[0]
            mask = (h_in.detach().cpu() != 0).reshape(-1, h_in.shape[-1])
            key = (layer_idx, expert_idx)
            if key not in self.buffers:
                self.buffers[key] = {"hidden": [], "mask": []}
            self.buffers[key]["mask"].append(mask)

            self._total_samples += mask.shape[0]
            if self._total_samples % self._flush_every < mask.shape[0]:
                self._flush()
        return hook

    def _register_hooks(self):
        layers = self.model.model.layers
        for li, layer in enumerate(layers):
            if self.arch == "mixtral":
                experts = layer.block_sparse_moe.experts
            else:
                experts = [layer.mlp]

            for ei, expert in enumerate(experts):
                mlp = expert if self.arch == "mixtral" else layer.mlp

                # Hook 1: hidden state (input to FFN)
                h1 = mlp.register_forward_hook(
                    self._make_hidden_hook(li, ei)
                )

                # Hook 2: which neurons fired
                if self.arch == "bamboo":
                    h2 = mlp.down_proj.register_forward_pre_hook(
                        self._make_down_mask_hook(li, ei)
                    )
                else:
                    target = None
                    if hasattr(mlp, "act_fn"):
                        target = mlp.act_fn
                    elif hasattr(mlp, "activation_fn"):
                        target = mlp.activation_fn
                    else:
                        target = mlp
                    h2 = target.register_forward_hook(
                        self._make_activation_hook(li, ei)
                    )

                self.hooks += [h1, h2]

    # ── data management ────────────────────────────────────────────────────

    def _flush(self):
        """Append buffered tensors to per-(layer, expert) .pt files."""
        for (li, ei), data in self.buffers.items():
            if not data["hidden"] or not data["mask"]:
                continue

            hidden = torch.cat(data["hidden"], dim=0)   # (N, H)
            mask   = torch.cat(data["mask"],   dim=0)   # (N, D)

            # Trim to same length (hidden captured once per token, mask same)
            n = min(hidden.shape[0], mask.shape[0])
            hidden, mask = hidden[:n], mask[:n]

            outfile = self.output_dir / f"layer_{li:03d}_expert_{ei:03d}.pt"

            if outfile.exists():
                existing = torch.load(outfile, map_location="cpu")
                hidden = torch.cat([existing["hidden"], hidden], dim=0)
                mask   = torch.cat([existing["mask"],   mask],   dim=0)

            torch.save({"hidden": hidden, "mask": mask}, outfile)

        # Clear buffers
        self.buffers = {}

    def collect(
        self,
        tokenizer,
        corpus_dir: str,
        max_tokens: int = 1_000_000,
        chunk_size: int = 256,
        device: str = "cuda",
    ):
        files = sorted(Path(corpus_dir).glob("**/*.txt"))
        total = 0
        for f in files:
            if total >= max_tokens:
                break
            text = f.read_text(encoding="utf-8", errors="replace")
            ids  = tokenizer.encode(text, add_special_tokens=False)
            for i in range(0, len(ids), chunk_size):
                if total >= max_tokens:
                    break
                chunk = ids[i : i + chunk_size]
                inp = torch.tensor([chunk]).to(device)
                with torch.no_grad():
                    self.model(input_ids=inp)
                total += len(chunk)
        self._flush()
        print(f"Collected {total:,} tokens → {self.output_dir}")
        self.remove_hooks()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─── Trainer ──────────────────────────────────────────────────────────────────

class PredictorTrainer:
    """
    Trains one NeuronPredictor per (layer, expert) pair from the
    data files produced by ProfilingDataCollector.
    """

    def __init__(
        self,
        data_dir: str,
        predictor_dir: str,
        rank: int = 64,
        device: str = "cuda",
    ):
        self.data_dir      = Path(data_dir)
        self.predictor_dir = Path(predictor_dir)
        self.predictor_dir.mkdir(parents=True, exist_ok=True)
        self.rank   = rank
        self.device = device

    def train_one(
        self,
        data_file: Path,
        epochs: int = 3,
        batch_size: int = 256,
        lr: float = 1e-3,
    ) -> dict:
        data     = torch.load(data_file, map_location="cpu")
        hidden   = data["hidden"].float()           # (N, H)
        mask     = data["mask"].float()             # (N, D)  0/1

        hidden_dim       = hidden.shape[1]
        intermediate_dim = mask.shape[1]

        predictor = NeuronPredictor(hidden_dim, intermediate_dim, self.rank)
        predictor = predictor.to(self.device)

        dataset = TensorDataset(hidden, mask)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             num_workers=2, pin_memory=True)

        optim = torch.optim.AdamW(predictor.parameters(), lr=lr)

        metrics = {}
        for epoch in range(epochs):
            total_loss = 0.0
            total_tp = total_fp = total_fn = total_tn = 0

            for h_batch, m_batch in loader:
                h_batch = h_batch.to(self.device)
                m_batch = m_batch.to(self.device)

                logits = predictor.up(predictor.down(h_batch))   # skip sigmoid for BCE loss
                loss   = F.binary_cross_entropy_with_logits(logits, m_batch)

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

                total_loss += loss.item() * h_batch.shape[0]

                with torch.no_grad():
                    pred = (torch.sigmoid(logits) >= 0.5)
                    tgt  = m_batch.bool()
                    total_tp += (pred &  tgt).sum().item()
                    total_fp += (pred & ~tgt).sum().item()
                    total_fn += (~pred &  tgt).sum().item()
                    total_tn += (~pred & ~tgt).sum().item()

            n = len(dataset)
            prec    = total_tp / (total_tp + total_fp + 1e-8)
            recall  = total_tp / (total_tp + total_fn + 1e-8)
            f1      = 2 * prec * recall / (prec + recall + 1e-8)
            # Fraction of actual activations we would fetch (recall)
            # Fraction of fetches that are wasted (1 - precision)
            metrics[f"epoch_{epoch+1}"] = {
                "loss":       total_loss / n,
                "precision":  prec,
                "recall":     recall,
                "f1":         f1,
                "waste_rate": 1.0 - prec,   # fraction of fetched neurons that didn't fire
                "miss_rate":  1.0 - recall,  # fraction of fired neurons we didn't fetch
            }
            print(f"  epoch {epoch+1}/{epochs}  loss={total_loss/n:.4f}  "
                  f"precision={prec:.3f}  recall={recall:.3f}  f1={f1:.3f}  "
                  f"waste={1-prec:.3f}  miss={1-recall:.3f}")

        return predictor, metrics

    def train_all(self, epochs: int = 3, batch_size: int = 256, lr: float = 1e-3):
        data_files = sorted(self.data_dir.glob("layer_*_expert_*.pt"))
        print(f"Training predictors for {len(data_files)} (layer, expert) pairs")

        all_metrics = {}
        for i, df in enumerate(data_files):
            name = df.stem
            print(f"\n[{i+1}/{len(data_files)}] {name}")
            predictor, metrics = self.train_one(df, epochs, batch_size, lr)

            out_path = self.predictor_dir / f"{name}.pt"
            predictor.save(str(out_path))
            all_metrics[name] = metrics

        # Save aggregate metrics
        metrics_path = self.predictor_dir / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nAll predictors saved → {self.predictor_dir}")
        print(f"Metrics → {metrics_path}")

        # Print summary
        self._print_summary(all_metrics)

    def _print_summary(self, all_metrics: dict):
        final_metrics = []
        for name, m in all_metrics.items():
            last = list(m.values())[-1]
            final_metrics.append(last)

        avg_prec   = sum(m["precision"]  for m in final_metrics) / len(final_metrics)
        avg_recall = sum(m["recall"]     for m in final_metrics) / len(final_metrics)
        avg_waste  = sum(m["waste_rate"] for m in final_metrics) / len(final_metrics)
        avg_miss   = sum(m["miss_rate"]  for m in final_metrics) / len(final_metrics)

        print(f"\n── Predictor Summary ─────────────────────────────────────")
        print(f"  Avg precision:  {avg_prec:.3f}  (fetched neurons that fired)")
        print(f"  Avg recall:     {avg_recall:.3f}  (fired neurons that were fetched)")
        print(f"  Avg waste rate: {avg_waste:.3f}  (fraction of fetches wasted)")
        print(f"  Avg miss rate:  {avg_miss:.3f}  (fraction of firings missed)")
        print(f"\n  Effective fetch reduction vs naive (load all cold):")
        # If sparsity is S (fraction of neurons that fire) and recall is R,
        # effective fetch = S * R + (1-S) * waste
        # Naive fetch = all cold neurons = 1.0
        # We print this once we know S from profiling.
        print(f"  (Run with --profile to compute vs profiled sparsity)")
        print(f"──────────────────────────────────────────────────────────")


# ─── Predictor registry (inference time) ──────────────────────────────────────

class PredictorRegistry:
    """
    Loads all trained predictors into VRAM and provides fast lookup
    by (layer_idx, expert_idx) for use in the inference engine.
    """

    def __init__(self, predictor_dir: str, device: str = "cuda"):
        self.predictors: dict[tuple[int,int], NeuronPredictor] = {}
        self.device = device
        self._load_all(Path(predictor_dir))

    def _load_all(self, predictor_dir: Path):
        files = sorted(predictor_dir.glob("layer_*_expert_*.pt"))
        print(f"Loading {len(files)} neuron predictors into {self.device}...")
        for f in files:
            # Parse layer and expert indices from filename
            parts = f.stem.split("_")   # ["layer", "003", "expert", "001"]
            li = int(parts[1])
            ei = int(parts[3])
            self.predictors[(li, ei)] = NeuronPredictor.load(str(f), self.device)
        print(f"  → {len(self.predictors)} predictors resident in VRAM")

    def predict(
        self,
        layer_idx: int,
        expert_idx: int,
        hidden_state: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Returns boolean mask of predicted-active neurons for this
        (layer, expert) pair given the current hidden state.
        hidden_state: (hidden_dim,) or (batch, hidden_dim)
        """
        predictor = self.predictors.get((layer_idx, expert_idx))
        if predictor is None:
            return None
        with torch.no_grad():
            # The engine runs its hidden states at the model's compute dtype
            # (e.g. float16), but predictors are trained and stored in float32.
            # Cast at this boundary so the matmul dtypes agree regardless of
            # what precision the caller is running.
            hidden_state = hidden_state.to(next(predictor.parameters()).dtype)
            return predictor.predict_mask(hidden_state, threshold)

    def vram_usage_mb(self) -> float:
        total = 0
        for p in self.predictors.values():
            total += sum(par.numel() * par.element_size() for par in p.parameters())
        return total / 1e6


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Desdemona neuron predictor")
    sub = parser.add_subparsers(dest="cmd")

    # collect
    c = sub.add_parser("collect", help="Collect (hidden, activation) training data")
    c.add_argument("--model",   required=True)
    c.add_argument("--corpus",  required=True)
    c.add_argument("--output",  default="predictor_data/")
    c.add_argument("--tokens",  type=int, default=1_000_000)
    c.add_argument("--device",  default="cuda")
    c.add_argument("--trust-remote-code", action="store_true",
                   help="Needed for checkpoints with custom modeling code "
                        "(e.g. TurboSparse/Bamboo)")

    # train
    t = sub.add_parser("train", help="Train predictors from collected data")
    t.add_argument("--data",    required=True, help="Data dir from collect step")
    t.add_argument("--output",  default="predictors/")
    t.add_argument("--epochs",  type=int, default=3)
    t.add_argument("--batch",   type=int, default=256)
    t.add_argument("--rank",    type=int, default=64)
    t.add_argument("--lr",      type=float, default=1e-3)
    t.add_argument("--device",  default="cuda")

    args = parser.parse_args()

    if args.cmd == "collect":
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, use_fast=True, trust_remote_code=args.trust_remote_code
        )
        model     = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map=args.device,
            trust_remote_code=args.trust_remote_code,
        )
        model.eval()
        from profiler import detect_arch
        arch      = detect_arch(model)
        collector = ProfilingDataCollector(model, arch, args.output)
        collector.collect(tokenizer, args.corpus, args.tokens, device=args.device)

    elif args.cmd == "train":
        trainer = PredictorTrainer(args.data, args.output, args.rank, args.device)
        trainer.train_all(args.epochs, args.batch, args.lr)

    else:
        parser.print_help()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
