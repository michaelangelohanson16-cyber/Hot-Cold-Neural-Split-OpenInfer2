"""
magnitude.py -- magnitude-based "fired" criterion for gated FFNs (SwiGLU / dReLU).

Why this exists. The old criterion counted a neuron as fired when
silu(gate) > 0. For SwiGLU that is a sign test, not a contribution test: on
Hermes-3-Llama-3.1-8B it keeps 23.9% of neurons while the set that carries 99%
of |silu(g)*u| mass needs 82.6% (desdemona-x3 docs/OPENINFER_STEP0_RESULTS.md
section 2). Anything built on the sign mask silently drops real work.

Replacement, used by profiler.py, predictor.py and engine.py:
  * mass   = |a| with a = act(g) * u (the down_proj input), per neuron.
  * kXX    = per token, the smallest share of neurons whose |a| sums to XX% of
             that token's total |a| (step 0's sparsity_profile.py metric).
  * tau_L  = a per-layer CATS threshold on |act(g)| (arXiv 2404.08763),
             calibrated as the p-quantile of |act(g)| over calibration tokens.
             A neuron is active iff |act(g)| >= tau_L. tau_L = 0 keeps every
             neuron, so the gate is exact by construction.

Everything here is plain torch on whatever device the inputs live on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import torch

MASS_LEVELS = (0.90, 0.95, 0.99)


def active_mask(act_g: torch.Tensor, tau) -> torch.Tensor:
    """Neuron is active iff |act(g)| >= tau (tau scalar or broadcastable).
    tau <= 0 marks every neuron active (the exact gate)."""
    if not torch.is_tensor(tau) and float(tau) <= 0.0:
        return torch.ones_like(act_g, dtype=torch.bool)
    return act_g.abs() >= tau


def mass_shares(abs_a: torch.Tensor, levels: Iterable[float] = MASS_LEVELS) -> dict:
    """abs_a: (N_tokens, I) = |act(g)*u|. Returns {"k90": share, ...}: the mean
    over tokens of the minimal share of neurons whose |a| sums to the level."""
    a = abs_a.float()
    srt = torch.sort(a, dim=1, descending=True).values
    cs = torch.cumsum(srt, dim=1)
    tot = cs[:, -1:].clamp(min=1e-30)
    frac = cs / tot
    I = a.shape[1]
    out = {}
    for m in levels:
        k = (frac < m).sum(dim=1) + 1          # neurons needed to reach mass m
        out[f"k{int(round(m * 100))}"] = float(k.float().mean() / I)
    return out


def quantile_large(x: torch.Tensor, qs: Iterable[float]) -> list[float]:
    """Exact quantiles of a large tensor (torch.quantile caps at 16M elements)."""
    flat = x.reshape(-1).float()
    srt = torch.sort(flat).values
    n = srt.numel()
    out = []
    for q in qs:
        if q <= 0:
            out.append(0.0)
            continue
        idx = min(n - 1, max(0, int(round(q * (n - 1)))))
        out.append(float(srt[idx]))
    return out


class MassAccumulator:
    """Per-layer running statistics for one (layer, expert) FFN.

    feed(act_g, a) with act_g = act(gate) and a = the down_proj input
    (act(g)*u, or act(g)*act(u) for dReLU), both (N, I).
    Keeps: legacy >0 counts, per-neuron |a| mass, kXX shares, a bounded
    reservoir of |act(g)| samples for threshold calibration, and per-neuron
    active counts under any thresholds already known (set_thresholds)."""

    def __init__(self, intermediate_dim: int, reservoir: int = 4_000_000, seed: int = 0):
        self.I = intermediate_dim
        self.tokens = 0
        self.pos_counts = torch.zeros(intermediate_dim, dtype=torch.float64)
        self.mass = torch.zeros(intermediate_dim, dtype=torch.float64)
        self.k_sums = {f"k{int(round(m*100))}": 0.0 for m in MASS_LEVELS}
        self._res_cap = reservoir
        self._res: list[torch.Tensor] = []
        self._res_n = 0
        self._gen = torch.Generator().manual_seed(seed)
        self.thresholds: dict[str, float] = {}
        self.active_counts: dict[str, torch.Tensor] = {}

    def set_thresholds(self, thresholds: dict[str, float]):
        self.thresholds = dict(thresholds)
        self.active_counts = {k: torch.zeros(self.I, dtype=torch.float64) for k in thresholds}

    @torch.no_grad()
    def feed(self, act_g: torch.Tensor, a: torch.Tensor):
        act_g = act_g.reshape(-1, act_g.shape[-1])
        a = a.reshape(-1, a.shape[-1])
        n = a.shape[0]
        self.tokens += n
        self.pos_counts += (act_g > 0).sum(0).double().cpu()
        abs_a = a.abs().float()
        self.mass += abs_a.sum(0).double().cpu()
        ks = mass_shares(abs_a)
        for k, v in ks.items():
            self.k_sums[k] += v * n
        ag = act_g.abs().float()
        for name, tau in self.thresholds.items():
            self.active_counts[name] += (ag >= tau).sum(0).double().cpu()
        # Reservoir of |act(g)| for calibration: uniform subsample per batch.
        flat = ag.reshape(-1).cpu()
        keep = min(flat.numel(), max(1, self._res_cap // 64))
        idx = torch.randint(0, flat.numel(), (keep,), generator=self._gen)
        self._res.append(flat[idx])
        self._res_n += keep
        while self._res_n > self._res_cap and len(self._res) > 1:
            self._res_n -= self._res[0].numel()
            self._res.pop(0)

    def calibrate(self, sparsities: Iterable[float]) -> dict[str, float]:
        """CATS tau per target sparsity p: the p-quantile of |act(g)|."""
        sample = torch.cat(self._res) if self._res else torch.zeros(1)
        sp = list(sparsities)
        return {f"{p:.2f}": v for p, v in zip(sp, quantile_large(sample, sp))}

    def summary(self) -> dict:
        n = max(self.tokens, 1)
        out = {
            "tokens": self.tokens,
            "pos_frac_legacy": float(self.pos_counts.sum() / (n * self.I)),
            **{k: v / n for k, v in self.k_sums.items()},
        }
        for name, c in self.active_counts.items():
            out[f"active_frac@{name}"] = float(c.sum() / (n * self.I))
        return out

    def frequencies(self, name: Optional[str] = None) -> list[float]:
        """Per-neuron active frequency under threshold `name` (magnitude
        criterion); falls back to normalised mass when no threshold is set.
        This is what splitter.py ranks hot neurons by."""
        n = max(self.tokens, 1)
        if name is not None and name in self.active_counts:
            return (self.active_counts[name] / n).tolist()
        m = self.mass / max(float(self.mass.max()), 1e-30)
        return m.tolist()


def save_thresholds(path: str, taus: dict[int, dict[str, float]], meta: dict | None = None):
    """thresholds.json: {"meta": {...}, "layers": {"<L>": {"0.40": tau, ...}}}"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"meta": meta or {}, "layers": {str(k): v for k, v in taus.items()}}, f, indent=1)


def load_thresholds(path: str, sparsity: Optional[float]) -> dict[int, float]:
    """Per-layer tau for one target sparsity key; sparsity None or 0 -> all 0."""
    if not sparsity:
        with open(path) as f:
            layers = json.load(f)["layers"]
        return {int(k): 0.0 for k in layers}
    key = f"{sparsity:.2f}"
    with open(path) as f:
        layers = json.load(f)["layers"]
    return {int(k): float(v[key]) for k, v in layers.items()}
