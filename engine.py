"""
desdemona_neural/engine.py

Runtime inference engine for desdemona-x3.

One dense base model whose FFN neurons are split hot/cold:
  - Shared attention weights live in VRAM (FP8 at rest, FP16 compute)
  - Hot FFN neurons (frequently activated) live in VRAM (FP8)
  - Cold FFN neurons live on NVMe (INT4), fetched async per token

Per-token pipeline:
  Layer 0:  predict active cold neurons (layer 0) → async NVMe fetch
            meanwhile: compute attention(layer 0) + hot FFN(layer 0)
            wait for fetch → add cold FFN(layer 0) contribution
  Layer 1:  ... same, with layer 1's fetch submitted during layer 0's FFN
  ...

The predictor (desdemona_neural.predictor.PredictorRegistry) prefetches
specific neurons, so only the predicted-active subset of cold neurons is
read per layer. On a ReLU-sparsified base (~90% of neurons inactive per
token) this brings NVMe fetch time below CUDA compute time. Without a
predictor the engine still works — it fetches every cold neuron per layer,
which is correct but slow; train predictors to get the speedup.

Geometry (hidden dim, FFN width, layer count, heads, vocab, RoPE theta,
activation function) is read from split_config.json — written by
splitter.py's split-dense mode from the base model's own HF config —
NOT hardcoded, so the engine follows whatever base model was split.

Key design choices:
  - Thread pool for NVMe reads (Windows lacks io_uring; async overlap via
    futures achieves the same compute/IO pipelining)
  - Cross-layer prefetch: layer L+1's cold fetch is submitted using layer
    L's post-attention hidden state (PowerInfer-style lookahead — the
    predictor input is one sublayer early, an accepted approximation)
  - FFN activation is config-driven:
      "silu" → silu(gate) ⊙ up          (SwiGLU: Llama/Mistral/Nemotron)
      "relu" → relu(gate) ⊙ up          (ReLU-sparsified bases, e.g.
                                          TurboSparse/ProSparse checkpoints)
    (The pre-refactor engine computed gate ⊙ silu(gate) ⊙ up — an extra
    factor of gate. Never hit production: the engine had never run.)
  - INT4 cold weights dequantized on the fly to FP32 before matmul
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from magnitude import active_mask, load_thresholds
from predictor import PredictorRegistry
from splitter  import ColdWeightStore


# ─── Attention (shared, always in VRAM) ───────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    """
    GQA sized from split_config.json (e.g. 32 Q heads / 8 KV heads for both
    Mistral-7B and Llama-3.1-8B geometry).
    """

    def __init__(self, layer_idx: int, weights: dict[str, torch.Tensor], cfg: dict):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_q       = cfg["num_attention_heads"]
        self.n_kv      = cfg["num_key_value_heads"]
        self.head_dim  = cfg["hidden_dim"] // self.n_q
        self.hidden    = cfg["hidden_dim"]

        # Weights are FP8 at rest; cast to FP16 before compute
        prefix = f"layer_{layer_idx:03d}."
        self.q_weight = weights[prefix + "q_proj"].to(torch.float16)
        self.k_weight = weights[prefix + "k_proj"].to(torch.float16)
        self.v_weight = weights[prefix + "v_proj"].to(torch.float16)
        self.o_weight = weights[prefix + "o_proj"].to(torch.float16)

        # Attention pre-norm (input_layernorm applied before Q/K/V)
        self.norm = nn.RMSNorm(self.hidden, eps=cfg["rms_norm_eps"])
        norm_w = weights.get(prefix + "input_layernorm")
        if norm_w is not None:
            self.norm.weight = nn.Parameter(norm_w.to(torch.float16))

    def forward(
        self,
        x: torch.Tensor,          # (B, T, H)
        cos: torch.Tensor,        # (T, head_dim//2)
        sin: torch.Tensor,        # (T, head_dim//2)
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, H = x.shape

        x_n = self.norm(x)
        q = F.linear(x_n, self.q_weight)
        k = F.linear(x_n, self.k_weight)
        v = F.linear(x_n, self.v_weight)

        q = q.view(B, T, self.n_q,  self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_kv = (k, v)

        groups = self.n_q // self.n_kv
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=(kv_cache is None))
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        out = F.linear(out, self.o_weight)

        return out, new_kv


# ─── RoPE ─────────────────────────────────────────────────────────────────────

def llama3_inv_freq(inv: torch.Tensor, head_dim: int, rope_scaling: Optional[dict]) -> torch.Tensor:
    """Llama-3.1 'llama3' RoPE frequency scaling (as in HF transformers'
    _compute_llama3_parameters). Without it the engine's rotations differ from
    the checkpoint's at every position, not only past 8k: low-frequency
    components are divided by `factor`. Other/absent scaling: unchanged."""
    if not rope_scaling or rope_scaling.get("rope_type", rope_scaling.get("type")) != "llama3":
        return inv
    import math
    factor = rope_scaling["factor"]
    lo, hi = rope_scaling["low_freq_factor"], rope_scaling["high_freq_factor"]
    old = rope_scaling["original_max_position_embeddings"]
    lo_wl, hi_wl = old / lo, old / hi
    wl = 2 * math.pi / inv
    smooth = (old / wl - lo) / (hi - lo)
    return torch.where(wl > lo_wl, inv / factor,
                       torch.where(wl < hi_wl, inv, (1 - smooth) * inv / factor + smooth * inv))


def build_rope_cache(
    max_seq_len: int,
    head_dim: int,
    theta: float,            # Llama-3.1: 500000, Mistral-7B: 10000 — from config
    device: torch.device = torch.device("cpu"),
    rope_scaling: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    half  = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device).double() / half))
    freqs = llama3_inv_freq(freqs, head_dim, rope_scaling)
    t     = torch.arange(max_seq_len, device=device).double()
    emb   = torch.outer(t, freqs)
    return emb.cos().float(), emb.sin().float()


def _apply_rope(
    x: torch.Tensor,   # (B, n_heads, T, D)
    cos: torch.Tensor, # (T, D//2)
    sin: torch.Tensor,
) -> torch.Tensor:
    B, H, T, D = x.shape
    half = D // 2
    x1, x2 = x[..., :half], x[..., half:]
    # The rope cache is built in float32 for precision; cast to the activation
    # dtype here so q/k don't silently upcast (which would leave v, that skips
    # rope, at a mismatched dtype going into scaled_dot_product_attention).
    cos_t = cos[:T].unsqueeze(0).unsqueeze(0).to(x.dtype)
    sin_t = sin[:T].unsqueeze(0).unsqueeze(0).to(x.dtype)
    rotated = torch.cat([x1 * cos_t - x2 * sin_t,
                         x1 * sin_t + x2 * cos_t], dim=-1)
    return rotated


# ─── Hot/cold FFN ─────────────────────────────────────────────────────────────

def _gated_act(g: torch.Tensor, u: torch.Tensor,
               activation: str, act_on_both: bool) -> torch.Tensor:
    act = F.relu if activation == "relu" else F.silu
    if act_on_both:
        # dReLU family (TurboSparse/Bamboo): act(gate) ⊙ act(up) — verified
        # against the checkpoint's own modeling_bamboo.py BambooMLP.forward
        return act(g) * act(u)
    return act(g) * u          # standard gated FFN (SwiGLU etc.)


def sparse_ffn(
    x: torch.Tensor,                      # (B, H)
    activation: str,
    act_on_both: bool,
    gate_hot:  Optional[torch.Tensor],    # (n_hot, H) FP16
    up_hot:    Optional[torch.Tensor],
    down_hot:  Optional[torch.Tensor],    # (H, n_hot)
    gate_cold: Optional[torch.Tensor],    # (n_cold, H) FP32 (dequantized)
    up_cold:   Optional[torch.Tensor],
    down_cold: Optional[torch.Tensor],    # (H, n_cold)
) -> torch.Tensor:
    """
    Gated FFN assembled from hot (VRAM) + cold (NVMe-fetched) neuron subsets.
    Because down_proj is a sum over neurons, computing hot and cold halves
    separately and adding is exact — no approximation beyond INT4 quant.
    """
    results = []

    # down_hot / down_cold are stored (H, n_neurons); F.linear(h, W) computes
    # h @ W.T, so passing them directly maps the (B, n_neurons) activation to
    # (B, H). (An earlier .t() here double-transposed and broke the matmul.)
    if gate_hot is not None and gate_hot.shape[0] > 0:
        g = F.linear(x, gate_hot)
        u = F.linear(x, up_hot)
        results.append(F.linear(_gated_act(g, u, activation, act_on_both), down_hot))

    if gate_cold is not None and gate_cold.shape[0] > 0:
        xf = x.float()
        g = F.linear(xf, gate_cold)
        u = F.linear(xf, up_cold)
        results.append(
            F.linear(_gated_act(g, u, activation, act_on_both), down_cold).to(x.dtype)
        )

    if not results:
        return torch.zeros_like(x)
    return sum(results)


def _act_fn(activation: str):
    return F.relu if activation == "relu" else F.silu


def dense_ffn(x, w_gate, w_up, w_down, activation="silu", act_on_both=False):
    """Reference dense gated FFN: down(act(gate x) * up x). w_down is (H, I)."""
    act = _act_fn(activation)
    g = F.linear(x, w_gate)
    u = F.linear(x, w_up)
    if act_on_both:
        u = act(u)
    return F.linear(act(g) * u, w_down)


def exact_gate_ffn(
    x: torch.Tensor,                  # (N, H)
    gate_all: torch.Tensor,           # (I, H) EVERY gate row, resident
    tau,                              # float, or (I,) tensor; <= 0 keeps all
    activation: str,
    act_on_both: bool,
    hot_ids: torch.Tensor,            # (n_hot,) long, on x.device, sorted
    up_hot: Optional[torch.Tensor],   # (n_hot, H)
    down_hot: Optional[torch.Tensor], # (H, n_hot)
    cold_ids: torch.Tensor,           # (n_cold,) long, on x.device, sorted
    cold_rows,                        # callable(ids long tensor) -> (up (n,H), down (H,n))
    mode: str = "gather",
    stats: Optional[dict] = None,
) -> torch.Tensor:
    """
    CATS-style exact gate (arXiv 2404.08763) over a hot/cold split.

    1. g = W_gate x is computed EXACTLY for all I neurons (all gate rows stay
       resident), s = act(g).
    2. A neuron is active for a token iff |s| >= tau (magnitude.active_mask).
    3. Only active neurons' up rows and down columns are computed (hot) or
       fetched (cold). Inactive neurons contribute exactly 0 by definition of
       the threshold, so nothing a predictor got wrong can change the output:
       the predictor is only a prefetch hint inside `cold_rows`.

    mode="gather": compute only the union of active rows (the speed path;
                   at tau=0 equal to dense up to fp summation order).
    mode="masked": assemble full-width up/down in neuron order and multiply
                   the activation by the 0/1 mask (the verification path; at
                   tau=0 it performs the same ops on the same values as
                   dense_ffn, so its output is bitwise identical).
    """
    act = _act_fn(activation)
    I, H = gate_all.shape
    s = act(F.linear(x, gate_all.to(x.dtype)))
    keep = active_mask(s, tau)
    any_active = keep.any(0)
    need_cold = cold_ids[any_active[cold_ids]] if cold_ids.numel() else cold_ids
    if stats is not None:
        stats["tokens"] = stats.get("tokens", 0) + x.shape[0]
        stats["active"] = stats.get("active", 0) + int(keep.sum())
        stats["union_rows"] = stats.get("union_rows", 0) + int(any_active.sum())
        stats["cold_needed"] = stats.get("cold_needed", 0) + int(need_cold.numel())
        stats["calls"] = stats.get("calls", 0) + 1
        stats["I"] = I

    if mode == "masked":
        up_full = x.new_zeros(I, H)
        down_full = x.new_zeros(H, I)
        if hot_ids.numel():
            up_full[hot_ids] = up_hot.to(x.dtype)
            down_full[:, hot_ids] = down_hot.to(x.dtype)
        if need_cold.numel():
            uc, dc = cold_rows(need_cold)
            up_full[need_cold] = uc.to(x.dtype)
            down_full[:, need_cold] = dc.to(x.dtype)
        u = F.linear(x, up_full)
        if act_on_both:
            u = act(u)
        return F.linear(s * u * keep.to(x.dtype), down_full)

    if mode != "gather":
        raise ValueError(f"unknown exact-gate mode {mode!r}")
    out = None
    if hot_ids.numel():
        sel = any_active[hot_ids]
        if bool(sel.all()):
            ids, uh, dh = hot_ids, up_hot, down_hot
        else:
            idx = sel.nonzero(as_tuple=True)[0]
            ids, uh, dh = hot_ids[idx], up_hot[idx], down_hot[:, idx]
        if ids.numel():
            u = F.linear(x, uh.to(x.dtype))
            if act_on_both:
                u = act(u)
            out = F.linear(s[:, ids] * u * keep[:, ids].to(x.dtype), dh.to(x.dtype))
    if need_cold.numel():
        uc, dc = cold_rows(need_cold)
        u = F.linear(x, uc.to(x.dtype))
        if act_on_both:
            u = act(u)
        c = F.linear(s[:, need_cold] * u * keep[:, need_cold].to(x.dtype), dc.to(x.dtype))
        out = c if out is None else out + c
    return out if out is not None else torch.zeros_like(x)


class SparseFFNLayer(nn.Module):
    """
    One FFN layer with hot neurons resident in VRAM and cold neurons
    fetched from NVMe.

    gate_mode="exact" (default): every gate row is resident (hot rows from the
    hot file, cold rows read once from the cold store at load), the gate is
    computed exactly, and only neurons with |act(g)| >= tau_L get their up and
    down rows computed or fetched (exact_gate_ffn). The predictor, if present,
    only prefetches; a miss is fetched synchronously and costs time, never
    accuracy. tau_L = 0 reproduces the dense FFN of the split weights.

    gate_mode="predictor": the legacy path. Cold neurons the predictor does
    not select are dropped, so the output is approximate. Kept only for
    comparison with old numbers.

    Internally reuses the split artifacts' (layer, expert) indexing with
    expert fixed at 0 — the dense model is stored as a single "expert 0"
    so ColdWeightStore, the hot-file naming, and the predictor registry
    all work unchanged.
    """

    _EID = 0    # dense model == single expert 0 in the split artifacts

    def __init__(
        self,
        layer_idx: int,
        hot_dir:   str,
        cold_store: ColdWeightStore,
        predictor:  Optional[PredictorRegistry],
        cfg:        dict,
        device:     torch.device,
        executor:   ThreadPoolExecutor,
        ffn_norm_weight: Optional[torch.Tensor] = None,
        gate_mode:  str = "exact",
        tau:        float = 0.0,
        exact_mode: str = "gather",
    ):
        super().__init__()
        if gate_mode not in ("exact", "predictor"):
            raise ValueError(f"gate_mode must be 'exact' or 'predictor', got {gate_mode!r}")
        self.layer_idx  = layer_idx
        self.cold_store = cold_store
        self.predictor  = predictor
        self.device     = device
        self.executor   = executor
        self.activation  = cfg.get("hidden_act", "silu")
        self.act_on_both = cfg.get("act_on_both", False)
        self.inter_dim   = cfg["intermediate_dim"]
        self.gate_mode   = gate_mode
        self.tau         = float(tau)
        self.exact_mode  = exact_mode
        self.stats: dict = {}

        # Pre-FFN norm (post_attention_layernorm — applied before FFN)
        self.input_layernorm = nn.RMSNorm(cfg["hidden_dim"], eps=cfg["rms_norm_eps"])
        if ffn_norm_weight is not None:
            self.input_layernorm.weight = nn.Parameter(ffn_norm_weight.to(device))

        # Hot weights for this layer
        self.hot: Optional[dict[str, torch.Tensor]] = None
        p = Path(hot_dir) / f"layer_{layer_idx:03d}_expert_{self._EID:03d}.pt"
        if p.exists():
            w = torch.load(str(p), map_location=device)
            self.hot = {
                "neuron_ids": w["neuron_ids"].to(device).long(),
                "gate": w["gate"].to(torch.float16),
                "up":   w["up"].to(torch.float16),
                "down": w["down"].to(torch.float16),
            }
        self._hot_id_set: set[int] = (
            set(self.hot["neuron_ids"].tolist()) if self.hot else set()
        )
        hot_mask = torch.zeros(self.inter_dim, dtype=torch.bool)
        if self.hot:
            hot_mask[self.hot["neuron_ids"].cpu()] = True
        self.hot_ids  = hot_mask.nonzero(as_tuple=True)[0].to(device)
        self.cold_ids = (~hot_mask).nonzero(as_tuple=True)[0].to(device)
        if self.hot is not None:
            # exact_gate_ffn wants hot rows sorted by neuron id
            order = torch.argsort(self.hot["neuron_ids"])
            for k in ("gate", "up"):
                self.hot[k] = self.hot[k][order]
            self.hot["down"] = self.hot["down"][:, order]
            self.hot["neuron_ids"] = self.hot["neuron_ids"][order]

        self.gate_all: Optional[torch.Tensor] = None
        if gate_mode == "exact":
            self.gate_all = self._assemble_gate()

    def _assemble_gate(self) -> torch.Tensor:
        """All I gate rows, resident: hot rows as stored, cold rows read once."""
        H = self.input_layernorm.weight.shape[0]
        g = torch.zeros(self.inter_dim, H, dtype=torch.float16, device=self.device)
        if self.hot is not None:
            g[self.hot["neuron_ids"]] = self.hot["gate"]
        if self.cold_ids.numel():
            r = self.cold_store.read_neurons(self.layer_idx, self._EID, self.cold_ids.tolist())
            if r is None:
                raise ValueError(f"layer {self.layer_idx}: cold store has none of the cold neurons")
            g[self.cold_ids] = r[0].to(self.device, torch.float16)
        return g

    def _fetch_cold(self, neuron_ids: list[int]):
        """Blocking cold fetch — runs in the thread pool to overlap with GPU.
        Returns (ids, gate, up, down) for the ids actually present."""
        if not neuron_ids:
            return [], None, None, None
        result = self.cold_store.read_neurons(self.layer_idx, self._EID, neuron_ids)
        if result is None:
            return [], None, None, None
        gate, up, down = result
        return (list(neuron_ids), gate.to(self.device), up.to(self.device), down.to(self.device))

    def _predicted_cold_ids(self, x_normed: torch.Tensor) -> Optional[list[int]]:
        if self.predictor is None:
            return None
        h_query = x_normed[:, -1, :].mean(0, keepdim=True)
        mask = self.predictor.predict(self.layer_idx, self._EID, h_query)
        if mask is None:
            return None
        ids = mask[0].nonzero(as_tuple=True)[0].tolist()
        return [i for i in ids if i not in self._hot_id_set]

    def _cold_ids_for(self, x_normed: torch.Tensor) -> list[int]:
        """Cold neuron ids to fetch ahead of the FFN. Exact mode: the
        predictor's hint, or nothing (rows are fetched once the exact gate
        says which are needed). Legacy mode: predictor-filtered, or all."""
        pred = self._predicted_cold_ids(x_normed)
        if pred is not None:
            return pred
        if self.gate_mode == "exact":
            return []
        return [i for i in range(self.inter_dim) if i not in self._hot_id_set]

    def submit_cold_fetch(self, x: torch.Tensor):
        """Submit this layer's NVMe read now so it overlaps with the caller's
        GPU compute. Returns a future to pass back into forward()."""
        with torch.no_grad():
            x_n = self.input_layernorm(x)
        return self.executor.submit(self._fetch_cold, self._cold_ids_for(x_n))

    def _cold_rows_from(self, prefetched):
        """cold_rows callable for exact_gate_ffn: serve needed ids from the
        prefetch where it hit, read the misses synchronously."""
        pids, _, pup, pdown = prefetched

        def rows(need: torch.Tensor):
            n = need.numel()
            H = self.gate_all.shape[1]
            up = torch.empty(n, H, dtype=torch.float32, device=self.device)
            down = torch.empty(H, n, dtype=torch.float32, device=self.device)
            pos = torch.full((self.inter_dim,), -1, dtype=torch.long, device=self.device)
            if pids:
                pos[torch.tensor(pids, device=self.device)] = torch.arange(len(pids), device=self.device)
            p = pos[need]
            hit = p >= 0
            if bool(hit.any()):
                up[hit] = pup[p[hit]].float()
                down[:, hit] = pdown[:, p[hit]].float()
            miss = (~hit).nonzero(as_tuple=True)[0]
            if miss.numel():
                r = self.cold_store.read_neurons(self.layer_idx, self._EID, need[miss].tolist())
                up[miss] = r[1].to(self.device)
                down[:, miss] = r[2].to(self.device)
            self.stats["prefetch_hits"] = self.stats.get("prefetch_hits", 0) + int(hit.sum())
            self.stats["prefetch_misses"] = self.stats.get("prefetch_misses", 0) + int(miss.numel())
            self.stats["prefetch_waste"] = self.stats.get("prefetch_waste", 0) + max(0, len(pids) - int(hit.sum()))
            return up, down
        return rows

    def forward(self, x: torch.Tensor, prefetch_future=None) -> torch.Tensor:
        B, T, H = x.shape
        residual = x
        x = self.input_layernorm(x)

        future = prefetch_future or self.executor.submit(
            self._fetch_cold, self._cold_ids_for(x)
        )

        gate_hot = up_hot = down_hot = None
        if self.hot is not None:
            gate_hot, up_hot, down_hot = self.hot["gate"], self.hot["up"], self.hot["down"]

        prefetched = future.result()

        if self.gate_mode == "exact":
            out = exact_gate_ffn(
                x.reshape(B * T, H), self.gate_all, self.tau,
                self.activation, self.act_on_both,
                self.hot_ids, up_hot, down_hot,
                self.cold_ids, self._cold_rows_from(prefetched),
                mode=self.exact_mode, stats=self.stats,
            )
            return residual + out.reshape(B, T, H)

        _, gate_cold, up_cold, down_cold = prefetched
        out = sparse_ffn(
            x.reshape(B * T, H), self.activation, self.act_on_both,
            gate_hot, up_hot, down_hot,
            gate_cold, up_cold, down_cold,
        )
        return residual + out.reshape(B, T, H)


# ─── Full Model ───────────────────────────────────────────────────────────────

class DesdemonaEngine(nn.Module):
    """
    desdemona-x3 inference engine: dense base model, hot/cold FFN split,
    predictor-guided NVMe prefetch with cross-layer pipelining.

    Expects split artifacts produced by splitter.py's split-dense mode:
      split_dir/split_config.json
      split_dir/hot_weights/shared.pt
      split_dir/hot_weights/layer_XXX_expert_000.pt
      split_dir/cold_weights.bin + cold_index.npy
    """

    def __init__(
        self,
        split_dir: str,
        predictor_dir: Optional[str] = None,
        device: str = "cuda",
        n_threads: int = 8,
        max_seq_len: int = 8192,
        gate_mode: str = "exact",
        thresholds: Optional[str] = None,
        sparsity: Optional[float] = None,
        exact_mode: str = "gather",
    ):
        super().__init__()
        self.device   = torch.device(device)
        self.executor = ThreadPoolExecutor(max_workers=n_threads)

        with open(Path(split_dir) / "split_config.json") as f:
            self.cfg = json.load(f)
        if self.cfg.get("num_experts", 1) != 1:
            raise ValueError(
                f"split_dir contains a {self.cfg['num_experts']}-expert MoE split; "
                "this engine serves dense split-dense artifacts only."
            )

        self.num_layers = self.cfg["num_layers"]
        hidden          = self.cfg["hidden_dim"]
        vocab           = self.cfg["vocab_size"]
        head_dim        = hidden // self.cfg["num_attention_heads"]
        self.max_seq    = min(max_seq_len, self.cfg.get("max_position_embeddings", max_seq_len))

        # Shared weights (embed, attention, norms, lm_head)
        print("Loading shared weights...")
        shared = torch.load(
            str(Path(split_dir) / "hot_weights" / "shared.pt"),
            map_location=self.device,
        )
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.embed_tokens.weight = nn.Parameter(shared["embed_tokens"].to(self.device))
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = nn.Parameter(shared["lm_head"].to(self.device))
        self.final_norm = nn.RMSNorm(hidden, eps=self.cfg["rms_norm_eps"])
        if "final_norm" in shared:
            self.final_norm.weight = nn.Parameter(shared["final_norm"].to(self.device))

        # Predictor registry (optional — engine runs without, just slower)
        self.predictor = None
        if predictor_dir and Path(predictor_dir).exists():
            print("Loading predictor registry...")
            self.predictor = PredictorRegistry(predictor_dir, device=device)
            print(f"  Predictor VRAM: {self.predictor.vram_usage_mb():.0f} MB")

        # Cold weight store (memory-mapped NVMe file)
        self.cold_store = ColdWeightStore(
            str(Path(split_dir) / "cold_weights.bin"),
            str(Path(split_dir) / "cold_index.npy"),
            self.cfg,
        )

        print(f"Loading {self.num_layers} attention layers...")
        self.attention_layers = nn.ModuleList([
            GroupedQueryAttention(li, shared, self.cfg)
            for li in range(self.num_layers)
        ])

        print(f"Loading {self.num_layers} sparse FFN layers (hot weights)...")
        hot_dir = str(Path(split_dir) / "hot_weights")
        # Per-layer tau_L from profiler.py's thresholds.json; none -> tau 0 (exact)
        taus = load_thresholds(thresholds, sparsity) if thresholds else {}
        self.ffn_layers = nn.ModuleList([
            SparseFFNLayer(
                li, hot_dir, self.cold_store, self.predictor,
                self.cfg, self.device, self.executor,
                ffn_norm_weight=shared.get(f"layer_{li:03d}.post_attention_layernorm"),
                gate_mode=gate_mode, tau=taus.get(li, 0.0), exact_mode=exact_mode,
            )
            for li in range(self.num_layers)
        ])

        self.cos_cache, self.sin_cache = build_rope_cache(
            self.max_seq, head_dim, self.cfg["rope_theta"], device=self.device,
            rope_scaling=self.cfg.get("rope_scaling"),
        )

        print("Engine ready.")

    def forward(
        self,
        input_ids: torch.Tensor,        # (B, T)
        kv_caches: Optional[list] = None,
    ) -> tuple[torch.Tensor, list]:
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)

        # RoPE positions continue from the KV cache length during decode
        past_len = kv_caches[0][0].shape[2] if kv_caches else 0
        cos = self.cos_cache[past_len:past_len + T]
        sin = self.sin_cache[past_len:past_len + T]

        new_kv_caches = []

        # Kick off layer 0's cold fetch now (overlaps with layer 0 attention)
        next_prefetch = self.ffn_layers[0].submit_cold_fetch(x)

        for li in range(self.num_layers):
            kv = kv_caches[li] if kv_caches else None
            attn_out, new_kv = self.attention_layers[li](x, cos, sin, kv)
            x = x + attn_out
            new_kv_caches.append(new_kv)

            # Capture this layer's in-flight fetch, then submit the next
            # layer's using the current hidden state (lookahead approximation)
            this_prefetch = next_prefetch
            if li + 1 < self.num_layers:
                next_prefetch = self.ffn_layers[li + 1].submit_cold_fetch(x)
            else:
                next_prefetch = None

            x = self.ffn_layers[li](x, this_prefetch)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, new_kv_caches

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: int | list[int] | tuple[int, ...] | None = None,
    ) -> list[int]:
        """
        Autoregressive generation with KV caching.
        Returns generated token ids only (prompt excluded).
        eos_token_id should come from the tokenizer — it is model-specific
        and no longer defaulted here.
        """
        if eos_token_id is None:
            eos_set: set[int] = set()
        elif isinstance(eos_token_id, int):
            eos_set = {eos_token_id}
        else:
            eos_set = set(eos_token_id)

        kv_caches = None
        generated = []
        t0 = time.perf_counter()

        logits, kv_caches = self.forward(input_ids, kv_caches)
        next_token = _sample(logits[:, -1, :], temperature, top_p)
        generated.append(next_token.item())

        for _ in range(max_new_tokens - 1):
            if next_token.item() in eos_set:
                break
            logits, kv_caches = self.forward(next_token.unsqueeze(0), kv_caches)
            next_token = _sample(logits[:, -1, :], temperature, top_p)
            generated.append(next_token.item())

        elapsed = time.perf_counter() - t0
        print(f"Generated {len(generated)} tokens in {elapsed:.2f}s "
              f"({len(generated) / elapsed:.1f} tok/s)")
        return generated

    def close(self):
        self.cold_store.close()
        self.executor.shutdown(wait=False)


# ─── Sampling ─────────────────────────────────────────────────────────────────

def _sample(
    logits: torch.Tensor,    # (B, vocab)
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    if temperature == 0.0:
        return logits.argmax(dim=-1)

    logits = logits / temperature
    probs  = F.softmax(logits, dim=-1)

    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    mask   = (cumsum - sorted_probs) < top_p
    sorted_probs[~mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_ids.gather(-1, idx).squeeze(-1)


# ─── Benchmark ────────────────────────────────────────────────────────────────

def benchmark(
    split_dir: str,
    predictor_dir: Optional[str] = None,
    n_warmup: int = 5,
    n_tokens: int = 50,
    prompt_len: int = 32,
    device: str = "cuda",
):
    """Measure tok/s with a dummy prompt — validates NVMe/CUDA overlap."""
    model = DesdemonaEngine(split_dir, predictor_dir, device)
    vocab = model.cfg["vocab_size"]
    dummy = torch.randint(0, vocab, (1, prompt_len), device=device)

    print(f"\nWarm-up ({n_warmup} tokens)...")
    model.generate(dummy, max_new_tokens=n_warmup, temperature=0.0)

    print(f"Benchmark ({n_tokens} tokens)...")
    t0 = time.perf_counter()
    model.generate(dummy, max_new_tokens=n_tokens, temperature=0.0)
    tok_per_sec = n_tokens / (time.perf_counter() - t0)

    print(f"\nResult: {tok_per_sec:.1f} tok/s")
    model.close()
    return tok_per_sec


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    g = sub.add_parser("generate", help="Run inference")
    g.add_argument("--split-dir",    required=True)
    g.add_argument("--predictor",    default=None)
    g.add_argument("--tokenizer",    required=True,
                   help="HF id or local path — must match the split base model")
    g.add_argument("--prompt",       default="What is justice according to Plato?")
    g.add_argument("--max-tokens",   type=int, default=200)
    g.add_argument("--temperature",  type=float, default=0.7)
    g.add_argument("--device",       default="cuda")

    b = sub.add_parser("benchmark", help="Measure tok/s")
    b.add_argument("--split-dir",   required=True)
    b.add_argument("--predictor",   default=None)
    b.add_argument("--n-tokens",    type=int, default=50)
    b.add_argument("--prompt-len",  type=int, default=32)
    b.add_argument("--device",      default="cuda")

    args = parser.parse_args()

    if args.cmd == "generate":
        # trust_remote_code so tokenizers from checkpoints with custom code
        # (e.g. the dReLU/bamboo TurboSparse family) load non-interactively.
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        ids = tok(args.prompt, return_tensors="pt").input_ids.to(args.device)

        model = DesdemonaEngine(args.split_dir, args.predictor, args.device)
        out_ids = model.generate(
            ids, max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            eos_token_id=tok.eos_token_id,
        )
        print("\n" + tok.decode(out_ids))
        model.close()

    elif args.cmd == "benchmark":
        benchmark(args.split_dir, args.predictor,
                  n_tokens=args.n_tokens, prompt_len=args.prompt_len,
                  device=args.device)

    else:
        parser.print_help()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
