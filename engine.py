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

def build_rope_cache(
    max_seq_len: int,
    head_dim: int,
    theta: float,            # Llama-3.1: 500000, Mistral-7B: 10000 — from config
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    half  = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    t     = torch.arange(max_seq_len, device=device).float()
    emb   = torch.outer(t, freqs)
    return emb.cos(), emb.sin()


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


class SparseFFNLayer(nn.Module):
    """
    One FFN layer with hot neurons resident in VRAM and cold neurons
    fetched from NVMe, guided by the activation predictor.

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
    ):
        super().__init__()
        self.layer_idx  = layer_idx
        self.cold_store = cold_store
        self.predictor  = predictor
        self.device     = device
        self.executor   = executor
        self.activation  = cfg.get("hidden_act", "silu")
        self.act_on_both = cfg.get("act_on_both", False)
        self.inter_dim   = cfg["intermediate_dim"]

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
                "neuron_ids": w["neuron_ids"],
                "gate": w["gate"].to(torch.float16),
                "up":   w["up"].to(torch.float16),
                "down": w["down"].to(torch.float16),
            }
        self._hot_id_set: set[int] = (
            set(self.hot["neuron_ids"].tolist()) if self.hot else set()
        )

    def _fetch_cold(self, neuron_ids: list[int]):
        """Blocking cold fetch — runs in the thread pool to overlap with GPU."""
        result = self.cold_store.read_neurons(self.layer_idx, self._EID, neuron_ids)
        if result is None:
            return None, None, None
        gate, up, down = result
        return gate.to(self.device), up.to(self.device), down.to(self.device)

    def _cold_ids_for(self, x_normed: torch.Tensor) -> list[int]:
        """Cold neuron ids to fetch — predictor-filtered when available."""
        if self.predictor is not None:
            h_query = x_normed[:, -1, :].mean(0, keepdim=True)
            mask = self.predictor.predict(self.layer_idx, self._EID, h_query)
            if mask is not None:
                ids = mask[0].nonzero(as_tuple=True)[0].tolist()
                return [i for i in ids if i not in self._hot_id_set]
        # No predictor: fetch every cold neuron (correct, but slow)
        return [i for i in range(self.inter_dim) if i not in self._hot_id_set]

    def submit_cold_fetch(self, x: torch.Tensor):
        """Submit this layer's NVMe read now so it overlaps with the caller's
        GPU compute. Returns a future to pass back into forward()."""
        with torch.no_grad():
            x_n = self.input_layernorm(x)
        return self.executor.submit(self._fetch_cold, self._cold_ids_for(x_n))

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

        gate_cold, up_cold, down_cold = future.result()

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
        self.ffn_layers = nn.ModuleList([
            SparseFFNLayer(
                li, hot_dir, self.cold_store, self.predictor,
                self.cfg, self.device, self.executor,
                ffn_norm_weight=shared.get(f"layer_{li:03d}.post_attention_layernorm"),
            )
            for li in range(self.num_layers)
        ])

        self.cos_cache, self.sin_cache = build_rope_cache(
            self.max_seq, head_dim, self.cfg["rope_theta"], device=self.device
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
