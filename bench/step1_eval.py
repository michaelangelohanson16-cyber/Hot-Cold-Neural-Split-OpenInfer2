"""
bench/step1_eval.py -- Step 1 gates for the mask fix on a real model.

Layer-streamed Llama-3.x forward on one GPU, fp32 compute from the HF bf16
safetensors, with the FFN of every non-dense variant computed by the engine's
own exact_gate_ffn (engine.py). One weight pass serves every variant, so the
comparison is same-weights, same-kernels, same-shapes: only the FFN differs.

Phases (run in order, each writes JSON into --out):
  calib  CATS tau_L per layer from held-out Desdemona text (|silu(g)| quantiles),
         per-neuron magnitude frequency (for the hot/cold split), and the
         magnitude profile (>0 legacy, k90/95/99, mass kept, 5-token union)
         on the step-0 text, to cross-check step 0's numpy numbers.
  ppl    PPL on the step-0 text for dense, exact-gate tau=0 (masked and gather)
         and the tau sweep; bitwise / max-abs logit comparison against dense.
  gen    25-question greedy generation (first N tokens), dense vs exact gate
         tau=0 vs the chosen sparse tau: token identity and logit diffs.
  ffnbench  batch-1 timing of one resident fp16 FFN layer: dense vs the
         exact-gate gather path at each tau (what step 1 does to speed).

Usage: python bench/step1_eval.py --phase calib|ppl|gen|ffnbench --model DIR \
         --out DIR [--ppl-text F] [--calib-text F] [--questions F]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import dense_ffn, exact_gate_ffn, llama3_inv_freq  # noqa: E402
from magnitude import mass_shares, quantile_large  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
DEV = torch.device(os.environ.get("OI_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
SPARSITIES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ─── weights ──────────────────────────────────────────────────────────────────

class Weights:
    """bf16 safetensors reader: plain reads (no mmap faults), tensors to GPU."""

    def __init__(self, d: str):
        self.d = d
        self.cfg = json.load(open(os.path.join(d, "config.json")))
        self.idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.hdr = {}
        self.bytes_read = 0
        self.read_s = 0.0

    def _h(self, fn):
        if fn not in self.hdr:
            with open(os.path.join(self.d, fn), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self.hdr[fn] = (json.loads(f.read(n)), 8 + n)
        return self.hdr[fn]

    def meta(self, name):
        fn = self.idx[name]
        h, base = self._h(fn)
        m = h[name]
        assert m["dtype"] == "BF16", (name, m["dtype"])
        return fn, base, m

    def get(self, name, device=DEV) -> torch.Tensor:
        fn, base, m = self.meta(name)
        s, e = m["data_offsets"]
        t0 = time.perf_counter()
        buf = torch.empty((e - s) // 2, dtype=torch.int16)
        with open(os.path.join(self.d, fn), "rb", buffering=0) as f:
            f.seek(base + s)
            f.readinto(memoryview(buf.numpy()).cast("B"))
        self.bytes_read += e - s
        self.read_s += time.perf_counter() - t0
        return buf.view(torch.bfloat16).reshape(m["shape"]).to(device, non_blocking=False)

    def rows(self, name, ids) -> torch.Tensor:
        fn, base, m = self.meta(name)
        s, _ = m["data_offsets"]
        mm = np.memmap(os.path.join(self.d, fn), dtype=np.uint16, mode="r",
                       offset=base + s, shape=tuple(m["shape"]))
        r = torch.from_numpy(np.ascontiguousarray(mm[np.asarray(ids)]).view(np.int16))
        return r.view(torch.bfloat16)


class Layers:
    """Per-layer bf16 weights, LRU-resident on the GPU up to a byte budget."""
    NAMES = ("input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
             "self_attn.v_proj.weight", "self_attn.o_proj.weight",
             "post_attention_layernorm.weight", "mlp.gate_proj.weight",
             "mlp.up_proj.weight", "mlp.down_proj.weight")

    def __init__(self, W: Weights, resident_bytes: int):
        self.W = W
        self.budget = resident_bytes
        self.cache: dict[int, dict] = {}
        self.order: list[int] = []

    def __call__(self, L):
        if L in self.cache:
            return self.cache[L]
        w = {n: self.W.get(f"model.layers.{L}.{n}") for n in self.NAMES}
        nb = sum(t.numel() * 2 for t in w.values())
        if (len(self.cache) + 1) * nb <= self.budget:
            self.cache[L] = w
            self.order.append(L)
        return w


# ─── model pieces ─────────────────────────────────────────────────────────────

class Model:
    def __init__(self, model_dir, resident_gb):
        self.W = Weights(model_dir)
        c = self.cfg = self.W.cfg
        self.H, self.I = c["hidden_size"], c["intermediate_size"]
        self.NH, self.NKV = c["num_attention_heads"], c["num_key_value_heads"]
        self.HD = self.H // self.NH
        self.NL = int(os.environ.get("OI_LAYERS") or c["num_hidden_layers"])   # smoke tests only
        self.eps = c["rms_norm_eps"]
        inv = 1.0 / (c["rope_theta"] ** (torch.arange(0, self.HD, 2, dtype=torch.float64) / self.HD))
        self.inv = llama3_inv_freq(inv, self.HD, c.get("rope_scaling"))
        self.layers = Layers(self.W, int(resident_gb * 2**30))
        self.norm_w = self.W.get("model.norm.weight").float()
        self.lm_head = self.W.get("lm_head.weight")              # bf16 on GPU
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.bos = c["bos_token_id"]

    def embed(self, ids_2d: torch.Tensor) -> torch.Tensor:
        flat = ids_2d.reshape(-1).cpu().numpy()
        uniq, inv = np.unique(flat, return_inverse=True)
        e = self.W.rows("model.embed_tokens.weight", uniq).to(DEV).float()
        return e[torch.from_numpy(inv).to(DEV)].reshape(*ids_2d.shape, self.H)

    def rms(self, x, w):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps) * w

    def rope(self, x, pos):
        # x (B, heads, T, HD); pos (B, T) int64
        ang = pos.double().unsqueeze(-1) * self.inv.to(x.device)
        cos = ang.cos().float().unsqueeze(1)
        sin = ang.sin().float().unsqueeze(1)
        cos = torch.cat([cos, cos], -1)
        sin = torch.cat([sin, sin], -1)
        h = self.HD // 2
        rot = torch.cat([-x[..., h:], x[..., :h]], -1)
        return x * cos + rot * sin

    def attn(self, w, x, pos, kv=None, key_valid=None):
        """x (B,T,H) fp32. kv: dict with k,v (B,NKV,Tmax,HD) and 'n' filled.
        key_valid (B, Tk) bool marks non-pad keys."""
        B, T, _ = x.shape
        h = self.rms(x, w["input_layernorm.weight"].float())
        q = F.linear(h, w["self_attn.q_proj.weight"].float()).view(B, T, self.NH, self.HD).transpose(1, 2)
        k = F.linear(h, w["self_attn.k_proj.weight"].float()).view(B, T, self.NKV, self.HD).transpose(1, 2)
        v = F.linear(h, w["self_attn.v_proj.weight"].float()).view(B, T, self.NKV, self.HD).transpose(1, 2)
        q, k = self.rope(q, pos), self.rope(k, pos)
        if kv is not None:
            n = kv["n"]
            kv["k"][:, :, n:n + T] = k
            kv["v"][:, :, n:n + T] = v
            k = kv["k"][:, :, :n + T]
            v = kv["v"][:, :, :n + T]
        Tk = k.shape[2]
        rep = self.NH // self.NKV
        k = k.repeat_interleave(rep, 1)
        v = v.repeat_interleave(rep, 1)
        qi = torch.arange(Tk - T, Tk, device=x.device)
        allowed = torch.arange(Tk, device=x.device)[None, :] <= qi[:, None]       # (T, Tk)
        mask = allowed[None, None]
        if key_valid is not None:
            # pad queries see only themselves, so no row is all -inf (NaN);
            # real queries never see pad keys
            diag = torch.arange(Tk, device=x.device)[None, :] == qi[:, None]
            mask = (mask & key_valid[:, None, None, :Tk]) | diag[None, None]
        bias = torch.zeros(mask.shape, device=x.device).masked_fill(~mask, float("-inf"))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        o = o.transpose(1, 2).reshape(B, T, self.H)
        return x + F.linear(o, w["self_attn.o_proj.weight"].float())

    def ffn_in(self, w, x):
        return self.rms(x, w["post_attention_layernorm.weight"].float())

    def logits_chunks(self, h, chunk=16384):
        V = self.lm_head.shape[0]
        for c0 in range(0, V, chunk):
            yield c0, F.linear(h, self.lm_head[c0:c0 + chunk].float())


# ─── variants ─────────────────────────────────────────────────────────────────

class Variant:
    def __init__(self, name, kind, mode="gather", taus=None, hot_frac=0.65):
        self.name, self.kind, self.mode = name, kind, mode
        self.taus = taus            # list per layer (floats) or None -> 0
        self.hot_frac = hot_frac
        self.stats = [dict() for _ in range(64)]

    def ffn(self, M, w, L, xn, hot_order):
        """xn (N, H) fp32 normed FFN input."""
        Wg = w["mlp.gate_proj.weight"].float()
        Wu = w["mlp.up_proj.weight"].float()
        Wd = w["mlp.down_proj.weight"].float()
        if self.kind == "dense":
            return dense_ffn(xn, Wg, Wu, Wd)
        tau = 0.0 if self.taus is None else float(self.taus[L])
        n_hot = int(round(self.hot_frac * M.I))
        order = hot_order[L]
        hot_ids = order[:n_hot].sort().values
        cold_ids = order[n_hot:].sort().values
        cold_rows = lambda ids: (Wu[ids], Wd[:, ids])        # noqa: E731 (exact rows)
        return exact_gate_ffn(xn, Wg, tau, "silu", False, hot_ids, Wu[hot_ids], Wd[:, hot_ids],
                              cold_ids, cold_rows, mode=self.mode, stats=self.stats[L])


def load_hot_order(out, M):
    p = Path(out) / "calib.json"
    if p.exists():
        c = json.load(open(p))
        return [torch.tensor(c["hot_order"][str(L)], device=DEV) for L in range(M.NL)]
    return [torch.arange(M.I, device=DEV) for _ in range(M.NL)]


def load_taus(out, sp):
    c = json.load(open(Path(out) / "calib.json"))
    return [c["taus"][str(L)][f"{sp:.2f}"] for L in range(len(c["taus"]))]


def encode(M, text, n):
    ids = M.tok.encode(text).ids
    if not ids or ids[0] != M.bos:
        ids = [M.bos] + ids
    return ids[:n]


# ─── phase: calib ─────────────────────────────────────────────────────────────

@torch.no_grad()
def phase_calib(M, a):
    out = {"taus": {}, "hot_order": {}, "heldout_profile": {}, "ppltext_profile": {}}
    texts = {"heldout": encode(M, open(a.calib_text, encoding="utf-8").read(), a.calib_tokens),
             "ppltext": encode(M, open(a.ppl_text, encoding="utf-8").read(), int(os.environ.get("OI_TOKENS", 1024)))}
    log("calib tokens", {k: len(v) for k, v in texts.items()})
    xs = {k: M.embed(torch.tensor([v], device=DEV)) for k, v in texts.items()}
    pos = {k: torch.arange(len(v), device=DEV)[None] for k, v in texts.items()}
    taus_by_layer = {}
    for L in range(M.NL):
        w = {k_: t.float() for k_, t in M.layers(L).items()}
        for k in texts:
            x = M.attn(w, xs[k], pos[k])
            xn = M.ffn_in(w, x)[0]
            g = F.linear(xn, w["mlp.gate_proj.weight"].float())
            s = F.silu(g)
            u = F.linear(xn, w["mlp.up_proj.weight"].float())
            A = (s * u)[1:].abs()                    # BOS excluded, as in step 0
            S = s[1:].abs()
            prof = {"pos_frac_legacy": float((s[1:] > 0).float().mean()), **mass_shares(A)}
            if k == "heldout":
                tq = quantile_large(S, SPARSITIES)
                taus_by_layer[L] = {f"{p:.2f}": t for p, t in zip(SPARSITIES, tq)}
                out["taus"][str(L)] = taus_by_layer[L]
                # hot order: per-neuron active frequency at tau(0.40), ties by mass
                act = (S >= taus_by_layer[L]["0.40"]).float().mean(0)
                mass = A.sum(0)
                score = act * 1e6 + mass / mass.max()
                out["hot_order"][str(L)] = torch.argsort(score, descending=True).tolist()
            tl = taus_by_layer[L]
            for p in SPARSITIES:
                keep = S >= tl[f"{p:.2f}"]
                n = keep.shape[0]
                win = keep[4:].clone()
                for d in range(1, 5):
                    win |= keep[4 - d:n - d]
                prof[f"sp{p:.2f}"] = {
                    "active": float(keep.float().mean()),
                    "mass_kept": float((A * keep).sum() / A.sum()),
                    "union5": float(win.float().mean()),
                }
            out[f"{k}_profile"][str(L)] = prof
            # residual continues dense
            xs[k] = x + F.linear(s * u, w["mlp.down_proj.weight"].float())[None]
        log(f"calib L{L} k90={out['ppltext_profile'][str(L)]['k90']:.3f} "
            f">0={out['ppltext_profile'][str(L)]['pos_frac_legacy']:.3f} "
            f"tau40={taus_by_layer[L]['0.40']:.4f}")
    # dense PPL on the step-0 text as a cross-check of the forward (step 0: 12.98)
    h = M.rms(xs["ppltext"][0], M.norm_w)
    ids = torch.tensor(texts["ppltext"], device=DEV)
    out["dense_ppl_check"] = nll_stats(M, h, ids)
    summ = {}
    for k in ("ppltext_profile", "heldout_profile"):
        lay = out[k].values()
        summ[k] = {m: float(np.mean([d[m] for d in lay])) for m in ("pos_frac_legacy", "k90", "k95", "k99")}
        for p in SPARSITIES:
            for m in ("active", "mass_kept", "union5"):
                summ[k][f"sp{p:.2f}_{m}"] = float(np.mean([d[f"sp{p:.2f}"][m] for d in lay]))
    out["summary"] = summ
    json.dump(out, open(Path(a.out) / "calib.json", "w"))
    log("calib summary", json.dumps(summ["ppltext_profile"]), "dense ppl", out["dense_ppl_check"])


def nll_stats(M, h, ids):
    """h (T, H) final-normed; returns PPL over all T-1 predictions and over
    the second half (positions >= T/2, llama-perplexity's scoring window)."""
    T = h.shape[0]
    lse = torch.full((T,), -float("inf"), device=DEV)
    tgt = torch.zeros(T, device=DEV)
    nxt = ids[1:]
    for c0, lg in M.logits_chunks(h):
        lse = torch.logaddexp(lse, torch.logsumexp(lg, 1))
        sel = (nxt >= c0) & (nxt < c0 + lg.shape[1])
        r = sel.nonzero(as_tuple=True)[0]
        tgt[r] = lg[r, nxt[r] - c0]
    nll = (lse[:-1] - tgt[:-1])
    half = T // 2
    return {"tokens": T, "nll": float(nll.mean()), "ppl": float(nll.mean().exp()),
            "ppl_2nd_half": float(nll[half:].mean().exp()), "n_2nd_half": int(nll[half:].numel())}


# ─── phase: ppl ───────────────────────────────────────────────────────────────

@torch.no_grad()
def phase_ppl(M, a):
    hot_order = load_hot_order(a.out, M)
    V = [Variant("dense", "dense"),
         Variant("exact_tau0_masked", "exact", "masked"),
         Variant("exact_tau0_gather", "exact", "gather")]
    for p in a.sweep:
        V.append(Variant(f"cats_{p:.2f}", "exact", "gather", load_taus(a.out, p)))
    ids_l = encode(M, open(a.ppl_text, encoding="utf-8").read(), int(os.environ.get("OI_TOKENS", 1024)))
    ids = torch.tensor(ids_l, device=DEV)
    x0 = M.embed(ids[None])
    pos = torch.arange(len(ids_l), device=DEV)[None]
    xs = {v.name: x0.clone() for v in V}
    t0 = time.time()
    for L in range(M.NL):
        w = {k_: t.float() for k_, t in M.layers(L).items()}
        for v in V:
            x = M.attn(w, xs[v.name], pos)
            xn = M.ffn_in(w, x)[0]
            xs[v.name] = x + v.ffn(M, w, L, xn, hot_order)[None]
        if L % 8 == 7:
            log(f"ppl L{L} {time.time()-t0:.0f}s")
    res = {}
    hd = M.rms(xs["dense"][0], M.norm_w)
    for v in V:
        h = M.rms(xs[v.name][0], M.norm_w)
        r = nll_stats(M, h, ids)
        r["hidden_equal_dense"] = bool(torch.equal(h, hd))
        maxd, agree = 0.0, None
        best_v = torch.full((h.shape[0],), -float("inf"), device=DEV); arg_v = torch.zeros(h.shape[0], dtype=torch.long, device=DEV)
        best_d = best_v.clone(); arg_d = arg_v.clone()
        all_eq = True
        for (c0, lg), (_, ld) in zip(M.logits_chunks(h), M.logits_chunks(hd)):
            all_eq &= bool(torch.equal(lg, ld))
            maxd = max(maxd, float((lg - ld).abs().max()))
            mv, iv = lg.max(1); md, idd = ld.max(1)
            upd = mv > best_v; best_v = torch.where(upd, mv, best_v); arg_v = torch.where(upd, iv + c0, arg_v)
            upd = md > best_d; best_d = torch.where(upd, md, best_d); arg_d = torch.where(upd, idd + c0, arg_d)
        r["logits_bitwise_equal_dense"] = all_eq
        r["logits_max_abs_diff"] = maxd
        r["top1_agree_dense"] = float((arg_v == arg_d).float().mean())
        st = [s for s in v.stats if s]
        if st:
            r["active_frac"] = float(np.mean([s["active"] / (s["tokens"] * s["I"]) for s in st]))
            r["union_rows_frac"] = float(np.mean([s["union_rows"] / (s["calls"] * s["I"]) for s in st]))
        res[v.name] = r
        log(v.name, json.dumps(r))
    base = res["dense"]["ppl"]
    for v in res.values():
        v["ppl_rise_pct"] = 100 * (v["ppl"] / base - 1)
    res["_meta"] = {"tokens": len(ids_l), "seconds": time.time() - t0,
                    "bytes_read": M.W.bytes_read, "read_s": M.W.read_s}
    json.dump(res, open(Path(a.out) / "ppl.json", "w"), indent=1)


# ─── phase: gen ───────────────────────────────────────────────────────────────

SYSTEM = "You are a careful research librarian. Answer the question directly."


def chat_ids(M, q):
    t = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n"
         f"<|im_start|>assistant\n")
    return [M.bos] + M.tok.encode(t, add_special_tokens=False).ids


@torch.no_grad()
def phase_gen(M, a):
    hot_order = load_hot_order(a.out, M)
    qs = json.load(open(a.questions))[: int(os.environ.get("OI_QUESTIONS", 999))]
    prompts = [chat_ids(M, q["question"]) for q in qs]
    B = len(prompts)
    Lp = max(len(p) for p in prompts)
    N = a.gen_tokens
    Tmax = Lp + N
    pad = torch.tensor([Lp - len(p) for p in prompts], device=DEV)
    ids0 = torch.zeros(B, Lp, dtype=torch.long, device=DEV)
    for b, p in enumerate(prompts):
        ids0[b, Lp - len(p):] = torch.tensor(p, device=DEV)
    key_valid = torch.arange(Tmax, device=DEV)[None, :] >= pad[:, None]
    V = [Variant("dense", "dense")]
    for name in a.gen_variants:
        if name == "exact_tau0_masked":
            V.append(Variant(name, "exact", "masked"))
        elif name == "exact_tau0_gather":
            V.append(Variant(name, "exact", "gather"))
        else:                                   # cats_0.40 etc.
            p = float(name.split("_")[1])
            V.append(Variant(name, "exact", "gather", load_taus(a.out, p)))
    kv = {v.name: [{"k": torch.zeros(B, M.NKV, Tmax, M.HD, device=DEV),
                    "v": torch.zeros(B, M.NKV, Tmax, M.HD, device=DEV), "n": 0}
                   for _ in range(M.NL)] for v in V}
    toks = {v.name: [] for v in V}
    margins = {v.name: [] for v in V}
    logit_diff = {v.name: [] for v in V}
    cur = {v.name: ids0 for v in V}
    t0 = time.time()
    for step in range(N):
        T = cur[V[0].name].shape[1]
        n = kv[V[0].name][0]["n"]
        pos = (torch.arange(n, n + T, device=DEV)[None, :] - pad[:, None]).clamp(min=0)
        xs = {v.name: M.embed(cur[v.name]) for v in V}
        for L in range(M.NL):
            w = {k_: t.float() for k_, t in M.layers(L).items()}
            for v in V:
                x = M.attn(w, xs[v.name], pos, kv[v.name][L], key_valid)
                xn = M.ffn_in(w, x).reshape(-1, M.H)
                xs[v.name] = x + v.ffn(M, w, L, xn, hot_order).reshape(x.shape)
        for v in V:
            for L in range(M.NL):
                kv[v.name][L]["n"] += T
        last = {}
        for v in V:
            h = M.rms(xs[v.name][:, -1], M.norm_w)
            last[v.name] = torch.cat([lg for _, lg in M.logits_chunks(h)], 1)
        for v in V:
            lg = last[v.name]
            top2 = lg.topk(2, 1)
            nt = top2.indices[:, 0]
            toks[v.name].append(nt.tolist())
            margins[v.name].append((top2.values[:, 0] - top2.values[:, 1]).tolist())
            logit_diff[v.name].append(float((lg - last["dense"]).abs().max()))
            cur[v.name] = nt[:, None]
        if step % 8 == 0:
            log(f"gen step {step}/{N} {time.time()-t0:.0f}s read {M.W.bytes_read/1e9:.1f} GB")
    res = {"questions": B, "tokens_per_answer": N, "seconds": time.time() - t0, "variants": {}}
    dense = np.array(toks["dense"]).T          # (B, N)
    for v in V:
        tv = np.array(toks[v.name]).T
        same_prefix = []
        for b in range(B):
            d = np.nonzero(tv[b] != dense[b])[0]
            same_prefix.append(int(d[0]) if d.size else N)
        st = [s for s in v.stats if s]
        res["variants"][v.name] = {
            "answers_identical": int(sum(sp == N for sp in same_prefix)),
            "first_divergence": same_prefix,
            "logits_max_abs_diff_last_token": max(logit_diff[v.name]),
            "min_dense_margin_at_divergence": [
                float(margins["dense"][sp][b]) if sp < N else None for b, sp in enumerate(same_prefix)],
            "active_frac": float(np.mean([s["active"] / (s["tokens"] * s["I"]) for s in st])) if st else 1.0,
            "text": [M.tok.decode(tv[b].tolist()) for b in range(B)],
        }
        log(v.name, "identical", res["variants"][v.name]["answers_identical"], "/", B,
            "maxdiff", res["variants"][v.name]["logits_max_abs_diff_last_token"])
    json.dump(res, open(Path(a.out) / f"gen{a.tag}.json", "w"), indent=1)


# ─── phase: ffnbench ──────────────────────────────────────────────────────────

@torch.no_grad()
def phase_ffnbench(M, a):
    """One FFN layer resident in fp16, batch 1, CUDA-timed: dense vs exact-gate
    gather (all rows hot, so only compute is measured, no transfers)."""
    L = 16
    w = M.layers(L)
    Wg, Wu, Wd = (w[f"mlp.{n}_proj.weight"].half() for n in ("gate", "up", "down"))
    c = json.load(open(Path(a.out) / "calib.json"))
    x = torch.randn(1, M.H, device=DEV, dtype=torch.half) * 0.05
    # realistic input: a normed hidden state from the calibration text is not
    # stored; use the tau of this layer on random-normal input scaled so the
    # active fraction matches calibration (reported alongside).
    all_ids = torch.arange(M.I, device=DEV)
    empty = torch.zeros(0, dtype=torch.long, device=DEV)

    def timeit(fn, n=200):
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(n):
            fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / n

    res = {"layer": L, "dtype": "fp16", "batch": 1}
    res["dense_ms"] = timeit(lambda: dense_ffn(x, Wg, Wu, Wd))
    s = F.silu(F.linear(x, Wg)).abs().float()
    for p in (0.0,) + tuple(a.sweep):
        tau = 0.0 if p == 0 else float(quantile_large(s, [p])[0])
        st = {}
        fn = lambda: exact_gate_ffn(x, Wg, tau, "silu", False, all_ids, Wu, Wd,  # noqa: E731
                                    empty, None, mode="gather", stats=None)
        exact_gate_ffn(x, Wg, tau, "silu", False, all_ids, Wu, Wd, empty, None, stats=st)
        res[f"gather_{p:.2f}_ms"] = timeit(fn)
        res[f"gather_{p:.2f}_active"] = st["active"] / st["I"]
    res["calib_tau40_layer16"] = c["taus"][str(L)]["0.40"]
    res["peak_vram_mib"] = torch.cuda.max_memory_allocated() / 2**20
    log(json.dumps(res))
    json.dump(res, open(Path(a.out) / "ffnbench.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["calib", "ppl", "gen", "ffnbench"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ppl-text")
    ap.add_argument("--calib-text")
    ap.add_argument("--calib-tokens", type=int, default=2048)
    ap.add_argument("--questions")
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--sweep", type=float, nargs="*", default=[0.2, 0.3, 0.4, 0.5, 0.6])
    ap.add_argument("--gen-variants", nargs="*", default=["exact_tau0_masked", "exact_tau0_gather"])
    ap.add_argument("--tag", default="", help="suffix for the gen output file")
    ap.add_argument("--resident-gb", type=float, default=0.0)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    M = Model(a.model, a.resident_gb)
    t0 = time.time()
    {"calib": phase_calib, "ppl": phase_ppl, "gen": phase_gen, "ffnbench": phase_ffnbench}[a.phase](M, a)
    if DEV.type == "cuda":  # noqa
        log(f"phase {a.phase} done in {time.time()-t0:.0f}s; peak VRAM "
            f"{torch.cuda.max_memory_allocated()/2**20:.0f} MiB allocated, "
            f"{torch.cuda.max_memory_reserved()/2**20:.0f} MiB reserved")


if __name__ == "__main__":
    main()
