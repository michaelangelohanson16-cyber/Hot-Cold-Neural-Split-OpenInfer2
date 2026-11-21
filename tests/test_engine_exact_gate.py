"""
Engine-level exactness of the exact gate (the mask fix).

Unlike test_ffn_exactness.py, which checks the hot+cold decomposition on a
reference function, these tests drive the engine's own SparseFFNLayer.forward
over a real on-disk split (hot .pt file + cold_weights.bin in split_dense's
record layout) and compare with a dense FFN over the very same weight values:

  * tau = 0, mode "masked": bitwise identical to the dense FFN;
  * tau = 0, mode "gather": equal up to fp32 summation order;
  * tau > 0: equal to the dense FFN with act(g) zeroed where |act(g)| < tau
    (the CATS definition), i.e. nothing else is dropped;
  * a wrong predictor changes nothing: misses are fetched, not dropped.
Plus unit tests of magnitude.py and of the profiler's magnitude hooks.
"""

import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import SparseFFNLayer, dense_ffn  # noqa: E402
from magnitude import MassAccumulator, active_mask, mass_shares, quantile_large  # noqa: E402
from splitter import (  # noqa: E402
    HIDDEN_DIM, PAGE_ALIGN, ColdWeightStore, _pack_records,
)

H = HIDDEN_DIM
I = 48


def _build_split(tmp, seed=0, n_hot=16):
    torch.manual_seed(seed)
    gate = torch.randn(I, H) * 0.03
    up = torch.randn(I, H) * 0.03
    down = torch.randn(H, I) * 0.03
    perm = torch.randperm(I)
    hot_ids = perm[:n_hot].sort().values
    cold_ids = perm[n_hot:].sort().values
    hot_dir = os.path.join(tmp, "hot_weights")
    os.makedirs(hot_dir, exist_ok=True)
    torch.save({
        "neuron_ids": hot_ids,
        "gate": gate[hot_ids].to(torch.float8_e4m3fn),
        "up": up[hot_ids].to(torch.float8_e4m3fn),
        "down": down[:, hot_ids].to(torch.float8_e4m3fn),
    }, os.path.join(hot_dir, "layer_000_expert_000.pt"))
    cold_bin = os.path.join(tmp, "cold_weights.bin")
    with open(cold_bin, "wb") as f:
        f.write(struct.pack("<HHIq", 0, 0, len(cold_ids), 0))
        recs = _pack_records(cold_ids, gate[cold_ids], up[cold_ids], down[:, cold_ids].T)
        f.write(recs.tobytes())
        n = 16 + recs.nbytes
        if n % PAGE_ALIGN:
            f.write(b"\0" * (PAGE_ALIGN - n % PAGE_ALIGN))
    idx = np.zeros((1, 1, 2), dtype=np.uint64)
    idx[0, 0] = (0, len(cold_ids))
    np.save(os.path.join(tmp, "cold_index.npy"), idx)
    cfg = {"hidden_dim": H, "intermediate_dim": I, "rms_norm_eps": 1e-5, "hidden_act": "silu"}
    store = ColdWeightStore(cold_bin, os.path.join(tmp, "cold_index.npy"), cfg)
    return hot_dir, store, cfg, hot_ids, cold_ids


def _layer(tmp, tau=0.0, mode="gather", predictor=None, seed=0):
    hot_dir, store, cfg, hot_ids, cold_ids = _build_split(tmp, seed)
    ex = ThreadPoolExecutor(2)
    layer = SparseFFNLayer(0, hot_dir, store, predictor, cfg, torch.device("cpu"), ex,
                           gate_mode="exact", tau=tau, exact_mode=mode)
    return layer, store


def _effective_weights(layer, store):
    """The exact weight values the layer computes with, as full matrices."""
    gate = layer.gate_all.float()
    up = torch.zeros(I, H)
    down = torch.zeros(H, I)
    hid = layer.hot["neuron_ids"]
    up[hid] = layer.hot["up"].float()
    down[:, hid] = layer.hot["down"].float()
    _, cu, cd = store.read_neurons(0, 0, layer.cold_ids.tolist())
    up[layer.cold_ids] = cu
    down[:, layer.cold_ids] = cd
    return gate, up, down


def _x(seed=1, n=5):
    torch.manual_seed(seed)
    return torch.randn(1, n, H)


def test_tau0_masked_is_bitwise_dense(tmp_path):
    layer, store = _layer(str(tmp_path), tau=0.0, mode="masked")
    g, u, d = _effective_weights(layer, store)
    x = _x()
    with torch.no_grad():
        y = layer(x)
        xn = layer.input_layernorm(x).reshape(-1, H)
        ref = x + dense_ffn(xn, g, u, d).reshape(x.shape)
    assert torch.equal(y, ref), f"max diff {(y - ref).abs().max().item():.3e}"
    store.close()


def test_tau0_gather_equals_dense(tmp_path):
    layer, store = _layer(str(tmp_path), tau=0.0, mode="gather")
    g, u, d = _effective_weights(layer, store)
    x = _x()
    with torch.no_grad():
        y = layer(x)
        xn = layer.input_layernorm(x).reshape(-1, H)
        ref = x + dense_ffn(xn, g, u, d).reshape(x.shape)
    assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5), (y - ref).abs().max()
    assert layer.stats["active"] == layer.stats["tokens"] * I     # nothing skipped
    store.close()


def test_tau_positive_drops_only_below_threshold(tmp_path):
    for mode in ("gather", "masked"):
        sub = tmp_path / mode
        sub.mkdir()
        layer0, store0 = _layer(str(sub), tau=0.0)
        g, u, d = _effective_weights(layer0, store0)
        x = _x(n=7)
        xn = layer0.input_layernorm(x).reshape(-1, H)
        s = F.silu(F.linear(xn, g))
        tau = float(s.abs().median())
        store0.close()
        sub2 = tmp_path / (mode + "_t")
        sub2.mkdir()
        layer, store = _layer(str(sub2), tau=tau, mode=mode)
        with torch.no_grad():
            y = layer(x)
            ref = x + F.linear(s * (s.abs() >= tau) * F.linear(xn, u), d).reshape(x.shape)
        assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5), (mode, (y - ref).abs().max())
        frac = layer.stats["active"] / (layer.stats["tokens"] * I)
        assert 0.3 < frac < 0.7, frac
        store.close()


class _WrongPredictor:
    """Predicts a fixed, mostly wrong set of cold neurons."""
    def __init__(self, ids):
        self.ids = ids

    def predict(self, layer_idx, expert_idx, h, threshold=0.5):
        m = torch.zeros(1, I, dtype=torch.bool)
        m[0, self.ids] = True
        return m


def test_predictor_miss_costs_time_not_accuracy(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    layer_np, s1 = _layer(str(a), tau=0.0)
    x = _x(n=3)
    with torch.no_grad():
        y0 = layer_np(x)
    cold = layer_np.cold_ids.tolist()
    layer_p, s2 = _layer(str(b), tau=0.0, predictor=_WrongPredictor(cold[:3]))
    with torch.no_grad():
        fut = layer_p.submit_cold_fetch(x)
        y1 = layer_p(x, fut)
    assert torch.equal(y0, y1)
    assert layer_p.stats["prefetch_hits"] == 3
    assert layer_p.stats["prefetch_misses"] == len(cold) - 3
    s1.close(); s2.close()


# ─── magnitude.py ─────────────────────────────────────────────────────────────

def test_mass_shares_known_values():
    a = torch.tensor([[10.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    k = mass_shares(a, (0.5, 0.99))
    # token 0 needs 1/4 of neurons for any level; token 1 needs 2/4 for 50%, 4/4 for 99%
    assert abs(k["k50"] - (0.25 + 0.5) / 2) < 1e-6
    assert abs(k["k99"] - (0.25 + 1.0) / 2) < 1e-6


def test_active_mask_tau0_keeps_everything():
    s = torch.tensor([[-0.0, 0.0, -1e-9, 3.0]])
    assert active_mask(s, 0.0).all()
    assert active_mask(s, 1e-6).tolist() == [[False, False, False, True]]


def test_quantile_large_matches_torch():
    x = torch.randn(10_000)
    for q in (0.1, 0.5, 0.9):
        assert abs(quantile_large(x, [q])[0] - float(torch.quantile(x, q))) < 1e-2


def test_sign_mask_is_lossy_magnitude_is_not():
    """The bug being fixed: silu(g) > 0 drops neurons that carry mass."""
    torch.manual_seed(0)
    g = torch.randn(64, 512)
    u = torch.randn(64, 512)
    s = F.silu(g)
    a = s * u
    sign_kept = (a.abs() * (s > 0)).sum() / a.abs().sum()
    exact_kept = (a.abs() * active_mask(s, 0.0)).sum() / a.abs().sum()
    assert exact_kept == 1.0
    assert sign_kept < 0.99     # silu's negative lobe carries real mass


# ─── profiler hooks ───────────────────────────────────────────────────────────

class _MLP(nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _Tiny(nn.Module):
    def __init__(self, h=16, i=40, n=2):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        for _ in range(n):
            L = nn.Module()
            L.mlp = _MLP(h, i)
            self.model.layers.append(L)

    def forward(self, x):
        for L in self.model.layers:
            x = x + L.mlp(x)
        return x


def test_profiler_records_mass_not_sign():
    from profiler import NeuronProfiler
    torch.manual_seed(0)
    m = _Tiny()
    prof = NeuronProfiler(m, "llama")
    x = torch.randn(1, 30, 16)
    with torch.no_grad():
        m(x)
        mlp = m.model.layers[0].mlp
        s = F.silu(mlp.gate_proj(x))
        a = (s * mlp.up_proj(x)).reshape(-1, 40)
    acc = prof.acc[0][0]
    assert acc.tokens == 30
    assert torch.allclose(acc.mass.float(), a.abs().sum(0), rtol=1e-5, atol=1e-6)
    summ = acc.summary()
    assert abs(summ["pos_frac_legacy"] - float((s > 0).float().mean())) < 1e-6
    assert 0 < summ["k90"] <= summ["k95"] <= summ["k99"] <= 1
    taus = prof.calibrate_thresholds((0.5,))
    assert set(taus) == {0, 1}
    prof.remove_hooks()


def test_accumulator_thresholds_count_active():
    acc = MassAccumulator(4)
    acc.set_thresholds({"t": 0.5})
    g = torch.tensor([[0.1, -0.6, 0.7, 0.0]])
    acc.feed(g, g)
    assert acc.frequencies("t") == [0.0, 1.0, 1.0, 0.0]
