"""
The engine's central mathematical claim, tested directly: because down_proj
is linear in the neuron dimension, computing the FFN as (hot half) + (cold
half) is EXACT -- identical to computing the full FFN with the same
per-neuron weight precisions in one pass. The only approximation in the
pipeline is the quantization itself, which is measured (not just asserted)
here.

Covers both activation conventions the engine supports:
  standard gated FFN:   silu(gate) * up
  dReLU / act_on_both:  relu(gate) * relu(up)   (TurboSparse/Bamboo family)
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splitter import HIDDEN_DIM, dequantize_int4, quantize_int4  # noqa: E402

I_HOT, I_COLD = 32, 32
I_TOTAL = I_HOT + I_COLD


def _int4_roundtrip_rows(w):
    """Apply the full cold-path precision to each row of w: INT4 groupwise
    weights AND the FP16 rounding the disk format applies to group scales."""
    out = torch.empty_like(w)
    for r in range(w.shape[0]):
        packed, scales = quantize_int4(w[r])
        out[r] = dequantize_int4(
            packed, scales.to(torch.float16).float(), w.shape[1])
    return out


def _fp8_roundtrip(w):
    """Apply the hot-path precision (FP8 e4m3) exactly as the splitter does."""
    return w.to(torch.float8_e4m3fn).to(torch.float32)


def _ffn(x, wg, wu, wd, act_on_both):
    g = wg @ x
    u = wu @ x
    h = F.relu(g) * F.relu(u) if act_on_both else F.silu(g) * u
    return wd @ h


def _run_case(act_on_both, seed):
    torch.manual_seed(seed)
    x = torch.randn(HIDDEN_DIM)
    wg = torch.randn(I_TOTAL, HIDDEN_DIM) * 0.05
    wu = torch.randn(I_TOTAL, HIDDEN_DIM) * 0.05
    wd = torch.randn(HIDDEN_DIM, I_TOTAL) * 0.05

    hot = slice(0, I_HOT)
    cold = slice(I_HOT, I_TOTAL)

    # Mixed-precision weights: hot rows/cols at FP8, cold rows/cols at INT4 --
    # exactly the precisions the split assigns.
    wg_m = torch.cat([_fp8_roundtrip(wg[hot]), _int4_roundtrip_rows(wg[cold])])
    wu_m = torch.cat([_fp8_roundtrip(wu[hot]), _int4_roundtrip_rows(wu[cold])])
    wd_m = torch.cat(
        [_fp8_roundtrip(wd[:, hot]), _int4_roundtrip_rows(wd[:, cold].T).T],
        dim=1,
    )

    # Reference: full FFN over all neurons at once, mixed precision.
    y_full = _ffn(x, wg_m, wu_m, wd_m, act_on_both)

    # Engine decomposition: hot half + cold half, computed separately.
    y_hot = _ffn(x, wg_m[hot], wu_m[hot], wd_m[:, hot], act_on_both)
    y_cold = _ffn(x, wg_m[cold], wu_m[cold], wd_m[:, cold], act_on_both)
    y_split = y_hot + y_cold

    # Exactness of the decomposition (fp32 summation-order noise only).
    assert torch.allclose(y_split, y_full, atol=1e-5, rtol=1e-5), (
        f"hot+cold decomposition diverged from the full FFN "
        f"(act_on_both={act_on_both}): max abs diff "
        f"{(y_split - y_full).abs().max().item():.3e}"
    )

    # Quantization fidelity, measured against full fp32. The expected figure
    # is not small: INT4 absmax over Gaussian 128-groups has a quantization
    # step of absmax/7 with absmax ~ 2.8*sigma, giving ~12% RMS per-weight
    # error, and dot-product outputs inherit roughly that relative error.
    # Measured values of ~9-16% on random weights are therefore the true
    # cost of the scheme, reported here as data. The assertion's job is only
    # to catch catastrophic breakage (wrong scales, sign corruption), which
    # would show up as >>25%.
    y_fp32 = _ffn(x, wg, wu, wd, act_on_both)
    rel = ((y_full - y_fp32).norm() / y_fp32.norm().clamp(min=1e-9)).item()
    print(f"  act_on_both={act_on_both}: decomposition exact; "
          f"mixed-precision vs fp32 relative L2 error = {rel:.4%}")
    assert rel < 0.25, f"quantization error implies broken scales: {rel:.2%}"
    return rel


def test_decomposition_exact_swiglu():
    for seed in range(3):
        _run_case(act_on_both=False, seed=seed)


def test_decomposition_exact_drelu():
    for seed in range(3):
        _run_case(act_on_both=True, seed=seed)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
