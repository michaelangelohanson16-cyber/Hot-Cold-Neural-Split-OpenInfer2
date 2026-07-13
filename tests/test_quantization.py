"""
INT4 quantization: round-trip fidelity within the absmax bound, edge cases,
and packing invariants. These are the precision guarantees the README
claims are 'the only precision loss anywhere in the pipeline'.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splitter import dequantize_int4, quantize_int4  # noqa: E402


def test_roundtrip_within_absmax_bound():
    """Absmax group quantization guarantees per-element error <= scale/2,
    where scale = group_absmax / 7. Verify the bound holds exactly."""
    torch.manual_seed(0)
    for group_size in (128,):
        x = torch.randn(4096) * 3.0
        packed, scales = quantize_int4(x, group_size)
        y = dequantize_int4(packed, scales, 4096, group_size)

        groups_x = x.reshape(-1, group_size)
        groups_err = (y - x).abs().reshape(-1, group_size)
        bound = (groups_x.abs().max(dim=1).values.clamp(min=1e-8) / 7.0) / 2.0
        assert (groups_err.max(dim=1).values <= bound + 1e-6).all(), \
            "round-trip error exceeded the absmax INT4 bound"


def test_roundtrip_zero_vector():
    x = torch.zeros(4096)
    packed, scales = quantize_int4(x)
    y = dequantize_int4(packed, scales, 4096)
    assert torch.allclose(y, x), "zero vector must survive round-trip exactly"


def test_packing_density():
    """Two INT4 values per byte: packed length must be N // 2."""
    x = torch.randn(4096)
    packed, _ = quantize_int4(x)
    assert len(packed) == 4096 // 2


def test_short_vector_below_group_size():
    """N < group_size is explicitly allowed by the implementation."""
    x = torch.randn(100)
    packed, scales = quantize_int4(x, group_size=128)
    y = dequantize_int4(packed, scales, 100, group_size=128)
    err = (y - x).abs().max().item()
    bound = (x.abs().max().item() / 7.0) / 2.0
    assert err <= bound + 1e-6


def test_extreme_values_survive_sign_and_scale():
    """Large-magnitude and negative values keep sign and approximate scale."""
    x = torch.tensor([100.0, -100.0, 50.0, -50.0] * 32)  # len 128, one group
    packed, scales = quantize_int4(x)
    y = dequantize_int4(packed, scales, 128)
    assert (torch.sign(y) == torch.sign(x)).all()
    assert (y - x).abs().max() <= (100.0 / 7.0) / 2.0 + 1e-4


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
