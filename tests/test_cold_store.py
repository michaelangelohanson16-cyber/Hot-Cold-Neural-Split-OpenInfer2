"""
ColdWeightStore vs. the on-disk format: a synthetic cold file written in
split_dense's exact record layout must read back byte-faithfully -- right
neurons, right values, right shapes, correct handling of missing ids and
empty sections, and a warm offset cache that changes nothing.
"""

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splitter import HIDDEN_DIM, ColdWeightStore  # noqa: E402
from tests.common import build_synthetic_cold_file  # noqa: E402


def _make_store(tmp, num_layers=4, n_cold=16, seed=0):
    cold_bin, cold_idx, config, truth = build_synthetic_cold_file(
        tmp, num_layers=num_layers, n_cold=n_cold, seed=seed
    )
    import numpy as np
    return ColdWeightStore(cold_bin, cold_idx, config), truth


def test_readback_values_exact():
    """Values read through the store must equal the independently computed
    INT4 round-trip -- same quantization code path, so equality is exact."""
    with tempfile.TemporaryDirectory() as tmp:
        store, truth = _make_store(tmp)
        try:
            for (li, nid), vecs in truth.items():
                out = store.read_neurons(li, 0, [nid])
                assert out is not None, f"neuron {nid} missing from layer {li}"
                gate, up, down = out
                assert gate.shape == (1, HIDDEN_DIM)
                assert up.shape == (1, HIDDEN_DIM)
                assert down.shape == (HIDDEN_DIM, 1)
                assert torch.equal(gate[0], vecs["gate"])
                assert torch.equal(up[0], vecs["up"])
                assert torch.equal(down[:, 0], vecs["down"])
        finally:
            store.close()


def test_batch_read_shapes_and_order():
    with tempfile.TemporaryDirectory() as tmp:
        store, truth = _make_store(tmp)
        try:
            layer0_ids = sorted(nid for (li, nid) in truth if li == 0)
            out = store.read_neurons(0, 0, layer0_ids)
            gate, up, down = out
            n = len(layer0_ids)
            assert gate.shape == (n, HIDDEN_DIM)
            assert down.shape == (HIDDEN_DIM, n)
            for row, nid in enumerate(layer0_ids):
                assert torch.equal(gate[row], truth[(0, nid)]["gate"])
        finally:
            store.close()


def test_missing_ids_skipped_and_all_missing_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        store, truth = _make_store(tmp)
        try:
            layer0_ids = [nid for (li, nid) in truth if li == 0]
            absent = max(layer0_ids) + 1000
            # mixed present/absent: absent id silently skipped
            out = store.read_neurons(0, 0, [layer0_ids[0], absent])
            assert out is not None and out[0].shape[0] == 1
            # all absent: None
            assert store.read_neurons(0, 0, [absent, absent + 1]) is None
        finally:
            store.close()


def test_offset_cache_consistency():
    """Second read of the same section uses the cached offset map and must
    return identical values."""
    with tempfile.TemporaryDirectory() as tmp:
        store, truth = _make_store(tmp)
        try:
            layer1_ids = [nid for (li, nid) in truth if li == 1][:4]
            first = store.read_neurons(1, 0, layer1_ids)
            assert (1, 0) in store._offset_cache
            second = store.read_neurons(1, 0, layer1_ids)
            for a, b in zip(first, second):
                assert torch.equal(a, b)
        finally:
            store.close()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
