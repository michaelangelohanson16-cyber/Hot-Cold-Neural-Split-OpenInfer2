"""
Shared fixtures: build a synthetic cold-weight file that is byte-compatible
with splitter.split_dense()'s on-disk format, without needing any model.

The record layout is replicated from split_dense's inner loop exactly:
  section header:  struct '<HHIq'  (layer:u16, expert:u16, n_cold:u32, reserved:u64)
  neuron record:   '<H' neuron_id + gate/up/down INT4-packed + 3x FP16 group scales,
                   padded to the 64-byte record boundary
  section:         padded to the 4096-byte page boundary
"""

import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splitter import (  # noqa: E402
    ALIGN_BYTES,
    HIDDEN_DIM,
    NEURON_RECORD_BYTES,
    PAGE_ALIGN,
    dequantize_int4,
    quantize_int4,
)


def build_synthetic_cold_file(out_dir, num_layers=4, n_cold=16, seed=0):
    """
    Write cold_weights.bin + cold_index.npy for `num_layers` layers with
    `n_cold` random cold neurons each (non-consecutive neuron ids).

    Returns (cold_bin_path, cold_index_path, config, truth) where truth maps
    (layer, neuron_id) -> dict of the INT4-roundtripped float32 vectors the
    store must reproduce exactly (same quantize/dequantize code path).
    """
    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    cold_bin = os.path.join(out_dir, "cold_weights.bin")
    cold_idx = os.path.join(out_dir, "cold_index.npy")

    cold_index = np.zeros((num_layers, 1, 2), dtype=np.uint64)
    truth = {}
    byte_cursor = 0

    with open(cold_bin, "wb") as f:
        for li in range(num_layers):
            # Non-consecutive ids, deliberately unsorted on disk order.
            nids = torch.randperm(n_cold * 3)[:n_cold].tolist()

            section_start = byte_cursor
            header = struct.pack("<HHIq", li, 0, n_cold, 0)
            f.write(header)
            byte_cursor += len(header)

            for nid in nids:
                gate = torch.randn(HIDDEN_DIM)
                up = torch.randn(HIDDEN_DIM)
                down = torch.randn(HIDDEN_DIM)

                gate_p, gate_s = quantize_int4(gate)
                up_p, up_s = quantize_int4(up)
                down_p, down_s = quantize_int4(down)

                # The disk format stores group scales as FP16, so the values
                # the store reproduces are dequantized with FP16-rounded
                # scales -- the ground truth must apply the same rounding.
                # (On-disk fidelity = INT4 weights + FP16 scales, not INT4 +
                # FP32 scales; the difference is real and this fixture
                # originally missed it.)
                truth[(li, nid)] = {
                    "gate": dequantize_int4(
                        gate_p, gate_s.to(torch.float16).float(), HIDDEN_DIM),
                    "up": dequantize_int4(
                        up_p, up_s.to(torch.float16).float(), HIDDEN_DIM),
                    "down": dequantize_int4(
                        down_p, down_s.to(torch.float16).float(), HIDDEN_DIM),
                }

                record = struct.pack("<H", nid)
                record += gate_p + up_p + down_p
                record += gate_s.to(torch.float16).numpy().tobytes()
                record += up_s.to(torch.float16).numpy().tobytes()
                record += down_s.to(torch.float16).numpy().tobytes()
                pad = (ALIGN_BYTES - len(record) % ALIGN_BYTES) % ALIGN_BYTES
                record += b"\x00" * pad
                assert len(record) == NEURON_RECORD_BYTES
                f.write(record)
                byte_cursor += len(record)

            remainder = byte_cursor % PAGE_ALIGN
            if remainder:
                pad = PAGE_ALIGN - remainder
                f.write(b"\x00" * pad)
                byte_cursor += pad

            cold_index[li, 0] = (section_start, n_cold)

    np.save(cold_idx, cold_index)
    config = {"hidden_dim": HIDDEN_DIM, "intermediate_dim": n_cold * 3}
    return cold_bin, cold_idx, config, truth
