"""
Component benchmarks that require no model download.

1. ColdWeightStore.read_neurons throughput -- the known Python-loop
   bottleneck, measured rather than asserted. Reported against a raw-mmap
   byte-copy baseline over the same records, which isolates how much of the
   cost is Python-level per-neuron work (slicing, dequantization, tensor
   construction) versus the I/O itself. Run on a warm page cache by design:
   this measures the code-path floor, not NVMe latency.

2. dequantize_int4 in isolation.

3. Prefetch-predictor forward latency at real geometry (4096 -> 64 -> 14336),
   the '<0.1 ms on GPU' figure claimed in predictor.py's own docstring --
   checked on GPU when available, with a CPU figure alongside.

Usage:  python bench/bench_cold_read.py
"""

import os
import statistics
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import NeuronPredictor  # noqa: E402
from splitter import (  # noqa: E402
    HIDDEN_DIM,
    NEURON_RECORD_BYTES,
    ColdWeightStore,
    dequantize_int4,
    quantize_int4,
)
from tests.common import build_synthetic_cold_file  # noqa: E402

NUM_LAYERS = 16
N_COLD = 256
BATCH = 64          # neurons per read_neurons call (a plausible prefetch batch)
READ_CALLS = 200    # total timed calls, spread across layers


def env_banner():
    import platform
    print("environment:")
    print(f"  python {sys.version.split()[0]}, torch {torch.__version__}")
    print(f"  cpu: {platform.processor()}")
    if torch.cuda.is_available():
        print(f"  gpu: {torch.cuda.get_device_name(0)}")
    print()


def bench_cold_store(tmp):
    print(f"building synthetic cold file: {NUM_LAYERS} layers x {N_COLD} "
          f"cold neurons ({NUM_LAYERS * N_COLD * NEURON_RECORD_BYTES / 1e6:.1f} MB)...")
    t0 = time.perf_counter()
    cold_bin, cold_idx, config, truth = build_synthetic_cold_file(
        tmp, num_layers=NUM_LAYERS, n_cold=N_COLD, seed=0
    )
    print(f"  built in {time.perf_counter() - t0:.1f}s\n")

    store = ColdWeightStore(cold_bin, cold_idx, config)
    per_layer_ids = {
        li: [nid for (l, nid) in truth if l == li] for li in range(NUM_LAYERS)
    }

    # Warm the page cache and the store's offset maps once, so the timed
    # loop measures the steady-state code path.
    for li in range(NUM_LAYERS):
        store.read_neurons(li, 0, per_layer_ids[li][:BATCH])

    g = torch.Generator().manual_seed(1)
    timings = []
    total_neurons = 0
    for call in range(READ_CALLS):
        li = call % NUM_LAYERS
        ids = per_layer_ids[li]
        picks = torch.randperm(len(ids), generator=g)[:BATCH].tolist()
        batch_ids = [ids[p] for p in picks]
        t0 = time.perf_counter()
        out = store.read_neurons(li, 0, batch_ids)
        timings.append(time.perf_counter() - t0)
        total_neurons += out[0].shape[0]

    total_s = sum(timings)
    neurons_per_s = total_neurons / total_s
    effective_mb_s = neurons_per_s * NEURON_RECORD_BYTES / 1e6
    print("ColdWeightStore.read_neurons (warm cache, batch "
          f"{BATCH} neurons/call, {READ_CALLS} calls):")
    print(f"  median call latency: {statistics.median(timings) * 1e3:.2f} ms")
    print(f"  throughput:          {neurons_per_s:,.0f} neurons/s")
    print(f"  effective bandwidth: {effective_mb_s:,.1f} MB/s of cold records")

    # Baseline: raw byte copies of the same records via the same mmap --
    # no struct parsing, no dequant, no tensor construction.
    mm = store._mm
    offsets = []
    for li in range(NUM_LAYERS):
        sec = store._section_offsets(li, 0)
        offsets.extend(sec.values())
    t0 = time.perf_counter()
    picks = torch.randperm(len(offsets), generator=g)[: READ_CALLS * BATCH]
    for p in picks.tolist():
        _ = bytes(mm[offsets[p]: offsets[p] + NEURON_RECORD_BYTES])
    raw_s = time.perf_counter() - t0
    raw_neurons_per_s = len(picks) / raw_s
    print(f"  raw-mmap baseline:   {raw_neurons_per_s:,.0f} records/s "
          f"({raw_neurons_per_s * NEURON_RECORD_BYTES / 1e6:,.1f} MB/s)")
    print(f"  => Python-level decode overhead: "
          f"{raw_neurons_per_s / neurons_per_s:,.1f}x over raw bytes\n")

    store.close()
    return neurons_per_s, effective_mb_s, raw_neurons_per_s


def bench_dequant():
    x = torch.randn(HIDDEN_DIM)
    packed, scales = quantize_int4(x)
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        dequantize_int4(packed, scales, HIDDEN_DIM)
    dt = time.perf_counter() - t0
    print(f"dequantize_int4 (len {HIDDEN_DIM}): "
          f"{n / dt:,.0f} vectors/s ({dt / n * 1e6:.0f} us/vector)")
    print(f"  (three vectors per neuron record -> "
          f"{n / dt / 3:,.0f} neurons/s dequant ceiling)\n")


def bench_predictor():
    print("prefetch predictor forward (4096 -> rank 64 -> 14336, batch 1):")
    for device in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        model = NeuronPredictor(4096, 14336, rank=64).to(device).eval()
        x = torch.randn(1, 4096, device=device)
        with torch.no_grad():
            for _ in range(50):  # warmup
                model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times = []
            for _ in range(1000):
                t0 = time.perf_counter()
                model(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        med = statistics.median(times) * 1e3
        claim = "  [docstring claims <0.1 ms on GPU]" if device == "cuda" else ""
        print(f"  {device}: median {med:.3f} ms{claim}")
    print()


if __name__ == "__main__":
    env_banner()
    with tempfile.TemporaryDirectory() as tmp:
        bench_cold_store(tmp)
    bench_dequant()
    bench_predictor()
