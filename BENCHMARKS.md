# Tests and Measured Benchmarks

Everything below was produced by the committed test suite (`tests/`) and
benchmark script (`bench/bench_cold_read.py`); the unedited benchmark
output is in `bench_results.log`. None of it requires a model download —
this is the component tier: correctness of the math and format, and
measured throughput of the code paths. End-to-end tokens-per-second on a
real model remains unmeasured and is stated as such.

**Environment:** Python 3.11.9, PyTorch 2.12.0-dev cu128, Intel (Family 6
Model 197) CPU, NVIDIA RTX 5070 Laptop GPU, Windows 11.

---

## Test suite: 14 tests, all passing

`python -m pytest tests/ -v`

**Quantization (`test_quantization.py`, 5 tests).** INT4 absmax group
round-trip error stays within the analytic bound (step/2 per element,
step = group absmax / 7) on every group; zero vectors survive exactly;
packing is two values per byte; short vectors and extreme values are
handled.

**Cold-store format (`test_cold_store.py`, 4 tests).** A synthetic
`cold_weights.bin` written in `split_dense`'s exact record layout reads
back byte-faithfully through `ColdWeightStore`: correct neurons, exact
values, correct shapes, absent ids skipped, empty sections and the
offset cache handled. One subtlety surfaced while building the fixture:
on-disk fidelity is INT4 weights **plus FP16-rounded group scales** —
the ground truth must apply the same FP16 rounding or values mismatch.
The store is faithful to what the format actually stores.

**FFN decomposition (`test_ffn_exactness.py`, 2 tests, 6 seeds).** The
engine's central mathematical claim, verified directly: computing the
FFN as (hot half) + (cold half) is exact — identical (to fp32
summation-order noise, tolerance 1e-5) to computing the full FFN with
the same per-neuron precisions in one pass, under both activation
conventions (SwiGLU `silu(g)*u` and dReLU `relu(g)*relu(u)`). The
measured cost of the mixed-precision scheme itself, against full fp32
on random Gaussian weights: **9.5–17.5% relative L2 output error**
across seeds. That figure is the analytically expected one (INT4 absmax
over Gaussian 128-groups gives ~12% RMS per-weight error), reported
here as data rather than hidden: it is the true precision cost of the
format on unstructured weights, and the reason the scheme's viability
on a real model is an empirical question about real weight structure,
not something these tests settle.

**Predictor (`test_predictor.py`, 3 tests).** Trained through
`PredictorTrainer`'s own code path on synthetic low-rank-separable
activation data: **F1 = 0.941** (precision 0.941, recall 0.941, base
rate 0.50) after 25 epochs. Save/load round-trip preserves outputs
bit-for-bit at fp32 tolerance; mask thresholding matches probabilities.
This validates the training loop and serialization, not accuracy on any
real model.

---

## Benchmarks: the bottleneck, measured

`python bench/bench_cold_read.py` (full output: `bench_results.log`)

Synthetic cold file, 16 layers × 256 cold neurons (26.2 MB), warm page
cache — so these numbers measure the code path, not NVMe latency.

| Path | Throughput | Effective bandwidth |
|---|---|---|
| `ColdWeightStore.read_neurons` (batch 64/call) | **917 neurons/s** (median 66 ms/call) | **5.9 MB/s** |
| Raw mmap byte-copy of the same records | 377,407 records/s | 2,415 MB/s |
| `dequantize_int4` in isolation (len 4096) | 3,508 vectors/s (285 µs/vector) | — |

**Interpretation, plainly.** The cold read path is **412× slower than
raw byte access to the same data**, and the arithmetic pins the cause:
three dequantization calls per neuron at 285 µs each (~855 µs) accounts
for nearly all of the ~1.09 ms per-neuron cost. The path is CPU-bound
on per-vector Python/torch dequantization — the storage is idle. At 917
neurons/s, any realistic per-token cold fetch (hundreds of neurons per
layer) implies seconds per token: **the engine cannot be fast in its
current form, and the fix is not I/O tuning but batching the decode** —
unpacking and scaling a whole batch of records as single vectorized
tensor operations instead of one neuron at a time. The raw-mmap
baseline (2.4 GB/s warm) is the ceiling that says how much headroom
that fix has. This confirms, with numbers, the bottleneck the README
previously described in prose.

**Prefetch predictor forward** (4096 → rank 64 → 14336, batch 1, median
of 1000):

| Device | Latency |
|---|---|
| RTX 5070 Laptop GPU | 0.417 ms |
| CPU | 0.302 ms |

The predictor source previously claimed "<0.1 ms on GPU." **Measured,
that claim does not hold on this hardware at batch 1** — kernel-launch
overhead dominates, and the CPU is actually faster at this size. The
docstring has been corrected to the measured figures. The practical
consequence is mild (32 layers × ~0.4 ms ≈ 13 ms of predictor cost per
token if run serially on GPU, less on CPU or batched), but the claim is
now a measurement instead of an assertion.

---

## What this tier does not cover

No profile, split, predictor, or generation run has been executed
against a real model — those require the ~15 GB TurboSparse-Mistral
download and hours of profiling, and every claim about end-to-end
tokens-per-second remains unmade. What this tier establishes: the
arithmetic the engine depends on is correct (decomposition exactness,
quantization bounds, format fidelity, predictor learnability), and the
performance work needed before an end-to-end run is worth benchmarking
is now quantified rather than suspected.
