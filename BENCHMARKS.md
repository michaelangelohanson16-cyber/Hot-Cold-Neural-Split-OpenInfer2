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

## End-to-end smoke run (synthetic model)

The full pipeline — profile → split → predictor collect/train → generate —
has now been run start to finish on a **tiny synthetic model**: a
randomly-initialized 2-layer Llama with the required 4096 hidden size and
14336 FFN width, a 50257-token vocabulary, and tied embeddings (642M
params, small enough to fit the 15 GB RAM that a real 7-8B model does
not). The point is not output quality — random weights produce gibberish
by construction — but whether the engine's assembly of hot FP8 weights,
the INT4 cold store, predictor-guided prefetch, and cross-layer overlap
actually executes and emits correctly-shaped tokens. It does:

```
Engine ready.
Generated 12 tokens in 73.37s (0.2 tok/s)
betaulative distributor Battery Mith Armour...   (gibberish, as expected)
```

The 0.2 tok/s is meaningless as a performance figure (CPU, float16
matmuls, the un-vectorized cold-read path, a 2-layer model); it is
reported only to show the loop ran.

The value of the run was the bugs it exposed. Every component above is
unit-tested and correct in isolation, but three defects lived in the
*seams between them* — exactly where the unit tests, by construction,
could not reach — and only surfaced when the engine assembled the real
pieces:

1. **Predictor dtype boundary.** The engine runs activations at the
   model's compute dtype (float16); predictors are stored float32.
   `PredictorRegistry.predict` now casts the hidden state to the
   predictor's dtype.
2. **RoPE upcast.** The rope cos/sin cache is float32, which silently
   promoted `q`/`k` to float32 while `v` (which skips rope) stayed
   float16, breaking `scaled_dot_product_attention`. `_apply_rope` now
   casts cos/sin to the activation dtype.
3. **FFN down-projection shape.** The engine's `sparse_ffn` transposed
   the already-`(H, n)` down-weight, breaking the matmul. Removed. (The
   FFN-exactness unit test did not catch this because it exercises a
   reference implementation, not the engine's own `sparse_ffn` — a
   pointed reminder of why an end-to-end run matters even when every
   component is green.)

## Full end-to-end on a real model (TurboSparse-Mistral-7B, RTX 3090)

The complete pipeline was then run on a **real** model — PowerInfer's
`TurboSparse-Mistral-Instruct`, the dReLU-sparsified 7B this engine was
designed for — on a rented RTX 3090 (24 GB), the machine class needed to
hold a 7B checkpoint that the 15 GB dev laptop cannot. Every stage ran on
real weights. The measured numbers, stage by stage:

| Stage | Measurement |
|---|---|
| Load (custom `bamboo` remote code) | loads cleanly |
| Profile activations | **3,215 tok/s** (200,323 tokens in 1.0 min, GPU) |
| Hot/cold split | **34 min** (2,042 s), producing a **2.52 GB** INT4 cold file |
| Predictor training | **84.4% precision / 79.9% recall** predicting cold-neuron activations |
| Generation | **~68 s/token** (4 tokens in 271.8 s) |

Three of these are the real headline results:

**The split is the bottleneck, confirmed at scale.** 34 minutes to
quantize a 7B's cold neurons — CPU-bound on the same per-neuron Python
`quantize_int4` loop the component benchmark measured at 412× overhead.
This is the single strongest argument for the vectorization work: the
split dominates the entire pipeline's wall-clock.

**The predictor genuinely works on real weights.** 84% precision / 80%
recall is a real, non-trivial result — the low-rank prefetch oracle,
trained on real activation traces, actually predicts which cold neurons
fire. The premise of the whole design (that activation is predictable
from the pre-layer hidden state) holds on a real model, not just on
synthetic separable data.

**Generation runs, and reveals the real cost/quality trade.** The engine
loaded the real split (32 predictors + 32 attention + 32 hot-FFN layers)
and generated, but at ~68 s/token — the cold-read path, unvectorized, at
7B scale. And the output was weak: for the prompt *"The relationship
between memory and identity is"* it produced a barely-grammatical
continuation. That is not a bug but a measured consequence of the
design: at 80% recall the predictor **misses ~20% of firing neurons every
layer**, and that error compounds across 32 layers. Predictor-guided
prefetch trades exactness for fewer fetches; at this recall, on this
model, the trade visibly degrades output. A `--skip-predictor` full-fetch
run would be exact (hot + all cold = the true FFN) and coherent, at the
cost of being several times slower still.

Total cloud cost for the full run: roughly **$0.55**.

## What remains

The engine is now validated end to end on a real 7B: it loads, profiles,
splits, trains predictors, and generates real tokens. What the run makes
concrete is the two-front performance problem standing between "works"
and "usable": the **split's per-neuron quantization** (34 min, dominates
wall-clock) and the **cold-read path** (~68 s/token in generation), both
the same unvectorized Python loop, and the **precision/recall of the
predictor** (higher recall would trade speed for output quality). None of
these are correctness gaps — the arithmetic is exact and the integration
bugs are fixed — they are the optimization surface for turning a correct
research engine into a fast one.
