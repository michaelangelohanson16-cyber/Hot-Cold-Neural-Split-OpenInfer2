# Hot-Cold-Neural-Split-OpenInfer2

A from-scratch PC/NVMe hot-cold sparse-FFN inference engine: profile which
FFN neurons a model actually activates, keep the frequently-firing ones
resident in VRAM (FP8), and stream the rest from an NVMe-backed cold store
(INT4) on prediction — with cross-layer overlap so the NVMe fetch for layer
*L+1* runs while layer *L*'s GPU compute is still in flight. The goal:
serve a dense 7-8B model inside an 8 GB VRAM budget by exploiting how
sparse a model's real activation pattern is per token.

## What this is, precisely

This adapts one specific idea from **PowerInfer-2** (Xue, Song, Mi, Zheng,
Xia, Chen — SJTU IPADS, ["Fast Large Language Model Inference on a
Smartphone"](https://arxiv.org/abs/2406.06282), 2024) — that cold,
rarely-activated neurons don't need to live in fast memory at all; they can
be tiered to persistent storage and streamed in only when a predictor says
they're about to fire. **PowerInfer-2 itself targets smartphones**: NPU +
CPU compute, UFS flash storage, dynamically-sized "neuron clusters" that
adapt to batch size for parallel decoding strategies like Best-of-N. None
of that hardware-specific machinery is what this repo is attempting.
What's adapted here is the underlying idea — persistent-storage neuron
tiering driven by a learned predictor — reimplemented for **PC hardware**:
GPU + NVMe, per-neuron (not cluster) granularity, pure PyTorch.

**PowerInfer-2's own source code is not publicly available**, only the
paper. The original **PowerInfer** (Song et al., SJTU IPADS,
[MIT-licensed, open source](https://github.com/SJTU-IPADS/PowerInfer),
8,000+ stars) is the actual public ancestor this idea traces back to —
GPU-hot / CPU-RAM-cold neuron splitting on PC hardware, which this repo's
NVMe-cold variant is closer to architecturally than PI2's mobile design,
even though the storage-tiering-plus-predictor framing that motivated this
specific implementation comes from the PI2 paper.

## How it works

1. **`profiler.py`** — registers forward hooks on every FFN layer, runs a
   corpus through the model, and records per-neuron activation frequency.
   Handles both standard gated FFNs (SwiGLU) and the dReLU/"Bamboo" family
   (TurboSparse checkpoints) correctly — the latter's shared activation
   function fires twice per forward with a built-in post-hoc mask, so
   hooking it directly would double-count; the profiler hooks
   `down_proj`'s input instead, which is the correct place to read the
   real per-neuron contribution.
2. **`splitter.py`** — given the profile, partitions every FFN neuron into
   hot (top-N by activation frequency, kept in VRAM as FP8) and cold
   (written to a page-aligned NVMe binary as INT4 with per-group absmax
   scales). Includes the memory-mapped `ColdWeightStore` reader the engine
   uses at inference time.
3. **`predictor.py`** — trains a small low-rank (hidden→64→intermediate)
   model per layer to predict, from the hidden state just before a layer,
   which cold neurons are about to fire — so only the predicted-active
   subset needs to be fetched, not the whole cold set. Tracks precision,
   recall, and the two costs that actually matter for a prefetcher: waste
   rate (fetched but didn't fire) and miss rate (fired but wasn't
   fetched).
4. **`engine.py`** — the runtime: per-token, submits layer *L+1*'s cold
   fetch using layer *L*'s post-attention hidden state (a one-sublayer-
   early lookahead — an accepted approximation, documented as such),
   overlapping NVMe I/O with GPU compute via a thread pool. Hot and cold
   FFN halves are computed separately and summed — because `down_proj` is
   linear in the neuron dimension, this is exact, not approximate; the
   only precision loss anywhere in the pipeline is the INT4 quantization
   itself.

Geometry (hidden dim, layer count, heads, activation function, etc.) is
read from the base model's own HF config at split time, not hardcoded —
`--base` accepts any 4096-hidden Llama/Mistral-family model.


## Running it

```bash
pip install torch numpy transformers

python profiler.py --model <hf-id-or-path> --corpus <dir-of-txt-files> \
    --output profile.json --tokens 10000000

python splitter.py split-dense --base <hf-id-or-path> --profile profile.json \
    --output split/

python predictor.py collect --model <hf-id-or-path> --corpus <dir> \
    --output predictor_data/
python predictor.py train --data predictor_data/ --output predictors/

python engine.py generate --split-dir split/ --predictor predictors/ \
    --tokenizer <hf-id-or-path> --prompt "..."
```

A 4096-hidden model (Mistral-7B/Llama-3.1-8B geometry) is required — the
cold-record binary layout is compile-time sized for it.


## Tests and measured benchmarks

The component tier is tested and benchmarked without any model download —
14 tests covering INT4 round-trip bounds, byte-level cold-store format
fidelity, exactness of the hot+cold FFN decomposition under both
activation conventions, and predictor learnability, plus measured
throughput of the cold read path and the prefetch predictor:

```bash
python -m pytest tests/ -v
python bench/bench_cold_read.py
```

Full results, with interpretation, in [BENCHMARKS.md](BENCHMARKS.md), and
the unedited benchmark output in `bench_results.log`. The headline
findings are reported there whether they flatter the project or not: the
decomposition math is verified exact, the on-disk format reads back
byte-faithfully, and the current per-neuron cold read path measures 412×
slower than raw access to the same bytes — CPU-bound on dequantization,
which is the concrete work item standing between this design and a
meaningful end-to-end benchmark.

The full pipeline (profile → split → predict → generate) **has been run
end to end on a real 7B** — PowerInfer's dReLU-sparse
TurboSparse-Mistral-Instruct, on a rented RTX 3090. Measured: profiling
at 3,215 tok/s, a 34-minute hot/cold split producing a 2.52 GB INT4 cold
file, a prefetch predictor hitting 84% precision / 80% recall on real
activations, and generation at ~68 s/token. The split time and the
generation latency both trace to the same unvectorized per-neuron loop,
and the 80%-recall predictor visibly degrades output quality — all
honest, measured, and detailed (with the three integration bugs the
first end-to-end run exposed and fixed) in [BENCHMARKS.md](BENCHMARKS.md).


## License

MIT — see `LICENSE`.
