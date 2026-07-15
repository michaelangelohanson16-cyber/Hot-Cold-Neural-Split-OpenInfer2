#!/usr/bin/env bash
#
# One-shot end-to-end run on a fresh GPU box (>=24 GB VRAM).
# Profiles a real sparse model, builds the hot/cold split, trains the
# prefetch predictors, and generates text through the engine -- the full
# pipeline on real weights, which the 15 GB dev laptop cannot hold.
#
# Usage, on a fresh RunPod / Lambda / Vast Ubuntu+CUDA box:
#     bash run_end_to_end.sh
#
# Tunables (env vars):
#     MODEL         HF id of the base model (default: TurboSparse-Mistral,
#                   the dReLU-sparse model the engine is designed for --
#                   the one that actually exercises the sparsity premise)
#     PROF_TOKENS   profiling token budget         (default 200000)
#     PRED_TOKENS   predictor-data token budget    (default 100000)
#     HOT_COUNT     hot neurons per layer          (default 2048)
#     GEN_TOKENS    tokens to generate at the end  (default 40)
#     PROMPT        generation prompt
set -euo pipefail

MODEL="${MODEL:-PowerInfer/TurboSparse-Mistral-Instruct}"
PROF_TOKENS="${PROF_TOKENS:-200000}"
PRED_TOKENS="${PRED_TOKENS:-100000}"
HOT_COUNT="${HOT_COUNT:-2048}"
GEN_TOKENS="${GEN_TOKENS:-40}"
PROMPT="${PROMPT:-The relationship between memory and identity is}"
WORK="${WORK:-/workspace/openinfer}"

banner() { printf '\n\033[1;36m==== %s ====\033[0m\n' "$1"; }

banner "0. Environment"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

banner "1. Fetch code + deps"
mkdir -p "$WORK" && cd "$WORK"
if [ ! -d repo ]; then
    git clone --depth 1 https://github.com/michaelangelohanson16-cyber/Hot-Cold-Neural-Split-OpenInfer2.git repo
fi
cd repo
# Many GPU-cloud base images ship a PEP-668 "externally managed" system
# Python; --break-system-packages is the pragmatic choice on a throwaway pod.
export PIP_BREAK_SYSTEM_PACKAGES=1
pip install -q --upgrade "transformers>=4.44" numpy safetensors accelerate sentencepiece

banner "2. Build a small profiling corpus (public-domain Gutenberg plain text)"
mkdir -p corpus
# A few books give a few hundred K tokens of varied English -- enough to
# get non-degenerate per-neuron activation frequencies.
for id in 1342 84 2701 1661; do
    [ -f "corpus/$id.txt" ] || curl -sL "https://www.gutenberg.org/files/$id/$id-0.txt" -o "corpus/$id.txt" || true
done
wc -l corpus/*.txt | tail -1

banner "3. Profile activations  (MODEL=$MODEL)"
time python profiler.py --model "$MODEL" --corpus corpus \
    --output out/profile.json --tokens "$PROF_TOKENS" --chunk 512 \
    --device cuda --trust-remote-code

banner "4. Hot/cold split -> FP8 hot + INT4 cold NVMe file"
time python splitter.py split-dense --base "$MODEL" \
    --profile out/profile.json --output out/split \
    --hot-count "$HOT_COUNT" --trust-remote-code

banner "5. Collect predictor data + train prefetch predictors"
time python predictor.py collect --model "$MODEL" --corpus corpus \
    --output out/pred_data --tokens "$PRED_TOKENS" \
    --device cuda --trust-remote-code
time python predictor.py train --data out/pred_data \
    --output out/predictors --epochs 3 --device cuda

banner "6. Generate through the engine (real model, real weights)"
time python engine.py generate --split-dir out/split \
    --predictor out/predictors --tokenizer "$MODEL" \
    --prompt "$PROMPT" --max-tokens "$GEN_TOKENS" --device cuda

banner "7. Benchmark tokens/sec"
python engine.py benchmark --split-dir out/split \
    --predictor out/predictors --n-tokens 50 --device cuda || true

banner "Done"
echo "Split artifacts and logs are under $WORK/repo/out"
echo "Cold file size:"; du -h out/split/cold_weights.bin 2>/dev/null || true
