#!/bin/bash
#SBATCH -p i64m1tga800ue
#SBATCH -J dcr-bench-encode
#SBATCH -o slurm_bench_encode_%j.out
#SBATCH -e slurm_bench_encode_%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#
# Encode the VSRG benchmark charts for the legacy and canon arms.
# Submit from the repo root:
#     sbatch scripts/slurm/benchmark_encode.slurm
# Override:
#     CROPS=legacy sbatch ...       # one arm only
#     NUM_SHARDS=4 sbatch ...       # array-like sharding inside one job
set -uo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO" || exit 1
export PATH=/opt/slurm/bin:$PATH
export CUDA_HOME="${CUDA_HOME:-/hpc2ssd/softwares/cuda/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/hpc2ssd/softwares/anaconda3/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-$REPO/data/raw/hf_cache}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$REPO/.cache/hf_modules}"
mkdir -p "$HF_HOME" "$HF_MODULES_CACHE" "$REPO/results/benchmark/encode"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CM3P_ROOT="${CM3P_ROOT:-$REPO/external/cm3p}"
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-8}"
export PYTHONUNBUFFERED=1

PY="${PY:-$HOME/venvs/cm3p/bin/python}"
CROPS="${CROPS:-legacy,canon}"
MODEL_DIR="${MODEL_DIR:-$REPO/data/raw/cm3p_model_26cc694}"
OSU_ROOT="${OSU_ROOT:-$REPO/.cache/benchmark/VSRG-DanEstimation-Benchmark/extracted}"
OUT_DIR="${OUT_DIR:-$REPO/results/benchmark/encode}"
MANIFEST="${MANIFEST:-$REPO/data/processed/benchmark_manifest.parquet}"
NUM_SHARDS="${NUM_SHARDS:-1}"
LIMIT="${LIMIT:-0}"
DEVICE="${DEVICE:-cuda}"
THREADS="${THREADS:-8}"

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) ==="
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
"$PY" - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY

if [ -n "${ONLY_SHARD:-}" ]; then
  SHARD_LIST="$ONLY_SHARD"
else
  SHARD_LIST="$(seq 0 $((NUM_SHARDS - 1)))"
fi

for CROP in ${CROPS//,/ }; do
  for SHARD in $SHARD_LIST; do
    echo "=== encode crop=$CROP shard=$SHARD/$NUM_SHARDS ==="
    "$PY" scripts/benchmark_encode.py \
      --manifest "$MANIFEST" --osu-root "$OSU_ROOT" \
      --crop "$CROP" --shard "$SHARD" --num-shards "$NUM_SHARDS" \
      --limit "$LIMIT" --model-dir "$MODEL_DIR" --out-dir "$OUT_DIR" \
      --device "$DEVICE" --threads "$THREADS" --dtype float32 --win-batch 8 || exit 1
  done
  if [ "${ASSEMBLE:-1}" = "1" ]; then
    "$PY" scripts/benchmark_encode.py --assemble --crop "$CROP" \
      --out-dir "$OUT_DIR" --manifest "$MANIFEST" \
      --out-parquet "data/processed/benchmark_content_${CROP}.parquet" || exit 1
  fi
done
echo "=== done ==="
date
