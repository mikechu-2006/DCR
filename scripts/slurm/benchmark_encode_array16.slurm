#!/bin/bash
#SBATCH -p i64m512ue
#SBATCH -J dcr-bench-enc16
#SBATCH -o slurm_bench_enc16_%A_%a.out
#SBATCH -e slurm_bench_enc16_%A_%a.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-31
#
# 32 CPU tasks = 16 legacy shards + 16 canon shards, 8 CPUs each.
# Re-encodes from scratch into a fresh OUT_DIR; the earlier 4-shard run is kept
# under results/benchmark/encode for audit.
set -euo pipefail
ID="${SLURM_ARRAY_TASK_ID:?}"
if [ "$ID" -lt 16 ]; then
  export CROPS=legacy
  export ONLY_SHARD="$ID"
else
  export CROPS=canon
  export ONLY_SHARD="$((ID - 16))"
fi
export NUM_SHARDS=16
export OUT_DIR="results/benchmark/encode2"
export ASSEMBLE=0
export DEVICE=cpu
export THREADS=8
exec bash scripts/slurm/benchmark_encode.slurm
