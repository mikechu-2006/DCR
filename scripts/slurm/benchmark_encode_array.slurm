#!/bin/bash
#SBATCH -p i64m512ue
#SBATCH -J dcr-bench-enc-arr
#SBATCH -o slurm_bench_enc_%A_%a.out
#SBATCH -e slurm_bench_enc_%A_%a.err
#SBATCH -n 16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-7
#
# 8 CPU tasks: 0-3 legacy shards, 4-7 canon shards.  Each task encodes one
# shard; assemble once after the array finishes.
set -euo pipefail
ID="${SLURM_ARRAY_TASK_ID:?}"
if [ "$ID" -lt 4 ]; then
  export CROPS=legacy
  export ONLY_SHARD="$ID"
else
  export CROPS=canon
  export ONLY_SHARD="$((ID - 4))"
fi
export NUM_SHARDS=4
export ASSEMBLE=0
export DEVICE=cpu
export THREADS=16
exec bash scripts/slurm/benchmark_encode.slurm
