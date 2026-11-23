#!/bin/bash
#SBATCH -p i64m512ue
#SBATCH -J dcr-bench-encode-cpu
#SBATCH -o slurm_bench_encode_cpu_%j.out
#SBATCH -e slurm_bench_encode_cpu_%j.err
#SBATCH -n 16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#
# CPU fallback for the benchmark encoder (the GPU torch build needs a newer
# driver than the A800 nodes provide).  This wrapper only sets env defaults and
# delegates to benchmark_encode.slurm, whose #SBATCH lines are ignored when it
# is executed with bash.
set -euo pipefail
export DEVICE=cpu
export THREADS="${THREADS:-16}"
export CROPS="${CROPS:-legacy}"
export NUM_SHARDS="${NUM_SHARDS:-1}"
exec bash scripts/slurm/benchmark_encode.slurm
