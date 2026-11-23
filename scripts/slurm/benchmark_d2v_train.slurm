#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-bench-d2v-train
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o slurm_bench_d2v_train_%j.out
#SBATCH -e slurm_bench_d2v_train_%j.err
#
# D2V counterpart of RM-L: linear_extra on the randommerge set,
# content = 512-d D2V holdout vectors, extra = msd_overall,sunny_star.
set -euo pipefail
cd "$HOME/DCR"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1
mkdir -p results/benchmark/models results/benchmark/d2v_linear/train
echo "[$(date +%H:%M:%S)] host=$(hostname) d2v train start"
python3 -u scripts/step2_cm3p_rank.py \
  --in data/processed/step1_randommerge_pairbest_content.parquet \
  --content-path data/processed/chart_content_d2v_holdout.parquet \
  --signal-table data/processed/rank_signals_1k.parquet \
  --extra-signals msd_overall,sunny_star \
  --model linear_extra --lam 1e-3 \
  --val-frac 0 --seed 0 --threads 8 \
  --name cm3p_linear_extra \
  --fit-all \
  --save-model results/benchmark/models/D2V_L_full.pt \
  --out-signals results/benchmark/d2v_linear/train/D2V_L_signals.parquet \
  --out-json results/benchmark/d2v_linear/train/D2V_L.json
echo "[$(date +%H:%M:%S)] D2V_BENCHMARK_TRAIN_DONE"
