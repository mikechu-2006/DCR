#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J d2v-base-p2
#SBATCH -o slurm_d2v_base_p2_%A_%a.out
#SBATCH -e slurm_d2v_base_p2_%A_%a.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#
# Experiment 3 baseline OOF: P2-holdout, 5 chart-disjoint folds.
# 0 = linear_extra, 1 = mlp_extra.
set -euo pipefail
cd "$HOME/DCR"
PY="$HOME/.conda/envs/DCR/bin/python"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
MODELS=(linear_extra mlp_extra)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
OUT="results/d2v_residual/baseline/p2/${MODEL}_oof"
echo "[$(date +%H:%M:%S)] host=$(hostname) p2 $MODEL"
"$PY" scripts/step2_cm3p_rank.py \
  --in data/processed/step1_1k_pairbest_d2v_holdout.parquet \
  --content-path data/processed/chart_content_cm3p_local.parquet \
  --signal-table data/processed/rank_signals_1k.parquet \
  --extra-signals msd_overall,sunny_star \
  --model "$MODEL" --chart-only --chart-folds 5 --chart-seed 0 \
  --val-frac 0 --epochs 98 --seed 0 --threads 8 \
  --name "base_${MODEL}" \
  --out-signals "$OUT.parquet" --out-json "$OUT.json"
echo "[$(date +%H:%M:%S)] DONE p2 $MODEL"
