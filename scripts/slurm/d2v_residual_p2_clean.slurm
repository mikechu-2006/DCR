#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J d2v-res-p2c
#SBATCH -o slurm_d2v_res_p2c_%A_%a.out
#SBATCH -e slurm_d2v_res_p2c_%A_%a.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-9%10
#
# Experiment 3, clean P2: chart-disjoint fold tables. The residual for fold k
# is fitted on train rows of other charts and evaluated on test rows of fold k.
set -euo pipefail
cd "$HOME/DCR"
PY="$HOME/.conda/envs/DCR/bin/python"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
FOLD=$((SLURM_ARRAY_TASK_ID / 2))
MODELS=(linear_extra mlp_extra)
MODEL="${MODELS[$((SLURM_ARRAY_TASK_ID % 2))]}"
OUT="results/d2v_residual/p2_clean/fold${FOLD}/${MODEL}"
echo "[$(date +%H:%M:%S)] host=$(hostname) p2 fold=$FOLD $MODEL"
"$PY" scripts/fit_d2v_residual.py \
  --protocol p2 --seed 0 --model "$MODEL" \
  --split "data/processed/d2v_p2_folds/p2_fold${FOLD}.parquet" \
  --baseline-oof "results/d2v_residual/baseline/p2/${MODEL}_oof.parquet" \
  --baseline-col "base_${MODEL}" \
  --d2v-path data/processed/chart_content_d2v_holdout.parquet \
  --signal-table data/processed/rank_signals_1k.parquet \
  --lam-grid 0,0.001,0.01,0.1,1,10 --val-frac 0.2 --bootstrap 400 \
  --out-signals "${OUT}/signals.parquet" --out-json "${OUT}/result.json"
echo "[$(date +%H:%M:%S)] DONE p2 fold=$FOLD $MODEL"
