#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J d2v-res-p4
#SBATCH -o slurm_d2v_res_p4_%A_%a.out
#SBATCH -e slurm_d2v_res_p4_%A_%a.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-9%10
#
# Experiment 3, P4-holdout residual: 5 seeds x 2 baseline models.
# ID -> seed = ID//2+1, model = (linear_extra, mlp_extra)[ID%2].
set -euo pipefail
cd "$HOME/DCR"
PY="$HOME/.conda/envs/DCR/bin/python"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
SEED=$((SLURM_ARRAY_TASK_ID / 2 + 1))
MODELS=(linear_extra mlp_extra)
MODEL="${MODELS[$((SLURM_ARRAY_TASK_ID % 2))]}"
BASE="results/d2v_residual/baseline/p4/s${SEED}/${MODEL}"
OUT="results/d2v_residual/p4/s${SEED}/${MODEL}"
echo "[$(date +%H:%M:%S)] host=$(hostname) p4 s$SEED $MODEL"
"$PY" scripts/fit_d2v_residual.py \
  --protocol p4 --seed "$SEED" --model "$MODEL" \
  --split "data/processed/step1_1k_pairbest_p4_d2vhold_s${SEED}.parquet" \
  --baseline-oof "${BASE}_oof.parquet" \
  --baseline-full "${BASE}_full.parquet" \
  --baseline-col "base_${MODEL}" \
  --d2v-path data/processed/chart_content_d2v_holdout.parquet \
  --signal-table data/processed/rank_signals_1k.parquet \
  --lam-grid 0,0.001,0.01,0.1,1,10 --val-frac 0.2 --bootstrap 400 \
  --out-signals "${OUT}/signals.parquet" --out-json "${OUT}/result.json"
echo "[$(date +%H:%M:%S)] DONE p4 s$SEED $MODEL"
