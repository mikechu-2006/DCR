#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J d2v-base-p4
#SBATCH -o slurm_d2v_base_p4_%A_%a.out
#SBATCH -e slurm_d2v_base_p4_%A_%a.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-9%10
#
# Experiment 3 baseline OOF: P4-holdout, 5 chart-disjoint folds + full 6x6.
# ID -> seed = ID//2+1, model = (linear_extra, mlp_extra)[ID%2].
set -euo pipefail
cd "$HOME/DCR"
PY="$HOME/.conda/envs/DCR/bin/python"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
SEED=$((SLURM_ARRAY_TASK_ID / 2 + 1))
MODELS=(linear_extra mlp_extra)
MODEL="${MODELS[$((SLURM_ARRAY_TASK_ID % 2))]}"
SPLIT="data/processed/step1_1k_pairbest_p4_d2vhold_s${SEED}.parquet"
OUT="results/d2v_residual/baseline/p4/s${SEED}/${MODEL}"
echo "[$(date +%H:%M:%S)] host=$(hostname) p4 s$SEED $MODEL"
# 1) OOF scores for the 6x6 training charts (5 chart-disjoint folds)
"$PY" scripts/step2_cm3p_rank.py \
  --in "$SPLIT" \
  --content-path data/processed/chart_content_cm3p_local.parquet \
  --signal-table data/processed/rank_signals_1k.parquet \
  --extra-signals msd_overall,sunny_star \
  --model "$MODEL" --chart-only --chart-folds 5 --chart-seed 0 \
  --val-frac 0 --epochs 98 --seed 0 --threads 8 \
  --name "base_${MODEL}" \
  --out-signals "${OUT}_oof.parquet" --out-json "${OUT}_oof.json"
# 2) full 6x6 fit, used for the 4x4 test charts
"$PY" scripts/step2_cm3p_rank.py \
  --in "$SPLIT" \
  --content-path data/processed/chart_content_cm3p_local.parquet \
  --signal-table data/processed/rank_signals_1k.parquet \
  --extra-signals msd_overall,sunny_star \
  --model "$MODEL" --val-frac 0 --epochs 98 --seed 0 --threads 8 \
  --name "base_${MODEL}" \
  --out-signals "${OUT}_full.parquet" --out-json "${OUT}_full.json"
echo "[$(date +%H:%M:%S)] DONE p4 s$SEED $MODEL"
