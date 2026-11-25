#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-first2-a6
#SBATCH -o slurm_first2_a6_%A_%a.out
#SBATCH -e slurm_first2_a6_%A_%a.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#
# A6 control: first2 labels shuffled across pairs (training-side second label
# is noise).  If A6 ~ A1, the small A1 gain is not first2-specific.
set -euo pipefail
cd "$HOME/DCR"
PY="$HOME/.conda/envs/DCR/bin/python"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
SEED=$((SLURM_ARRAY_TASK_ID + 1))
OUT="results/first2_aux/1k_p4/s${SEED}"
echo "[$(date +%H:%M:%S)] host=$(hostname) seed=$SEED A6"
"$PY" scripts/step2_cm3p_rank_aux.py \
  --in "data/processed/step1_1k_pairbest_p4_s${SEED}.parquet" \
  --first2 data/processed/step1_1k_pairbest_first2_shuffled.parquet \
  --aux-label-col loss_first2_shuffled --aux-scope all \
  --content-path data/processed/chart_content_cm3p_local.parquet \
  --signal-table data/processed/rank_signals_1k.parquet \
  --extra-signals msd_overall,sunny_star \
  --aux-weight-grid 0,0.05,0.1,0.2,0.5,1.0 --val-frac 0.2 \
  --name first2_a6 \
  --out-signals "$OUT/a6.parquet" --out-json "$OUT/a6.json"
echo "[$(date +%H:%M:%S)] DONE s$SEED A6"
