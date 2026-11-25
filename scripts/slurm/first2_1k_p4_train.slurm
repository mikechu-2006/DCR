#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-first2-1k
#SBATCH -o slurm_first2_1k_%A_%a.out
#SBATCH -e slurm_first2_1k_%A_%a.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-19
#
# Experiment 1, 1k P4 5-seed confirmation: A0-A3 per seed.
set -euo pipefail
cd "$HOME/DCR"
PY="$HOME/.conda/envs/DCR/bin/python"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
SEED=$((SLURM_ARRAY_TASK_ID / 4 + 1))
ARM_IDX=$((SLURM_ARRAY_TASK_ID % 4))
ARMS=(a0 a1 a2 a3)
ARM="${ARMS[$ARM_IDX]}"
case "$ARM" in
  a0) SCOPE=all; GRID=0; VAL=0 ;;
  a1) SCOPE=all; GRID=0,0.05,0.1,0.2,0.5,1.0; VAL=0.2 ;;
  a2) SCOPE=n_plays_gt2; GRID=0,0.05,0.1,0.2,0.5,1.0; VAL=0.2 ;;
  a3) SCOPE=support; GRID=0,0.05,0.1,0.2,0.5,1.0; VAL=0.2 ;;
  *) echo "bad arm $ARM"; exit 2 ;;
esac
OUT="results/first2_aux/1k_p4/s${SEED}"
echo "[$(date +%H:%M:%S)] host=$(hostname) seed=$SEED arm=$ARM scope=$SCOPE"
"$PY" scripts/step2_cm3p_rank_aux.py \
  --in "data/processed/step1_1k_pairbest_p4_s${SEED}.parquet" \
  --first2 data/processed/step1_1k_pairbest_first2.parquet \
  --content-path data/processed/chart_content_cm3p_local.parquet \
  --signal-table data/processed/rank_signals_1k.parquet \
  --extra-signals msd_overall,sunny_star \
  --aux-scope "$SCOPE" --aux-weight-grid "$GRID" --val-frac "$VAL" \
  --name "first2_${ARM}" \
  --out-signals "$OUT/${ARM}.parquet" \
  --out-json "$OUT/${ARM}.json"
echo "[$(date +%H:%M:%S)] DONE s$SEED $ARM"
