#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-first2-pilot
#SBATCH -o slurm_first2_pilot_%A_%a.out
#SBATCH -e slurm_first2_pilot_%A_%a.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#
# Experiment 1 pilot: randommerge P4 s1, linear_extra + msd/sunny/log_n.
# A0 = baseline; A1 = first2 all pairs; A2 = first2 only n_plays>2;
# A3 = first2 only first2-ACC support (0.90, 0.998].
set -euo pipefail
cd "$HOME/DCR"
PY="$HOME/.conda/envs/DCR/bin/python"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
NAMES=(a0 a1 a2 a3)
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"
case "$NAME" in
  a0) SCOPE=all; GRID=0; VAL=0 ;;
  a1) SCOPE=all; GRID=0,0.05,0.1,0.2,0.5,1.0; VAL=0.2 ;;
  a2) SCOPE=n_plays_gt2; GRID=0,0.05,0.1,0.2,0.5,1.0; VAL=0.2 ;;
  a3) SCOPE=support; GRID=0,0.05,0.1,0.2,0.5,1.0; VAL=0.2 ;;
  *) echo "bad arm $NAME"; exit 2 ;;
esac
echo "[$(date +%H:%M:%S)] host=$(hostname) arm=$NAME scope=$SCOPE job=${SLURM_JOB_ID:-?}"
"$PY" scripts/step2_cm3p_rank_aux.py \
  --in data/processed/step1_randommerge_c_p4_s1.parquet \
  --first2 data/processed/step1_randommerge_first2.parquet \
  --content-path data/processed/chart_content_cm3p_local.parquet \
  --signal-table data/processed/rank_signals_randommerge_od.parquet \
  --extra-signals msd_overall,sunny_star,log_n \
  --aux-scope "$SCOPE" --aux-weight-grid "$GRID" --val-frac "$VAL" \
  --name "first2_${NAME}" \
  --out-signals "results/first2_aux/pilot/${NAME}.parquet" \
  --out-json "results/first2_aux/pilot/${NAME}.json"
echo "[$(date +%H:%M:%S)] DONE $NAME"
