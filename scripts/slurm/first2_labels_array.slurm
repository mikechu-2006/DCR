#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-first2
#SBATCH -o slurm_first2_%A_%a.out
#SBATCH -e slurm_first2_%A_%a.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#
# Stage 0b: first-two-record best labels for 1k / 10k / full.
# randommerge was already produced by first2_labels.slurm (job 14078610).
set -euo pipefail
cd "$HOME/DCR"
PY="$HOME/.conda/envs/DCR/bin/python"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
DATASETS=(1k 10k full)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
case "$DS" in
  1k)
    STEP0=data/processed/step0_1k.csv
    PAIRS=data/processed/step1_1k_pairbest.parquet
    ;;
  10k)
    STEP0=data/processed/step0_10k.parquet
    PAIRS=data/processed/step1_10k_pairbest.parquet
    ;;
  full)
    STEP0=data/processed/step0_full.parquet
    PAIRS=data/processed/step1_full_pairbest_content.parquet
    ;;
  *) echo "bad dataset $DS"; exit 2;;
esac
OUT="data/processed/step1_${DS}_pairbest_first2.parquet"
JSON="data/processed/step1_${DS}_pairbest_first2.json"
echo "[$(date +%H:%M:%S)] host=$(hostname) dataset=$DS job=${SLURM_JOB_ID:-?}"
"$PY" scripts/step1_first2_labels.py --step0 "$STEP0" --pairs "$PAIRS" --out "$OUT" --json "$JSON"
echo "[$(date +%H:%M:%S)] DONE $DS"
