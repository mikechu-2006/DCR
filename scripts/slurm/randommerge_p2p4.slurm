#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-rm-p2p4
#SBATCH -o slurm_rm_p2p4_%A_%a.out
#SBATCH -e slurm_rm_p2p4_%A_%a.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#
# One array task = one (protocol, split seed).
#   tasks 0-4 -> P2 seeds 1-5   (pair-random resplit, chart-disjoint folds)
#   tasks 5-9 -> P4 seeds 1-5   (player x chart double cold start)
#
# Content arm: legacy audio-off CM3P (data/processed/chart_content_cm3p_local.parquet),
# the only table that covers the merged 20,388 charts.  canon is worse in this repo's
# own P2/P4 test; audio-on covers only 32.7% of the merged charts.
#
# Splits are prebuilt on the laptop and synced to data/processed/:
#   step1_randommerge_c_p2_s{1..5}.parquet, step1_randommerge_c_p4_s{1..5}.parquet
set -euo pipefail
ID="${SLURM_ARRAY_TASK_ID:?}"
if [ "$ID" -lt 5 ]; then
  PROTO=p2; SEED=$((ID + 1))
else
  PROTO=p4; SEED=$((ID - 4))
fi
cd "$HOME/DCR"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1
MODELS="${MODELS:-linear_extra,linear}"
echo "[$(date +%H:%M:%S)] host=$(hostname) proto=$PROTO seed=$SEED models=$MODELS"
python3 scripts/run_p2_p4_multiseed.py \
  --protocols "$PROTO" --seeds "$SEED" \
  --models "$MODELS" \
  --threads 8 \
  --split-p2 data/processed/step1_randommerge_c_p2 \
  --split-p4 data/processed/step1_randommerge_c_p4 \
  --out-dir results/randommerge_p2p4_hpc
echo "[$(date +%H:%M:%S)] DONE $PROTO s$SEED"
