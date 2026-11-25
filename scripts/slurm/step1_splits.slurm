#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-splits
#SBATCH -o slurm_splits_%A_%a.out
#SBATCH -e slurm_splits_%A_%a.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#
# Stage 0b: P2/P4 5-seed splits for 1k / 10k / full.
# randommerge already has its splits.  One dataset x protocol per task.
set -euo pipefail
cd "$HOME/DCR"
PY="$HOME/.conda/envs/DCR/bin/python"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
DATASETS=(1k 1k 10k 10k full full)
PROTOS=(p2 p4 p2 p4 p2 p4)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
PROTO="${PROTOS[$SLURM_ARRAY_TASK_ID]}"
case "$DS" in
  1k) IN=data/processed/step1_1k_pairbest.parquet ;;
  10k) IN=data/processed/step1_10k_pairbest.parquet ;;
  full) IN=data/processed/step1_full_pairbest_content.parquet ;;
  *) echo "bad dataset $DS"; exit 2;;
esac
PREFIX="data/processed/step1_${DS}_pairbest_${PROTO}"
echo "[$(date +%H:%M:%S)] host=$(hostname) dataset=$DS protocol=$PROTO in=$IN"
if [ "$PROTO" = "p2" ]; then
  "$PY" scripts/step1_pairbest_resplit.py --in "$IN" --out-prefix "$PREFIX" --seeds 1,2,3,4,5
else
  "$PY" scripts/step1_p4_split.py --in "$IN" --out-prefix "$PREFIX" --seeds 1,2,3,4,5
fi
echo "[$(date +%H:%M:%S)] DONE $DS $PROTO"
