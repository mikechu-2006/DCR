#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-rm-pool-feat
#SBATCH -o slurm_rm_pool_feat_%A_%a.out
#SBATCH -e slurm_rm_pool_feat_%A_%a.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#
# Part B stage 1: one (protocol, split seed) per task.
#   ID 0-4 -> P2 seeds 1-5; 5-9 -> P4 seeds 1-5.
# Fits the per-seed A0 head (P2: 5 chart folds aligned with the downstream
# model; P4: the P4 train rows only) and builds A2/A2b/A2c/A3 for that seed.
set -euo pipefail
ID="${SLURM_ARRAY_TASK_ID:?}"
if [ "$ID" -lt 5 ]; then PROTO=p2; SEED=$((ID + 1)); else PROTO=p4; SEED=$((ID - 4)); fi
cd "$HOME/DCR"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
echo "[$(date +%H:%M:%S)] host=$(hostname) proto=$PROTO seed=$SEED"
python3 scripts/run_pool_randommerge.py \
  --stage features --protocols "$PROTO" --seeds "$SEED" --threads 8
echo "[$(date +%H:%M:%S)] DONE $PROTO s$SEED features"
