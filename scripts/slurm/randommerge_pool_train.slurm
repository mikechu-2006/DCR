#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-rm-pool-train
#SBATCH -o slurm_rm_pool_train_%A_%a.out
#SBATCH -e slurm_rm_pool_train_%A_%a.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-69%12
#
# Part B stage 2: one (protocol, seed, variant) per task, linear_extra only.
#   RUN = ID // 7 -> 0-4 P2 seeds 1-5; 5-9 P4 seeds 1-5
#   VAR = ID % 7 -> a0,a0l,a1,a2,a2b,a2c,a3
# No MLP: random10k shows no gain over linear.
set -euo pipefail
ID="${SLURM_ARRAY_TASK_ID:?}"
RUN=$((ID / 7))
VAR_IDX=$((ID % 7))
if [ "$RUN" -lt 5 ]; then PROTO=p2; SEED=$((RUN + 1)); else PROTO=p4; SEED=$((RUN - 4)); fi
VARIANTS=(a0 a0l a1 a2 a2b a2c a3)
VAR="${VARIANTS[$VAR_IDX]}"
cd "$HOME/DCR"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
echo "[$(date +%H:%M:%S)] host=$(hostname) proto=$PROTO seed=$SEED variant=$VAR"
python3 scripts/run_pool_randommerge.py \
  --stage train --protocols "$PROTO" --seeds "$SEED" --variants "$VAR" --threads 8
echo "[$(date +%H:%M:%S)] DONE $PROTO s$SEED $VAR"
