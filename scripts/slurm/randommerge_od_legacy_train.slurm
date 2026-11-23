#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-od-legacy-train
#SBATCH -o slurm_rm_od_train_%A_%a.out
#SBATCH -e slurm_rm_od_train_%A_%a.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-39%12
#
# Legacy L0-L3: one (protocol, split seed, arm) per task, linear_extra only.
#   RUN = ID // 4 -> 0-4 P2 seeds 1-5; 5-9 P4 seeds 1-5
#   ARM = ID % 4  -> l0,l1,l2,l3
# L0 = legacy A0 + msd_overall,sunny_star (control)
# L1 = +log_n; L2 = +od; L3 = +log_n+od
set -euo pipefail
ID="${SLURM_ARRAY_TASK_ID:?}"
RUN=$((ID / 4))
ARM_IDX=$((ID % 4))
if [ "$RUN" -lt 5 ]; then PROTO=p2; SEED=$((RUN + 1)); else PROTO=p4; SEED=$((RUN - 4)); fi
ARMS=(l0 l1 l2 l3)
ARM="${ARMS[$ARM_IDX]}"
cd "$HOME/DCR"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
echo "[$(date +%H:%M:%S)] host=$(hostname) proto=$PROTO seed=$SEED arm=$ARM"
python3 scripts/run_od_legacy.py \
  --stage train --protocols "$PROTO" --seeds "$SEED" --arms "$ARM" --threads 8
echo "[$(date +%H:%M:%S)] DONE $PROTO s$SEED $ARM"
