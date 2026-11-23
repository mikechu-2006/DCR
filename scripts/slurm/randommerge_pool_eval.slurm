#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-rm-pool-eval
#SBATCH -o slurm_rm_pool_eval_%A_%a.out
#SBATCH -e slurm_rm_pool_eval_%A_%a.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#
# Part B stage 3: merge the seven variant signals + fixed signals, then
# step2_rank_eval (common maps / strict pairs / ceiling all) and
# paired_delta_nll for A2b vs A0/A0L, A1 vs A0, A2/A2c/A3 vs A0L.
set -euo pipefail
ID="${SLURM_ARRAY_TASK_ID:?}"
if [ "$ID" -lt 5 ]; then PROTO=p2; SEED=$((ID + 1)); else PROTO=p4; SEED=$((ID - 4)); fi
cd "$HOME/DCR"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
echo "[$(date +%H:%M:%S)] host=$(hostname) proto=$PROTO seed=$SEED"
python3 scripts/run_pool_randommerge.py \
  --stage eval --protocols "$PROTO" --seeds "$SEED" --threads 8 --force-eval
echo "[$(date +%H:%M:%S)] DONE $PROTO s$SEED eval"
