#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-od-legacy-eval
#SBATCH -o slurm_rm_od_eval_%A_%a.out
#SBATCH -e slurm_rm_od_eval_%A_%a.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#
# Per (protocol, seed): merge L0-L3 + fixed, rank eval, paired deltas.
set -euo pipefail
ID="${SLURM_ARRAY_TASK_ID:?}"
if [ "$ID" -lt 5 ]; then PROTO=p2; SEED=$((ID + 1)); else PROTO=p4; SEED=$((ID - 4)); fi
cd "$HOME/DCR"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
echo "[$(date +%H:%M:%S)] host=$(hostname) proto=$PROTO seed=$SEED"
python3 scripts/run_od_legacy.py \
  --stage eval --protocols "$PROTO" --seeds "$SEED" --threads 8 --force-eval
echo "[$(date +%H:%M:%S)] DONE $PROTO s$SEED eval"
