#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-rm-mlp
#SBATCH -o slurm_rm_mlp_%A_%a.out
#SBATCH -e slurm_rm_mlp_%A_%a.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-19
#
# One array task = one (model, protocol, split seed).
#   ID  0-9  -> model mlp_extra; 10-19 -> model mlp
#   ID%10 0-4 -> P2 seeds 1-5;   5-9   -> P4 seeds 1-5
#
# 98 epochs / 5 chart folds (the runner's defaults), model seed 0.  Same splits,
# content table and eval as the linear wave (job 14021385), so the MLP rows drop
# straight into results/randommerge_p2p4_hpc/summary.csv.
set -euo pipefail
ID="${SLURM_ARRAY_TASK_ID:?}"
BASE=$((ID % 10))
if [ "$ID" -lt 10 ]; then MODEL=mlp_extra; else MODEL=mlp; fi
if [ "$BASE" -lt 5 ]; then PROTO=p2; SEED=$((BASE + 1)); else PROTO=p4; SEED=$((BASE - 4)); fi
cd "$HOME/DCR"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1
echo "[$(date +%H:%M:%S)] host=$(hostname) proto=$PROTO seed=$SEED model=$MODEL"
python3 scripts/run_p2_p4_multiseed.py \
  --protocols "$PROTO" --seeds "$SEED" \
  --models "$MODEL" \
  --threads 8 --epochs 98 \
  --split-p2 data/processed/step1_randommerge_c_p2 \
  --split-p4 data/processed/step1_randommerge_c_p4 \
  --out-dir results/randommerge_p2p4_hpc
echo "[$(date +%H:%M:%S)] DONE $PROTO s$SEED $MODEL"
