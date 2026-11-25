#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-p4sets-train
#SBATCH -o slurm_p4sets_train_%A_%a.out
#SBATCH -e slurm_p4sets_train_%A_%a.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#
# Experiment 2b: audio-on vs audio-off on the beatmapset-held-out P4.
# One array task per arm; the runner loops over 5 seeds x 4 methods.
set -euo pipefail
cd "$HOME/DCR"
PY="$HOME/.conda/envs/DCR/bin/python"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
ARMS=(audio legacy)
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
if [ "$ARM" = "audio" ]; then
  CONTENT=data/processed/chart_content_cm3p_audio_on.parquet
else
  CONTENT=data/processed/chart_content_cm3p_local.parquet
fi
OUT="results/audio_newset/p4sets/${ARM}"
echo "[$(date +%H:%M:%S)] host=$(hostname) arm=$ARM content=$CONTENT"
"$PY" scripts/run_p2_p4_multiseed.py \
  --protocols p4 --seeds 1,2,3,4,5 \
  --models linear,linear_extra,mlp,mlp_extra \
  --split-p4 data/processed/step1_1k_pairbest_p4sets \
  --content-path "$CONTENT" \
  --out-dir "$OUT" --threads 8
echo "[$(date +%H:%M:%S)] DONE $ARM"
