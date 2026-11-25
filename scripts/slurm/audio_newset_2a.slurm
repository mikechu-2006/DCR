#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-audio-2a
#SBATCH -o slurm_audio_2a_%j.out
#SBATCH -e slurm_audio_2a_%j.err
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# Experiment 2a: recompute audio vs legacy on new-beatmapset test pairs
# using the existing P4 predictions (no retraining).
set -euo pipefail
cd "$HOME/DCR"
PY="$HOME/.conda/envs/DCR/bin/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
echo "[$(date +%H:%M:%S)] host=$(hostname) 2a"
"$PY" scripts/eval_newset_audio.py \
  --split-prefix data/processed/step1_1k_pairbest_p4_nogap \
  --audio-dir results/p4_multiseed_audio/audio/p4 \
  --legacy-dir results/p4_multiseed_audio/legacy/p4 \
  --meta data/processed/beatmap_meta_1k.csv \
  --seeds 1,2,3,4,5 --bootstrap 400 \
  --out-dir results/audio_newset
echo "[$(date +%H:%M:%S)] DONE 2a"
