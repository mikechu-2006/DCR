#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-p4sets-subset
#SBATCH -o slurm_p4sets_subset_%j.out
#SBATCH -e slurm_p4sets_subset_%j.err
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# Strict cross-set subset of the beatmapset-held-out P4: all test sets are
# unseen by construction; new_cross additionally requires the two charts to
# come from different sets.
set -euo pipefail
cd "$HOME/DCR"
PY="$HOME/.conda/envs/DCR/bin/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
echo "[$(date +%H:%M:%S)] host=$(hostname) p4sets subset"
"$PY" scripts/eval_newset_audio.py \
  --split-prefix data/processed/step1_1k_pairbest_p4sets \
  --audio-dir results/audio_newset/p4sets/audio/p4 \
  --legacy-dir results/audio_newset/p4sets/legacy/p4 \
  --meta data/processed/beatmap_meta_1k.csv \
  --seeds 1,2,3,4,5 --bootstrap 400 \
  --out-dir results/audio_newset/p4sets/subset
echo "[$(date +%H:%M:%S)] DONE p4sets subset"
