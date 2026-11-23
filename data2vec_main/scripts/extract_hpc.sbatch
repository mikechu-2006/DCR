#!/bin/bash
#SBATCH -J d2v_extract
#SBATCH -p emergency_gpu
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o slurm_d2v_extract_%j.out
#SBATCH -e slurm_d2v_extract_%j.err
set -euo pipefail
cd "$HOME/DCR/data2vec_main"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
PY=$HOME/.conda/envs/DCR/bin/python
echo "[extract] host=$(hostname) date=$(date) job=$SLURM_JOB_ID"
$PY -m d2v_main.extract --config configs/audio_style_base.yaml \
  --ckpt runs/main_holdout/checkpoints/last.pt \
  --out ../data/processed/chart_content_d2v_holdout.parquet \
  --device cuda --batch-size 32 --num-workers 6 --tag d2v_main_holdout
