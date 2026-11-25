#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-e3-teacher
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/behavior/e3_teacher_%j.out
#SBATCH -e logs/behavior/e3_teacher_%j.err
#
# Fit mf_dot on H4 train and build the K=64 reference-player signature.
set -euo pipefail
cd "$HOME/DCR"
mkdir -p logs/behavior results/behavior/e3/teacher_s1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
PY=$HOME/.conda/envs/DCR/bin/python

echo "[$(date +%H:%M:%S)] host=$(hostname) teacher fit start"
$PY scripts/step2_teacher_signature.py \
  --fit \
  --train-table data/processed/step1_full_holdout_s1.parquet \
  --tag full --dim 8 --epochs 60 --device cpu \
  --reference-players 64 --min-pairs 30 --holdout-frac 0.2 --seed 0 \
  --out results/behavior/e3/teacher_s1/signature.parquet \
  --json results/behavior/e3/teacher_s1/signature.json
echo "[$(date +%H:%M:%S)] E3_TEACHER_DONE"
