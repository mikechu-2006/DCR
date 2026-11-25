#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-e3-eval
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o logs/behavior/e3_eval_%j.out
#SBATCH -e logs/behavior/e3_eval_%j.err
set -euo pipefail
cd "$HOME/DCR"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
PY=$HOME/.conda/envs/DCR/bin/python
$PY scripts/behavior_arm_eval.py \
  --runs runs/behavior/e3 \
  --split-table data/processed/step1_full_holdout_s1.parquet \
  --arms rank,mean,full,shuffled \
  --pairs full:mean,full:rank,shuffled:rank,mean:rank \
  --out-dir runs/behavior/e3/eval --bootstrap 2000
echo "[$(date +%H:%M:%S)] E3_EVAL_DONE"
