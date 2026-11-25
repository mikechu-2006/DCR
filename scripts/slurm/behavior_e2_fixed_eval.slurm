#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-e2-fixed-eval
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o logs/behavior/e2_fixed_eval_%j.out
#SBATCH -e logs/behavior/e2_fixed_eval_%j.err
set -euo pipefail
cd "$HOME/DCR"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
PY=$HOME/.conda/envs/DCR/bin/python
$PY scripts/behavior_arm_eval.py \
  --runs runs/behavior/e2_fixed \
  --split-table data/processed/step1_full_holdout_judge_s1.parquet \
  --arms rank,judge,judge_shuffled,single_grade_miss,judge_none \
  --pairs judge:rank,judge_shuffled:rank,single_grade_miss:rank,judge_none:rank,judge:judge_shuffled,judge:judge_none \
  --out-dir runs/behavior/e2_fixed/eval --bootstrap 2000
echo "[$(date +%H:%M:%S)] E2_FIXED_EVAL_DONE"
