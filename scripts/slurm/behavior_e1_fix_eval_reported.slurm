#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-e1fix-reported
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/behavior/e1fix_reported_%j.out
#SBATCH -e logs/behavior/e1fix_reported_%j.err
#
# Diagnostic: strict-pair re-analysis of the *reported* legacy d_test files.
# These were produced by the old nondeterministic export; keep them separate
# from the deterministic re-score.
set -euo pipefail
cd "$HOME/DCR"
mkdir -p logs/behavior runs/behavior/e1_fix/reported/s1/eval
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
PY=$HOME/.conda/envs/DCR/bin/python
$PY scripts/e1_fix_eval.py \
  --runs runs/behavior/e1 \
  --split-table data/processed/step1_full_holdout_s1.parquet \
  --baseline-dir results/behavior/baselines/h4_s1 \
  --out-dir runs/behavior/e1_fix/reported/s1/eval \
  --cache-dir data/processed/rank_eval_cache \
  --bootstrap 2000 --seed 0
echo E1FIX_REPORTED_EVAL_DONE
