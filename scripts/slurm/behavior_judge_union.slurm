#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-judge-union
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/behavior/judge_union_%j.out
#SBATCH -e logs/behavior/judge_union_%j.err
#
# Merge the per-source judgement tables onto the union play table.
set -euo pipefail
cd "$HOME/DCR"
mkdir -p logs/behavior data/processed
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
PY=$HOME/.conda/envs/DCR/bin/python

echo "[$(date +%H:%M:%S)] host=$(hostname) union judge start"
$PY scripts/step0_union_judge.py \
  --judge 1k=data/processed/step0_judge_1k.parquet \
  --judge 10k=data/processed/step0_judge_10k.parquet \
  --judge randommerge=data/processed/step0_judge_random_20260713.parquet \
  --judge randommerge=data/processed/step0_judge_random_20260801.parquet \
  --judge randommerge=data/processed/step0_judge_random_20260901.parquet \
  --judge randommerge=data/processed/step0_judge_random_20261105.parquet \
  --plays data/processed/step0_full.parquet \
  --out data/processed/step0_judge_full.parquet \
  --report data/processed/step0_judge_full.json
echo "[$(date +%H:%M:%S)] JUDGE_UNION_DONE"
