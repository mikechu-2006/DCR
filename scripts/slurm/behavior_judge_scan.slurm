#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-judge-scan
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --array=0-5
#SBATCH -o logs/behavior/judge_scan_%A_%a.out
#SBATCH -e logs/behavior/judge_scan_%A_%a.err
#
# Six-grade judgement side tables for the four sources.  Each array task is
# independent; the union job depends on all of them.
set -euo pipefail
cd "$HOME/DCR"
mkdir -p logs/behavior data/processed
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
PY=$HOME/.conda/envs/DCR/bin/python

echo "[$(date +%H:%M:%S)] host=$(hostname) task=$SLURM_ARRAY_TASK_ID start"
case "$SLURM_ARRAY_TASK_ID" in
0)
  $PY scripts/step0_judgements.py \
    --dump-dir data/raw/extracted/2026_09_01_performance_mania_top_1000 \
    --tag 1k --source-id 1 \
    --out data/processed/step0_judge_1k.parquet
  ;;
1)
  $PY scripts/step0_judgements.py \
    --dump-dir data/raw/extracted_10k/2026_09_01_performance_mania_top_10000 \
    --beatmap-src data/raw/extracted/2026_09_01_performance_mania_top_1000 \
    --tag 10k --source-id 2 \
    --out data/processed/step0_judge_10k.parquet
  ;;
2)
  $PY scripts/step0_judgements.py \
    --dump-dir data/raw/extracted_random/2026_07_13_performance_mania_random_10000 \
    --beatmap-src data/raw/extracted/2026_09_01_performance_mania_top_1000 \
    --tag random_20260713 --source-id 3 \
    --out data/processed/step0_judge_random_20260713.parquet
  ;;
3)
  $PY scripts/step0_judgements.py \
    --dump-dir data/raw/extracted_random/2026_08_01_performance_mania_random_10000 \
    --beatmap-src data/raw/extracted/2026_09_01_performance_mania_top_1000 \
    --tag random_20260801 --source-id 3 \
    --out data/processed/step0_judge_random_20260801.parquet
  ;;
4)
  $PY scripts/step0_judgements.py \
    --dump-dir data/raw/extracted_random/2026_09_01_performance_mania_random_10000 \
    --beatmap-src data/raw/extracted/2026_09_01_performance_mania_top_1000 \
    --tag random_20260901 --source-id 3 \
    --out data/processed/step0_judge_random_20260901.parquet
  ;;
5)
  $PY scripts/step0_judgements.py \
    --dump-dir data/raw/extracted_random/2026_11_05_performance_mania_random_10000 \
    --beatmap-src data/raw/extracted/2026_09_01_performance_mania_top_1000 \
    --tag random_20261105 --source-id 3 \
    --out data/processed/step0_judge_random_20261105.parquet
  ;;
esac
echo "[$(date +%H:%M:%S)] JUDGE_SCAN_DONE task=$SLURM_ARRAY_TASK_ID"
