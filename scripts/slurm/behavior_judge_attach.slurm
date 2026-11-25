#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-judge-attach
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/behavior/judge_attach_%j.out
#SBATCH -e logs/behavior/judge_attach_%j.err
#
# Attach the six counts to step1/pairbest, validate against count_total and
# the .osu object count, then build the judge-aware H4 splits.
set -euo pipefail
cd "$HOME/DCR"
mkdir -p logs/behavior data/processed
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
PY=$HOME/.conda/envs/DCR/bin/python

echo "[$(date +%H:%M:%S)] host=$(hostname) attach start"
$PY scripts/step1_attach_judgements.py \
  --in data/processed/step1_full_pairbest.parquet \
  --judge data/processed/step0_judge_full.parquet \
  --meta data/processed/beatmap_meta_full.csv \
  --osu-dir data/osu_cache \
  --out data/processed/step1_full_pairbest_judge.parquet \
  --json data/processed/step1_full_pairbest_judge.json

echo "[$(date +%H:%M:%S)] attach done; H4 judge splits"
$PY scripts/step1_holdout_split.py \
  --in data/processed/step1_full_pairbest_judge.parquet \
  --holdout-ids data2vec_main/splits/holdout/eval_ids.txt \
  --out-prefix data/processed/step1_full_holdout_judge \
  --seeds 1,2,3,4,5 --player-frac 0.6 --bins 5 --val-chart-frac 0.1
echo "[$(date +%H:%M:%S)] JUDGE_ATTACH_DONE"
