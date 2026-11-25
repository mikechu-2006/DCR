#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-h4-split
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/behavior/h4_split_%j.out
#SBATCH -e logs/behavior/h4_split_%j.err
#
# H4: fixed D2V-unseen chart holdout, 5 player splits, chart-disjoint val.
set -euo pipefail
cd "$HOME/DCR"
mkdir -p logs/behavior data/processed
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
PY=$HOME/.conda/envs/DCR/bin/python

echo "[$(date +%H:%M:%S)] host=$(hostname) H4 split start"
$PY scripts/step1_holdout_split.py \
  --in data/processed/step1_full_pairbest.parquet \
  --holdout-ids data2vec_main/splits/holdout/eval_ids.txt \
  --out-prefix data/processed/step1_full_holdout \
  --seeds 1,2,3,4,5 --player-frac 0.6 --bins 5 --val-chart-frac 0.1
echo "[$(date +%H:%M:%S)] H4_SPLIT_DONE"
