#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-judge-h4-resplit
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/behavior/judge_h4_resplit_%j.out
#SBATCH -e logs/behavior/judge_h4_resplit_%j.err
set -euo pipefail
cd "$HOME/DCR"
mkdir -p logs/behavior data/processed
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
PY=$HOME/.conda/envs/DCR/bin/python

echo "[$(date +%H:%M:%S)] judge H4 resplit start"
$PY scripts/step1_holdout_split.py \
  --in data/processed/step1_full_pairbest_judge.parquet \
  --holdout-ids data2vec_main/splits/holdout/eval_ids.txt \
  --out-prefix data/processed/step1_full_holdout_judge \
  --seeds 1,2,3,4,5 --player-frac 0.6 --bins 5 --val-chart-frac 0.1 --force
echo "[$(date +%H:%M:%S)] JUDGE_H4_RESPLIT_DONE"
