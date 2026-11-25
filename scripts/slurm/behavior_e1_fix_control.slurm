#!/bin/bash
#SBATCH -p i64m1tga800ue
#SBATCH -J dcr-e1fix-control
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o logs/behavior/e1fix_control_%j.out
#SBATCH -e logs/behavior/e1fix_control_%j.err
#
# F4: frozen D2V, K4 vs all-window 2x2.  Features are cached once; only the
# shared ChartHead is trained for each condition.
set -euo pipefail
cd "$HOME/DCR"
mkdir -p logs/behavior runs/behavior/e1_fix/control/s1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
PY=$HOME/.conda/envs/DCR/bin/python
$PY scripts/e1_frozen_window_control.py \
  --split-table data/processed/step1_full_holdout_s1.parquet \
  --osu-dir data/osu_cache \
  --ckpt data2vec_main/runs/main_holdout/checkpoints/last.pt \
  --config data2vec_main/configs/audio_style_base.yaml \
  --window-sec 16 --frame-hz 100 \
  --players-per-batch 4 --charts-per-player 6 --windows-per-chart 4 \
  --pair-views 1000000 --eval-every 2000 \
  --hid 256 --dropout 0.1 --lr-head 5e-4 --wd-head 0.05 \
  --seed 0 --device cuda --conditions c0,c1,c2,c3 \
  --out-dir runs/behavior/e1_fix/control/s1
echo E1FIX_CONTROL_DONE
