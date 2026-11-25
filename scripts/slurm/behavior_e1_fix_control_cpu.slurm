#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-e1fix-control-cpu
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/behavior/e1fix_control_cpu_%j.out
#SBATCH -e logs/behavior/e1fix_control_cpu_%j.err
#
# F4 head-only control on cached frozen features (no encoder forward).
set -euo pipefail
cd "$HOME/DCR"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
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
  --seed 0 --device cpu --conditions c0,c1,c2,c3 \
  --out-dir runs/behavior/e1_fix/control/s1
echo E1FIX_CONTROL_CPU_DONE
