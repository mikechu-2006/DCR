#!/bin/bash
#SBATCH -p debug
#SBATCH -J dcr-e1-bench
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/behavior/e1_bench_%j.out
#SBATCH -e logs/behavior/e1_bench_%j.err
#
# 30-step GPU throughput / correctness smoke on the real D2V checkpoint.
set -euo pipefail
cd "$HOME/DCR"
mkdir -p logs/behavior runs/behavior
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
PY=$HOME/.conda/envs/DCR/bin/python

echo "[$(date +%H:%M:%S)] host=$(hostname) e1 bench start"
$PY scripts/step2_chart_encoder.py \
  --task rank --init d2v \
  --ckpt data2vec_main/runs/main_holdout/checkpoints/last.pt \
  --config data2vec_main/configs/audio_style_base.yaml \
  --freeze topk --unfreeze-blocks 4 \
  --split-table data/processed/step1_full_holdout_s1.parquet \
  --osu-dir data/osu_cache \
  --window-sec 16 --frame-hz 100 --windows-per-chart 4 \
  --players-per-batch 4 --charts-per-player 6 \
  --pair-views 1000000 --benchmark-views 30 --max-steps 30 \
  --eval-every 0 --log-every 5 --hid 256 \
  --lr-encoder 1e-4 --lr-head 1e-3 --bf16 --device cuda --seed 0 \
  --out-dir runs/behavior/bench_e1_s1
echo "[$(date +%H:%M:%S)] E1_BENCH_DONE"
