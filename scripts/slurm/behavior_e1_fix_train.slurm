#!/bin/bash
#SBATCH -p i64m1tga800ue
#SBATCH -J dcr-e1fix-train
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH --array=0-2
#SBATCH -o logs/behavior/e1fix_train_%A_%a.out
#SBATCH -e logs/behavior/e1fix_train_%A_%a.err
#
# E1_fix clean seed 1: fixed 1M pair views, no early stop, export last.
# SEED is the H4 player-split seed (1..5), not the model seed.
set -euo pipefail
cd "$HOME/DCR"
S=${SEED:-1}
mkdir -p logs/behavior runs/behavior/e1_fix/clean/s$S
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
PY=$HOME/.conda/envs/DCR/bin/python
IN=data/processed/step1_full_holdout_s$S.parquet
case "$SLURM_ARRAY_TASK_ID" in
  0) ARM=unfrozen_k4; INIT_ARGS="--init d2v --ckpt data2vec_main/runs/main_holdout/checkpoints/last.pt --freeze topk --unfreeze-blocks 4" ;;
  1) ARM=random; INIT_ARGS="--init random --freeze none" ;;
  2) ARM=frozen_samehead; INIT_ARGS="--init d2v --ckpt data2vec_main/runs/main_holdout/checkpoints/last.pt --freeze all" ;;
  *) echo "bad array task"; exit 1 ;;
esac
OUT=runs/behavior/e1_fix/clean/s$S/$ARM
echo "[$(date +%H:%M:%S)] host=$(hostname) arm=$ARM split=$S start"
$PY scripts/step2_chart_encoder.py \
  --task rank $INIT_ARGS \
  --config data2vec_main/configs/audio_style_base.yaml \
  --split-table "$IN" --osu-dir data/osu_cache \
  --window-sec 16 --frame-hz 100 --windows-per-chart 4 --max-windows 24 \
  --players-per-batch 4 --charts-per-player 6 \
  --pair-views 1000000 --patience 0 --eval-every 2000 --save-every 1000 \
  --log-every 50 --lr-encoder 5e-5 --lr-head 5e-4 \
  --bf16 --device cuda --seed 0 --export-checkpoint last \
  --out-dir "$OUT"
echo "[$(date +%H:%M:%S)] E1FIX_TRAIN_DONE arm=$ARM split=$S"
