#!/bin/bash
#SBATCH -p i64m1tga800ue
#SBATCH -J dcr-e1-train
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH --array=0-2
#SBATCH -o logs/behavior/e1_train_%A_%a.out
#SBATCH -e logs/behavior/e1_train_%A_%a.err
#
# Experiment 1, seed 1: 0 = unfrozen top-4, 1 = random init.
# The frozen D2V reference is the H4 baseline array (d2v_linear/d2v_mlp).
set -euo pipefail
cd "$HOME/DCR"
S=${SEED:-1}
mkdir -p logs/behavior runs/behavior/e1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
PY=$HOME/.conda/envs/DCR/bin/python
IN=data/processed/step1_full_holdout_s$S.parquet

if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  ARM=unfrozen_k4
  INIT_ARGS="--init d2v --ckpt data2vec_main/runs/main_holdout/checkpoints/last.pt --freeze topk --unfreeze-blocks 4"
elif [ "$SLURM_ARRAY_TASK_ID" = "1" ]; then
  ARM=random
  INIT_ARGS="--init random --freeze none"
else
  ARM=frozen_samehead
  INIT_ARGS="--init d2v --ckpt data2vec_main/runs/main_holdout/checkpoints/last.pt --freeze all"
fi
OUT=runs/behavior/e1/$ARM/s$S

echo "[$(date +%H:%M:%S)] host=$(hostname) arm=$ARM seed=$S start"
$PY scripts/step2_chart_encoder.py \
  --task rank $INIT_ARGS \
  --config data2vec_main/configs/audio_style_base.yaml \
  --split-table "$IN" --osu-dir data/osu_cache \
  --window-sec 16 --frame-hz 100 --windows-per-chart 4 \
  --players-per-batch 4 --charts-per-player 6 \
  --pair-views 1000000 --eval-every 2000 --patience 5 \
  --log-every 50 --save-every 1000 \
  --lr-encoder 5e-5 --lr-head 5e-4 --bf16 --device cuda --seed 0 \
  --out-dir "$OUT"

echo "[$(date +%H:%M:%S)] arm=$ARM seed=$S eval"
$PY scripts/step2_rank_eval.py \
  --in "$IN" --signal-table "$OUT/d_test.parquet" --signals d \
  --split test --ceiling-fit all --strict-pairs --bootstrap 200 \
  --cache-dir data/processed/rank_eval_cache \
  --out "$OUT/eval.json"
echo "[$(date +%H:%M:%S)] E1_TRAIN_DONE arm=$ARM seed=$S"
