#!/bin/bash
#SBATCH -p i64m1tga800ue
#SBATCH -J dcr-e2-train
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --array=0-3
#SBATCH -o logs/behavior/e2_train_%A_%a.out
#SBATCH -e logs/behavior/e2_train_%A_%a.err
#
# Experiment 2, seed 1, H4:
#   0 = rank only (judge head present but weight 0; same rows/budget)
#   1 = + six-grade proportions
#   2 = + shuffled proportions (row/column-sum-preserving placebo)
#   3 = + miss-only (mechanism)
set -euo pipefail
cd "$HOME/DCR"
S=1
mkdir -p logs/behavior runs/behavior/e2
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
PY=$HOME/.conda/envs/DCR/bin/python
IN=data/processed/step1_full_holdout_judge_s$S.parquet
case "$SLURM_ARRAY_TASK_ID" in
0) ARM=rank; AUX=none; JW=0.0; EXTRA="" ;;
1) ARM=judge; AUX=judge; JW=1.0; EXTRA="" ;;
2) ARM=judge_shuffled; AUX=judge_shuffled; JW=1.0; EXTRA="" ;;
3) ARM=single_grade_miss; AUX=single_grade; JW=1.0; EXTRA="--single-grade-col n_miss" ;;
esac
D=runs/behavior/e2/$ARM/s$S

echo "[$(date +%H:%M:%S)] host=$(hostname) E2 arm=$ARM start"
$PY scripts/step2_chart_encoder.py \
  --task judge --aux "$AUX" --judge-weight "$JW" $EXTRA \
  --judge-class-weights sqrt_inv \
  --init d2v --ckpt data2vec_main/runs/main_holdout/checkpoints/last.pt \
  --config data2vec_main/configs/audio_style_base.yaml \
  --freeze topk --unfreeze-blocks 4 \
  --split-table "$IN" --osu-dir data/osu_cache \
  --window-sec 16 --frame-hz 100 --max-windows 24 \
  --players-per-batch 3 --charts-per-player 5 \
  --pair-views 300000 --eval-every 1000 --patience 5 \
  --log-every 25 --save-every 500 --eval-batch-charts 4 \
  --lr-encoder 1e-4 --lr-head 1e-3 --bf16 --device cuda --seed 0 \
  --export-checkpoint last --out-dir "$D"

echo "[$(date +%H:%M:%S)] E2 arm=$ARM eval"
$PY scripts/step2_rank_eval.py \
  --in "$IN" --signal-table "$D/d_test.parquet" --signals d \
  --split test --ceiling-fit all --strict-pairs --bootstrap 200 \
  --cache-dir data/processed/rank_eval_cache \
  --out "$D/eval.json"
echo "[$(date +%H:%M:%S)] E2_TRAIN_DONE arm=$ARM"
