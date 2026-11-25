#!/bin/bash
#SBATCH -p i64m1tga800ue
#SBATCH -J dcr-e2-fixed
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --array=0-4
#SBATCH -o logs/behavior/e2_fixed_%A_%a.out
#SBATCH -e logs/behavior/e2_fixed_%A_%a.err
#
# E2 fixed-budget rerun (seed 1, H4).  Every arm trains 300k pair views;
# --patience 0 disables early stopping so the pilot's 7k-vs-11k-step
# difference cannot explain the result.
#   0 rank / 1 six-grade sqrt_inv / 2 shuffled / 3 miss-only / 4 six-grade unweighted
set -euo pipefail
cd "$HOME/DCR"
S=1
mkdir -p logs/behavior runs/behavior/e2_fixed
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
PY=$HOME/.conda/envs/DCR/bin/python
IN=data/processed/step1_full_holdout_judge_s$S.parquet
case "$SLURM_ARRAY_TASK_ID" in
0) ARM=rank; AUX=none; JW=0.0; CW=sqrt_inv; EXTRA="" ;;
1) ARM=judge; AUX=judge; JW=1.0; CW=sqrt_inv; EXTRA="" ;;
2) ARM=judge_shuffled; AUX=judge_shuffled; JW=1.0; CW=sqrt_inv; EXTRA="" ;;
3) ARM=single_grade_miss; AUX=single_grade; JW=1.0; CW=sqrt_inv; EXTRA="--single-grade-col n_miss" ;;
4) ARM=judge_none; AUX=judge; JW=1.0; CW=none; EXTRA="" ;;
esac
D=runs/behavior/e2_fixed/$ARM/s$S

echo "[$(date +%H:%M:%S)] host=$(hostname) E2-fixed arm=$ARM start"
$PY scripts/step2_chart_encoder.py \
  --task judge --aux "$AUX" --judge-weight "$JW" $EXTRA \
  --judge-class-weights "$CW" \
  --init d2v --ckpt data2vec_main/runs/main_holdout/checkpoints/last.pt \
  --config data2vec_main/configs/audio_style_base.yaml \
  --freeze topk --unfreeze-blocks 4 \
  --split-table "$IN" --osu-dir data/osu_cache \
  --window-sec 16 --frame-hz 100 --max-windows 24 \
  --players-per-batch 3 --charts-per-player 5 \
  --pair-views 300000 --eval-every 1000 --patience 0 \
  --log-every 25 --save-every 1000 --eval-batch-charts 4 \
  --lr-encoder 1e-4 --lr-head 1e-3 --bf16 --device cuda --seed 0 \
  --export-checkpoint last --out-dir "$D"

echo "[$(date +%H:%M:%S)] E2-fixed arm=$ARM eval"
$PY scripts/step2_rank_eval.py \
  --in "$IN" --signal-table "$D/d_test.parquet" --signals d \
  --split test --ceiling-fit all --strict-pairs --bootstrap 200 \
  --cache-dir data/processed/rank_eval_cache \
  --out "$D/eval.json"
echo "[$(date +%H:%M:%S)] E2_FIXED_TRAIN_DONE arm=$ARM"
