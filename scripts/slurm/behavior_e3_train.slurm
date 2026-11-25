#!/bin/bash
#SBATCH -p i64m1tga800ue
#SBATCH -J dcr-e3-train
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --array=0-3
#SBATCH -o logs/behavior/e3_train_%A_%a.out
#SBATCH -e logs/behavior/e3_train_%A_%a.err
#
# Experiment 3, seed 1, H4:
#   0 = d-only rank-NLL
#   1 = + mean teacher difficulty (scalar control, B1)
#   2 = + K=64 reference-player signature (B2)
#   3 = + shuffled signature (placebo, B3)
set -euo pipefail
cd "$HOME/DCR"
S=1
mkdir -p logs/behavior runs/behavior/e3
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
PY=$HOME/.conda/envs/DCR/bin/python
IN=data/processed/step1_full_holdout_s$S.parquet
SIG=results/behavior/e3/teacher_s1/signature.parquet
case "$SLURM_ARRAY_TASK_ID" in
0) ARM=rank; TASK=rank; AUX=none; TW=0.0; EXTRA="" ;;
1) ARM=mean; TASK=teacher; AUX=mean; TW=1.0; EXTRA="--teacher-dim 1" ;;
2) ARM=full; TASK=teacher; AUX=full; TW=1.0; EXTRA="--teacher-dim 64" ;;
3) ARM=shuffled; TASK=teacher; AUX=shuffled; TW=1.0; EXTRA="--teacher-dim 64" ;;
esac
D=runs/behavior/e3/$ARM/s$S

echo "[$(date +%H:%M:%S)] host=$(hostname) E3 arm=$ARM start"
ARGS=(--task "$TASK" --aux "$AUX" --teacher-weight "$TW" $EXTRA)
if [ "$TASK" = teacher ]; then ARGS+=(--teacher-signature "$SIG"); fi
$PY scripts/step2_chart_encoder.py "${ARGS[@]}" \
  --init d2v --ckpt data2vec_main/runs/main_holdout/checkpoints/last.pt \
  --config data2vec_main/configs/audio_style_base.yaml \
  --freeze topk --unfreeze-blocks 4 \
  --split-table "$IN" --osu-dir data/osu_cache \
  --window-sec 16 --frame-hz 100 --max-windows 24 \
  --players-per-batch 4 --charts-per-player 6 \
  --pair-views 300000 --eval-every 1000 --patience 5 \
  --log-every 25 --save-every 500 --eval-batch-charts 8 \
  --lr-encoder 1e-4 --lr-head 1e-3 --bf16 --device cuda --seed 0 \
  --export-checkpoint last --out-dir "$D"

echo "[$(date +%H:%M:%S)] E3 arm=$ARM eval"
$PY scripts/step2_rank_eval.py \
  --in "$IN" --signal-table "$D/d_test.parquet" --signals d \
  --split test --ceiling-fit all --strict-pairs --bootstrap 200 \
  --cache-dir data/processed/rank_eval_cache \
  --out "$D/eval.json"
echo "[$(date +%H:%M:%S)] E3_TRAIN_DONE arm=$ARM"
