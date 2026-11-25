#!/bin/bash
#SBATCH -p debug
#SBATCH -J dcr-e1-pilot
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH --array=0-2
#SBATCH -o logs/behavior/e1_pilot_%A_%a.out
#SBATCH -e logs/behavior/e1_pilot_%A_%a.err
#
# Short 60k-view pilot on the debug GPU (30 min cap).  Same three arms; output
# is directional only, not the main result.
set -euo pipefail
cd "$HOME/DCR"
S=1
mkdir -p logs/behavior runs/behavior/e1_pilot
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
D=runs/behavior/e1_pilot/$ARM/s$S

echo "[$(date +%H:%M:%S)] host=$(hostname) pilot arm=$ARM start"
$PY scripts/step2_chart_encoder.py \
  --task rank $INIT_ARGS \
  --config data2vec_main/configs/audio_style_base.yaml \
  --split-table "$IN" --osu-dir data/osu_cache \
  --window-sec 16 --frame-hz 100 --windows-per-chart 4 \
  --players-per-batch 4 --charts-per-player 6 \
  --pair-views 60000 --eval-every 0 --patience 0 \
  --log-every 50 --save-every 500 \
  --lr-encoder 5e-5 --lr-head 5e-4 --bf16 --device cuda --seed 0 \
  --out-dir "$D"

echo "[$(date +%H:%M:%S)] pilot arm=$ARM eval"
$PY scripts/step2_rank_eval.py \
  --in "$IN" --signal-table "$D/d_test.parquet" --signals d \
  --split test --ceiling-fit all --strict-pairs --bootstrap 100 \
  --cache-dir data/processed/rank_eval_cache \
  --out "$D/eval.json"
echo "[$(date +%H:%M:%S)] E1_PILOT_DONE arm=$ARM"
