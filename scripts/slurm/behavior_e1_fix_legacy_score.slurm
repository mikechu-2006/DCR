#!/bin/bash
#SBATCH -p i64m1tga800ue
#SBATCH -J dcr-e1fix-legacy
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/behavior/e1fix_legacy_%j.out
#SBATCH -e logs/behavior/e1fix_legacy_%j.err
#
# Deterministic re-score of the legacy seed-1 checkpoints.  These are the
# last-saved checkpoint.pt files, not necessarily the in-memory final weights
# used by the original report; label them legacy/last-available.
set -euo pipefail
cd "$HOME/DCR"
mkdir -p logs/behavior runs/behavior/e1_fix/legacy/s1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
PY=$HOME/.conda/envs/DCR/bin/python
IN=data/processed/step1_full_holdout_s1.parquet
for ARM in unfrozen_k4 random frozen_samehead; do
  OUT=runs/behavior/e1_fix/legacy/s1/$ARM/score
  echo "[$(date +%H:%M:%S)] score $ARM -> $OUT"
  $PY scripts/step2_chart_encoder.py --score-only --task rank \
    --checkpoint "runs/behavior/e1/$ARM/s1/checkpoint.pt" \
    --config data2vec_main/configs/audio_style_base.yaml \
    --split-table "$IN" --osu-dir data/osu_cache \
    --max-windows 24 --repeat-eval 2 --eval-batch-charts 8 \
    --eval-diff-tol 1e-6 --device cuda --seed 0 --out-dir "$OUT"
done
echo E1FIX_LEGACY_SCORE_DONE
