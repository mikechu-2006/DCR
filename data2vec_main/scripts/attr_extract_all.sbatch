#!/bin/bash
#SBATCH -J d2v_attr_ext
#SBATCH -p i64m1tga800ue
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o slurm_attr_extract_%j.out
#SBATCH -e slurm_attr_extract_%j.err
set -euo pipefail
cd "$HOME/DCR/data2vec_main"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
PY=$HOME/.conda/envs/DCR/bin/python
MODE=${MODE:?set MODE=s or d}
if [ "$MODE" = "s" ]; then
  ARMS="a0cont a12 a20 a27"; IDS="../data/processed/charts_1k.txt"; SUFFIX="1k"; RUNDIR="runs/attr"
elif [ "$MODE" = "d" ]; then
  ARMS="a0d a12d a20d a27d"; IDS="splits/mapset_holdout/eval_ids.txt"; SUFFIX="mclean"; RUNDIR="runs/attr_d"
else
  echo "bad MODE=$MODE"; exit 2
fi
echo "[attr-extract] mode=$MODE host=$(hostname) date=$(date)"
for ARM in $ARMS; do
  CKPT="$RUNDIR/$ARM/checkpoints/last.pt"
  OUT="../data/processed/chart_content_d2v_${ARM}-${SUFFIX}.parquet"
  echo "[attr-extract] ARM=$ARM ckpt=$CKPT out=$OUT"
  test -f "$CKPT" || { echo "missing ckpt $CKPT"; exit 3; }
  $PY -m d2v_main.extract --config configs/audio_style_base.yaml \
    --ckpt "$CKPT" --ids-file "$IDS" --out "$OUT" \
    --device cuda --batch-size 32 --num-workers 6 --tag "d2v_${ARM}"
done
echo "[attr-extract] done $(date)"
