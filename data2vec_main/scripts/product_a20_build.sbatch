#!/bin/bash
#SBATCH -J d2v_a20d_prod
#SBATCH -p emergency_gpu
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o slurm_d2v_a20d_prod_%j.out
#SBATCH -e slurm_d2v_a20d_prod_%j.err
#
# PRODUCT build for the a20d structured-attribute D2V encoder:
#   1. pretrain from scratch on ALL 21,949 whitelist charts (not the
#      16,487-chart mapset-disjoint evaluation split)
#   2. encode the whitelist  -> chart_content_d2v_a20d_prod.parquet
#   3. encode the VSRG benchmark staging -> benchmark_content_d2v_a20d_prod.parquet
#
# The linear head / isotonic calibration are fit locally afterwards
# (scripts/export_a20d_head.py) with the same contract as the existing deploy JSONs.
set -euo pipefail
cd "$HOME/DCR"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$HOME/DCR/data2vec_main"
PY=$HOME/.conda/envs/DCR/bin/python
CFG=data2vec_main/configs/audio_style_a20_product.yaml

echo "[prod] host=$(hostname) date=$(date) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true

echo "[prod] 1/3 pretrain on ALL cached charts (no --ids-file)"
$PY -m d2v_main.train --config "$CFG" \
  --out-dir data2vec_main/runs/product/a20_full \
  --device cuda --epochs 20 --batch-size 16 --num-workers 6 --log-every 100

CKPT=data2vec_main/runs/product/a20_full/checkpoints/last.pt
test -f "$CKPT" || { echo "missing ckpt $CKPT"; exit 3; }

echo "[prod] 2/3 encode whitelist"
$PY -m d2v_main.extract --config "$CFG" --ckpt "$CKPT" \
  --out data/processed/chart_content_d2v_a20d_prod.parquet \
  --device cuda --batch-size 32 --num-workers 6 --tag d2v_a20d_prod

echo "[prod] 3/3 encode benchmark + finalize"
$PY -m d2v_main.extract --config "$CFG" --ckpt "$CKPT" \
  --osu-dir data/benchmark_d2v_osu \
  --ids-file data/processed/benchmark_d2v_ids.txt \
  --out data/processed/benchmark_content_d2v_a20d_prod_raw.parquet \
  --device cuda --batch-size 32 --num-workers 6 --tag benchmark_d2v_a20d_prod
$PY scripts/finalize_benchmark_d2v.py \
  --inp data/processed/benchmark_content_d2v_a20d_prod_raw.parquet \
  --out data/processed/benchmark_content_d2v_a20d_prod.parquet
echo "[prod] DONE $(date)"
