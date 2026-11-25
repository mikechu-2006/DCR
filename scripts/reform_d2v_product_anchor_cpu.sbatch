#!/bin/bash
# Encode the Reform anchors with the exact a20d product checkpoint used by the
# product linear head. This job is submitted with afterok:<product-build-job>.
#SBATCH --job-name=reform-d2v-prod-anchors
#SBATCH --partition=long_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/reform_d2v_product_anchors_%j.log

set -euo pipefail
cd "$HOME/DCR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$HOME/DCR/data2vec_main"

"$HOME/.conda/envs/DCR/bin/python" -m d2v_main.extract \
  --config data2vec_main/configs/audio_style_a20_product.yaml \
  --ckpt data2vec_main/runs/product/a20_full/checkpoints/last.pt \
  --osu-dir data/reform/inference_osu \
  --ids-file data/reform/d2v_ids.txt \
  --index-cache data/reform/d2v_product_anchor_chart_index.json \
  --out data/reform/chart_content_d2v_a20d_product_anchors.parquet \
  --device cpu \
  --batch-size 16 \
  --num-workers 6 \
  --tag d2v_a20d_product
