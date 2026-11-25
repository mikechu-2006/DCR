#!/bin/bash
#SBATCH --job-name=reform-d2v
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_reform_d2v_%j.log

set -euo pipefail
cd "$HOME/DCR"
export PYTHONPATH="$HOME/DCR/data2vec_main"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
PY="$HOME/.conda/envs/DCR/bin/python"
CKPT=data2vec_main/runs/attr_d/a20d/checkpoints/last.pt
test -s "$CKPT"
test "$(find data/reform/inference_osu -maxdepth 1 -name '*.osu' | wc -l)" -eq 117

"$PY" -m d2v_main.extract \
  --config data2vec_main/configs/audio_style_a20.yaml \
  --ckpt "$CKPT" \
  --osu-dir data/reform/inference_osu \
  --ids-file data/reform/d2v_ids.txt \
  --out data/reform/chart_content_d2v_a20d_anchors.parquet \
  --index-cache data/reform/d2v_anchor_index.json \
  --device cuda --batch-size 32 --num-workers 4 --tag reform_a20d_anchor
