#!/bin/bash
#SBATCH -J d2v_a20d_export
#SBATCH -p i64m1tga800ue
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o slurm_d2v_a20d_export_%j.out
#SBATCH -e slurm_d2v_a20d_export_%j.err
#
# Side task: encode the whitelist (21,949 .osu) and the VSRG benchmark (1,382)
# with the Protocol-D a20d checkpoint (+tempo/beat/bar/SV/kiai, 20 channels),
# so a linear_extra head can be fitted/exported locally for packaging.
set -euo pipefail
cd "$HOME/DCR"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$HOME/DCR/data2vec_main"
PY=$HOME/.conda/envs/DCR/bin/python
CKPT=data2vec_main/runs/attr_d/a20d/checkpoints/last.pt
test -f "$CKPT" || { echo "missing ckpt $CKPT"; exit 3; }
test -f data/processed/benchmark_d2v_ids.txt || { echo "missing benchmark ids"; exit 3; }
echo "[a20d-export] host=$(hostname) date=$(date) job=$SLURM_JOB_ID"
echo "[a20d-export] whitelist encode 21,949 charts"
$PY -m d2v_main.extract \
  --config data2vec_main/configs/audio_style_a20.yaml \
  --ckpt "$CKPT" \
  --out data/processed/chart_content_d2v_a20d_full.parquet \
  --device cuda --batch-size 32 --num-workers 6 --tag d2v_a20d_full
echo "[a20d-export] benchmark encode 1,382 charts"
$PY -m d2v_main.extract \
  --config data2vec_main/configs/audio_style_a20.yaml \
  --ckpt "$CKPT" \
  --osu-dir data/benchmark_d2v_osu \
  --ids-file data/processed/benchmark_d2v_ids.txt \
  --out data/processed/benchmark_content_d2v_a20d_raw.parquet \
  --device cuda --batch-size 32 --num-workers 6 --tag benchmark_d2v_a20d
echo "[a20d-export] finalize benchmark table"
$PY scripts/finalize_benchmark_d2v.py \
  --inp data/processed/benchmark_content_d2v_a20d_raw.parquet \
  --out data/processed/benchmark_content_d2v_a20d.parquet
echo "[a20d-export] DONE $(date)"
