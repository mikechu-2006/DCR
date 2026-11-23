#!/bin/bash
#SBATCH -p emergency_gpu
#SBATCH -J dcr-bench-d2v-enc
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o slurm_bench_d2v_enc_%j.out
#SBATCH -e slurm_bench_d2v_enc_%j.err
#
# Encode the 1,382 VSRG benchmark charts with the D2V holdout encoder.
# Staging dir data/benchmark_d2v_osu/{0..1381}.osu; mapping in
# data/processed/benchmark_d2v_map.parquet.
set -euo pipefail
cd "$HOME/DCR"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
PY=$HOME/.conda/envs/DCR/bin/python
export PYTHONPATH="$HOME/DCR/data2vec_main"
echo "[$(date +%H:%M:%S)] host=$(hostname) d2v benchmark encode start"
$PY -m d2v_main.extract \
  --config data2vec_main/configs/audio_style_base.yaml \
  --ckpt data2vec_main/runs/main_holdout/checkpoints/last.pt \
  --osu-dir data/benchmark_d2v_osu \
  --ids-file data/processed/benchmark_d2v_ids.txt \
  --out data/processed/benchmark_content_d2v.parquet \
  --device cuda --batch-size 32 --num-workers 6 --tag benchmark_d2v
echo "[$(date +%H:%M:%S)] D2V_BENCHMARK_ENCODE_DONE"
