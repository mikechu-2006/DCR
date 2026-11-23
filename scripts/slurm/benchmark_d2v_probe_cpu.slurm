#!/bin/bash
#SBATCH -p emergency_cpu
#SBATCH -J dcr-d2v-probe
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o slurm_d2v_probe_%j.out
#SBATCH -e slurm_d2v_probe_%j.err
set -euo pipefail
cd "$HOME/DCR"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
PY=$HOME/.conda/envs/DCR/bin/python
export PYTHONPATH="$HOME/DCR/data2vec_main"
echo "[$(date +%H:%M:%S)] probe start host=$(hostname)"
/usr/bin/time -v $PY -m d2v_main.extract \
  --config data2vec_main/configs/audio_style_base.yaml \
  --ckpt data2vec_main/runs/main_holdout/checkpoints/last.pt \
  --osu-dir data/benchmark_d2v_osu \
  --ids-file data/processed/benchmark_d2v_probe_ids.txt \
  --out /tmp/bench_d2v_probe.parquet \
  --device cpu --batch-size 8 --num-workers 2 --tag probe_d2v 2>&1 | tail -40
echo "[$(date +%H:%M:%S)] PROBE_DONE"
