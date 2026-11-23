#!/bin/bash
#SBATCH -J d2v_bench
#SBATCH -p debug
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 00:25:00
#SBATCH -o slurm_d2v_bench_%j.out
#SBATCH -e slurm_d2v_bench_%j.err
set -euo pipefail
cd "$HOME/DCR/data2vec_main"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
PY=$HOME/.conda/envs/DCR/bin/python
export PYTHONNOUSERSITE=1
echo "[bench] host=$(hostname) date=$(date) py=$PY"
$PY - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu", torch.cuda.get_device_name(0), round(p.total_memory / 1e9, 1), "GB")
PY
$PY -m d2v_main.train --config configs/audio_style_base.yaml \
  --out-dir runs/bench_hpc_b16 --max-steps 120 --epochs 1 \
  --batch-size 16 --num-workers 6 --device cuda --log-every 10
