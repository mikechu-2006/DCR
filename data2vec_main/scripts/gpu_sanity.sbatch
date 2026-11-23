#!/bin/bash
#SBATCH -J d2v_sanity
#SBATCH -p debug
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 00:05:00
#SBATCH -o slurm_d2v_sanity_%j.out
#SBATCH -e slurm_d2v_sanity_%j.err
set -euo pipefail
if [ -x /hpc2ssd/softwares/anaconda3/bin/python ]; then PY=/hpc2ssd/softwares/anaconda3/bin/python; else PY=python; fi
echo "host=$(hostname) date=$(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
$PY - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), "cuda", torch.version.cuda)
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    print("matmul", float((x @ x).float().abs().mean()))
PY
