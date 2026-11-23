#!/bin/bash
#SBATCH -J d2v_main
#SBATCH -p i64m1tga800ue
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o slurm_d2v_main_%j.out
#SBATCH -e slurm_d2v_main_%j.err
set -euo pipefail
cd "$HOME/DCR/data2vec_main"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
PY=$HOME/.conda/envs/DCR/bin/python
echo "[main] host=$(hostname) date=$(date) job=$SLURM_JOB_ID"
$PY - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
test -f cache/chart_index_100.json || $PY -m d2v_main.index_cache --osu-dir data/osu_cache --frame-hz 100
test -f splits/holdout/pretrain_ids.txt || { echo "missing splits/holdout/pretrain_ids.txt"; exit 3; }
$PY -m d2v_main.train --config configs/audio_style_base.yaml \
  --ids-file splits/holdout/pretrain_ids.txt \
  --out-dir runs/main_holdout \
  --device cuda --epochs 20 --batch-size 16 --num-workers 6 --log-every 50
