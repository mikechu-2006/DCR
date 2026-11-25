#!/bin/bash
# Short GPU run using the exact legacy CM3P representation consumed by RM-L.
#SBATCH --job-name=reform-cm3p-legacy-gpu
#SBATCH --partition=emergency_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=results/reform_cm3p_legacy_gpu_%j.log

set -euo pipefail
cd "$HOME/DCR"
mkdir -p data/reform results
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-cm3p-legacy-gpu-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"

"$HOME/.conda/envs/DCR/bin/python" scripts/step0c_chart_content.py \
  --source osu-dir \
  --ids-file data/reform/cm3p_ids.csv \
  --osu-dir data/reform/inference_osu \
  --model-repo .cache/cm3p_model_26cc694 \
  --crop legacy \
  --mania-columns off \
  --windows off \
  --infer-batch 8 \
  --audio off \
  --device cuda \
  --core-audit 0 \
  --out data/reform/chart_content_cm3p_legacy_gpu.parquet
