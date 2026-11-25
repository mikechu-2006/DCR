#!/bin/bash
# Recompute Reform Dan CM3P vectors with the exact legacy loader used by RM-L.
# The trained heads consume the non-windowed BeatmapFilesDataset representation.
#SBATCH --job-name=reform-cm3p-legacy
#SBATCH --partition=long_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/reform_cm3p_legacy_cpu_%j.log

set -euo pipefail
cd "$HOME/DCR"
mkdir -p data/reform results
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-cm3p-legacy-${SLURM_JOB_ID:-local}"
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
  --device cpu \
  --core-audit 0 \
  --out data/reform/chart_content_cm3p_legacy_cpu.parquet
