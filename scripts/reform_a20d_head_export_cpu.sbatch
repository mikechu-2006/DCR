#!/bin/bash
# Export the a20d product linear head and its VSRG benchmark package after the
# product encoder has built the full and benchmark content tables.
#SBATCH --job-name=reform-a20d-head
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=results/reform_a20d_head_export_%j.log

set -euo pipefail
cd "$HOME/DCR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

"$HOME/.conda/envs/DCR/bin/python" scripts/export_a20d_head.py \
  --key D2V_A20D_P
