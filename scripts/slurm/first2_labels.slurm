#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-first2-rm
#SBATCH -o slurm_first2_rm_%j.out
#SBATCH -e slurm_first2_rm_%j.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# Experiment 1, Stage 0b: best-of-first-two labels for randommerge.
# Inputs are the canonical step0 merge and the content-filtered pairbest table.
# The script is self-contained; it does not import the step1 module.
set -euo pipefail
cd "$HOME/DCR"
PY="$HOME/.conda/envs/DCR/bin/python"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
echo "[$(date +%H:%M:%S)] host=$(hostname) job=${SLURM_JOB_ID:-?}"
"$PY" scripts/step1_first2_labels.py \
  --step0 data/processed/step0_randommerge.parquet \
  --pairs data/processed/step1_randommerge_pairbest_content.parquet \
  --out data/processed/step1_randommerge_first2.parquet \
  --json data/processed/step1_randommerge_first2.json
echo "[$(date +%H:%M:%S)] DONE"
