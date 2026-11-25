#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-p4sets-split
#SBATCH -o slurm_p4sets_split_%j.out
#SBATCH -e slurm_p4sets_split_%j.err
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
cd "$HOME/DCR"
PY="$HOME/.conda/envs/DCR/bin/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
"$PY" scripts/step1_p4_split_sets.py \
  --in data/processed/step1_1k_pairbest_nogap.parquet \
  --meta data/processed/beatmap_meta_1k.csv \
  --out-prefix data/processed/step1_1k_pairbest_p4sets \
  --seeds 1,2,3,4,5 --force
