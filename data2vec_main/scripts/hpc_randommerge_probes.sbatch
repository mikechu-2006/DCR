#!/bin/bash
#SBATCH -J d2v_rm_cpu
#SBATCH -p i64m512u
#SBATCH -n 16
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o slurm_d2v_rm_cpu_%j.out
#SBATCH -e slurm_d2v_rm_cpu_%j.err
set -euo pipefail
cd "$HOME/DCR"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
PY=$HOME/.conda/envs/DCR/bin/python
$PY scripts/run_p2_p4_multiseed.py --protocols p2,p4 --seeds 1,2,3,4,5 \
  --models linear,linear_extra \
  --content-path data/processed/chart_content_d2v_holdout.parquet \
  --signal-table data/processed/rank_signals_randommerge_od.parquet \
  --split-p2 data/processed/step1_randommerge_c_p2 \
  --split-p4 data/processed/step1_randommerge_c_p4 \
  --out-dir results/d2v_main/randommerge_d2v_hpc --threads 16
echo "RANDOMMERGE_HPC_DONE"
