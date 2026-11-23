#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-rm-pool-shared
#SBATCH -o slurm_rm_pool_shared_%j.out
#SBATCH -e slurm_rm_pool_shared_%j.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#
# Part B stage 0: label-free A0/A0L/A1 features on the canon window store.
# Everything downstream depends on this job.
set -euo pipefail
cd "$HOME/DCR"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
python3 scripts/run_pool_randommerge.py --stage shared --threads 8
echo "[$(date +%H:%M:%S)] DONE shared"
