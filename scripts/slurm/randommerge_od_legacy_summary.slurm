#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-od-legacy-sum
#SBATCH -o slurm_rm_od_sum_%j.out
#SBATCH -e slurm_rm_od_sum_%j.err
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
cd "$HOME/DCR"
export PYTHONUNBUFFERED=1
python3 scripts/run_od_legacy.py --stage summarize
echo "[$(date +%H:%M:%S)] DONE summarize"
