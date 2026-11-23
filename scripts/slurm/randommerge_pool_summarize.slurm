#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-rm-pool-sum
#SBATCH -o slurm_rm_pool_sum_%j.out
#SBATCH -e slurm_rm_pool_sum_%j.err
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#
# Part B stage 4: summary.csv / summary_mean_std.csv / summary_pairs.csv /
# summary.md from all rank_eval.json + paired_delta.json.
set -euo pipefail
cd "$HOME/DCR"
export PYTHONUNBUFFERED=1
python3 scripts/run_pool_randommerge.py --stage summarize
echo "[$(date +%H:%M:%S)] DONE summarize"
