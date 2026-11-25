#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-e1-sentinel
#SBATCH -n 1
#SBATCH --mem=1G
#SBATCH -t 00:10:00
#SBATCH -o logs/behavior/e1_sentinel_%j.out
#SBATCH -e logs/behavior/e1_sentinel_%j.err
set -euo pipefail
cd "$HOME/DCR"
echo "[$(date +%H:%M:%S)] E1_SENTINEL"
