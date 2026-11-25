#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-phase0-sentinel
#SBATCH -n 1
#SBATCH --mem=1G
#SBATCH -t 00:10:00
#SBATCH -o logs/behavior/phase0_sentinel_%j.out
#SBATCH -e logs/behavior/phase0_sentinel_%j.err
set -euo pipefail
cd "$HOME/DCR"
echo "[$(date +%H:%M:%S)] PHASE0_SENTINEL"
