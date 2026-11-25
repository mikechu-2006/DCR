#!/bin/bash
#SBATCH -J a17-probe
#SBATCH -p i64m512u
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o slurm_a17_probe_%j.out
#SBATCH -e slurm_a17_probe_%j.err
set -euo pipefail
cd "$HOME/DCR"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
PY=$HOME/.conda/envs/DCR/bin/python
SIGNALS=data/processed/rank_signals_1k.parquet
SPLIT=data/processed/step1_mclean25_p4sets_cached
echo "[a17-probe] host=$(hostname) date=$(date)"
for spec in "a17d:data/processed/chart_content_d2v_a17d-mclean.parquet:results/d2v_attr_d_cached/a17d/setp4" "a20d_reprobe:data/processed/chart_content_d2v_a20d-mclean.parquet:results/d2v_attr_d_cached/a20d_reprobe/setp4"; do
  ARM=$(printf '%s' "$spec" | cut -d: -f1)
  C=$(printf '%s' "$spec" | cut -d: -f2)
  O=$(printf '%s' "$spec" | cut -d: -f3)
  echo "[a17-probe] $ARM"
  $PY scripts/run_p2_p4_multiseed.py --protocols p4 --seeds 1 \
    --models linear,linear_extra --content-path "$C" --signal-table "$SIGNALS" \
    --split-p4 "$SPLIT" --out-dir "$O" --threads 8
done
echo "[a17-probe] DONE $(date)"
