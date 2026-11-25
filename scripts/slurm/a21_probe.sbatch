#!/bin/bash
#SBATCH -J a21-probe
#SBATCH -p i64m512u
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o slurm_a21_probe_%j.out
#SBATCH -e slurm_a21_probe_%j.err
set -eo pipefail
cd "$HOME/DCR"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
PY=$HOME/.conda/envs/DCR/bin/python
SIGNALS=data/processed/rank_signals_1k.parquet
SPLIT=data/processed/step1_mclean25_p4sets_cached
echo "[a21-probe] host=$(hostname) date=$(date)"
for spec in "a21d:data/processed/chart_content_d2v_a21d-mclean.parquet:results/d2v_attr_d_cached/a21d/setp4" "a20d_reprobe:data/processed/chart_content_d2v_a20d-mclean.parquet:results/d2v_attr_d_cached/a20d_reprobe/setp4"; do
  ARM=$(printf '%s' "$spec" | cut -d: -f1)
  C=$(printf '%s' "$spec" | cut -d: -f2)
  O=$(printf '%s' "$spec" | cut -d: -f3)
  echo "[a21-probe] $ARM"
  $PY scripts/run_p2_p4_multiseed.py --protocols p4 --seeds 1 \
    --models linear,linear_extra --content-path "$C" --signal-table "$SIGNALS" \
    --split-p4 "$SPLIT" --out-dir "$O" --threads 8
done
echo "[a21-probe] DONE $(date)"
