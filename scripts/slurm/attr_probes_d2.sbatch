#!/bin/bash
#SBATCH -J d2v_attr_dprobe2
#SBATCH -p i64m512u
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o slurm_attr_dprobe2_%j.out
#SBATCH -e slurm_attr_dprobe2_%j.err
set -euo pipefail
cd "$HOME/DCR"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
PY=$HOME/.conda/envs/DCR/bin/python
SIGNALS=data/processed/rank_signals_1k.parquet
SPLIT=data/processed/step1_mclean25_p4sets_cached
echo "[dprobe2] host=$(hostname) date=$(date) split=$SPLIT"
test -f "$SPLIT""_s1.parquet" || { echo "missing split"; exit 3; }
for ARM in a0d a12d a20d a27d; do
  C=$(printf 'data/processed/chart_content_d2v_%s-mclean.parquet' "$ARM")
  O=$(printf 'results/d2v_attr_d_cached/%s/setp4' "$ARM")
  echo "[dprobe2] setp4 $ARM"
  $PY scripts/run_p2_p4_multiseed.py --protocols p4 --seeds 1 \
    --models linear,linear_extra --content-path "$C" --signal-table "$SIGNALS" \
    --split-p4 "$SPLIT" --out-dir "$O" --threads 8
done
echo "[dprobe2] setp4 CM3P legacy reference"
$PY scripts/run_p2_p4_multiseed.py --protocols p4 --seeds 1 \
  --models linear,linear_extra,mlp,mlp_extra \
  --content-path data/processed/chart_content_cm3p_local.parquet --signal-table "$SIGNALS" \
  --split-p4 "$SPLIT" --out-dir results/d2v_attr_d_cached/cm3p_legacy/setp4 --threads 8
echo "[dprobe2] DONE $(date)"
