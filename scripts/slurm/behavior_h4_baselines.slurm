#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-h4-base
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --array=1-5
#SBATCH -o logs/behavior/h4_base_%A_%a.out
#SBATCH -e logs/behavior/h4_base_%A_%a.err
#
# H4 baselines for one split seed: fixed signals (Sunny / MinaCalc / star),
# CM3P + external linear/MLP, frozen D2V linear/MLP.
set -euo pipefail
cd "$HOME/DCR"
S=$SLURM_ARRAY_TASK_ID
OUT=results/behavior/baselines/h4_s$S
mkdir -p "$OUT" logs/behavior data/processed
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
PY=$HOME/.conda/envs/DCR/bin/python
IN=data/processed/step1_full_holdout_s$S.parquet
SIG=data/processed/rank_signals_1k.parquet

echo "[$(date +%H:%M:%S)] host=$(hostname) seed=$S fixed signals"
$PY scripts/step2_rank_eval.py \
  --in "$IN" --signal-table "$SIG" \
  --signals sunny_star,msd_overall,star \
  --split test --ceiling-fit all --strict-pairs \
  --bootstrap 200 \
  --out "$OUT/fixed.json"

echo "[$(date +%H:%M:%S)] seed=$S CM3P+ext MLP"
$PY scripts/step2_cm3p_rank.py \
  --in "$IN" \
  --content-path data/processed/chart_content_cm3p_local_canon.parquet \
  --content-col embedding \
  --signal-table "$SIG" --extra-signals msd_overall,sunny_star \
  --model mlp_extra --lam 1e-2 --hid 64 --depth 2 --epochs 98 \
  --val-frac 0.1 --seed 0 --threads 8 \
  --name cm3p_ext_mlp_h4_s$S \
  --out-signals "$OUT/cm3p_ext_mlp.parquet" \
  --out-json "$OUT/cm3p_ext_mlp.json"

echo "[$(date +%H:%M:%S)] seed=$S CM3P linear"
$PY scripts/step2_cm3p_rank.py \
  --in "$IN" \
  --content-path data/processed/chart_content_cm3p_local_canon.parquet \
  --content-col embedding \
  --model linear --lam 1e-3 --val-frac 0.1 --seed 0 --threads 8 \
  --name cm3p_linear_h4_s$S \
  --out-signals "$OUT/cm3p_linear.parquet" \
  --out-json "$OUT/cm3p_linear.json"

echo "[$(date +%H:%M:%S)] seed=$S D2V linear"
$PY scripts/step2_cm3p_rank.py \
  --in "$IN" \
  --content-path data/processed/chart_content_d2v_holdout.parquet \
  --content-col embedding \
  --model linear --lam 1e-3 --val-frac 0.1 --seed 0 --threads 8 \
  --name d2v_linear_h4_s$S \
  --out-signals "$OUT/d2v_linear.parquet" \
  --out-json "$OUT/d2v_linear.json"

echo "[$(date +%H:%M:%S)] seed=$S D2V MLP"
$PY scripts/step2_cm3p_rank.py \
  --in "$IN" \
  --content-path data/processed/chart_content_d2v_holdout.parquet \
  --content-col embedding \
  --model mlp --lam 1e-2 --hid 64 --depth 2 --epochs 98 \
  --val-frac 0.1 --seed 0 --threads 8 \
  --name d2v_mlp_h4_s$S \
  --out-signals "$OUT/d2v_mlp.parquet" \
  --out-json "$OUT/d2v_mlp.json"

echo "[$(date +%H:%M:%S)] H4_BASE_DONE seed=$S"
