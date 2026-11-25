#!/bin/bash
#SBATCH -J bench-attr-fit
#SBATCH -p debug
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 00:28:00
#SBATCH -o slurm_bench_attr_fit_%j.out
#SBATCH -e slurm_bench_attr_fit_%j.err
# One arm: randommerge linear_extra fit + benchmark inference (debug CPU).
set -eo pipefail
cd "$HOME/DCR"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
PY=$HOME/.conda/envs/DCR/bin/python
if [ -z "$ARM" ]; then echo "set ARM=a0d|a17d|a20d"; exit 2; fi
OUT=results/benchmark_attr
PRED=$OUT/pred
MODELS=$OUT/models
mkdir -p "$PRED" "$MODELS" "$OUT/train/$ARM"
for ref in RM_L D2V_L M1_L; do
  if [ -f "results/benchmark_ref/$ref.parquet" ]; then
    cp "results/benchmark_ref/$ref.parquet" "$PRED/$ref.parquet"
  fi
done
TAG=$(printf '%s' "$ARM" | tr '[:lower:]' '[:upper:]')
C="data/processed/chart_content_d2v_$ARM-all.parquet"
CB="data/processed/benchmark_content_$ARM.parquet"
echo "[bench-fit] arm=$ARM host=$(hostname) date=$(date)"
test -f "$C" || { echo "missing $C"; exit 3; }
test -f "$CB" || { echo "missing $CB"; exit 3; }
$PY scripts/step2_cm3p_rank.py \
  --in data/processed/step1_randommerge_pairbest_content.parquet \
  --content-path "$C" \
  --signal-table data/processed/rank_signals_1k.parquet \
  --extra-signals msd_overall,sunny_star \
  --model linear_extra --lam 1e-3 --val-frac 0 --seed 0 --threads 8 \
  --name cm3p_linear_extra --fit-all \
  --save-model "$MODELS/$TAG-full.pt" \
  --out-signals "$OUT/train/$ARM/signals.parquet" \
  --out-json "$OUT/train/$ARM/train.json"
$PY scripts/benchmark_infer.py \
  --model "$MODELS/$TAG-full.pt" \
  --content "$CB" \
  --signals data/processed/benchmark_signals.parquet \
  --out "$PRED/$TAG.parquet"
echo "[bench-fit] DONE arm=$ARM $(date)"
