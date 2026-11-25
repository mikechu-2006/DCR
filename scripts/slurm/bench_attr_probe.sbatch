#!/bin/bash
#SBATCH -J bench-attr-probe
#SBATCH -p i64m512u
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o slurm_bench_attr_probe_%j.out
#SBATCH -e slurm_bench_attr_probe_%j.err
# Per arm: fit linear_extra on randommerge, infer benchmark predictions; then run
# the 5-fold chart-group LOCV isotonic calibration over arms + references.
set -eo pipefail
cd "$HOME/DCR"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
PY=$HOME/.conda/envs/DCR/bin/python
OUT=results/benchmark_attr
PRED=$OUT/pred
MODELS=$OUT/models
mkdir -p "$PRED" "$MODELS"
for ref in RM_L D2V_L M1_L; do
  if [ -f "results/benchmark_ref/$ref.parquet" ]; then
    cp "results/benchmark_ref/$ref.parquet" "$PRED/$ref.parquet"
  fi
done
for ARM in a0d a17d a20d; do
  TAG=$(printf '%s' "$ARM" | tr '[:lower:]' '[:upper:]')
  C="data/processed/chart_content_d2v_$ARM-all.parquet"
  CB="data/processed/benchmark_content_$ARM.parquet"
  echo "[bench-probe] $ARM content=$C"
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
done
echo "[bench-probe] calibration"
$PY scripts/benchmark_calibrate_eval.py \
  --pred-dir "$PRED" \
  --out-dir "$OUT/eval" \
  --runs A0D,A17D,A20D,RM_L,D2V_L,M1_L \
  --bootstrap 1000
echo "[bench-probe] DONE $(date)"
