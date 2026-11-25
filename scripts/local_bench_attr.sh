#!/bin/bash
# Local (8-core) benchmark ablation probe+calibration for a0d/a17d/a20d.
# The HPC CPU jobs kept being cancelled externally; this is a short job.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/benchmark_attr
PRED=$OUT/pred
MODELS=$OUT/models
mkdir -p "$PRED" "$MODELS" "$OUT/train/a0d" "$OUT/train/a17d" "$OUT/train/a20d"
for ref in RM_L D2V_L M1_L; do
  if [ -f "results/benchmark/pred/$ref.parquet" ]; then cp "results/benchmark/pred/$ref.parquet" "$PRED/$ref.parquet"; fi
done
PY=python3
for ARM in a0d a17d a20d; do
  TAG=$(printf '%s' "$ARM" | tr '[:lower:]' '[:upper:]')
  C="data/processed/chart_content_d2v_$ARM-all.parquet"
  CB="data/processed/benchmark_content_$ARM.parquet"
  echo "[local-bench] fit $ARM ($(date))"
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
  echo "[local-bench] infer $ARM ($(date))"
  $PY scripts/benchmark_infer.py \
    --model "$MODELS/$TAG-full.pt" \
    --content "$CB" \
    --signals data/processed/benchmark_signals.parquet \
    --out "$PRED/$TAG.parquet"
done
echo "[local-bench] calibrate ($(date))"
$PY scripts/benchmark_calibrate_eval.py \
  --pred-dir "$PRED" \
  --out-dir "$OUT/eval" \
  --runs A0D,A17D,A20D,RM_L,D2V_L,M1_L \
  --bootstrap 1000
{
  echo "# VSRG benchmark ablation (local run) — $(date)"
  echo
  echo "## eval files"
  find "$OUT/eval" -maxdepth 2 -type f 2>/dev/null | sort
  echo
  for f in "$OUT"/eval/summary.csv "$OUT"/eval/summary.json "$OUT"/eval/metrics.json; do
    if [ -f "$f" ]; then echo "## $(basename "$f")"; echo '~~~'; cat "$f"; echo '~~~'; fi
  done
} > "$OUT/STATUS_BENCH_LOCAL.md"
echo "[local-bench] DONE $(date)"
