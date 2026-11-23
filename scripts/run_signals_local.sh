#!/bin/bash
# Extract ISOR v3.1 (DCR MSD 505 injected) and Koishi v5.1 signals.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p data/processed/signals_raw logs

echo "[$(date +%H:%M:%S)] ISOR 10K start"
for k in $(seq 0 7); do
  python3 scripts/run_isor_signals.py \
    --charts data/processed/charts_10k.txt \
    --out "data/processed/signals_raw/isor_10k_s${k}.ndjson" \
    --shard "$k/8" > "logs/isor_10k_s${k}.log" 2>&1 &
done
wait
echo "[$(date +%H:%M:%S)] ISOR 10K done"

echo "[$(date +%H:%M:%S)] Koishi 10K start"
for k in $(seq 0 7); do
  node scripts/run_koishi_signals.mjs \
    data/processed/charts_10k_koishi.tsv \
    "data/processed/signals_raw/koishi_10k_s${k}.ndjson" \
    "$k/8" > "logs/koishi_10k_s${k}.log" 2>&1 &
done
wait
echo "[$(date +%H:%M:%S)] Koishi 10K done"

echo "[$(date +%H:%M:%S)] ISOR randommerge start"
for k in $(seq 0 7); do
  python3 scripts/run_isor_signals.py \
    --charts data/processed/charts_randommerge.txt \
    --out "data/processed/signals_raw/isor_rm_s${k}.ndjson" \
    --shard "$k/8" > "logs/isor_rm_s${k}.log" 2>&1 &
done
wait
echo "[$(date +%H:%M:%S)] ISOR randommerge done"
