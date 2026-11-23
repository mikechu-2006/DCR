#!/bin/bash
set -euo pipefail
cd /home/mikechu/Documents/Projects/DCR
mkdir -p results/d2v_main/probe_logs
PY=python3
run() {
  name=$1; inp=$2; content=$3
  out=results/d2v_main/$name
  echo "=== $name ==="
  $PY scripts/step2_cm3p_rank.py --in "$inp" --content-path "$content" \
    --model linear --chart-folds 5 --threads 8 \
    --out-signals "$out"_signals.parquet \
    --out-json "$out"_rank.json 2>&1 | tee "results/d2v_main/probe_logs/$name.log"
}
run p2_all_d2v     data/processed/step1_1k_pairbest.parquet               data/processed/chart_content_d2v_holdout.parquet
run p2_all_cm3p    data/processed/step1_1k_pairbest.parquet               data/processed/chart_content_cm3p_local_canon.parquet
run p2_hold_d2v    data/processed/step1_1k_pairbest_d2v_holdout.parquet   data/processed/chart_content_d2v_holdout.parquet
run p2_hold_cm3p   data/processed/step1_1k_pairbest_d2v_holdout.parquet   data/processed/chart_content_cm3p_local_canon.parquet
echo "ALL_P2_PROBES_DONE"
