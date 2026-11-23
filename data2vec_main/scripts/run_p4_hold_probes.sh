#!/bin/bash
set -euo pipefail
cd /home/mikechu/Documents/Projects/DCR
mkdir -p results/d2v_main/p4_hold_logs
run() {
  tag=$1; content=$2
  echo "=== P4-holdout $tag ==="
  python3 scripts/run_p2_p4_multiseed.py --protocols p4 --seeds 1,2,3,4,5 \
    --models linear,linear_extra \
    --content-path "$content" \
    --split-p4 data/processed/step1_1k_pairbest_p4_d2vhold \
    --out-dir "results/d2v_main/p4_hold_$tag" --threads 8 \
    2>&1 | tee "results/d2v_main/p4_hold_logs/$tag.log"
}
run d2v  data/processed/chart_content_d2v_holdout.parquet
run cm3p data/processed/chart_content_cm3p_local_canon.parquet
echo "ALL_P4_HOLD_DONE"
