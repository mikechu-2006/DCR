#!/bin/bash
set -euo pipefail
cd /home/mikechu/Documents/Projects/DCR
mkdir -p results/d2v_main/mlp_logs
run() {
  tag=$1; proto=$2; content=$3
  echo "=== $tag $proto ==="
  python3 scripts/run_p2_p4_multiseed.py --protocols "$proto" --seeds 1,2,3,4,5 \
    --models mlp,mlp_extra --content-path "$content" \
    --out-dir "results/d2v_main/${proto}_$tag" --threads 8 \
    2>&1 | tee "results/d2v_main/mlp_logs/$tag.log"
}
run d2v_mlp p4 data/processed/chart_content_d2v_holdout.parquet
run cm3p_legacy_mlp p4 data/processed/chart_content_cm3p_local.parquet
run d2v_mlp p2 data/processed/chart_content_d2v_holdout.parquet
run cm3p_legacy_mlp p2 data/processed/chart_content_cm3p_local.parquet
echo "ALL_MLP_DONE"
