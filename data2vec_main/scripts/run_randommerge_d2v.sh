#!/bin/bash
set -euo pipefail
cd /home/mikechu/Documents/Projects/DCR
mkdir -p results/d2v_main/randommerge_logs
python3 scripts/run_p2_p4_multiseed.py --protocols p2,p4 --seeds 1,2,3,4,5 \
  --models linear,linear_extra \
  --content-path data/processed/chart_content_d2v_holdout.parquet \
  --signal-table data/processed/rank_signals_randommerge_od.parquet \
  --split-p2 data/processed/step1_randommerge_c_p2 \
  --split-p4 data/processed/step1_randommerge_c_p4 \
  --out-dir results/d2v_main/randommerge_d2v --threads 6 \
  2>&1 | tee results/d2v_main/randommerge_logs/d2v.log
echo "RANDOMMERGE_D2V_DONE"
