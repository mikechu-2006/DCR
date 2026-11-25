#!/bin/bash
# Run the a17/a19/a20 set-P4 probe trio on the LOCAL CPU (HPC CPU queue was
# stuck at Priority). Content tables were produced on the HPC A800 jobs; only
# the rank-NLL probe is run locally.
set -uo pipefail
ROOT="/home/mikechu/Documents/Projects/DCR"
OUT="$ROOT/results/d2v_attr_run"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/a19_local.log") 2>&1
cd "$ROOT"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
echo "=== local probe trio start $(date)"
SIGNALS=data/processed/rank_signals_1k.parquet
SPLIT=data/processed/step1_mclean25_p4sets_cached
for spec in \
  "a17d_rerun:data/processed/chart_content_d2v_a17d-mclean.parquet:results/d2v_attr_d_cached/a17d_rerun/setp4" \
  "a19d:data/processed/chart_content_d2v_a19d-mclean.parquet:results/d2v_attr_d_cached/a19d/setp4" \
  "a20d_rerun:data/processed/chart_content_d2v_a20d-mclean.parquet:results/d2v_attr_d_cached/a20d_rerun/setp4"; do
  ARM=$(printf '%s' "$spec" | cut -d: -f1)
  C=$(printf '%s' "$spec" | cut -d: -f2)
  O=$(printf '%s' "$spec" | cut -d: -f3)
  echo "[local-probe] $ARM $(date)"
  python3 scripts/run_p2_p4_multiseed.py --protocols p4 --seeds 1 \
    --models linear,linear_extra --content-path "$C" --signal-table "$SIGNALS" \
    --split-p4 "$SPLIT" --out-dir "$O" --threads 8 || echo "[local-probe] FAILED $ARM"
done
echo "=== C3/C4 diagnostics a19d $(date)"
python3 scripts/content_mapper_probe.py --content-path data/processed/chart_content_d2v_a19d-mclean.parquet \
  --mode both --tag a19d --mapper-out results/d2v_attr_diag/mapper_probe_a19d.json \
  --attr-out results/d2v_attr_diag/attr_decodability_a19d.json 2>&1 | tail -20
echo "=== C3/C4 diagnostics a17d_rerun (sanity) $(date)"
python3 scripts/content_mapper_probe.py --content-path data/processed/chart_content_d2v_a17d-mclean.parquet \
  --mode both --tag a17d --mapper-out results/d2v_attr_diag/mapper_probe_a17d.json \
  --attr-out results/d2v_attr_diag/attr_decodability_a17d.json 2>&1 | tail -14
{
  echo "# a19 ablation — local CPU probe fallback ($(date))"
  echo
  echo "Content tables: a17d/a20d produced earlier on HPC; a19d produced on HPC job 14128340/14128341 (train/extract COMPLETED)."
  echo "Probe (linear/linear_extra, set-P4 seed 1) run locally because the HPC CPU probe job 14128342 was stuck at Priority and was cancelled."
  echo
  echo "## summaries"
  find "$ROOT/results/d2v_attr_d_cached" \( -path '*a17d_rerun*' -o -path '*a19d*' -o -path '*a20d_rerun*' \) -name summary.csv 2>/dev/null | sort
} > "$OUT/STATUS_A19_LOCAL.md"
echo "=== local probe trio DONE $(date)"
