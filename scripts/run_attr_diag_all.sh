#!/bin/bash
# C3 (mapper-ID probe) + C4 (attribute decodability) for every pilot content table.
# Local CPU, a few minutes. Usage: bash scripts/run_attr_diag_all.sh
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/d2v_attr_diag
mkdir -p "$OUT"
for arm in a0cont a12 a20 a27; do
  C=$(printf 'data/processed/chart_content_d2v_%s-1k.parquet' "$arm")
  M=$(printf '%s/mapper_probe_%s.json' "$OUT" "$arm")
  A=$(printf '%s/attr_decodability_%s.json' "$OUT" "$arm")
  L=$(printf '%s/logs_%s.log' "$OUT" "$arm")
  [ -f "$C" ] || { echo "[diag] skip $arm (no content)"; continue; }
  python3 scripts/content_mapper_probe.py --content-path "$C" --mode both \
    --tag "$arm" --mapper-out "$M" --attr-out "$A" > "$L" 2>&1 \
    && echo "[diag] $arm OK" || echo "[diag] $arm FAILED (see $L)"
done
for arm in a0d a12d a20d a27d; do
  C=$(printf 'data/processed/chart_content_d2v_%s-mclean.parquet' "$arm")
  M=$(printf '%s/mapper_probe_%s.json' "$OUT" "$arm")
  A=$(printf '%s/attr_decodability_%s.json' "$OUT" "$arm")
  L=$(printf '%s/logs_%s.log' "$OUT" "$arm")
  [ -f "$C" ] || { echo "[diag] skip $arm (no content)"; continue; }
  python3 scripts/content_mapper_probe.py --content-path "$C" --mode both \
    --tag "$arm" --mapper-out "$M" --attr-out "$A" > "$L" 2>&1 \
    && echo "[diag] $arm OK" || echo "[diag] $arm FAILED (see $L)"
done
python3 - <<'PY'
import json, glob, os
rows = {}
for p in sorted(glob.glob('results/d2v_attr_diag/attr_decodability_*.json')):
    arm = os.path.basename(p)[len('attr_decodability_'):-5]
    if arm in ('d2v8', 'meta12'):
        continue
    d = json.load(open(p))
    rows.setdefault(arm, {})['c4'] = {k: round(v.get('r2_test', float('nan')), 3)
                                     for k, v in d.get('targets', {}).items()}
for p in sorted(glob.glob('results/d2v_attr_diag/mapper_probe_*.json')):
    arm = os.path.basename(p)[len('mapper_probe_'):-5]
    if arm in ('d2v8', 'meta12'):
        continue
    d = json.load(open(p))
    rows.setdefault(arm, {})['c3_top1'] = round(d.get('top1_accuracy_test', float('nan')), 4)
    rows.setdefault(arm, {})['c3_majority'] = round(d.get('majority_baseline_test_accuracy', float('nan')), 4)
out = 'results/d2v_attr_diag/summary_arms.json'
json.dump(rows, open(out, 'w'), indent=1)
print(json.dumps(rows, indent=1))
PY
