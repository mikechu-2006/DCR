#!/bin/bash
# Wait for the benchmark-attribute probe+calibration CPU job, pull results,
# and write results/benchmark_attr/STATUS_BENCH.md. Run in the background.
set -uo pipefail
HOST="aiaa3111_l01_066@hpc2login.hpc.hkust-gz.edu.cn"
ROOT="/home/mikechu/Documents/Projects/DCR"
OUT="$ROOT/results/benchmark_attr"
JID="14130198"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/bench_watch.log") 2>&1
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25"
RENV='export PATH=/opt/slurm/bin:$PATH; cd $HOME/DCR'
START=$(date +%s)
echo "=== watching bench-attr-probe job $JID ($(date))"
EMPTY=0
while true; do
  NOW=$(date +%s)
  if [ $((NOW - START)) -gt 10800 ]; then echo "=== timeout 3h"; break; fi
  RAW=$($SSH "$HOST" "$RENV; echo BEGIN; squeue -h -j $JID -o '%i|%T|%r' 2>/dev/null; echo END" 2>/dev/null)
  if ! printf '%s' "$RAW" | grep -q 'END'; then echo "[wait] ssh hiccup $(date)"; sleep 90; continue; fi
  BODY=$(printf '%s\n' "$RAW" | sed -n '/BEGIN/,/END/p' | grep '|')
  ACTIVE=$(printf '%s\n' "$BODY" | awk -F'|' 'NF>=2 { if ($2=="RUNNING"||$2=="CONFIGURING"||$2=="COMPLETING") n++; else if ($2=="PENDING" && $3 !~ /DependencyNeverSatisfied/) n++ } END {print n+0}')
  if [ "$ACTIVE" -eq 0 ]; then EMPTY=$((EMPTY+1)); else EMPTY=0; fi
  echo "[wait] $(date) active=$ACTIVE streak=$EMPTY"
  [ "$EMPTY" -ge 2 ] && break
  sleep 90
done
echo "=== collecting ($(date))"
$SSH "$HOST" "$RENV; sacct -n -P -j $JID --format=JobID%12,JobName%20,State,ExitCode,Elapsed 2>/dev/null" > "$OUT/sacct_probe.txt" || true
cat "$OUT/sacct_probe.txt"
rsync -az "$HOST:~/DCR/results/benchmark_attr/" "$OUT/" 2>/dev/null || true
rsync -az --include 'slurm_bench_attr_*' --exclude '*' "$HOST:~/DCR/" "$OUT/logs/" 2>/dev/null || true
{
  echo "# VSRG benchmark ablation: a0d vs a17d vs a20d (isotonic 5-fold) — $(date)"
  echo
  echo "job: $JID"; echo
  echo "## sacct"; echo '~~~'; cat "$OUT/sacct_probe.txt"; echo '~~~'
  echo; echo "## output files"
  find "$OUT" -maxdepth 3 -name '*.csv' -o -maxdepth 3 -name '*.json' -o -maxdepth 3 -name '*.md' 2>/dev/null | sort | head -40
} > "$OUT/STATUS_BENCH.md"
echo "=== DONE $(date)"
