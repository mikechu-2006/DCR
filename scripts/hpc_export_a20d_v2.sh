#!/bin/bash
# Robust side-task orchestrator (v2): sync benchmark staging -> submit a20d
# content extraction -> wait -> pull tables. Survives HPC login-node/SSH hiccups
# by retrying every step; writes STATUS_SIDE2.md and exits (wakes the agent).
set -uo pipefail
HPC="aiaa4051_083@hpc2login.hpc.hkust-gz.edu.cn"
ROOT="/home/mikechu/Documents/Projects/DCR"
OUT="$ROOT/results/d2v_a20d_export"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/orchestrator_v2.log") 2>&1
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o ServerAliveInterval=15 -o ServerAliveCountMax=3"
RSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o ServerAliveInterval=15 -o ServerAliveCountMax=3"
RENV='export PATH=/opt/slurm/bin:$PATH; cd $HOME/DCR'
cd "$ROOT"

retry_rsync() {
  local i
  for i in $(seq 1 120); do
    if rsync -az --partial --inplace --timeout=180 -e "$RSH" "$@" >> "$OUT/rsync.log" 2>&1; then
      echo "[rsync] ok attempt $i $(date)"; return 0
    fi
    echo "[rsync] attempt $i failed $(date); sleep 60"
    sleep 60
  done
  return 1
}

echo "=== [1/4] sync small inputs ($(date))"
retry_rsync data/processed/benchmark_d2v_ids.txt data/processed/benchmark_d2v_map.parquet \
  data/processed/benchmark_signals.parquet "$HPC:~/DCR/data/processed/" || { echo FATAL_SYNC_SMALL; exit 1; }
echo "=== [1/4] sync benchmark osu staging ($(date))"
retry_rsync data/benchmark_d2v_osu/ "$HPC:~/DCR/data/benchmark_d2v_osu/" || { echo FATAL_SYNC_STAGING; exit 1; }
retry_rsync data2vec_main/scripts/export_a20d_extract.sbatch "$HPC:~/DCR/data2vec_main/scripts/" || { echo FATAL_SYNC_SBATCH; exit 1; }
retry_rsync scripts/finalize_benchmark_d2v.py "$HPC:~/DCR/scripts/" || { echo FATAL_SYNC_FINALIZE; exit 1; }

echo "=== [2/4] submit extraction job ($(date))"
JID=""
for i in $(seq 1 120); do
  OUT2=$($SSH "$HPC" "$RENV; sbatch data2vec_main/scripts/export_a20d_extract.sbatch" 2>&1)
  JID=$(printf '%s' "$OUT2" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+' | tail -1)
  [ -n "$JID" ] && break
  echo "[submit] attempt $i: $OUT2"
  sleep 60
done
if [ -z "$JID" ]; then
  printf '# a20d export blocked\n\nsubmit failed at %s\n' "$(date)" > "$OUT/STATUS_SIDE2.md"
  echo FATAL_SUBMIT; exit 1
fi
echo "[a20d-export] job=$JID" | tee "$OUT/job2.txt"

echo "=== [3/4] waiting ($(date))"
EMPTY=0
while true; do
  RAW=$($SSH "$HPC" "$RENV; echo BEGIN; squeue -h -j $JID -o '%i|%T|%r' 2>/dev/null; echo END" 2>/dev/null)
  if ! printf '%s' "$RAW" | grep -q 'END'; then echo "[wait] ssh hiccup $(date); sleep 90"; sleep 90; continue; fi
  BODY=$(printf '%s\n' "$RAW" | sed -n '/BEGIN/,/END/p' | grep '|')
  ACTIVE=$(printf '%s\n' "$BODY" | awk -F'|' 'NF>=2 { if ($2=="RUNNING"||$2=="CONFIGURING"||$2=="COMPLETING") n++; else if ($2=="PENDING" && $3 !~ /DependencyNeverSatisfied/) n++ } END {print n+0}')
  if [ "$ACTIVE" -eq 0 ]; then EMPTY=$((EMPTY+1)); else EMPTY=0; fi
  echo "[wait] $(date) active=$ACTIVE streak=$EMPTY"
  [ "$EMPTY" -ge 2 ] && break
  sleep 90
done

echo "=== [4/4] collecting ($(date))"
$SSH "$HPC" "$RENV; sacct -n -P -j $JID --format=JobID%12,JobName%20,State,ExitCode,Elapsed 2>/dev/null" > "$OUT/sacct2.txt" || true
cat "$OUT/sacct2.txt"
retry_rsync "$HPC:~/DCR/data/processed/chart_content_d2v_a20d_full.parquet" "$ROOT/data/processed/" || echo PULL_FULL_FAIL
retry_rsync "$HPC:~/DCR/data/processed/benchmark_content_d2v_a20d.parquet" "$ROOT/data/processed/" || echo PULL_BENCH_FAIL
retry_rsync --include 'slurm_d2v_a20d_export_*' --exclude '*' "$HPC:~/DCR/" "$OUT/logs/" || true
{
  echo "# a20d content export for packaging — $(date)"
  echo; echo "job: $JID"; echo; echo "## sacct"; echo '~~~'; cat "$OUT/sacct2.txt"; echo '~~~'
  echo; echo "## local tables"
  ls -lh "$ROOT/data/processed/chart_content_d2v_a20d_full.parquet" \
         "$ROOT/data/processed/benchmark_content_d2v_a20d.parquet" 2>/dev/null
} > "$OUT/STATUS_SIDE2.md"
echo "=== DONE $(date)"
