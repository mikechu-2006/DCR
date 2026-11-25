#!/bin/bash
# Side task orchestrator: sync benchmark staging to HPC, submit the a20d content
# extraction job, wait, pull the two content tables back, write STATUS_SIDE.md.
set -uo pipefail
HPC="aiaa4051_083@hpc2login.hpc.hkust-gz.edu.cn"
ROOT="/home/mikechu/Documents/Projects/DCR"
OUT="$ROOT/results/d2v_a20d_export"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/orchestrator.log") 2>&1
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25"
RENV='export PATH=/opt/slurm/bin:$PATH; cd $HOME/DCR'
cd "$ROOT"

echo "=== [1/4] syncing side-task inputs to HPC ($(date))"
rsync -az data/processed/benchmark_d2v_ids.txt data/processed/benchmark_d2v_map.parquet \
  data/processed/benchmark_signals.parquet "$HPC:~/DCR/data/processed/" || exit 1
rsync -az data/benchmark_d2v_osu/ "$HPC:~/DCR/data/benchmark_d2v_osu/" || exit 1
rsync -az data2vec_main/scripts/export_a20d_extract.sbatch "$HPC:~/DCR/data2vec_main/scripts/" || exit 1
rsync -az scripts/finalize_benchmark_d2v.py "$HPC:~/DCR/scripts/" || exit 1

echo "=== [2/4] submitting extraction job ($(date))"
JID=$($SSH "$HPC" "$RENV; sbatch data2vec_main/scripts/export_a20d_extract.sbatch" 2>&1 | grep -oE '[0-9]+' | tail -1)
if [ -z "$JID" ]; then
  echo "=== FATAL: submission failed (quota or scheduler)"
  printf '# a20d export blocked\n\nsubmit failed at %s\n' "$(date)" > "$OUT/STATUS_SIDE.md"
  exit 1
fi
echo "[a20d-export] job=$JID" | tee "$OUT/job.txt"

echo "=== [3/4] waiting ($(date))"
EMPTY=0
while true; do
  RAW=$($SSH "$HPC" "$RENV; echo BEGIN; squeue -h -j $JID -o '%i|%T|%r' 2>/dev/null; echo END" 2>/dev/null)
  if ! printf '%s' "$RAW" | grep -q 'END'; then echo "[wait] ssh hiccup $(date)"; sleep 90; continue; fi
  BODY=$(printf '%s\n' "$RAW" | sed -n '/BEGIN/,/END/p' | grep '|')
  ACTIVE=$(printf '%s\n' "$BODY" | awk -F'|' 'NF>=2 { if ($2=="RUNNING"||$2=="CONFIGURING"||$2=="COMPLETING") n++; else if ($2=="PENDING" && $3 !~ /DependencyNeverSatisfied/) n++ } END {print n+0}')
  if [ "$ACTIVE" -eq 0 ]; then EMPTY=$((EMPTY+1)); else EMPTY=0; fi
  echo "[wait] $(date) active=$ACTIVE streak=$EMPTY"
  [ "$EMPTY" -ge 2 ] && break
  sleep 90
done

echo "=== [4/4] collecting ($(date))"
$SSH "$HPC" "$RENV; sacct -n -P -j $JID --format=JobID%12,JobName%20,State,ExitCode,Elapsed 2>/dev/null" > "$OUT/sacct.txt" || true
cat "$OUT/sacct.txt"
rsync -az "$HPC:~/DCR/data/processed/chart_content_d2v_a20d_full.parquet" "$ROOT/data/processed/" 2>/dev/null || true
rsync -az "$HPC:~/DCR/data/processed/benchmark_content_d2v_a20d.parquet" "$ROOT/data/processed/" 2>/dev/null || true
rsync -az "$HPC:~/DCR/data/processed/benchmark_content_d2v_a20d_raw.parquet" "$ROOT/data/processed/" 2>/dev/null || true
rsync -az --include 'slurm_d2v_a20d_export_*' --exclude '*' "$HPC:~/DCR/" "$OUT/logs/" 2>/dev/null || true
{
  echo "# a20d content export for packaging — $(date)"
  echo; echo "job: $JID"; echo; echo "## sacct"; echo '~~~'; cat "$OUT/sacct.txt"; echo '~~~'
  echo; echo "## local tables"
  ls -lh "$ROOT/data/processed/chart_content_d2v_a20d_full.parquet" \
         "$ROOT/data/processed/benchmark_content_d2v_a20d.parquet" 2>/dev/null
} > "$OUT/STATUS_SIDE.md"
echo "=== DONE $(date)"
