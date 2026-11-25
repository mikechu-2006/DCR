#!/bin/bash
# Sync the cached-filtered set-P4 split, submit the D-stage probe rerun, wait,
# pull results and write STATUS4.md. Run in the background.
set -uo pipefail
HPC="aiaa4051_083@hpc2login.hpc.hkust-gz.edu.cn"
ROOT="/home/mikechu/Documents/Projects/DCR"
OUT="$ROOT/results/d2v_attr_run"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/dprobe2.log") 2>&1
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25"
RENV='export PATH=/opt/slurm/bin:$PATH; cd $HOME/DCR'
cd "$ROOT"

echo "=== syncing split + filtered table + sbatch ($(date))"
rsync -az data/processed/step1_mclean25_pairbest_cached.parquet \
  data/processed/step1_mclean25_p4sets_cached_s1.parquet \
  data/processed/step1_mclean25_p4sets_cached_summary.json \
  "$HPC:~/DCR/data/processed/" || exit 1
rsync -az scripts/slurm/attr_probes_d2.sbatch "$HPC:~/DCR/scripts/slurm/" || exit 1

echo "=== submitting D probe rerun ($(date))"
JID=$($SSH "$HPC" "$RENV; sbatch scripts/slurm/attr_probes_d2.sbatch" 2>&1 | grep -oE '[0-9]+' | tail -1)
if [ -z "$JID" ]; then echo "=== FATAL submit failed"; exit 1; fi
echo "[dprobe2] job=$JID" | tee "$OUT/job_dprobe2.txt"

echo "=== waiting ($(date))"
EMPTY=0
while true; do
  RAW=$($SSH "$HPC" "$RENV; echo BEGIN; squeue -h -j $JID -o '%i|%T|%r' 2>/dev/null; echo END" 2>/dev/null)
  if ! printf '%s' "$RAW" | grep -q 'END'; then echo "[wait] ssh hiccup"; sleep 90; continue; fi
  BODY=$(printf '%s\n' "$RAW" | sed -n '/BEGIN/,/END/p' | grep '|')
  ACTIVE=$(printf '%s\n' "$BODY" | awk -F'|' 'NF>=2 { if ($2=="RUNNING"||$2=="CONFIGURING"||$2=="COMPLETING") n++; else if ($2=="PENDING" && $3 !~ /DependencyNeverSatisfied/) n++ } END {print n+0}')
  if [ "$ACTIVE" -eq 0 ]; then EMPTY=$((EMPTY+1)); else EMPTY=0; fi
  echo "[wait] $(date) active=$ACTIVE streak=$EMPTY"
  [ "$EMPTY" -ge 2 ] && break
  sleep 90
done

echo "=== collecting ($(date))"
$SSH "$HPC" "$RENV; sacct -n -P -j $JID --format=JobID%12,JobName%20,State,ExitCode,Elapsed 2>/dev/null" > "$OUT/sacct_dprobe2.txt" || true
cat "$OUT/sacct_dprobe2.txt"
mkdir -p "$ROOT/results/d2v_attr_d_cached"
rsync -az "$HPC:~/DCR/results/d2v_attr_d_cached/" "$ROOT/results/d2v_attr_d_cached/" 2>/dev/null || true
rsync -az --include 'slurm_attr_dprobe2_*' --exclude '*' "$HPC:~/DCR/" "$OUT/logs/" 2>/dev/null || true
{
  echo "# D-stage set-P4 probe rerun (cached-filtered split, 1 seed) — $(date)"
  echo; echo "job: $JID"; echo; echo "## sacct"; echo '~~~'; cat "$OUT/sacct_dprobe2.txt"; echo '~~~'
  echo; echo "## results"; find "$ROOT/results/d2v_attr_d_cached" -name summary.csv 2>/dev/null | sort
} > "$OUT/STATUS4.md"
echo "=== DONE $(date)"
