#!/bin/bash
# Wait for the two running probe jobs (S-probe 14114227, D-probe 14114228),
# then pull all 1-seed results/content tables/logs and write STATUS3b.md.
# Started after the machine restart killed the previous monitor; the HPC jobs
# themselves kept running.
set -uo pipefail
HPC="aiaa4051_083@hpc2login.hpc.hkust-gz.edu.cn"
ROOT="/home/mikechu/Documents/Projects/DCR"
OUT="$ROOT/results/d2v_attr_run"
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"
exec > >(tee -a "$OUT/finish.log") 2>&1
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25"
RENV='export PATH=/opt/slurm/bin:$PATH; cd $HOME/DCR'
IDS="14114227,14114228"
START=$(date +%s)
echo "=== waiting for probe jobs $IDS ($(date))"
EMPTY=0
while true; do
  NOW=$(date +%s)
  if [ $((NOW - START)) -gt 14400 ]; then echo "=== timeout after 4h"; break; fi
  RAW=$($SSH "$HPC" "$RENV; echo BEGIN; squeue -h -j $IDS -o '%i|%T|%r' 2>/dev/null; echo END" 2>/dev/null)
  if ! printf '%s' "$RAW" | grep -q 'END'; then echo "[wait] ssh hiccup $(date)"; sleep 90; continue; fi
  BODY=$(printf '%s\n' "$RAW" | sed -n '/BEGIN/,/END/p' | grep '|')
  ACTIVE=$(printf '%s\n' "$BODY" | awk -F'|' 'NF>=2 { if ($2=="RUNNING"||$2=="CONFIGURING"||$2=="COMPLETING") n++; else if ($2=="PENDING" && $3 !~ /DependencyNeverSatisfied/) n++ } END {print n+0}')
  if [ "$ACTIVE" -eq 0 ]; then EMPTY=$((EMPTY+1)); echo "[wait] $(date) active=0 streak=$EMPTY"; [ "$EMPTY" -ge 2 ] && break; sleep 90; else EMPTY=0; echo "[wait] $(date) active=$ACTIVE"; sleep 120; fi
done
echo "=== collecting ($(date))"
$SSH "$HPC" "$RENV; sacct -n -P -j $IDS --format=JobID%12,JobName%20,State,ExitCode,Elapsed 2>/dev/null" > "$OUT/sacct_finish.txt" || true
cat "$OUT/sacct_finish.txt"
mkdir -p "$ROOT/results/d2v_attr" "$ROOT/results/d2v_attr_d" "$ROOT/data/processed"
rsync -az "$HPC:~/DCR/results/d2v_attr/" "$ROOT/results/d2v_attr/" 2>/dev/null || true
rsync -az "$HPC:~/DCR/results/d2v_attr_d/" "$ROOT/results/d2v_attr_d/" 2>/dev/null || true
rsync -az "$HPC:~/DCR/data/processed/" "$ROOT/data/processed/" \
  --include 'chart_content_d2v_a*-1k.parquet' --include 'chart_content_d2v_a*-mclean.parquet' \
  --exclude '*' 2>/dev/null || true
rsync -az --include 'slurm_attr_*' --exclude '*' "$HPC:~/DCR/" "$LOGS/" 2>/dev/null || true
{
  echo "# Attribute pilot final collection — $(date)"
  echo; echo "## sacct"; echo '~~~'; cat "$OUT/sacct_finish.txt"; echo '~~~'
  echo "## S summaries"; find "$ROOT/results/d2v_attr" -name summary.csv 2>/dev/null | sort
  echo "## D summaries"; find "$ROOT/results/d2v_attr_d" -name summary.csv 2>/dev/null | sort
  echo "## content tables"; ls -1 "$ROOT/data/processed/"chart_content_d2v_a0cont-1k.parquet "$ROOT/data/processed/"chart_content_d2v_a12-1k.parquet "$ROOT/data/processed/"chart_content_d2v_a20-1k.parquet "$ROOT/data/processed/"chart_content_d2v_a27-1k.parquet "$ROOT/data/processed/"chart_content_d2v_a0d-mclean.parquet "$ROOT/data/processed/"chart_content_d2v_a12d-mclean.parquet "$ROOT/data/processed/"chart_content_d2v_a20d-mclean.parquet "$ROOT/data/processed/"chart_content_d2v_a27d-mclean.parquet 2>/dev/null
} > "$OUT/STATUS3b.md"
echo "=== DONE $(date)"
