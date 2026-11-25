#!/bin/bash
# Resilient shuffle-control chain: the HPC account is currently at its
# AssocGrpSubmitJobsLimit (another session's array job holds the quota), so this
# script retries the train submission every 5 min (up to 6 h), then submits the
# dependent extraction + probe jobs, waits, pulls results and writes STATUS5c.md.
set -uo pipefail
HPC="aiaa4051_083@hpc2login.hpc.hkust-gz.edu.cn"
ROOT="/home/mikechu/Documents/Projects/DCR"
OUT="$ROOT/results/d2v_attr_run"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/shuffle_retry.log") 2>&1
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25"
RENV='export PATH=/opt/slurm/bin:$PATH; cd $HOME/DCR'

submit_raw() {
  # $1 = shell env assignments, $2 = sbatch args+script; prints job id or empty
  local out id
  out=$($SSH "$HPC" "$RENV; $1 sbatch $2" 2>&1)
  id=$(printf '%s' "$out" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+' | tail -1)
  if [ -z "$id" ]; then echo "[submit] failed: $out" >&2; fi
  printf '%s' "$id"
}

echo "=== shuffle retry chain start $(date)"
TID=""
for attempt in $(seq 1 72); do
  TID=$(submit_raw "ARM=a20shuf CONFIG=configs/audio_style_a20_shuf.yaml EPOCHS=20 LR=1e-4 IDS_FILE=splits/mapset_holdout/pretrain_ids.txt OUT_DIR=runs/attr_d/a20shuf" "data2vec_main/scripts/attr_train.sbatch")
  [ -n "$TID" ] && break
  echo "[retry] $(date) train submit blocked (attempt $attempt); sleep 300"
  sleep 300
done
if [ -z "$TID" ]; then
  echo "=== FATAL: quota never freed; giving up"
  printf '# Shuffle control blocked\n\nAssocGrpSubmitJobsLimit persisted for 6h at %s\n' "$(date)" > "$OUT/STATUS5c.md"
  exit 1
fi
echo "[shuffle-train] $TID"
EID=$(submit_raw "ARM=a20shuf CONFIG=configs/audio_style_a20_shuf.yaml CKPT=runs/attr_d/a20shuf/checkpoints/last.pt IDS_FILE=splits/mapset_holdout/eval_ids.txt OUT=../data/processed/chart_content_d2v_a20shuf-mclean.parquet TAG=d2v_a20shuf-mclean" "--dependency=afterok:$TID data2vec_main/scripts/attr_extract.sbatch")
echo "[shuffle-extract] $EID"
PID=""
if [ -n "$EID" ]; then
  PID=$(submit_raw "CONTENT=data/processed/chart_content_d2v_a20shuf-mclean.parquet OUT_DIR=results/d2v_attr_d_cached/a20shuf/setp4 SEEDS=1 MODELS=linear,linear_extra SPLIT_P4=data/processed/step1_mclean25_p4sets_cached" "--dependency=afterok:$EID scripts/slurm/attr_probe_cpu.sbatch")
  echo "[shuffle-probe] $PID"
fi
{ echo "train: $TID"; echo "extract: $EID"; echo "probe: $PID"; } > "$OUT/jobs_shuffle_retry.txt"
IDS=$(printf '%s' "$TID,$EID,$PID" | tr -d ' ')
echo "=== waiting $(date) ids=$IDS"
EMPTY=0
while true; do
  RAW=$($SSH "$HPC" "$RENV; echo BEGIN; squeue -h -j $IDS -o '%i|%T|%r' 2>/dev/null; echo END" 2>/dev/null)
  if ! printf '%s' "$RAW" | grep -q 'END'; then echo "[wait] ssh hiccup"; sleep 120; continue; fi
  BODY=$(printf '%s\n' "$RAW" | sed -n '/BEGIN/,/END/p' | grep '|')
  ACTIVE=$(printf '%s\n' "$BODY" | awk -F'|' 'NF>=2 { if ($2=="RUNNING"||$2=="CONFIGURING"||$2=="COMPLETING") n++; else if ($2=="PENDING" && $3 !~ /DependencyNeverSatisfied/) n++ } END {print n+0}')
  if [ "$ACTIVE" -eq 0 ]; then EMPTY=$((EMPTY+1)); else EMPTY=0; fi
  echo "[wait] $(date) active=$ACTIVE streak=$EMPTY"
  [ "$EMPTY" -ge 2 ] && break
  sleep 120
done
echo "=== collecting $(date)"
$SSH "$HPC" "$RENV; sacct -n -P -j $IDS --format=JobID%12,JobName%20,State,ExitCode,Elapsed 2>/dev/null" > "$OUT/sacct_shuffle_retry.txt" || true
cat "$OUT/sacct_shuffle_retry.txt"
rsync -az "$HPC:~/DCR/results/d2v_attr_d_cached/a20shuf/" "$ROOT/results/d2v_attr_d_cached/a20shuf/" 2>/dev/null || true
rsync -az "$HPC:~/DCR/data/processed/" "$ROOT/data/processed/" \
  --include 'chart_content_d2v_a20shuf-mclean.parquet' --exclude '*' 2>/dev/null || true
{ echo "# Shuffle negative control (retry chain) — $(date)"; echo; echo "## jobs"; cat "$OUT/jobs_shuffle_retry.txt"; echo; echo "## sacct"; echo '~~~'; cat "$OUT/sacct_shuffle_retry.txt"; echo '~~~'; echo; echo "## results"; find "$ROOT/results/d2v_attr_d_cached/a20shuf" -name summary.csv 2>/dev/null; } > "$OUT/STATUS5c.md"
echo "=== DONE $(date)"
