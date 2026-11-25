#!/bin/bash
# Decisive negative control for the D-stage a20 gain: same 20-channel input, but
# the whole attribute block is rolled per chart (statistics preserved, alignment
# to note onsets destroyed).  Train -> extract -> set-P4 probe -> pull results.
set -uo pipefail
HPC="aiaa4051_083@hpc2login.hpc.hkust-gz.edu.cn"
ROOT="/home/mikechu/Documents/Projects/DCR"
OUT="$ROOT/results/d2v_attr_run"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/shuffle.log") 2>&1
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25"
RENV='export PATH=/opt/slurm/bin:$PATH; cd $HOME/DCR'
cd "$ROOT"

submit() {
  local envs="$1"; local script="$2"; local out id attempt
  for attempt in $(seq 1 30); do
    out=$($SSH "$HPC" "$RENV; $envs sbatch $script" 2>&1)
    id=$(printf '%s' "$out" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
    if [ -n "$id" ]; then printf '%s\n' "$id"; return 0; fi
    echo "[submit] attempt $attempt failed: $out"; sleep 60
  done
  return 1
}

echo "=== syncing shuffle code/config ($(date))"
rsync -az data2vec_main/d2v_main/attributes.py "$HPC:~/DCR/data2vec_main/d2v_main/" || exit 1
rsync -az data2vec_main/configs/audio_style_a20_shuf.yaml "$HPC:~/DCR/data2vec_main/configs/" || exit 1
rsync -az data2vec_main/scripts/test_attribute_channels.py "$HPC:~/DCR/data2vec_main/scripts/" || true

echo "=== submitting shuffle chain ($(date))"
TID=$(submit "ARM=a20shuf CONFIG=configs/audio_style_a20_shuf.yaml EPOCHS=20 LR=1e-4 IDS_FILE=splits/mapset_holdout/pretrain_ids.txt OUT_DIR=runs/attr_d/a20shuf" "data2vec_main/scripts/attr_train.sbatch")
[ -z "$TID" ] && { echo "FATAL train submit"; exit 1; }
echo "[shuffle-train] $TID"
EID=$(submit "ARM=a20shuf CONFIG=configs/audio_style_a20_shuf.yaml CKPT=runs/attr_d/a20shuf/checkpoints/last.pt IDS_FILE=splits/mapset_holdout/eval_ids.txt OUT=../data/processed/chart_content_d2v_a20shuf-mclean.parquet TAG=d2v_a20shuf-mclean" "--dependency=afterok:$TID data2vec_main/scripts/attr_extract.sbatch")
[ -z "$EID" ] && { echo "FATAL extract submit"; exit 1; }
echo "[shuffle-extract] $EID"
PID=$(submit "CONTENT=data/processed/chart_content_d2v_a20shuf-mclean.parquet OUT_DIR=results/d2v_attr_d_cached/a20shuf/setp4 SEEDS=1 MODELS=linear,linear_extra SPLIT_P4=data/processed/step1_mclean25_p4sets_cached" "--dependency=afterok:$EID scripts/slurm/attr_probe_cpu.sbatch")
[ -z "$PID" ] && { echo "FATAL probe submit"; exit 1; }
echo "[shuffle-probe] $PID"
{ echo "train: $TID"; echo "extract: $EID"; echo "probe: $PID"; } > "$OUT/jobs_shuffle.txt"
IDS="$TID,$EID,$PID"

echo "=== waiting ($(date))"
EMPTY=0
while true; do
  RAW=$($SSH "$HPC" "$RENV; echo BEGIN; squeue -h -j $IDS -o '%i|%T|%r' 2>/dev/null; echo END" 2>/dev/null)
  if ! printf '%s' "$RAW" | grep -q 'END'; then echo "[wait] ssh hiccup"; sleep 90; continue; fi
  BODY=$(printf '%s\n' "$RAW" | sed -n '/BEGIN/,/END/p' | grep '|')
  ACTIVE=$(printf '%s\n' "$BODY" | awk -F'|' 'NF>=2 { if ($2=="RUNNING"||$2=="CONFIGURING"||$2=="COMPLETING") n++; else if ($2=="PENDING" && $3 !~ /DependencyNeverSatisfied/) n++ } END {print n+0}')
  if [ "$ACTIVE" -eq 0 ]; then EMPTY=$((EMPTY+1)); else EMPTY=0; fi
  echo "[wait] $(date) active=$ACTIVE streak=$EMPTY"
  [ "$EMPTY" -ge 2 ] && break
  sleep 90
done

echo "=== collecting ($(date))"
$SSH "$HPC" "$RENV; sacct -n -P -j $IDS --format=JobID%12,JobName%20,State,ExitCode,Elapsed 2>/dev/null" > "$OUT/sacct_shuffle.txt" || true
cat "$OUT/sacct_shuffle.txt"
rsync -az "$HPC:~/DCR/results/d2v_attr_d_cached/a20shuf/" "$ROOT/results/d2v_attr_d_cached/a20shuf/" 2>/dev/null || true
rsync -az "$HPC:~/DCR/data/processed/" "$ROOT/data/processed/" \
  --include 'chart_content_d2v_a20shuf-mclean.parquet' --exclude '*' 2>/dev/null || true
{ echo "# Shuffle negative control — $(date)"; echo; echo "## jobs"; cat "$OUT/jobs_shuffle.txt"; echo; echo "## sacct"; echo '~~~'; cat "$OUT/sacct_shuffle.txt"; echo '~~~'; } > "$OUT/STATUS5.md"
echo "=== DONE $(date)"
