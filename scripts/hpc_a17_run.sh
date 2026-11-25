#!/bin/bash
# a17 ablation on the NEW HPC account: a20d minus SV/kiai (17 channels).
#   train a17d from scratch 20 ep on the mapset-disjoint pool
#   -> extract 5,462 eval charts -> set-P4 probe (a17d + a20d re-probe) -> pull
set -uo pipefail
HOST="aiaa3111_l01_066@hpc2login.hpc.hkust-gz.edu.cn"
ROOT="/home/mikechu/Documents/Projects/DCR"
OUT="$ROOT/results/d2v_attr_run"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/a17_run.log") 2>&1
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25"
RENV='export PATH=/opt/slurm/bin:$PATH; cd $HOME/DCR'

submit() {
  local envs="$1"; local script="$2"; local out id attempt
  for attempt in $(seq 1 30); do
    out=$($SSH "$HOST" "$RENV; $envs sbatch $script" 2>&1)
    id=$(printf '%s' "$out" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+' | tail -1)
    if [ -n "$id" ]; then printf '%s\n' "$id"; return 0; fi
    echo "[submit] attempt $attempt failed: $out"
    sleep 60
  done
  return 1
}

echo "=== a17 run start $(date)"
TID=$(submit "ARM=a17d CONFIG=configs/audio_style_a17.yaml EPOCHS=20 LR=1e-4 IDS_FILE=splits/mapset_holdout/pretrain_ids.txt OUT_DIR=runs/attr_d/a17d" "data2vec_main/scripts/attr_train.sbatch")
if [ -z "$TID" ]; then echo "FATAL train submit"; exit 1; fi
echo "[a17-train] $TID"
EID=$(submit "ARM=a17d CONFIG=configs/audio_style_a17.yaml CKPT=runs/attr_d/a17d/checkpoints/last.pt IDS_FILE=splits/mapset_holdout/eval_ids.txt OUT=../data/processed/chart_content_d2v_a17d-mclean.parquet TAG=d2v_a17d-mclean" "--dependency=afterok:$TID data2vec_main/scripts/attr_extract.sbatch")
if [ -z "$EID" ]; then echo "FATAL extract submit"; exit 1; fi
echo "[a17-extract] $EID"
PID=$(submit "" "--dependency=afterok:$EID scripts/slurm/a17_probe.sbatch")
if [ -z "$PID" ]; then echo "FATAL probe submit"; exit 1; fi
echo "[a17-probe] $PID"
{ echo "train: $TID"; echo "extract: $EID"; echo "probe: $PID"; } > "$OUT/jobs_a17.txt"
IDS="$TID,$EID,$PID"
echo "=== waiting $(date)"
EMPTY=0
while true; do
  RAW=$($SSH "$HOST" "$RENV; echo BEGIN; squeue -h -j $IDS -o '%i|%T|%r' 2>/dev/null; echo END" 2>/dev/null)
  if ! printf '%s' "$RAW" | grep -q 'END'; then echo "[wait] ssh hiccup"; sleep 90; continue; fi
  BODY=$(printf '%s\n' "$RAW" | sed -n '/BEGIN/,/END/p' | grep '|')
  ACTIVE=$(printf '%s\n' "$BODY" | awk -F'|' 'NF>=2 { if ($2=="RUNNING"||$2=="CONFIGURING"||$2=="COMPLETING") n++; else if ($2=="PENDING" && $3 !~ /DependencyNeverSatisfied/) n++ } END {print n+0}')
  if [ "$ACTIVE" -eq 0 ]; then EMPTY=$((EMPTY+1)); else EMPTY=0; fi
  echo "[wait] $(date) active=$ACTIVE streak=$EMPTY"
  [ "$EMPTY" -ge 2 ] && break
  sleep 120
done
echo "=== collecting $(date)"
$SSH "$HOST" "$RENV; sacct -n -P -j $IDS --format=JobID%12,JobName%20,State,ExitCode,Elapsed 2>/dev/null" > "$OUT/sacct_a17.txt" || true
cat "$OUT/sacct_a17.txt"
rsync -az "$HOST:~/DCR/results/d2v_attr_d_cached/a17d/" "$ROOT/results/d2v_attr_d_cached/a17d/" 2>/dev/null || true
rsync -az "$HOST:~/DCR/results/d2v_attr_d_cached/a20d_reprobe/" "$ROOT/results/d2v_attr_d_cached/a20d_reprobe/" 2>/dev/null || true
rsync -az "$HOST:~/DCR/data/processed/" "$ROOT/data/processed/" --include 'chart_content_d2v_a17d-mclean.parquet' --exclude '*' 2>/dev/null || true
{ echo "# a17 ablation (a20d minus SV/kiai) — $(date)"; echo; echo "jobs:"; cat "$OUT/jobs_a17.txt"; echo; echo "sacct:"; echo '~~~'; cat "$OUT/sacct_a17.txt"; echo '~~~'; echo; echo "summaries:"; find "$ROOT/results/d2v_attr_d_cached" -path '*a17*' -name summary.csv 2>/dev/null; find "$ROOT/results/d2v_attr_d_cached" -path '*a20d_reprobe*' -name summary.csv 2>/dev/null; } > "$OUT/STATUS_A17.md"
echo "=== a17 run DONE $(date)"
