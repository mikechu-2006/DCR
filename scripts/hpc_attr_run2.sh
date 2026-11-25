#!/bin/bash
# Re-run the failed attribute arms (malformed-.osu parser fix) and complete the
# 1-seed pilot: 6 training jobs -> extraction -> CPU probes -> pull results.
#
# The two 8ch control arms (S a0cont, D a0d) already COMPLETED; this script only
# redoes the 6 attribute arms plus the downstream extraction/probe chain.
#
# Run in the background:  bash scripts/hpc_attr_run2.sh
set -uo pipefail

HPC="aiaa4051_083@hpc2login.hpc.hkust-gz.edu.cn"
ROOT="/home/mikechu/Documents/Projects/DCR"
OUT="$ROOT/results/d2v_attr_run"
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"
LOG="$OUT/orchestrator2.log"
exec > >(tee -a "$LOG") 2>&1
START_TS=$(date +%s)

SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25"
RENV='export PATH=/opt/slurm/bin:$PATH; cd $HOME/DCR'

submit() {
  local envs="$1"; local script="$2"; local out id attempt
  for attempt in $(seq 1 40); do
    out=$($SSH "$HPC" "$RENV; $envs sbatch $script" 2>&1)
    id=$(printf '%s' "$out" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
    if [ -n "$id" ]; then printf '%s\n' "$id"; return 0; fi
    echo "[submit] attempt $attempt failed: $out"
    sleep 60
  done
  return 1
}

echo "=== [0/5] cancelling stale dependency jobs + cleaning partial arms ($(date))"
$SSH "$HPC" "$RENV; scancel 14101489 14101490 14101491 14101495 2>/dev/null; true" || true
$SSH "$HPC" "$RENV; rm -rf data2vec_main/runs/attr/a12 data2vec_main/runs/attr/a20 data2vec_main/runs/attr/a27 data2vec_main/runs/attr_d/a12d data2vec_main/runs/attr_d/a20d data2vec_main/runs/attr_d/a27d; echo CLEANED" || true

echo "=== [1/5] submitting Protocol S attribute arms ($(date))"
S_IDS=""
for spec in "a12:configs/audio_style_a12.yaml" "a20:configs/audio_style_a20.yaml" "a27:configs/audio_style_a27.yaml"; do
  ARM=$(printf '%s' "$spec" | cut -d: -f1)
  CONFIG=$(printf '%s' "$spec" | cut -d: -f2)
  id=$(submit "ARM=$ARM CONFIG=$CONFIG INIT_FROM=runs/main_holdout/checkpoints/last.pt EPOCHS=5 LR=5e-5 OUT_DIR=runs/attr/$ARM" "data2vec_main/scripts/attr_train.sbatch")
  if [ -z "$id" ]; then echo "=== FATAL: S submit failed for $ARM"; exit 1; fi
  S_IDS="$S_IDS $id"; echo "[S-train] ARM=$ARM job=$id"
done
S_IDS_TRIMMED=$(echo $S_IDS | xargs | tr ' ' ':')

echo "=== [2/5] submitting Protocol D attribute arms ($(date))"
D_IDS=""
for spec in "a12d:configs/audio_style_a12.yaml" "a20d:configs/audio_style_a20.yaml" "a27d:configs/audio_style_a27.yaml"; do
  ARM=$(printf '%s' "$spec" | cut -d: -f1)
  CONFIG=$(printf '%s' "$spec" | cut -d: -f2)
  id=$(submit "ARM=$ARM CONFIG=$CONFIG EPOCHS=20 LR=1e-4 IDS_FILE=splits/mapset_holdout/pretrain_ids.txt OUT_DIR=runs/attr_d/$ARM" "data2vec_main/scripts/attr_train.sbatch")
  if [ -z "$id" ]; then echo "=== FATAL: D submit failed for $ARM"; exit 1; fi
  D_IDS="$D_IDS $id"; echo "[D-train] ARM=$ARM job=$id"
done
D_IDS_TRIMMED=$(echo $D_IDS | xargs | tr ' ' ':')

echo "=== [3/5] submitting extraction jobs ($(date))"
SE_ID=$(submit "MODE=s" "--dependency=afterok:$S_IDS_TRIMMED data2vec_main/scripts/attr_extract_all.sbatch")
if [ -z "$SE_ID" ]; then echo "=== FATAL: S extract submit failed"; exit 1; fi
DE_ID=$(submit "MODE=d" "--dependency=afterok:$D_IDS_TRIMMED data2vec_main/scripts/attr_extract_all.sbatch")
if [ -z "$DE_ID" ]; then echo "=== FATAL: D extract submit failed"; exit 1; fi
echo "[S-extract] job=$SE_ID"; echo "[D-extract] job=$DE_ID"

echo "=== [4/5] submitting probe jobs ($(date))"
SP_ID=$(submit "STAGE=s" "--dependency=afterok:$SE_ID scripts/slurm/attr_all_probes.sbatch")
if [ -z "$SP_ID" ]; then echo "=== FATAL: S probe submit failed"; exit 1; fi
DP_ID=$(submit "STAGE=d" "--dependency=afterok:$DE_ID scripts/slurm/attr_all_probes.sbatch")
if [ -z "$DP_ID" ]; then echo "=== FATAL: D probe submit failed"; exit 1; fi
echo "[S-probe] job=$SP_ID"; echo "[D-probe] job=$DP_ID"

ALL_IDS="$S_IDS $D_IDS $SE_ID $DE_ID $SP_ID $DP_ID"
IDS_CSV=$(echo $ALL_IDS | xargs | tr ' ' ',')
{
  echo "protocol_s_train: $S_IDS"
  echo "protocol_d_train: $D_IDS"
  echo "extract_s: $SE_ID"
  echo "extract_d: $DE_ID"
  echo "probe_s: $SP_ID"
  echo "probe_d: $DP_ID"
} > "$OUT/jobs2.txt"
echo "=== submitted:"; cat "$OUT/jobs2.txt"

echo "=== [5/5] monitoring (DependencyNeverSatisfied treated as terminal) ($(date))"
while true; do
  NOW=$(date +%s)
  if [ $((NOW - START_TS)) -gt 21600 ]; then echo "=== monitor timeout after 6h"; break; fi
  RAW=$($SSH "$HPC" "$RENV; squeue -h -j $IDS_CSV -o '%i|%T|%r' 2>/dev/null" 2>/dev/null)
  ACTIVE=$(printf '%s\n' "$RAW" | awk -F'|' 'NF>=2 { if ($2=="RUNNING" || $2=="CONFIGURING" || $2=="COMPLETING" || $2=="STAGE_OUT") n++; else if ($2=="PENDING" && $3 !~ /DependencyNeverSatisfied/) n++ } END { print n+0 }')
  if [ "$ACTIVE" -eq 0 ]; then echo "=== no active tracked jobs ($(date))"; break; fi
  echo "[monitor] $(date) active=$ACTIVE"
  sleep 120
done

echo "=== collecting ($(date))"
$SSH "$HPC" "$RENV; sacct -n -P -j $IDS_CSV --format=JobID%12,JobName%20,State,ExitCode,Elapsed 2>/dev/null" > "$OUT/sacct2.txt" || true
cat "$OUT/sacct2.txt"
mkdir -p "$ROOT/results/d2v_attr" "$ROOT/results/d2v_attr_d"
rsync -az --info=progress2 "$HPC:~/DCR/results/d2v_attr/" "$ROOT/results/d2v_attr/" 2>/dev/null || true
rsync -az --info=progress2 "$HPC:~/DCR/results/d2v_attr_d/" "$ROOT/results/d2v_attr_d/" 2>/dev/null || true
rsync -az "$HPC:~/DCR/data/processed/" "$ROOT/data/processed/" \
  --include 'chart_content_d2v_a*-1k.parquet' --include 'chart_content_d2v_a*-mclean.parquet' \
  --exclude '*' 2>/dev/null || true
rsync -az --include 'slurm_attr_*' --include 'slurm_d2v_*' --exclude '*' \
  "$HPC:~/DCR/" "$LOGS/" 2>/dev/null || true

{
  echo "# Attribute pilot re-run (1 split seed) — $(date)"
  echo
  echo "## jobs"
  echo '~~~'
  cat "$OUT/jobs2.txt"
  echo '~~~'
  echo "## sacct"
  echo '~~~'
  cat "$OUT/sacct2.txt"
  echo '~~~'
  echo "## result files"
  find "$ROOT/results/d2v_attr" "$ROOT/results/d2v_attr_d" \( -name 'summary.md' -o -name 'summary.csv' -o -name 'rank_eval.json' \) 2>/dev/null | sort
} > "$OUT/STATUS2.md"
echo "=== DONE $(date)"
