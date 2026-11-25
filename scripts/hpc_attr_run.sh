#!/bin/bash
# Orchestrate the 1-seed attribute-channel pilot on the HKUST-GZ HPC.
#
# Flow (all jobs are independent of the local machine once submitted):
#   1. wait for the login-node chart index cache (~15 min; started separately)
#   2. submit Protocol S: 4 continuation runs (A0-cont/A1/A2/A3, 5 epochs, 1 seed)
#      and Protocol D: 4 from-scratch runs (A0'/A1'/A2'/A3', 20 epochs, mapset split)
#   3. submit GPU extraction jobs with --dependency=afterok on the training jobs
#   4. submit CPU probe jobs with --dependency=afterok on the extraction jobs
#   5. poll squeue until every tracked job has left the queue or a timeout hits
#   6. pull results/logs to the local checkout and write results/d2v_attr_run/STATUS.md
#
# Run it in the background:  bash scripts/hpc_attr_run.sh
set -uo pipefail

HPC="aiaa4051_083@hpc2login.hpc.hkust-gz.edu.cn"
ROOT="/home/mikechu/Documents/Projects/DCR"
OUT="$ROOT/results/d2v_attr_run"
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"
LOG="$OUT/orchestrator.log"
exec > >(tee -a "$LOG") 2>&1
START_TS=$(date +%s)

SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25"
RENV='export PATH=/opt/slurm/bin:$PATH; cd $HOME/DCR'

remote() { $SSH "$HPC" "$RENV; $*"; }

submit() {
  # submit "<env assignments>" "<sbatch script> [--dependency=...]" -> job id on stdout
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

# ---------------------------------------------------------------- 1. cache
echo "=== [1/6] waiting for HPC chart index cache ($(date))"
CACHE_OK=0
for i in $(seq 1 100); do
  if $SSH "$HPC" 'test -f $HOME/DCR/data2vec_main/cache/chart_index_100.json' 2>/dev/null; then
    CACHE_OK=1; break
  fi
  sleep 45
done
if [ "$CACHE_OK" != "1" ]; then
  echo "=== FATAL: chart index cache not ready after 75 min; aborting"
  printf '# Attribute pilot aborted\n\nchart_index_100.json not ready at %s\n' "$(date)" > "$OUT/STATUS.md"
  exit 1
fi
echo "=== cache ready ($(date))"

# ------------------------------------------------------- 2. training jobs
echo "=== [2/6] submitting Protocol S continuation runs ($(date))"
S_IDS=""
for spec in "a0cont:configs/audio_style_base.yaml" "a12:configs/audio_style_a12.yaml" "a20:configs/audio_style_a20.yaml" "a27:configs/audio_style_a27.yaml"; do
  ARM=$(printf '%s' "$spec" | cut -d: -f1)
  CONFIG=$(printf '%s' "$spec" | cut -d: -f2)
  id=$(submit "ARM=$ARM CONFIG=$CONFIG INIT_FROM=runs/main_holdout/checkpoints/last.pt EPOCHS=5 LR=5e-5 OUT_DIR=runs/attr/$ARM" "data2vec_main/scripts/attr_train.sbatch")
  if [ -z "$id" ]; then echo "=== FATAL: S train submit failed for $ARM"; exit 1; fi
  S_IDS="$S_IDS $id"; echo "[S-train] ARM=$ARM job=$id"
done
S_IDS_TRIMMED=$(echo $S_IDS | xargs | tr ' ' ':')

echo "=== submitting Protocol D from-scratch runs ($(date))"
D_IDS=""
for spec in "a0d:configs/audio_style_base.yaml" "a12d:configs/audio_style_a12.yaml" "a20d:configs/audio_style_a20.yaml" "a27d:configs/audio_style_a27.yaml"; do
  ARM=$(printf '%s' "$spec" | cut -d: -f1)
  CONFIG=$(printf '%s' "$spec" | cut -d: -f2)
  id=$(submit "ARM=$ARM CONFIG=$CONFIG EPOCHS=20 LR=1e-4 IDS_FILE=splits/mapset_holdout/pretrain_ids.txt OUT_DIR=runs/attr_d/$ARM" "data2vec_main/scripts/attr_train.sbatch")
  if [ -z "$id" ]; then echo "=== FATAL: D train submit failed for $ARM"; exit 1; fi
  D_IDS="$D_IDS $id"; echo "[D-train] ARM=$ARM job=$id"
done
D_IDS_TRIMMED=$(echo $D_IDS | xargs | tr ' ' ':')

# ------------------------------------------------------ 3. extraction jobs
echo "=== [3/6] submitting extraction jobs ($(date))"
SE_ID=$(submit "MODE=s" "--dependency=afterok:$S_IDS_TRIMMED data2vec_main/scripts/attr_extract_all.sbatch")
if [ -z "$SE_ID" ]; then echo "=== FATAL: S extract submit failed"; exit 1; fi
echo "[S-extract] job=$SE_ID"
DE_ID=$(submit "MODE=d" "--dependency=afterok:$D_IDS_TRIMMED data2vec_main/scripts/attr_extract_all.sbatch")
if [ -z "$DE_ID" ]; then echo "=== FATAL: D extract submit failed"; exit 1; fi
echo "[D-extract] job=$DE_ID"

# ---------------------------------------------------------- 4. probe jobs
echo "=== [4/6] submitting CPU probe jobs ($(date))"
SP_ID=$(submit "STAGE=s" "--dependency=afterok:$SE_ID scripts/slurm/attr_all_probes.sbatch")
if [ -z "$SP_ID" ]; then echo "=== FATAL: S probe submit failed"; exit 1; fi
echo "[S-probe] job=$SP_ID"
DP_ID=$(submit "STAGE=d" "--dependency=afterok:$DE_ID scripts/slurm/attr_all_probes.sbatch")
if [ -z "$DP_ID" ]; then echo "=== FATAL: D probe submit failed"; exit 1; fi
echo "[D-probe] job=$DP_ID"

ALL_IDS="$S_IDS $D_IDS $SE_ID $DE_ID $SP_ID $DP_ID"
printf '%s\n' "$S_IDS $D_IDS $SE_ID $DE_ID $SP_ID $DP_ID" | tr ' ' '\n' | grep -E '^[0-9]+$' > "$OUT/job_ids.txt"
{
  echo "protocol_s_train: $S_IDS"
  echo "protocol_d_train: $D_IDS"
  echo "extract_s: $SE_ID"
  echo "extract_d: $DE_ID"
  echo "probe_s: $SP_ID"
  echo "probe_d: $DP_ID"
} > "$OUT/jobs.txt"
echo "=== submitted jobs:"; cat "$OUT/jobs.txt"

# --------------------------------------------------------------- 5. wait
echo "=== [5/6] monitoring jobs until the queue is empty ($(date))"
while true; do
  NOW=$(date +%s)
  if [ $((NOW - START_TS)) -gt 32400 ]; then
    echo "=== monitor timeout after 9h"; break
  fi
  RUNNING=$($SSH "$HPC" "$RENV; squeue -h -u \$USER -o '%i' 2>/dev/null | tr -d ' '" 2>/dev/null)
  REMAIN=0
  for id in $ALL_IDS; do
    if printf '%s\n' "$RUNNING" | grep -qx "$id"; then REMAIN=$((REMAIN + 1)); fi
  done
  if [ "$REMAIN" -eq 0 ]; then echo "=== tracked jobs left the queue ($(date))"; break; fi
  echo "[monitor] $(date) tracked_remaining=$REMAIN"
  sleep 120
done

# --------------------------------------------------------- 6. collect
echo "=== [6/6] collecting accounting and results ($(date))"
$SSH "$HPC" "$RENV; sacct -n -P -j $(echo $ALL_IDS | tr ' ' ',') --format=JobID,JobName%22,State,ExitCode,Elapsed 2>/dev/null" > "$OUT/sacct.txt" || true
echo "=== sacct:"; cat "$OUT/sacct.txt"

mkdir -p "$ROOT/results/d2v_attr" "$ROOT/results/d2v_attr_d"
rsync -az --info=progress2 "$HPC:~/DCR/results/d2v_attr/" "$ROOT/results/d2v_attr/" 2>/dev/null || true
rsync -az --info=progress2 "$HPC:~/DCR/results/d2v_attr_d/" "$ROOT/results/d2v_attr_d/" 2>/dev/null || true
rsync -az "$HPC:~/DCR/data/processed/" "$ROOT/data/processed/" \
  --include 'chart_content_d2v_a*-1k.parquet' --include 'chart_content_d2v_a*-mclean.parquet' \
  --exclude '*' 2>/dev/null || true
rsync -az --include 'slurm_attr_*' --include 'slurm_d2v_*' --exclude '*' \
  "$HPC:~/DCR/" "$LOGS/" 2>/dev/null || true

{
  echo "# Attribute-channel pilot run (1 split seed) — $(date)"
  echo
  echo "Started: $(date -d @$START_TS '+%F %T')"
  echo
  echo "## Jobs"
  echo '~~~'
  cat "$OUT/jobs.txt"
  echo '~~~'
  echo
  echo "## sacct"
  echo '~~~'
  cat "$OUT/sacct.txt"
  echo '~~~'
  echo
  echo "## Result files"
  find "$ROOT/results/d2v_attr" "$ROOT/results/d2v_attr_d" \( -name 'summary.md' -o -name 'summary.csv' -o -name 'rank_eval.json' \) 2>/dev/null | sort
  echo
  echo "## Slurm logs"
  ls -1 "$LOGS" 2>/dev/null | head -40
} > "$OUT/STATUS.md"
echo "=== DONE $(date)"
