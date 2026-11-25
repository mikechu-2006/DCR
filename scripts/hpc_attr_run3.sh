#!/bin/bash
# Third (and final) attempt at the 1-seed attribute pilot.
#
# Sequence:
#   0. LOCAL full-cache regression scan of the fixed parser (21,949 charts,
#      parse + attribute window build).  Abort before touching HPC if any chart
#      still fails.
#   1. cancel the currently queued jobs from the failed second attempt
#   2. rsync the fixed attributes.py
#   3. submit 6 attribute training jobs, 2 GPU extraction jobs, 2 CPU probe jobs
#      (slurm afterok dependency chain; the 8ch controls S-a0cont and D-a0d are
#      already completed from attempt 1)
#   4. robust monitor: ssh hiccups are retried, DependencyNeverSatisfied is
#      treated as terminal, and an empty squeue must be confirmed twice
#   5. pull results/logs, write results/d2v_attr_run/STATUS3.md
#
# Run in the background:  bash scripts/hpc_attr_run3.sh
set -uo pipefail

HPC="aiaa4051_083@hpc2login.hpc.hkust-gz.edu.cn"
ROOT="/home/mikechu/Documents/Projects/DCR"
OUT="$ROOT/results/d2v_attr_run"
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"
LOG="$OUT/orchestrator3.log"
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

# ------------------------------------------------ 0. local regression scan
echo "=== [0/5] local full-cache parser scan ($(date))"
cd "$ROOT"
python - <<'PY'
import sys, time
sys.path.insert(0, 'data2vec_main'); sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
from d2v_main.attributes import ChannelSpec, parse_attributes, build_attribute_window
cfg = {'data': {'channels': {'groups': {'release': {}, 'tempo': {}, 'sv': {},
                                        'kiai': {}, 'meta': {}}}}}
spec = ChannelSpec.from_config(cfg)
paths = sorted(p for p in Path('data/osu_cache').glob('*.osu') if p.stem.isdigit())
t0 = time.time(); bad = 0
for i, p in enumerate(paths, 1):
    try:
        a = parse_attributes(p, 100)
        for start in (0, 4000):
            w = build_attribute_window(a, start, 1600, 100, spec)
            assert w.shape == (19, 1600) and np.isfinite(w).all()
    except Exception as exc:
        bad += 1
        if bad <= 10:
            print('FAIL', p.name, type(exc).__name__, exc, flush=True)
    if i % 5000 == 0:
        print('scan', i, 'bad', bad, 'elapsed', round(time.time() - t0, 1), flush=True)
print('SCAN_RESULT checked=%d bad=%d elapsed=%.1f' % (len(paths), bad, time.time() - t0))
sys.exit(0 if bad == 0 else 1)
PY
SCAN_RC=$?
if [ "$SCAN_RC" != "0" ]; then
  echo "=== FATAL: local parser scan failed (rc=$SCAN_RC); not submitting"
  printf '# Attempt 3 aborted\n\nLocal full-cache parser scan failed at %s\n' "$(date)" > "$OUT/STATUS3.md"
  exit 1
fi
echo "=== local scan clean ($(date))"

# ------------------------------------- 1. cancel stale attempt-2 jobs
echo "=== [1/5] cancelling stale attempt-2 jobs ($(date))"
OLD_IDS=$(grep -oE '[0-9]+' "$OUT/jobs2.txt" 2>/dev/null | xargs | tr ' ' ',')
if [ -n "$OLD_IDS" ]; then
  $SSH "$HPC" "$RENV; scancel $OLD_IDS 2>/dev/null; true" || true
fi

# ------------------------------------------------ 2. sync fixed parser
echo "=== [2/5] syncing fixed code ($(date))"
rsync -az data2vec_main/d2v_main/attributes.py "$HPC:~/DCR/data2vec_main/d2v_main/" || exit 1
rsync -az data2vec_main/scripts/test_attribute_channels.py "$HPC:~/DCR/data2vec_main/scripts/" || true

# ------------------------------------------------ 3. submit training
echo "=== [3/5] submitting training jobs ($(date))"
S_IDS=""
for spec in "a12:configs/audio_style_a12.yaml" "a20:configs/audio_style_a20.yaml" "a27:configs/audio_style_a27.yaml"; do
  ARM=$(printf '%s' "$spec" | cut -d: -f1); CONFIG=$(printf '%s' "$spec" | cut -d: -f2)
  id=$(submit "ARM=$ARM CONFIG=$CONFIG INIT_FROM=runs/main_holdout/checkpoints/last.pt EPOCHS=5 LR=5e-5 OUT_DIR=runs/attr/$ARM" "data2vec_main/scripts/attr_train.sbatch")
  if [ -z "$id" ]; then echo "=== FATAL: S submit $ARM"; exit 1; fi
  S_IDS="$S_IDS $id"; echo "[S-train] $ARM job=$id"
done
D_IDS=""
for spec in "a12d:configs/audio_style_a12.yaml" "a20d:configs/audio_style_a20.yaml" "a27d:configs/audio_style_a27.yaml"; do
  ARM=$(printf '%s' "$spec" | cut -d: -f1); CONFIG=$(printf '%s' "$spec" | cut -d: -f2)
  id=$(submit "ARM=$ARM CONFIG=$CONFIG EPOCHS=20 LR=1e-4 IDS_FILE=splits/mapset_holdout/pretrain_ids.txt OUT_DIR=runs/attr_d/$ARM" "data2vec_main/scripts/attr_train.sbatch")
  if [ -z "$id" ]; then echo "=== FATAL: D submit $ARM"; exit 1; fi
  D_IDS="$D_IDS $id"; echo "[D-train] $ARM job=$id"
done
S_TRIM=$(echo $S_IDS | xargs | tr ' ' ':')
D_TRIM=$(echo $D_IDS | xargs | tr ' ' ':')

# ------------------------------------------------ 4. extraction + probes
echo "=== [4/5] submitting extraction + probes ($(date))"
SE_ID=$(submit "MODE=s" "--dependency=afterok:$S_TRIM data2vec_main/scripts/attr_extract_all.sbatch")
DE_ID=$(submit "MODE=d" "--dependency=afterok:$D_TRIM data2vec_main/scripts/attr_extract_all.sbatch")
if [ -z "$SE_ID" ] || [ -z "$DE_ID" ]; then echo "=== FATAL: extract submit"; exit 1; fi
SP_ID=$(submit "STAGE=s" "--dependency=afterok:$SE_ID scripts/slurm/attr_all_probes.sbatch")
DP_ID=$(submit "STAGE=d" "--dependency=afterok:$DE_ID scripts/slurm/attr_all_probes.sbatch")
if [ -z "$SP_ID" ] || [ -z "$DP_ID" ]; then echo "=== FATAL: probe submit"; exit 1; fi
echo "[S-extract] $SE_ID"; echo "[D-extract] $DE_ID"; echo "[S-probe] $SP_ID"; echo "[D-probe] $DP_ID"

ALL_IDS="$S_IDS $D_IDS $SE_ID $DE_ID $SP_ID $DP_ID"
IDS_CSV=$(echo $ALL_IDS | xargs | tr ' ' ',')
{
  echo "protocol_s_train: $S_IDS"
  echo "protocol_d_train: $D_IDS"
  echo "extract_s: $SE_ID"
  echo "extract_d: $DE_ID"
  echo "probe_s: $SP_ID"
  echo "probe_d: $DP_ID"
} > "$OUT/jobs3.txt"
echo "=== submitted:"; cat "$OUT/jobs3.txt"

# ------------------------------------------------ 5. robust monitor
echo "=== [5/5] monitoring ($(date))"
EMPTY_STREAK=0
while true; do
  NOW=$(date +%s)
  if [ $((NOW - START_TS)) -gt 28800 ]; then echo "=== monitor timeout after 8h"; break; fi
  RAW=$($SSH "$HPC" "$RENV; echo SQUEUE_BEGIN; squeue -h -j $IDS_CSV -o '%i|%T|%r' 2>/dev/null; echo SQUEUE_END" 2>/dev/null)
  if ! printf '%s' "$RAW" | grep -q 'SQUEUE_END'; then
    echo "[monitor] $(date) ssh hiccup; retry"; sleep 90; continue
  fi
  BODY=$(printf '%s\n' "$RAW" | sed -n '/SQUEUE_BEGIN/,/SQUEUE_END/p' | grep '|')
  ACTIVE=$(printf '%s\n' "$BODY" | awk -F'|' 'NF>=2 { if ($2=="RUNNING"||$2=="CONFIGURING"||$2=="COMPLETING"||$2=="STAGE_OUT") n++; else if ($2=="PENDING" && $3 !~ /DependencyNeverSatisfied/) n++ } END {print n+0}')
  if [ "$ACTIVE" -eq 0 ]; then
    EMPTY_STREAK=$((EMPTY_STREAK + 1))
    echo "[monitor] $(date) active=0 (empty_streak=$EMPTY_STREAK)"
    if [ "$EMPTY_STREAK" -ge 2 ]; then echo "=== confirmed no active tracked jobs ($(date))"; break; fi
    sleep 90
  else
    EMPTY_STREAK=0
    echo "[monitor] $(date) active=$ACTIVE"
    sleep 120
  fi
done

# ------------------------------------------------ collect
echo "=== collecting ($(date))"
$SSH "$HPC" "$RENV; sacct -n -P -j $IDS_CSV --format=JobID%12,JobName%20,State,ExitCode,Elapsed 2>/dev/null" > "$OUT/sacct3.txt" || true
cat "$OUT/sacct3.txt"
mkdir -p "$ROOT/results/d2v_attr" "$ROOT/results/d2v_attr_d"
rsync -az --info=progress2 "$HPC:~/DCR/results/d2v_attr/" "$ROOT/results/d2v_attr/" 2>/dev/null || true
rsync -az --info=progress2 "$HPC:~/DCR/results/d2v_attr_d/" "$ROOT/results/d2v_attr_d/" 2>/dev/null || true
rsync -az "$HPC:~/DCR/data/processed/" "$ROOT/data/processed/" \
  --include 'chart_content_d2v_a*-1k.parquet' --include 'chart_content_d2v_a*-mclean.parquet' \
  --exclude '*' 2>/dev/null || true
rsync -az --include 'slurm_attr_*' --include 'slurm_d2v_*' --exclude '*' \
  "$HPC:~/DCR/" "$LOGS/" 2>/dev/null || true
{
  echo "# Attribute pilot attempt 3 (1 split seed) — $(date)"
  echo
  echo "## jobs"; echo '~~~'; cat "$OUT/jobs3.txt"; echo '~~~'
  echo "## sacct"; echo '~~~'; cat "$OUT/sacct3.txt"; echo '~~~'
  echo "## result files"
  find "$ROOT/results/d2v_attr" "$ROOT/results/d2v_attr_d" \( -name 'summary.md' -o -name 'summary.csv' -o -name 'rank_eval.json' \) 2>/dev/null | sort
} > "$OUT/STATUS3.md"
echo "=== DONE $(date)"
