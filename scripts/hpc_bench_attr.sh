#!/bin/bash
# VSRG Dan benchmark ablation for a0d / a17d / a20d with 5-fold chart-group
# isotonic calibration (same protocol as docs/benchmark_d2v.md):
#   1. relay the a0d ckpt from the OLD account to the NEW account (737 MB pipe)
#   2. sync the patched calibration script + reference predictions
#   3. GPU: encode all 21,949 charts + the 1,382 benchmark charts per arm
#   4. CPU: fit linear_extra on randommerge, infer benchmark, calibrate 5-fold
#   5. pull results, write results/benchmark_attr/STATUS_BENCH.md
set -uo pipefail
OLD="aiaa4051_083@hpc2login.hpc.hkust-gz.edu.cn"
NEW="aiaa3111_l01_066@hpc2login.hpc.hkust-gz.edu.cn"
ROOT="/home/mikechu/Documents/Projects/DCR"
OUT="$ROOT/results/benchmark_attr"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/bench_attr.log") 2>&1
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25"
RENV='export PATH=/opt/slurm/bin:$PATH; cd $HOME/DCR'
START=$(date +%s)
echo "=== bench-attr start $(date)"

submit() {
  local envs="$1"; local script="$2"; local out id attempt
  for attempt in $(seq 1 30); do
    out=$($SSH "$NEW" "$RENV; $envs sbatch $script" 2>&1)
    id=$(printf '%s' "$out" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+' | tail -1)
    if [ -n "$id" ]; then printf '%s\n' "$id"; return 0; fi
    echo "[submit] attempt $attempt failed: $out"; sleep 60
  done
  return 1
}

echo "=== [1/4] relay a0d checkpoint old->new ($(date))"
$SSH "$NEW" 'mkdir -p $HOME/DCR/data2vec_main/runs/attr_d/a0d/checkpoints' || exit 1
OLD_MD5=$($SSH "$OLD" 'md5sum $HOME/DCR/data2vec_main/runs/attr_d/a0d/checkpoints/last.pt' 2>/dev/null | awk '{print $1}')
echo "[relay] old md5=$OLD_MD5"
if [ -z "$OLD_MD5" ]; then echo "[relay] FATAL cannot read old a0d ckpt"; exit 1; fi
NEW_MD5=""
if $SSH "$NEW" 'test -f $HOME/DCR/data2vec_main/runs/attr_d/a0d/checkpoints/last.pt'; then
    NEW_MD5=$($SSH "$NEW" 'md5sum $HOME/DCR/data2vec_main/runs/attr_d/a0d/checkpoints/last.pt' 2>/dev/null | awk '{print $1}')
fi
if true; then
  if [ "$OLD_MD5" = "$NEW_MD5" ]; then
    echo "[relay] already present with matching md5"
  else
    echo "[relay] streaming..."
    $SSH "$OLD" 'cat $HOME/DCR/data2vec_main/runs/attr_d/a0d/checkpoints/last.pt' | \
      $SSH "$NEW" 'cat > $HOME/DCR/data2vec_main/runs/attr_d/a0d/checkpoints/last.pt'
    NEW_MD5=$($SSH "$NEW" 'md5sum $HOME/DCR/data2vec_main/runs/attr_d/a0d/checkpoints/last.pt' 2>/dev/null | awk '{print $1}')
    if [ "$OLD_MD5" != "$NEW_MD5" ]; then echo "[relay] FATAL md5 mismatch new=$NEW_MD5"; exit 1; fi
    echo "[relay] md5 OK"
  fi
fi

echo "=== [2/4] sync code + reference predictions ($(date))"
rsync -az "$ROOT/scripts/benchmark_calibrate_eval.py" "$NEW:~/DCR/scripts/" || exit 1
rsync -az "$ROOT/scripts/slurm/bench_attr_extract.sbatch" "$ROOT/scripts/slurm/bench_attr_probe.sbatch" "$NEW:~/DCR/scripts/slurm/" || exit 1
$SSH "$NEW" 'mkdir -p $HOME/DCR/results/benchmark_ref' || exit 1
for ref in RM_L D2V_L M1_L; do
  rsync -az "$ROOT/results/benchmark/pred/$ref.parquet" "$NEW:~/DCR/results/benchmark_ref/" || exit 1
done
echo "[sync] done"

echo "=== [3/4] submit GPU extraction jobs ($(date))"
IDS=""
for ARM in a0d a17d a20d; do
  id=$(submit "ARM=$ARM" "scripts/slurm/bench_attr_extract.sbatch")
  if [ -z "$id" ]; then echo "FATAL extract submit $ARM"; exit 1; fi
  IDS="$IDS $id"; echo "[extract] $ARM job=$id"
done
IDS_TRIMMED=$(echo $IDS | xargs | tr ' ' ':')

echo "=== [4/4] submit CPU probe+calibration job ($(date))"
PID=$(submit "" "--dependency=afterok:$IDS_TRIMMED scripts/slurm/bench_attr_probe.sbatch")
if [ -z "$PID" ]; then echo "FATAL probe submit"; exit 1; fi
echo "[probe] job=$PID"
{ echo "extract: $IDS"; echo "probe: $PID"; } > "$OUT/jobs.txt"

echo "=== waiting ($(date))"
ALL="$IDS $PID"
EMPTY=0
while true; do
  NOW=$(date +%s)
  if [ $((NOW - START)) -gt 14400 ]; then echo "=== timeout 4h"; break; fi
  RAW=$($SSH "$NEW" "$RENV; echo BEGIN; squeue -h -j $(echo $ALL | xargs | tr ' ' ',') -o '%i|%T|%r' 2>/dev/null; echo END" 2>/dev/null)
  if ! printf '%s' "$RAW" | grep -q 'END'; then echo "[wait] ssh hiccup"; sleep 90; continue; fi
  BODY=$(printf '%s\n' "$RAW" | sed -n '/BEGIN/,/END/p' | grep '|')
  ACTIVE=$(printf '%s\n' "$BODY" | awk -F'|' 'NF>=2 { if ($2=="RUNNING"||$2=="CONFIGURING"||$2=="COMPLETING") n++; else if ($2=="PENDING" && $3 !~ /DependencyNeverSatisfied/) n++ } END {print n+0}')
  if [ "$ACTIVE" -eq 0 ]; then EMPTY=$((EMPTY+1)); else EMPTY=0; fi
  echo "[wait] $(date) active=$ACTIVE streak=$EMPTY"
  [ "$EMPTY" -ge 2 ] && break
  sleep 120
done

echo "=== collecting ($(date))"
$SSH "$NEW" "$RENV; sacct -n -P -j $(echo $ALL | xargs | tr ' ' ',') --format=JobID%12,JobName%20,State,ExitCode,Elapsed 2>/dev/null" > "$OUT/sacct.txt" || true
cat "$OUT/sacct.txt"
mkdir -p "$OUT"
rsync -az "$NEW:~/DCR/results/benchmark_attr/" "$OUT/" 2>/dev/null || true
{
  echo "# VSRG benchmark ablation: a0d vs a17d vs a20d (isotonic 5-fold) — $(date)"
  echo; echo "## jobs"; cat "$OUT/jobs.txt"; echo
  echo "## sacct"; echo '~~~'; cat "$OUT/sacct.txt"; echo '~~~'; echo
  echo "## eval files"
  find "$OUT/eval" -name '*.csv' -o -name '*.json' -o -name '*.md' 2>/dev/null | head -20
} > "$OUT/STATUS_BENCH.md"
echo "=== DONE $(date)"
