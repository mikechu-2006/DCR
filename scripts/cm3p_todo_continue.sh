#!/bin/bash
# =============================================================================
#  Continue the 2,786-chart run after the first pass.
#
#  First pass produced 1,634/2,786 vectors.  The gap is AUDIO, not compute:
#     * HF WebDataset covered 863 charts,
#     * osu.direct added 680 before its per-IP quota ran out,
#     * 1,151 charts (775 mapsets) still have no audio.
#
#  This driver spends the remaining osu.direct quota in bounded rounds (so a
#  round never mass-retries into a 403 storm), waits when a round makes no
#  progress, then re-runs the embedding array (resume) and re-assembles.
#
#      nohup bash scripts/cm3p_todo_continue.sh > logs/todo_continue.log 2>&1 &
# =============================================================================
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO" || exit 1
PY=${PY:-$HOME/venvs/cm3p/bin/python}
IDS=${IDS_CSV:-data/processed/cm3p_ids_todo_2786.csv}
HF_OUT=${HF_OUT:-data/todo_audio_hf}
FB_OUT=${FB_OUT:-data/todo_audio_fallback}
MISS=${MISS:-results/cm3p_todo_official/missing_now.csv}
ROUNDS=${ROUNDS:-12}
PER_ROUND=${PER_ROUND:-170}       # charts per quota round (~= mapsets)
SLEEP_EMPTY=${SLEEP_EMPTY:-21600}  # 6 h when a round stages nothing
export PATH=/opt/slurm/bin:$PATH
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export LD_LIBRARY_PATH=/hpc2ssd/softwares/anaconda3/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1

say() { echo; echo "=== $(date '+%F %T') $*"; }

count_missing() {
  "$PY" - "$IDS" "$HF_OUT" "$FB_OUT" "$MISS" <<'EOF'
import glob, json, sys
from pathlib import Path
import pandas as pd
ids_csv, hf_out, fb_out, miss_csv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
audio = set()
for d in (hf_out, fb_out):
    for p in Path(d).glob("*"):
        if p.suffix.lower() in (".mp3", ".ogg", ".wav", ".flac", ".m4a") and p.stem.isdigit():
            audio.add(int(p.stem))
todo = pd.read_csv(ids_csv)
miss = todo[~todo.beatmap_id.isin(audio)]
Path(miss_csv).parent.mkdir(parents=True, exist_ok=True)
miss.to_csv(miss_csv, index=False)
print(f"{len(audio)}/{len(todo)} charts have audio; {len(miss)} missing "
      f"({miss.beatmapset_id.nunique()} mapsets)")
EOF
}

for r in $(seq 1 "$ROUNDS"); do
  say "round $r/$ROUNDS"
  count_missing
  n=$(( $(wc -l < "$MISS") - 1 ))
  [ "$n" -le 0 ] && { echo "all charts have audio"; break; }
  before=$(ls "$FB_OUT" 2>/dev/null | grep -cE '^[0-9]+\.(mp3|ogg|wav|flac|m4a)$')
  "$PY" scripts/cm3p_probe_fetch_audio.py --sample "$MISS" --osu-dir data/osu_cache \
      --out-dir "$FB_OUT" --workers 2 --limit "$PER_ROUND" 2>&1 | tail -15
  after=$(ls "$FB_OUT" 2>/dev/null | grep -cE '^[0-9]+\.(mp3|ogg|wav|flac|m4a)$')
  echo "round $r staged $((after - before)) new audio file(s)"
  if [ "$after" -le "$before" ]; then
    echo "no progress -> waiting $SLEEP_EMPTY s for the quota window"
    sleep "$SLEEP_EMPTY"
  fi
done

say "final coverage"
count_missing

say "submit embedding array (resume; only new audio is computed)"
jobid=$(sbatch scripts/slurm/cm3p_embed_todo.slurm | awk '{print $NF}')
echo "array job: $jobid"
while squeue -h -j "$jobid" -o "%T" 2>/dev/null | grep -qE "PENDING|RUNNING|CONFIGURING"; do
  sleep 120
done

say "assemble"
"$PY" scripts/cm3p_embed_todo.py --assemble --ids-csv "$IDS" \
    --in-dir results/cm3p_todo_official/shards \
    --out-parquet data/processed/chart_content_cm3p_local_audio.parquet
say "done"
