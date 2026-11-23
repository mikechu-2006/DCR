#!/bin/bash
# =============================================================================
#  Finish the audio gap with the OFFICIAL osu! download (session cookie).
#
#  Context: HPC is NOT IP-blocked.  A single request with cookie + Referer
#  returns 302 -> bm*.ppy.sh in ~3 s; running 3 in parallel made the site stop
#  answering (per-IP concurrency), so this driver is SEQUENTIAL with a small
#  delay.  ~775 mapsets, estimated 5-10 h.
#
#  1. stop leftover fetches (osu.direct quota rounds / parallel tests)
#  2. recompute the missing charts
#  3. official sequential fetch -> data/todo_audio_fallback
#  4. re-run the embedding array (resume) and assemble the parquet
#
#      nohup bash scripts/cm3p_osu_fetch_finish.sh > logs/osu_fetch_finish.log 2>&1 &
# =============================================================================
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO" || exit 1
PY=${PY:-$HOME/venvs/cm3p/bin/python}
IDS=${IDS_CSV:-data/processed/cm3p_ids_todo_2786.csv}
HF_OUT=${HF_OUT:-data/todo_audio_hf}
FB_OUT=${FB_OUT:-data/todo_audio_fallback}
MISS=${MISS:-results/cm3p_todo_official/missing_now.csv}
WORKERS=${WORKERS:-1}
DELAY=${DELAY:-1.0}
export PATH=/opt/slurm/bin:$PATH
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export LD_LIBRARY_PATH=/hpc2ssd/softwares/anaconda3/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1

say() { echo; echo "=== $(date '+%F %T') $*"; }

count_missing() {
  "$PY" - "$IDS" "$HF_OUT" "$FB_OUT" "$MISS" <<'EOF'
import sys
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
covered = len(set(todo.beatmap_id) & audio)
print(f"covered {covered:,}/{len(todo):,}; missing {len(miss):,} "
      f"({miss.beatmapset_id.nunique():,} mapsets)")
EOF
}

# 1. leftover quota fetch (harmless if absent)
pkill -f cm3p_todo_continue.sh 2>/dev/null || true
pkill -f cm3p_probe_fetch_audio.py 2>/dev/null || true
pkill -f "cm3p_fetch_audio_osu.py.*workers 3" 2>/dev/null || true
sleep 2

say "coverage before"
count_missing

say "official sequential fetch (workers=$WORKERS delay=$DELAY)"
"$PY" scripts/cm3p_fetch_audio_osu.py --sample "$MISS" --osu-dir data/osu_cache \
    --out-dir "$FB_OUT" --session-file ~/.osu_session \
    --workers "$WORKERS" --delay "$DELAY" --timeout 180 2>&1 | tail -40

say "coverage after the official fetch"
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
