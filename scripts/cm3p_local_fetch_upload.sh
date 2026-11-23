#!/bin/bash
# =============================================================================
#  LOCAL-side driver: download the missing mapset audio with the user's own
#  osu! session, keep ONLY the audio, and rsync it to the HPC in batches.
#
#  Why local: after a few requests the HPC IP is throttled by osu.ppy.sh
#  (TLS handshake timeouts), while the user's network works.  Parallel requests
#  trigger the same throttle even locally, so this is strictly sequential.
#
#  Disk safety: .osz is parsed in memory; only <beatmap_id>.<ext> audio is
#  written (~3-5 MB/set).  Each batch is uploaded and then deleted locally, so
#  the peak local footprint is one batch (~500 MB).
#
#      bash scripts/cm3p_local_fetch_upload.sh
# =============================================================================
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || exit 1
PY=.cache/cm3p_venv/bin/python
MISS=results/cm3p_todo_official/missing_for_local.csv
OUT=.cache/osu_audio
DONE=$OUT/done_ids.txt
BATCH=${BATCH:-150}
LOG=results/cm3p_todo_official/local_fetch.log
HPC=aiaa3111_l01_008@hpc2login.hpc.hkust-gz.edu.cn
HPC_DIR=/hpc2hdd/home/aiaa3111_l01_008/DCR/data/todo_audio_fallback
mkdir -p "$OUT" results/cm3p_todo_official
touch "$DONE"

say() { echo; echo "=== $(date '+%F %T') $*"; }

empty=0
while true; do
  free_mb=$(df -m . | awk 'NR==2 {print $4}')
  if [ "$free_mb" -lt 2000 ]; then
    say "only ${free_mb} MB free -> stop before filling the disk"
    break
  fi
  "$PY" - "$MISS" "$DONE" "$OUT/batch.csv" "$BATCH" <<'EOF'
import sys
import pandas as pd
miss_csv, done_txt, out_csv, batch = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
done = set()
try:
    with open(done_txt) as fh:
        done = {int(x) for x in fh.read().split() if x.strip().isdigit()}
except FileNotFoundError:
    pass
df = pd.read_csv(miss_csv)
df = df[~df.beatmap_id.isin(done)].head(batch)
df.to_csv(out_csv, index=False)
print(f"batch {len(df)} chart(s), {df.beatmapset_id.nunique()} set(s)")
EOF
  n=$(( $(wc -l < "$OUT/batch.csv") - 1 ))
  if [ "$n" -le 0 ]; then say "no pending charts"; break; fi

  say "downloading $n chart(s) (sequential)"
  "$PY" scripts/cm3p_fetch_audio_osu.py --sample "$OUT/batch.csv" --out-dir "$OUT" \
      --session-file .cache/osu_session --workers 1 --delay 0.3 --timeout 180 \
      >> "$LOG" 2>&1
  rc=$?

  new=$("$PY" - "$OUT/manifest_osu.jsonl" "$DONE" <<'EOF'
import json, sys
man, done_txt = sys.argv[1], sys.argv[2]
ok = []
try:
    for line in open(man, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("ok"):
                ok.append(int(r["beatmap_id"]))
except FileNotFoundError:
    pass
done = set()
try:
    done = {int(x) for x in open(done_txt).read().split() if x.strip().isdigit()}
except FileNotFoundError:
    pass
new = sorted(set(ok) - done)
with open(done_txt, "a") as fh:
    for x in new:
        fh.write(f"{x}\n")
print(len(ok), len(new))
EOF
)
  say "manifest ok/new: $new (rc=$rc)"

  n_audio=$(find "$OUT" -maxdepth 1 -type f \( -name '*.mp3' -o -name '*.ogg' -o -name '*.wav' -o -name '*.flac' \) | wc -l)
  say "uploading $n_audio audio file(s) to HPC"
  rsync -a --ignore-existing \
      --include='*.mp3' --include='*.ogg' --include='*.wav' --include='*.flac' --exclude='*' \
      "$OUT/" "$HPC:$HPC_DIR/" || { say "rsync failed"; sleep 30; }
  find "$OUT" -maxdepth 1 -type f \( -name '*.mp3' -o -name '*.ogg' -o -name '*.wav' -o -name '*.flac' \) -delete
  say "local audio cleared"

  if echo "$new" | awk '{exit !($2 == 0)}'; then
    empty=$((empty + 1))
  else
    empty=0
  fi
  if [ "$empty" -ge 2 ]; then say "two empty batches -> stop"; break; fi
done

say "starting HPC embedding + assembly"
ssh "$HPC" "cd /hpc2hdd/home/aiaa3111_l01_008/DCR && setsid nohup bash scripts/cm3p_embed_finish.sh > logs/embed_finish.log 2>&1 < /dev/null &"
say "done (watch logs/embed_finish.log on the HPC)"
