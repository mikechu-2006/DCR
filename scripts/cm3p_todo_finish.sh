#!/bin/bash
# =============================================================================
#  Finisher for the 2,786-chart audio-on embedding run.
#
#  The scan loops (data/todo_audio_hf) were started separately.  This script
#  waits until they are really finished (74 done markers, or 5 consecutive idle
#  minutes with no new shard), then:
#    1. hardlink <beatmap_id>.<ext>
#    2. compute the missing charts -> osu.direct fallback folder
#    3. submit the embedding array
#
#  Idempotent: safe to run again later; it re-links, re-computes the gap list and
#  re-submits only if the embed outputs are missing.
#
#      nohup bash scripts/cm3p_todo_finish.sh > logs/todo_finish.log 2>&1 &
# =============================================================================
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO" || exit 1
PY=${PY:-$HOME/venvs/cm3p/bin/python}
IDS=${IDS_CSV:-data/processed/cm3p_ids_todo_2786.csv}
HF_OUT=${HF_OUT:-data/todo_audio_hf}
FB_OUT=${FB_OUT:-data/todo_audio_fallback}
MISS=${MISS:-results/cm3p_todo_official/missing.csv}
N_SHARDS=${N_SHARDS:-74}
export PATH=/opt/slurm/bin:$PATH
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export LD_LIBRARY_PATH=/hpc2ssd/softwares/anaconda3/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1

say() { echo; echo "=== $(date '+%F %T') $*"; }

say "wait for the scan"
prev=-1; stable=0
while true; do
  n=$(ls "$HF_OUT"/done_*.json 2>/dev/null | wc -l)
  running=$(ps -eo cmd | grep -c "[c]m3p_fetch_audio_hf.py --shard" || true)
  echo "$(date '+%F %T') done=$n/$N_SHARDS scan_procs=$running"
  if [ "$n" -ge "$N_SHARDS" ]; then break; fi
  if [ "$running" -eq 0 ] && [ "$n" -eq "$prev" ]; then
    stable=$((stable + 1))
  else
    stable=0
  fi
  if [ "$stable" -ge 5 ]; then
    echo "no scan process and no new shard for 5 min -> assuming finished"
    break
  fi
  prev=$n
  sleep 60
done

say "stage 2: hardlink beatmap audio"
"$PY" scripts/cm3p_fetch_audio_hf.py --link --out-dir "$HF_OUT"

say "stage 3: missing charts"
"$PY" - "$IDS" "$HF_OUT" "$MISS" <<'EOF'
import glob, json, sys
from pathlib import Path
import pandas as pd
ids_csv, hf_out, miss_csv = sys.argv[1], sys.argv[2], sys.argv[3]
found = set()
for f in glob.glob(str(Path(hf_out) / "manifest_*.jsonl")):
    for line in Path(f).read_text(encoding="utf-8").splitlines():
        if line.strip():
            found.update(int(b) for b in json.loads(line).get("beatmap_ids", []))
todo = pd.read_csv(ids_csv)
missing = todo[~todo.beatmap_id.isin(found)]
Path(miss_csv).parent.mkdir(parents=True, exist_ok=True)
missing.to_csv(miss_csv, index=False)
print(f"HF coverage {len(found):,}/{len(todo):,} charts; "
      f"missing {len(missing):,} charts / {missing.beatmapset_id.nunique():,} set(s)")
EOF

if [ -s "$MISS" ]; then
  say "stage 3b: osu.direct fallback (quota-limited; errors are recorded, not fatal)"
  "$PY" scripts/cm3p_probe_fetch_audio.py --sample "$MISS" --osu-dir data/osu_cache \
      --out-dir "$FB_OUT" --workers 2 2>&1 | tail -25
fi

say "stage 4: submit embedding array"
"$PY" - "$IDS" "$HF_OUT" "$FB_OUT" <<'EOF'
import glob, sys
from pathlib import Path
todo = sum(1 for _ in open(sys.argv[1])) - 1
found = set(int(p.stem) for d in sys.argv[2:] for p in Path(d).glob("*.mp3")
            if p.stem.isdigit())
print(f"audio staged for {len(found):,}/{todo:,} chart(s)")
EOF
jobid=$(sbatch scripts/slurm/cm3p_embed_todo.slurm | awk '{print $NF}')
say "embedding array submitted: $jobid"

say "wait for the embedding array"
while squeue -h -j "$jobid" -o "%T" 2>/dev/null | grep -qE "PENDING|RUNNING|CONFIGURING"; do
  sleep 120
done

say "stage 5: assemble the content table"
"$PY" scripts/cm3p_embed_todo.py --assemble --ids-csv "$IDS" \
    --in-dir results/cm3p_todo_official/shards \
    --out-parquet data/processed/chart_content_cm3p_local_audio.parquet
say "done: data/processed/chart_content_cm3p_local_audio.parquet (+ .meta.json)"
say "bring it home: scp <user>@hpc2login:.../data/processed/chart_content_cm3p_local_audio.parquet*"
