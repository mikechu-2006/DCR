#!/bin/bash
# =============================================================================
#  One-shot driver for the 2,786-chart audio-on embedding run.
#
#  Stages (all on the HPC; nothing is stored locally):
#    1. stream-scan the 74 shards of project-riz/osu-beatmaps in two disjoint
#       loops (odd/even), keeping only the audio of samples that contain a
#       wanted beatmap id.  Resumable: done_<shard>.json skips finished shards.
#    2. hardlink <beatmap_id>.<ext> for every manifest entry.
#    3. compute the missing charts; fetch their mapsets from osu.direct into
#       data/todo_audio_fallback (per-IP quota; whatever fits is fetched).
#    4. submit the embedding array (scripts/slurm/cm3p_embed_todo.slurm).
#
#  Run detached from the login node:
#      nohup bash scripts/cm3p_todo_pipeline.sh > logs/todo_pipeline.log 2>&1 &
# =============================================================================
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO" || exit 1
PY=${PY:-$HOME/venvs/cm3p/bin/python}
IDS=${IDS_CSV:-data/processed/cm3p_ids_todo_2786.csv}
HF_OUT=${HF_OUT:-data/todo_audio_hf}
FB_OUT=${FB_OUT:-data/todo_audio_fallback}
MISS=${MISS:-results/cm3p_todo_official/missing.csv}
export PATH=/opt/slurm/bin:$PATH
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export LD_LIBRARY_PATH=/hpc2ssd/softwares/anaconda3/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1

say() { echo; echo "=== $(date '+%F %T') $*"; }

say "stage 1/4: scan shards (2 disjoint loops)"
mkdir -p "$HF_OUT" logs
for start in 1 2; do
  nohup bash -c "for i in \$(seq $start 2 73); do       $PY scripts/cm3p_fetch_audio_hf.py --shard \$i --ids-csv $IDS           --out-dir $HF_OUT --work-dir /tmp/hf_shard --mode stream       || echo FAIL \$i; done" > "logs/hf_loop_$start.log" 2>&1 &
done
# wait for both loops (their python children finish before the loops do)
while ps -eo cmd | grep -q "[c]m3p_fetch_audio_hf.py --shard"; do sleep 60; done
say "scan loops finished: $(ls "$HF_OUT"/done_*.json 2>/dev/null | wc -l) shard(s) done"

say "stage 2/4: hardlink beatmap audio"
"$PY" scripts/cm3p_fetch_audio_hf.py --link --out-dir "$HF_OUT"

say "stage 3/4: missing charts -> osu.direct fallback"
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
print(f"found {len(found):,}/{len(todo):,} charts in the HF dataset; "
      f"missing {len(missing):,} charts / {missing.beatmapset_id.nunique():,} set(s)")
EOF
if [ -s "$MISS" ]; then
  "$PY" scripts/cm3p_probe_fetch_audio.py --sample "$MISS" --osu-dir data/osu_cache \
      --out-dir "$FB_OUT" --workers 2 2>&1 | tail -20
fi

say "stage 4/4: submit embedding array"
sbatch scripts/slurm/cm3p_embed_todo.slurm
say "done (watch: squeue -u $USER; results: results/cm3p_todo_official/shards/)"
