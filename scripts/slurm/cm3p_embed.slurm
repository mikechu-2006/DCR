#!/bin/bash
# =============================================================================
#  Slurm job 2/2 - CM3P chart embeddings.  GPU job.  See docs/cm3p_hpc_slurm.md.
#
#  Submit from the repo root, with the .osu cache already in place:
#      sbatch scripts/slurm/cm3p_embed.slurm
#
#  WHAT GETS ENCODED (EMBED_MANIFEST)
#      default: manifests/whitelist_1k.csv = the WHOLE whitelist, 21,949 charts.
#      data/osu_cache now holds every one of them, so one no-audio run covers the
#      corpus with a single convention, instead of mixing the 11,877 published
#      audio-based rows with no-audio rows.  The mix gate exists to judge exactly
#      that mixture; a full re-embed never mixes, so a "do-not-mix" verdict is
#      reported but does not stop the job (GATE_STOP=1 forces a stop).
#          EMBED_MANIFEST=manifests/cm3p_todo_1k.csv sbatch ...    # gaps only
#
#  OUTPUT (paths relative to the repo root)
#      data/processed/chart_content_cm3p_local.parquet   encoded rows, source=1
#      data/processed/chart_content_cm3p_probe.parquet   the 256-chart gate
#      data/processed/cm3p_mix_report.json               the gate verdict
#      MERGE=1 additionally folds the local rows into the canonical table
#      data/processed/chart_content_cm3p.parquet (new row wins per beatmap_id).
#
#  Override anything on the command line:
#      sbatch -p <queue> --time=08:00:00 scripts/slurm/cm3p_embed.slurm
#      INFER_BATCH=4 SKIP_PROBE=1 sbatch scripts/slurm/cm3p_embed.slurm
#      LIMIT=32 sbatch scripts/slurm/cm3p_embed.slurm     # first 32 charts only (debug)
#
#  RESUMABLE at every stage: step0c only computes ids that are absent from its
#  --out table, so a time-limit kill only costs the batch in flight -- resubmit
#  the very same command and it picks up where it stopped.
#
#  This file ALSO runs without Slurm (useful when sbatch itself is broken):
#      bash scripts/slurm/cm3p_embed.slurm 2>&1 | tee logs/cm3p_embed_$(date +%F).log
#  Every SLURM_* variable below is read with a default, so nothing breaks outside
#  of a job.
#
#  The compute node is assumed OFFLINE.  pip, the CM3P checkout, the HF weights
#  and the .osu files must already be on disk -- docs/cm3p_hpc_slurm.md section 0.
# =============================================================================
#SBATCH -p i64m1tga800u
#SBATCH -J cm3p-embed
#SBATCH -o slurm_cm3p_embed_%j.out
#SBATCH -e slurm_cm3p_embed_%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

set -uo pipefail
# Slurm spools the batch script elsewhere, so "$0" is NOT this file and
"$(dirname "$0")" is NOT scripts/slurm/ -- resolving the repo from $0 silently
# breaks.  SLURM_SUBMIT_DIR is the directory sbatch was invoked from; that is the
# contract here.  REPO can be passed explicitly at submit time, which is the only
# reliable override when you are not standing in the repo:
#     REPO=/scratch/$USER/DSR sbatch scripts/slurm/cm3p_embed.slurm
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO" || exit 1
echo "job starts in: $(pwd)"
if [ ! -f scripts/step0c_chart_content.py ]; then
  echo "ERROR: submit this job from the repo root (currently in $REPO)." >&2
  echo "       cd <repo> && sbatch scripts/slurm/cm3p_embed.slurm" >&2
  echo "       or  REPO=/abs/path/<repo> sbatch <abs>/scripts/slurm/cm3p_embed.slurm" >&2
  exit 1
fi

# module load anaconda3 cuda/12.4      # <- uncomment / adapt for your site
PY="${PY:-python}"

# ---- what to encode, and where the outputs go --------------------------------
TAG="${TAG:-1k}"
OSU_CACHE="${OSU_CACHE:-$REPO/data/osu_cache}"
EMBED_MANIFEST="${EMBED_MANIFEST:-manifests/whitelist_1k.csv}"
FETCH_MANIFEST="${FETCH_MANIFEST:-$EMBED_MANIFEST}"       # only used by FETCH_FIRST=1
PROBE_MANIFEST="${PROBE_MANIFEST:-manifests/cm3p_probe_1k.csv}"
EMBED_OUT="${EMBED_OUT:-data/processed/chart_content_cm3p_local.parquet}"
PROBE_OUT="${PROBE_OUT:-data/processed/chart_content_cm3p_probe.parquet}"
MIX_JSON="${MIX_JSON:-data/processed/cm3p_mix_report.json}"
CANONICAL="${CANONICAL:-data/processed/chart_content_cm3p.parquet}"
INFER_BATCH="${INFER_BATCH:-8}"
WORKERS="${WORKERS:-4}"

# ---- offline environment -----------------------------------------------------
# HF weights: prefer a cache copied INTO the folder (data/raw/hf_cache), else use
# an explicit HF_HOME, else the default ~/.cache/huggingface warmed on the login
# node (docs/cm3p_hpc_slurm.md section 1).
if [ -z "${HF_HOME:-}" ] && [ -d "$REPO/data/raw/hf_cache" ]; then
  HF_HOME="$REPO/data/raw/hf_cache"
fi
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CM3P_ROOT="${CM3P_ROOT:-$REPO/external/cm3p}"
export CM3P_ATTN="${CM3P_ATTN:-sdpa}"
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-8}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

echo "=== job ${SLURM_JOB_ID:-local (not under Slurm)} on $(hostname) ==="
date
echo "repo          $REPO"
echo "python        $($PY -V 2>&1)"
echo "HF_HOME       $HF_HOME (offline=$HF_HUB_OFFLINE)"
echo "CM3P_ROOT     $CM3P_ROOT   attn=$CM3P_ATTN"
echo "encode        $EMBED_MANIFEST  ->  $EMBED_OUT"
echo "osu cache     $OSU_CACHE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || true

# ---- slider: not on PyPI under this name, and compute nodes cannot reach GitHub
# CM3P imports it at module level (it IS the .osu parser), so a missing slider is
# fatal.  The wheel is vendored in vendor/ -- see vendor/README.md.
if ! "$PY" -c "import slider" >/dev/null 2>&1; then
  WHL="$(ls vendor/slider-*.whl 2>/dev/null | head -1)"
  if [ -n "$WHL" ]; then
    echo "slider missing -> installing from $WHL"
    if "$PY" -m pip install --user --no-index --no-deps "$WHL" >/dev/null 2>&1; then
      echo "  pip install OK"
    else
      echo "  pip unavailable/read-only home -> unpacking the wheel onto PYTHONPATH"
      "$PY" - "$WHL" <<'PYEOF'
import sys, zipfile
zipfile.ZipFile(sys.argv[1]).extractall("vendor/_unpacked")
PYEOF
      export PYTHONPATH="$PWD/vendor/_unpacked:${PYTHONPATH:-}"
      echo "  PYTHONPATH=$PYTHONPATH"
    fi
  else
    echo "ERROR: slider is missing and vendor/slider-*.whl was not found." >&2
    echo "       The copied folder must include vendor/ (see vendor/README.md)." >&2
    exit 1
  fi
fi

# ---- preflight A: the imports step0c will actually make (10 s, not 10 min) ---
"$PY" - <<'EOF' || exit 1
import os, sys
sys.path.insert(0, os.environ["CM3P_ROOT"])
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
assert torch.cuda.is_available(), "no GPU visible: check --gres=gpu:1 and the site cuda module"
import transformers
print("transformers", transformers.__version__, transformers.__file__)
try:
    from transformers.processing_utils import CommonKwargs, AudioKwargs   # noqa: F401
except ImportError as exc:
    print("FATAL: transformers", transformers.__version__, "cannot work with CM3P:", exc)
    print("      CommonKwargs exists only in transformers 4.42-4.57:")
    print("        <=4.41 has no such class;  >=5.0 removed it.  Both fail identically.")
    print("fix:  python -m pip install -U transformers==4.55.0   (same env as PY)")
    print("      then check where transformers.__file__ points; ~/.local can shadow conda")
    raise SystemExit(1)
from cm3p.processing_cm3p import CM3PProcessor           # noqa: F401  (pulls in slider)
from cm3p.modeling_cm3p import CM3PModel                 # noqa: F401
# step0c ALSO imports upstream BeatmapFilesDataset, which reaches config.py -> hydra
# at module level.  A preflight is only worth its 10 seconds if it makes the same
# imports the job makes: without this line a missing hydra-core surfaces ~10 min
# later, inside stage 2, as "No module named hydra" (found the hard way).
try:
    from utils.beatmap_files_dataset import BeatmapFilesDataset   # noqa: F401
except ImportError as exc:
    print("FATAL: cannot import the CM3P dataset:", exc)
    if "hydra" in str(exc):
        print("      CM3P config.py imports hydra at module level (it only builds the")
        print("      train.py ConfigStore) and utils/mmrs_dataset.py pulls it in, so the")
        print("      embed path needs it even though Hydra itself is never used.")
        print("fix:  python -m pip install hydra-core>=1.3,<1.4")
    print("      full list: requirements-cm3p.txt / docs/cm3p_hpc_slurm.md section 1")
    raise SystemExit(1)
print("imports OK (cm3p + hydra + slider reachable)")
EOF

# ---- preflight B: corpus, weights and probe inputs must already be on disk ---
"$PY" - "$EMBED_MANIFEST" "$OSU_CACHE" "$PROBE_MANIFEST" <<'EOF' || exit 1
import csv, os, pathlib, sys

want_path, cache, probe_path = sys.argv[1], sys.argv[2], sys.argv[3]

def n_ids(path):
    try:
        with open(path, newline="") as fh:
            return sum(1 for r in csv.reader(fh) if r and r[0].strip().isdigit())
    except OSError:
        return 0

if not os.path.exists(want_path):
    raise SystemExit(f"ERROR: {want_path} not found -- the copied folder must include manifests/")

have = {p[:-4] for p in os.listdir(cache) if p.endswith(".osu")} if os.path.isdir(cache) else set()
with open(want_path, newline="") as fh:
    want = {r[0].strip() for r in csv.reader(fh) if r and r[0].strip().isdigit()}
missing = want - have
print(f"targets       {len(want):,} charts; {len(have):,} .osu in the cache; "
      f"{len(missing):,} of the targets missing")
if not have:
    raise SystemExit("""ERROR: the .osu cache is empty.
  * data/osu_cache (~1.2 GB, 21,949 files) must be part of the copied folder, or
  * fetch it on a node that has network:
      python scripts/step0d_fetch_osu.py --ids-file manifests/whitelist_1k.csv --out-dir data/osu_cache --workers 4 --delay 0.25
    (or set FETCH_FIRST=1 when the compute node itself has network)""")
if missing:
    print(f"  WARNING: {len(missing):,} target charts have no .osu and will be skipped")

hf = pathlib.Path(os.environ["HF_HOME"]) / "hub" / "models--OliBomby--CM3P"
if not hf.is_dir():
    raise SystemExit(f"""ERROR: the CM3P weights are not under {hf}.
  Compute nodes are offline, so warm the cache once on the LOGIN node:
      export CM3P_ROOT=$PWD/external/cm3p
      python -c "import os,sys;sys.path.insert(0,os.environ['CM3P_ROOT']);from cm3p.modeling_cm3p import CM3PModel;CM3PModel.from_pretrained('OliBomby/CM3P',trust_remote_code=True)"
  or copy a warmed ~/.cache/huggingface into data/raw/hf_cache before copying
  the folder.  docs/cm3p_hpc_slurm.md section 1 has both recipes.""")

if not os.path.exists(probe_path):
    raise SystemExit(f"ERROR: {probe_path} not found -- manifests/ must be copied")
print("preflight OK: corpus, weights and probe inputs are in place")
EOF

# ---- stage 0 (optional): fetch the .osu files on the compute node ------------
# Only with FETCH_FIRST=1, i.e. when the compute nodes DO have outbound network.
# The normal path is that data/osu_cache travelled with the copied folder.
if [ "${FETCH_FIRST:-0}" = "1" ]; then
  echo; echo "=== stage 0: fetching .osu (FETCH_FIRST=1, $FETCH_MANIFEST) ==="
  "$PY" scripts/step0d_fetch_osu.py --ids-file "$FETCH_MANIFEST" \
      --out-dir "$OSU_CACHE" --workers "$WORKERS" --delay 0.25 || exit 1
fi

# ---- how the mix verdict is treated ------------------------------------------
# A PARTIAL embed manifest is meant to be merged with the published rows, so a
# "do-not-mix" verdict must stop the job.  A FULL re-embed replaces every row and
# never mixes, so the verdict is reported but the run continues.  GATE_STOP=0/1
# overrides the automatic choice.
full_run="$("$PY" - "$EMBED_MANIFEST" <<'EOF'
import csv, sys

def n(path):
    try:
        with open(path, newline="") as fh:
            return sum(1 for r in csv.reader(fh) if r and r[0].strip().isdigit())
    except OSError:
        return 0

print(1 if n(sys.argv[1]) >= n("manifests/whitelist_1k.csv") > 0 else 0)
EOF
)"
if [ -z "${GATE_STOP:-}" ]; then
  if [ "$full_run" = "1" ]; then GATE_STOP=0; else GATE_STOP=1; fi
fi
if [ "$full_run" = "1" ]; then
  echo "scope         whole whitelist -> nothing is mixed, GATE_STOP=$GATE_STOP"
else
  echo "scope         subset -> merged with the published rows, GATE_STOP=$GATE_STOP"
fi

# ---- stage 1: the mix gate (256 charts, ~1 min on a GPU) ---------------------
# Re-encode 256 charts that HAVE published vectors with the same no-audio path the
# full run uses, then compare geometries (docs/cm3p_remote_runbook.md section 4).
if [ "${SKIP_PROBE:-0}" != "1" ]; then
  echo; echo "=== stage 1/3: PROBE (no-audio vs published vectors) ==="
  "$PY" scripts/step0c_chart_content.py --source osu-dir \
      --ids-file "$PROBE_MANIFEST" --osu-dir "$OSU_CACHE" --audio off \
      --infer-batch "$INFER_BATCH" --out "$PROBE_OUT" || exit 1

  "$PY" scripts/step0e_validate_mix.py --computed "$PROBE_OUT" --json "$MIX_JSON" || exit 1

  code="$("$PY" -c "import json;print(json.load(open('$MIX_JSON'))['code'])")"
  echo ">>> mix verdict: $code"
  if [ "$code" = "do-not-mix" ] && [ "$GATE_STOP" = "1" ]; then
    echo ">>> STOPPING: the no-audio rows may not be mixed with the published table."
    echo ">>> See docs/cm3p_remote_runbook.md section 4 for the two ways out."
    exit 2
  fi
else
  echo "SKIP_PROBE=1 -- skipping the mix gate (not recommended)"
fi

# ---- stage 2: encode EMBED_MANIFEST ------------------------------------------
echo; echo "=== stage 2/3: FULL RUN ($EMBED_MANIFEST) -> $EMBED_OUT ==="
# LIMIT=N is a debug knob: encode only the first N targets (handy for a first run
# on a new cluster -- same code path, just fewer batches).  The partial table it
# writes is a valid checkpoint, so a later full run resumes from it.
LIMIT_ARGS=""
if [ "${LIMIT:-0}" != "0" ]; then
  echo "LIMIT=$LIMIT -- debugging on the first $LIMIT charts only"
  LIMIT_ARGS="--limit $LIMIT"
fi
"$PY" scripts/step0c_chart_content.py --source osu-dir \
    --ids-file "$EMBED_MANIFEST" --osu-dir "$OSU_CACHE" --audio off \
    --infer-batch "$INFER_BATCH" --out "$EMBED_OUT" $LIMIT_ARGS || exit 1

# ---- stage 3 (optional): fold the local rows into the canonical table --------
if [ "${MERGE:-0}" = "1" ]; then
  echo; echo "=== stage 3/3: MERGE $EMBED_OUT -> $CANONICAL (new row wins) ==="
  "$PY" scripts/step0c_chart_content.py --source table --in "$EMBED_OUT" \
      --tag "$TAG" --out "$CANONICAL" || exit 1
else
  echo; echo "MERGE=0 -- the canonical table was left alone.  To fold it in later:"
  echo "    python scripts/step0c_chart_content.py --source table --in $EMBED_OUT --tag $TAG"
fi

# ---- summary -----------------------------------------------------------------
echo; echo "=== done ==="
date
"$PY" - "$EMBED_OUT" "$EMBED_MANIFEST" <<'EOF'
import sys
import pandas as pd

out = pd.read_parquet(sys.argv[1], columns=["beatmap_id", "source", "n_windows", "checksum_ok"])
want = pd.read_csv(sys.argv[2], usecols=["beatmap_id"])
ids = set(want.beatmap_id.astype("int64").tolist())
got = set(out.beatmap_id.astype("int64").tolist())
print(f"{sys.argv[1]}: {len(out):,} rows, {out.beatmap_id.nunique():,} unique charts")
print(f"coverage of {sys.argv[2]}: {len(got & ids):,}/{len(ids):,} = {len(got & ids) / max(len(ids), 1):.2%}")
print("source counts:", {int(k): int(v) for k, v in out.source.value_counts().items()})
bad = int(out.checksum_ok.eq(False).sum())
if bad:
    print(f"checksum_ok=False: {bad:,} (a later .osu revision -- kept and flagged)")
missing = sorted(ids - got)
if missing:
    print(f"missing {len(missing):,} target ids, e.g. {missing[:10]} (no .osu in the cache)")
EOF
echo
echo "next steps:"
echo "  * bring back $EMBED_OUT and $MIX_JSON"
echo "  * use it directly:  step2 --chart-content cm3p --content-path $EMBED_OUT"
echo "  * or fold it in:    python scripts/step0c_chart_content.py --source table --in $EMBED_OUT --tag $TAG"

