#!/bin/bash
# =============================================================================
#  Slurm job 2/2 - CM3P embeddings for the remaining whitelist charts.  GPU job.
#
#  Submit from the repo root, AFTER the .osu cache is populated:
#      sbatch scripts/slurm/cm3p_embed.slurm
#
#  Override anything on the command line, e.g.:
#      sbatch -p <other-queue> --time=08:00:00 --gres=gpu:1 scripts/slurm/cm3p_embed.slurm
#      INFER_BATCH=4 SKIP_PROBE=1 sbatch scripts/slurm/cm3p_embed.slurm
#
#  The job is RESUMABLE at every stage: step0c only computes ids that are absent
#  from its --out table, so if Slurm kills you at the time limit, just re-submit.
#  (That is why the per-chart work is checkpointed into the parquet, not held in RAM.)
#
#  Environment: this script assumes the compute node is OFFLINE.  Everything that
#  needs the network (pip, the CM3P checkout, the HF weights, the .osu files) must
#  already exist - see docs/cm3p_hpc_slurm.md step 0.
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
# Slurm copies the batch script into its own spool directory, so "$0" is NOT this file and
# "$(dirname "$0")" is NOT scripts/slurm/ -- resolving the repo from $0 silently breaks.
# SLURM_SUBMIT_DIR is the directory sbatch was invoked from; that is the contract here.
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO" || exit 1
if [ ! -f scripts/step0c_chart_content.py ]; then
  echo "ERROR: submit this job from the repo root (currently in $REPO)." >&2
  echo "       e.g.  cd <repo> && sbatch scripts/slurm/cm3p_embed.slurm" >&2
  exit 1
fi

# module load anaconda3 cuda/12.4      # <- uncomment / adapt for your site
PY="${PY:-python}"

# ---- offline mode: never let a worker reach for the network -----------------
export HF_HOME="${HF_HOME:-$REPO/data/raw/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CM3P_ROOT="${CM3P_ROOT:-$REPO/external/cm3p}"
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

OSU_CACHE="${OSU_CACHE:-$REPO/data/raw/osu_cache}"
PROBE_OUT=data/processed/chart_content_cm3p_probe.parquet
ADD_OUT="${ADD_OUT:-data/processed/chart_content_cm3p_add.parquet}"
MIX_JSON=data/processed/cm3p_mix_report.json
INFER_BATCH="${INFER_BATCH:-8}"

echo "=== job $SLURM_JOB_ID on $(hostname) ==="
date
echo "repo        $REPO"
echo "python      $($PY -V 2>&1)"
echo "HF_HOME     $HF_HOME (offline=$HF_HUB_OFFLINE)"
echo "CM3P_ROOT   $CM3P_ROOT"
echo "osu cache   $($OSU_CACHE)  -> $(ls "$OSU_CACHE"/*.osu 2>/dev/null | wc -l) files"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

# ---- slider: not on PyPI under this name, and compute nodes cannot reach GitHub ----
# CM3P imports it at module level (it IS the .osu parser), so a missing slider is fatal.
# The wheel is vendored in vendor/ -- see vendor/README.md.
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
    echo "       See vendor/README.md (the folder upload must include vendor/)." >&2
    exit 1
  fi
fi

# ---- preflight: fail loudly in 10 s instead of 10 min downstream ------------
$PY - <<'EOF' || exit 1
import os, sys
sys.path.insert(0, os.environ["CM3P_ROOT"])
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
assert torch.cuda.is_available(), "no GPU visible: check --gres=gpu:1 and the site's cuda module"
from cm3p.processing_cm3p import CM3PProcessor           # noqa: F401  (pulls in slider)
from cm3p.modeling_cm3p import CM3PModel                 # noqa: F401
print("cm3p import OK")
missing = [p for p in ("manifests/cm3p_todo_1k.csv", "manifests/cm3p_probe_1k.npz",
                       "scripts/step0c_chart_content.py") if not os.path.exists(p)]
assert not missing, f"missing inputs: {missing} -- submit from the repo root"
EOF

# ---- stage 0 (optional): fetch the .osu files from the compute node ----------
# Only set FETCH_FIRST=1 if the compute nodes DO have outbound network.  If they do not
# (the usual case), fetch on the login node first -- see docs/cm3p_hpc_slurm.md section 2.
if [ "${FETCH_FIRST:-0}" = "1" ]; then
  echo; echo "=== stage 0: fetching .osu (FETCH_FIRST=1) ==="
  "$PY" scripts/step0d_fetch_osu.py --ids-file manifests/cm3p_todo_1k.csv \
      --out-dir "$OSU_CACHE" --workers "${WORKERS:-4}" || exit 1
  "$PY" scripts/step0d_fetch_osu.py --ids-file manifests/cm3p_probe_1k.csv \
      --out-dir "$OSU_CACHE" --workers "${WORKERS:-4}" || exit 1
fi

n_osu="$(ls "$OSU_CACHE"/*.osu 2>/dev/null | wc -l)"
if [ "$n_osu" -lt 100 ]; then
  cat >&2 <<EOF
ERROR: only $n_osu .osu files in $OSU_CACHE -- nothing to encode yet.

The .osu bodies are NOT in the uploaded folder (data/ is a skeleton on purpose).
Do ONE of these, then re-submit:
  a) compute nodes have internet:  FETCH_FIRST=1 sbatch scripts/slurm/cm3p_embed.slurm
  b) they do not (usual case):     run this on the LOGIN node, then re-submit
       python scripts/step0d_fetch_osu.py --ids-file manifests/cm3p_todo_1k.csv  --out-dir $OSU_CACHE --workers 4
       python scripts/step0d_fetch_osu.py --ids-file manifests/cm3p_probe_1k.csv --out-dir $OSU_CACHE --workers 4
EOF
  exit 1
fi

# ---- stage 1: the mix gate (256 charts, ~1 min) -----------------------------
if [ "${SKIP_PROBE:-0}" != "1" ]; then
  echo; echo "=== stage 1/2: PROBE (no-audio vs published vectors) ==="
  "$PY" scripts/step0c_chart_content.py --source osu-dir \
      --ids-file manifests/cm3p_probe_1k.csv --osu-dir "$OSU_CACHE" --audio off \
      --infer-batch "$INFER_BATCH" --out "$PROBE_OUT" || exit 1

  "$PY" scripts/step0e_validate_mix.py --computed "$PROBE_OUT" --json "$MIX_JSON" || exit 1

  code="$("$PY" -c "import json;print(json.load(open('$MIX_JSON'))['code'])")"
  echo ">>> mix verdict: $code"
  if [ "$code" = "do-not-mix" ]; then
    echo ">>> STOPPING: the no-audio rows may not be mixed with the published table."
    echo ">>> See docs/cm3p_remote_runbook.md section 4 for the two ways out."
    exit 2
  fi
else
  echo "SKIP_PROBE=1 -- skipping the mix gate (not recommended)"
fi

# ---- stage 2: the remaining 10,072 charts ----------------------------------
echo; echo "=== stage 2/2: FULL RUN -> $ADD_OUT ==="
"$PY" scripts/step0c_chart_content.py --source osu-dir \
    --ids-file manifests/cm3p_todo_1k.csv --osu-dir "$OSU_CACHE" --audio off \
    --infer-batch "$INFER_BATCH" --out "$ADD_OUT" || exit 1

echo; echo "=== done ==="
date
ls -la "$ADD_OUT" "$MIX_JSON" 2>/dev/null || true
echo
echo "bring these two files back to the machine that has the 11,877 published rows,"
echo "then merge there:"
echo "    python scripts/step0c_chart_content.py --source table --in <ADD_OUT> --tag 1k"
