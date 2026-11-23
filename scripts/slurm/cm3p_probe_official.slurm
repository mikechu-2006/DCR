#!/bin/bash
# =============================================================================
#  Slurm job - CM3P official-embedding probe (audio ON), CPU partition.
#
#  Goal: reproduce the published 512-d vectors (OliBomby/CM3P-Embeddings-244K,
#  audio on) for a sample of official charts and report the cosine.  The target
#  is >= 0.99 for every chart; the summary prints mean/median/min and counts.
#
#  The compute node is OFFLINE.  Before submitting:
#    1) stage audio on the LOGIN node (only <beatmap_id>.<ext>; no .osz kept):
#         python scripts/cm3p_probe_fetch_audio.py \
#             --sample results/cm3p_probe_official/sample.csv \
#             --out-dir data/probe_audio --workers 4
#    2) make sure the 26cc694 checkpoint snapshot is at MODEL_DIR
#       (data/raw/cm3p_model_26cc694; HF_HOME cache also works if it contains
#       the remote-code .py files -- the login-node download does).
#
#  Submit from the repo root:
#      sbatch scripts/slurm/cm3p_probe_official.slurm
#      LIMIT=5 sbatch scripts/slurm/cm3p_probe_official.slurm     # pilot
#      AUDIO=off LOCAL_TABLE=... sbatch ...                       # no-audio control
#
#  Runs without Slurm too (handy on the login node for a tiny pilot):
#      LIMIT=2 bash scripts/slurm/cm3p_probe_official.slurm
# =============================================================================
#SBATCH -p i64m512u
#SBATCH -J cm3p-probe
#SBATCH -o slurm_cm3p_probe_%j.out
#SBATCH -e slurm_cm3p_probe_%j.err
#SBATCH -n 16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -uo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO" || exit 1
echo "job starts in: $(pwd)"

PY=${PY:-$HOME/venvs/cm3p/bin/python}
SAMPLES=${SAMPLE:-results/cm3p_probe_official/sample.csv}
AUDIO_DIR=${AUDIO_DIR:-data/probe_audio}
MODEL_DIR=${MODEL_DIR:-data/raw/cm3p_model_26cc694}
OUT=${OUT:-results/cm3p_probe_official/probe.jsonl}
SUMMARY=${SUMMARY:-results/cm3p_probe_official/summary.json}
VECTORS=${VECTORS:-results/cm3p_probe_official/vectors.npz}
AUDIO=${AUDIO:-on}
LOCAL_TABLE=${LOCAL_TABLE:-}
LIMIT=${LIMIT:-0}
PAD_MODE=${PAD_MODE:-longest}
DTYPE=${DTYPE:-float32}
WIN_BATCH=${WIN_BATCH:-4}
DEVICE=${DEVICE:-cpu}
CROP=${CROP:-legacy}

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# Do not import TensorFlow/Flax on a node that merely has them installed.
export USE_TF=0 USE_FLAX=0
# The site anaconda's libstdc++ is newer than /lib/x86_64-linux-gnu's, which
# sklearn (pulled in by transformers.generation) needs.  Harmless if absent.
CONDA_LIB=${CONDA_LIB:-/hpc2ssd/softwares/anaconda3/lib}
if [ -d "$CONDA_LIB" ]; then
  export LD_LIBRARY_PATH="$CONDA_LIB:${LD_LIBRARY_PATH:-}"
fi
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-16}
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) ==="
date
echo "repo       $REPO"
echo "python     $PY ($($PY -V 2>&1))"
echo "model      $MODEL_DIR"
echo "sample     $SAMPLES"
echo "audio      $AUDIO  dir=$AUDIO_DIR  limit=$LIMIT"
echo "target     official table  (pad=$PAD_MODE dtype=$DTYPE crop=$CROP)"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true
ls -la "$MODEL_DIR" 2>/dev/null | head -5 || true
[ "$AUDIO" = on ] && { echo "staged audio: $(ls "$AUDIO_DIR" | grep -cE '^[0-9]+\.(mp3|ogg|wav|flac|m4a)$') file(s)"; }

"$PY" - <<'EOF' || exit 1
import sys
import torch
import transformers
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
assert transformers.__version__.startswith(("4.4", "4.5")), \
    f"CM3P needs transformers 4.42-4.57, found {transformers.__version__}"
EOF

LIMIT_ARGS=""
if [ "$LIMIT" != "0" ]; then
  LIMIT_ARGS="--limit $LIMIT"
  echo "LIMIT=$LIMIT -- pilot run"
fi
LOCAL_ARGS=""
if [ -n "$LOCAL_TABLE" ]; then
  LOCAL_ARGS="--local-table $LOCAL_TABLE"
fi

t0=$(date +%s)
"$PY" scripts/cm3p_official_probe.py \
    --sample "$SAMPLES" \
    --audio "$AUDIO" \
    --audio-dir "$AUDIO_DIR" \
    --model-dir "$MODEL_DIR" \
    --official-table data/processed/chart_content_cm3p.parquet \
    $LOCAL_ARGS \
    --osu-dir data/osu_cache \
    --device "$DEVICE" --dtype "$DTYPE" --pad-mode "$PAD_MODE" \
    --win-batch "$WIN_BATCH" --crop "$CROP" $LIMIT_ARGS \
    --out "$OUT" --summary "$SUMMARY" --save-vectors "$VECTORS"
rc=$?
echo "probe exit=$rc  elapsed=$(( $(date +%s) - t0 ))s"
date
exit $rc
