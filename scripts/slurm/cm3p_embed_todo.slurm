#!/bin/bash
# =============================================================================
#  Slurm array - audio-on CM3P embeddings for the whole todo manifest.
#
#  Config = the one validated by the official probe (mean cos 0.9975):
#  audio on, checkpoint processor defaults (hitspeed/kiai/SV on, snapping off),
#  16 s / 16 s windows, max_length 2000, per-window normalize + mean + renormalize.
#
#  Each task writes results/cm3p_todo_official/shards/shard_XX.jsonl with the
#  512-d vector per chart; a killed task resumes from its own file.
#
#  After the array:
#      python scripts/cm3p_embed_todo.py --assemble \
#          --in-dir results/cm3p_todo_official/shards \
#          --out-parquet data/processed/chart_content_cm3p_local_audio.parquet
#
#  Submit:  sbatch scripts/slurm/cm3p_embed_todo.slurm
# =============================================================================
#SBATCH -p i64m512u
#SBATCH -J cm3p-todo
#SBATCH -o slurm_cm3p_todo_%A_%a.out
#SBATCH -e slurm_cm3p_todo_%A_%a.err
#SBATCH -n 16
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --array=0-15%8

set -uo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO" || exit 1
PY=${PY:-$HOME/venvs/cm3p/bin/python}
IDS=${IDS_CSV:-data/processed/cm3p_ids_todo_2786.csv}
AUDIO_DIRS=${AUDIO_DIRS:-data/todo_audio_hf,data/todo_audio_fallback}
MODEL_DIR=${MODEL_DIR:-data/raw/cm3p_model_26cc694}
OUTDIR=${OUTDIR:-results/cm3p_todo_official/shards}
NSHARDS=${NUM_SHARDS:-16}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export USE_TF=0 USE_FLAX=0
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-16}
export LD_LIBRARY_PATH=/hpc2ssd/softwares/anaconda3/lib:${LD_LIBRARY_PATH:-}

echo "=== embed shard $SHARD/$NSHARDS on $(hostname) ==="
date
mkdir -p "$OUTDIR" logs
"$PY" scripts/cm3p_embed_todo.py --shard "$SHARD" --num-shards "$NSHARDS" \
    --ids-csv "$IDS" --audio-dir "$AUDIO_DIRS" --osu-dir data/osu_cache \
    --model-dir "$MODEL_DIR" --out-dir "$OUTDIR" \
    --device cpu --dtype float32 --pad-mode longest --win-batch 4 --crop legacy
rc=$?
echo "exit=$rc"
date
exit $rc
