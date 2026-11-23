#!/bin/bash
# =============================================================================
#  Slurm array - fetch original osu! audio for the todo charts from the
#  Hugging Face WebDataset project-riz/osu-beatmaps (no mirror quota).
#
#  One task scans one 1 GiB shard: it downloads the tar with parallel HTTP
#  Range requests, walks it, and saves the audio of every sample whose
#  beatmaps list contains a wanted beatmap id.  The tar is deleted afterwards.
#
#  After the array, run the hardlink step once (login node or a short task):
#      python scripts/cm3p_fetch_audio_hf.py --link --out-dir data/todo_audio_hf
#
#  Submit:  sbatch scripts/slurm/cm3p_fetch_hf.slurm
#  Test:    sbatch --array=0-1 --time=00:30:00 scripts/slurm/cm3p_fetch_hf.slurm
# =============================================================================
#SBATCH -p i64m512u
#SBATCH -J cm3p-hf
#SBATCH -o slurm_cm3p_hf_%A_%a.out
#SBATCH -e slurm_cm3p_hf_%A_%a.err
#SBATCH -n 8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-73%3

set -uo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO" || exit 1
PY=${PY:-$HOME/venvs/cm3p/bin/python}
IDS=${IDS_CSV:-data/processed/cm3p_ids_todo_2786.csv}
OUT=${OUT_DIR:-data/todo_audio_hf}
WORK=${WORK_DIR:-/tmp/hf_shard_${SLURM_ARRAY_TASK_ID:-0}}
WORKERS=${WORKERS:-4}
CHUNK_MB=${CHUNK_MB:-64}
NUM=${NUM_SHARDS:-74}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/hpc2ssd/softwares/anaconda3/lib:${LD_LIBRARY_PATH:-}

echo "=== shard $SHARD/$NUM on $(hostname) ==="
date
mkdir -p "$OUT" "$WORK" logs
"$PY" scripts/cm3p_fetch_audio_hf.py --shard "$SHARD" --num-shards "$NUM" \
    --ids-csv "$IDS" --out-dir "$OUT" --work-dir "$WORK" \
    --chunk-mb "$CHUNK_MB" --workers "$WORKERS"
rc=$?
echo "exit=$rc"
date
exit $rc
