#!/bin/bash
#SBATCH -J d2v_attr_extract
#SBATCH -p i64m1tga800ue
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o slurm_attr_extract_%j.out
#SBATCH -e slurm_attr_extract_%j.err
#
# Encode charts with an attribute-channel checkpoint -> 512-d chart content table
# (plan section 5.2 step 2).  One A800, about 8 min for the 6,828-chart pool.
#
# Env vars (submit from ~/DCR):
#   ARM          required, arm name -> default CONFIG/CKPT/OUT derive from it
#   CONFIG       default configs/audio_style_${ARM}.yaml (must match the ckpt)
#   CKPT         default runs/attr/${ARM}/checkpoints/last.pt
#   IDS_FILE     default splits/holdout/pretrain_ids.txt
#                (Protocol S extraction: data/processed/charts_1k.txt)
#   OUT          default ../data/processed/chart_content_d2v_${ARM}-${TAG}.parquet
#   TAG          default holdout; provenance tag written into the table (e.g. d2v_a20_v1)
#   BATCH        default 32
#   NUM_WORKERS  default 6
#
# Examples:
#   sbatch --export=ALL,ARM=a20,TAG=d2v_a20_v1,IDS_FILE=../data/processed/charts_1k.txt \
#          data2vec_main/scripts/attr_extract.sbatch
set -euo pipefail

ARM="${ARM:?set ARM=a12|a20|a27|a0cont}"
CONFIG="${CONFIG:-configs/audio_style_${ARM}.yaml}"
CKPT="${CKPT:-runs/attr/${ARM}/checkpoints/last.pt}"
IDS_FILE="${IDS_FILE:-splits/holdout/pretrain_ids.txt}"
TAG="${TAG:-holdout}"
OUT="${OUT:-../data/processed/chart_content_d2v_${ARM}-${TAG}.parquet}"
BATCH="${BATCH:-32}"
NUM_WORKERS="${NUM_WORKERS:-6}"

cd "$HOME/DCR/data2vec_main"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
PY=$HOME/.conda/envs/DCR/bin/python

LOG="slurm_attr_extract_${ARM}_${SLURM_JOB_ID:-local}.out"
exec > >(tee -a "$LOG") 2>&1

echo "[attr_extract] host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-local}"
echo "[attr_extract] arm=$ARM config=$CONFIG ckpt=$CKPT"
echo "[attr_extract] ids=$IDS_FILE out=$OUT tag=$TAG batch=$BATCH workers=$NUM_WORKERS"

test -f "$CONFIG" || { echo "[attr_extract] missing config: $CONFIG"; exit 3; }
test -f "$CKPT" || { echo "[attr_extract] missing ckpt: $CKPT"; exit 3; }
test -f "$IDS_FILE" || { echo "[attr_extract] missing ids file: $IDS_FILE"; exit 3; }
mkdir -p "$(dirname "$OUT")"

$PY -m d2v_main.extract \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  --ids-file "$IDS_FILE" \
  --out "$OUT" \
  --device cuda \
  --batch-size "$BATCH" \
  --num-workers "$NUM_WORKERS" \
  --tag "$TAG"

echo "[attr_extract] DONE out=$OUT date=$(date)"
