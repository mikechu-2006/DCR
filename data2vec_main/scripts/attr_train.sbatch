#!/bin/bash
#SBATCH -J d2v_attr_train
#SBATCH -p i64m1tga800ue
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o slurm_attr_train_%j.out
#SBATCH -e slurm_attr_train_%j.err
#
# Attribute-channel data2vec continuation pretraining (Protocol S, plan section 4.4).
# Mirrors train_hpc.sbatch: one A800, 5 epochs on splits/holdout/pretrain_ids.txt,
# initialised from the 8ch holdout checkpoint with --init-from ... --expand-channels.
#
# Env vars (submit from ~/DCR):
#   ARM         required, arm name; default config is configs/audio_style_<ARM>.yaml
#               e.g. ARM=a12 / a20 / a27 / a0cont
#   CONFIG      default configs/audio_style_${ARM}.yaml
#   INIT_FROM   optional checkpoint to continue from; when non-empty the run adds
#               --init-from "$INIT_FROM" --expand-channels (zero-init new channels)
#   EPOCHS      default 5   (steps are THIS run's epochs; --init-from is not --resume)
#   LR          optional -> --lr (Protocol S uses 5e-5; empty = config value)
#   NUM_WORKERS default 6
#   IDS_FILE    default splits/holdout/pretrain_ids.txt
#   OUT_DIR     default runs/attr/${ARM}
#   EXTRA       optional extra args, word-split, e.g. EXTRA="--batch-size 8 --log-every 20"
#
# Examples:
#   sbatch --export=ALL,ARM=a0cont,INIT_FROM=runs/main_holdout/checkpoints/last.pt \
#          data2vec_main/scripts/attr_train.sbatch
#   sbatch --export=ALL,ARM=a20,INIT_FROM=runs/main_holdout/checkpoints/last.pt,LR=5e-5 \
#          data2vec_main/scripts/attr_train.sbatch
#
# No --resume on purpose: --init-from starts a fresh optimizer/LR schedule, while
# --resume would continue the old step counter (wrong total_steps for a 5-epoch run).
set -euo pipefail

ARM="${ARM:?set ARM=a12|a20|a27|a0cont (expects configs/audio_style_<ARM>.yaml)}"
CONFIG="${CONFIG:-configs/audio_style_${ARM}.yaml}"
INIT_FROM="${INIT_FROM:-}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-}"
NUM_WORKERS="${NUM_WORKERS:-6}"
IDS_FILE="${IDS_FILE:-splits/holdout/pretrain_ids.txt}"
OUT_DIR="${OUT_DIR:-runs/attr/${ARM}}"
EXTRA="${EXTRA:-}"

cd "$HOME/DCR/data2vec_main"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
PY=$HOME/.conda/envs/DCR/bin/python

# Log file names carry the ARM so 4 arms submitted the same night stay legible;
# SLURM's own -o/-e files are still keyed by job id.
LOG="slurm_attr_train_${ARM}_${SLURM_JOB_ID:-local}.out"
exec > >(tee -a "$LOG") 2>&1

echo "[attr_train] host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-local}"
echo "[attr_train] arm=$ARM config=$CONFIG epochs=$EPOCHS lr=${LR:-<config>} workers=$NUM_WORKERS"
echo "[attr_train] init_from=${INIT_FROM:-<none>} ids=$IDS_FILE out=$OUT_DIR"

test -f "$CONFIG" || { echo "[attr_train] missing config: $CONFIG"; ls configs; exit 3; }
test -f "$IDS_FILE" || { echo "[attr_train] missing ids file: $IDS_FILE"; exit 3; }
if [ -n "$INIT_FROM" ] && [ ! -f "$INIT_FROM" ]; then
  echo "[attr_train] missing INIT_FROM ckpt: $INIT_FROM"; exit 3
fi

# The attribute channels do not change n_frames (plan C4), so the existing 8ch
# chart_index_100.json is still valid.  Build it only if absent (about 4 min,
# normally already built once on the login node); never rebuild inside the job.
if [ ! -f cache/chart_index_100.json ]; then
  echo "[attr_train] index cache missing -> building cache/chart_index_100.json"
  $PY -m d2v_main.index_cache --osu-dir data/osu_cache --frame-hz 100
else
  echo "[attr_train] index cache present: cache/chart_index_100.json"
fi

ARGS=(--config "$CONFIG"
      --ids-file "$IDS_FILE"
      --out-dir "$OUT_DIR"
      --device cuda
      --epochs "$EPOCHS"
      --num-workers "$NUM_WORKERS")
[ -n "$LR" ] && ARGS+=(--lr "$LR")
if [ -n "$INIT_FROM" ]; then
  ARGS+=(--init-from "$INIT_FROM" --expand-channels)
fi
# EXTRA is deliberately word-split (it is a multi-arg escape hatch).
# shellcheck disable=SC2206
[ -n "$EXTRA" ] && ARGS+=($EXTRA)

echo "[attr_train] python -m d2v_main.train ${ARGS[*]}"
$PY -m d2v_main.train "${ARGS[@]}"

echo "[attr_train] DONE arm=$ARM out=$OUT_DIR date=$(date)"
