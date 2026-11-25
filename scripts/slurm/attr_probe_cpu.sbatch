#!/bin/bash
#SBATCH -J dcr-attr-probe
#SBATCH -p i64m512u
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o slurm_attr_probe_%j.out
#SBATCH -e slurm_attr_probe_%j.err
#
# CPU rank-NLL probes on an attribute-channel content table (plan sections 5.2/5.3).
# Same runner as the existing D2V probes (scripts/run_p2_p4_multiseed.py), so each
# arm's P4/P4-hold numbers are paired-comparable with the 8ch baseline tables.
#
# Env vars (submit from ~/DCR):
#   CONTENT      required, e.g. data/processed/chart_content_d2v_a20-1k.parquet
#   PROTOCOL     default p4   (p2, p4 or p2,p4)
#   SEEDS        default 1    (e.g. "1,2,3,4,5" for the screening gate)
#   MODELS       default linear,linear_extra
#   SPLIT_P4     optional P4 split prefix, e.g. data/processed/step1_1k_pairbest_p4
#                or data/processed/step1_1k_pairbest_p4_d2vhold (P4-hold side check)
#   SPLIT_P2     optional P2 split prefix (runner default when empty)
#   SIGNAL_TABLE default data/processed/rank_signals_1k.parquet
#   OUT_DIR      required, e.g. results/d2v_attr/a20/p4
#   EXTRA        optional extra args, word-split (e.g. EXTRA="--force --quick")
#   THREADS      default 8
#
# Examples (env is exported by default; commas in SEEDS must not go through
# --export=A=1,2,3 - sbatch splits that on commas, so set vars in the shell):
#   CONTENT=data/processed/chart_content_d2v_a20-1k.parquet \
#   OUT_DIR=results/d2v_attr/a20/p4 SEEDS=1,2,3,4,5 \
#     sbatch --export=ALL scripts/slurm/attr_probe_cpu.sbatch
#   CONTENT=data/processed/chart_content_d2v_a20-1k.parquet \
#   OUT_DIR=results/d2v_attr/a20/p4hold \
#   SPLIT_P4=data/processed/step1_1k_pairbest_p4_d2vhold \
#     sbatch --export=ALL scripts/slurm/attr_probe_cpu.sbatch
set -euo pipefail

CONTENT="${CONTENT:?set CONTENT=data/processed/chart_content_d2v_<arm>-<tag>.parquet}"
OUT_DIR="${OUT_DIR:?set OUT_DIR=results/d2v_attr/<arm>/<probe>}"
PROTOCOL="${PROTOCOL:-p4}"
SEEDS="${SEEDS:-1}"
MODELS="${MODELS:-linear,linear_extra}"
SPLIT_P4="${SPLIT_P4:-}"
SPLIT_P2="${SPLIT_P2:-}"
SIGNAL_TABLE="${SIGNAL_TABLE:-data/processed/rank_signals_1k.parquet}"
EXTRA="${EXTRA:-}"
THREADS="${THREADS:-8}"

cd "$HOME/DCR"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
PY=$HOME/.conda/envs/DCR/bin/python

echo "[attr_probe] host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-local}"
echo "[attr_probe] content=$CONTENT protocol=$PROTOCOL seeds=$SEEDS models=$MODELS"
echo "[attr_probe] signal_table=$SIGNAL_TABLE out=$OUT_DIR threads=$THREADS"
echo "[attr_probe] split_p4=${SPLIT_P4:-<runner default>} split_p2=${SPLIT_P2:-<runner default>}"

test -f "$CONTENT" || { echo "[attr_probe] missing content table: $CONTENT"; exit 3; }
test -f "$SIGNAL_TABLE" || { echo "[attr_probe] missing signal table: $SIGNAL_TABLE"; exit 3; }
if [ -n "$SPLIT_P4" ] && [ ! -f "${SPLIT_P4}_s1.parquet" ]; then
  echo "[attr_probe] WARNING: ${SPLIT_P4}_s1.parquet missing (seeds must be synced)"
fi
if [ -n "$SPLIT_P2" ] && [ ! -f "${SPLIT_P2}_s1.parquet" ]; then
  echo "[attr_probe] WARNING: ${SPLIT_P2}_s1.parquet missing (seeds must be synced)"
fi
mkdir -p "$OUT_DIR"

ARGS=(--protocols "$PROTOCOL"
      --seeds "$SEEDS"
      --models "$MODELS"
      --content-path "$CONTENT"
      --signal-table "$SIGNAL_TABLE"
      --out-dir "$OUT_DIR"
      --threads "$THREADS")
[ -n "$SPLIT_P4" ] && ARGS+=(--split-p4 "$SPLIT_P4")
[ -n "$SPLIT_P2" ] && ARGS+=(--split-p2 "$SPLIT_P2")
# EXTRA is deliberately word-split (multi-arg escape hatch).
# shellcheck disable=SC2206
[ -n "$EXTRA" ] && ARGS+=($EXTRA)

echo "[attr_probe] python scripts/run_p2_p4_multiseed.py ${ARGS[*]}"
$PY scripts/run_p2_p4_multiseed.py "${ARGS[@]}"

echo "[attr_probe] DONE content=$CONTENT out=$OUT_DIR date=$(date)"
