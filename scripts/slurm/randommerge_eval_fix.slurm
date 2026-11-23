#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-rm-evalfix
#SBATCH -o slurm_rm_evalfix_%A_%a.out
#SBATCH -e slurm_rm_evalfix_%A_%a.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#
# Re-run the rank eval with ALL model signals (linear + MLP).
#
# Why: run_p2_p4_multiseed.py assembles signals.parquet from the models of the
# CURRENT run only, and evaluate() skips when rank_eval.json already exists.  The
# MLP wave therefore trained fine but its eval was skipped, and signals.parquet
# was overwritten with the last model only.  The per-model tables are intact in
# signals/<name>.parquet; this job merges them and evaluates the full 7-signal set.
set -euo pipefail
ID="${SLURM_ARRAY_TASK_ID:?}"
if [ "$ID" -lt 5 ]; then PROTO=p2; SEED=$((ID + 1)); else PROTO=p4; SEED=$((ID - 4)); fi
cd "$HOME/DCR"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
RUN="results/randommerge_p2p4_hpc/$PROTO/s$SEED"
SPLIT="data/processed/step1_randommerge_c_${PROTO}_s${SEED}.parquet"
python3 - "$RUN" <<'PY'
import glob, sys
import pandas as pd
run = sys.argv[1]
base = pd.read_parquet(run + "/signals.parquet")
fixed = base[["beatmap_id", "sunny_star", "daniel_star", "star"]]
out = fixed
for p in sorted(glob.glob(run + "/signals/*.parquet")):
    s = pd.read_parquet(p)
    col = [c for c in s.columns if c != "beatmap_id"][0]
    out = out.merge(s[["beatmap_id", col]], on="beatmap_id", how="outer")
out.to_parquet(run + "/signals_all.parquet", index=False)
print("merged", out.shape, list(out.columns))
PY
python3 scripts/step2_rank_eval.py \
  --in "$SPLIT" \
  --signal-table "$RUN/signals_all.parquet" \
  --signals cm3p_linear_extra,cm3p_mlp_extra,cm3p_linear,cm3p_mlp_h64_L2,sunny_star,daniel_star,star \
  --split test --ceiling-fit all --common-maps --strict-pairs \
  --folds 2 --bootstrap 0 --seed 0 \
  --cache-dir data/processed/rank_eval_cache \
  --out "$RUN/rank_eval.json"
echo "[$(date +%H:%M:%S)] DONE $PROTO s$SEED"
