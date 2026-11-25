#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-first2-a6-eval
#SBATCH -o slurm_first2_a6_eval_%A_%a.out
#SBATCH -e slurm_first2_a6_eval_%A_%a.err
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-4
set -euo pipefail
cd "$HOME/DCR"
PY="$HOME/.conda/envs/DCR/bin/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
SEED=$((SLURM_ARRAY_TASK_ID + 1))
OUT="results/first2_aux/1k_p4/s${SEED}"
"$PY" - <<PY
import pandas as pd
from pathlib import Path
out = Path("$OUT")
frames = []
for arm in ["a0", "a6"]:
    p = out / f"{arm}.parquet"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    d = pd.read_parquet(p)
    col = f"first2_{arm}"
    frames.append(d[["beatmap_id", col]].set_index("beatmap_id"))
sig = pd.concat(frames, axis=1).reset_index()
sig.to_parquet(out / "signals_a6.parquet", index=False)
print(sig.shape)
PY
"$PY" scripts/paired_delta_nll.py \
  --in "data/processed/step1_1k_pairbest_p4_s${SEED}.parquet" \
  --signal-table "$OUT/signals_a6.parquet" \
  --signals first2_a0,first2_a6 \
  --pairs first2_a6:first2_a0 \
  --split test --bootstrap 400 --seed 0 \
  --out "$OUT/paired_delta_a6.json"
"$PY" scripts/step2_rank_eval.py \
  --in "data/processed/step1_1k_pairbest_p4_s${SEED}.parquet" \
  --signal-table "$OUT/signals_a6.parquet" \
  --signals first2_a0,first2_a6 \
  --split test --ceiling-fit all --common-maps --strict-pairs \
  --folds 2 --bootstrap 0 --seed 0 \
  --cache-dir data/processed/rank_eval_cache \
  --out "$OUT/rank_eval_a6.json"
echo "[$(date +%H:%M:%S)] DONE eval A6 s$SEED"
