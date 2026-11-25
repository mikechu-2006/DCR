#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-first2-eval
#SBATCH -o slurm_first2_eval_%j.out
#SBATCH -e slurm_first2_eval_%j.err
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# Assemble A0-A3 signals and run paired Delta-NLL + rank eval on P4 s1.
set -euo pipefail
cd "$HOME/DCR"
PY="$HOME/.conda/envs/DCR/bin/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
OUT=results/first2_aux/pilot
"$PY" - <<'PY'
import pandas as pd
from pathlib import Path
out = Path("results/first2_aux/pilot")
frames = []
for arm in ["a0", "a1", "a2", "a3"]:
    p = out / f"{arm}.parquet"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    d = pd.read_parquet(p)
    col = f"first2_{arm}"
    if col not in d.columns:
        raise SystemExit(f"{p}: missing {col}")
    frames.append(d[["beatmap_id", col]].set_index("beatmap_id"))
sig = pd.concat(frames, axis=1).reset_index()
sig.to_parquet(out / "signals_pilot.parquet", index=False)
print(sig.shape)
PY
"$PY" scripts/paired_delta_nll.py \
  --in data/processed/step1_randommerge_c_p4_s1.parquet \
  --signal-table "$OUT/signals_pilot.parquet" \
  --signals first2_a0,first2_a1,first2_a2,first2_a3 \
  --pairs first2_a1:first2_a0,first2_a2:first2_a0,first2_a3:first2_a0 \
  --split test --bootstrap 400 --seed 0 \
  --out "$OUT/paired_delta.json"
"$PY" scripts/step2_rank_eval.py \
  --in data/processed/step1_randommerge_c_p4_s1.parquet \
  --signal-table "$OUT/signals_pilot.parquet" \
  --signals first2_a0,first2_a1,first2_a2,first2_a3 \
  --split test --ceiling-fit all --common-maps --strict-pairs \
  --folds 2 --bootstrap 0 --seed 0 \
  --cache-dir data/processed/rank_eval_cache \
  --out "$OUT/rank_eval.json"
echo "[$(date +%H:%M:%S)] DONE eval"
