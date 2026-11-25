#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-p4sets-compare
#SBATCH -o slurm_p4sets_compare_%j.out
#SBATCH -e slurm_p4sets_compare_%j.err
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
cd "$HOME/DCR"
PY="$HOME/.conda/envs/DCR/bin/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
OUT=results/audio_newset/p4sets
"$PY" scripts/compare_multiseed_runs.py \
  --baseline "$OUT/legacy" --variant "$OUT/audio" \
  --out "$OUT/audio_vs_legacy.md"
"$PY" - <<'PY'
import pandas as pd
from pathlib import Path
out=Path('results/audio_newset/p4sets')
for arm in ['legacy','audio']:
    p=out/arm/'summary_mean_std.csv'
    print('==',arm,p.exists())
    if p.exists():
        print(pd.read_csv(p).round(5).to_string(index=False))
PY
