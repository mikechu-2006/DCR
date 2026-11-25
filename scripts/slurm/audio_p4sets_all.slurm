#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-p4sets
#SBATCH -o slurm_p4sets_%j.out
#SBATCH -e slurm_p4sets_%j.err
#SBATCH -n 16
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# Experiment 2b, one job:
#   1) build beatmapset-held-out P4 splits (5 seeds);
#   2) train audio and legacy arms in parallel (8 threads each);
#   3) compare audio vs legacy on the same set-P4 splits.
set -euo pipefail
cd "$HOME/DCR"
PY="$HOME/.conda/envs/DCR/bin/python"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
OUT=results/audio_newset/p4sets
mkdir -p "$OUT"
echo "[$(date +%H:%M:%S)] host=$(hostname) split"
"$PY" scripts/step1_p4_split_sets.py \
  --in data/processed/step1_1k_pairbest_nogap.parquet \
  --meta data/processed/beatmap_meta_1k.csv \
  --out-prefix data/processed/step1_1k_pairbest_p4sets \
  --seeds 1,2,3,4,5 --force
echo "[$(date +%H:%M:%S)] train audio + legacy in parallel"
"$PY" scripts/run_p2_p4_multiseed.py \
  --protocols p4 --seeds 1,2,3,4,5 \
  --models linear,linear_extra,mlp,mlp_extra \
  --split-p4 data/processed/step1_1k_pairbest_p4sets \
  --content-path data/processed/chart_content_cm3p_audio_on.parquet \
  --out-dir "$OUT/audio" --threads 8 > "$OUT/audio.log" 2>&1 &
P1=$!
"$PY" scripts/run_p2_p4_multiseed.py \
  --protocols p4 --seeds 1,2,3,4,5 \
  --models linear,linear_extra,mlp,mlp_extra \
  --split-p4 data/processed/step1_1k_pairbest_p4sets \
  --content-path data/processed/chart_content_cm3p_local.parquet \
  --out-dir "$OUT/legacy" --threads 8 > "$OUT/legacy.log" 2>&1 &
P2=$!
wait $P1
wait $P2
echo "[$(date +%H:%M:%S)] compare"
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
print('== compare markdown')
p=out/'audio_vs_legacy.md'
if p.exists():
    print(p.read_text()[:4000])
PY
echo "[$(date +%H:%M:%S)] DONE"
