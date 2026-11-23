#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J dcr-rm-bench-linear
#SBATCH -o slurm_rm_bench_linear_%j.out
#SBATCH -e slurm_rm_bench_linear_%j.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#
# Train the RM-L full-data linear_extra model on the merged randommerge set.
# Content/signal table hashes are identical to the laptop versions:
#   chart_content_cm3p_local.parquet  0a730398...
#   rank_signals_1k.parquet           2d3e86da...
# Benchmark inference/eval run on the laptop with the same tables.
set -euo pipefail
cd "$HOME/DCR"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1
mkdir -p results/benchmark/models results/benchmark/rm_linear/train results/benchmark/logs

echo "[$(date +%H:%M:%S)] host=$(hostname) train start"
python3 -u scripts/step2_cm3p_rank.py \
  --in data/processed/step1_randommerge_pairbest_content.parquet \
  --content-path data/processed/chart_content_cm3p_local.parquet \
  --signal-table data/processed/rank_signals_1k.parquet \
  --extra-signals msd_overall,sunny_star \
  --model linear_extra --lam 1e-3 \
  --val-frac 0 --seed 0 --threads 8 \
  --name cm3p_linear_extra \
  --fit-all \
  --save-model results/benchmark/models/RM_L_full.pt \
  --out-signals results/benchmark/rm_linear/train/RM_L_signals.parquet \
  --out-json results/benchmark/rm_linear/train/RM_L.json
echo "[$(date +%H:%M:%S)] train done"

# G2 self-check. benchmark_infer.py expects a row_id column; make temporary
# views of the training content/signal tables with row_id = beatmap_id.
python3 - <<'PY'
import pandas as pd
c = pd.read_parquet('data/processed/chart_content_cm3p_local.parquet')
c['row_id'] = c['beatmap_id'].astype('int64')
c[['row_id', 'embedding', 'n_windows']].to_parquet(
    '/tmp/rm_content_row_id.parquet', index=False)
s = pd.read_parquet('data/processed/rank_signals_1k.parquet')
s['row_id'] = s['beatmap_id'].astype('int64')
s[['row_id', 'msd_overall', 'sunny_star']].to_parquet(
    '/tmp/rm_signals_row_id.parquet', index=False)
print('temp tables written')
PY

python3 -u scripts/benchmark_infer.py \
  --model results/benchmark/models/RM_L_full.pt \
  --content /tmp/rm_content_row_id.parquet \
  --signals /tmp/rm_signals_row_id.parquet \
  --out results/benchmark/rm_linear/train/RM_L_selfcheck.parquet

python3 - <<'PY'
import pandas as pd
tr = pd.read_parquet(
    'results/benchmark/rm_linear/train/RM_L_signals.parquet'
).rename(columns={'beatmap_id': 'row_id'})
sc = pd.read_parquet(
    'results/benchmark/rm_linear/train/RM_L_selfcheck.parquet'
)[['row_id', 'd_m']]
m = tr.merge(sc, on='row_id', how='inner')
assert len(m) == 20388, f'selfcheck rows {len(m)}'
d = (m['cm3p_linear_extra'] - m['d_m']).abs()
assert d.max() < 1e-6, f'selfcheck max abs diff {d.max()}'
print(f'selfcheck OK: {len(m)} charts, max abs diff {d.max():.3e}')
PY
rm -f /tmp/rm_content_row_id.parquet /tmp/rm_signals_row_id.parquet
echo "[$(date +%H:%M:%S)] RM_BENCH_LINEAR_DONE"
