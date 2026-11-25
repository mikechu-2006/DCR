#!/bin/bash
#SBATCH -J bench-attr-calib
#SBATCH -p debug
#SBATCH -n 8
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o slurm_bench_attr_calib_%j.out
#SBATCH -e slurm_bench_attr_calib_%j.err
# 5-fold chart-group isotonic calibration over the three ablation arms + refs.
set -eo pipefail
cd "$HOME/DCR"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
PY=$HOME/.conda/envs/DCR/bin/python
OUT=results/benchmark_attr
for ref in RM_L D2V_L M1_L; do
  if [ -f "results/benchmark_ref/$ref.parquet" ]; then
    cp "results/benchmark_ref/$ref.parquet" "$OUT/pred/$ref.parquet"
  fi
done
$PY scripts/benchmark_calibrate_eval.py \
  --pred-dir "$OUT/pred" \
  --out-dir "$OUT/eval" \
  --runs A0D,A17D,A20D,RM_L,D2V_L,M1_L \
  --bootstrap 1000
echo "[bench-calib] DONE $(date)"
