#!/bin/bash
#SBATCH -J d2v_mlp_cpu
#SBATCH -p i64m512u
#SBATCH -n 16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o slurm_d2v_mlp_cpu_%j.out
#SBATCH -e slurm_d2v_mlp_cpu_%j.err
set -euo pipefail
cd "$HOME/DCR"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
PY=$HOME/.conda/envs/DCR/bin/python
run() {
  tag=$1; proto=$2; content=$3
  echo "=== $tag $proto ==="
  $PY scripts/run_p2_p4_multiseed.py --protocols "$proto" --seeds 1,2,3,4,5 \
    --models mlp,mlp_extra --content-path "$content" \
    --out-dir "results/d2v_main/${proto}_${tag}_hpc" --threads 16
}
run d2v_mlp p4 data/processed/chart_content_d2v_holdout.parquet
run cm3p_legacy_mlp p4 data/processed/chart_content_cm3p_local.parquet
run d2v_mlp p2 data/processed/chart_content_d2v_holdout.parquet
run cm3p_legacy_mlp p2 data/processed/chart_content_cm3p_local.parquet
echo "MLP_HPC_DONE"
