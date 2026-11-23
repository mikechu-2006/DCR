#!/bin/bash
# Assemble signals, build subsets, sync to HPC, submit 10 rank-eval tasks.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 scripts/assemble_signals.py \
  --isor-glob 'data/processed/signals_raw/isor_10k_s*.ndjson' \
  --koishi-glob 'data/processed/signals_raw/koishi_10k_s*.ndjson' \
  --dataset 10k \
  --out-long data/processed/signals_long_10k.parquet \
  --out-wide data/processed/signals_10k.parquet

python3 scripts/assemble_signals.py \
  --isor-glob 'data/processed/signals_raw/isor_rm_s*.ndjson' \
  --dataset randommerge \
  --out-long data/processed/signals_long_randommerge.parquet \
  --out-wide data/processed/signals_randommerge.parquet

python3 scripts/make_subsets.py \
  --in data/processed/step1_10k_pairbest.parquet \
  --signals data/processed/signals_10k.parquet \
  --dataset 10k \
  --out-dir data/processed/eval_subsets

python3 scripts/make_subsets.py \
  --in data/processed/step1_randommerge_pairbest_content.parquet \
  --signals data/processed/signals_randommerge.parquet \
  --dataset randommerge \
  --out-dir data/processed/eval_subsets

ssh hpc2login.hpc.hkust-gz.edu.cn 'mkdir -p ki_eval/scripts/slurm ki_eval/data/processed/eval_subsets ki_eval/results/ki_eval/10k ki_eval/results/ki_eval/rm ki_eval/data/processed/rank_eval_cache_ki'

rsync -avz --no-perms --omit-dir-times \
  data/processed/signals_10k.parquet \
  data/processed/signals_randommerge.parquet \
  data/processed/step1_10k_pairbest.parquet \
  hpc2login.hpc.hkust-gz.edu.cn:ki_eval/data/processed/

rsync -avz --no-perms --omit-dir-times data/processed/eval_subsets/ \
  hpc2login.hpc.hkust-gz.edu.cn:ki_eval/data/processed/eval_subsets/

rsync -avz --no-perms --omit-dir-times scripts/slurm/ki_eval.slurm \
  hpc2login.hpc.hkust-gz.edu.cn:ki_eval/scripts/slurm/

ssh hpc2login.hpc.hkust-gz.edu.cn 'cd ki_eval && export PATH=/opt/slurm/bin:$PATH && sbatch scripts/slurm/ki_eval.slurm'
