#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J ki-eval
#SBATCH -o slurm_ki_%A_%a.out
#SBATCH -e slurm_ki_%A_%a.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#
# Koishi / ISOR rank eval on DCR 10K + randommerge.
# folds=1: b_full is in-sample and is used only for Pearson/Spearman;
# NLL is independent of the ceiling. captured is not reported.
set -uo pipefail
ID="${SLURM_ARRAY_TASK_ID:?}"
cd "$HOME/ki_eval"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
mkdir -p results/ki_eval/10k results/ki_eval/rm logs

case "$ID" in
  0) IN=data/processed/eval_subsets/step1_10k_all.parquet;
     SIG=data/processed/signals_10k.parquet; SIGNALS=sunny_star,isor_v31_dp;
     OUT=results/ki_eval/10k/all_isor.json; TAG=10k_all_isor ;;
  1) IN=data/processed/eval_subsets/step1_10k_all.parquet;
     SIG=data/processed/signals_10k.parquet; SIGNALS=sunny_star,isor_v31_dp,koishi_v5_dan;
     OUT=results/ki_eval/10k/all_common.json; TAG=10k_all_common ;;
  2) IN=data/processed/eval_subsets/step1_10k_rc20.parquet;
     SIG=data/processed/signals_10k.parquet; SIGNALS=sunny_star,isor_v31_dp;
     OUT=results/ki_eval/10k/rc20_isor.json; TAG=10k_rc20_isor ;;
  3) IN=data/processed/eval_subsets/step1_10k_rc80.parquet;
     SIG=data/processed/signals_10k.parquet; SIGNALS=sunny_star,isor_v31_dp,koishi_v5_dan;
     OUT=results/ki_eval/10k/rc80_common.json; TAG=10k_rc80_common ;;
  4) IN=data/processed/eval_subsets/step1_10k_hardrc.parquet;
     SIG=data/processed/signals_10k.parquet; SIGNALS=sunny_star,isor_v31_dp,koishi_v5_dan;
     OUT=results/ki_eval/10k/hardrc_common.json; TAG=10k_hardrc_common ;;
  5) IN=data/processed/eval_subsets/step1_10k_koishi_native.parquet;
     SIG=data/processed/signals_10k.parquet; SIGNALS=sunny_star,isor_v31_dp,koishi_v5_dan;
     OUT=results/ki_eval/10k/koishi_native.json; TAG=10k_koishi_native ;;
  6) IN=data/processed/eval_subsets/step1_randommerge_all.parquet;
     SIG=data/processed/signals_randommerge.parquet; SIGNALS=sunny_star,isor_v31_dp;
     OUT=results/ki_eval/rm/all_isor.json; TAG=rm_all_isor ;;
  7) IN=data/processed/eval_subsets/step1_randommerge_rc20.parquet;
     SIG=data/processed/signals_randommerge.parquet; SIGNALS=sunny_star,isor_v31_dp;
     OUT=results/ki_eval/rm/rc20_isor.json; TAG=rm_rc20_isor ;;
  8) IN=data/processed/eval_subsets/step1_randommerge_ln.parquet;
     SIG=data/processed/signals_randommerge.parquet; SIGNALS=sunny_star,isor_v31_dp;
     OUT=results/ki_eval/rm/ln_isor.json; TAG=rm_ln_isor ;;
  9) IN=data/processed/eval_subsets/step1_10k_ln.parquet;
     SIG=data/processed/signals_10k.parquet; SIGNALS=sunny_star,isor_v31_dp;
     OUT=results/ki_eval/10k/ln_isor.json; TAG=10k_ln_isor ;;
  *) echo "bad id $ID"; exit 2 ;;
esac

echo "[$(date +%H:%M:%S)] host=$(hostname) id=$ID tag=$TAG"
python3 -W ignore scripts/step2_rank_eval.py \
  --in "$IN" --signal-table "$SIG" --signals "$SIGNALS" \
  --split all --ceiling-fit all --common-maps \
  --folds 1 --bootstrap 0 --seed 0 --min-l 10 \
  --cache-dir "data/processed/rank_eval_cache_ki/$TAG" \
  --out "$OUT"
echo "[$(date +%H:%M:%S)] DONE id=$ID tag=$TAG"
