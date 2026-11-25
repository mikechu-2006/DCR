#!/bin/bash
#SBATCH -J d2v_attr_probe
#SBATCH -p i64m512u
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o slurm_attr_probes_%j.out
#SBATCH -e slurm_attr_probes_%j.err
set -euo pipefail
cd "$HOME/DCR"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
PY=$HOME/.conda/envs/DCR/bin/python
STAGE=${STAGE:?set STAGE=s or d}
SIGNALS="--signal-table data/processed/rank_signals_1k.parquet"
echo "[attr-probe] stage=$STAGE host=$(hostname) date=$(date)"
if [ "$STAGE" = "s" ]; then
  for ARM in a0cont a12 a20 a27; do
    C="data/processed/chart_content_d2v_${ARM}-1k.parquet"
    echo "[attr-probe] p4 $ARM"
    $PY scripts/run_p2_p4_multiseed.py --protocols p4 --seeds 1 \
      --models linear,linear_extra --content-path "$C" $SIGNALS \
      --out-dir "results/d2v_attr/$ARM/p4" --threads 8
    echo "[attr-probe] p4_hold $ARM"
    $PY scripts/run_p2_p4_multiseed.py --protocols p4 --seeds 1 \
      --models linear,linear_extra --content-path "$C" $SIGNALS \
      --split-p4 data/processed/step1_1k_pairbest_p4_d2vhold \
      --out-dir "results/d2v_attr/$ARM/p4_hold" --threads 8
    if [ -f "data/processed/step1_1k_pairbest_p2_s1.parquet" ]; then
      echo "[attr-probe] p2 $ARM"
      $PY scripts/run_p2_p4_multiseed.py --protocols p2 --seeds 1 \
        --models linear,linear_extra --content-path "$C" $SIGNALS \
        --out-dir "results/d2v_attr/$ARM/p2" --threads 8
    fi
  done
elif [ "$STAGE" = "d" ]; then
  for ARM in a0d a12d a20d a27d; do
    C="data/processed/chart_content_d2v_${ARM}-mclean.parquet"
    echo "[attr-probe] setp4 $ARM"
    $PY scripts/run_p2_p4_multiseed.py --protocols p4 --seeds 1 \
      --models linear,linear_extra --content-path "$C" $SIGNALS \
      --split-p4 data/processed/step1_mclean25_p4sets \
      --out-dir "results/d2v_attr_d/$ARM/setp4" --threads 8
  done
  echo "[attr-probe] setp4 CM3P legacy reference"
  $PY scripts/run_p2_p4_multiseed.py --protocols p4 --seeds 1 \
    --models linear,linear_extra,mlp,mlp_extra \
    --content-path data/processed/chart_content_cm3p_local.parquet $SIGNALS \
    --split-p4 data/processed/step1_mclean25_p4sets \
    --out-dir "results/d2v_attr_d/cm3p_legacy/setp4" --threads 8
else
  echo "bad STAGE=$STAGE"; exit 2
fi
echo "[attr-probe] done $(date)"
