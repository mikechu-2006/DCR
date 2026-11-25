#!/bin/bash
#SBATCH -J bench-attr-ext
#SBATCH -p i64m1tga800ue
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o slurm_bench_attr_ext_%j.out
#SBATCH -e slurm_bench_attr_ext_%j.err
# Encode (a) all 21,949 charts for randommerge probe training and (b) the 1,382
# VSRG benchmark staging charts with one attribute-arm ckpt. ARM=a0d|a17d|a20d
set -eo pipefail
cd "$HOME/DCR/data2vec_main"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
PY=$HOME/.conda/envs/DCR/bin/python
if [ -z "$ARM" ]; then echo "set ARM=a0d|a17d|a20d"; exit 2; fi
CKPT=runs/attr_d/$ARM/checkpoints/last.pt
echo "[bench-ext] host=$(hostname) date=$(date) arm=$ARM ckpt=$CKPT"
test -f "$CKPT" || { echo "missing ckpt $CKPT"; exit 3; }
TAG="d2v_$ARM"
# NOTE: ChartStore resolves relative --osu-dir against the repository root
# (data2vec_main/../), so pass a repo-relative path, not ../data/...
ALL="../data/processed/chart_content_d2v_$ARM-all.parquet"
if [ -f "$ALL" ]; then
  echo "[bench-ext] all-chart content already present: $ALL"
else
  $PY -m d2v_main.extract --config configs/audio_style_base.yaml \
    --ckpt "$CKPT" --out "$ALL" \
    --device cuda --batch-size 32 --num-workers 6 --tag "$TAG-all"
fi
$PY -m d2v_main.extract --config configs/audio_style_base.yaml \
  --ckpt "$CKPT" --osu-dir data/benchmark_d2v_osu \
  --ids-file ../data/processed/benchmark_d2v_ids.txt \
  --out "../data/processed/benchmark_content_$ARM-raw.parquet" \
  --device cuda --batch-size 32 --num-workers 6 --tag "benchmark_$ARM"
$PY ../scripts/finalize_benchmark_d2v.py \
  --inp "../data/processed/benchmark_content_$ARM-raw.parquet" \
  --out "../data/processed/benchmark_content_$ARM.parquet"
echo "[bench-ext] DONE arm=$ARM $(date)"
