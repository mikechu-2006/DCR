#!/bin/bash
# Bootstrap the NEW HPC account (aiaa3111_l01_066) for the a17 ablation:
#   1. create ~/.conda/envs/DCR (python 3.11 + torch CUDA + numpy/pandas/pyarrow/...)
#   2. rsync code + osu_cache + the small probe tables from the local checkout
#   3. GPU sanity (torch cuda / bf16 matmul) on the debug partition
# Logs to results/d2v_attr_run/a17_setup.log; exits non-zero on failure.
set -uo pipefail
HOST="aiaa3111_l01_066@hpc2login.hpc.hkust-gz.edu.cn"
ROOT="/home/mikechu/Documents/Projects/DCR"
OUT="$ROOT/results/d2v_attr_run"
mkdir -p "$OUT"
LOG="$OUT/a17_setup.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== a17 setup start $(date)"

echo "=== [1/4] remote venv + pip packages ($(date))"
ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25 "$HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
BASE=/hpc2ssd/softwares/anaconda3/bin/python
ENVDIR=$HOME/.conda/envs/DCR
PY=$ENVDIR/bin/python
if [ ! -x "$PY" ]; then
  echo "[remote] creating venv at $ENVDIR (from shared python 3.11)"
  mkdir -p "$(dirname "$ENVDIR")"
  "$BASE" -m venv "$ENVDIR"
fi
"$PY" -m pip install --no-cache-dir --upgrade pip
"$PY" -m pip install --no-cache-dir numpy pandas pyarrow pyyaml tabulate scikit-learn
"$PY" -m pip install --no-cache-dir torch==2.5.1
"$PY" -c "import torch,numpy,pandas,pyarrow,yaml; print('IMPORT_OK', torch.__version__, torch.version.cuda)"
REMOTE
RC=$?
if [ "$RC" != "0" ]; then echo "=== FATAL env setup rc=$RC"; exit 1; fi
echo "=== env OK $(date)"

echo "=== [2/4] rsync code + data ($(date))"
ssh -o BatchMode=yes "$HOST" 'mkdir -p $HOME/DCR/data2vec_main/splits/mapset_holdout $HOME/DCR/data/processed $HOME/DCR/data/osu_cache $HOME/DCR/scripts/slurm $HOME/DCR/dcr' || exit 1
rsync -az --exclude '__pycache__' "$ROOT/data2vec_main/d2v_main/" "$HOST:~/DCR/data2vec_main/d2v_main/" || exit 1
rsync -az --exclude '__pycache__' "$ROOT/data2vec_main/scripts/" "$HOST:~/DCR/data2vec_main/scripts/" || exit 1
rsync -az --exclude '__pycache__' "$ROOT/data2vec_main/configs/" "$HOST:~/DCR/data2vec_main/configs/" || exit 1
rsync -az --exclude '__pycache__' "$ROOT/dcr/" "$HOST:~/DCR/dcr/" || exit 1
rsync -az --exclude '__pycache__' --exclude '*.parquet' --exclude '*.npz' --exclude '*.npy' \
  "$ROOT/scripts/" "$HOST:~/DCR/scripts/" || exit 1
rsync -az "$ROOT/data2vec_main/splits/mapset_holdout/" "$HOST:~/DCR/data2vec_main/splits/mapset_holdout/" || exit 1
rsync -az "$ROOT/data/processed/step1_mclean25_pairbest_cached.parquet" \
  "$ROOT/data/processed/step1_mclean25_p4sets_cached_s1.parquet" \
  "$ROOT/data/processed/step1_mclean25_p4sets_cached_summary.json" \
  "$ROOT/data/processed/rank_signals_1k.parquet" \
  "$ROOT/data/processed/chart_content_d2v_a20d-mclean.parquet" \
  "$ROOT/data/processed/d2v_attr_audit_per_chart.parquet" \
  "$ROOT/data/processed/beatmap_meta_full.csv" \
  "$HOST:~/DCR/data/processed/" || exit 1
echo "=== rsyncing osu_cache (1.2 GB) $(date)"
rsync -az "$ROOT/data/osu_cache/" "$HOST:~/DCR/data/osu_cache/" || exit 1
echo "=== data OK $(date)"

echo "=== [3/4] remote data checks ($(date))"
ssh -o BatchMode=yes "$HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
cd $HOME/DCR
echo "osu files: $(ls data/osu_cache/*.osu | wc -l)"
echo "pretrain ids: $(wc -l < data2vec_main/splits/mapset_holdout/pretrain_ids.txt)"
echo "eval ids: $(wc -l < data2vec_main/splits/mapset_holdout/eval_ids.txt)"
ls -lh data/processed/step1_mclean25_p4sets_cached_s1.parquet data/processed/chart_content_d2v_a20d-mclean.parquet
ls data2vec_main/configs/audio_style_a17.yaml
REMOTE
RC=$?
if [ "$RC" != "0" ]; then echo "=== FATAL remote checks rc=$RC"; exit 1; fi

echo "=== [4/4] GPU sanity on debug partition ($(date))"
ssh -o BatchMode=yes "$HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
export PATH=/opt/slurm/bin:$PATH
PY=$HOME/.conda/envs/DCR/bin/python
cat > $HOME/gpu_sanity_torch.py <<'PY'
import torch
print("TORCH", torch.__version__, "CUDA_BUILD", torch.version.cuda, "AVAIL", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name(0))
    x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    print("BF16_MATMUL", float((x @ x).float().abs().mean()))
PY
srun -p debug --gres=gpu:1 -n 1 --mem=16G -t 00:10:00 "$PY" "$HOME/gpu_sanity_torch.py"
REMOTE
RC=$?
if [ "$RC" != "0" ]; then echo "=== FATAL GPU sanity rc=$RC"; exit 1; fi
echo "=== a17 setup DONE $(date)"
