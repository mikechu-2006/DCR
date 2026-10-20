#!/bin/bash
# =============================================================================
#  Step 2 - 用 MinaCalc 前向模型拟合玩家 6 维技能 + 预测器参数。
#  GPU job.  See docs/step2_fit_skills.md.
#
#  提交（在仓库根目录）：
#      sbatch scripts/slurm/step2_fit.slurm
#      ROUNDS=5 LIMIT_PLAYS=100000 sbatch scripts/slurm/step2_fit.slurm   # 调参
#
#  依赖的前置产物（都已随 data/ 同步过去）：
#      data/processed/step1_1k.parquet      行 + split
#      data/processed/charts_1k.npz         扁平谱面张量（6.13M 区间 / 27M jack 事件）
#
#  输出：
#      results/fit_1k/X_iter{1..R}.npy      每轮玩家技能
#      results/fit_1k/history.json          RMSE 轨迹 + theta 轨迹
#
#  没有 Slurm 也能直接跑：bash scripts/slurm/step2_fit.slurm
# =============================================================================
#SBATCH -p i64m1tga800u
#SBATCH -J step2-fit
#SBATCH -o slurm_step2_fit_%j.out
#SBATCH -e slurm_step2_fit_%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

set -uo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO" || exit 1
echo "job starts in: $(pwd)"

PY="${PY:-python}"
OUT="${OUT:-results/fit_1k}"
ROUNDS="${ROUNDS:-3}"
BATCH_PLAYS="${BATCH_PLAYS:-4096}"
BATCH_ITV="${BATCH_ITV:-3000000}"
EPOCHS_A="${EPOCHS_A:-2}"
ITERS_B1="${ITERS_B1:-30}"
ITERS_B2="${ITERS_B2:-15}"
LIMIT_ARGS=""
if [ "${LIMIT_PLAYS:-0}" != "0" ]; then LIMIT_ARGS="--limit-plays $LIMIT_PLAYS"; fi
# FREEZE_THETA=1 只跑 Phase A（玩家步），theta 保持 MinaCalc 原值
THETA_ARGS=""
if [ "${FREEZE_THETA:-0}" = "1" ]; then THETA_ARGS="--freeze-theta"; fi
# EVAL_TEST_BATCHES=0 表示 test 全量评估
EVAL_BATCHES="${EVAL_TEST_BATCHES:-0}"

export PYTHONUNBUFFERED=1 TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-8}"

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) ==="; date
$PY -V
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true

$PY - <<'EOF' || exit 1
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
assert torch.cuda.is_available(), "no GPU: check --gres=gpu:1"
EOF

ls -la data/processed/charts_1k.npz data/processed/step1_1k.parquet || exit 1

echo "=== fit: rounds=$ROUNDS batch=$BATCH_PLAYS itv=$BATCH_ITV ==="
$PY scripts/step2_fit_skills.py \
    --device cuda --out "$OUT" --rounds "$ROUNDS" \
    --batch-plays "$BATCH_PLAYS" --batch-itv "$BATCH_ITV" \
    --epochs-a "$EPOCHS_A" --iters-b1 "$ITERS_B1" --iters-b2 "$ITERS_B2" \
    --eval-test-batches "$EVAL_BATCHES" $THETA_ARGS $LIMIT_ARGS || exit 1

echo; echo "=== done ==="; date
$PY - "$OUT" <<'EOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "history.json"
h = json.loads(p.read_text())
for row in h["history"]:
    print(f'round {row["round"]}: train {row["rmse_train"]:.5f}  test {row["rmse_test"]:.5f}')
print("theta:", [round(v, 4) for v in h["history"][-1]["theta"]])
EOF
