#!/usr/bin/env bash
# ==============================================================================
# DCR -> HPC（HKUST-GZ hpc2login）整包同步
#
# 用法：
#   cd /home/mikechu/Documents/Projects/DCR
#   bash copy_folder.sh                              # 前台跑（tmux 里跑更稳）
#   tmux new -d -s push 'bash copy_folder.sh 2>&1 | tee -a push.log'   # 推荐
#
# 断线后：原样重跑即可续传（已传完的文件被秒跳过，未传完的从 .rsync-partial 续）
# 前置：ssh-copy-id -i ~/.ssh/id_ed25519.pub hpc2login.hpc.hkust-gz.edu.cn
#
# 为什么不用 --append-verify：rsync 明确禁止 --append* 与 --partial-dir 同时使用
#   （"rsync: --append cannot be used with --partial-dir"）。
#   两者选一：--partial-dir 是更安全的那个 —— 断点留在远端 .rsync-partial/ 里，
#   下次续传时它作为 delta 的 basis 被复用，传完由 rsync 自己删除；
#   而 --append 只是"假定已有部分是正确前缀"，半截文件被写坏时反而会出错。
# ==============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO" || exit 1

REMOTE=aiaa3111_l01_008@hpc2login.hpc.hkust-gz.edu.cn
R=/hpc2hdd/home/aiaa3111_l01_008/DCR

# rsync 自己的 ssh：心跳防呆（登录节点空闲超时会掐连接）
export RSYNC_RSH="ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o TCPKeepAlive=yes"

# 远端目录在每条 rsync 里就地建好（不能依赖上一条连接的状态）
MKDIRS="mkdir -p $R/data $R/data/raw $R/data/raw/extracted $R/data/raw/extracted_10k $R/data/raw/cm3p_embeddings $R/data/processed $R/data/interim $R/data/osu_cache"

# -a 归档  -z 传中压缩  -h 人类可读
# --partial --partial-dir=.rsync-partial  断点落在 <目标目录>/.rsync-partial/，可续传
# --info=name,progress2                   整批百分比（判断"是不是卡住了"看这个）
# --timeout=120                           卡死 120 s 就断开重试，而不是挂一整夜
# --exclude=.rsync-partial/               别把断点目录本身再同步一遍
OPTS=(-a -z -h
      --partial --partial-dir=.rsync-partial
      --timeout=120
      --info=name,progress2
      --exclude='.rsync-partial/')

run() {
  echo
  echo "=================== $* ==================="
  date
  "$@"
  local rc=$?
  echo "[exit=$rc] $(date)"
  return $rc
}

FAIL=0

# ---------- ① 代码（不含 data/、experiments/、legacy/ 与缓存）----------
# 唯一带 --delete 的批次：--exclude data/ 保护远端 data（远端 osu_cache 里有本地没有的 .osu）
run rsync "${OPTS[@]}" --rsync-path="$MKDIRS && rsync" --delete \
  --exclude='data/' --exclude='.git/' --exclude='experiments/' --exclude='legacy/' \
  --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='slurm_*.out' --exclude='slurm_*.err' \
  --exclude='.minacalc_cache/' \
  --exclude='data/processed/charts_1k.npz' \
  ./ "$REMOTE:$R/" || FAIL=1

# ---------- ② 数据 10.1 G ----------
# 这三批永不 --delete：远端可能有本地没有的文件（fetch 作业新下的 .osu 等）
for d in data/raw data/processed data/interim; do
  run rsync "${OPTS[@]}" --rsync-path="$MKDIRS && rsync" "$d/" "$REMOTE:$R/$d/" || FAIL=1
done

# ---------- ③ 小文件压缩包 192 M（21,950 个 .osu 压成一个文件）----------
run rsync "${OPTS[@]}" --rsync-path="$MKDIRS && rsync" \
  data/osu_cache.zip "$REMOTE:$R/data/" || FAIL=1

echo
if [ "$FAIL" -eq 0 ]; then
  echo "ALL DONE (all batches exit=0)  $(date)"
else
  echo "SOME BATCHES FAILED —— 修好网络后原样重跑本脚本即可续传  $(date)"
fi

cat <<'EOF'

--------------- 传完后在远端做的事（手动，确认计数无误再删 zip）---------------
ssh aiaa3111_l01_008@hpc2login.hpc.hkust-gz.edu.cn '
  cd /hpc2hdd/home/aiaa3111_l01_008/DCR/data
  unzip -q -n osu_cache.zip -d .        # -n 只补缺的，不动远端已有的 ~9,802 个
  ls osu_cache/*.osu | wc -l            # 期望 21,950
  # 计数确认无误后再执行下一行：
  # rm -f osu_cache.zip
'
EOF

cat <<'EOF'

--------------- 代码同步：远端 git pull（2026-10-13 起可用）---------------
远端 /hpc2hdd/home/aiaa3111_l01_008/DCR 已经是正常 git checkout（.git 已 rsync 过去，
之后不再随本脚本同步），仓库是 public，HTTPS 免密钥：

    ssh aiaa3111_l01_008@hpc2login.hpc.hkust-gz.edu.cn
    cd /hpc2hdd/home/aiaa3111_l01_008/DCR
    git pull          # 已设 pull.ff=only：只做 fast-forward，不会冒合并提交
    git status        # 只应看到 slurm_*.out/.err 这类未跟踪文件

本地流程：改代码 → git add/commit → git push → 远端 git pull。
数据（data/）仍走本脚本的 rsync 批次；代码批次保留作兜底（本地始终是已提交状态）。
⚠️ 不要在服务器上 commit（以本地为准）；.git 不在 rsync 范围，pull 出来的 refs 不会被覆盖。
远端 .git 万一丢了，重建：
    git init -q && git remote add origin https://github.com/mikechu-2006/DCR.git
    git fetch origin && git checkout -f -B main origin/main && git config pull.ff only
EOF
