#!/usr/bin/env bash
# 断线后：直接重跑 bash push.sh 即可续传，已传完的文件会被秒跳过
set -uo pipefail
cd "$(dirname "$0")"

REMOTE=aiaa3111_l01_008@hpc2login.hpc.hkust-gz.edu.cn
R=/hpc2hdd/home/aiaa3111_l01_008/DCR
SSH="ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o TCPKeepAlive=yes"

# 远端目录一次建好（--rsync-path 在一条连接里做完，省得单独 ssh）
mkdir -p data/raw/extracted data/raw/extracted_10k data/raw/cm3p_embeddings \
         data/processed data/interim data/osu_cache
RSYNC_PATH="--rsync-path=mkdir -p $R/data/raw/extracted $R/data/raw/extracted_10k $R/data/raw/cm3p_embeddings $R/data/processed $R/data/interim $R/data/osu_cache && rsync"

OPTS=(-a -z -h --partial --partial-dir=.rsync-partial --append-verify \
      --info=name,progress2 --human-readable)

run() { echo; echo "=================== $* ==================="; date; "$@"; echo "[exit=$?] $(date)"; }

# ① 代码 27 M（唯一带 --delete 的批次：--exclude data/ 保护远端 data）
run rsync "${OPTS[@]}" $RSYNC_PATH --delete \
  --exclude 'data/' --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' \
  ./ "$REMOTE:$R/"

# ② 数据 10.1 G（永不 --delete：远端可能有本地没有的文件）
run rsync "${OPTS[@]}" $RSYNC_PATH data/raw/       "$REMOTE:$R/data/raw/"
run rsync "${OPTS[@]}" $RSYNC_PATH data/processed/ "$REMOTE:$R/data/processed/"
run rsync "${OPTS[@]}" $RSYNC_PATH data/interim/   "$REMOTE:$R/data/interim/"

# ③ 小文件压缩包 192 M
run rsync "${OPTS[@]}" $RSYNC_PATH data/osu_cache.zip "$REMOTE:$R/data/"

echo; echo "ALL DONE"; date