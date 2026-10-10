# HPC / Slurm 执行手册：补齐剩余白名单谱面的 CM3P 向量

> 场景：整个文件夹已上传到集群（**没有走 git**），调度器是 Slurm，GPU 队列 `i64m1tga800u`。
> 上游手册：[docs/cm3p_remote_runbook.md](cm3p_remote_runbook.md)（讲"算什么"）；本文讲"在 Slurm 上怎么跑"。
> 两个作业脚本：`scripts/slurm/cm3p_fetch.slurm`（要网）、`scripts/slurm/cm3p_embed.slurm`（GPU、离线）。

---

## 0. 上传之后先确认三件事

    cd <你上传的文件夹>          # 下文一律记作 $REPO
    pwd && ls scripts/step0c_chart_content.py manifests/cm3p_todo_1k.csv
    ls external/cm3p/cm3p >/dev/null 2>&1 && echo "CM3P checkout 在" || echo "CM3P checkout 缺"
    du -sh . ; df -h . | tail -1

| 检查项 | 为什么 |
|---|---|
| `manifests/` 里三个文件都在 | 这是远程唯一的输入清单，缺了就没法知道要算哪些图 |
| `external/cm3p/` 在不在 | 不在就要在有网的登录节点 `git clone` 一次 |
| 磁盘配额 | `.osu` 缓存 ~300 MB + HF 权重 1–2 GB；**家目录配额通常很小**，建议整份放到 scratch |

**建议**：把 `HF_HOME` 和 `OSU_CACHE` 指到 scratch，例如

    export HF_HOME=/scratch/$USER/dcr/hf_cache
    export OSU_CACHE=/scratch/$USER/dcr/osu_cache

⚠️ 这两个变量在**提交作业时也要带同样的值**，否则计算节点会去别处找权重而失败（见 §3）。

---

## 1. 登录节点（有网）一次做完四件事

    module load anaconda3                 # 按站点实际模块名调整
    PY=python

    # ① 依赖：只装推理侧。不要装 requirements.txt（它把 torch 钉在 CPU 版）
    $PY -m pip install --user "transformers>=4.48" accelerate numpy pandas pyarrow
    # 若无 CUDA 版 torch：
    $PY -m pip install --user torch --index-url https://download.pytorch.org/whl/cu124

    # ⚠️ slider 不要走 GitHub（集群连不上），也不要 `pip install slider`（PyPI 上是另一个项目）。
    #    仓库自带编好的 wheel，直接本地装：
    $PY -m pip install --user --no-index --no-deps vendor/slider-0.8.2-py3-none-any.whl
    $PY -c "import slider; print('slider', slider.__file__)"

    # ② CM3P checkout（已上传就跳过）
    ls external/cm3p/cm3p >/dev/null 2>&1 || \
        git clone --depth 1 https://github.com/OliBomby/CM3P.git external/cm3p

    # ③ 预热 HF 权重 —— 关键一步，计算节点通常是离线的
    export CM3P_ROOT=$PWD/external/cm3p
    $PY - <<'EOF'
    import os, sys; sys.path.insert(0, os.environ["CM3P_ROOT"])
    from cm3p.processing_cm3p import CM3PProcessor
    from cm3p.modeling_cm3p import CM3PModel
    CM3PProcessor.from_pretrained("OliBomby/CM3P")
    CM3PModel.from_pretrained("OliBomby/CM3P", trust_remote_code=True)
    print("HF weights cached under", os.environ.get("HF_HOME", "~/.cache/huggingface"))
    EOF

    # ④ 连通性测试：计算节点到底能不能上网？（用 5 分钟的小作业试）
    srun -n 1 --time=00:05:00 bash -c \
      'curl -sS -o /dev/null -w "osu=%{http_code}\n" --max-time 20 https://osu.ppy.sh/osu/222593'

---

## 1.9 两个作业分别是什么（为什么一个是 CPU、一个是 GPU）

| 作业 | 队列 | 资源 | 干什么 | 为什么这么配 |
|---|---|---|---|---|
| `cm3p_fetch.slurm` | **不写 `-p`**，走站点默认队列 | 4 核 / 8G / 6 h | 1 万次小 HTTP GET，把 `.osu` 下下来 | **纯网络 IO，用不着 GPU**。GPU 队列是稀缺资源（要排队、时限常常更短），拿它下文件是浪费，也会挤掉别人的训练 |
| `cm3p_embed.slurm` | `i64m1tga800u` | **1 GPU** + 8 核 / 64G / 24 h | 探针 → 混用 gate → 10,072 张推理 | CM3P 是 22 层 Transformer，GPU 必需；那 8 个核是给 dataloader/tokenizer 和 `OMP_NUM_THREADS` 的，不是拿来算矩阵的 |

如果站点要求 CPU 作业也必须指定队列，把 `cm3p_fetch.slurm` 里的注释打开（`#SBATCH -p <cpu-partition>`），
队列名用 `sinfo` / `sinfo -s` 查。GPU 作业换队列不用改文件，命令行覆盖即可：
`sbatch -p <other> scripts/slurm/cm3p_embed.slurm`。

**只有在计算节点完全不能上网时**（§1④ 不是 200），下载才挪到登录节点用 tmux 跑 —— 那不是"CPU 任务"，
只是不需要调度器（§2B）。

---

## 2. 下载 `.osu`：根据 §1④ 的结果二选一

**A. 计算节点能上网（`osu=200`）** → 提交下载作业：

    sbatch scripts/slurm/cm3p_fetch.slurm
    #    默认 4 并发、约 30–60 min；超过 2 h 会超时，重投即可（已经下好的不会重下）

**B. 计算节点不能上网**（更常见）→ 在**登录节点**用 tmux 跑（负载很轻：1 万次小 GET）：

    tmux new -s fetch
    $PY scripts/step0d_fetch_osu.py --ids-file manifests/cm3p_todo_1k.csv \
        --out-dir "$OSU_CACHE" --workers 4 --delay 0.25
    $PY scripts/step0d_fetch_osu.py --ids-file manifests/cm3p_probe_1k.csv \
        --out-dir "$OSU_CACHE" --workers 4 --delay 0.25
    #    Ctrl-b 然后 d 脱离；tmux attach -t fetch 回来看

两个清单都要下：`cm3p_todo_1k.csv`（10,072 张，要算的）+ `cm3p_probe_1k.csv`（256 张，探针用的）。
跑完确认：`ls "$OSU_CACHE"/*.osu | wc -l` 应该在 1 万上下。

---

## 3. 提交 GPU 作业

    sbatch scripts/slurm/cm3p_embed.slurm

如果 §0 里改过路径，提交时带上同名变量：

    HF_HOME=/scratch/$USER/dcr/hf_cache OSU_CACHE=/scratch/$USER/dcr/osu_cache \
        sbatch scripts/slurm/cm3p_embed.slurm

脚本自带的资源请求（可以在命令行覆盖）：

| 指令 | 值 | 说明 |
|---|---|---|
| `-p` | `i64m1tga800u` | 改队列就 `sbatch -p <其他>` 覆盖（用 `sinfo` 查） |
| `-n` | 8 | CPU 核；同时喂给 `OMP_NUM_THREADS` |
| `--gres` | `gpu:1` | **必须**是 `gpu:1` 这种写法；官方示例里的 `--gres 1` 是错的 |
| `--mem` | 64G | 十万级 `.osu` 解析 + 模型 |
| `--time` | `1-00:00:00` | A800 上推理估 20–40 min，给足 24 h 不用管队列策略；跑完节点立刻释放，不会浪费 |
| `-o/-e` | `slurm_cm3p_embed_%j.out/.err` | `%j` 是作业号；**在仓库根目录提交**，文件就落在这里 |

作业内部顺序（这就是为什么它必须离线自洽）：

    环境自检 → 预检（GPU / import / 输入清单，10 秒内失败而不是 10 分钟后）
      → 探针 256 张 → 混用 gate（do-not-mix 则 exit 2 停住）
      → 全量 10,072 张 → 写 data/processed/chart_content_cm3p_add.parquet

调参用环境变量：`INFER_BATCH=4`（显存不够时）、`SKIP_PROBE=1`（不建议）、`PY=...`。
例：`INFER_BATCH=4 sbatch scripts/slurm/cm3p_embed.slurm`

---

## 4. 监控

    squeue -u $USER                       # 排队/运行状态
    tail -f slurm_cm3p_embed_<jobid>.out  # 实时输出（脚本里开了 PYTHONUNBUFFERED）
    sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
    scancel <jobid>                       # 想停就停，已算的不会丢

**时间给得很足**（推理作业 24 h、下载作业 6 h），正常情况用不掉。真被杀了也不用从头来：
`step0c` 只算 `--out` 表里还没有的 id，`step0d` 只下 md5 还不对的文件，两步都幂等，重投命令完全一样：

    sbatch scripts/slurm/cm3p_embed.slurm

---

## 5. 探针 gate（别跳）

输出里会有一段 `VERDICT [ok|caution|do-not-mix]`。判据与三种应对见
[docs/cm3p_remote_runbook.md](cm3p_remote_runbook.md) §4。要点重复一遍：

**已发布的 244K 表是带音频算的，你这里只能不带音频算**，两者是不同的输入分布。
不同图之间的余弦本来就有 **0.576** 那么高，所以"看着 0.9 挺像"**不算通过**；
要看 `Mantel corr ≥ 0.8` 且 `recall@1 ≥ 0.95`。
`do-not-mix` 时作业会 `exit 2` 停住，全量不会开始 —— 这是故意的。

探针顺带是吞吐标定：256 张的耗时 ×40 ≈ 全量耗时，先看这个再决定 `--time` 给多少。

---

## 6. 把结果拿回来合并

需要带回**两个**文件：

    data/processed/chart_content_cm3p_add.parquet      # 新算的 ~10k 行
    data/processed/cm3p_mix_report.json                # 探针结论，必须一起归档

在**有那 11,877 行已发布表的机器上**（也就是本地）执行：

    python scripts/step0c_chart_content.py --source table \
        --in data/processed/chart_content_cm3p_add.parquet --tag 1k

幂等：按 `beatmap_id` 求并集，新行覆盖旧行，已有的 11,877 行不动。
合并后 `source` 列会区分 `0 = HF 预计算` / `1 = 本地无音频`，
`cm3p_rev` 记录模型 revision —— 审计"这一行是哪种口径算的"就靠这两列。

---

## 7. 常见报错

| 现象 | 原因 / 处理 |
|---|---|
| `no GPU visible` | `--gres=gpu:1` 漏了，或站点要先 `module load cuda` |
| `ModuleNotFoundError: slider` | **集群连不上 GitHub，所以不能用 `pip install "slider @ git+..."`**；PyPI 上的 `slider` 还是另一个项目。装仓库自带的 wheel：`pip install --user --no-index --no-deps vendor/slider-0.8.2-py3-none-any.whl`（作业会自己试一次，失败则解包到 `vendor/_unpacked` 挂 PYTHONPATH） |
| `OSError: We couldn't connect to huggingface.co` | 计算节点离线且权重没预热（§1③），或 `HF_HOME` 提交时和预热时不一致 |
| `sbatch: error: getcwd failed: No such file or directory` | **你当前所在的目录已经被删掉/改名了**（常见于把上传的文件夹又移动或重命名过一次），跟作业脚本无关。`cd ~` 再 `cd <仓库>`，或直接 `REPO=/绝对/路径/DSR sbatch scripts/slurm/cm3p_embed.slurm` |
| `ERROR: submit this job from the repo root` | `SLURM_SUBMIT_DIR` 不是仓库根。要么先 `cd` 到仓库再提交，要么用上面的 `REPO=...` 显式指定 |
| `no such file: manifests/...` | 作业没在仓库里跑。先 `cd` 到仓库再提交，或用 `REPO=...` |
| CUDA OOM | `INFER_BATCH=4`（或 2）重投；只影响吞吐，不影响结果 |
| 大量 `missing`（404/451） | 谱面已删/受限，不会进内容表，step2 会把它们丢掉并报数，不会瞎猜 |
| 大量 `mismatch` | 谱面被 mapper 改过；文件保留、`checksum_ok=False`，单独统计 |
| 作业到点被杀 | 直接重投，幂等（§4） |

---

## 8. `sbatch` 自己起不来：`slurm_set_addr: Unable to resolve`

```
sbatch: error: get_addr_info: getaddrinfo() failed: Name or service not known
sbatch: error: slurm_set_addr: Unable to resolve "mgt02"
sbatch: error: Unable to establish control machine address
sbatch: error: Batch job submission failed: No such file or directory
```

**这不是作业脚本的问题。** `mgt02` 是 `/etc/slurm/slurm.conf` 里 `SlurmctldHost` 写的主机名，
Slurm 客户端解析不出它 —— 典型原因是**集群 DNS 挂了**（校园网抽风时很常见），而 `/etc/hosts` 没有兜底。
`sbatch` 根本没走到"读你的脚本"这一步，所以改脚本、改参数都没用。

### 30 秒定位

    hostname; hostname -i
    getent hosts mgt02 || echo "DNS 解析不了 mgt02"
    grep -i "^SlurmctldHost" /etc/slurm/slurm.conf 2>/dev/null || echo "读不到 slurm.conf"
    sinfo 2>&1 | head -3        # 其它 slurm 命令是不是也这样

### 处理顺序

**1) 重试 / 换登录节点。** DNS 抖动常常几分钟就恢复；同集群的另一台登录节点（`yq_mgt01` 之类）可能是好的。

**2) 用 IP 绕开 DNS（最可能有效）。** 复制一份配置，把 `SlurmctldHost` 换成 IP，再用 `SLURM_CONF` 指过去：

    IP=$(hostname -i | awk '{print $1}')
    [ -z "$IP" ] && IP=$(getent hosts "$(hostname)" | awk '{print $1}' | head -1)
    mkdir -p ~/.slurm
    sed -E "s/^SlurmctldHost=[^(]+/SlurmctldHost=${IP}/" /etc/slurm/slurm.conf > ~/.slurm/slurm.conf
    grep -i "^SlurmctldHost" ~/.slurm/slurm.conf          # 确认已变成 IP

    export SLURM_CONF=$HOME/.slurm/slurm.conf
    sinfo | head                                          # 先试这个
    sbatch scripts/slurm/cm3p_embed.slurm

`sed` 里的 `[^(]+` 是为了保留 `mgt02(6817)` 这种带端口的写法。
前提是**控制器就在你登录的这台机器上**（名字对得上时通常如此）；如果控制器是另一台，用管理员给的 IP。

**3) 完全绕开调度器。** 作业脚本不依赖 Slurm —— 所有 `SLURM_*` 变量都带默认值，
`#SBATCH` 行对 bash 来说只是注释。拿到一台 GPU 节点的 shell 后直接跑：

    ssh <gpu-node>                     # 或站点提供的交互方式
    cd <仓库绝对路径>
    module load anaconda3
    bash scripts/slurm/cm3p_embed.slurm 2>&1 | tee logs/cm3p_embed_$(date +%F).log

⚠️ 这会独占节点、绕过 fair-share，有些站点明令禁止；只在前两条都失败时用，并尽快结束。

**4) 报给管理员**（把这段原样发过去最快）：

    登录节点上所有 slurm 客户端命令都失败：
      sbatch: error: slurm_set_addr: Unable to resolve "mgt02"
      sbatch: error: Unable to establish control machine address
    getent hosts mgt02 无结果；/etc/slurm/slurm.conf 里 SlurmctldHost=mgt02。
    疑似 DNS / /etc/hosts 解析问题。
