# HPC / Slurm 执行手册：补齐剩余白名单谱面的 CM3P 向量

> 场景：整个文件夹已上传到集群（**没有走 git**），调度器是 Slurm，GPU 队列 `i64m1tga800u`。
> 上游手册：[docs/cm3p_remote_runbook.md](cm3p_remote_runbook.md)（讲"算什么"）；本文讲"在 Slurm 上怎么跑"。
> 两个作业脚本：`scripts/slurm/cm3p_fetch.slurm`（要网）、`scripts/slurm/cm3p_embed.slurm`（GPU、离线）。

---

## 0. 上传之后先确认这几件事

    cd <你上传的文件夹>          # 下文一律记作 $REPO
    pwd && ls scripts/step0c_chart_content.py manifests/whitelist_1k.csv
    ls external/cm3p/cm3p >/dev/null 2>&1 && echo "CM3P checkout 在" || echo "CM3P checkout 缺"
    ls data/osu_cache/*.osu 2>/dev/null | wc -l     # 期望 21,949（整份文件夹拷过来时）
    ls vendor/slider-*.whl                          # 推理要用，缺了 embed 作业会明确报错
    du -sh . ; df -h . | tail -1

| 检查项 | 为什么 |
|---|---|
| `manifests/` 里清单和探针夹具都在 | 全量白名单 `whitelist_1k.csv` + 256 张探针 `cm3p_probe_1k.*`，缺了不知道要算哪些图、也没法做 gate |
| `data/osu_cache/` 有 21,949 个 `.osu` | 整份文件夹拷贝时会一起过来（只有正文，~1.2 GB）；没拷到就在登录节点补下（§2） |
| `external/cm3p/` 在不在 | 不在就要在有网的登录节点 `git clone` 一次 |
| `vendor/slider-*.whl` 在不在 | 计算节点连不上 GitHub，slider 只能从仓库自带的 wheel 装 |
| 磁盘配额 | `.osu` 缓存 ~1.2 GB + HF 权重 1–2 GB；**家目录配额通常很小**，建议整份放到 scratch |

**建议**：把 `HF_HOME` 和 `OSU_CACHE` 指到 scratch，例如

    export HF_HOME=/scratch/$USER/dcr/hf_cache
    export OSU_CACHE=/scratch/$USER/dcr/osu_cache

⚠️ 这两个变量在**提交作业时也要带同样的值**，否则计算节点会去别处找权重而失败（见 §3）。
`cm3p_embed.slurm` 的默认值是"文件夹里自带"的路径：`OSU_CACHE=$REPO/data/osu_cache`、
HF 权重优先 `$REPO/data/raw/hf_cache`（有就自动用），所以整份拷贝的自洽场景不需要额外导出变量。

### 0.5 代码同步：远端 `git pull`

上传的文件夹从 2026-10-13 起带 `.git`，远端 `/hpc2hdd/.../DCR` 就是正常 git checkout；
仓库是 **public**，走 HTTPS 免密钥，所以改代码以后不用再 rsync：

    ssh aiaa3111_l01_008@hpc2login.hpc.hkust-gz.edu.cn
    cd /hpc2hdd/home/aiaa3111_l01_008/DCR
    git pull          # 已设 pull.ff=only，只做 fast-forward
    git status        # 只应看到 slurm_*.out/.err 这类未跟踪文件

本地流程：改完 → `git add/commit/push` → 远端 `git pull`。数据（`data/`）仍走
`copy_folder.sh` 的 rsync 批次；代码批次保留作兜底（本地始终处于已提交状态）。
⚠️ **不要在服务器上 commit**（以本地为准），`.git` 不在 rsync 范围内，pull 出来的 refs 不会被覆盖。
远端 `.git` 万一丢了，重建：

    git init -q && git remote add origin https://github.com/mikechu-2006/DCR.git
    git fetch origin && git checkout -f -B main origin/main && git config pull.ff only

---

## 1. 登录节点（有网）一次做完四件事

    module load anaconda3                 # 按站点实际模块名调整
    PY=python

    # ① 依赖：只装推理侧。不要装 requirements.txt（那是管线 B 的环境，torch 钉在 CPU 版）；
    #    推理侧依赖的单一来源是 requirements-cm3p.txt，下面的 pip 行与它等价。
    # transformers 必须钉在 4.42–4.57 这个窗口里，两头都会炸：
    #   <=4.41 → 没有 processing_utils.CommonKwargs → ImportError
    #   >=5.0  → 这个类被删掉了                     → 一模一样的 ImportError
    # 上游 CM3P 的 Dockerfile 钉的就是 transformers==4.55.0，照抄最省事。
    # 千万不要用 --user 装 transformers：~/.local 会在 sys.path 里压过 conda 环境，
    # 而且不带上限时 pip 会直接拿 5.x（CommonKwargs 已被删除）。
    # 已经被 --user 装坏了就这样修回来（两个方向都能修）：
    #   $PY -m pip install --user -U "transformers==4.55.0"
    #    hydra-core 也在这张清单里：上游 config.py 在模块顶层 import hydra（它只是 train.py
    #    的 ConfigStore），而 utils/mmrs_dataset.py → utils/beatmap_files_dataset.py 会把它
    #    拽进来 —— 少了它 --source osu-dir 会死在 "No module named 'hydra'"，而且是在预检
    #    之后、真正开始跑的时候才炸。发行名是 hydra-core，import 名才是 hydra。
    #    （等价写法：$PY -m pip install -r requirements-cm3p.txt）
    $PY -m pip install -U "transformers==4.55.0" "hydra-core>=1.3,<1.4" accelerate numpy pandas pyarrow
    $PY -c "import transformers;print(transformers.__version__, transformers.__file__)"
    $PY -c "from transformers.processing_utils import CommonKwargs;print('CommonKwargs ok')"
    $PY -c "import hydra;print('hydra', hydra.__version__)"
    # 若无 CUDA 版 torch：
    $PY -m pip install --user torch --index-url https://download.pytorch.org/whl/cu124

    # ⚠️ slider 不要走 `pip install "slider @ git+..."`（它只有 legacy setup.py，pip 的
    #    build isolation 会报 "setuptools is not available in the build environment"）。
    #    也不要 `pip install slider`（PyPI 上那是另一个项目）。
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
| `cm3p_fetch.slurm` | **不写 `-p`**，走站点默认队列 | 4 核 / 8G / 6 h | 2.2 万次小 HTTP GET，把 `.osu` 下下来（只补算时 1 万次） | **纯网络 IO，用不着 GPU**。GPU 队列是稀缺资源（要排队、时限常常更短），拿它下文件是浪费，也会挤掉别人的训练 |
| `cm3p_embed.slurm` | `i64m1tga800u` | **1 GPU** + 8 核 / 64G / 24 h | 探针 → 混用 gate → 21,949 张推理（只补算时 10,072 张） | CM3P 是 22 层 Transformer，GPU 必需；那 8 个核是给 dataloader/tokenizer 和 `OMP_NUM_THREADS` 的，不是拿来算矩阵的 |

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
    $PY scripts/step0d_fetch_osu.py --out-dir "$OSU_CACHE" --workers 4 --delay 0.25
    #    默认清单就是全量白名单（manifests/whitelist_1k.csv，21,949 张）；
    #    只补 HF 表缺的 10,072 张：--ids-file manifests/cm3p_todo_1k.csv
    #    Ctrl-b 然后 d 脱离；tmux attach -t fetch 回来看

默认下**整个白名单**（`manifests/whitelist_1k.csv`，21,949 张；256 张探针是它的子集，
不用单独下一遍）。缓存里**只有 `.osu` 正文**，没有 `.osz`、没有音频、没有图片。
跑完确认：`ls "$OSU_CACHE"/*.osu | wc -l` 全量应在 2 万上下，只补算在 1 万上下。

---

## 3. 提交 GPU 作业

    sbatch scripts/slurm/cm3p_embed.slurm

默认编码**整个白名单**（`manifests/whitelist_1k.csv`，21,949 张——缓存里 100% 都有），
只算 HF 表缺的那 10,072 张就加 `EMBED_MANIFEST=manifests/cm3p_todo_1k.csv`。

如果 §0 里改过路径，提交时带上同名变量：

    HF_HOME=/scratch/$USER/dcr/hf_cache OSU_CACHE=/scratch/$USER/dcr/osu_cache \
        sbatch scripts/slurm/cm3p_embed.slurm

HF 权重：脚本按 `data/raw/hf_cache` → `HF_HOME` → `~/.cache/huggingface` 的顺序找。
整份文件夹拷过来时，把预热好的 `~/.cache/huggingface` 一起放进 `data/raw/hf_cache`
就自带权重了；三处都没有会在预检 B 里 10 秒内报错，并打印预热命令。

脚本自带的资源请求（可以在命令行覆盖）：

| 指令 | 值 | 说明 |
|---|---|---|
| `-p` | `i64m1tga800u` | 改队列就 `sbatch -p <其他>` 覆盖（用 `sinfo` 查） |
| `-n` | 8 | CPU 核；同时喂给 `OMP_NUM_THREADS` |
| `--gres` | `gpu:1` | **必须**是 `gpu:1` 这种写法；官方示例里的 `--gres 1` 是错的 |
| `--mem` | 64G | 两万级 `.osu` 解析 + 模型 |
| `--time` | `1-00:00:00` | A800 上 21,949 张估 45–90 min（只补 10,072 张 20–40 min）；给足 24 h 不用管队列策略，跑完节点立刻释放 |
| `-o/-e` | `slurm_cm3p_embed_%j.out/.err` | `%j` 是作业号；**在仓库根目录提交**，文件就落在这里 |

作业内部顺序（这就是为什么它必须离线自洽）：

    环境自检 → 预检 A（GPU / torch / transformers 窗口 / cm3p+hydra+slider，10 秒内失败）
      → 预检 B（.osu 语料、HF 权重、探针清单：缺什么直接给出补法）
      → 探针 256 张 → 混用 gate
      → 全量 21,949 张 → 写 data/processed/chart_content_cm3p_local.parquet
      → MERGE=1 时并进 data/processed/chart_content_cm3p.parquet
      → 覆盖率汇总（缺哪些 id、checksum_ok=False 几张）

调参用环境变量：`INFER_BATCH=4`（显存不够时）、`SKIP_PROBE=1`（不建议）、
`EMBED_MANIFEST=...`、`EMBED_OUT=...`、`MERGE=1`、`GATE_STOP=0/1`、
`FETCH_FIRST=1`（计算节点有网时自己下 `.osu`）、`PY=...`。
例：`INFER_BATCH=4 sbatch scripts/slurm/cm3p_embed.slurm`

---

## 4. 监控

    squeue -u $USER                       # 排队/运行状态
    tail -f slurm_cm3p_embed_<jobid>.out  # 实时输出（脚本里开了 PYTHONUNBUFFERED）
    sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
    scancel <jobid>                       # 想停就停，已算的不会丢

**时间给得很足**（推理作业 24 h、下载作业 6 h），正常情况用不掉。`step0c` 只在**跑完时**写一次 `--out` 表：
表写成就幂等（重投只补表里没有的 id），但**中途被杀会丢掉这一次的全部计算**——所以 `--time` 不要砍
（全量 21,949 张在 A800 上约 45–90 min）。`step0d` 是每张下完就落盘，随时可续。重投命令完全一样：

    sbatch scripts/slurm/cm3p_embed.slurm

---

## 5. 探针 gate（别跳）

输出里会有一段 `VERDICT [ok|caution|do-not-mix]`。判据与三种应对见
[docs/cm3p_remote_runbook.md](cm3p_remote_runbook.md) §4。要点重复一遍：

**已发布的 244K 表是带音频算的，你这里只能不带音频算**，两者是不同的输入分布。
不同图之间的余弦本来就有 **0.576** 那么高，所以"看着 0.9 挺像"**不算通过**；
要看 `Mantel corr ≥ 0.8` 且 `recall@1 ≥ 0.95`。
`do-not-mix` 的处理分两种：**补算再合并**（`EMBED_MANIFEST=manifests/cm3p_todo_1k.csv`）
会 `exit 2` 停住，全量不开始——这是故意的；**全量重算**（默认清单）不涉及混用，
verdict 只记录不停（`GATE_STOP=1` 可强制停）。

探针顺带是吞吐标定：256 张的耗时 ×40 ≈ 全量耗时，先看这个再决定 `--time` 给多少。

---

## 6. 把结果拿回来合并

需要带回**两个**文件：

    data/processed/chart_content_cm3p_local.parquet    # 全量 21,949 行（只补算时 ~10k 行）
    data/processed/cm3p_mix_report.json                # 探针结论，必须一起归档

在**有那 11,877 行已发布表的机器上**（也就是本地）执行：

    # (a) 直接当内容表用（口径完全统一，不涉及混用）：
    #     step2 --chart-content cm3p --content-path data/processed/chart_content_cm3p_local.parquet
    # (b) 或者并进规范表：按 beatmap_id 求并集，新行覆盖旧行 —— 全量重算会把
    #     21,949 行全部换成 source=1 的本地无音频行，"覆盖子集 54%" 不再存在
    python scripts/step0c_chart_content.py --source table \
        --in data/processed/chart_content_cm3p_local.parquet --tag 1k

幂等：按 `beatmap_id` 求并集，新行覆盖旧行；只补算时已发布的行不动。
合并后 `source` 列会区分 `0 = HF 预计算` / `1 = 本地无音频`，
`cm3p_rev` 记录模型 revision —— 审计"这一行是哪种口径算的"就靠这两列。

缓存抓完后，把白名单更新成表并让 step0 剔除抓不到的谱面（详见
[docs/step0_preprocessing.md](step0_preprocessing.md) §2.4）：

    python scripts/step0g_whitelist.py --tag 1k --cache data/osu_cache \
        --out data/processed/whitelist_1k.csv \
        --manifest-out manifests/whitelist_1k.csv
    python scripts/step0_preprocess.py --dump-dir <1k dump> \
        --whitelist data/processed/whitelist_1k.csv

---

## 7. 常见报错

| 现象 | 原因 / 处理 |
|---|---|
| `no GPU visible` | `--gres=gpu:1` 漏了，或站点要先 `module load cuda` |
| `torch._inductor.exc.InductorError: PermissionError: [Errno 13] Permission denied: 'nvcc'` | CM3P 的 `compiled_head` 带 `@torch.compile(dynamic=True)`，模型 config 打开 `reference_compile` 时 TorchInductor 会去调 `nvcc`，而集群上的 `nvcc` 对你不可执行。**推理不需要编译器**：`export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1`（`cm3p_embed.slurm` 已默认导出，`step0c` 自己也会 `setdefault`）。预检 A 会打印 `dynamo disable = True`；显示 False 说明还在编译路径上。想查 `nvcc` 本身：`which nvcc; ls -l "$(which nvcc)"; nvcc --version` —— `Errno 13`（找到但不可执行）常见于 CUDA 模块没 `module load`、二进制在 `noexec` 挂载的 `/home` 下、或站点没给该用户执行权限；`Errno 2`（No such file）才是根本没装 |
| `Can not execute setup.py since setuptools is not available in the build environment` | `pip install "slider @ git+..."` 的 build isolation 拿不到 setuptools（不是网络问题）。**首选**装仓库自带的 wheel：`pip install --user --no-index --no-deps vendor/slider-0.8.2-py3-none-any.whl`；**次选**（手上没有 vendor/ 时）：`pip install --user -U setuptools wheel` 之后原命令加 `--no-build-isolation` |
| `ImportError: cannot import name 'CommonKwargs' from transformers.processing_utils` | transformers 落在窗口外：**≤4.41 没有这个类，≥5.0 又把它删了**（只有 4.42–4.57 有）。`pip install -U "transformers==4.55.0"`（上游 Dockerfile 的钉法）；升完仍报错就看 `transformers.__file__` 指向哪，多半是 `~/.local` 那份在遮蔽 conda 环境。先 `python -c "import transformers;print(transformers.__file__)"` 看指向；要交给环境管就先 `ls -d ~/.local/lib/python*/site-packages/transformers*` 确认，再 `rm -rf` 掉那两份，最后 `python -m pip install -U "transformers==4.55.0"` |
| `ModuleNotFoundError: slider` | 没装成功。PyPI 上的 `slider` 是**另一个项目**，永远不要 `pip install slider`。作业预检会自己从 vendor/ 装一次；pip 不可用或家目录只读时，它会解包到 `vendor/_unpacked` 并挂到 PYTHONPATH |
| `OSError: We couldn't connect to huggingface.co` | 计算节点离线且权重没预热（§1③），或 `HF_HOME` 提交时和预热时不一致 |
| `sbatch: error: getcwd failed: No such file or directory` | **你当前所在的目录已经被删掉/改名了**（常见于把上传的文件夹又移动或重命名过一次），跟作业脚本无关。`cd ~` 再 `cd <仓库>`，或直接 `REPO=/绝对/路径/DSR sbatch scripts/slurm/cm3p_embed.slurm` |
| `ERROR: submit this job from the repo root` | `SLURM_SUBMIT_DIR` 不是仓库根。要么先 `cd` 到仓库再提交，要么用上面的 `REPO=...` 显式指定 |
| `no such file: manifests/...` | 作业没在仓库里跑。先 `cd` 到仓库再提交，或用 `REPO=...` |
| CUDA OOM | `INFER_BATCH=4`（或 2）重投；只影响吞吐，不影响结果 |
| 大量 `missing`（404/451） | 谱面已删/受限，缓存里永远不会有。`step0g` 标 `cacheable=False`，step0 把谱面连同它的 play 一起剔除，清单在 `whitelist_1k_dropped.csv` |
| 大量 `mismatch` | 谱面被 mapper 改过；文件保留、`checksum_ok=False`，单独统计 |
| 作业到点被杀 | 直接重投，幂等（§4） |
| 改资源配置后**保存脚本卡住**（网页 IDE 显示 "Saving ...: Writing into file..."） | 编辑器的"保存"不是一次 12 KB 写：它通常**写临时文件 → rename → fsync**，随后 watcher/indexer 再 stat 一遍目录；家目录/scratch 是网络文件系统（NFS/Lustre/CephFS），每个元数据操作都要排队。常见诱因：**同一个文件系统上还在跑 rsync/scp 拷贝那 10 GB 的文件夹**、网页 IDE 打开了仓库根目录（`data/osu_cache` 2.2 万个文件 + `data/raw/extracted` 6.5 GB 会引发 stat 风暴）、文件系统本身降级。<br>先量一下：`time dd if=/dev/zero of=~/.wt bs=4k count=1 conv=fsync; rm ~/.wt`、`pgrep -a rsync`、`df -hT ~`、`uptime`。<br>**最快的绕法：根本不改文件** —— `sbatch` 的命令行选项覆盖 `#SBATCH`：<br>`sbatch -p <队列> --time=08:00:00 --mem=64G --gres=gpu:1 -n 8 scripts/slurm/cm3p_embed.slurm`<br>一定要改文件就用 `sed -i` 或 `rsync` 单文件覆盖，别开网页编辑器；也尽量别用网页 IDE 打开仓库根目录（打开 `scripts/slurm/` 即可）。 |

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
