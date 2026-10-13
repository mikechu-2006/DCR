# 远程执行手册：补齐剩余白名单谱面的 CM3P 向量

> 目标：让 21,949 张 4K mania 白名单**每一张都有 `.osu` 正文**（缓存只放谱面本身，
> 不含音频/图片），然后在远程 GPU 上算 CM3P 向量。
> 两种口径：**补算** HF 表没有的 10,072 张（`manifests/cm3p_todo_1k.csv`），
> 或**用完整缓存按同一口径重算全部 21,949 张**（`manifests/whitelist_1k.csv`）。
> 前置方案：[docs/cm3p_step2_integration_plan.md](cm3p_step2_integration_plan.md)；
> 已有实测：[docs/cm3p_step2_results.md](cm3p_step2_results.md)。
> **本手册里的程序在本机只编写、未执行**；计算全部在远程 GPU 机器上做。
> **如果远程是 Slurm 集群（文件夹上传、无 git）**，直接看 [docs/cm3p_hpc_slurm.md](cm3p_hpc_slurm.md)：
> 作业脚本在 `scripts/slurm/`。

---

## 0. 为什么不能在本机做

| 约束 | 本机现状 |
|---|---|
| 缺 `.osu` 正文 | dump 只有元数据；本地缓存在 `data/osu_cache`（**只有 `.osu` 正文**，无音频/图片），远端要自己抓 |
| CM3P 是 22 层 d=768 / 每 16 s 窗最多 4000 token 的 Transformer | 本机 `torch 2.13.0+cpu`、无 CUDA |
| 需要 transformers + HF checkpoint | 本机未装 `transformers` |

仓库里自带的、让远程**从 git clone 就能开跑**的三件东西：

| 文件 | 内容 | 大小 |
|---|---|---|
| `manifests/whitelist_1k.csv` | **全量白名单 21,949** 个 `beatmap_id,checksum,beatmapset_id`（`step0d` 的默认清单） | 1.0 MB |
| `manifests/cm3p_todo_1k.csv` | 其中**尚无 CM3P 向量的 10,072** 个（只补算时用） | 481 KB |
| `manifests/cm3p_probe_1k.csv` | **256** 张**已有向量**的谱面（用于混用探针） | 10 KB |
| `manifests/cm3p_probe_1k.npz` | 上述 256 张的**已发布参考向量** 256×512 float32 | 476 KB |

---

## 1. 远程机器上：clone

    git clone git@github.com:mikechu-2006/DCR.git
    cd DCR          # 远程仓库名是 DCR，本地目录名是 DenseConstantRegressor，同一个仓库

    # 只需要这些；data/ 是空的骨架，这是故意的
    ls manifests/ scripts/step0d_fetch_osu.py scripts/step0c_chart_content.py

**不需要**拷贝本地那 14 GB 数据，也不需要 615 MB 的 244K 预计算表。

---

## 1.5 已经本地验证过的前提（2026-10-09）

用清单里第一张图（`beatmap_id=222593`）实测过下载路径，三条假设全部成立：

| 假设 | 实测 |
|---|---|
| `https://osu.ppy.sh/osu/{id}` 免鉴权可下 | ✅ HTTP 200，50,405 字节，`osu file format v14` |
| 下载到的 `.osu` 带 `BeatmapID`（上游 `BeatmapFilesDataset` 解析时需要，缺了会抛异常） | ✅ `BeatmapID:222593` / `BeatmapSetID:79839` |
| md5 与 2026-09-01 快照一致（**否则说明谱面被改过**） | ✅ `425e8fc376f35e65ac2153264aad8b3a`，与清单逐位相同 |

⇒ 下下来的就是当年被玩的那个版本，`checksum_ok=True` 会成立。少数被 mapper 更新过的图会标 False，
**保留并单独统计，不静默丢**。

---

## 2. 依赖（只装推理侧，**不要**装 requirements.txt）

`requirements.txt` 把 torch 钉在 CPU 版（那是管线 B 的环境），而这里要 GPU。
推理侧依赖的单一来源是仓库根的 `requirements-cm3p.txt`，最小集合：

    # 1) CUDA 版 torch（按机器的驱动选 cu124 / cu121 / cu118）
    pip install torch --index-url https://download.pytorch.org/whl/cu124

    # 2) CM3P 推理侧依赖（= requirements-cm3p.txt）
    pip install -r requirements-cm3p.txt
    #   两个必须钉死的：
    #   * transformers==4.55.0：processing_utils.CommonKwargs 只在 4.42–4.57 存在，
    #     ≤4.41 没有它、≥5.0 又删了它，两头都是同一个 ImportError。
    #   * hydra-core>=1.3,<1.4：上游 config.py 在模块顶层 import hydra（只是 train.py 的
    #     ConfigStore），utils/mmrs_dataset.py → utils/beatmap_files_dataset.py 会把它拽进来，
    #     所以少了它 --source osu-dir 会死在 "No module named 'hydra'"。发行名 hydra-core，
    #     import 名才是 hydra。
    # slider 不要走 GitHub（目标集群连不上），也不要 pip install slider（PyPI 上是另一个项目）：
    pip install --user --no-index --no-deps vendor/slider-0.8.2-py3-none-any.whl

    python -c "import torch;print(torch.__version__, torch.cuda.is_available())"   # 必须是 True

`slider` 是 CM3P 解析 `.osu` 的**解析器本体**（`cm3p/parsing_cm3p.py` 在模块顶层 import 它，stub 不掉），
漏了会在加载 processor 时炸。它只发布在 GitHub，而目标集群连不上 GitHub —— 所以仓库里
**自带编好的纯 Python wheel**（`vendor/`，含 LGPLv3 许可证原文），理由与重建方法见 `vendor/README.md`；
`scripts/slurm/cm3p_embed.slurm` 会在预检里自动装它（pip 不可用时退化为解包挂 PYTHONPATH）。

---

## 3. 执行

### 3.1 一条命令（推荐）

    bash scripts/remote_cm3p_run.sh 2>&1 | tee logs/cm3p_remote_$(date +%F).log

它按顺序做 6 件事：环境自检 → clone CM3P → 装依赖 → 下 `.osu` → **探针 gate** → 全量推理。
每一步都幂等：已正确的 `.osu` 不重下，已算过的 id 不重算，中断后重跑即可。

可用环境变量跳过某几步：`SKIP_INSTALL=1`、`SKIP_FETCH=1`、`SKIP_PROBE=1`（不建议）、`WORKERS=4`、
`FETCH=...`（下哪些，默认全量白名单）、`EMBED=...`（算哪些，默认也是全量白名单）、`MERGE=1`
（算完直接并进规范内容表）。

### 3.2 分步（想看清楚每步在干什么时）

    # ① 下 .osu：默认就是整个白名单（21,949 张，含 256 张探针），可断点续传，
    #    md5 与 2026-09-01 快照核对。缓存里只有 .osu 正文（~60 KB/张），
    #    没有 .osz、没有音频、没有图片。
    python scripts/step0d_fetch_osu.py --workers 6 --delay 0.25
    #    只想补 HF 表缺的那 10,072 张：--ids-file manifests/cm3p_todo_1k.csv
    #    先干跑看看会下什么：加 --dry-run

    # ② 探针：用**与全量完全相同的 no-audio 路径**重算 256 张"已有向量"的谱面
    python scripts/step0c_chart_content.py --source osu-dir \
        --ids-file manifests/cm3p_probe_1k.csv --osu-dir data/osu_cache --audio off \
        --out data/processed/chart_content_cm3p_probe.parquet

    # ③ 混用 gate：这 256 个重算向量和已发布的参考向量像不像？
    python scripts/step0e_validate_mix.py \
        --computed data/processed/chart_content_cm3p_probe.parquet \
        --json data/processed/cm3p_mix_report.json

    # ④ 只有 gate 通过才跑全量。缓存现在是完整的，所以可以按同一口径重算全部：
    python scripts/step0c_chart_content.py --source osu-dir \
        --ids-file manifests/whitelist_1k.csv --osu-dir data/osu_cache --audio off \
        --out data/processed/chart_content_cm3p_local.parquet
    #    只想补 HF 表缺的 10,072 张：--ids-file manifests/cm3p_todo_1k.csv

`--infer-batch`（默认 8，单位是**谱面**）按显存调；`--batch` 在 `osu-dir` 模式下没有意义。

---

## 4. 探针 gate 怎么读（这一步不能跳）

**问题**：已发布的 244K 表是**带音频**算的；远程只有 `.osu`，只能**不带音频**算。
两者是不同的输入分布，直接拼在一起会注入一个与谱面无关的域偏移，而且**在下游指标里看不出来**。

`step0e` 会打印三组数：

| 指标 | 含义 | 怎么判 |
|---|---|---|
| `cosine same`（mean/median/min） | 同一张图，重算 vs 已发布 | ≥0.99 放心；0.90–0.99 存疑；<0.90 不要混 |
| `cosine different`（参考空间） | **不同**谱面之间的余弦 —— 校准用 | 实测约 **0.576**，说明 CM3P 向量本来就挤在一个锥里，**不能拿"0.9 看着挺高"当通过** |
| `Mantel corr` / `recall@1` | 成对相似度相关性 / 最近邻一致性（尺度无关） | Mantel ≥0.8 且 recall@1 ≥0.95 才算几何保住了 |

**三种结论与对应动作**：

* `MIX OK` → 直接合并，两半可以放进同一个实验；
* `MIX WITH CAUTION` → 可以合并，但**任何混合结果都必须同时报探针数字**；
  更稳的做法是拿音频（`.osz`，需要 osu! 账号）把 11,877 张按同一口径重算一遍；
* `DO NOT MIX` → 停。两条路：要么补音频重算，要么把两个子集**分开做实验**、分别报。

探针顺便还是**吞吐标定**：256 张的耗时 ×40 ≈ 全量耗时，先看这个再决定要不要跑全量。

---

## 5. 把结果带回来（在**本地**执行）

    # 从本地机器拉回两个文件
    scp <user>@<host>:<repo>/data/processed/chart_content_cm3p_local.parquet   data/processed/
    scp <user>@<host>:<repo>/data/processed/cm3p_mix_report.json               data/processed/

全量重算的输出**自带 21,949 行**（`source=1`，无音频），两种用法都行：

    # (a) 直接当内容表用：口径完全统一，不涉及 HF 音频行与本地行的混用
    python scripts/step2_predict.py --tag 1k --models mf_dot --dim 32 --epochs 60 \
        --chart-content cm3p --content-path data/processed/chart_content_cm3p_local.parquet

    # (b) 合并进规范表：按 beatmap_id 求并集，新行覆盖旧行 —— 21,949 行全部换成
    #     本地无音频行（HF 那 11,877 行按 id 被覆盖，"覆盖子集 54%"不再存在）
    python scripts/step0c_chart_content.py --source table \
        --in data/processed/chart_content_cm3p_local.parquet --tag 1k

    # 看覆盖率与来源构成
    python -c "import pandas as pd; t=pd.read_parquet('data/processed/chart_content_cm3p.parquet',columns=['beatmap_id','source']); print(len(t)); print(t.source.value_counts())"

`source` 列会区分 `0 = HF 预计算` / `1 = 本地无音频` / `2 = 本地有音频`，
`cm3p_rev` 记录模型 revision —— **这是事后审计"这一行是哪种口径算出来的"的唯一依据，不要丢**。

缓存抓完后，把白名单更新成表，并让 step0 剔除抓不到的谱面（详见
[docs/step0_preprocessing.md](step0_preprocessing.md) §2.4）：

    # ② 白名单表：全部 21,949 张 + 缓存状态列 cached/valid/md5_ok/cacheable
    python scripts/step0g_whitelist.py --tag 1k --cache data/osu_cache \
        --out data/processed/whitelist_1k.csv \
        --manifest-out manifests/whitelist_1k.csv
    #    21,949 行里 cacheable=True 的才进白名单；抓不到的写在
    #    data/processed/whitelist_1k_dropped.csv（含原因 missing/failed/…）

    # ③ 重跑 step0：cacheable=False 的谱面连同它们的 play 一起被剔除，
    #    这样内容表的覆盖率不再是 54%
    python scripts/step0_preprocess.py \
        --dump-dir data/raw/extracted/2026_09_01_performance_mania_top_1000 \
        --whitelist data/processed/whitelist_1k.csv

合并完成后重跑 step2（先过滤后划分的逻辑已经就位）：

    python scripts/step2_predict.py --tag 1k --models mf_dot --dim 32 --epochs 60 --chart-content cm3p
    python scripts/step2_predict.py --tag 1k --models mf_dot --dim 32 --epochs 60 --chart-content cm3p --p-drift time

---

## 6. 资源与时间（估算，用探针实测校准）

| 阶段 | 量 | 估算 |
|---|---|---|
| 下载 `.osu` | 21,949 个请求（只补算时 10,328），6 并发 + 0.25 s 间隔 | **60–100 min**（受限于 osu! 响应；实测 ~2 请求/s，6 并发以上吞吐基本不再涨。请保持礼貌，别把 `--workers` 拉到 16） |
| 磁盘 | `.osu` 缓存（**只有正文**） | 全量 ~1.3 GB（只补算 ~600 MB） |
| CM3P 权重 | `OliBomby/CM3P` | 1–2 GB（HF 缓存） |
| 推理 | ~10.3k 图 × 每图 3–5 个 16 s 窗 ≈ **4 万个窗** | A100 ≈ 20–40 min；4090 ≈ 40–90 min；**CPU ≈ 天级，不要试** |
| 显存 | 22 层 × 4000 token，batch 8 | ~12–16 GB（`--infer-batch` 4 可降到 ~8 GB） |

---

## 7. 故障排查

| 现象 | 原因 / 处理 |
|---|---|
| `ModuleNotFoundError: slider` | 装 `vendor/slider-0.8.2-py3-none-any.whl`（**不要**从 GitHub 装，集群连不上；PyPI 上的同名包是另一个项目），见 §2 与 `vendor/README.md` |
| `No module named 'cm3p'` | CM3P 没 clone 到 `external/cm3p`（`step0c` 会自动 `sys.path.insert` 该目录） |
| CUDA OOM | 降 `--infer-batch`（8 → 4 → 2）；它只影响吞吐，不影响结果 |
| 大量 `missing`（HTTP 404/451） | 谱面已删除或受限，缓存里永远不会有。`step0g` 标 `cacheable=False`，step0 把谱面连同它的 play 一起剔除（清单在 `whitelist_1k_dropped.csv` / `_report.json`），**不会瞎猜** |
| 大量 `mismatch`（md5 不符） | 谱面被 mapper 更新过。文件保留、`checksum_ok=False` 标记，单独统计 —— 不要静默丢弃 |
| 下载一直超时 | `--delay` 调大、`--workers` 调小；已下好的不会重下 |
| HF 下载模型失败 | 设 `HF_ENDPOINT=https://hf-mirror.com` 再试 |

---

## 8. 相关文件一览

| 文件 | 角色 |
|---|---|
| `manifests/whitelist_1k.csv` | 全量白名单 id 清单（21,949 行，`step0d` 的默认输入） |
| `scripts/step0d_fetch_osu.py` | 按清单下 `.osu`（md5 校验、可续传、限流；只下正文） |
| `scripts/step0g_whitelist.py` | 白名单 + 缓存状态表（step0 `--whitelist` 的输入） |
| `scripts/step0c_chart_content.py --source osu-dir` | 跑 CM3P beatmap tower，按 16 s 窗求平均并重新归一化，落内容表 |
| `scripts/step0e_validate_mix.py` | no-audio 与已发布向量能否混用的判定 |
| `scripts/remote_cm3p_run.sh` | 上面三步的一键封装（含 clone + 装依赖） |
| `manifests/*` | 任务清单 + 探针参考向量（仓库自带，clone 即用） |
