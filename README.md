# DenseConstantRegressor

仅通过谱面信息计算难度，并在玩家行为数据集上验证。数据集经过激进的清理处理，
包括但不限于删除所有单独的（玩家，谱面）对游玩记录。

现有方法论包括 MinaCalc 以及其衍生、CM3P 谱面向量化，均已在玩家行为数据集上验证。
同时提供 bias、mf_dot 等 oracle 方法以方便比较。

[中文](#中文) · [English](#english)

---

## 中文

### 简介

默认观测单位是**一个 (玩家段, 谱面) 对**（step1/pairbest，口径 A）：

~~~text
step0/preprocess  ->  play 表（5 列）+ beatmap_meta（17 列）
step1/pairbest    ->  玩家 >= 20 条、1 年空窗拆玩家、每对只留最高成绩、
                      单次对剔除、最高 ACC ∈ (0.90, 0.998]、标签 log(1-ACC)、
                      pair-random 70/30
step2/*           ->  bias / mf_dot / minacalc，或 CM3P 内容向量模型
~~~

旧的 play 级清洗 `step1/clean` 仍保留（显式 `version: clean`），它的 test 是未清洗的
play，不能和 pairbest 的 test 混用。

### 两条评价轴

| 轴 | 观测单位 | 预测目标 | 指标 | 入口 |
|---|---|---|---|---|
| **loss 回归** | (玩家段, 谱面) 一行 | `loss = -log(1-ACC)` | RMSE / Spearman | `step2/predict`、`step2/cm3p_mlp` |
| **难度序** | 同一玩家的 (谱面, 谱面) 比较 | 两张图哪张更难 | 成对 BT NLL / `captured` | `scripts/step2_rank_eval.py`、`scripts/step2_cm3p_rank.py` |

两条轴回答不同问题，**数字不能互相比较**：RMSE 可以靠每张谱面的自由截距 `bm` 拿到；
rank-NLL 把 `bm` 换成内容信号后，只评价"序"。定义、归一化与常见坑见
[docs/metrics.md](docs/metrics.md)，实证比较见 [docs/rank_eval.md](docs/rank_eval.md)。

### 方法论

| 方法 | 入口 | 说明 |
|---|---|---|
| **bias / mf_dot / mirt** | `step2/predict` | oracle 基线：玩家/谱面偏置、玩家×谱面点积、IRT 式链接 |
| **CM3P 线性** | `step2/predict` + 内容表 | 冻结 512 维 CM3P 向量，PCA 后与玩家向量点积 |
| **CM3P 内容锚定 MLP** | `step2/cm3p_mlp` | 内容塔把谱面映射到交互空间，再与玩家向量点积 |
| **MinaCalc** | [`minacalc/`](minacalc/README.md) | 逐区间可微打分 + C++ 区间导出（独立包，不注册为 step） |
| **rank-NLL 评测** | `scripts/step2_rank_eval.py` | 难度序的成对 BT NLL；cross-fitted ceiling / captured；可作 step2 训练损失 |
| **CM3P → 难度序** | `scripts/step2_cm3p_rank.py` | 只从 CM3P 内容学习 per-chart 难度 d_m，按 rank-NLL 训练 |

口径与主结果见 [docs/data.md](docs/data.md) 与 [docs/models.md](docs/models.md)。

### 快速开始

~~~bash
pip install -r requirements.txt        # CM3P 推理另见 requirements-cm3p.txt

# 数据（约 14 GB，不入库）
cd data/raw
curl -LO https://data.ppy.sh/2026_09_01_performance_mania_top_1000.tar.bz2
curl -LO https://data.ppy.sh/2026_09_01_performance_mania_top_10000.tar.bz2
tar -xjf 2026_09_01_performance_mania_top_1000.tar.bz2  -C extracted/
tar -xjf 2026_09_01_performance_mania_top_10000.tar.bz2 -C extracted_10k/
cd ../..

python main.py -c configs/exp_pairbest_1k.yaml    # 默认 A + bias/mf_dot
python main.py -c configs/exp_pairbest_10k.yaml   # 默认 A，10k
python main.py -c configs/exp_cm3p_1k.yaml        # CM3P 内容向量（线性）
python main.py -c configs/exp_cm3p_mlp_1k.yaml    # CM3P 内容锚定 MLP
python main.py -c configs/exp_full_1k.yaml        # raw dump 全流程（legacy clean）

# 自检（合成张量，不需要 data/）
python scripts/test_rank_metrics.py               # rank-NLL / BT / ceiling
python scripts/test_step2_cm3p_mlp.py             # CM3P MLP
python scripts/test_step2_rank_eval.py            # rank runner 端到端

# 难度序：1k pairbest 上的 MinaCalc
python scripts/step2_rank_eval.py \
    --in data/processed/step1_1k_pairbest.parquet \
    --signal-table data/processed/rank_signals_1k.parquet \
    --signals msd_overall --split all --min-l 10 --folds 2 --bootstrap 300 \
    --out results/rank_eval_1k/msd.json

# CM3P → per-chart 难度（--quick 只做烟测，完整约 30 min CPU）
python scripts/run_cm3p_rank_experiments.py --quick
~~~

一次实验 = 一个 YAML = 一个实验目录；默认新建在 `experiments/<时间戳>[_{name}]/`，
含配置快照、日志、`artifacts/`、`results/`、`manifest.json` 与可重放的 `run.sh`。
编排细节见 [docs/procedure.md](docs/procedure.md)；实验目录的本地索引可用
`python scripts/index_experiments.py` 生成。

### 目录

~~~text
main.py               唯一入口：读 configs/*.yaml，按 step 顺序调用 scripts/
dcr/                  编排包：配置、实验目录、step 注册表、流水线、rank 指标
configs/              实验配置，一个 YAML = 一次实验
scripts/              管线 step、rank 评测/训练与运维脚本（见 scripts/README.md）
preprocess/           step0/step1 的最小可运行参考
minacalc/             MinaCalc 逐区间可微打分（独立包）
docs/                 procedure / data / models / metrics / rank_eval / cm3p_rank
manifests/            白名单清单、CM3P 远程清单、256 图混用探针（入库）
vendor/               HPC 环境无法联网时用的 slider wheel
data/                 数据骨架；实际数据约 14 GB，不入库
experiments/          每次运行的配置、日志、中间产物与结果（整体不入库）
results/              可直接引用的结果；索引见 results/README.md
legacy/               退役脚本、归档文档、历史日志、scratch 探针（含 MANIFEST.tsv）
~~~

### 结果速览

#### 1. loss 回归（legacy play 级 clean）

测试集按 `split=test` 取，指标为 loss 上的 RMSE / Spearman：

| 模型 | 1k RMSE | 1k Spearman | 10k RMSE | 10k Spearman |
|---|---|---|---|---|
| bias | 1.5236 | 0.7330 | 1.3727 | 0.6874 |
| mf_dot | 1.4548 | 0.7632 | 1.3327 | 0.7079 |
| **mirt** | **1.4582** | **0.7633** | **1.3331** | **0.7096** |

#### 2. loss 回归（pairbest，口径 A）

标签支撑与观测单位不同，RMSE **不能**和上表直接比。mf_dot dim=8：

| tag | RMSE | Spearman | pp_p50 |
|---|---|---|---|
| 1k | 0.7015 | 0.6749 | 0.4327 |
| 10k | 0.6844 | 0.6966 | 0.5607 |

CM3P 线性（clean 口径、覆盖子集）为 1.57595；更细的表与消融见
[docs/models.md](docs/models.md)。

#### 3. 难度序 rank-NLL（1k pairbest，12.48M 次比较，ceiling 0.56288）

| 信号 | 来源 | NLL | captured |
|---|---|---|---|
| `daniel_star` | osumania_map_analyser | **0.63156** | **0.4728** |
| `sunny_star` | osumania_map_analyser | 0.63165 | 0.4721 |
| `ett_068_overall` | MinaCalc 0.68 | 0.63506 | 0.4459 |
| `msd_overall` | 本地 MinaCalc 505 | 0.64017 | 0.4067 |
| `star` | osu! star rating | 0.66085 | 0.2479 |

全部 full-coverage 信号共享同一 pair 集与 ceiling，可以直接比；LN / RC 子集、
逐玩家相关与置信区间见 [docs/rank_eval.md](docs/rank_eval.md) §4。

#### 4. CM3P 学到的难度序

| 口径 | 模型 | NLL | captured |
|---|---|---|---|
| 新玩家 val | CM3P + external MLP | **0.5476** | 1.40* |
| 新玩家 val | CM3P MLP h64 L2 | 0.5680 | 1.20* |
| 新玩家 val | CM3P 线性 | 0.6021 | 0.88 |
| 新玩家 val | Sunny（固定信号） | 0.6367 | 0.54 |
| 新 chart P2 | CM3P + external 线性 | **0.5756** | **0.9013** |
| 新 chart P2 | CM3P 线性 | 0.6148 | 0.6006 |
| 新 chart P2 | CM3P MLP h64 L2 | 0.6152 | 0.5976 |

\* val ceiling（0.58909）本身欠拟合，captured >1 不代表"超过上界"；比较模型看 NLL 与
paired ΔNLL。P1（同玩家、同 chart 的新 pair）的数字不能当泛化结论。完整表、消融、
negative controls 与 paired Δ 见 [docs/cm3p_rank.md](docs/cm3p_rank.md)。

P2 口径的 per-chart Pearson(d,b)（同 4,565 张 test 图、同 ceiling `b`）：CM3P + external
线性 **0.6025**、CM3P MLP 0.5613、CM3P 线性 0.5546、Sunny 0.3713；Pearson 与 NLL 基本
同序（MLP 与线性在 Pearson 上互换），只作诊断，主指标仍是 NLL / captured（§5.3）。

### 文档

| 文档 | 内容 |
|---|---|
| [docs/procedure.md](docs/procedure.md) | **管线与编排**：YAML、实验目录、step 版本、输入解析、扩展 |
| [docs/data.md](docs/data.md) | **数据**：step0 的保留规则与两张表、step1 pairbest / clean、规模与消融 |
| [docs/models.md](docs/models.md) | **模型**：oracle 公式与结果、CM3P 线性/MLP、MinaCalc 入口 |
| [docs/metrics.md](docs/metrics.md) | **指标**：难度序的成对 NLL、cross-fitted ceiling、captured、逐玩家 Spearman、训练目标 |
| [docs/rank_eval.md](docs/rank_eval.md) | **实验**：MinaCalc / osumania_map_analyser（含 LN）的难度序评测 |
| [docs/cm3p_rank.md](docs/cm3p_rank.md) | **实验**：CM3P → per-chart 难度序；M0–M2、P1/P2、new-player val、content ceiling |
| [results/README.md](results/README.md) | **结果索引**：每个产物是什么、由哪条命令生成、在哪篇文档里解读 |
| [scripts/README.md](scripts/README.md) | **脚本索引**：step、rank 训练/评测、分析与运维脚本 |
| [preprocess/README.md](preprocess/README.md) | step0/step1 精简参考与可运行代码 |
| [minacalc/README.md](minacalc/README.md) | MinaCalc 包与 C++ 导出 |
| [legacy/README.md](legacy/README.md) | 归档内容索引；逐文件去向见 [legacy/MANIFEST.tsv](legacy/MANIFEST.tsv) |

### 说明

* **数据不入库**：原始 dump 与中间产物合计约 14 GB，只保留 `data/` 骨架。
* **实验目录不入库**：`experiments/` 整体被忽略；本机可用
  `python scripts/index_experiments.py` 生成 `experiments/INDEX.md`。
  参数与结论请落到 `configs/` 与 `docs/`。
* **结果入库规则**：JSON / CSV / Markdown / PNG 入库，`*.npz`、`*.parquet`、
  `*.npy` 不入库（可由命令重建）；见 [results/README.md](results/README.md)。
* **`external/` 不入库**：`minacalc-standalone`、`cm3p`、`bobert` 都是独立克隆。
* Python 3.13；依赖见 [requirements.txt](requirements.txt)，CM3P GPU 侧见
  [requirements-cm3p.txt](requirements-cm3p.txt)。

### 许可

MIT

---

## English

Compute difficulty from **beatmap information only**, validated on a player-behaviour
dataset that is aggressively cleaned: every play row belonging to a single
(player, beatmap) pair is removed, so the default observation unit is one
**(player-epoch, beatmap) pair** (step1/pairbest, protocol A).

| Step | Default version | Artifact |
|---|---|---|
| **0 Preprocess** | `step0/preprocess` | play table (5 cols) + beatmap metadata (17 cols) |
| **1 Pair-best clean (A)** | `step1/pairbest` | one row per pair; see [docs/data.md](docs/data.md) |
| **1' Play-level clean** | `step1/clean` | play table + `split` column; requires `version: clean` |
| **2 Models** | `step2/predict`, `step2/cm3p_mlp` | oracles and CM3P content-vector models |
| **Rank eval** | `scripts/step2_rank_eval.py` | pairwise BT NLL for a per-chart difficulty |

**Two evaluation axes.** *Loss regression* predicts `log(1-ACC)` per (player, beatmap)
row and reports RMSE / Spearman. *Rank-NLL* asks whether a difficulty ordering predicts
which of two charts the same player performs better on; it is normalized by a
cross-fitted Bradley-Terry ceiling (captured). The two axes answer different questions
and their numbers are not comparable; see [docs/metrics.md](docs/metrics.md).

**Methods.** Oracle baselines (`bias`, `mf_dot`, `mirt`), CM3P content vectors
(linear dot product and content-anchored MLP), MinaCalc (the standalone
[`minacalc/`](minacalc/README.md) package), and CM3P-based per-chart difficulty learned
directly against rank-NLL (`scripts/step2_cm3p_rank.py`). The 1k rank-NLL results for
MinaCalc and osumania_map_analyser are in [docs/rank_eval.md](docs/rank_eval.md); the CM3P
difficulty experiments are in [docs/cm3p_rank.md](docs/cm3p_rank.md).

**Quick start.**

~~~bash
pip install -r requirements.txt
# download the 1k/10k dumps into data/raw/ (commands above)

python main.py -c configs/exp_pairbest_1k.yaml     # default protocol A
python main.py -c configs/exp_cm3p_1k.yaml         # CM3P content vectors
python main.py -c configs/exp_cm3p_mlp_1k.yaml     # content-anchored MLP

python scripts/test_rank_metrics.py                # rank-NLL self-tests
python scripts/run_cm3p_rank_experiments.py --quick
~~~

Each run creates `experiments/<timestamp>[_{name}]/` with `config.yaml`, `logs/`,
`artifacts/`, `results/`, `manifest.json` and a replayable `run.sh`.
See [docs/procedure.md](docs/procedure.md).

**Results.** The headline numbers are in the tables above: play-level clean RMSE
(bias 1.5236 / mf_dot 1.4548 / mirt 1.4582 on 1k), pairbest protocol A
(mf_dot dim=8: 0.7015 on 1k), rank-NLL captured for MinaCalc `msd_overall` (0.4067)
and for the strongest analyser signals (0.47), and CM3P per-chart difficulty
(player-disjoint val NLL 0.5476 with external features; chart-disjoint captured 0.9013,
per-chart Pearson against the BT scale 0.6025 vs 0.3713 for Sunny).
Every artifact is indexed in [results/README.md](results/README.md).

Retired scripts, archived docs, historical logs and the rank-NLL development scratch
live in [legacy/](legacy/README.md). No data is tracked; `experiments/` and
`external/` are ignored. License: MIT.
