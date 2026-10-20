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
step2/*           ->  bias / mf_dot / mirt，或 CM3P 内容向量模型
~~~

旧的 play 级清洗 `step1/clean` 仍保留（显式 `version: clean`），它的 test 是未清洗的
play，不能和 pairbest 的 test 混用。

### 方法论

| 方法 | 入口 | 说明 |
|---|---|---|
| **bias / mf_dot / mirt** | `step2/predict` | oracle 基线：玩家/谱面偏置、玩家×谱面点积、IRT 式链接 |
| **CM3P 线性** | `step2/predict` + 内容表 | 冻结 512 维 CM3P 向量，PCA 后与玩家向量点积 |
| **CM3P 内容锚定 MLP** | `step2/cm3p_mlp` | 内容塔把谱面映射到交互空间，再与玩家向量点积 |
| **MinaCalc** | [`minacalc/`](minacalc/README.md) | 逐区间可微打分 + C++ 区间导出（独立包，不注册为 step） |

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

python main.py -c configs/exp_pairbest_1k.yaml --dry-run
python main.py -c configs/exp_pairbest_1k.yaml --set steps.step2.params.dim=32 --set name=dim32
~~~

一次实验 = 一个 YAML = 一个实验目录；默认新建在 `experiments/<时间戳>[_{name}]/`，
含配置快照、日志、`artifacts/`、`results/`、`manifest.json` 与可重放的 `run.sh`。
编排细节见 [docs/procedure.md](docs/procedure.md)。

### 目录

~~~text
main.py               唯一入口：读 configs/*.yaml，按 step 顺序调用 scripts/
dcr/                  编排包：配置、实验目录、step 注册表、流水线
configs/              实验配置，一个 YAML = 一次实验
scripts/              管线 step + CM3P 运维脚本
preprocess/           step0/step1 的最小可运行参考
minacalc/             MinaCalc 逐区间可微打分（独立包）
docs/                 procedure.md / data.md / models.md
manifests/            白名单清单、CM3P 远程清单、256 图混用探针（入库）
vendor/               HPC 环境无法联网时用的 slider wheel
data/                 数据骨架；实际数据约 14 GB，不入库
experiments/          每次运行的配置、日志、中间产物与结果（整体不入库）
legacy/               退役脚本、归档文档、历史日志与旧结果（含 MANIFEST.tsv）
results/              直接跑 scripts/ 时的默认产物目录
~~~

### 结果速览

测试集按 `split=test` 取，指标为 loss 上的 RMSE / Spearman。下表是 **legacy play 级
clean** 总体（test 是未清洗的 play）：

| 模型 | 1k RMSE | 1k Spearman | 10k RMSE | 10k Spearman |
|---|---|---|---|---|
| bias | 1.5236 | 0.7330 | 1.3727 | 0.6874 |
| mf_dot | 1.4548 | 0.7632 | 1.3327 | 0.7079 |
| **mirt** | **1.4582** | **0.7633** | **1.3331** | **0.7096** |

默认 A 的观测单位与标签支撑不同，RMSE 不能与上表直接比：mf_dot dim=8 在
1k 为 RMSE 0.7015 / Spearman 0.6749 / pp_p50 0.4327，10k 为
0.6844 / 0.6966 / 0.5607。CM3P 的数字见 [docs/models.md](docs/models.md)。

### 文档

| 文档 | 内容 |
|---|---|
| [docs/procedure.md](docs/procedure.md) | **管线与编排**：YAML、实验目录、step 版本、输入解析、扩展 |
| [docs/data.md](docs/data.md) | **数据**：step0 的保留规则与两张表、step1 pairbest / clean、规模与消融 |
| [docs/models.md](docs/models.md) | **模型**：oracle 公式与结果、CM3P 线性/MLP、MinaCalc 入口 |
| [preprocess/README.md](preprocess/README.md) | step0/step1 精简参考与可运行代码 |
| [minacalc/README.md](minacalc/README.md) | MinaCalc 包与 C++ 导出 |
| [legacy/README.md](legacy/README.md) | 归档内容索引；逐文件去向见 [legacy/MANIFEST.tsv](legacy/MANIFEST.tsv) |

### 说明

* **数据不入库**：原始 dump 与中间产物合计约 14 GB，只保留 `data/` 骨架。
* **实验目录不入库**：`experiments/` 整体被忽略；参数与结论请落到 `configs/` 与 `docs/`。
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

**Methods.** Oracle baselines (`bias`, `mf_dot`, `mirt`), CM3P content vectors
(linear dot product and content-anchored MLP), and MinaCalc (the standalone
[`minacalc/`](minacalc/README.md) package). See [docs/models.md](docs/models.md).

**Quick start.**

~~~bash
pip install -r requirements.txt
# download the 1k/10k dumps into data/raw/ (commands above)

python main.py -c configs/exp_pairbest_1k.yaml     # default protocol A
python main.py -c configs/exp_cm3p_1k.yaml         # CM3P content vectors
python main.py -c configs/exp_cm3p_mlp_1k.yaml     # content-anchored MLP
~~~

Each run creates `experiments/<timestamp>[_{name}]/` with `config.yaml`, `logs/`,
`artifacts/`, `results/`, `manifest.json` and a replayable `run.sh`.
See [docs/procedure.md](docs/procedure.md).

Retired scripts, archived docs and historical logs live in
[legacy/](legacy/README.md). No data is tracked; `experiments/` and `external/` are
ignored. License: MIT.
