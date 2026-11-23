# DenseConstantRegressor

仅通过谱面信息计算难度，并在玩家行为数据集上验证。数据集经过激进的清理处理，
包括但不限于删除所有单独的（玩家，谱面）对游玩记录。

现有方法论包括 MinaCalc 及其衍生、CM3P 谱面向量化、data2vec 自监督内容编码器，
以及直接对难度序训练的 per-chart 潜变量；同时提供 bias、mf_dot 等 oracle 方法以方便比较。

> **2026-11-23**：1k/10k pairbest 主线稳定；新增 randommerge 合并集（20,388 图）、
> VSRG Dan Benchmark 外部评测、Koishi/ISOR 信号复测与 data2vec（route B）首轮结果。
> 不同数据集 / 不同 ceiling 的绝对 NLL 不可互比，各节都标注了口径。

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

| 集合 | 规模 | 用途 |
|---|---|---|
| **1k pairbest** | 6,828 图 / 109,706 对 | 快速迭代、1k 主表、P2/P4 |
| **10k pairbest** | 16,061 图 / 1,513,746 对 / 10,266 玩家段 | 全量 rank-NLL、Koishi/ISOR |
| **randommerge** | 453,406 对 / 20,388 图 / 12,758 玩家 | 补低难度覆盖、窗口池化、外部 benchmark 训练集 |
| **VSRG benchmark** | 1,382 图（1,146 有 bid） | 唯一带外部 Dan 标签的留出评测 |

1k/10k 来自 top-1000/top-10000 玩家 dump；randommerge 合并 4 个
`performance_mania_random_10000` 快照（2026-07-13 / 08-01 / 09-01 / 11-05），在 step0
的 play 行上合并后重建全部对（**0 个对因合并“复活”**），单个 random 快照的 ≤3★ 对
占比 23.4%，而 top-10k 只有 1.36%。VSRG benchmark 的 1,382 行只用于外部评测；
它与 randommerge 训练集有 201 个重叠 bid，第 6 节的 RM-L 结果给了排除重叠后的数字，
benchmark 的 MAE 也不能和 DCR 的 RMSE / NLL 换算。

### 两条评价轴

| 轴 | 观测单位 | 预测目标 | 指标 | 入口 |
|---|---|---|---|---|
| **loss 回归** | (玩家段, 谱面) 一行 | `loss = -log(1-ACC)` | RMSE / Spearman | `step2/predict`、`step2/cm3p_mlp` |
| **难度序** | 同一玩家的 (谱面, 谱面) 比较 | 两张图哪张更难 | 成对 BT NLL / `captured` | `scripts/step2_rank_eval.py`、`scripts/step2_cm3p_rank.py` |

两条轴回答不同问题，**数字不能互相比较**：RMSE 可以靠每张谱面的自由截距 `bm` 拿到；
rank-NLL 把 `bm` 换成内容信号后，只评价“序”。定义、归一化与常见坑见
[docs/metrics.md](docs/metrics.md)，实证比较见 [docs/rank_eval.md](docs/rank_eval.md)。

外部 benchmark 的 **Dan MAE** 是独立的第三个评测口径（外部人工标签），它依赖 benchmark 的
`expected` 做校准；这个数字与 RMSE、rank-NLL 都不可互比。

### 主要结论

* **1k P2/P4**：CM3P + external MLP 在 5 个划分种子上稳定最好（test NLL **0.5660 / 0.5734**）；
  完整 audio-on 表进一步到 **0.5585 / 0.5673**。
* **randommerge**：external 线性最好（**0.5833 / 0.5827**）；1k 上 MLP 的优势**没有迁移**
  （MLP +0.0050 / +0.0091，5/5 更差）。加 `log_n` 有 5/5 的小幅增益。
* **VSRG Dan Benchmark**：所有 CM3P 系组合都**没有超过 isotonic Sunny（0.3367）**；
  最好的 RM-L 为 **0.4216**，仍差 0.085。published 算法里 Koishi 0.1540 最强。
* **Koishi / ISOR**：在 benchmark 上的 Dan-MAE 优势没有迁移到 DCR 的 ACC 序；
  Koishi 只覆盖 6.57% 的图，ISOR 在 randommerge 上反而更差（+0.0153）。
* **data2vec（route B）**：P2 0.5863、P4 0.5930，优于 canon+线性，但**仍不及**
  legacy CM3P + external MLP，且 P4 主口径含 transductive 成分。
* **canon 裁剪**：token −43.7%、截断 0.42%→0，但 rank 上 39/40 个组合变差；
  主线继续用 legacy 内容表。

### 方法论

| 方法 | 入口 | 说明 |
|---|---|---|
| **bias / mf_dot / mirt** | `step2/predict` | oracle 基线：玩家/谱面偏置、玩家×谱面点积、IRT 式链接 |
| **CM3P 线性** | `step2/predict` + 内容表 | 冻结 512 维 CM3P 向量，PCA 后与玩家向量点积 |
| **CM3P 内容锚定 MLP** | `step2/cm3p_mlp` | 内容塔把谱面映射到交互空间，再与玩家向量点积 |
| **data2vec（route B）** | [`data2vec_main/`](data2vec_main/README.md) | 100 Hz×8 通道 piano roll → 50 Hz/512d；span-mask Transformer + EMA teacher；产出 512d 内容向量 |
| **MinaCalc** | [`minacalc/`](minacalc/README.md) | 逐区间可微打分 + C++ 区间导出（独立包，不注册为 step） |
| **rank-NLL 评测** | `scripts/step2_rank_eval.py` | 难度序的成对 BT NLL；cross-fitted ceiling / captured；可作 step2 训练损失 |
| **CM3P → 难度序** | `scripts/step2_cm3p_rank.py` | 只从 CM3P 内容学习 per-chart 难度 d_m，按 rank-NLL 训练 |
| **窗口池化 A0–A6** | `scripts/cm3p_pool.py` | 16 s 窗口的均值 / 分位 / TCN；randommerge 上 A0–A3 已测，A4–A6 未跑 |

内容表有两种：**legacy**（audio-off，512 维，主线）与 **canon**（裁剪 hitsound/volume、
kiai、mania SV）。canon 省 43.7% token、便于缓存，但在 rank 目标上一致变差（见下文
“canon 裁剪”），所以主线仍用 legacy。

### 快速开始

~~~bash
pip install -r requirements.txt        # CM3P 推理另见 requirements-cm3p.txt

# 数据（原始 dump 约 13 GB；全部本地数据约 18 GB，不入库）
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

# data2vec（route B）自检；完整训练 / 抽取见 data2vec_main/README.md
cd data2vec_main && python scripts/smoke_test.py && cd ..
~~~

一次实验 = 一个 YAML = 一个实验目录；默认新建在 `experiments/<时间戳>[_{name}]/`，
含配置快照、日志、`artifacts/`、`results/`、`manifest.json` 与可重放的 `run.sh`。
编排细节见 [docs/procedure.md](docs/procedure.md)；实验目录的本地索引可用
`python scripts/index_experiments.py` 生成。randommerge、benchmark 与 data2vec 的
完整入口分别见各自文档。

### 目录

~~~text
main.py               唯一入口：读 configs/*.yaml，按 step 顺序调用 scripts/
dcr/                  编排包：配置、实验目录、step 注册表、流水线、rank 指标
configs/              实验配置，一个 YAML = 一次实验
scripts/              管线 step、rank 评测/训练与运维脚本（见 scripts/README.md）
preprocess/           step0/step1 的最小可运行参考
minacalc/             MinaCalc 逐区间可微打分（独立包）
data2vec_main/        route B：audio-style data2vec 的训练、抽取与 probe（独立子项目）
data2vec/             data2vec 论文与原始实现笔记
docs/                 procedure / data / models / metrics / rank_eval / cm3p_rank /
                      randommerge / benchmark / data2vec
manifests/            白名单清单、CM3P 远程清单、256 图混用探针（入库）
vendor/               HPC 环境无法联网时用的 slider wheel
data/                 数据骨架；实际数据约 18 GB，不入库
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

#### 4. CM3P 学到的难度序（1k）

内容表为 legacy（audio-off）512 维。P1 = 同玩家、同 chart 的新 pair；P2 = 新 chart
（chart-disjoint）；new-player val = 玩家不相交的验证集。下表是**单划分**
（split seed 20260928）的旧表；P2 三行未含 external MLP，只作历史对照：

| 口径 | 模型 | NLL | captured |
|---|---|---|---|
| 新玩家 val | CM3P + external MLP | **0.5476** | 1.40* |
| 新玩家 val | CM3P MLP h64 L2 | 0.5680 | 1.20* |
| 新玩家 val | CM3P 线性 | 0.6021 | 0.88 |
| 新玩家 val | Sunny（固定信号） | 0.6367 | 0.54 |
| 新 chart P2 | CM3P + external 线性 | **0.5756** | **0.9013** |
| 新 chart P2 | CM3P 线性 | 0.6148 | 0.6006 |
| 新 chart P2 | CM3P MLP h64 L2 | 0.6152 | 0.5976 |

\* val ceiling（0.58909）本身欠拟合，captured >1 不代表“超过上界”；比较模型看 NLL 与
paired ΔNLL。P1（同玩家、同 chart 的新 pair）的数字不能当泛化结论。完整表、消融、
negative controls 与 paired Δ 见 [docs/cm3p_rank.md](docs/cm3p_rank.md)。

**P2/P4 5 个划分种子复核**（种子只改 train/test 划分，模型种子固定 0；详见
[docs/p4_multiseed.md](docs/p4_multiseed.md)）：

| 口径 | 模型 | test NLL | Pearson(d,b) | Spearman(d,b) |
|---|---|---|---|---|
| P2 新 chart | **CM3P + external MLP** | **0.5660 ± 0.0031** | **0.6281 ± 0.0081** | **0.5853 ± 0.0101** |
| P2 新 chart | CM3P + external 线性 | 0.5779 ± 0.0033 | 0.5822 ± 0.0122 | 0.5523 ± 0.0086 |
| P4 双重冷启动 | **CM3P + external MLP** | **0.5734 ± 0.0043** | **0.5881 ± 0.0264** | **0.5426 ± 0.0232** |
| P4 双重冷启动 | CM3P + external 线性 | 0.5848 ± 0.0049 | 0.5374 ± 0.0391 | 0.5094 ± 0.0326 |

CM3P-only 线性 / MLP 在 P2 为 0.6162 / 0.6166、P4 为 0.6241 / 0.6254（MLP **没有增益**）；
Sunny / Daniel 约 0.628–0.631、star 约 0.660。**CM3P + external MLP 是 1k 上唯一在两个
口径都稳定胜出 external 线性的方法**（paired Δ -0.0118 / -0.0114，5/5 种子）；P4 只比 P2
差 ~0.007–0.009 nats，但 test 集与 ceiling 不同，**绝对 NLL 不可跨口径比较**。

**完整 audio-on 表复跑（2026-11-19）**：published 4,037 张 + HPC 新算 2,782 张合并成
6,819 张，与新划分的 audio-off legacy 做配对对照；4 个 embedding 方法在两个口径上全部
变好（ΔNLL **−0.0076 … −0.0218**，8/8 个 protocol×method 组合 **5/5 种子为负**），
CM3P + external MLP 仍最好（P2 **0.5585**、P4 **0.5673**）。published 向量用官方
processor 复现：98 张可编码图 cos 均值 **0.99745**、95/98 ≥0.99（
[docs/cm3p_official_probe.md](docs/cm3p_official_probe.md)）。详见
[docs/p4_multiseed_audio.md](docs/p4_multiseed_audio.md)。

**canon 裁剪（2026-11-09 / 11-20）**：固定 checkpoint 配置（`add_snapping=False`、
`add_cls=True`、`max_length=2000`）只删 hitsound/volume、kiai、mania SV；400 图审计
token/图 8,041→4,524（**−43.7%**）、4.90→2.76 token/note、撞 2,000 上限的窗口
18/4,251→0。同一协议只换内容表，4 个 embedding 方法在 P2/P4 **一致变差**
（ΔNLL **+0.0047…+0.0208**，40 个 seed×方法组合 39 个为正；external MLP
P2 0.56602→0.58574、P4 0.57343→0.59425）。**rank 目标继续用 legacy；canon 的价值是
token −43.7% 与可缓存，不是分数。** 详见 [docs/cm3p_crop_windows.md](docs/cm3p_crop_windows.md)、
[docs/p4_multiseed_canon.md](docs/p4_multiseed_canon.md)。

#### 5. randommerge 合并集（20,388 图）

把 4 个 random 快照在 **step0 的 play 行**合并，再重建全部对：
4,536,108 play / 25,105 玩家 / 22,200 图 / 2,999,619 对；step1 输出 453,601 对 /
12,759 玩家；内容过滤后 **453,406 对 / 20,388 图 / 12,758 玩家**。合并**没有复活任何
单次对**（0 个），收益主要是覆盖与规模（重复 play 2.8%）。数据报告见
[docs/random_merge_report.md](docs/random_merge_report.md)。

低于 3★ 的样本在 top-10k 里只有 **1.36%**，被抽样和 `ACC>0.998` 清洗一起删掉；
random 快照是现成解药（单个快照 ≤3★ 占 23.4%）。**不要**用 osu! API 批量爬 best：
违反使用条款，而且 best 只有通过局，与 pairbest 口径不兼容（
[docs/low_difficulty_data.md](docs/low_difficulty_data.md)）。

模型均为 `linear_extra`（λ1e-3，内容表 legacy 512 维）；P2/P4 各 5 个 split seed，
模型 seed 固定 0：

| 模型 | P2 NLL | P4 NLL | 说明 |
|---|---:|---:|---|
| legacy + msd/sunny（L0） | 0.58326 ± 0.00145 | 0.58271 ± 0.00514 | 基线 |
| **+ log_n（L1）** | **0.58247 ± 0.00153** | **0.58202 ± 0.00513** | Δ −0.00079 / −0.00069，5/5 |
| + log_n + OD（L3） | 0.58234 ± 0.00148 | 0.58196 ± 0.00518 | 再多 −0.00013 / −0.00006；OD 近噪声 |
| MLP + external | 0.58824 ± 0.00153 | 0.59178 ± 0.00502 | **+0.0050 / +0.0091，5/5 更差** |
| Sunny | 0.60448 ± 0.00126 | 0.60390 ± 0.00666 | 固定信号 |

canon 窗口池化 A0–A3 也测了：A0L 0.59504 / 0.59611、A2b 0.59322 / 0.59680
（A2b − A0L：P2 −0.00182、5/5 更好，P4 +0.00069、5/5 更差）；但仍比 legacy 差
**0.00996 / 0.01409**，canon 的 A0 相对 legacy 的代价是 **+0.01254 / +0.01418**。
A4–A6（softmax / minacalc / TCN）未跑，**不要引用 T3 结论**
（[docs/randommerge_pool_results.md](docs/randommerge_pool_results.md)）。

**结论**：合并集上 **external 线性最好，MLP 一致更差**；P4 比 P2 低 0.00054
（1k 上 P4 反而差 +0.007–0.009），冷启动代价在大图集上变小。captured 因 ceiling
欠拟合为 1.10–1.23，**不可引用**；只报 NLL / Pearson / Spearman。现行推荐是
`linear_extra --extra-signals msd_overall,sunny_star,log_n`（P2 **0.58247**、
P4 **0.58202**）。

#### 6. 外部基准：VSRG Dan Benchmark

1,382 张 4K 谱面（RC 1,246 / LN 102 / course 34；1,146 行有 beatmap id），
benchmark 快照 `e15f89b`；5 折按 chart-group 的 LOCV（同 bid 的 rate 变体同组），
isotonic 校准为主、linear 为对照，全部 out-of-fold。M1 = CM3P + external MLP，
M2 = CM3P MLP；L = legacy 内容表，C = canon；RM-L = randommerge `linear_extra`。

| 方法 | MAE | MAE 95% CI | Spearman |
|---|---:|---|---:|
| M1-L | 0.7411 | [0.6810, 0.8054] | 0.8437 |
| M1-C | 0.7918 | [0.7249, 0.8688] | 0.8450 |
| M2-L | 1.1489 | [1.0728, 1.2388] | 0.6609 |
| M2-C | 1.1248 | [1.0442, 1.2185] | 0.6823 |
| **RM-L（randommerge）** | **0.4216** | **[0.3965, 0.4481]** | **0.8652** |
| `sunny_star` isotonic | **0.3367** | [0.3155, 0.3587] | 0.9060 |
| `msd_overall` isotonic | 0.6286 | [0.5882, 0.6697] | 0.7849 |
| [Sunny, MSD] 线性 | 0.6568 | [0.6297, 0.6883] | 0.9021 |
| 常数 median | 1.9195 | [1.8151, 2.0377] | −0.0094 |

**所有 CM3P 系组合都没有超过 isotonic Sunny。** RM-L 比旧的 M1-L 大幅改善
（0.7411→0.4216，paired Δ −0.3195 [−0.3802, −0.2658]；排除 201 个训练重叠 bid 后
0.4419，Δ −0.3545），但仍比 Sunny 差 +0.0849 [0.0672, 0.1038]。改善集中在低/中难度：
`expected<11` 的 192 行 Δ −1.523、11–15 的 1,006 行 Δ −0.143、>15 的 184 行
Δ −0.031（CI 跨 0）；按 msd 分带 Q4 也只有 −0.001（CI 跨 0）。RC 0.3655、
LN 1.0733、course 0.5227（旧 M1-L 的 LN/course 是 2.11 / 2.25）。

按 `expected >= 12` 在子集内**重拟合**（post-hoc，1,142 行）后：M1-L **0.3194**、
M1-C **0.3108**、Sunny **0.2396**、MSD **0.3990**；M1-L 胜 MSD 0.0796、负 Sunny
0.0798。全量 0.7411 主要由 `expected<12` 的 240 行（MAE 2.3815）拉动。

published 算法（覆盖率不同，80–100%）：Koishi **0.1540**、XDC1 0.2096、Roxy 0.2125、
Azusa 0.2628、ISOR 0.2672、Mixed 0.2879、Daniel 0.4224、Companella 0.4934、
Sunny 0.5763、P1h 0.8314。RM-L 0.4216 与 Daniel 同量级，仍落后前六个。结论与限制见
[docs/benchmark_vsrg.md](docs/benchmark_vsrg.md)、
[docs/benchmark_linear_extra.md](docs/benchmark_linear_extra.md)。

#### 7. Koishi / ISOR 在 DCR 难度序上的复测

信号来源是 benchmark 的 Dan 估计器：ISOR v3.1、Koishi v5.1；在 DCR 的 ACC 序上算
rank-NLL，与 `sunny_star` 对比。**主看 ΔNLL，不跨图集比绝对值。**

| 图集 | Sunny NLL | 信号 NLL | ΔNLL | 覆盖 |
|---|---:|---:|---:|---|
| 10K ALL（16,061 图） | 0.60673 | ISOR 0.60527 | −0.00146 | 100% |
| 10K RC20（9,928 图） | 0.61720 | ISOR 0.61459 | −0.00261 | 100% |
| 10K LN（6,133 图） | 0.57411 | ISOR 0.57857 | **+0.00446** | 100% |
| 10K Koishi-native（1,056 图） | 0.54964 | Koishi 0.54518 | −0.00446 | **6.57% 图** |
| randommerge ALL（20,388 图） | 0.60510 | ISOR 0.62041 | **+0.01531** | 100% |

10K 全量与 RC20 上 ISOR 的 NLL 略低，但 Spearman / 逐玩家相关也略低；LN 是
out-of-domain，ISOR 更差。**benchmark 上的 Dan-MAE 优势没有迁移到 DCR 的 ACC 序**：
Koishi 只在 1,056 张高难图上可用，randommerge 上 ISOR 全面更差。本轮没有外部 Dan
留出集，不能证明“对 Dan 标签过拟合”；也没有 bootstrap CI。详细口径见
[docs/koishi_isor_report.md](docs/koishi_isor_report.md)。

#### 8. data2vec（route B，from-scratch audio-style）

把 4K 谱面当成连续信号：100 Hz × 8 通道 piano roll（4 列 onset + 4 列 hold）→
共享 1D conv（stride 2）→ 50 Hz × 512d → span mask（p=0.065、长 10 帧，mask ratio
约 0.49）→ 12 层 / 512d student Transformer，回归 EMA teacher 顶层 6 层 LayerNorm
平均表示（仅 masked 帧，Smooth-L1 β=1.0）。没有离散词表，也没有 8-bit code。

holdout 协议用 **20,584** 张图训练 20 epoch / 25,720 step（1×A800，47m02s，
9.114 step/s，loss 0.524→0.054）；为 **21,949** 张图抽取 512d 向量（218,525 个
16 s 窗口，L2=1.0）。预训练与抽取的完整命令见
[data2vec_main/README.md](data2vec_main/README.md)。

**P2**（5 折 chart-disjoint，linear，λ1e-3）：

| 口径 | D2V NLL | D2V+ext NLL | CM3P canon NLL | CM3P canon+ext NLL |
|---|---:|---:|---:|---:|
| 全 6,828 图 | **0.5863** | **0.5810** | 0.6240 | 0.6009 |
| 仅 1,365 holdout 图 | **0.5971** | **0.5939** | 0.6370 | 0.6167 |

**P4**（新玩家 + 新图，5 seeds，model seed 0；ceiling 0.56540 ± 0.00859）：

| 方法 | P4 NLL | captured | Pearson(d,b) |
|---|---:|---:|---:|
| D2V 线性 | 0.59301 ± 0.00417 | 0.7859 | 0.5373 |
| **D2V + external 线性** | **0.58561 ± 0.00459** | **0.8437** | **0.5411** |
| D2V MLP | 0.59681 ± 0.00281 | 0.7567 | 0.5283 |
| D2V MLP + external | 0.58605 ± 0.00460 | 0.8403 | 0.5310 |
| CM3P canon 线性 | 0.63050 ± 0.00270 | 0.4914 | 0.4520 |
| CM3P canon + external | 0.60335 ± 0.00515 | 0.7037 | 0.4892 |
| **CM3P legacy MLP + external** | **0.57343 ± 0.00433** | — | **0.5881** |

**读法**：D2V 在 P2 比 canon+线性好 0.0377、P4 好 0.0375（P4 paired 5/5）；但它只赢
**canon + 线性**。换成最强基线（1k legacy 表 + external MLP），P2 0.56602 /
P4 0.57343，D2V+ext 的 0.5810 / 0.58561 都更差。D2V MLP 没有增益。P4 主口径还有
**transductive 成分**：编码器预训练见过 6,828 张中的 80%（5,462 张）；P4-holdout
干净口径（编码器未见过的 1,365 图）D2V 线性 **0.60471**、+ext **0.60280**，
canon 线性 0.65211、+ext 0.62848，但仍未与 legacy MLP+ext 对照。P2 MLP 只跑完 3/5
折；HPC 批量（job 14076958 / 14076959）截至 2026-11-23 20:25 仍 pending，
**不能引用**。

### 文档

| 文档 | 内容 |
|---|---|
| [docs/procedure.md](docs/procedure.md) | **管线与编排**：YAML、实验目录、step 版本、输入解析、扩展 |
| [docs/data.md](docs/data.md) | **数据**：step0 的保留规则与两张表、step1 pairbest / clean、规模与消融 |
| [docs/models.md](docs/models.md) | **模型**：oracle 公式与结果、CM3P 线性/MLP、MinaCalc 入口 |
| [docs/metrics.md](docs/metrics.md) | **指标**：难度序的成对 NLL、cross-fitted ceiling、captured、逐玩家 Spearman、训练目标 |
| [docs/experiments.md](docs/experiments.md) | **总表**：全部实验的 P2/P4 sign-NLL / Pearson / Spearman、随机合并集、10K、ISOR/Koishi 与 VSRG benchmark（含高难子集与缺失清单） |
| [docs/rank_eval.md](docs/rank_eval.md) | **实验**：MinaCalc / osumania_map_analyser（含 LN）的难度序评测 |
| [docs/cm3p_rank.md](docs/cm3p_rank.md) | **实验**：CM3P → per-chart 难度序；M0–M2、P1/P2、new-player val、content ceiling |
| [docs/cm3p_crop_windows.md](docs/cm3p_crop_windows.md) | **管线**：canon 裁剪 T0–T2、token 审计、窗口导出；T3 池化未跑 |
| [docs/cm3p_official_probe.md](docs/cm3p_official_probe.md) | **数据**：官方 CM3P 向量复现（cos 0.99745）与 audio-on 覆盖 |
| [docs/p4_multiseed.md](docs/p4_multiseed.md) | **实验**：P2/P4 5 个划分种子；7 方法的 NLL / Pearson / Spearman mean±std、配对 Δ、局限 |
| [docs/p4_multiseed_audio.md](docs/p4_multiseed_audio.md) | **实验**：完整 audio-on 内容表（published + local audio）在 P2/P4 上的配对复测；4 张音频缺口 |
| [docs/p4_multiseed_canon.md](docs/p4_multiseed_canon.md) | **实验**：canon vs legacy 同协议配对；39/40 个 seed×方法组合变差 |
| [docs/p4_multiseed_pub.md](docs/p4_multiseed_pub.md) | **实验**：published 子集三臂复测；published 对 local audio-off 40/40 更好 |
| [docs/random_merge_report.md](docs/random_merge_report.md) | **数据**：4 个 random 快照合并、453,601 对重建、0 个复活对、2.8% 重复 play |
| [randommerge_p2p4_results.md](docs/randommerge_p2p4_results.md) · [pool](docs/randommerge_pool_results.md) · [log_n/OD](docs/randommerge_od_legacy_results.md) | **实验**：20,388 图 P2/P4；external 线性 0.5833/0.5827、MLP 负结果、A0–A3 池化、log_n/OD |
| [docs/benchmark_vsrg.md](docs/benchmark_vsrg.md) | **外部评测**：4 个 CM3P 组合的 5 折 chart-group LOCV、isotonic 校准与负迁移 |
| [docs/benchmark_linear_extra.md](docs/benchmark_linear_extra.md) | **外部评测**：randommerge linear_extra 的 OOF MAE 0.4216、难度分层与重叠检查 |
| [docs/koishi_isor_report.md](docs/koishi_isor_report.md) | **实验**：Koishi / ISOR 在 10K 与 randommerge 上的 rank-NLL；覆盖率与边界 |
| [docs/low_difficulty_data.md](docs/low_difficulty_data.md) | **数据**：低难度为何被删、random 快照实测、为什么不用 API 批量爬 |
| [data2vec_main/README.md](data2vec_main/README.md) | **实验**：route B from-scratch audio-style data2vec；预训练、抽取、P2/P4 probe 与 caveat |
| [results/README.md](results/README.md) | **结果索引**：每个产物是什么、由哪条命令生成、在哪篇文档里解读 |
| [scripts/README.md](scripts/README.md) | **脚本索引**：step、rank 训练/评测、分析与运维脚本 |
| [preprocess/README.md](preprocess/README.md) | step0/step1 精简参考与可运行代码 |
| [minacalc/README.md](minacalc/README.md) | MinaCalc 包与 C++ 导出 |
| [legacy/README.md](legacy/README.md) | 归档内容索引；逐文件去向见 [legacy/MANIFEST.tsv](legacy/MANIFEST.tsv) |

计划、评审、调研与一次性运维快照（data2vec 2×2、MLM 反思、封装、HPC 队列、各实验执行计划）已收进
[legacy/docs/](legacy/docs/)；`docs/` 只保留当前口径、结果与总表。

### 说明

* **数据不入库**：原始 dump（13 GB）、`data/processed`（3.9 GB）与 `.osu` 缓存
  （1.2 GB）合计约 18 GB，只保留 `data/` 骨架。
* **实验目录不入库**：`experiments/` 整体被忽略；`data2vec_main/runs`、
  `data2vec_main/splits` 也被它自己的 `.gitignore` 忽略。本机可用
  `python scripts/index_experiments.py` 生成 `experiments/INDEX.md`。
  参数与结论请落到 `configs/` 与 `docs/`。
* **结果入库规则**：JSON / CSV / Markdown / PNG 入库，`*.npz`、`*.parquet`、
  `*.npy` 不入库（可由命令重建）；见 [results/README.md](results/README.md)。
* **`external/` 不入库**：`minacalc-standalone`、`cm3p`、`bobert` 都是独立克隆。
* **口径边界**：1k、10k、randommerge 与 benchmark 的绝对 NLL / RMSE / MAE 不可互比；
  `captured` 依赖各自的 ceiling（randommerge 的 ceiling 欠拟合，captured >1 不引用）；
  benchmark 的校准使用了 `expected`，这是外部评测最大的解释限制。
* Python 3.13；依赖见 [requirements.txt](requirements.txt)，CM3P GPU 侧见
  [requirements-cm3p.txt](requirements-cm3p.txt)。

### 许可

MIT

---

## English

DCR predicts 4K mania difficulty from **chart content only**, validated on player-behaviour
data cleaned so aggressively that every singleton (player, beatmap) pair is removed.
Methods include MinaCalc and derivatives, CM3P content vectors, a from-scratch
audio-style data2vec encoder, and per-chart difficulty learned directly against pairwise
rank-NLL; oracle baselines (bias, mf_dot, mirt) are kept for comparison.

**Data.** Three in-domain sets — 1k/10k pairbest (6,828 / 16,061 charts) and a
randommerge set built from four `performance_mania_random_10000` snapshots
(453,406 pairs / 20,388 charts / 12,758 players) — plus the external VSRG Dan Benchmark
(1,382 charts, 1,146 with beatmap ids). Absolute RMSE / NLL / MAE are not comparable
across these sets or across ceilings.

**Two axes and one external check.** *Loss regression* predicts `log(1-ACC)` per
(player, beatmap) row and reports RMSE / Spearman. *Rank-NLL* asks whether a difficulty
ordering predicts which of two charts the same player performs better on; it is
normalized by a cross-fitted Bradley-Terry ceiling (`captured`). The benchmark's Dan MAE
is an external-label check and is calibrated with the benchmark's `expected` values.
See [docs/metrics.md](docs/metrics.md).

**Headline results (2026-11-23).**

* **1k P2/P4 (5 split seeds).** CM3P + external MLP is best at **0.5660 / 0.5734**
  nats; the full audio-on table lowers this to **0.5585 / 0.5673**. CM3P-only MLP does
  not beat linear (0.6166 / 0.6254).
* **randommerge (20,388 charts).** External linear is best at **0.58326 / 0.58271**;
  MLP is worse by **+0.0050 / +0.0091** (5/5 seeds); `log_n` gives a small 5/5 gain
  (0.58247 / 0.58202). The 1k MLP advantage does not transfer.
* **VSRG Dan Benchmark.** Isotonic Sunny is **0.3367** MAE; the best CM3P-family model
  (RM-L, randommerge linear_extra) is **0.4216** and still loses by +0.0849. Published
  Koishi/XDC1/Azusa/ISOR/Roxy/Mixed are 0.154–0.288; RM-L is on par with Daniel (0.4224).
* **Koishi / ISOR on DCR rank.** Their benchmark Dan-MAE advantage does **not** transfer
  to the DCR ACC ordering: ISOR is −0.0015 on 10K ALL, +0.0045 on LN, and +0.0153 on
  randommerge; Koishi only covers 1,056 charts (6.57%).
* **data2vec (route B).** P2 0.5863 / P4 0.5930; it beats canon+linear but **not**
  legacy CM3P + external MLP (0.5660 / 0.5734). The main P4 protocol has a transductive
  component; the clean P4-holdout is 0.60471 / 0.60280.
* **canon cropping.** −43.7% tokens and truncation 0.42%→0, but rank-NLL is worse in
  39/40 seed×method cells; legacy remains the default content table.

**Quick start.**

~~~bash
pip install -r requirements.txt
# download the 1k/10k dumps into data/raw/ (commands above)

python main.py -c configs/exp_pairbest_1k.yaml     # default protocol A
python main.py -c configs/exp_cm3p_1k.yaml         # CM3P content vectors
python main.py -c configs/exp_cm3p_mlp_1k.yaml     # content-anchored MLP

python scripts/test_rank_metrics.py                # rank-NLL self-tests
python scripts/run_cm3p_rank_experiments.py --quick
cd data2vec_main && python scripts/smoke_test.py && cd ..
~~~

Each run creates `experiments/<timestamp>[_{name}]/` with `config.yaml`, `logs/`,
`artifacts/`, `results/`, `manifest.json` and a replayable `run.sh`.
See [docs/procedure.md](docs/procedure.md).

**Results and docs.** The Chinese section above is the authoritative table set. The main
result indexes are [results/README.md](results/README.md) and
[scripts/README.md](scripts/README.md). Deep dives:
[docs/p4_multiseed.md](docs/p4_multiseed.md) (1k P2/P4),
[docs/randommerge_p2p4_results.md](docs/randommerge_p2p4_results.md) (randommerge),
[docs/benchmark_vsrg.md](docs/benchmark_vsrg.md) and
[docs/benchmark_linear_extra.md](docs/benchmark_linear_extra.md) (external benchmark),
[docs/koishi_isor_report.md](docs/koishi_isor_report.md) (Koishi/ISOR), and
[data2vec_main/README.md](data2vec_main/README.md) (route B).

Retired scripts, archived docs, historical logs and the rank-NLL development scratch
live in [legacy/](legacy/README.md). No data is tracked; `experiments/` and
`external/` are ignored. License: MIT.
