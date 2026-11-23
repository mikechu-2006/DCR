# 实验：用 CM3P 内容向量优化 rank-NLL

本文件记录用冻结 CM3P 512 维内容向量学习 **per-chart 难度** `d_m`，并直接以
[docs/metrics.md](metrics.md) 的 rank-NLL（成对 Bradley–Terry NLL）为训练目标和评测口径的实验。
方法论部分先写清楚"学什么、怎么算、怎么比"，结果表在 §5。

实现与产物：

* 训练：`scripts/step2_cm3p_rank.py`
* 实验驱动：`scripts/run_cm3p_rank_experiments.py`
* 配对 ΔNLL：`scripts/paired_delta_nll.py`
* 新玩家 val 表：`scripts/player_disjoint_val.py`
* 官方评测：`scripts/step2_rank_eval.py`（2026-10-31 修复 ceiling 拟合/评测拆分后）
* 指标实现：`dcr/rank_metrics.py`
* 原始 JSON/parquet：`results/cm3p_rank_1k/`

> 时间口径：2026-10-31，1k pairbest，单机 CPU（6 线程，无 CUDA）。所有 test 行只用于评测，
> 不参与任何拟合；超参数在 train 内部按 player cluster 的 20% 验证集选择。
> 2026-11-03 用同一 runner 重跑 P1/P2 官方评测以补 `pearson_bhat`（旧 JSON 无此字段），
> 其余字段逐位复现。

---

## 0. 摘要

* 主口径是 **player-disjoint 验证**（train 内按 cluster 抽 20% 玩家；val ceiling 0.5891，
  coverage 0.883）：CM3P+external MLP val NLL **0.5476**，CM3P-only MLP **0.5680**，
  CM3P-only linear **0.6021**，最好的固定外部信号 Sunny **0.6367**。内容模型在新玩家上明显
  更好；MSD+Sunny 的线性 blend 只有 0.6372，说明外部信号单独不够，CM3P 是主信号。
* **P1（新 pair、同 player/chart）**的 test NLL 更低（MLP+external 0.5317），但 P1 的 signal
  用评测玩家的其他 pair 拟合，ceiling 却按玩家 cross-fit；因此 P1 的 captured 会 >1
  （MLP+external 1.238），**不能当 captured 用**。P1 只回答"同一批玩家、新 pair"。
* **P2（5 折 chart-disjoint）**是"新 chart"口径：CM3P-only linear captured **0.6006**
  （NLL 0.6148），CM3P-only MLP **0.5976**（0.6152，**没有增益**），而
  **CM3P+external linear 0.9013**（NLL 0.5756）。CM3P-only MLP 的 P1 增益主要在
  seen chart 上，不能泛化到新图；外部信号是能跨图泛化的内容。同一口径的 per-chart
  Pearson(d,b)：external linear **0.6025**、CM3P MLP 0.5613、CM3P linear 0.5546、
  Sunny 0.3713（§5.3）。**5 个划分种子的复核（§5.6）**补上了当时没跑的
  CM3P+external MLP：P2 0.5660 ± 0.0031，比 external linear 0.5779 ± 0.0033 好
  0.0118 nats（5/5 种子）；所以"非线性对新 chart 无效"只适用于 CM3P-only。
* **in-sample content ceiling**：CM3P-only linear 0.5879（captured_content 0.808），
  CM3P+external linear 0.5618（>1 的 oracle，见 §5.5）；linear 模型离各自 oracle 只差
  0.002–0.004 nats，说明线性族的拟合已接近极限，剩余增益来自非线性（MLP）和外部信号。
* 外部信号（MSD+Sunny）与 CM3P 互补：val 从 MLP-only 0.5680 → **0.5476**；
  CM3P+external linear 的 val 0.5767 也优于 CM3P-only linear 的 0.6021（λ1e-3）。
* **ceiling 是 noisy 的参照，不是硬上界**：val/P1 的 captured 都会 >1；比较系统时优先看
  val NLL、P2 和 paired ΔNLL，不要只看 captured 的绝对值。
* **发布表子集**（4,042 图）：published linear strict **0.588** / all-pair 0.593，
  captured 1.305*（subset ceiling 0.6126，coverage 0.926）；同子集 local linear 0.6122，
  音频版好约 0.019 nats；local MLP+external 在同子集 0.5714（all-pair）仍最好。
  captured 不能和主表混。

---

## 1. 问题与目标

rank-NLL 回答的问题是：**系统给出的 per-chart 难度序，能不能预测同一个玩家对两张图
哪一张表现更好**。它把玩家能力消掉，只看同一玩家内的两个谱面比较：

~~~text
y_pij = 1[ s_pi > s_pj ]          s = -loss = -log(1-ACC)
q_ij  = sigmoid( g(d_j) - g(d_i) )
L(d)  = - sum_{i<j} [ n_ij log q_ij + n_ji log(1-q_ij) ]
~~~

其中 `n_ij` 是所有玩家中 `i` 胜 `j` 的次数，`d_m` 是系统难度（越高越难）。
**这个指标没有玩家项**：它把所有玩家对同一对谱面的比较聚合成 counts，边际化掉玩家。
因此本实验要学的是

~~~text
d_m = f_theta(C_m)        C_m = 冻结的 CM3P 512 维内容向量
~~~

而不是 step2 的 per-(玩家,谱面) 预测。完整 CM3P MLP 的玩家×谱面交互在这个似然里
被边际化掉，不能直接改善它；如果目标是交互，需要另定义一个 per-player conditional NLL。

**要避免的循环**：如果把每张图的 `b_m` 也作为自由参数，那就是 ceiling 本身；
本实验里 `f_theta` 只能看内容，不能看 chart id，也不能看该图的任何成绩。

---

## 2. 数据

### 2.1 step1 / pairbest（1k）

| 项 | 值 |
|---|---|
| 输入 | `data/processed/step1_1k_pairbest.parquet` |
| 观测单位 | 一行一个 (玩家段, 谱面) 对 |
| 行 / 玩家段 / cluster / 谱面 | 109,706 / 1,039 / 942 / 6,828 |
| train / test 行 | 76,567 / 33,139 |
| train 唯一谱面对 / 比较数 | 2,433,345 / 6,065,303 |
| test 唯一谱面对 / 比较数 | 766,369 / 1,144,617 |
| 划分 | pair-random 70/30（同一玩家、同一谱面可以同时出现在 train/test，只是 pair 不同） |

> **P1 的重要限制**：pair-random 划分让同一谱面的其他 train pair 参与拟合，所以 P1 是
> "新 pair、同 chart"，不是"新 chart"。要声称泛化到新图，必须看 §3.4 的 P2。

### 2.2 CM3P 内容表

| 表 | 谱面数 | 覆盖 step1 谱面 | 说明 | 用途 |
|---|---|---|---|---|
| `chart_content_cm3p_local.parquet` | 21,949 | **6,828 / 6,828（100%）** | 无音频 CM3P 编码（audio off），512 维单位向量 | 主实验 |
| `chart_content_cm3p.parquet` | 11,877 | 4,042 / 6,828（59.2%） | 官方发布表 | 仅作子集稳健性检查 |

`data/processed/cm3p_mix_report.json` 的结论是 **DO NOT MIX**（同图余弦仅约 0.21），
所以两张表**不能拼接**。本地表覆盖全部 step1 谱面，是主实验；发布表只在 §5.3 的独立子集里报，
它有自己的 map 集、自己的 ceiling、自己的 baseline，不能和主表的 captured 直接比。

> **内容表示的限制（2026-11）**：上游 CM3P 从未发出 `[MANIA_COLUMN_*]` token
> （`parsing_cm3p.py:501` 写 `Group.column`，`tokenization_cm3p.py:191` 读
> `Group.mania_column`，而 dataclass 字段是后者且从未被赋值）。因此本文所有 `C` 与 `d`
> 都是 **column-blind** 的：列号（jack/trill/chord 的左右手结构）不在输入里。这是上游表示
> 层面的已知局限，不是本实验的 bug。离线审计、opt-in patch 的证据与代价见
> [docs/cm3p_crop_windows.md](cm3p_crop_windows.md) §0。

### 2.3 外部难度信号（M2 用）

`data/processed/rank_signals_1k.parquet` 里的 per-chart 列，全部 content-only、无成绩：

* `msd_overall`（本地 MinaCalc 505）
* `sunny_star` / `daniel_star`（osummania_map_analyser）
* `star`（osu! star）
* `notes_per_sec`

M2 用 `msd_overall` + `sunny_star`（主表最强的一维信号 + 一个不同来源的 pattern 难度），
并单独拟合 `ext_blend`（只用这两个外部信号）作为对照。所有外部信号在 train charts 上
z-score；不参与 test 拟合。

这两列在本批 21,949 张图上高度相关（Pearson **r ≈ 0.944**），所以 "MSD+Sunny" 实际上
接近一个 pattern 难度方向；external blend 的 val 0.6372 与 Sunny 单独 0.6367 几乎相同。
**"external" 指把这两个 per-chart 信号作为输入特征，不是外部验证集/外部数据**。

---

## 3. 方法

### 3.1 目标函数：精确的 marginal BT NLL

对每张谱面 `m`，`d_m = f_theta(C_m)`，`z = (d - mean(d)) / sd(d)`。训练集上的
所有 within-player pair 先聚合成 counts `n_ab` / `n_ba`（`a` 胜 `b` 的次数），
训练目标是

~~~text
q_ab = sigmoid( tau * (z_b - z_a) )
L(theta, tau) = - sum_{a<b} [ n_ab log q_ab + n_ba log(1 - q_ab) ] / N
N = sum_{a<b} (n_ab + n_ba)
~~~

要点：

* **counts 是充分统计量**。不要先把 counts 变成经验概率 `n_ab/(n_ab+n_ba)` 再算交叉熵；
  58% 的谱面对只出现一次，plug-in 目标会带很大噪声。训练直接用 counts（`build_pair_table`）。
* **ties 0.5/0.5**，与指标一致。
* **没有 per-chart 自由参数**。`d_m` 只来自 `C_m`；`b_m` 自由就是 ceiling。
* **全局仿射不变**。评测端会重新 z-score 并重估 `beta`，所以输出的全局 scale/offset 无关；
  但**间距（shape）有所谓**，linear link 不是纯序数指标。
* 训练时固定 `tau = 1` 并在每次前向里 z-score `d`；评测端用一维 MLE 重估 `beta`。
  两者只差一个全局 scale，排序/形状一致。

### 3.2 模型（M0–M2）

| 级别 | 形式 | 参数量 | 回答的问题 |
|---|---|---|---|
| **M0** | `d = w'C` | 512 + 1 | CM3P 有没有线性 rank 信号；必做基线 |
| **M1** | `d = w'C + MLP(C)` | 512 + 内容塔 | 非线性内容交互是否额外有用 |
| **M2** | `d = w'C + v'S` 或 `d = w'[C,S] + MLP([C,S])` | 514 + 塔 | CM3P 是否超过/互补外部难度信号 |
| **M2 baseline** | `d = v'S`（只外部信号） | 2 + 1 | 外部信号能解释多少；CM3P 的增量 |
| **M4** | 在评测 pair 上拟合同一模型（in-sample） | — | 内容族的 ceiling；oracle，不是泛化结果 |

MLP 结构：`d = Lin(C) + Out(MLP(LayerNorm(C)))`；一层或两层 hidden（64/128）、GELU、
dropout 0/0.1；`Out` 零初始化，线性路径用线性探针初始化。这样 MLP 只是对线性模型的
残差修正，step 0 严格等于线性探针。

**"external" 的准确含义**：S = [msd_overall, sunny_star]（train charts 上 z-score）。

~~~text
external linear : d = w'C + v'S                 (512 + 2 参数)
external MLP    : d = w'[C,S] + MLP([C,S])      (514 维输入，38,727 参数)
ext blend       : d = v'S                       (只用 2 个外部信号)
~~~

它不是外部验证，也不是把玩家成绩/label 加进特征；两个信号都是 content-only 的谱面难度
估计（MinaCalc 从 .osu 算，Sunny 从 osumania_map_analyser 算，均不使用成绩）。

### 3.3 训练与验证

* 内容标准化：每一维在 **train charts** 上 z-score（content-only，不看成绩）。
* 线性模型：L-BFGS + 解析梯度，L2 网格 `1e-3, 1e-2, 1e-1` 由验证集 NLL 选；选后在
  全部 train 行上重拟合。
* MLP：full-batch Adam，lr `3e-3`，线性路径 wd `1e-4`、塔 wd `1e-4`；最多 100 epoch，
  patience 20；按验证集 NLL 早停，再用最佳 epoch 在全部 train 行上重拟合。
* 验证：按 `player_id_base` 抽 20% cluster（1k 上 203/942 个），其余 812 个玩家拟合；
  **test 行不参与拟合、不参与早停**。
* 种子集成：最佳 MLP 配置在全部 train 上以 3 个 seed 重拟合，对 z-scored `d` 取平均。
* CPU full-batch；`torch.set_num_threads(6)`；无 CUDA。

> 计算限制：线性模型的 L2 由单次 80/20 player-cluster 验证选择；MLP 也只用同一验证集做
> 早停/选型，没有做 5 折 CV（full-batch MLP 在 CPU 上 5 折成本过高）。test 从未参与选择。

### 3.4 评测协议

| 协议 | 训练 | 评测 | ceiling | 回答 |
|---|---|---|---|---|
| **P1** 新 pair、同 chart | train 行 | test 行 | b 用全部行的 fold-train 玩家拟合，只在 test 行评测 | 已有该图日志时，排序新 pair |
| **P2** 新 chart | 5 折 chart-disjoint：每折只用 **两张图都在训练 chart 折** 的 train 行 | test 行；每张图用"没看过该图"的折模型打分 | b 用全部行的 fold-train 玩家拟合，只在 test 行评测 | 能否泛化到没见过的图 |
| **published subset** | 发布表 4,042 图，P1 | 对应 test 行 | 同一"全行拟合 / 子集评测"口径 | 音频版 CM3P 在覆盖子集上是否不同 |
| **content ceiling** | 在 test pair 上 in-sample 拟合 | test pair | 同一 ceiling | 内容族还能不能降 NLL；oracle |
| **P4** 双重冷启动 | 6×6（train 玩家 × train 谱面） | 4×4（test 玩家 × test 谱面） | b 用全部行的 fold-train 玩家拟合 | 见 [p4_multiseed.md](p4_multiseed.md) |

P2 的组合方式：第 k 折的模型给所有图打分，但只把第 k 折（held-out）图的 d 写进最终
signal；每张图的 d 都来自没看过该图成绩的模型。跨折的 pair 由两个不同折模型各提供一个
d，但每个 d 对其图都是 out-of-sample。官方评测仍用全部 test 行。

**P1 的口径警告（重要）**：P1 的 test 行与 train 共享**玩家**和**谱面**，signal 的 d 用这些
玩家的其他 train 行拟合；ceiling 却按 `player_id_base` cross-fit，评测玩家完全不参与 `b`
的拟合。所以 P1 的 captured 不是"新玩家"口径，模型甚至可能略低于 ceiling（captured > 1）。
本次 MLP 就出现了 0.5619 < ceiling 0.5628。这不是泄漏了 test pair，而是 signal 和 ceiling 的
玩家口径不同。**新玩家**看 20% held-out cluster 的 val NLL；**新 chart**看 P2；不要把 P1
captured 和"超过 ceiling"当成同一件事。

---

## 4. 指标口径（与 2026-10-31 修复后的 runner 一致）

### 4.1 NLL 与 link

~~~text
q_ij = sigmoid( beta * (z_j - z_i) )
z    = (d - mean(d)) / sd(d)        # 在评测 signal 的 map 集上标准化
NLL  = - sum_{i<j} [ n_ij log q_ij + n_ji log(1-q_ij) ] / N
~~~

`beta >= 0` 用一维凸问题的导数二分求 MLE（`_fit_beta_1d`）。若方向反了，
`beta = 0`，NLL = log2；不会偷偷拟合负斜率。`z` 只在评测 map 集上标准化，
所以 `d` 的全局 scale/offset 不影响 NLL，但单调非线性变换会改变 NLL。

### 4.2 ceiling

自由 per-chart `b_m` 的 BT MLE 是任何全局序能达到的最优成对似然（ceiling）。按
`player_id_base` 做 K=2 折 cross-fit：

1. 用**全部行**（train+test）的 fold-train 玩家拟合 `b`；
2. 只在当前 `--split`（本实验是 test）的 fold-test 玩家行上算 NLL；
3. 只保留"两张图都在 fold-train 出现过"的 pair；
4. ceiling = 覆盖 pair 上的 NLL / 覆盖比较数。

这一步是 2026-10-31 修复的核心：旧 runner 在 `--split test` 时连 `b` 也只用 test 行
拟合，test 只有 33k 行、4,565 张图、1.14M 次比较，6,828 个自由参数严重欠拟合。
1k 实测旧口径 ceiling = **0.64018**（coverage 0.833），比 MSD 的 signal NLL 0.63849 还高，
captured 会算出 **1.032**；正确口径 ceiling = **0.56276**（coverage 0.9396），MSD 的
captured 是 **0.419**（全 pair）/ **0.440**（strict pair）。

### 4.3 captured / bits_saved

~~~text
bits_saved = 1 - NLL / log2
captured   = (log2 - NLL) / (log2 - ceiling_nll)
~~~

`captured` 是主指标；`bits_saved` 是相对"全 50/50"的熵降。ceiling 本身有噪声
（1k split-half 的 b 相关约 0.58），所以 captured 是**下界**；系统间排序仍有效，
绝对值要谨慎。报告必须同时给 `ceiling_nll` / `ceiling_coverage`，不能只报 raw NLL。

### 4.4 strict pairs：分子分母同集

默认 runner 的 captured 分子用该 split 的全部 pair，分母用 ceiling 覆盖的 pair；
1k test 上两者相差约 6%（1,144,617 vs 1,075,472）。本实验的所有正式数字加
`--strict-pairs`：signal NLL 只在 ceiling 覆盖的 pair 上算，`n_strict` 必须等于
`ceiling_n`。输出同时保留 `captured_all` / `nll_all` 作对照。

### 4.5 权重：micro / macro / lw

* `nll_per_instance` 是 **per-comparison**（micro）加权，约 `l_p^2` 权重，重度玩家主导；
* 逐玩家 Spearman 的 l-weighted 是 `l_p` 权重；macro 是玩家等权。
三者不是同一个总体。主指标是 micro NLL；同时报 `per_player.lw` / `macro` /
`fisher_l3`，并注意"micro 赢、macro 输"的可能。

### 4.6 置信区间与比较

* 按 `player_id_base` cluster bootstrap 400 次（同一玩家的多个 epoch 不拆开）；
* 跨信号比较用**同一 cluster、同一 pair、固定 ceiling** 的 paired ΔNLL 更合适；
  两个独立 CI 是否重叠不是配对检验；
* 不同 map 集/不同 ceiling/不同 coverage 的 captured **不可比**。多信号用
  `--common-maps` 取交集，让它们共享 pair 集与 ceiling。

### 4.7 这张表的"不可比"清单

1. 1k pairbest 与 legacy clean 的 RMSE 不可比；
2. CM3P 主表（6,828 图）与发布表子集（4,042 图）的 captured 不可比；
3. P1 与 P2 的 captured 不可比（P2 是真正的新图口径）；
4. strict 与 all-pair captured 不可比（差约 1-2 个点）；
5. 任何没有同时报 `ceiling_nll` / `ceiling_coverage` 的 captured 都不可信；
6. P1 captured 与 player-disjoint val / P2 不可比：P1 的 signal 见过评测玩家的其他 pair。

---

## 5. 结果

### 5.1 P1（新 pair、同 player/chart）—— 描述性

> P1 的 signal 用评测玩家的其他 train pair 拟合，ceiling 却按玩家 cross-fit；trained model 的
> captured >1 是 player-in-sample 的必然结果，不是"超过 ceiling"。这一列只描述"同批玩家、
> 新 pair"；公平比较看 §5.2 的 val 和 §5.3 的 P2。

| signal | test NLL（strict） | captured* | lw | val NLL |
|---|---|---|---|---|
| CM3P MLP+external（M2） | **0.53173** | 1.2380* | 0.6109 | **0.54760** |
| CM3P MLP ensemble（M1） | 0.54908 | 1.1049* | 0.5809 | —（单 seed 0.56797） |
| CM3P MLP h64 L2（M1） | 0.55248 | 1.0788* | 0.5721 | 0.56797 |
| CM3P MLP h128 L1 | 0.55644 | 1.0484* | 0.5692 | 0.57328 |
| CM3P MLP h64 L1 | 0.55710 | 1.0434* | 0.5671 | 0.57486 |
| CM3P+external linear λ1e-3（M2） | 0.56079 | 1.0151* | 0.5764 | 0.57669 |
| CM3P+external linear λ1e-2 | 0.57113 | 0.9358 | 0.5636 | 0.58696 |
| CM3P linear λ1e-3（M0） | 0.58989 | 0.7920 | 0.5028 | 0.60205 |
| external blend（M2 baseline） | 0.62704 | 0.5070 | 0.5071 | 0.63720 |
| Sunny star | 0.62718 | 0.5059 | 0.5098 | 0.63666 |
| Daniel star | 0.62595 | 0.5154 | 0.5041 | 0.63718 |
| MSD | 0.63577 | 0.4401 | 0.4484 | 0.64543 |
| star | 0.66084 | 0.2477 | 0.3244 | 0.66286 |
| notes_per_sec | 0.65813 | 0.2685 | 0.3065 | 0.66373 |

`*` captured >1：P1 signal 见过评测玩家的其他 pair，ceiling 的 `b` 不含这些玩家；
**不要把它和 fixed 信号的 captured / ceiling 直接比**。test NLL 本身是真实的（test pair 未参与
拟合），但 captured 的分母不对等。MLP+external 的 test 与 val 来自同配置的两次重拟合
（no-val 全量 vs 80% 早停 + 全量重拟合），差异在 seed/早停噪声内。

### 5.2 Player-disjoint 验证（新玩家）—— 主比较

train 内按 `player_id_base` 抽 20% cluster（203/942）；其余 812 个玩家拟合。val 有
1,047,304 个 pair / 1,616,117 次比较。val ceiling 用 80% fit 玩家拟合 `b`、2 折 cross-fit，
只在 held-out 玩家的 pair 上算：

**val ceiling = 0.58909，coverage 0.883，n = 1,427,149。**

| signal | val NLL | val captured* |
|---|---|---|
| CM3P+external MLP（M2） | **0.54760** | 1.399* |
| CM3P MLP h64 L2（M1） | 0.56797 | 1.203* |
| CM3P MLP h128 L1 | 0.57328 | 1.152* |
| CM3P MLP h64 L1 | 0.57486 | 1.137* |
| CM3P+external linear λ1e-3 | 0.57669 | 1.119* |
| CM3P+external linear λ1e-2 | 0.58696 | 1.020* |
| CM3P linear λ1e-3 | 0.60205 | 0.876 |
| Sunny star | 0.63666 | 0.543 |
| Daniel star | 0.63718 | 0.538 |
| external blend | 0.63720 | 0.538 |
| MSD | 0.64543 | 0.459 |
| star | 0.66286 | 0.291 |
| notes_per_sec | 0.66373 | 0.283 |

`*` val captured 用 0.58909 的 val ceiling；>1 说明 val ceiling（80% 玩家拟合的 6,828 个
per-chart `b`，coverage 0.883）本身欠拟合，不能当硬上界。**这一节的排序看 val NLL**；
fixed 信号不训练、不看玩家，val NLL 与内容模型在同一个 held-out 玩家集上算，可以直接比。
这也是全文最干净的"新玩家"比较。

### 5.3 P2：5 折 chart-disjoint（新 chart）

每折用"两张图都在训练 chart 折"的 train 行拟合；第 k 折的图只用没看过该图的折模型打分。

`Pearson(d,b)` / `Spearman(d,b)` 在全部 4,565 张 test 谱面上、对同一个 ceiling `b`
（`--ceiling-fit all` 的 `b_full`）计算，与 JSON 的 `pearson_bhat` / `rho_bhat` 一致；
固定信号（Sunny / Daniel / MSD / external blend）不随 P1/P2 改变，值相同。

| signal | test NLL（strict） | captured | lw | Pearson(d,b) | Spearman(d,b) | 说明 |
|---|---|---|---|---|---|---|
| CM3P+external linear λ1e-3, CD | **0.57563** | **0.9013** | 0.5458 | **0.6025** | 0.5623 | 5 折；本表最好（未含 external MLP，见 §5.6） |
| CM3P linear λ1e-3, CD | 0.61484 | 0.6006 | 0.4380 | 0.5546 | 0.5192 | 5 折；P1 0.58989 → +0.025 |
| CM3P MLP h64 L2, CD | 0.61522 | 0.5976 | 0.4328 | 0.5613 | 0.5291 | 5 折，98 epoch；**没有超过 linear** |
| Sunny / Daniel / MSD | 0.62718 / 0.62595 / 0.63577 | 0.5059 / 0.5154 / 0.4401 | 0.5098 / 0.5041 / 0.4484 | 0.3713 / 0.3739 / 0.3575 | 0.3920 / 0.3925 / 0.3744 | fixed，同 test pair |
| external blend | 0.62704 | 0.5070 | 0.5071 | 0.3710 | 0.3913 | fixed 2-signal blend（值同 P1） |

**P2 的核心结论**：CM3P-only 的 MLP 在 P1（seen chart）上比 linear 好 0.035 nats，但在
chart-disjoint 后 **0.6152 vs 0.6148，没有增益**；它的非线性主要拟合了 seen chart 的
内容细节。能跨图泛化的是**外部内容信号**：CD extra linear 0.5756 / captured 0.9013，
比 CD linear 低 0.039 nats。

**Pearson(d,b) 怎么读**（per-chart 辅助指标，不是主指标）：

* 四个关心的模型按 Pearson 排：CM3P+external **0.6025** > CM3P MLP **0.5613** >
  CM3P linear **0.5546** > Sunny **0.3713**。与 NLL/captured 基本同序，唯一互换是
  MLP 与 linear：MLP 的 Pearson 高 0.007，NLL 却几乎相同（0.61522 vs 0.61484，
  paired Δ≈0），不足以改变"CM3P-only 的非线性对新 chart 没有增益"的结论。
* 训练模型的 Pearson **高于** Spearman（0.6025 > 0.5623、0.5546 > 0.5192），表里的
  fixed 信号相反（Sunny 0.3713 < 0.3920，Daniel/MSD 同向）。原因是模型在 linear link 下
  拟合 BT 似然，`d` 与 `b` 近似线性；Sunny/MSD 与 ACC-BT 尺度只是单调相关，scale 和
  尾部没对齐。所以 Pearson 混合了"序对不对"和"scale 像不像"，不能替代 NLL。
* `b` 本身有噪声（split-half 可信度 0.736，见 [rank_eval.md](rank_eval.md) §3），Pearson
  被衰减；除以 √0.736≈0.858 粗略校正后约为 0.702 / 0.654 / 0.647 / 0.433（同上顺序），
  只作量级参考。

（P2 的官方 captured 用与 P1 相同的 test ceiling 0.56276、strict pair；CD 模型仍看过 test
玩家的其他 chart 行，所以不是纯"新玩家+新 chart"；新玩家看 §5.2。）

### 5.4 Published CM3P 子集（4,042 / 6,828 图）

> 发布表与本地表是不同表示，**不能混**。子集有自己的 test pair 集和 ceiling，baseline 必须
> 在同一子集上重算；这里的 captured 不能和 §5.1/§5.2 直接比。

| signal | subset NLL | captured | 说明 |
|---|---|---|---|
| CM3P published linear λ1e-3 | **0.58799**（strict；all-pair 0.59308） | **1.305*** | 4,042 图；18,150 test 行；subset ceiling **0.61258**，coverage 0.926，n 307,610 |
| 同子集 local linear λ1e-3 | 0.61222（all-pair） | — | audio-off；同 4,042 图上比 published 差 **0.019** |
| 同子集 local MLP+external | 0.57142（all-pair） | — | audio-off+external；同子集最好 |
| Sunny / Daniel / MSD | 0.63589 / 0.63513 / 0.64412（strict） | 0.7107 / 0.7201 / 0.6086 | fixed，同子集 |

* published linear 的 captured >1 是 player-in-sample（同 §5.1）；subset ceiling 0.61258
比主表 0.56276 高很多（只有 4,042 图、coverage 0.926），captured 的绝对值不能和主表混。
**音频版 CM3P 在同一个 4,042 图上确实比 audio-off 本地表好 0.019 nats**（0.5931 vs 0.6122），
但本地 MLP+external 在同子集仍有 0.5714（all-pair）。

> 2026-11-09 在**同一子集**上按 P2/P4 协议补了 5 个划分种子的配对复测
> （[p4_multiseed_pub.md](p4_multiseed_pub.md)）：published 对 local audio-off 在
> 4 个 embedding 方法上 **40/40 个 seed×方法组合全部更好**（ΔNLL −0.011…−0.029），
> CM3P-only 的增益最大、叠 external 后收窄；canon 的代价在子集上同样复现。

### 5.5 Content ceiling、ablation、paired Δ

| family | in-sample test NLL（oracle） | captured_content |
|---|---|---|
| CM3P linear λ1e-3 | **0.58786** | 0.808 |
| CM3P+external linear λ1e-3 | **0.56175** | 1.008* |

`*` in-sample oracle 在 test pair 上拟合，**不是泛化结果**；extra linear 的 oracle 略低于
cross-fitted BT ceiling 0.56276，说明 in-sample/player 口径与 ceiling 的差异已在噪声量级。
CM3P linear 的 P1 strict NLL 0.58989，离自己的 oracle 0.58786 只差 **0.0020 nats**；
extra linear 的 P1 strict 0.56079，与 oracle 0.56175 在噪声内（模型甚至略低）。
MLP 的 val NLL 0.568（CM3P-only）和 0.548（+external）已经低于 linear 的 in-sample
oracle 0.5879——增益来自非线性与外部信号，不是线性拟合不够。

**Negative controls（关键）**

| control | test NLL | val NLL | 说明 |
|---|---|---|---|
| CM3P+external linear（真实） | 0.56557 | 0.57669 | 外部信号 + CM3P |
| 同模型，MSD/Sunny 在 chart 间打乱 | **0.59223** | **0.60215** | 外部增益消失，回到 CM3P-only（0.59243/0.60205） |
| CM3P linear（真实） | 0.59243 | 0.60205 | |
| 同模型，CM3P 向量在 chart 间打乱 | **0.68083** | **0.68228** | 内容增益消失，接近 log2 |

打乱方式：external 控制把 `msd_overall`/`sunny_star` 在 21,949 张图上独立随机重排
（两者相关从 0.944 → 0.004）；content 控制把 CM3P 的 512 维向量在谱面间重排。**test 行
在两个控制里都只用于评测**。如果模型还能从打乱的特征拿到 0.56/0.59，就说明有泄漏或 bug；
结果显示外部增益和内容增益都来自真实的 chart-level 信号，不是 test label。

**模型级验证（val NLL / best epoch / 参数量）**

| model | val NLL | test NLL（all-pair） | best epoch | params |
|---|---|---|---|---|
| CM3P linear λ1e-3 | 0.60205 | 0.59243 | — | 513 |
| CM3P MLP h64 L1 | 0.57486 | 0.56188 | 99 | 34,433 |
| CM3P MLP h64 L2 | 0.56797 | 0.55720 | 98 | 34,945 |
| CM3P MLP h128 L1 | 0.57328 | 0.56085 | 100 | 66,177 |
| CM3P MLP ensemble (3 seeds) | — | 0.55381 | 98 | 3 × 34,945 |
| CM3P+external linear λ1e-3 | 0.57669 | 0.56557 | — | 515 |
| CM3P+external MLP (val run) | **0.54760** | 0.53650 | 99 | 38,727 |

**Paired ΔNLL**（同一 cluster、同一 test pair；负值 = A 更好）：

| A : B | Δ NLL | 95% CI | P(Δ<0) |
|---|---|---|---|
| MLP+external : CM3P MLP h64 L2 | **-0.02031** | [-0.02480, -0.01541] | 1.000 |
| MLP+external : CM3P linear | **-0.05553** | [-0.06752, -0.04879] | 1.000 |
| MLP+external : external blend | **-0.09332** | [-0.10816, -0.07900] | 1.000 |
| CM3P linear : Sunny | **-0.03771** | [-0.04819, -0.02362] | 1.000 |
| CM3P MLP h64 L2 : CM3P linear | -0.03523 | [-0.04108, -0.02907] | 1.000 |
| MLP ensemble : MLP h64 L2 | -0.00340 | [-0.00454, -0.00168] | 1.000 |
| CM3P+external linear : CM3P linear | -0.02686 | [-0.03205, -0.02095] | 1.000 |

（paired Δ 用 §7 的 `paired_delta_nll.py`；在 P1 test pair 上算。由于 P1 是 player-in-sample，
这些 Δ 描述"同批玩家、新 pair"；新玩家看 §5.2 的 val。）

**P2 paired Δ（chart-disjoint 信号，同一 test pair，all-pair NLL）**

| A : B | Δ NLL | 95% CI | P(Δ<0) |
|---|---|---|---|
| CD extra linear : CD linear | **-0.03657** | [-0.04259, -0.03011] | 1.000 |
| CD MLP : CD linear | -0.00001 | [-0.00317, +0.00454] | 0.480 |
| CD extra linear : CD MLP | **-0.03656** | [-0.04288, -0.02933] | 1.000 |

CD MLP 与 CD linear 的 Δ 为 0（CI 跨 0，P=0.48）：**CM3P-only 的非线性在 chart-disjoint
后没有增益**；CD extra 的 -0.037 才是能跨图泛化的部分（external MLP 见 §5.6）。

### 5.6 P2/P4 多划分种子复核（external MLP）

§5.3 的 P2 表只有 external linear，没有 external MLP。后续在 **5 个新的 train/test
划分种子**（P2：pair-random 70/30 重采样；P4：玩家 × 谱面各分层 60/40，6×6 train /
4×4 test）上补齐了 7 个方法；模型种子固定 0。test NLL 的 5-seed mean ± std：

| 方法 | P2（新 chart） | P4（新玩家 + 新 chart） |
|---|---|---|
| **CM3P + external MLP** | **0.56602 ± 0.00307** | **0.57343 ± 0.00433** |
| CM3P + external 线性 | 0.57786 ± 0.00335 | 0.58479 ± 0.00494 |
| CM3P 线性 (λ1e-3) | 0.61618 ± 0.00293 | 0.62411 ± 0.00345 |
| CM3P MLP h64 L2 | 0.61664 ± 0.00265 | 0.62540 ± 0.00653 |
| Sunny | 0.62868 ± 0.00207 | 0.63111 ± 0.00777 |
| Daniel | 0.62768 ± 0.00212 | 0.63102 ± 0.00601 |
| star | 0.66153 ± 0.00137 | 0.66021 ± 0.00779 |

* **external MLP 在 P2/P4 都是最好**，对 external 线性的 paired ΔNLL 分别为
  **-0.01183 ± 0.00051** / **-0.01137 ± 0.00540**，5/5 个划分种子为负。
* **CM3P-only MLP 仍没有增益**（P2 0.6166 vs 0.6162；P4 0.6254 vs 0.6241，Δ 符号混合）。
  所以"非线性不能跨新图"只适用于纯 CM3P；**external 特征 + 非线性**可以泛化。
* P2 的 5-seed 均值与已发布单划分（seed 20260928）在 ~1σ 内：external 线性 0.5779 vs
  0.5756、CM3P 线性 0.6162 vs 0.6148、CM3P MLP 0.6166 vs 0.6152；external MLP 当时
  没有正式 P2 数字（P1 val 0.5476 是 player-in-sample 口径）。
* P4 只比 P2 差 ~0.007–0.009 nats（内容模型），方法排序不变；P2/P4 的 test 集与
  ceiling 不同，绝对 NLL 不可跨口径比较。

完整表、配对 Δ、Pearson/Spearman、局限见 [p4_multiseed.md](p4_multiseed.md)。

---

## 6. 讨论与局限

### 6.1 主要发现

1. **CM3P 内容本身在 rank-NLL 上远强于最好的固定外部信号**。player-disjoint val：CM3P
   linear 0.6021、CM3P MLP 0.5680，而 Sunny 0.6367、Daniel 0.6372、MSD 0.6454。P1 的
   strict captured：CM3P linear 0.792 vs Sunny 0.506（同 test pair、同 ceiling）。
2. **非线性（MLP）带来实质提升**：val 从 linear 0.6021 → MLP 0.5680（-0.034），
   3 seed ensemble 在 P1 上再 -0.0034（ΔNLL，CI 不含 0）。线性模型已接近自己的
   in-sample content oracle（0.5899 vs 0.5879），所以剩余增益只能来自非线性/额外特征。
3. **外部信号与 CM3P 互补，不是替代**。MSD+Sunny 单独线性 blend 的 val 只有 0.6372；
   把两者加到 CM3P 后，val 0.5680 → **0.5476**，P1 test 0.5572 → 0.5365。
   paired ΔNLL（P1）：MLP+external vs MLP = **-0.0203**（CI [-0.0248, -0.0154]），
   vs linear = -0.0555，vs external blend = -0.0933；全部 P(Δ<0)=1.000。
   说明 CM3P 是主信号，外部难度系统补的是 CM3P audio-off 没覆盖的 pattern/OD 信息。
4. **content ceiling**：CM3P-only linear 的 in-sample oracle 0.5879（captured_content
   0.808），P1 strict 0.5899，差 0.002 nats；extra linear 的 oracle 0.5618，模型 0.5608
   在噪声内。线性族已经到顶；MLP 的 val 0.568 / 0.548 低于 linear oracle，增益来自非线性。
5. **P2（新 chart）**：CM3P-only linear captured 0.6006（NLL 0.6148），MLP CD 0.5976
   （0.6152，**没有增益**），CM3P+external linear CD **0.9013**（0.5756）。per-chart
   Pearson(d,b) 同向：0.5546 / 0.5613 / **0.6025**（Sunny 0.3713），MLP 只在 Pearson 上
   微高于 linear、NLL 不变（§5.3）。所以 **CM3P-only** MLP 的 P1/P1-val 增益主要在
   seen chart 上；能跨图泛化的是外部内容信号。**5 个划分种子的复核（§5.6）**进一步
   区分了两件事：CM3P-only 的非线性对新 chart 无效（0.6166 vs 0.6162），但
   CM3P+external MLP 比 external linear 好 0.0118 nats（5/5 种子，P4 上同样 5/5）。
   原来的"非线性对新玩家有效、对新 chart 无效"应改为"**CM3P-only 的非线性**对新
   chart 无效；external 特征上的非线性可以跨新图"。
6. **P1 captured >1 不是模型超过 ceiling**：P1 的 signal 用评测玩家的其他 pair 拟合，
   ceiling 的 `b` 按玩家 cross-fit、完全不含这些玩家；这是 player-in-sample。
   val captured 也 >1，因为 val ceiling（80% 玩家、coverage 0.883）本身欠拟合。
   **ceiling 是模型，不是真值**；captured 的绝对值不能当硬上界，系统比较用 val NLL、
   P2 和 paired ΔNLL。
7. **发布表子集**：published linear 在 4,042 图上 strict 0.588 / all-pair 0.593
   （captured 1.305*，subset ceiling 0.6126、coverage 0.926）；同子集 local linear 0.6122，
   音频版好约 0.019 nats；local MLP+external 0.5714（all-pair）仍最好。两张表不能混；
   captured >1 是 player-in-sample，不能和主表比。

### 6.2 这些数字不能怎么读

* **P1 captured >1 不是超过 ceiling**：P1 的 signal 见过评测玩家的其他 pair 和该 chart 的
  其他 train pair；ceiling 按玩家 cross-fit。P1 只回答"同批玩家、同 chart、新 pair"。
* **val 的 0.5476 不是"新玩家+新 chart"**：val 只换玩家，chart 仍是同一批；val NLL 是
  早停选出的，且 val captured 的分母只覆盖 88.3% 的 val pair，而 val NLL 是全 val pair。
* **P2 的 0.9013 不是"全新"**：P2 换 chart，但模型仍看过 test 玩家的其他 chart 行。
  P4（[p4_multiseed.md](p4_multiseed.md)）补了玩家+谱面双重冷启动，但 P4 是**独立
  划分协议**（不是 P2 的同一 test 集再抽玩家），P2 本身仍不是新玩家；P3 的正式
  runner 仍未做。
* **external 是 stacking**：external linear/MLP = CM3P + MSD/Sunny；Sunny 单独 val 0.637、
  CM3P 单独 0.602/0.568；0.548 是两个强信号的组合，不是 CM3P 单独的功劳。CM3P 的净贡献
  看 M0/M1。
* **负控制**：external 打乱后 test 0.59223 / val 0.60215（≈ CM3P-only 0.59243/0.60205），
  外部增益消失；CM3P 向量打乱后 test 0.68083 / val 0.68228（≈ log2），内容增益消失。
  说明高结果来自 chart-level 的真实信号，不是 test label 泄漏；但不改变上面的口径限制。

### 6.3 指标口径的经验

* **captured 的绝对值依赖 ceiling 的 coverage/噪声**。本次 P1/val 都出现 captured >1；
  报告必须同时给 `ceiling_nll` / `ceiling_coverage` / `n_strict`，否则不可解释。
* **P1 与 val/P2/P4 不是同一个总体**。P1 = 新 pair、同 player/chart；val = 新 player、
  同 chart；P2 = 新 chart；P4 = 新 player + 新 chart。论文主表应放 val（新玩家）、
  P2（新图）和 P4（双重冷启动）；P1 只作描述。
* **strict pair 与 all-pair** 的 captured 差 1–2 个点（MSD 0.4401 vs 0.4192）；必须写清楚。
* **micro NLL 与逐玩家 Spearman 大体同向但不完全同序**：MLP+external 的 lw 0.611 最高，
  MSD 的 lw 0.448 低于 Sunny 0.510；NLL 更看高高置信/大间隔 pair。主指标是 NLL。
* **固定信号与训练模型不能直接比 P1 captured**：fixed 信号不训练、不看玩家，P1 captured
  对它是公平的；训练模型见过评测玩家的其他 pair，P1 captured 被抬高。公平比较用 val。

### 6.4 局限

* **验证协议**：单次 80/20 player-cluster 验证，没有 5 折 CV；MLP 的 val 是早停模型，
  最终 test 模型在全部 train 上重拟合。val 是 player-disjoint 的估计，但 test 仍是
  pair-random（P1），所以 test 数比 val 低。
* **P3 未做正式 runner**：真正的 player-disjoint **test**（留出 cluster 的 test 行）没有
  跑官方 captured；val 已给出 player-disjoint 的 NLL 排序，但和 P1/P2 的 ceiling 不可直接混。
  P4 用独立划分补了双重冷启动的 5 个种子（[p4_multiseed.md](p4_multiseed.md)），但它不是
  同一 test 集上的 P2∩P3，且模型种子固定 0。
* **P2 的口径**：CD 模型仍看过 test 玩家的其他 chart 行；它不是"新玩家+新 chart"。
  P4 做了玩家+谱面双重冷启动，但它是独立划分协议；P2 本身仍不是新玩家。
* **本地 CM3P 是 audio-off**：主表 100% 覆盖；发布表只有 4,042/6,828（59.2%）且有音频，
  两者是不同表示（mix report：DO NOT MIX）。不能跨 coverage 比较 captured。
* **选择偏差**：top-1000、ACC∈(0.90, 0.998]、n_plays≥2；难度序在两端最有区分度，
  本结论不外推。
* **ACC 难度 ≠ pattern 难度**：MSD/star 测 pattern，本指标是 ACC 的相对表现；
  两者的 chart-level 相关只有约 0.24–0.26；CM3P 学到的是"对这批玩家的 ACC 难度"。
* **外部信号是既有难度系统**：M2 是 stacking/互补，不是纯内容模型；它的提升不能全部
  记在 CM3P 头上。M0（CM3P-only）和 M1 才是 CM3P 本身的贡献。
* **计算限制**：CPU、6 线程、MLP 98 epoch（CD）/100 epoch（P1 val）；更大模型/更久
  训练可能继续降低 NLL，但 linear 已接近 content oracle，主要剩余空间在非线性和外部特征。

---

## 7. 复现

~~~bash
# 全流程 driver（约 30 min CPU；--quick 只做 3 epoch 的管线烟测）
python scripts/run_cm3p_rank_experiments.py --force
# 注：driver 在 P1 模型之后会继续按 λ=1e-2 跑 CD/published；本次最终表在 P1 后手动停止，
# 用下面的 best-λ targeted 命令补 P2/ceiling/published。最终数字以
# results/cm3p_rank_1k/*_final.json、player_disjoint_val.json、paired_delta*.json 为准。

# 单个模型（P1）
python scripts/step2_cm3p_rank.py \
    --model linear --name cm3p_linear \
    --lam-grid 1e-3,1e-2,1e-1 --val-frac 0.2 \
    --out-signals results/cm3p_rank_1k/cm3p_linear.parquet \
    --out-json    results/cm3p_rank_1k/cm3p_linear.json

# 新 chart（P2），线性 + 外部信号（本次 λ1e-3 胜出）
python scripts/step2_cm3p_rank.py \
    --model linear_extra --chart-only --chart-folds 5 --chart-seed 0 \
    --val-frac 0 --lam 1e-3 --name cm3p_linear_extra_cd_l1e3 \
    --out-signals results/cm3p_rank_1k/cm3p_linear_extra_cd_l1e3.parquet \
    --out-json    results/cm3p_rank_1k/cm3p_linear_extra_cd_l1e3.json

# 新玩家 val 表（20% held-out cluster；约 3 min）
python scripts/player_disjoint_val.py \
    --signal-table results/cm3p_rank_1k/signals_p1.parquet \
    --signals msd_overall,sunny_star,daniel_star,star,notes_per_sec \
    --val-frac 0.2 --val-seed 0 --folds 2 \
    --out results/cm3p_rank_1k/player_disjoint_val.json

# 配对 ΔNLL（同一 cluster、同一 pair 集；负值 = A 更好）
python scripts/paired_delta_nll.py \
    --signal-table results/cm3p_rank_1k/signals_1k.parquet \
    --signals cm3p_linear_extra,cm3p_linear,ext_blend,sunny_star \
    --pairs cm3p_linear_extra:cm3p_linear,cm3p_linear_extra:ext_blend \
    --split test --bootstrap 400 \
    --out results/cm3p_rank_1k/paired_delta.json

# 官方评测：同一 ceiling、同一 pair 集、strict
python scripts/step2_rank_eval.py \
    --in data/processed/step1_1k_pairbest.parquet \
    --signal-table results/cm3p_rank_1k/signals_p1.parquet \
    --signals cm3p_linear,cm3p_mlp_h64_L1,cm3p_mlp_h64_L2,cm3p_mlp_h128_L1,cm3p_mlp_ens,cm3p_mlp_extra,cm3p_linear_extra,cm3p_linear_extra_l3,ext_blend,msd_overall,sunny_star,daniel_star,star,notes_per_sec \
    --split test --ceiling-fit all --common-maps --strict-pairs \
    --folds 2 --bootstrap 400 --cache-dir /tmp/rank_cache \
    --out results/cm3p_rank_1k/rank_eval_p1_final.json

# 官方评测：P2（5 折 chart-disjoint；同一 ceiling、同一 pair 集、strict；含 pearson_bhat）
python scripts/step2_rank_eval.py \
    --in data/processed/step1_1k_pairbest.parquet \
    --signal-table results/cm3p_rank_1k/signals_p2.parquet \
    --signals cm3p_linear_cd_l1e3,cm3p_linear_extra_cd_l1e3,cm3p_mlp_cd,msd_overall,sunny_star,daniel_star \
    --split test --ceiling-fit all --common-maps --strict-pairs \
    --folds 2 --bootstrap 400 --cache-dir /tmp/rank_cache \
    --out results/cm3p_rank_1k/rank_eval_p2_final.json

# P2/P4 多种子（5 个划分种子；训练 + 官方评测 + mean/std）
python scripts/run_p2_p4_multiseed.py --protocols p2,p4 --seeds 1,2,3,4,5 --threads 6
~~~

单元测试：

~~~bash
python scripts/test_rank_metrics.py          # 24/24
python scripts/test_step2_rank_eval.py       # runner end-to-end，0 failures
python scripts/test_step2_cm3p_mlp.py        # MLP 自检
~~~

---

## 8. 产物

| 文件 | 内容 |
|---|---|
| `results/cm3p_rank_1k/<name>.parquet` | 单个模型的 `beatmap_id + name` signal |
| `results/cm3p_rank_1k/<name>.json` | train/val/test NLL、best epoch/lam、参数量、耗时 |
| `results/cm3p_rank_1k/signals_p1.parquet` | P1 所有模型列 + 外部 baseline，官方评测输入 |
| `results/cm3p_rank_1k/signals_p2.parquet` | P2/CD 信号 + baseline |
| `results/cm3p_rank_1k/signals_pub.parquet` | 发布表子集信号 + baseline |
| `results/cm3p_rank_1k/rank_eval_p1_final.json` | P1 官方评测（strict / all-pair，含 `pearson_bhat` / `rho_bhat`） |
| `results/cm3p_rank_1k/rank_eval_p2_final.json` | P2 5 折 chart-disjoint 官方评测（含 `pearson_bhat` / `rho_bhat`） |
| `results/cm3p_rank_1k/rank_eval_published.json` | 发布表子集评测 |
| `results/cm3p_rank_1k/player_disjoint_val.json` / `.csv` | 新玩家 20% val 的 NLL / ceiling |
| `results/cm3p_rank_1k/paired_delta.json` / `_p2.json` | P1/P2 关键对比的配对 ΔNLL 与 95% CI |
| `results/cm3p_rank_1k/cm3p_linear_ceiling_l1e3.json` | CM3P linear 的 in-sample content oracle |
| `results/cm3p_rank_1k/cm3p_linear_extra_ceiling_l1e3.json` | CM3P+external linear 的 in-sample oracle |
| `results/p4_multiseed/summary.csv` / `summary_mean_std.csv` / `summary.md` | P2/P4 5 个划分种子 × 7 方法的 NLL / Pearson / Spearman 与 mean/std（见 [p4_multiseed.md](p4_multiseed.md)） |

本文所有数字都来自上述 JSON（§5.3 的 P2 表来自 `rank_eval_p2_final.json`）。
`summary.csv` 是训练 driver 的扁平汇总；其中的 P2 行来自更早的 2 折 / λ=1e-2 运行，
与 §5.3 的 5 折最终模型（λ1e-3）不同，引用以 `rank_eval_p2_final.json` 为准。
