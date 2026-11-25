# 行为监督三条路线：方法论与执行计划

> 2026-11-24。状态：**计划**，不是结果报告。
> 本文把旗舰模型提出的三条后续路线转成可执行的研究方案：每条路线给出
> 假设、数据、模型、对照、指标、判据、代码改动、算力与停止条件。
> 现有 [三实验计划](2026-11-23_three_experiments_plan.md) 处理的是 first2 标签、
> 整 beatmapset 留出与 data2vec 残差修正；本文是**另一组**问题，不替换前者。
>
> 阅读顺序：§0 执行摘要 → §1 共同口径 → §2/§3/§4 三条线 → §5 执行计划与判据 →
> §6 代码与产物 → §7 不能说的话。赶时间只看 §0、§2.5、§3.5、§4.5、§5.1。

## 0. 执行摘要

### 0.1 三条线在问什么

三条线是同一件事的三个深度：**让谱面编码器从人类行为里学到“难度”以外的东西**。

| # | 问题 | 监督信号 | 最小对照 | 主指标 | 优先级 |
|---|---|---|---|---|---|
| 1 | 同一玩家的胜负能否改善谱面表征？ | 同一玩家两张图的 ACC 胜负（成对 BT） | 冻结 D2V / 解冻上层 / 同架构随机初始化 | H4 test rank-NLL | **最高** |
| 2 | 整局的六档判定分布能否弱监督出局部难点？ | 每局六档计数（标签比例） | rank-NLL only vs + 判定比例 | H4/P4-full test rank-NLL | 高（推荐新方向） |
| 3 | “哪些玩家在哪些图上相对更好”能否蒸馏成谱面表示？ | mf_dot 教师对固定参考玩家的预测向量 | 平均难度 vs 整组参考玩家反应 | H4 test rank-NLL | 中（前两条获胜后） |

三条线的共同点：

* 编码器**只读原始谱面**：4 列 onset/hold 序列（8 通道 piano roll），不用音频、
  不用 OD/star/难度常数、不用玩家历史作为部署输入；
* 评价只用**同一玩家内两张图的胜负**（rank-NLL），玩家能力在成对比较里被消掉；
* 所有对照在**同一批留出谱面、同一批 test 对、同一个 ceiling** 上比较，
  只报配对 ΔNLL，不跨口径比绝对值；
* Sunny 与 MinaCalc 全程作为**独立基准**，不是训练信号，也不是要“打败”的对手。

### 0.2 为什么现在做

四个已经具备的条件：

1. **数据够大，且谱面数远小于对数**：full 并集有 **1,898,076** 个
   (玩家段, 谱面) 对、**20,945** 张谱面、**20,681** 个基础玩家
   （[full_union_report.md](../full_union_report.md) §1/§7）。同一张图有许多人演奏过，
   这正是“同一玩家比较两张图”的监督密度来源。
2. **六档判定原来就有，只是被丢掉了**：legacy 行含
   `countgeki/count300/countkatu/count100/count50/countmiss`，
   modern JSON 含 `perfect/great/good/ok/meh/miss`；现有 step0 只用它们算
   `loss = log(1-ACC)` 后丢弃（[step0_preprocess.py:425](../../scripts/step0_preprocess.py#L425)，
   权重见同文件 L89–L93）。判定分布的“形状”信息目前完全没进模型。
3. **有一个可用的行为教师**：`mf_dot`（矩阵分解）已经在
   `step2/predict` 里，能给出玩家×谱面的交互项；它的潜向量可以任意旋转，
   但固定参考玩家上的预测有确定含义（[models.md](../models.md) §2）。
4. **有一个可微的谱面编码器**：D2V route B 的 12L/512d 编码器读 8 通道 piano roll，
   已在 20,584 张图上完成遮蔽预训练；512-d 内容表在本地，完整 checkpoint 在 HPC
   （见 §5.1 P0.4；[data2vec_main/README.md](../../data2vec_main/README.md) §1/§6.2；
   [benchmark_d2v.md](../benchmark_d2v.md) §2.1）。

### 0.3 一个必须先说清楚的统计事实

在 pairbest 的 ACC 区间 (0.9, 0.998] 内，六档判定的**平均值**大致是：

| 档位 | 占判定比例 | 对 ACC 的权重 | 在同一 ACC 内的组内标准差 |
|---|---:|---:|---:|
| MAX（320） | 66.6% | 300（与 300 同权） | 0.065 |
| 300 | 29.0% | 300 | — |
| 200 | 0.52% | 200 | 0.0057 |
| 100 | 0.44% | 100 | 0.0029 |
| 50 | 0.14% | 50 | 0.0019 |
| miss | 3.31% | 0 | 0.0042 |

（1k legacy 前 60 万行中 ACC ∈ (0.9, 0.998]、总判定 ≥100 的 408,033 局；
这是描述性样本，不是 pairbest 精确分布；Gate 0 会用 pairbest 行重算。）

两个后果：

* **MAX/300 的拆分对 ACC 完全不可见**：两者权重都是 300，所以同一 ACC 可以对应
  完全不同的 MAX 率（组内标准差 0.065，约等于总体标准差的 58%）。这是判定分布
  里最大的一块“ACC 看不见”的信息。
* **坏档位（200/100/50/miss）在固定 ACC 下仍有显著变化**，而且 miss 与
  200/100/50 在固定 ACC 内是此消彼长的（组内相关约 −0.71 到 −0.78）。
  “99 个 300 + 1 miss”和“97 个 300 + 3 个 200”只是这个 trade-off 的一个例子。

所以第 2 条线不是“把 ACC 再喂一遍”，而是**给模型一个比 ACC 更细的行为结果**：
同样 99% ACC，MAX 率、坏档位组合、miss 与 200 的比例可以完全不同。

### 0.4 建议的执行顺序（一句话版）

> **第 1 周先建行为数据与 H4 划分；第 2–3 周完成第 1 条的冻结/解冻/随机三臂；
> 第 3–5 周在胜出的编码器上加第 2 条；第 4–6 周用冻结探针并行启动第 3 条；
> 第 6–7 周做 P4-full / P4-1k / VSRG 外部复核并写结论。**

顺序的理由：第 1 条决定“行为监督是否值得继续投入”。如果冻结 D2V 已经打平或
打赢解冻，说明行为监督没有改善表征，第 2、3 条应降级为“换标签/换读出”的问题，
而不是继续加编码器算力。如果解冻明显更好，第 2、3 条才有“在更好的表征上加更细的
监督”的意义。

### 0.5 什么算成功

不是“打赢 Sunny”。而是下面三条中至少一条成立，并且能被同一套留出图复现：

1. **表征层**：解冻上层比冻结 D2V 在 H4 test 上稳定低 ≥0.003 nats（5/5 种子，
   CI 排除 0），且 Pearson/Spearman 不变差；
2. **监督层**：在胜出编码器上，rank-NLL + 六档判定比例比 rank-NLL only 稳定低
   ≥0.003 nats（≥4/5 种子），而打乱判定比例的 placebo 没有同样增益；
3. **结构层**：蒸馏整组参考玩家的反应比只蒸馏平均难度稳定低 ≥0.003 nats
   （≥4/5 种子），而打乱签名的 placebo 没有同样增益。

即使三条都不成立，只要实验有足够功效并给出清楚的零结果，也是可交付的：
它告诉我们“继续扩大行为监督”是否值得，而不是把负结果藏起来。

### 0.6 E2/E3 执行更新（2026-11-24）

第 1 条已完成 clean seed 1：解冻 top-4 的 H4 test NLL 为 0.571056，比同头冻结
低 0.024273，两个 CI 都排除 0。因此第 2、3 条按本计划使用**解冻 top-4** 作为
backbone，而不是退回冻结臂。

第 2、3 条的 seed 1 已执行。E2 固定预算 5 臂完成：judge 相对 rank 仅
−0.001128，且 placebo 更低（−0.002577），H2 不成立；首轮 −0.010774 是早停
预算伪影。E3 完成：mean 臂比 rank 低 0.013217，但 full 签名比 mean 高
0.007098（两个 CI 排除 0），H3 不成立。详见
[E2 结果](../e2_judge_results.md)、[E3 结果](../e3_signature_results.md)。
实现中修掉了 hold 判定时间、placebo 行和、单档位数值、玩家基线聚合、长图
截断、事件级 QA、B1 均值目标、教师 QA 等问题；完整清单、命令、判据和 job id
见 [E2/E3 执行计划与问题记录](2026-11-24_e2_e3_plan.md)。

---

## 1. 共同口径

三条线共享同一套输入、划分、指标与基线。先把它钉死，后文只写各自不同的部分。

### 1.1 输入与编码器

**输入**：`.osu` 的 4K mania 对象（circle + hold），按
`data2vec_main/d2v_main/charts.py` 转成 100 Hz × 8 通道：

* 通道 0–3：四列的 onset，sigma=8 ms 高斯软化，clip 到 4；
* 通道 4–7：四列的 hold 状态（持续段为 1）。

**窗口**：16 s = 1600 个 100 Hz 帧；conv stride 2 后是 800 个 50 Hz token。
训练时每张图随机采 K 个 16 s 窗口（默认 K=4），验证/测试时用全部不重叠窗口
（与 `extract.py` 的图向量口径一致）。

**图向量**：

$$
z_m = \mathrm{Norm}\left(\frac{\sum_w v_w \cdot \frac{1}{v_w}\sum_{t=1}^{v_w} e_{m,w,t}}{\sum_w v_w}\right)
$$

其中 $e_{m,w,t}$ 是编码器在第 $w$ 个窗口第 $t$ 帧的输出，$v_w$ 是有效帧数，
$\mathrm{Norm}$ 是 L2 归一化（主口径；§2.4.5 有 L2 消融）。

**d 头**：LayerNorm(512) → Linear(512,256) → GELU → Dropout(0.1) → Linear(256,1)。
三臂共用同一个头；冻结臂也用它，不用旧的 `linear_extra` 头，避免“头不同”
成为解释。

**只读原谱**：不使用音频、OD/HP/AR、star、BPM、mapper、beatmap_id 的 one-hot。
`beatmap_id` 只作为数据索引，不作为特征。外部信号只在基线表里出现。

### 1.2 数据与划分

#### 1.2.1 full 并集的硬数字

| 项 | 值 | 来源 |
|---|---:|---|
| 合并后 play 行 | 17,293,289 | [full_union_report.md](../full_union_report.md) §1 |
| step1 对 | 1,898,076 | 同上 §7 |
| step1 谱面 | 20,945 | 同上 §1 |
| 基础玩家 / 玩家段 | 20,681 / 22,535 | 同上 §1 |
| 每图对数 p10/p50/p90 | 2 / 19 / 205（computed） | 本机统计 |
| ≥50 对的图 | 6,529 | 本机统计 |
| 有 D2V/CM3P 内容的图 | 20,832（99.46%） | [full_union_report.md](../full_union_report.md) §6 |
| 玩家内两两比较数 | 255,196,653 | 本机统计 |
| H4 seed 1 train 行 / 比较数 | 927,896 / 100,287,454 | 本机统计 |
| H4 seed 1 test 行 / 比较数 | 138,685 / 3,679,122 | 本机统计 |

#### 1.2.2 为什么需要一个新划分 H4

现有两个口径都不完全适合这三个问题：

* **P2**（新 chart，玩家仍在训练集）：玩家侧没有冷启动；对“新玩家”的结论不成立
  （[p4_multiseed.md](../p4_multiseed.md) §1.1）。
* **P4**（新玩家 + 新 chart，双重冷启动）：玩家和谱面都新，但 chart 的 60/40 是
  随机的；D2V 预训练用了 21,949 张 1k 白名单图里的 20,584 张，
  **P4 的 test chart 很可能被 D2V 预训练见过**（D2V README §6.5 的 caveat）。
  对第 1 条的“遮蔽预训练到底帮了多少”这是混淆：冻结/解冻臂见过 test chart 的
  无标签内容，随机臂没有。

**H4（Holdout-4）** 是 P4 的一个变体，只改 chart 侧：

| 项 | H4 定义 |
|---|---|
| test charts | 固定为 `data2vec_main/splits/holdout/eval_ids.txt` 的 1,365 张（D2V 预训练完全没见过；1,363 张在 full 里有对） |
| players | 按 `player_id_base` 聚类、活跃度 5 层分层，60/40 train/test；同一 base 的所有段只在一侧；5 个划分种子（1–5） |
| train | train players × 非 holdout charts |
| test | test players × holdout charts |
| dropped | 两个 off-diagonal block，不训练，只给 BT ceiling 拟合 $b$ |
| val | 从 train charts 里按 star/popularity 分层抽 10%，chart-disjoint，用于早停；不参与训练 |
| 标签 | 仍是 `loss = log(1-ACC)`，ACC ∈ (0.9, 0.998]，n_plays ≥ 2 |

H4 seed 1 的实际规模（本机用 `step1_full_pairbest.parquet` 复算）：

| block | 行 | 谱面 | 玩家段 |
|---|---:|---:|---:|
| train | 927,896 | 18,714 | 13,395 |
| test | 138,685 | 1,363 | 6,384 |
| dropped（test player × train chart） | 626,137 | — | — |
| dropped（train player × test chart） | 205,358 | — | — |

test 每图对数的 p10/p50/p90 = 10 / 48 / 233；≥20 对的 test 图 1,059 张。
5 个种子的 test 行在 136,608–139,177 之间，稳定。

> 注：上表的 train 行/比较数是**未扣 10% val charts、未扣 113 张无内容图**的毛数；
> 实际训练约少 10%（val）和 <1%（无内容），以 split JSON 为准。

**H4 的代价**：这 1,365 张图来自 1k pairbest 的 star 分层 20%，难度偏高：

| 图集 | star 均值 | star 中位数 |
|---|---:|---:|
| full 全部图 | 3.312 | 3.147 |
| full pairbest 图 | 3.391 | 3.231 |
| D2V holdout 图 | **4.711** | 4.714 |

所以 H4 是**高星偏置**的“全新图”测试。它适合回答第 1 条的预训练消融（干净），
但不足以单独支撑“低/中星也变好”的结论。补法是 §1.2.3 的 P4-full，以及可选的
H4-full（用新预训练划分把 holdout 扩到全难度分布，见 §5.1 Gate 1b）。

#### 1.2.3 三个评价协议

| 协议 | chart test | player test | test chart 被 D2V 预训练见过？ | 用途 | 优先级 |
|---|---|---|---|---:|---|
| **H4** | 固定 1,363 张 D2V-unseen | 40% | **否** | 第 1 条三臂；第 2/3 条的干净复核 | **主** |
| **P4-full** | 随机 40%（8,380 张，full 20,832 图内） | 40% | 多数是 | 第 2/3 条在全难度分布上的复核；第 1 条的稳健性（有 transductive caveat） | 副 |
| **P4-1k** | 随机 40%（2,730 张，1k 6,828 图内） | 40% | 多数是 | 与已有 P2/P4 数字对齐 | 副 |
| **P4-1k-holdout** | 1,365 张 D2V-unseen | 40% | 否 | 已有 D2V P4-holdout 的对照 | 参考 |

主表只报 H4；P4-full / P4-1k 是**同一方法在另一个 test 集上的符号复核**，
不混表、不比绝对 NLL。P4-full seed 1 的规模（本机用
`scripts/step1_p4_split.py` 在 full 上实跑）：

| block | 行 | 谱面 | 玩家（base） |
|---|---:|---:|---:|
| train (6×6) | 677,661 | 12,565 | 12,410 |
| test (4×4) | 307,178 | 8,380 | 8,271 |
| dropped | 913,237 | — | — |

P4-1k 的数字沿用 [p4_multiseed.md](../p4_multiseed.md) §2.2：
seed 1 train 39,825 / test 17,569；4,098 / 2,730 张图；ceiling 0.56540 ± 0.00859。

#### 1.2.4 防泄漏规则（所有实验硬约束）

1. test 行从不参与训练、早停、超参选择、教师拟合、参考玩家选择、归一化统计；
2. val 是 chart-disjoint 的 train 子集，只用于早停/选 λ；test 从不见；
3. 教师、参考玩家、辅助标签只用 train block；P4 的 off-diagonal 只给 ceiling，
   不给模型；
4. 外部信号（Sunny/MSD/star）只在基线表里，不进任何训练臂；
5. 每个 run 的 split JSON 必须记录 chart/player 的互斥校验；不过校验不落盘；
6. D2V 的内容表/checkpoint 版本写进结果 JSON；不同版本的绝对 NLL 不混比。

### 1.3 指标与统计

#### 1.3.1 主指标：rank-NLL

对玩家 $u$ 玩过的两张图 $i,j$，令 $s_{ui}=-\mathrm{loss}_{ui}$（越高越好），

$$
y_{uij} = \mathbf{1}[s_{ui} > s_{uj}], \qquad
q_{ij} = \sigma(d_j - d_i), \qquad
L(d) = -\frac{1}{N}\sum_{i<j}\left[n_{ij}\log q_{ij} + n_{ji}\log(1-q_{ij})\right]
$$

其中 $d$ 越高越难，$n_{ij}$ 是所有玩家中 $i$ 胜 $j$ 的次数；ties 按 0.5/0.5
（与 `build_pair_table` 的 `tie="half"` 一致）。$N$ 是参与比较的
pair 数。**这个指标没有玩家项**：同一玩家内的比较把玩家能力消掉，剩下的只有
谱面难度序。定义与实现见 [metrics.md](../metrics.md)、
[dcr/rank_metrics.py](../../dcr/rank_metrics.py)。

**ceiling**：把每张图的 $b_m$ 当自由参数，按玩家 cluster 2 折 cross-fit；
test NLL 只在 fold-test 玩家的 pair 上算。ceiling 是“任何全局序”的参照，不是硬上界
（[cm3p_rank.md](../cm3p_rank.md) §1；[p4_multiseed.md](../p4_multiseed.md) §4）。
主表必须同时给 `nll`、`ceiling_nll`、`captured`，否则覆盖子集不可比。

**strict-pairs**：signal NLL 只在 ceiling 覆盖的 pair 上算，与 `captured` 的分母
同一批 pair（`step2_rank_eval.py --strict-pairs`）。所有主表用 strict-pairs。

**链接**：评测时对 $d$ 重新拟合线性 link 的 $\beta$（`--link linear`，默认）。
所以 $d$ 的全局仿射变换不影响 NLL；但 $d$ 的**间距**会影响线性 link 的拟合，
这是模型输出尺度的一部分，不做额外标准化。

#### 1.3.2 配对 ΔNLL 与不确定性

对同一 test 集上的两个信号 A、B：

$$
\Delta\mathrm{NLL} = \mathrm{NLL}(A) - \mathrm{NLL}(B)
$$

负值 = A 更好。主表报：

* 5 个划分种子的 mean ± std（ddof=1）；
* 每个种子的 ΔNLL；
* “A 更好的种子数 / 5”；
* 95% CI：**player-cluster bootstrap**（沿用 `step2_rank_eval.py` 的
  `bootstrap_metric`）与 **chart-cluster bootstrap**（新增，按图重采样，
  pair 权重取两张图重采样次数的乘积；这是 two-way cluster 的近似）。
  两个 CI 都报，避免只挑一个。

**功效**：H4 test 有 138,685 行 / 3.68M 比较 / 1,363 图 / 6,118 玩家。
chart 级效应的有效样本约 1,363 张图；若每图 ΔNLL 的 std 约 0.05，
chart-cluster SE ≈ 0.0014，0.003 nats 的效应可检测。5 个划分种子的
between-seed std 可能更大（0.003–0.007），所以判据同时要求 5/5 符号一致，
不只依赖单一 CI。

#### 1.3.3 基线

每条线的主表都带下列基线；它们**不参与训练**，只在同一 test 对/ceiling 上算：

| 基线 | 来源 | 说明 |
|---|---|---|
| Sunny | `sunny_star`（rank_signals_1k.parquet） | 独立 Dan 估计器，覆盖 20,832 图 |
| MinaCalc | `msd_overall`（同上） | 独立难度，覆盖 20,832 图 |
| star | `star`（beatmap_meta_full） | 官方 star，弱基线 |
| CM3P + external MLP | `step2_cm3p_rank.py --model mlp_extra` | 强内容基线；H4 上重跑 |
| 冻结 D2V 线性/MLP | `step2_cm3p_rank.py --content-path chart_content_d2v_holdout.parquet` | 第 1 条冻结臂的旧头参照 |
| mf_dot | `step2/predict` | 第 3 条的教师；不作为 chart-only 基线 |

**不许做的事**：拿 H4 的绝对 NLL 和 1k P2/P4 的绝对 NLL 比；拿不同 ceiling 的
`captured` 比；拿不同内容表覆盖的图集比。所有比较只在同一 test 对、同一 ceiling、
同一 `n_scored` 上做。

### 1.4 算力与产物

* **GPU**：编码器训练在 HPC（A800/A40，bf16）；本地无 GPU（D2V README §5 问题 3）。
* **CPU**：冻结头、探针、教师、评测在本机或 HPC CPU 节点。
* **存储**：判定侧表约 1–2 GB；H4/P4 划分约 100 MB；每个 run 的 checkpoint
  0.7–1.5 GB；内容表 45 MB；预计总增量 <150 GB。
* **产物命名**：`results/behavior/<exp>/<arm>/s<seed>/` 下放
  `metrics.json / d_test.parquet / train_log.json / config.yaml`；
  汇总在 `results/behavior/summary/`。每个 run 必须能从
  `config.yaml + git rev + split JSON` 复现。

---

## 2. 第 1 条：用同一玩家的胜负直接训练谱面编码器

### 2.0 直觉：一个只比较同一玩家两张图的锦标赛

先把“胜负”说清楚：不是游戏胜负，而是同一个玩家在两张图上的 ACC 谁更高。
如果玩家 u 在 A 上 ACC 97%、在 B 上 94%，这次比较的“赢家”是 A。
模型要给每张图一个难度 $d$（越高越难），于是

$$
P(u \text{ 在 A 上更好}) = \sigma(d_B - d_A)
$$

因为 B 越难（$d_B$ 越大），玩家越可能在 A 上表现更好。两张图用同一个编码器，
各自算出一个 $d$；损失只比较这两个 $d$。

这像一个锦标赛：每张图是一个“选手”，每个玩家的两张图是一场“比赛”，
比赛结果是“哪张图对该玩家更容易”。很多玩家重复比较同一对图，
Bradley–Terry 模型就能估计出每张图的“实力”（难度）。区别是：
这里不是给每张图一个自由参数（那是 ceiling），而是让 $d$ 由原始音符算出来，
所以新图也能算。

**为什么玩家能力会消失**：同一个人的两张图比较，他整体强 5% 还是弱 5%，
对“A 比 B 高 3%”没有影响。所以模型不需要玩家 embedding；它只学
“哪些谱面特征让一张图对大多数人更难”。这也是它和第 3 条的区别：
第 1 条学的是**全局难度**，第 3 条才学“谁相对更喜欢/更擅长哪张图”。

一个数字例子：

* 玩家 u：A 的 ACC 0.97，B 的 ACC 0.94，所以 A 赢（$y=1$）。
* 模型输出 $d_A=1.2$、$d_B=1.5$：$q=\sigma(0.3)=0.574$，
  loss $=-\log 0.574=0.555$。
* 如果模型反过来输出 $d_A=2.0$、$d_B=1.0$：$q=\sigma(-1.0)=0.269$，
  loss $=1.31$。
* 梯度会把 $d_B$ 往上推、$d_A$ 往下推；编码器要在 B 的音符里找到
  “让它更难”的特征，在 A 里找到“让它更简单”的特征。

三臂在问什么：

| 臂 | 编码器 | 在问什么 | 如果它更好，说明什么 |
|---|---|---|---|
| 冻结 D2V | 固定 | 不改表征、只学读出，能做到多好？ | 预训练特征已经够用；行为监督只是读出问题 |
| 解冻 top-4 | 从同一 checkpoint 开始，改最后 4 层 | 让行为梯度改表征，是否有帮助？ | 行为监督改善了表征；第 2/3 条可以继续用这个编码器 |
| 随机初始化 | 不预训练，全部从零训练 | 没有遮蔽预训练，行为监督自己能学出表征吗？ | 预训练不是必需；也可能是随机臂欠训练（要看学习曲线） |

“解冻 vs 冻结”回答“行为监督能否改善表征”；“解冻 vs 随机”回答
“遮蔽预训练到底帮了多少”。两个问题一次实验拆开，是因为三臂的输入、
窗口、聚合、头、训练对、预算全都相同，只差编码器的初始化/冻结方式。

这也是为什么第 1 条要先做：如果冻结已经打平或打赢解冻，第 2/3 条就不该
继续加编码器算力，而应把精力放在标签/读出上。

### 2.1 假设与可执行定义

旗舰原话 → 可执行定义：

| 原话 | 本文定义 |
|---|---|
| “让同一玩家的胜负直接训练谱面编码器” | 对同一玩家段的两张图 $i,j$，用 $y=\mathbf{1}[\mathrm{ACC}_i>\mathrm{ACC}_j]$ 训练 $q_{ij}=\sigma(d_j-d_i)$；$d_m$ 是整张图的标量 |
| “输入沿用已有的四列 onset/hold 序列” | 8 通道 100 Hz piano roll，16 s 窗口，D2V conv+Transformer 编码器 |
| “两张谱共享同一个编码器，各自输出难度 d” | 同一个 $E_\theta$，同一套 pooling/head；不是 per-chart embedding |
| “保留完整谱面的监督，让梯度经过窗口聚合” | $d_m$ 由该图所有窗口的编码取加权平均后过 head；梯度从 BT loss 回到每个窗口、再回到编码器 |
| “模型可以自行学到哪些连续事件值得区分” | 不手工定义 jack/stream/LN 特征；编码器自己学局部表示，pooling 只做聚合 |

**三个可检验假设**：

* **H1a（行为监督改善表征）**：解冻上层 > 冻结 D2V。即“用行为梯度改编码器”
  比“只训练读出”在 H4 test 上更好。
* **H1b（遮蔽预训练有用）**：解冻上层 > 同架构随机初始化。即在同样的行为监督、
  同样的数据/预算下，D2V 的预训练初始化带来更好的图表示。
* **H1c（探索性）**：冻结 D2V vs 随机初始化。如果冻结已经很强，说明 D2V 作为
  **冻结特征提取器**有价值；如果随机更好，说明遮蔽预训练在这个监督下不是必需。

**估计量**：H4 test 上的 strict-pair rank-NLL；配对 ΔNLL 在同一 test 对/ceiling 上算。
不是 RMSE，不是 per-pair ACC 回归；RMSE 只作为诊断（§2.5.3）。

### 2.2 训练目标

对每个玩家段 $u$，取其训练集中玩过的图集合 $S_u$。对每一对 $i<j\in S_u$：

$$
y_{uij}=\mathbf{1}[\mathrm{loss}_{ui}<\mathrm{loss}_{uj}],\qquad
q_{ij}=\sigma(d_j-d_i),\qquad
\ell_{uij}=-[y_{uij}\log q_{ij}+(1-y_{uij})\log(1-q_{ij})]
$$

训练损失是 $\ell$ 在采样 pair 上的平均。ties（同一 loss）按 0.5 目标，与评测的
`tie="half"` 一致；不要用旧的 `tie="skip"`，否则训练目标与指标不一致
（`dcr/rank_metrics.py` 的 `pairwise_rank_loss` 已支持 `ties="half"`）。

**为什么玩家项可以没有**：若 $s_{ui}=\theta_u-d_i+\epsilon$，则
$s_{ui}-s_{uj}=d_j-d_i$，玩家能力 $\theta_u$ 在差值里消掉。所以 within-player
BT 只识别**难度差**；这正是 rank-NLL 的定义，也是“不需要玩家 embedding”的原因。
代价是：绝对难度水平不可识别（只识别相对序），所以 $d$ 的绝对值没有意义，
评测要重新拟合 link 的 $\beta$。

**与现有 `step2_cm3p_rank.py` 的关系**：那个脚本已经在优化**完全相同的**
marginal BT NLL，只是 $d_m=f_\theta(C_m)$ 的 $C_m$ 是冻结内容向量，
$f_\theta$ 是线性/MLP。本实验把 $f_\theta$ 换成“读原始音符的编码器 + pooling”，
并让梯度进入编码器。因此：

* 冻结臂是现有方法的直接延伸（D2V 内容 + 同一个 BT 目标），应能复现
  D2V 线性/MLP 的量级；这是 sanity check。
* 解冻/随机臂才是新东西；它们的对照对象是同一个头的冻结臂，不是旧论文的数字。

### 2.3 三臂与对照阶梯

#### 2.3.1 主三臂（旗舰指定）

| 项 | 冻结 | 解冻上层（top-4） | 随机初始化 |
|---|---|---|---|
| 初始化 | D2V holdout checkpoint | 同一 checkpoint | 同架构随机初始化 |
| conv（8→256→256→512） | 冻结 | 冻结 | 训练 |
| Transformer blocks 0–7 | 冻结 | 冻结 | 训练 |
| blocks 8–11 + final norm | 冻结 | **训练** | 训练 |
| d 头 | 训练 | 训练 | 训练 |
| pooling | 均值/有效帧加权 | 同 | 同 |
| 输入/窗口 | 同 | 同 | 同 |
| 训练 pairs | 同 | 同 | 同 |
| 损失 | BT | BT | BT |
| 早停 | chart-disjoint val rank-NLL | 同 | 同 |

“上层”默认 = 最后 4 个 Transformer block + final LayerNorm；conv 与底部 8 层冻结。
理由：D2V 的低层是通用 onset/hold 特征，高层才承载难度相关结构；先解冻 4 层是
算力/过拟合的折中。§2.4.5 有 k=2/6 的消融。

#### 2.3.2 对照阶梯（按需加跑）

主三臂只能回答“解冻 vs 冻结”“解冻 vs 随机”；它们各自混入了容量/初始化差异。
为了把机制拆开，准备下面三个二级对照（不是主表，结果放附录）：

| 编号 | 臂 | 目的 | 与主三臂的关系 |
|---|---|---|---|
| C1 | 冻结 D2V + 随机 4 层 Transformer adapter + 头 | 容量对照：给冻结臂同样多的可训练容量 | 若 C1 ≈ 解冻，增益是容量；若解冻 > C1，预训练上层有用 |
| C2 | 随机初始化 + 冻结 conv/底 8 层 + 训练 top-4/头 | 初始化对照：同样的可训练子集，随机 vs 预训练 | 若 C2 ≈ 解冻，top-4 的预训练权重不重要 |
| C3 | 随机初始化 + 全训练（即主三臂的随机臂） | 从零学全部 | 主三臂之一 |

C1/C2 只在主结果**方向不明确**时跑（例如解冻只比冻结好 0.001–0.003，
或随机臂异常强/弱）。不要一上来跑 6 个臂 × 5 seeds；先主三臂 1 seed 看方向。

#### 2.3.3 共享设置（避免“头不同/聚合不同”的解释）

| 设置 | 值 |
|---|---|
| 编码器 | D2V audio-style：conv 1.0M + 12L/512d/8heads/FFN2048，~39.1M trainable |
| 窗口 | 16 s，100 Hz，sigma=8 ms |
| 训练窗口数 K | 4（每图每步随机 4 个 16 s crop） |
| 评测窗口 | 全部不重叠 16 s 窗口，按有效帧数加权 |
| 图向量 | L2 归一化（主口径） |
| 头 | LayerNorm → Linear(512,256) → GELU → Dropout(0.1) → Linear(256,1) |
| 优化器 | AdamW，β=(0.9,0.999)，wd=0.01（encoder）/0.05（head） |
| lr | encoder 1e-4（解冻/随机），head 1e-3；cosine + 5% warmup |
| batch | 4 players × 6 charts（去重后 ≤24 张图；96 个窗口） |
| pair/step | 每玩家 15 对，约 60 对/step |
| 梯度裁剪 | 1.0 |
| 精度 | bf16（A800/A40），fp32 master |
| 早停 | val rank-NLL，patience=5，max 30 epoch |
| 模型种子 | 0（主）；1 只用于随机臂的初始化噪声检查 |
| 划分种子 | 1–5（H4 player split） |

**pair-view 预算**：H4 train 有 100.3M 个 within-player 比较。全跑一遍不现实，
也不必要。用**固定 pair-view 预算**而不是 epoch 数：

| 阶段 | pair views | 用途 |
|---|---:|---|
| smoke | 20k | 形状/梯度/内存 |
| pilot | 500k | 学习曲线、选 lr/早停 |
| main | 2M（若 val 仍在降则 5M） | 三臂主结果 |
| 三臂一致 | 同一预算 | 禁止某臂多跑 |

每 100k views 记一次 val；如果 2M 时 val 曲线仍明显下降，三臂一起加到 5M；
如果 2M 已平，三臂都用 2M。这个规则要在看 test 之前定下。


### 2.4 训练细节与实现

#### 2.4.1 数据采样

一个 step 的采样：

1. 从 train players 里抽 4 个段，要求每段在 train 里有 ≥2 张图；
2. 每段抽 6 张图（无放回；若不足 6 张则取全部），得到 ≤24 张唯一图；
3. 对每张唯一图随机采 4 个 16 s 窗口（随机起点，短图左侧 padding）；
4. 编码 96 个窗口 → 每图 4 个窗口向量 → 有效帧加权平均 → L2 → 头 → $d$；
5. 每个玩家在其 6 张图内形成 15 对，4 个玩家共 60 对；用 BT loss。

如果某玩家不足 2 张图，跳过；如果某图没有内容（113 张），从采样池剔除。
窗口起点用每 worker 的 RNG；不同 epoch 看到不同 crop，等价于数据增强。

伪代码：

~~~python
# charts: unique chart ids in this step; win_of: (M, K) window indices
x, valid, chart_idx = load_windows(charts, K=4)      # (M*K, 8, 1600)
feat, _ = encoder(x, valid_mask)                     # (M*K, 800, 512)
pooled = weighted_mean_by_chart(feat, valid, chart_idx)  # (M, 512)
z = F.normalize(head_norm(pooled), dim=-1)           # (M, 512)
d = head(z).squeeze(-1)                              # (M,)
loss = bt_nll(d, pair_i, pair_j, pair_y)             # mean over ~60 pairs
~~~

`bt_nll` 用 `dcr.rank_metrics.pairwise_rank_loss` 的公式，但输入是**图级 d**
和显式的 pair 索引，不是 per-row loss；不要直接复用它的“相邻行配对”实现。

#### 2.4.2 冻结臂

冻结臂不需要窗口采样：用 `chart_content_d2v_holdout.parquet` 的 512-d 向量
（已经是全窗口均值 + L2），直接训练头。这样：

* 与 D2V 的现有内容表口径完全一致；
* 头训练是标准监督学习，100M pair 的聚合 pair table 可以先用
  `build_pair_table` 建（若内存 >32 GB，按 chart-pair 分块聚合）；
* 结果应与 `step2_cm3p_rank.py --model mlp` 在 H4 上的量级一致；差太多说明
  头/训练协议写错了。

冻结臂的 pair table 只用 train block；val/test 不参与聚合。头训练用 L-BFGS
或 AdamW + 早停；因为只有头参数，收敛很快。

#### 2.4.3 随机臂

随机臂用同一个 `Data2VecModel` 架构初始化（conv + student；teacher 不参与），
全参数训练。注意：

* 随机臂没有预训练的位置编码统计，但 sinusoidal 位置是固定的，不是问题；
* 随机臂需要更多 step 才能收敛；用同样的 pair-view 预算 + 早停，若 val 仍降，
  允许把预算加到 5M，但三臂一致；
* 随机臂的初始化种子至少 0；如果结果与解冻臂接近，补 seed 1 看初始化噪声。

#### 2.4.4 关键实现检查

1. **梯度确实进入编码器**：解冻/随机臂在 backward 后检查
   `encoder.conv.weight.grad is None`（冻结臂应为 None/0；解冻臂 conv 为 None，
   blocks 8–11 非零；随机臂全部非零）。
2. **冻结参数不被 optimizer 更新**：optimizer 只接收 `requires_grad=True` 的参数；
   训练前后 hash 冻结权重，必须逐位相同。
3. **pooling 与评测一致**：训练用 4 个随机窗口，评测用全部窗口；在 smoke 里
   对同一张图用 4 窗口 vs 全窗口的 $d$ 相关性做检查（应 >0.95；否则训练/评测
   聚合不一致）。
4. **短图/坏图**：沿用 D2V 的 `min_valid_sec` 过滤；记录跳过数。
5. **测试图隔离**：H4 test chart 的 id 不出现在 train/val 的 chart 集合里；
   split JSON 里断言。
6. **确定性**：固定 seed、`torch.use_deterministic_algorithms` 可选；至少保证
   同一 config 重跑 val NLL 在 1e-4 内。

#### 2.4.5 消融清单

只在胜出的设置上跑，不要在每个臂上全跑一遍：

| 消融 | 值 | 目的 |
|---|---|---|
| 解冻层数 k | 2 / 4 / 6 | 容量 vs 预训练权重 |
| conv 是否解冻 | frozen / trainable | 低层特征是否也需要适配 |
| L2 归一化 | on / off | 幅度是否携带难度信息 |
| d 头 | linear / MLP | 非线性读出是否有用 |
| pooling | mean / attention | 固定聚合 vs 学习聚合 |
| 训练窗口 K | 4 / 8 / all | 训练/评测聚合差距 |

每个消融只跑 1 seed、与主臂同预算；如果主效应 <0.003，消融只作描述，
不据此宣布机制。

### 2.5 评价与判据

#### 2.5.1 主表

H4，5 个 player split seeds，strict-pairs，`--ceiling-fit all`：

| 方法 | test NLL（mean ± std） | Δ vs 冻结 | Δ vs 随机 | Pearson(d,b) | Spearman(d,b) | captured |
|---|---:|---:|---:|---:|---:|---:|
| 冻结 D2V | TBD | — | — | TBD | TBD | TBD |
| 解冻 top-4 | TBD | TBD | — | TBD | TBD | TBD |
| 随机初始化 | TBD | — | TBD | TBD | TBD | TBD |
| Sunny（参照） | TBD | — | — | TBD | TBD | TBD |
| MinaCalc（参照） | TBD | — | — | TBD | TBD | TBD |
| CM3P+ext MLP（参照） | TBD | — | — | TBD | TBD | TBD |

TBD 由 Gate 1 的冻结基线 + Phase 1 的三臂填入。Pearson/Spearman 是 per-chart
$d$ 与 cross-fitted $b$ 的相关（`pearson_bhat` / `rho_bhat`），只作描述；
主判据是 NLL。

#### 2.5.2 预注册判据

| 假设 | 支持（supported） | 不成立（not supported） | 不确定（inconclusive） |
|---|---|---|---|
| H1a 解冻 > 冻结 | Δ ≤ −0.003，5/5 种子为负，两个 95% CI 都排除 0，Pearson/Spearman 不差于 −0.005 | Δ ≥ 0，或 3/5 以上为正 | 方向对但 CI 含 0，或 3–4/5 |
| H1b 解冻 > 随机 | Δ ≤ −0.005，5/5，CI 排除 0 | Δ ≥ 0，或随机更好 | 方向对但 CI 含 0 |
| H1c 冻结 vs 随机 | 描述性；若 abs(Δ) ≥ 0.005 且 5/5 同号，报告方向 | — | — |

**不要**把“解冻比冻结低 0.001”写成“行为监督改善表征”；0.001 在 H4 的
between-seed 噪声内。0.003 是本文的“最小有意义效应”，0.010 是“强效应”。

**容量解释**：如果 H1a 成立但 C1（冻结 + 随机 adapter）也成立且差距 <0.001，
说明增益主要来自容量，不是“预训练上层被行为梯度改造”。这时结论要写成
“行为监督 + 足够容量改善读出”，不能写成“行为监督改善预训练表征”。

**H1b 的解释**：如果随机臂与解冻臂打平，说明在这个监督/数据量下，遮蔽预训练
的初始化不是必需；这不等于“D2V 没用”（冻结臂可能仍强），而是“从零训练也能
学会”。如果随机臂明显更差，才支持“遮蔽预训练提供可迁移的谱面先验”。

#### 2.5.3 诊断（不是主判据）

* **RMSE / MAE（ACC 尺度）**：把 $d$ 通过 link 映射回 loss，报告 test RMSE；
  只作诊断，因为 rank-NLL 只关心序。
* **分难度/分类型**：按 star 分箱（<3, 3–4, 4–5, 5–6, >6）、按 LN 比例
  （`ln_ratio`）、按 note 密度分箱，看 ΔNLL 是否集中在某一类。H4 高星偏置，
  分箱结果要标注样本量。
* **图级残差**：$d_m$ 与 ceiling $b_m$ 的差；看解冻臂是否只是把高星图整体拉高。
* **pair 级 calibration**：按 $|d_i-d_j|$ 分箱看 NLL；如果解冻臂只在近分对上
  更好，说明它学到了细粒度难度差。
* **窗口敏感性**：同一图用 4/8/全部窗口的 $d$ 的 Spearman；检查训练/评测
  aggregation 的差距。

#### 2.5.4 二级复核

* **P4-full**：用 full 的 60/40 chart split（8,380 test charts）重跑三臂的
  胜者 vs 冻结；符号一致才算稳健。注意 P4-full 的 test chart 多数被 D2V 预训练
  见过，对 H1b 有 transductive caveat；对 H1a（同一初始化下的解冻 vs 冻结）
  仍然有效，因为两臂都见过。
* **P4-1k**：与已有 P4 数字对齐；用 5 seeds，报配对 Δ。
* **VSRG benchmark**：只在最终模型上做一次，报告 isotonic 后的 OOF MAE，
  与 Sunny 0.3367、CM3P RM-L 0.4216、D2V-L 0.4527 比；这是外部效度，
  不是主指标（[benchmark_d2v.md](../benchmark_d2v.md) §0）。

### 2.6 算力

| 任务 | 资源 | 估计 |
|---|---|---:|
| H4 划分（5 seeds） | CPU | <10 min |
| 冻结臂（头 + pair table） | CPU 16 线程 | 0.5–2 h/seed |
| 解冻臂（2M views） | 1×A800，bf16 | 3–6 h/run |
| 随机臂（2M views） | 1×A800，bf16 | 3–8 h/run（可能更多 epoch） |
| 评测（H4 test） | CPU | 5–15 min/run |
| 三臂 × 3 seeds | 3×A800 | 约 1–2 天（含排队） |
| 三臂 × 5 seeds + C1/C2 | 3×A800 | 约 3–4 天 |

这些是**估计**，不是承诺。Gate 2 的 throughput benchmark 会用真实
step/s、显存、每 epoch 时间替换本表；如果 2M views 超出 6 h 队列上限，
用 checkpoint/resume 或减到 1M + 更长早停。

### 2.7 风险与缓解

| 风险 | 表现 | 缓解 |
|---|---|---|
| 解冻臂过拟合 | train NLL 降、val 升 | 早停、wd、dropout、少解冻（k=2）、冻结 conv |
| 冻结臂欠拟合 | 头很大仍 val 高 | C1 容量对照；头加大；lr 网格；不要用这个解释“解冻更好” |
| 随机臂欠训练 | val 仍在降 | 同预算下加到 5M views；报告学习曲线；不要提前宣布“预训练有用” |
| 训练/评测聚合不一致 | 4 窗口 d 与全窗口 d 相关性低 | 训练 K=8 或评测也用 4 窗口做一致性检查；主表用全窗口 |
| D2V checkpoint/内容表版本错 | 冻结臂数字漂移 | 记录 md5/rev；用 `extract.py` 重算 10 张图对齐 |
| H4 高星偏置 | 低星图没被测试 | P4-full 复核；或 Gate 1b 的 H4-full |
| BT 只识别相对序 | 绝对 d 无意义 | 不报绝对难度；评测重拟合 β；部署另做校准 |
| 100M pair table 内存 | OOM | 分块聚合；冻结臂用 sampled pairs 的 fallback |
| 6 h 队列上限 | job 被杀 | 每 500 step checkpoint；resume；或换长队列 |

## 3. 第 2 条：用整局判定分布弱监督局部难点

### 3.1 假设与可执行定义

旗舰原话 → 可执行定义：

| 原话 | 本文定义 |
|---|---|
| “整局的判定分布” | 一局（最佳 play）的六档计数 $c=(c_{\mathrm{MAX}},c_{300},c_{200},c_{100},c_{50},c_{\mathrm{miss}})$，$\sum c=N$ |
| “网络读取原始音符上下文，对每个判定事件输出六档概率” | 编码器帧特征 + 事件特征 → 每个 note 一个 6 类 softmax $p_{i,k}$ |
| “把全曲的预测概率求和，与这一局实际的六档计数比较” | $\hat{c}_k=\sum_i p_{i,k}$，用 $\hat{c}/N$ 与 $c/N$ 的交叉熵/KL |
| “训练时可用玩家在其他训练谱面上的表现估计水平” | 玩家基线 $b_u$（leave-one-chart-out 六档分布 + 技能标量），不是自由 embedding |
| “保留一个只读谱面的全局难度头，用原来的 rank-NLL 训练” | $d_m$ 头只吃图向量，用 §1.3 的 BT NLL |
| “部署时直接取这个头” | 推理只用 $d_m$；判定头是训练期辅助，不部署 |
| “把哪种键型导致这种错误留给模型学习” | 不手工标注难点；事件头 + 编码器自己学局部模式 |

**可检验假设**：

* **H2（主）**：在同一个编码器、同一批玩家/谱面、同一训练预算下，
  rank-NLL + 六档比例 比 rank-NLL only 在 H4 test 上 rank-NLL 更低。
* **H2-mech（机制）**：增益来自“判定分布的形状”，不是“多了一个辅助任务/更多参数”。
  判据是 +judge 优于 +judge-shuffled（placebo），且 +judge 优于 +single-grade
  （只加一个档位）。
* **H2-scope（范围）**：如果增益只出现在 LN 图或高星图，结论要限定；不能写成
  “所有图都变好”。

**为什么不是“换标签评价”**：主指标仍然是 $d_m$ 的 rank-NLL；判定分布只进训练，
不进评测。评测时 test 玩家的判定计数、test 图的判定计数都不参与任何拟合。

### 3.2 数据：六档判定侧表

#### 3.2.1 原始字段与映射

legacy 表（10k、randommerge 的源）行布局见
[step0_preprocess.py:425](../../scripts/step0_preprocess.py#L425)：

~~~text
0 score_id   1 beatmap_id  2 user_id   3 score     4 maxcombo  5 rank
6 count50    7 count100    8 count300   9 countmiss 10 countgeki 11 countkatu
12 perfect   13 enabled_mods 14 date      15 pp
~~~

modern 表（1k 的 `scores.sql`）的 JSON `statistics`：

~~~text
perfect -> MAX(320)   great -> 300   good -> 200
ok      -> 100        meh   -> 50    miss -> miss
~~~

统一成六列：

| 列 | legacy | modern | 权重（ACC） |
|---|---|---|---:|
| `n_max` | countgeki | perfect | 300 |
| `n300` | count300 | great | 300 |
| `n200` | countkatu | good | 200 |
| `n100` | count100 | ok | 100 |
| `n50` | count50 | meh | 50 |
| `n_miss` | countmiss | miss | 0 |
| `n_total` | 六列和 | 六列和 | — |

**已验证**：1k legacy 的 3,463,843 行里，有 meta 的 1,919,000 行
`sum(counts) == count_total` 的比例是 **100%**（0 mismatch）；modern 的
chart 193128 一局六档和 708 = `count_total` 708；.osu 对象数
（circle + hold）= 708，不是 taps + 2×holds（hold 只算一个判定）。
300 张 .osu 的 md5 与 `beatmap_meta_full.checksum` 全部匹配（300/300）。

**事件数**：一局判定的总数 = `count_total` = 4K 对象数（hold 只算一个事件；
规则上在 release 时判定，但 v1 同时取 head/tail 两个 gather 点，见 §3.3.2）。
所以事件头的事件数 = .osu 里的 circle + hold 数，
不是 2×hold。Gate 0 会对全部 20,832 张图重算这个等式；不匹配的图从第 2 条剔除。

#### 3.2.2 侧表构建

不改 step0 的 5 列主契约（避免影响所有下游），新增一个**侧表**：

~~~text
step0_judgements.py
  --dump-dir ... --tag 1k --whitelist ... --out data/processed/step0_judge_1k.parquet
输出列：
  player_id:int64, beatmap_id:int64, timestamp:datetime64[ns],
  n_max, n300, n200, n100, n50, n_miss, n_total : int32,
  source:int8, score_id:int64
~~~

* 复用 step0 的 4K/CL/去重规则；modern 与 legacy 的 dedup 与主表一致；
* 对 randommerge 的 4 个 snapshot 各跑一次，得到 4 张侧表；
* `step0_union_judge.py`：按 (player, beatmap_id, second) 合并，顺序与
  `step0_full.parquet` 的 `source_first` 一致；同一 key 的 counts 冲突时，
  取 loss 更低的那条（与主表标签一致），并记录冲突数；
* `step1_attach_judgements.py`：把 `step1_full_pairbest.parquet` 的每一对
  接到它**选中的最佳 play** 的 counts 上。join key =
  (player_id, beatmap_id, timestamp)；因为 union 按秒去重，key 唯一。
  输出 `step1_full_pairbest_judge.parquet`，在 step1 的 9 列后加
  `n_max...n_total, judge_ok`。

**选择偏差**：pairbest 选的是每对 loss 最小（ACC 最高）的 play，所以 counts 是
“最佳一局”的分布，不是平均分布。这与标签同源（标签也是最佳一局），是自洽的；
但它不是“这个玩家的典型一局”。v1 用最佳一局；§3.5.4 有“第一次 play / 平均 play”
的敏感性检查，不混主表。

**覆盖率 QA**（Gate 0 必须过）：

| 检查 | 通过线 |
|---|---|
| step1 每对都有 judge 行 | ≥99.9%；缺的行列出 |
| `n_total == count_total` | ≥99.9% |
| `n_total == .osu 对象数` | ≥99.9% |
| 每个玩家的 counts 和 >0 | 100% |
| 冲突 key 的 counts 差异 | 报告，不静默 |

不过线的图/对从第 2 条剔除；剔除数写进 JSON。

#### 3.2.3 分布（Gate 0 用 pairbest 重算）

第 0.3 节的样本统计说明信号存在：同一 ACC 下 MAX 率的组内 std 约 0.065，
坏档位也有可观的组内变化。Gate 0 要在 **pairbest 最佳 play** 上重算：

* 每档的边际比例；
* 按 ACC 0.001 分箱后，每档的组内 std；
* 玩家内（同一玩家不同图）与图内（同一图不同玩家）的方差分解；
* 六档的熵、MAX 率与 `ln_ratio` / `notes_per_sec` 的相关（描述性）。

目的是确认：在 (0.9, 0.998] 的标签支撑内，判定分布仍有足够的 chart 相关变异，
不是纯噪声。如果 chart 相关方差接近 0，第 2 条直接停止，不用训练。

### 3.3 模型

#### 3.3.1 总览

~~~text
                 ┌───────────────┐
  8ch piano roll →│  chart encoder │→ frame features (50 Hz, 512d)
                 └───────┬───────┘
                         ├── mean pool → z → d_head → d_m  (rank-NLL)
                         │
                         └── gather at events → h_i
                                                  │
   player baseline b_u (LOO 六档分布) ────────────┤
   player skill s_u (LOO 平均 loss) ──────────────┤
                                                  ▼
                                        event MLP → 6 logits → softmax
                                                  │
                                        sum over events → c_hat
                                                  │
                                        L_prop(c, c_hat/N)
~~~

* 编码器与第 1 条**完全相同**（同一个 checkpoint/初始化、同一个 pooling）；
* $d_m$ 头与第 1 条**完全相同**；第 2 条只加事件头和 $L_{prop}$；
* 部署时事件头丢弃，只保留 $d_m$ 头。所以“一张新谱进、一个难度分出”的接口不变。

#### 3.3.2 事件定义与特征

每个 4K 对象一个事件：

| 对象 | 判定时间 | 事件特征 |
|---|---|---|
| tap（circle） | onset | 帧特征 $h_i$ + 列 embedding(4) + type=0 + 局部密度 |
| hold | 尾判（release）；v1 同时取 head/tail 两个 gather 点，concat 或 mean | 帧特征 + 列 embedding + type=1 + hold 长度桶 + 局部密度 |

**为什么 hold 取尾判**：osu!mania 的 hold 在 release 时给判定；但现有数据只有
整局计数，无法逐事件验证。v1 用 head/tail 两个 gather 点的拼接，让模型自己选；
§3.5.4 有 head-only / tail-only / both 的消融。不要在没有 replay 的情况下声称
“我们知道是 head 还是 tail 导致的”。

**局部密度**（可选的 3 维，不依赖手工键型分类）：事件前后 200 ms 内的
note 数、同列 note 数、hold 覆盖数。这些是低层上下文，不命名 jack/stream/LN；
如果消融显示它们没用，就去掉。

**帧特征对齐**：编码器输出 50 Hz；事件时间（100 Hz）→ 帧索引
`round(t*50/1000)`，clamp 到窗口有效范围。事件按非重叠 16 s 窗口分配；
跨窗口的 hold 归到判定时间所在窗口。窗口 padding 的事件 mask 掉。

#### 3.3.3 玩家基线

对训练行 $(u,m)$，用玩家 $u$ 在**其他训练图**上的最佳 play counts 估计基线：

$$
b_{u,k} = \frac{\sum_{m'\ne m} c_{u,m',k} + \alpha \cdot \bar{c}_k}{\sum_{m'\ne m} N_{u,m'} + \alpha},\qquad
\alpha=5
$$

* 这是 leave-one-chart-out 的六档比例，$\bar{c}_k$ 是全局比例；
* $\alpha=5$ 的收缩避免 1–2 张图的玩家基线过噪；
* 技能标量 $s_u$ = 玩家在其他图上的平均 loss（越低越强），标准化到 train 的
  z-score；没有其他图的玩家用全局均值；
* **不用自由 player embedding**：v1 用固定的 LOO 基线，避免模型把目标图的
  counts 记进玩家向量。§3.4 有“可学习 player embedding”的消融，但只作为诊断。

事件 logits：

$$
\ell_{i,k} = \log(b_{u,k}+\epsilon) + f_k(h_i, e_i, s_u)
$$

其中 $f_k$ 是 2 层 MLP（输入 512 + 事件特征 + 1 维技能），最后一层零初始化，
所以训练开始时模型完全落在玩家基线上。$s_u$ 进入 MLP 允许“同一张图对不同水平
的人难度不同”的交互；如果显存/过拟合有问题，先做 additive（$f_k(h_i,e_i)$），
再加 $s_u$。

#### 3.3.4 标签比例损失

$$
p_{i,k}=\mathrm{softmax}(\ell_i)_k,\qquad
\hat{c}_k=\sum_i p_{i,k},\qquad
\hat{p}_k=\frac{\hat{c}_k}{\sum_j \hat{c}_j}
$$

主损失（多类比例交叉熵，等价于把 $\hat{p}$ 当 soft label 的 KL）：

$$
L_{prop} = -\sum_k w_k \cdot \frac{c_k}{N}\log(\hat{p}_k+\epsilon)
$$

* $w_k$ 是档位权重；默认 $w_k \propto 1/\sqrt{\text{freq}_k}$，归一化到均值 1。
  理由是 200/100/50/miss 很稀，不加权会被 MAX/300 淹没；权重从 **train** 的
  边际频率算，不用 test；
* 备选：Poisson-binomial 的方差加权高斯损失
  $\sum_k (c_k-\hat{c}_k)^2/\hat{\sigma}_k^2$，
  $\hat{\sigma}_k^2=\sum_i p_{i,k}(1-p_{i,k})$。主表用交叉熵，备选做消融；
* 不要用“把 counts 归一化后直接 MSE”作为主损失：它对稀有档位不敏感，且忽略
  了 counts 的尺度。可以做为第三个消融。

总损失：

$$
L = L_{rank} + \lambda L_{prop}
$$

* $\lambda$ 网格 {0.1, 0.3, 1.0, 3.0}；在 val 上按 **rank-NLL** 选，不按
  $L_{prop}$ 选（否则辅助目标反客为主）；
* rank-only 臂 $\lambda=0$；+judge 臂用选中的 $\lambda$；
* 两个臂的 encoder/头/预算/早停完全相同；只有 $\lambda$ 和事件头不同。

#### 3.3.5 为什么这叫 learning from label proportions

每个图是一个“bag”，bag 里是若干判定事件；我们只看到 bag 的六类计数，看不到
每个事件是哪一类。模型给每个事件一个类别分布，训练目标是让**预测的 bag 比例**
匹配**观测的 bag 比例**。这正是 learning from label proportions (LFLP) 的
多类别形式；与逐事件交叉熵不同，它对事件级标签不可识别，但对 bag 级比例是
可训练的。这个不可识别性也是限制：**全曲 counts 不能唯一确定哪些事件难**；
逐音热图必须用 replay 单独验证（§3.7）。

#### 3.3.6 训练实现

* 一个 step 采 4 个玩家 × 6 张图（与第 1 条同一采样器）；每图随机 4 个窗口；
* 编码 96 个窗口；对每图的事件 gather 帧特征，算 6 类 logits；按图 sum；
* 同一批图同时算 BT loss（图级 $d$）和 $L_{prop}$（图×玩家）；两个损失共享
  编码器；
* 事件数每图几百到几千；gather + MLP 是便宜的，显存主要在编码器；
* 如果一图的窗口数 >16，v1 截断到前 16 个非重叠窗口并记录；截断会改变 counts
  的比例，所以只对超长图做，并在 QA 里报告；
* 训练预算用 **row views**（图×玩家）而不是 epoch：pilot 200k，main 500k–1M；
  与第 1 条的 pair views 分开记，但两个臂一致。

伪代码：

~~~python
# charts: (M,) unique charts; each has K windows and E_m events
frame, valid = encode_chart_windows(charts, K=4)       # (M*K, 800, 512)
z = pool_to_chart(frame, valid, chart_idx)             # (M, 512)
d = d_head(z)                                          # (M,)
h = gather_events(frame, events, chart_idx)            # (E_total, 512)
logits = b_u[event_row] + event_mlp(h, event_feat, s_u[event_row])
p = softmax(logits, dim=-1)                            # (E_total, 6)
c_hat = scatter_sum(p, event_chart)                    # (M, 6)
L_prop = ce(c_obs / N, c_hat / N, weight=w)
L_rank = bt_nll(d, pair_i, pair_j, pair_y)
loss = L_rank + lam * L_prop
~~~

### 3.4 对照与消融

第 2 条的核心不是“加一个辅助头”，而是“判定分布的形状是否提供 chart 相关信号”。
所以对照必须能区分“信号”与“容量/正则/多任务”。

| 编号 | 臂 | 目的 | 预期 |
|---|---|---|---|
| A0 | rank-NLL only | 主基线 | — |
| A1 | + 六档比例（主） | H2 | 若 H2 成立，A1 < A0 |
| A2 | + 打乱六档比例（placebo） | 排除“多任务/容量”解释 | A2 ≈ A0 |
| A3 | + 单档位（miss / MAX / 200 三选一） | 判断信号是否集中在某一档 | 若 A3 ≈ A1，六档是冗余的 |
| A4 | + 玩家基线 only（事件头不看谱面特征） | 判断 chart 特征是否真的被用上 | A4 ≈ A0；A1 < A4 才说明谱面有用 |
| A5 | + 可学习 player embedding（诊断） | 检查玩家侧是否吸收了 chart 信号 | 只作诊断，不进主表 |

**A2 的打乱方式**：对每个**玩家**，把他名下的每图 counts 向量随机重排到他的
其他图上（保持该玩家的总 counts 和每图 N 不变，只破坏“哪张图对应哪个分布”）。
这样：
* 玩家基线 $b_u$ 不变；
* 每图的 counts 边际和 N 不变；
* 唯一被破坏的是 chart ↔ count 的对应关系。
如果 A2 仍然比 A0 好，说明增益来自“多一个辅助任务/更多参数/正则”，不是判定
分布本身。A2 必须用与 A1 完全相同的头、λ、预算。

**A3 的单档位**：只对一档做比例损失（例如只预测 miss 率），其他档不监督。
如果 miss-only 就能拿到 A1 的全部增益，说明“六档”是过度设计；如果 MAX-only
或 200-only 也有增益，说明不同档位携带不同信息。A3 是机制分析，不是主结果。

**A5 的诊断**：给每个训练玩家一个 8 维自由 embedding，替换 LOO 基线。如果
A5 的 rank-NLL 明显好于 A1，但 A5 的判定头在 val 上对**新图**没有更好，说明
玩家 embedding 记住了 chart 信号；这时不能用 A5 的结果支持 H2。

**超参预算**：A1 的 λ 在 val 上选；A2/A3 用与 A1 相同的 λ（不再单独搜），
避免 A2 因为 λ 不同而“看起来更差”。A0 没有 λ；但 A0 也做同样的 lr/早停搜索，
保证预算一致。

### 3.5 评价与判据

#### 3.5.1 主表

H4，5 个 player split seeds，strict-pairs：

| 方法 | test NLL（mean ± std） | Δ vs A0 | Δ vs A2 | 备注 |
|---|---:|---:|---:|---|
| A0 rank-only | TBD | — | — |  |
| A1 + 六档比例 | TBD | TBD | TBD | 主 |
| A2 + 打乱比例 | TBD | TBD | — | placebo |
| A3 + miss-only | TBD | TBD | — | 机制 |
| A3 + MAX-only | TBD | TBD | — | 机制 |
| A4 + 基线 only | TBD | TBD | — | 机制 |
| Sunny / MinaCalc / CM3P+ext | TBD | — | — | 参照 |

#### 3.5.2 预注册判据

| 假设 | 支持 | 不成立 | 不确定 |
|---|---|---|---|
| H2（A1 < A0） | Δ ≤ −0.003，≥4/5 种子为负，两个 95% CI 排除 0；A2 的 Δ 在 ±0.002 内且 CI 含 0 | A1 ≥ A0，或 A2 与 A1 同幅度改善 | 方向对但 CI 含 0，或 3/5 |
| H2 强 | Δ ≤ −0.005，5/5，CI 排除 0，A2 无效 | — | — |
| H2-mech（形状而非单档） | A1 比最好的 A3 再低 ≥0.002，且 A2 无效 | A3 ≈ A1，说明单档足够 | 差异在噪声内 |

**不要**把 A1 比 A0 低 0.001 写成“判定分布改善难度”；0.001 在 H4 的
between-seed 噪声内。也不要只报 A1 vs A0 而省略 A2；没有 placebo 的辅助任务
结果不可解释。

**即使 A1 仍落后 Sunny / MinaCalc**，只要配对 ΔNLL 稳定为负、placebo 无效，
就说明“多看到真实判定分布”值得继续投入；与 Sunny 的绝对差距不是这条线的判据。
反过来，如果 A1 与 A0 打平，即使绝对 NLL 很好看，也不能说判定分布有用。

#### 3.5.3 操纵检查（manipulation check）

在看 test 之前，必须在 val 上确认“判定头真的学到了 chart 相关信号”：

1. **事件级 NLL**：A1 的判定头在 val 图上的 per-event grade NLL，必须比
   “只用玩家基线、不看谱面”的 A4 低一个预先定的幅度。pilot 先测 A4 的 NLL，
   把阈值定为 A4 的 5%（或 0.02 nats/event，取更保守者），写进 config；
2. **图级边际**：A1 预测的每图 MAX 率 / miss 率，与 val 图实际 counts 的
   Spearman 必须 >0.2（pilot 定阈值）；如果接近 0，说明模型没学到 chart 变异；
3. **A2 的 sanity**：打乱后同一指标的 Spearman 应接近 0。

如果操纵检查失败，A1 的 rank-NLL 结果**不可解释**：不是“判定分布没用”，而是
“判定头没学会”。这时先修模型/数据，不要直接报零结果。

#### 3.5.4 分层与稳健性

* **LN vs RC**：用 `ln_ratio` / `patt_ln_percent` 分箱；判定分布对 LN 图
  可能更有用（hold 尾判、长条密度）。如果增益只在 LN，结论限定；
* **star 分箱**：H4 高星偏置；P4-full 复核低/中星；
* **玩家活跃度**：按每玩家对数分箱；判定头的 LOO 基线对低活跃玩家更噪，
  增益可能集中在高活跃玩家；
* **最佳一局 vs 第一次/平均**：用第一次 play 或该对全部 play 的平均 counts
  重跑 A1（不重跑 A0），看结论是否依赖“最佳一局”的选择；
* **hold 事件时间**：head-only / tail-only / both 三个消融，看 rank-NLL 与
  判定 NLL 的差异；不声称逐事件定位。

#### 3.5.5 二级复核

* **P4-full**：full 的 8,380 test charts，5 seeds；符号一致才算稳健；
* **P4-1k**：与已有数字对齐；
* **VSRG benchmark**：最终模型做一次 isotonic OOF MAE；判定头不参与部署。

如果 H2 在 H4 上成立但在 P4-full 上不成立，结论写成“在 D2V-unseen 的高星图上
成立，在全难度分布上未复现”；不要只报有利的口径。

### 3.6 算力

| 任务 | 资源 | 估计 |
|---|---|---:|
| 判定侧表（1k + 10k + randommerge） | CPU 8 线程 | 1–3 h |
| union + pairbest join + QA | CPU | 0.5–1 h |
| .osu 对象数校验（20,832 图） | CPU | 5–10 min |
| Phase 2a 冻结探针（特征缓存 + 头） | CPU/1 GPU | 1–2 h/seed |
| Phase 2b 端到端 A0/A1（各 500k–1M row views） | 1×A800 | 4–8 h/run |
| A0/A1 × 3 seeds | 3×A800 | 约 1 天 |
| A0/A1 × 5 seeds + A2/A3/A4 | 3×A800 | 约 2–3 天 |
| 评测 | CPU | 10–20 min/run |

Phase 2a 的目的不是主结果，而是：

1. 验证事件头能学到判定分布（操纵检查）；
2. 估算 A1 的效应大小与噪声，决定 Phase 2b 是否值得跑 5 seeds；
3. 选 λ 和 class weights 的候选范围。

Phase 2a 用冻结 D2V 的**每事件帧特征缓存**（20,832 图 × 事件数 × 512d，
float16 约 10–20 GB；可按图分片）。如果缓存太大，只缓存有 ≥5 对的图或
先做 1k 子集。

### 3.7 风险与限制

| 风险 | 表现 | 缓解 |
|---|---|---|
| ACC 冗余 | A1 只学到 ACC，没有额外信息 | 主信号是 MAX/300 与坏档位；A2 打乱；单档位消融；检查预测 MAX 率 |
| 稀有档位塌缩 | 预测 200/100/50/miss 全为 0 | class weights、focal、监控每档召回；A3 单档位 |
| 最佳一局选择偏差 | counts 不代表典型表现 | 第一次/平均 play 敏感性；结论限定“最佳一局” |
| counts 与 .osu 不匹配 | 事件数 ≠ N | Gate 0 全量校验；不匹配图剔除 |
| 玩家基线泄漏 | 模型把目标图 counts 记进玩家 | 只用 LOO；不用自由 embedding；A5 只作诊断 |
| hold 判定时间不确定 | 事件特征取错位置 | head/tail/both 消融；不声称逐事件定位 |
| 全曲 counts 不可识别 | 无法定位难点 | 不发布热图；需要 replay 才做定位 |
| 长图显存 | 事件/窗口太多 | 截断到 16 窗；gradient checkpointing；按图分片 |
| test counts 泄漏 | test 图 counts 进训练 | split 断言；test counts 只在评测/诊断用 |
| 多重比较 | 跑很多 λ/档位组合 | 主假设预注册；机制分析标 exploratory |

**最重要的限制**：整局六档计数只能告诉模型“错成了什么样”，不能告诉它
“哪个键型导致”。如果以后要发布逐音难点热图，必须用少量 replay 做定位验证；
本文的所有结果都不能当作逐音定位的证据。

---










## 4. 第 3 条：从“哪些玩家会在这张图上失分”学习谱面表示

### 4.1 假设与可执行定义

旗舰原话 → 可执行定义：

| 原话 | 本文定义 |
|---|---|
| “A、B 两张图的全局难度相同，但不同玩家在 A/B 上表现相反” | 教师预测的**参考玩家相对表现向量** $s_m\in\mathbb{R}^K$，在每张图内中心化，去掉全局难度 |
| “一个标量难度标签抹掉了这个差别” | $d_m$ 只是 $s_m$ 的图内均值；$s_m$ 保留玩家×图的交互 |
| “先用训练集成绩拟合一个行为教师，借用已有的 mf_dot” | `step2/predict --models bias,mf_dot`，只用 train block |
| “让教师预测一组固定参考玩家在各张训练图上的相对表现” | 选 K 个参考玩家，教师对每张训练图输出 K 个预测 loss，图内 z-score |
| “训练只读原谱的学生网络，同时预测这串值和原有全局难度” | 学生 = 编码器 + d 头 + K 维 s 头；$L=L_{rank}+\lambda L_{sig}$ |
| “教师、参考玩家以及辅助标签全部限于训练折” | 所有拟合/选择只用 train；test 玩家/图不参与 |

**可检验假设**：

* **H3（主）**：$d+\text{整组参考玩家}$ 比 $d+\text{平均难度（标量）}$ 在 H4 test
  上 rank-NLL 更低。
* **H3-mech**：$d+\text{整组}$ 优于 $d+\text{打乱签名}$（placebo），说明增益来自
  签名的 chart 相关结构，不是多任务/容量。
* **H3-scope**：如果增益只在某些玩家活跃度/图类型上，结论限定；不声称“学会了
  jack/stream/LN 这些技能维度”，只声称“保留了可迁移的玩家分歧”。

### 4.2 教师：mf_dot 与参考玩家签名

#### 4.2.1 教师模型

现有 oracle 基线：

$$
\hat{\ell}_{um} = b_0 + b_u[u] + b_m[m] + \langle P[u], C[m]\rangle
$$

* $P[u]\in\mathbb{R}^D$ 玩家潜向量，$C[m]\in\mathbb{R}^D$ 谱面潜向量；
* $D=8$ 或 16（[models.md](../models.md) §2 的默认 dim=16；默认口径 A 的
  mf_dot 用 dim=8，见同文 §2 的主结果表）；
* 训练目标：step2/predict 的 loss 回归（可选 `--rank-loss add`）；**只用 train block**；
* 产物：npz 潜向量 + 每对预测；或 `--save-preds` 的逐对预测表。

**为什么不用 mf_dot 当 chart-only 基线**：它含玩家项，不能给新谱一个纯 chart
的难度；它的作用是**教师**，不是学生。学生的部署接口仍然是“新谱 → 一个 d”。

这类“先从听众行为提取协同信号，再训练内容网络从音频预测它”的做法在音乐推荐里
有先例；这里把音频换成谱面、把听众换成玩家，教师从 mf_dot 换成“固定参考玩家的
反应向量”。

#### 4.2.2 参考玩家

从 **train players** 里选 K 个固定参考玩家：

* K=64（主）；16/128 做敏感性；
* 过滤：train 里 ≥30 个 pair（保证 $P[u]$ 可估）；
* 分层：按玩家平均 loss 的 5 个分位层 × 活跃度的 3 层，层内随机/取中位；
* 备选：对 $P[u]$ 做 k-means（K 个簇），每簇取离中心最近的玩家；
  这样参考玩家覆盖潜空间的多个方向，而不是只覆盖难度；
* 参考玩家一旦选定，**固定写入 JSON**（id 列表 + 选择规则 + seed），
  所有学生臂共用同一组；不允许每个 seed 重新选（否则签名不可比）。

#### 4.2.3 签名

对每张 train 图 $m$ 和参考玩家 $r$：

$$
\tilde{s}_m[r] = \hat{\ell}_{r m} - \frac{1}{K}\sum_{r'}\hat{\ell}_{r' m},
\qquad
s_m[r] = \frac{\tilde{s}_m[r]}{\mathrm{sd}_r(\tilde{s}_m)+\epsilon}
$$

* 图内中心化：去掉“这张图整体多难”，只留“谁相对更好/更差”；
* 图内标准化：去掉教师输出的整体尺度，避免 $L_{sig}$ 被大方差图主导；
* 可选：先对 $\hat{\ell}_{rm}$ 做玩家侧中心化（减去该参考玩家在所有图上的均值），
  再做图内中心化；两种都试，主表用“图内中心化 + 图内标准化”；
* **不要**把 $s_m$ 与 $d_m$ 一起标准化到同一尺度：$d$ 由 BT 学，$s$ 是辅助目标；
  两者的尺度不需要一致。

**可识别性**：mf_dot 的 $P$ 与 $C$ 可以同时做正交旋转而不改变
$\langle P,C\rangle$；所以潜向量本身没有固定含义。固定参考玩家后，
$s_m$ 是“这 K 个人在这张图上的相对表现”，有固定含义；图内中心化进一步去掉
可加常数。这个设计正是旗舰说的“潜向量可以任意旋转，参考玩家上的预测有固定含义”。

#### 4.2.4 教师 QA

在看学生结果之前，必须报告：

| 检查 | 目的 | 通过线 |
|---|---|---|
| 教师 held-out NLL | 教师对**没见过的玩家**是否有预测力 | 与 bias 基线比，ΔNLL < 0；否则签名不可信 |
| 5 个教师 seed 的签名相关 | 签名是否稳定 | 每图 $s_m$ 的跨 seed 平均相关 >0.7 |
| 签名方差分解 | 图内中心化后还剩多少方差 | 不能接近 0；否则没有分歧可学 |
| 与 $d$ 的相关 | 签名是否只是难度的非线性变换 | 每维与 $d$ 的 \|corr\| 不应全 >0.9 |
| 参考玩家覆盖 | 是否只选了高活跃/高段玩家 | 报告技能/活跃度分布 |

教师 held-out 的做法：按玩家 cluster 5 折，fit 4 折、预测 1 折；
只报告 fold-test 玩家的 NLL。不要用 in-sample NLL 作为教师质量。

### 4.3 学生

#### 4.3.1 架构

* 编码器：第 1 条胜出的设置（默认解冻 top-4；如果第 1 条冻结赢，则用冻结）；
* $d$ 头：与第 1 条完全相同；
* $s$ 头：LayerNorm → Linear(512,256) → GELU → Dropout(0.1) → Linear(256,K)；
* 部署只用 $d$ 头；$s$ 头是训练期辅助。

#### 4.3.2 损失

$$
L = L_{rank} + \lambda L_{sig},\qquad
L_{sig} = \frac{1}{K}\sum_{r=1}^{K}\left(\hat{s}_m[r]-s_m[r]\right)^2
$$

* $\lambda$ 网格 {0.1, 0.3, 1.0, 3.0}；在 val 上按 rank-NLL 选；
* 备选：cosine 相似度损失（对整体尺度更鲁棒）；主表用 MSE；
* 如果 $s$ 头容量过大导致过拟合，先降 hid 到 64 或加 wd；
* 训练预算与第 1 条同一 pair-view 预算；$s$ 目标对每张 train 图都有，
  不额外增加采样。

#### 4.3.3 对照

| 编号 | 臂 | 目的 | 预期 |
|---|---|---|---|
| B0 | d-only（rank-NLL） | 主基线 | — |
| B1 | d + 平均教师难度（标量） | 控制“多一个辅助标量” | B1 ≈ B0；若 B1 已很好，说明分歧不重要 |
| B2 | d + 整组 K 维签名（主） | H3 | B2 < B1 才支持分歧有用 |
| B3 | d + 打乱签名（placebo） | 排除多任务/容量 | B3 ≈ B0/B1 |
| B4 | d + K=16 / 128 | 维度敏感性 | 若 K 越大越好，支持分歧；若平，标量足够 |
| B5 | d + 单教师 seed vs 5-seed 集成 | 教师噪声 | 集成应更稳；若不稳，签名不可信 |

**B1 的“平均教师难度”**：$\bar{s}_m=\frac{1}{K}\sum_r\hat{\ell}_{rm}$，
即教师对参考玩家的平均预测（图内未中心化）。它与 $d_m$ 高度相关，但来源不同
（教师 vs 行为 BT）。B1 是“只蒸馏平均难度”的对照，正是旗舰要区分的对象。

**B3 的打乱方式**：把 $s_m$ 在 train 图之间随机重排（保持每维边际、破坏
chart ↔ 签名对应）。与第 2 条的 A2 同理。

### 4.4 评价与判据

#### 4.4.1 主表

H4，5 个 player split seeds，strict-pairs：

| 方法 | test NLL（mean ± std） | Δ vs B0 | Δ vs B1 | 备注 |
|---|---:|---:|---:|---|
| B0 d-only | TBD | — | — |  |
| B1 + 平均难度 | TBD | TBD | — | 标量对照 |
| B2 + K=64 签名 | TBD | TBD | TBD | 主 |
| B3 + 打乱签名 | TBD | TBD | — | placebo |
| B4 K=16 / 128 | TBD | TBD | — | 敏感性 |
| Sunny / MinaCalc / CM3P+ext | TBD | — | — | 参照 |

#### 4.4.2 预注册判据

| 假设 | 支持 | 不成立 | 不确定 |
|---|---|---|---|
| H3（B2 < B1） | Δ ≤ −0.003，≥4/5 种子为负，两个 95% CI 排除 0；B3 的 Δ 在 ±0.002 内 | B2 ≥ B1，或 B3 与 B2 同幅度改善 | 方向对但 CI 含 0 |
| H3 强 | Δ ≤ −0.005，5/5，CI 排除 0，B3 无效 | — | — |
| 维度 | B2(K=64) 比 B4(K=16) 低 ≥0.002，或 K=128 不再改善 | K=16 已足够，说明低维分歧就够 | 差异在噪声内 |

**不要**把“B2 比 B0 低 0.002”当成 H3 成立；H3 的核心是 B2 < B1，因为 B1 已经
包含了“多一个辅助目标”的收益。如果 B2 ≈ B1，说明整组参考玩家的分歧没有带来
超过平均难度的信息；这时应该报告“分歧对全局 rank-NLL 不重要”，而不是硬说
“保留玩家分歧有用”。

#### 4.4.3 操纵检查

1. **s 头对教师的拟合**：在 val 图上，$\hat{s}$ 与 $s$ 的平均相关必须 >0.3
   （pilot 定阈值）；否则辅助信号没学会，B2 的零结果不可解释；
2. **教师质量**：§4.2.4 的 held-out NLL 不过线，签名不可信；
3. **B3 的 sanity**：打乱后 s 头与教师签名的相关应接近 0。

#### 4.4.4 探索性诊断（不是主判据）

* **签名是否预测 test 玩家的残差**：对 test 图，取每个 test 玩家在该图上的
  相对 loss（图内 z-score）；用一半 test 图拟合“学生 $\hat{s}$ → 玩家残差”的
  ridge，在另一半上评估。这个映射用了 test 标签，只作探索性诊断，不能当主结果；
* **参考玩家敏感性**：换一组参考玩家（不同 seed），看 B2 的 rank-NLL 是否稳定；
* **签名维度**：对 $s_m$ 做 PCA，看前几维与 star/LN/密度的相关；描述性；
* **玩家活跃度分层**：低活跃玩家的教师预测更噪，签名可能更差；报告分箱。

### 4.5 算力

| 任务 | 资源 | 估计 |
|---|---|---:|
| mf_dot 教师（dim 8/16） | CPU 8 线程 | 10–30 min/seed |
| 签名构建 + QA | CPU | 10 min |
| 教师 5-seed 集成 | CPU | 1–2 h |
| B0–B3 端到端（各 2M pair views） | 1×A800 | 3–6 h/run |
| B0–B3 × 3 seeds | 3×A800 | 约 1 天 |
| B0–B3 × 5 seeds + B4 | 3×A800 | 约 2–3 天 |
| 评测 | CPU | 10–20 min/run |

Phase 3a 的冻结探针（冻结 D2V + d 头 + s 头）可以先跑，成本与第 1 条冻结臂
同一量级；它回答“签名是否可学”，不回答“编码器是否被改善”。Phase 3b 才用
第 1 条胜出的编码器做端到端。

### 4.6 风险与限制

| 风险 | 表现 | 缓解 |
|---|---|---|
| 教师 in-sample 过拟合 | 签名像 chart id | 教师正则/降维/5-seed 集成；held-out NLL；B3 placebo |
| 参考玩家不代表全体 | 签名只覆盖高段/高活跃 | 分层/聚类选人；报告覆盖；换一组做敏感性 |
| 签名与 d 冗余 | B2 ≈ B1 | 图内中心化；检查每维与 d 的相关；K 敏感性 |
| 教师 misspecification | mf_dot 欠拟合 | dim 8/16；`--rank-loss add`；教师 held-out NLL 门槛 |
| 旋转/尺度 | 签名不可复现 | 固定参考玩家；图内 z-score；写 JSON |
| 泄漏 | test 玩家/图进教师 | 只用 train block；split 断言 |
| 多任务容量 | B3 也改善 | B3 placebo；同头/同 λ/同预算 |
| K 太大过拟合 | K=128 更差 | K=16/64/128；wd；早停 |

**不要声称的东西**：不要声称学生“学会了 jack/stream/LN”这些命名技能维度。
我们只知道它保留了参考玩家之间的相对分歧；这些分歧是否对应人类可命名的技能，
需要额外的可解释性分析（PCA/探针），不在本文的判据内。

---

## 5. 执行计划

### 5.1 Phase 0：数据与基础设施（第 1 周）

目标：在跑任何 GPU 之前，把“判定侧表、H4 划分、基线、吞吐”四件事做完。
Phase 0 的任何一项不过，后面的结果都不可解释。

#### P0.1 判定侧表程序

| 项 | 内容 |
|---|---|
| 交付 | `scripts/step0_judgements.py` + `scripts/test_step0_judgements.py` |
| 输入 | dump 的 legacy/modern 表、whitelist、beatmap_meta |
| 输出 | `data/processed/step0_judge_{1k,10k,randommerge}.parquet` |
| 验收 | 合成行测试通过；1k 上 `sum(counts)==count_total` 的覆盖率 ≥99.9%；现代/legacy 同图同 play 的 counts 一致（抽样 1,000 对） |
| 依赖 | 无 |
| 算力 | CPU 1–3 h |

命令（目标 CLI）：

~~~bash
python scripts/step0_judgements.py \
  --dump-dir data/raw/extracted/2026_09_01_performance_mania_top_1000 \
  --tag 1k --whitelist data/processed/whitelist_1k.csv \
  --out data/processed/step0_judge_1k.parquet

python scripts/step0_judgements.py \
  --dump-dir data/raw/extracted_10k/2026_09_01_performance_mania_top_10000 \
  --beatmap-src data/raw/extracted/2026_09_01_performance_mania_top_1000 \
  --tag 10k --out data/processed/step0_judge_10k.parquet
~~~

randommerge 的 4 个 snapshot 各跑一次（命令同上，改 `--dump-dir` / `--tag`），
再用 `step0_union_judge.py` 合并成 `step0_judge_random.parquet`。

#### P0.2 union + pairbest join + QA

| 项 | 内容 |
|---|---|
| 交付 | `scripts/step0_union_judge.py`、`scripts/step1_attach_judgements.py`、QA JSON |
| 输入 | 1k/10k/randommerge 的 judge 侧表、`step0_full.parquet`、`step1_full_pairbest.parquet`、`beatmap_meta_full.csv` |
| 输出 | `step0_judge_full.parquet`、`step1_full_pairbest_judge.parquet` |
| 验收 | 每对 coverage ≥99.9%；`n_total==count_total` ≥99.9%；`n_total==.osu 对象数` ≥99.9%；冲突 key 全部列出；掉落行数写 JSON |
| 依赖 | P0.1 |
| 算力 | CPU 1–2 h |

~~~bash
python scripts/step0_union_judge.py \
  --judge 1k=data/processed/step0_judge_1k.parquet \
  --judge 10k=data/processed/step0_judge_10k.parquet \
  --judge randommerge=data/processed/step0_judge_random.parquet \
  --plays data/processed/step0_full.parquet \
  --out data/processed/step0_judge_full.parquet \
  --report data/processed/step0_judge_full.json

python scripts/step1_attach_judgements.py \
  --in data/processed/step1_full_pairbest.parquet \
  --judge data/processed/step0_judge_full.parquet \
  --meta data/processed/beatmap_meta_full.csv \
  --osu-dir data/osu_cache \
  --out data/processed/step1_full_pairbest_judge.parquet \
  --json data/processed/step1_full_pairbest_judge.json
~~~

#### P0.3 H4 划分

| 项 | 内容 |
|---|---|
| 交付 | `scripts/step1_holdout_split.py`、5 个 H4 split parquet + JSON |
| 输入 | `step1_full_pairbest.parquet`（或 judge 版）、`data2vec_main/splits/holdout/eval_ids.txt` |
| 输出 | `data/processed/step1_full_holdout_s{1..5}.parquet` |
| 验收 | test chart 全在 holdout 且不出现在 train/val；test base player 不出现在 train/val；val chart 与 train/test 互斥；test 行数与 §1.2.2 一致（±1%），train 行数 = 毛数减 val（约 −10%）；5 seeds 可复现 |
| 依赖 | 无（可与 P0.1 并行） |
| 算力 | CPU <10 min |

~~~bash
python scripts/step1_holdout_split.py \
  --in data/processed/step1_full_pairbest.parquet \
  --holdout-ids data2vec_main/splits/holdout/eval_ids.txt \
  --out-prefix data/processed/step1_full_holdout \
  --seeds 1,2,3,4,5 --player-frac 0.6 --bins 5 --val-chart-frac 0.1
~~~

第 2 条需要带六档 counts 的划分：等 P0.2 完成后，用同一个脚本、
`--in data/processed/step1_full_pairbest_judge.parquet`、
`--out-prefix data/processed/step1_full_holdout_judge` 再跑一次。
两条线的 split 种子/玩家分法必须逐位相同（同一个 `--seeds` 与 `--bins`）。

#### P0.4 checkpoint 与编码器

| 项 | 内容 |
|---|---|
| 交付 | checkpoint 位置确认、loader、10 张图对齐测试 |
| 输入 | `data2vec_main/runs/main_holdout/checkpoints/last.pt`（HPC） |
| 验收 | ckpt 可加载；`extract.py` 重算 10 张图与 `chart_content_d2v_holdout.parquet` 的 max abs diff <1e-5；记录 md5/rev |
| 依赖 | HPC 访问 |
| 算力 | GPU 5 min |

**当前事实**：本地没有完整 D2V checkpoint；只有 `data2vec_main/runs/smoke/checkpoints/last.pt`
（smoke 2L/64d，不能用）。HPC 路径见 [benchmark_d2v.md](../benchmark_d2v.md) §2.1：
`data2vec_main/runs/main_holdout/checkpoints/last.pt`。如果文件丢失：

* 用 `data2vec_main/configs/audio_style_base.yaml` + `splits/holdout/pretrain_ids.txt`
  重训 20 epoch（A800 约 47 min–1 h；README §6.2）；
* 重训后必须重抽 `chart_content_d2v_holdout.parquet`，并记录新 rev；
* 不要用旧内容表配新 checkpoint（图向量不一致）。

#### P0.5 训练器骨架与吞吐

| 项 | 内容 |
|---|---|
| 交付 | `scripts/step2_chart_encoder.py` 的 `--task rank` 骨架；20k views smoke；throughput JSON |
| 输入 | H4 s1、osu_cache、D2V ckpt |
| 验收 | 前向/反向跑通；梯度检查（§2.4.4）；显存/step/s/每 100k views 时间；6 h 队列内能完成 2M views 或能 resume |
| 依赖 | P0.3、P0.4 |
| 算力 | GPU 0.5–1 h |

~~~bash
cd $HOME/DCR
export PYTHONNOUSERSITE=1
PY=$HOME/.conda/envs/DCR/bin/python
$PY scripts/step2_chart_encoder.py \
  --task rank --init d2v \
  --ckpt data2vec_main/runs/main_holdout/checkpoints/last.pt \
  --freeze topk --unfreeze-blocks 4 \
  --split-table data/processed/step1_full_holdout_s1.parquet \
  --osu-dir data/osu_cache \
  --out-dir runs/behavior/smoke_e1 --benchmark-views 20000
~~~

#### P0.6 H4 基线

| 项 | 内容 |
|---|---|
| 交付 | Sunny / MinaCalc / star / 冻结 D2V / CM3P+ext 的 H4 test 指标 |
| 输入 | H4 s1–s5、rank_signals_1k.parquet、内容表 |
| 输出 | `results/behavior/baselines/h4_s{1..5}.json` |
| 验收 | 同一 test 对/ceiling；ceiling 与固定信号可复现；Sunny/MinaCalc 覆盖 1,363 图 |
| 依赖 | P0.3 |
| 算力 | CPU 1–2 h |

~~~bash
python scripts/step2_rank_eval.py \
  --in data/processed/step1_full_holdout_s1.parquet \
  --signal-table data/processed/rank_signals_1k.parquet \
  --signals sunny_star,msd_overall,star \
  --split test --ceiling-fit all --strict-pairs \
  --out results/behavior/baselines/h4_s1_fixed.json
~~~

**Phase 0 的 go/no-go**：

* P0.1–P0.3 全过 → 进入 Phase 1；
* 判定 QA 覆盖率 <99.9% 或对象数不匹配 >0.1% → 先修数据，第 2 条不启动；
* checkpoint 丢失且重训排不上 → 第 1 条先用冻结臂 + 随机臂（跳过解冻臂），
  但这只能回答 H1b 的一半；报告里写清楚。

### 5.2 Phase 1：第 1 条（第 1–3 周）

| 任务 | 内容 | 判据 | 算力 |
|---|---|---|---|
| P1.1 pilot | 三臂各 500k views，seed 1 | 学习曲线；选 lr/早停；估算 2M 时间 | GPU 3×1–2 h |
| P1.2 seed 1 | 三臂各 2M views | 解冻 − 冻结 ≤ −0.003 且随机更差 → 继续；否则先跑 C1/C2 | GPU 3×3–8 h |
| P1.3 seeds 2–3 | 三臂 | 5/5 符号一致；between-seed std | GPU 6×3–8 h |
| P1.4 C1/C2 | 仅在方向不明确时 | 容量 vs 初始化 | GPU 2×3–8 h |
| P1.5 二级协议 | 胜者 vs 冻结，P4-full/P4-1k | 符号一致 | GPU 10–20×2–6 h |

**决策门**：

* 解冻 > 冻结 → 解冻编码器作为第 2/3 条 backbone；
* 冻结 ≥ 解冻 → 第 2/3 条改用冻结 D2V 或轻量 adapter；报告写“行为监督没有
  改善表征”；
* 随机 ≥ 解冻 → 保留解冻 backbone，但不能声称“遮蔽预训练提供可迁移先验”。

### 5.3 Phase 2：第 2 条（第 3–5 周）

| 任务 | 内容 | 判据 | 算力 |
|---|---|---|---|
| P2.1 冻结探针 | A0/A1/A2/A3/A4，seed 1 | 事件头 val NLL 比 A4 低 ≥5%；MAX 率 Spearman >0.2；A2≈0 | CPU/GPU 1–2 h |
| P2.2 seed 1 | A0/A1 端到端 | A1 < A0 且方向稳定 | GPU 2×4–8 h |
| P2.3 seeds 2–3 | A0/A1 | ≥4/5 | GPU 4×4–8 h |
| P2.4 placebo/机制 | A2/A3/A4 | A2 无效；A1 优于最好 A3 ≥0.002 | GPU 4×4–8 h |
| P2.5 二级协议 | A1 vs A0，P4-full/P4-1k | 符号一致；LN/RC/star 分层 | GPU 10–20×2–6 h |

**决策门**：

* H2 支持 → 判定比例作为默认辅助损失；继续 5 seeds + P4-full；
* H2 不支持但操纵检查通过 → 报告有功效的零结果；第 3 条仍可做；
* 操纵检查失败 → 先修模型/数据，不报零结果；
* A2 与 A1 同幅度改善 → 结论改成“多任务/容量有效”。

### 5.4 Phase 3：第 3 条（第 4–6 周）

| 任务 | 内容 | 判据 | 算力 |
|---|---|---|---|
| P3.1 教师 | mf_dot dim 8/16，5 seeds；K=64；签名 QA | held-out NLL < bias；跨 seed 相关 >0.7 | CPU 1–2 h |
| P3.2 冻结探针 | B0/B1/B2/B3，5 seeds | s 头与 val 签名相关 >0.3；B3≈0 | CPU/GPU 5×0.5–1 h |
| P3.3 seed 1 | B0/B1/B2 端到端 | B2 < B1 且方向稳定 | GPU 3×3–6 h |
| P3.4 seeds 2–3 | B0/B1/B2 | ≥4/5 | GPU 6×3–6 h |
| P3.5 B3/B4 | placebo；K=16/128 | B3 无效；K 敏感性 | GPU 6×3–6 h |

**决策门**：

* H3 支持 → 报告“保留玩家分歧有助于全局 rank-NLL”；
* B2 ≈ B1 → 报告“平均难度足够，分歧对全局序不重要”；
* s 头拟合失败 → 先修教师/头，不报零结果；
* 教师 held-out NLL 不过线 → 停止第 3 条主结论。

### 5.5 Phase 4：外部复核与写作（第 6–7 周）

| 任务 | 内容 | 算力 |
|---|---|---:|
| P4.1 P4-full | 每条线胜者 vs 主基线，5 seeds | GPU 10–20 h |
| P4.2 P4-1k | 与已有 P2/P4 数字对齐 | GPU 5–10 h |
| P4.3 VSRG benchmark | 胜者导出内容/难度，isotonic 5 折 OOF MAE | CPU/GPU 2–4 h |
| P4.4 分层/稳健性 | LN/RC、star、活跃度、hold 时间、最佳一局 | CPU 2–4 h |
| P4.5 写作 | 结果表、失败分析、决策备忘录、复现命令 | — |

### 5.6 时间线（7 周）

| 周 | Phase 0 | 第 1 条 | 第 2 条 | 第 3 条 | Phase 4 |
|---|---|---|---|---|---|
| 1 | P0.1–P0.4；P0.5 smoke | P1.1 pilot | — | — | — |
| 2 | P0.5 throughput；P0.6 | P1.2 seed 1 | P2.1 数据/事件头 | P3.1 教师 | — |
| 3 | — | P1.3 seeds 2–3；P1.4 | P2.1 冻结探针 | P3.1 签名 QA | — |
| 4 | — | P1.5 P4-full | P2.2 seed 1 | P3.2 冻结探针 | — |
| 5 | — | 第 1 条结论 | P2.2 seeds 2–3；P2.3 | P3.3 seed 1 | — |
| 6 | — | — | P2.4 结论；P4-full | P3.3 seeds 2–3；P3.4 | P4.1–P4.4 |
| 7 | — | — | — | — | P4.5 写作 |

这是**并行版**：第 2/3 条的冻结探针与第 1 条的 GPU 训练可以同时跑（CPU vs GPU）。
单人串行按 P0 → P1 → P2 → P3，时间线拉长到 9–10 周；第 3 条的冻结探针仍可
提前到第 1 条训练期间做。

### 5.7 算力预算

| Phase | GPU-hours（低/高） | CPU-hours | 存储增量 |
|---|---:|---:|---:|
| P0 | 1 / 2 | 5–10 | 5 GB |
| P1 | 25 / 70 | 10–20 | 30 GB |
| P2 | 35 / 110 | 15–30 | 60 GB |
| P3 | 25 / 80 | 10–20 | 40 GB |
| P4 | 8 / 25 | 5–10 | 20 GB |
| **合计** | **94 / 287** | **45–90** | **~155 GB** |

* 低值 = 3 seeds、只跑主对照、跳过 C1/C2/B4；
* 高值 = 5 seeds、全 placebo/机制臂、P4-full + P4-1k；
* 不含排队等待；不含重跑/失败；
* 6 h 队列上限：每个 run 必须 checkpoint/resume，或拆成 2 个 job；
* 如果配额 <100 GPU-hours，优先顺序：P1 主三臂 > P2a 冻结探针 >
  P3a 冻结探针 > P2b > P3b > P4-full。

### 5.8 角色分工

| 角色 | 负责 | 关键交付 |
|---|---|---|
| 数据/基础设施 | P0.1–P0.5、P2.1 缓存、split/QA | 侧表、H4、训练器骨架、throughput |
| 建模 | P1.1–P1.4、P2.2、P3.3 | 三臂训练、事件头、签名蒸馏 |
| 评测 | P0.6、P1.5、P2.3、P3.2、P4 | 统一评测、配对 Δ、CI、分层 |
| 写作/决策 | P4.5、每周 gate | 结论表、零结果功效、决策备忘录 |

单人时四个角色顺序执行；至少把“评测”与“建模”分开一次（自己写完的模型，
隔一天再用独立脚本评一遍），避免评测脚本与训练脚本共享 bug。

### 5.9 风险登记与停止条件

| 编号 | 风险/条件 | 触发 | 动作 |
|---|---|---|---|
| K1 | 判定 QA 不过 | 覆盖率 <99.9% 或对象数不匹配 >0.1% | 停止第 2 条，先修数据 |
| K2 | 第 1 条冻结 ≥ 解冻 | 3 seeds 上 Δ ≥ 0 或 >−0.001 | 停止第 1 条扩展；第 2/3 条改用冻结 backbone |
| K3 | 随机 ≥ 解冻 | 3 seeds 上 Δ ≥ 0 | 保留解冻 backbone，但不声称“预训练有用” |
| K4 | 第 2 条操纵检查失败 | 事件头 val NLL 未超过 A4 阈值 | 停止第 2 条主跑；先修模型/数据 |
| K5 | 第 2 条 placebo ≈ A1 | A2 与 A1 同幅度且 CI 排除 0 | 停止 H2 声明；改写为“多任务/容量有效” |
| K6 | 第 3 条教师不过线 | held-out NLL ≥ bias，或签名跨 seed 相关 <0.5 | 停止第 3 条；签名不可信 |
| K7 | 第 3 条 B2 ≈ B1 | Δ 在 ±0.001 内 | 停止第 3 条扩展；报告“平均难度足够” |
| K8 | GPU 配额不足 | <100 GPU-hours | 按 §5.7 优先级砍；保 P1 + 两个冻结探针 |
| K9 | H4 高星偏置被质疑 | review 提出 | 补 P4-full；或 Gate 1b 的 H4-full |
| K10 | 6 h 队列杀 job | 无 checkpoint | 每 500 step 存；resume |
| K11 | 评测脚本与训练脚本共享 bug | 同一人写两边 | 独立评测脚本 + 手算 3 个 pair 的 NLL 对齐 |

**停止不是失败**：如果 K2/K4/K6 触发，写一份“为什么停、已经排除了什么”的
备忘录，比硬跑 5 seeds 更有价值。

---

## 6. 代码与产物

### 6.1 新增/修改文件

| 文件 | 状态 | 作用 | 主要输入 → 输出 |
|---|---|---|---|
| `scripts/step0_judgements.py` | 新 | dump → 六档侧表 | dump → `step0_judge_{tag}.parquet` |
| `scripts/step0_union_judge.py` | 新 | 多源侧表合并 | judge 表 + union plays → `step0_judge_full.parquet` |
| `scripts/step1_attach_judgements.py` | 新 | pairbest 接最佳 play counts | pairbest + judge + meta → `step1_full_pairbest_judge.parquet` |
| `scripts/step1_holdout_split.py` | 新 | H4 划分 | pairbest + holdout ids → `step1_full_holdout_s{k}.parquet` |
| `scripts/step2_chart_encoder.py` | 新 | 第 1/2/3 条的统一训练器 | H4/P4 split → `d_*.parquet / checkpoint / log` |
| `scripts/step2_teacher_signature.py` | 新 | mf_dot → K 维签名 | train table → `teacher_sig.parquet` |
| `scripts/rank_paired_delta.py` | 新 | 同 test 对上的配对 ΔNLL + CI | 两个 d 表 + split → `paired.json` |
| `scripts/summarize_behavior.py` | 新 | 跨 seed/arm 汇总 | run 目录 → `summary.csv/md` |
| `scripts/step2_rank_eval.py` | 改 | 加 `--per-pair-out` / chart-cluster bootstrap | 现有 + 新选项 |
| `scripts/step1_p4_split.py` | 改/复用 | P4-full 已可跑；加 `--judge` 可选 | 现有 |
| `dcr/steps.py` | 改 | 注册新 step 版本 | 现有 |
| `dcr/rank_metrics.py` | 改 | 加 chart-cluster bootstrap 辅助 | 现有 |
| `configs/exp_e1_*.yaml` | 新 | 第 1 条配置 | — |
| `configs/exp_e2_*.yaml` | 新 | 第 2 条配置 | — |
| `configs/exp_e3_*.yaml` | 新 | 第 3 条配置 | — |
| `scripts/slurm/behavior_*.slurm` | 新 | HPC 提交 | — |

#### 6.1.1 `step2_chart_encoder.py` 的 CLI 契约

~~~text
--task {rank,judge,teacher}           # 第 1/2/3 条
--init {d2v,random}
--ckpt PATH                           # init=d2v 时必填
--freeze {all,topk,none}
--unfreeze-blocks K                   # 默认 4
--split-table PATH                    # step1_holdout_s{k}.parquet
--split-col split                     # train/val/test/dropped
--osu-dir PATH
--window-sec 16 --frame-hz 100
--windows-per-chart 4
--players-per-batch 4 --charts-per-player 6
--pair-views 2000000 --row-views 0
--epochs 30 --patience 5
--lr-encoder 1e-4 --lr-head 1e-3 --wd-encoder 0.01 --wd-head 0.05
--bf16 --grad-clip 1.0
--judge-weight 0.0                    # task=judge
--judge-class-weights sqrt_inv        # none|sqrt_inv|inv
--teacher-signature PATH              # task=teacher
--teacher-weight 0.0
--aux {none,judge,judge_shuffled,single_grade,mean,full,shuffled}
--seed 0
--out-dir PATH
--resume PATH
--benchmark-views N                   # smoke/throughput
~~~

输出：

~~~text
<out-dir>/
  config.yaml          # 实际参数 + git rev + ckpt md5
  train_log.jsonl      # step, loss, val_nll, lr, grad_norm, time
  checkpoint.pt        # 最后一版 + best val
  d_train.parquet      # beatmap_id, split, d, arm, seed
  d_val.parquet
  d_test.parquet
  metrics.json         # 本 run 的 nll/ceiling/captured/pearson/rho
  events_val.parquet   # task=judge 时的预测比例（诊断）
~~~

#### 6.1.2 侧表 schema

`step0_judge_{tag}.parquet`：

| 列 | 类型 | 说明 |
|---|---|---|
| player_id | int64 | osu user id |
| beatmap_id | int64 | 4K mania 图 |
| timestamp | datetime64[ns] | play 时间（秒精度） |
| n_max | int32 | MAX/320 |
| n300 | int32 | 300 |
| n200 | int32 | 200 |
| n100 | int32 | 100 |
| n50 | int32 | 50 |
| n_miss | int32 | miss |
| n_total | int32 | 六列和 |
| source | int8 | 1k/10k/randommerge |
| score_id | int64 | legacy/modern 的 score id |

`step1_full_pairbest_judge.parquet` = step1 的 9 列 + 六档 + `n_total` +
`judge_ok`（bool）。join key = (player_id, beatmap_id, timestamp)。

### 6.2 配置示例

第 1 条（H4，解冻 top-4）：

~~~yaml
name: e1_unfrozen_k4
tag: full
steps:
  step1:
    version: holdout_split
    params:
      holdout_ids: data2vec_main/splits/holdout/eval_ids.txt
      seeds: [1, 2, 3, 4, 5]
      player_frac: 0.6
      bins: 5
      val_chart_frac: 0.1
  step2:
    version: chart_encoder
    params:
      task: rank
      init: d2v
      ckpt: data2vec_main/runs/main_holdout/checkpoints/last.pt
      freeze: topk
      unfreeze_blocks: 4
      split_table: data/processed/step1_full_holdout_s1.parquet
      osu_dir: data/osu_cache
      windows_per_chart: 4
      players_per_batch: 4
      charts_per_player: 6
      pair_views: 2000000
      epochs: 30
      lr_encoder: 0.0001
      lr_head: 0.001
      patience: 5
      seed: 0
      out_dir: results/behavior/e1/unfrozen_k4/s1
~~~

第 2 条（+ judge）：

~~~yaml
name: e2_judge
tag: full
steps:
  step2:
    version: chart_encoder
    params:
      task: judge
      init: d2v
      ckpt: data2vec_main/runs/main_holdout/checkpoints/last.pt
      freeze: topk
      unfreeze_blocks: 4
      split_table: data/processed/step1_full_holdout_judge_s1.parquet
      osu_dir: data/osu_cache
      judge_weight: 1.0
      judge_class_weights: sqrt_inv
      aux: judge
      pair_views: 2000000
      row_views: 500000
      epochs: 30
      patience: 5
      seed: 0
      out_dir: results/behavior/e2/judge/s1
~~~

第 3 条（+ 签名）：

~~~yaml
name: e3_signature
tag: full
steps:
  step2:
    version: chart_encoder
    params:
      task: teacher
      init: d2v
      ckpt: data2vec_main/runs/main_holdout/checkpoints/last.pt
      freeze: topk
      unfreeze_blocks: 4
      split_table: data/processed/step1_full_holdout_s1.parquet
      osu_dir: data/osu_cache
      teacher_signature: results/behavior/e3/teacher_sig_s1.parquet
      teacher_weight: 1.0
      aux: full
      pair_views: 2000000
      epochs: 30
      patience: 5
      seed: 0
      out_dir: results/behavior/e3/full/s1
~~~

**注意**：以上 YAML 的 `steps` 字段是目标接口；实现时要在
`dcr/steps.py` 注册 `step1/holdout_split`、`step2/chart_encoder` 等版本，
否则 `main.py` 不认识。也可以先用命令行直接跑，稳定后再注册。

### 6.3 Slurm 模板

~~~bash
#!/bin/bash
#SBATCH -p emergency_gpu
#SBATCH -J dcr-e1-unfrozen
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o slurm_e1_%j.out
#SBATCH -e slurm_e1_%j.err
set -euo pipefail
cd "$HOME/DCR"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
PY=$HOME/.conda/envs/DCR/bin/python

$PY scripts/step2_chart_encoder.py \
  --task rank --init d2v \
  --ckpt data2vec_main/runs/main_holdout/checkpoints/last.pt \
  --freeze topk --unfreeze-blocks 4 \
  --split-table data/processed/step1_full_holdout_s1.parquet \
  --osu-dir data/osu_cache \
  --pair-views 2000000 --epochs 30 --patience 5 \
  --out-dir results/behavior/e1/unfrozen_k4/s1
echo "[$(date +%H:%M:%S)] E1_DONE"
~~~

### 6.4 测试与 QA 清单

| 测试 | 内容 | 通过线 |
|---|---|---|
| `test_step0_judgements.py` | 合成 legacy/modern 行；dedup；counts 和 | 逐位等于手算 |
| `test_step1_attach_judgements.py` | 最佳 play 选择；join key；缺失 | 100% 命中；缺失列列出 |
| `test_step1_holdout_split.py` | chart/player 互斥；val 互斥；5 seeds | 断言全过 |
| `test_chart_encoder.py` | 20k views smoke；梯度；冻结 hash；短图 | 全过；显存记录 |
| `test_judge_proportions.py` | 事件数=N；sum(p)=1；损失梯度；打乱 | 全过 |
| `test_teacher_signature.py` | 参考玩家固定；签名 z-score；重排 | 全过 |
| `test_rank_paired_delta.py` | 手算 3 个 pair 的 NLL；配对差 | 与手算差 <1e-9 |
| 端到端 | H4 s1 冻结 vs 随机（小预算） | 解冻/随机梯度非零；冻结 hash 不变 |

### 6.5 结果表模板

`results/behavior/summary/<exp>_<protocol>.csv`：

| 列 | 说明 |
|---|---|
| exp | e1/e2/e3 |
| protocol | H4/P4-full/P4-1k |
| arm | frozen/top4/random/rank/judge/... |
| split_seed | 1–5 |
| nll | test strict-pair NLL |
| ceiling_nll | 同 test 对/ceiling |
| captured | (1 − nll/log2) / (1 − ceiling/log2) |
| n_scored | 参与比较的 pair 数 |
| coverage | ceiling 覆盖比例 |
| pearson_bhat / rho_bhat | per-chart d 与 b 的相关 |
| git_rev / ckpt_md5 | 版本 |

配对表 `results/behavior/summary/<exp>_paired.csv`：

| 列 | 说明 |
|---|---|
| arm_a / arm_b | 对比 |
| delta_mean / delta_std | 5 seeds |
| n_seeds_negative | A 更好的种子数 |
| ci_player_lo / hi | player-cluster bootstrap |
| ci_chart_lo / hi | chart-cluster bootstrap |
| verdict | supported/not/inconclusive |

---

## 7. 会说什么、不会说什么

### 7.1 可以说的

在**指定协议**（H4 / P4-full / P4-1k）和**指定对照**下，可以报告：

* 解冻编码器 vs 冻结 D2V 的配对 ΔNLL，以及 5 seeds 的符号一致性；
* 解冻编码器 vs 随机初始化的配对 ΔNLL；
* rank-NLL + 六档比例 vs rank-NLL only 的配对 ΔNLL，以及 placebo（打乱比例）
  是否同样改善；
* 整组参考玩家签名 vs 平均难度标量的配对 ΔNLL，以及 placebo（打乱签名）；
* 这些差异在 LN/RC、star 分箱、玩家活跃度上的分布（标注样本量）；
* 教师/事件头的操纵检查结果（它们是否真的学到了辅助信号）；
* 零结果及其功效（最小可检测效应）。

### 7.2 不可以说的

| 不能说 | 为什么 |
|---|---|
| “判定分布让模型学会了逐音难点” | 整局 counts 不能唯一确定事件级难度；没有 replay 定位验证 |
| “学生学会了 jack/stream/LN 这些技能维度” | 只验证了参考玩家分歧的可迁移性，没有命名技能的可解释性证据 |
| “行为监督有因果关系” | 这是预测实验；容量/初始化/数据量都可能解释差异，C1/C2 只能部分拆解 |
| “打赢了 Sunny / MinaCalc” | 主指标是 rank-NLL，不是 Dan MAE；外部基准只在 VSRG 上单独报告 |
| “H4 的绝对 NLL 比 P4 低/高” | test 集、ceiling、pair 集不同，绝对值不可比 |
| “模型可以部署了” | 部署需要校准、外部基准、稳定性/延迟/覆盖率验证；本文只到研究模型 |
| “低星图也变好” | H4 是高星偏置；除非 P4-full/H4-full 复现 |
| “遮蔽预训练没用” | 随机臂打平只说明“在这个监督/预算下初始化不是必需”，不等于 D2V 作为冻结特征没用 |
| “六档比例一定比 ACC 好” | 这是待检验假设；placebo/单档位/操纵检查都可能否定它 |

### 7.3 报告规则

1. 每个主表必须带：协议、ceiling、n_scored、coverage、5 seeds 的 mean±std、
   配对的种子符号数、两个 CI；
2. 每个辅助任务必须带 placebo 和操纵检查；没有这两个，辅助任务的结果不解释；
3. 零结果必须带功效；不能只写“不显著”；
4. 分层/探索性结果标注 exploratory，不与主假设混在一起；
5. 所有 run 的 config/git rev/ckpt md5 可追溯；不能只给一个汇总数字。

---

## 附录 A：已核查事实与复现命令

本附录记录写计划时在仓库里实际核查过的数字；不是从文档抄的。

### A.1 full 并集

~~~bash
python3 - <<'PY'
import pandas as pd
df = pd.read_parquet('data/processed/step1_full_pairbest.parquet')
print(len(df), df.beatmap_id.nunique(), df.player_id_base.nunique(), df.player_id.nunique())
print(df.groupby('split').size().to_dict())
print(df.groupby('beatmap_id').size().quantile([.1,.5,.9]).to_dict())
PY
~~~

结果：1,898,076 行；20,945 图；20,681 基础玩家；22,535 段；
train/test = 1,329,090 / 568,986；每图对数 p10/p50/p90 = 2/19/205；
≥50 对的图 6,529；玩家内两两比较 255,196,653。

### A.2 H4

用 `step1_full_pairbest.parquet` + `data2vec_main/splits/holdout/eval_ids.txt`，
按玩家活跃度 5 层 60/40、seed 1：

| block | 行 | 图 | 段 |
|---|---:|---:|---:|
| train | 927,896 | 18,714 | 13,395 |
| test | 138,685 | 1,363 | 6,384 |
| dropped (test player × train chart) | 626,137 | — | — |
| dropped (train player × test chart) | 205,358 | — | — |

test 每图对数 p10/p50/p90 = 10/48/233；≥20 对 1,059 图；
5 seeds 的 test 行 136,608–139,177。
玩家内比较：train 100,287,454；test 3,679,122。

> 注：train 数字未扣 10% val charts；实际训练行/比较数约少 10%。

### A.3 P4-full

~~~bash
python3 scripts/step1_p4_split.py \
  --in data/processed/step1_full_pairbest.parquet \
  --out-prefix /tmp/full_p4_test --seeds 1 --force
~~~

结果：players 12,410/8,271；charts 12,565/8,380；
rows dropped/train/test = 913,237 / 677,661 / 307,178。

### A.4 判定字段

1k legacy 表扫描（`osu_scores_mania_high.sql`）：

* 总行 3,463,843；有 `beatmap_meta_full` 的 1,919,000 行；
* `sum(counts) == count_total` 的 mismatch = **0**；
* chart 193128 的五条 play：counts 和都等于 708（= `count_total`）；
* 1k modern JSON 的 keys：`perfect/great/good/ok/meh/miss`；同一 play
  的 counts 与 legacy 一致（示例 user 315646：274 great + 396 perfect + 36 good
  + 1 ok + 1 miss = 708）。

1k legacy 前 60 万行中 ACC ∈ (0.9, 0.998]、N≥100 的 408,033 局：

| 档位 | 平均占比 | 同 ACC 组内 std |
|---|---:|---:|
| MAX | 0.666 | 0.065 |
| 300 | 0.290 | — |
| 200 | 0.0052 | 0.0057 |
| 100 | 0.0044 | 0.0029 |
| 50 | 0.0014 | 0.0019 |
| miss | 0.0331 | 0.0042 |

**注意**：这是按 score_id 顺序的前 60 万行，不是随机样本；它只用于说明
“信号存在”，不能代替 Gate 0 的 pairbest 精确统计。

### A.5 .osu 与 meta

* `data/osu_cache` 21,950 个文件；
* 随机 300 张，md5 与 `beatmap_meta_full.checksum` 全部匹配（300/300）；
* chart 193127/193128/193129/217976 的对象数 = circle + hold =
  `count_total`（422/708/217/234）；taps + 2×holds 不等于 `count_total`，
  所以 hold 只算一个判定。

### A.6 内容与基线覆盖

* `chart_content_d2v_holdout.parquet` 21,949 图 × 512d，L2 norm=1.0；
  与 full 图交集 20,832；
* `rank_signals_1k.parquet` 覆盖 20,832 图；H4 的 1,365 张 test 图 100% 覆盖；
* H4 test 的 star 均值 4.711 / 中位 4.714；full pairbest 图 3.391 / 3.231；
  全部 full 图 3.312 / 3.147；
* P4-1k 已有 5-seed 数字：ceiling 0.56540 ± 0.00859，
  CM3P+ext MLP 0.57343 ± 0.00433，Sunny 0.63111 ± 0.00777，
  Daniel 0.63102 ± 0.00601，star 0.66021 ± 0.00779。

### A.7 D2V checkpoint

* 本地：`data2vec_main/runs/smoke/checkpoints/last.pt`（smoke 2L/64d，不能用）；
* HPC：`data2vec_main/runs/main_holdout/checkpoints/last.pt`
  （[benchmark_d2v.md](../benchmark_d2v.md) §2.1）；
* 预训练：20,584 图、25,720 step、47m02s、A800、loss 0.524→0.054；
* 内容表：`data/processed/chart_content_d2v_holdout.parquet`；
* 如果 HPC checkpoint 不可用，重训命令见 `data2vec_main/README.md` §4。

### A.8 现有 D2V 数字的口径提醒

D2V README §6.5/§6.6 的 P4 数字（D2V 线性 0.59301、+ext 0.58561）与
P4-1k 的 CM3P 数字**只在该 runner/该内容表下可比**；它们不能与 H4 的绝对 NLL
比，也不能直接当作本计划的基线。本计划的冻结 D2V 基线必须在 H4 上用同一个
head/同一个 split 重跑。

---

## 附录 B：数学细节

### B.1 BT NLL 与玩家能力的消去

设玩家 $u$ 在图 $m$ 上的表现（越高越好）为

$$
s_{um} = \theta_u - d_m + \epsilon_{um}
$$

其中 $\theta_u$ 是玩家能力，$d_m$ 是图难度，$\epsilon$ 是噪声。对同一玩家的两张图
$i,j$：

$$
s_{ui}-s_{uj} = ( \theta_u-d_i )-( \theta_u-d_j ) + \epsilon_{ui}-\epsilon_{uj}
= d_j-d_i+\epsilon_{ui}-\epsilon_{uj}
$$

所以 $P(s_{ui}>s_{uj})=\sigma(d_j-d_i)$ 与 $\theta_u$ 无关。把每个玩家的所有
无序 pair 聚合成 counts $n_{ij},n_{ji}$，得到

$$
L(d)=-\sum_{i<j}\left[n_{ij}\log\sigma(d_j-d_i)+n_{ji}\log\sigma(d_i-d_j)\right]
$$

这正是 [dcr/rank_metrics.py](../../dcr/rank_metrics.py) 的 `nll_counts`。
**这不是一个 per-player 回归**：$d$ 只以差值出现，绝对水平和玩家项都被消掉。
因此 $d$ 的全局平移不变；评测重新拟合 link，所以全局缩放也不影响 NLL；
不能从 $d$ 读“绝对难度”；如果模型给每张图一个自由参数 $b_m$，那就是 ceiling。

### B.2 ties

若 $s_{ui}=s_{uj}$，目标取 0.5：

$$
\ell = -\left[0.5\log q+0.5\log(1-q)\right]
$$

这与 `build_pair_table(tie="half")` 和 `step2_rank_eval.py` 的 ties 口径一致。
训练时若用 `ties="skip"`，会丢掉 ties（在离散 ACC 上不少），导致训练目标与
评测目标不一致；本计划统一用 half。

### B.3 标签比例损失

设图 $m$ 有 $N_m$ 个判定事件，事件 $i$ 的六类概率 $p_{i,k}$，
观测 counts $c_{m,k}$，$\sum_k c_{m,k}=N_m$。定义

$$
\hat{c}_{m,k}=\sum_{i=1}^{N_m}p_{i,k},\qquad
\hat{p}_{m,k}=\frac{\hat{c}_{m,k}}{\sum_j\hat{c}_{m,j}}
$$

主损失：

$$
L_{prop}=-\sum_k w_k\,\frac{c_{m,k}}{N_m}\log(\hat{p}_{m,k}+\epsilon)
$$

它是 bag 级比例匹配，不是逐事件交叉熵；事件标签不可见，所以同一个 bag 比例
可以由很多不同的事件级分配产生。这正是“不能定位逐音难点”的数学原因。
梯度只通过 $\hat{c}_{m,k}$ 进入，每个事件拿到的梯度取决于“它所在 bag 的预测
比例与观测比例的差”，而不是它自己的真实标签。

**class weights**：$w_k\propto 1/\sqrt{\mathrm{freq}_k}$，从 train 的边际频率算。
不加权会被 MAX/300 淹没；$1/\mathrm{freq}$ 又会放大稀有档位噪声；$1/\sqrt{}$ 是折中，
pilot 比较 none/sqrt_inv/inv。

**Poisson-binomial 备选**：若事件近似独立，$\hat{c}_{m,k}$ 的方差是
$\sum_i p_{i,k}(1-p_{i,k})$；高斯损失
$L_{pb}=\sum_k (c_{m,k}-\hat{c}_{m,k})^2/(\hat{\sigma}_{m,k}^2+\epsilon)$
做消融。

### B.4 教师签名与旋转不变性

mf_dot 的交互项是双线性型 $\langle P[u],C[m]\rangle=P[u]^\top C[m]$。
若 $Q$ 是正交矩阵，令 $P'=PQ$、$C'=CQ$，则对每个玩家–谱面对

$$
\langle P'[u],C'[m]\rangle
= P[u]^\top Q\,Q^\top C[m]
= P[u]^\top C[m]
$$

（因为 $Q^\top Q=I$）。所以潜向量本身不可识别；任何旋转都给同一组预测。固定参考玩家集合 $R$ 后，

$$
s_m[r]=\frac{\hat{\ell}_{rm}-\frac{1}{K}\sum_{r'}\hat{\ell}_{r'm}}{\mathrm{sd}_r(\hat{\ell}_{\cdot m})+\epsilon}
$$

图内中心化去掉了 $b_m[m]$ 和 $\langle \bar{P},C[m]\rangle$，只留下
$\langle P[r]-\bar{P},C[m]\rangle$ 这一交互项；它在正交旋转下不变。所以参考
玩家固定后签名可复现；换一组参考玩家会换坐标，但分歧结构在正交变换意义下不变。

### B.5 功效与 CI

H4 test 有 1,363 张图、6,118 个玩家、138,685 行、3.68M 比较。
若每张图的 ΔNLL 标准差约 0.05，$\mathrm{SE}\approx 0.05/\sqrt{1363}\approx 0.0014$；
0.003 的效应约 2.1 SE。5 个 player split seeds 的 between-seed std 可能更大，
所以判据要求“5/5 符号一致 + CI 排除 0”，而不是只看单一 seed 的 p 值。

player-cluster bootstrap 重采样玩家，保留每个玩家的所有 pair；chart-cluster
bootstrap 重采样图，pair 的权重取两张图重采样次数的乘积（two-way cluster 的
近似）。两个 CI 都报；结论冲突时写清楚，不挑一个。

---

## 附录 C：预注册清单

在跑主实验之前，把下面每一项填好、签日期、写进 config；主实验开始后不再改。

### C.1 假设

| 编号 | 假设 | 方向 | 主对照 |
|---|---|---|---|
| H1a | 解冻上层 > 冻结 D2V | 解冻 − 冻结 < 0 | E1 top4 vs frozen |
| H1b | 解冻上层 > 随机初始化 | 解冻 − 随机 < 0 | E1 top4 vs random |
| H2 | rank-NLL + 六档比例 > rank-NLL only | A1 − A0 < 0 | E2 judge vs rank |
| H2-mech | 判定分布形状 > 单档位 | A1 − best(A3) < 0 | E2 judge vs single-grade |
| H3 | 整组参考玩家 > 平均难度 | B2 − B1 < 0 | E3 full vs mean |

### C.2 主指标与判据

* 主指标：H4 test strict-pair rank-NLL；配对 ΔNLL；
* 支持：Δ ≤ −0.003，≥4/5 种子为负，两个 95% CI 排除 0；
  H1b 用 −0.005；H3 强判据用 −0.005 + 5/5；
* 不成立：Δ ≥ 0，或 placebo 与主臂同幅度改善；
* 不确定：方向对但 CI 含 0，或 3/5；
* 多重比较：主假设 5 个；机制/分层标 exploratory，不做 Bonferroni 修正但
  不把它们当 confirmatory。

### C.3 数据与排除

| 项 | 规则 |
|---|---|
| 标签 | ACC ∈ (0.9, 0.998]，n_plays ≥ 2，step1 默认清洗 |
| 内容 | 只保留有 D2V 内容的图（20,832） |
| 判定 | 只保留 sum(counts)=count_total=.osu 对象数的对（Gate 0） |
| test | H4 的 1,363 张图；test 玩家 40% |
| val | train charts 的 10%，chart-disjoint |
| 排除 | 坏图、短图、counts 不匹配、无内容；排除数写 JSON |
| 种子 | 划分种子 1–5；模型种子 0（随机臂补 1） |

### C.4 操纵检查

| 检查 | 阈值 | 不过怎么办 |
|---|---|---|
| E2 事件头 vs A4 的 val NLL | 相对低 ≥5% 或绝对低 ≥0.02 nats/event，取保守 | 停止 E2 主跑，先修模型 |
| E2 预测 MAX 率 vs val 实际 | Spearman >0.2 | 同上 |
| E2 A2 打乱后的相关 | 接近 0 | 检查打乱实现 |
| E3 s 头 vs val 签名 | 平均相关 >0.3 | 停止 E3 主跑 |
| E3 教师 held-out NLL | < bias 基线 | 签名不可信，停止 |
| E3 5-seed 签名相关 | >0.7 | 换集成/正则 |
| E1 梯度/冻结 | 解冻/随机非零；冻结 hash 不变 | 修训练器 |

### C.5 停止规则

* K1 判定 QA 失败 → 停 E2；
* K2 3 seeds 上冻结 ≥ 解冻 → 停 E1 扩展；
* K4 E2 操纵检查失败 → 停 E2 主跑；
* K5 E2 placebo ≈ 主臂 → 停 H2 声明；
* K6 E3 教师失败 → 停 E3；
* K7 E3 B2 ≈ B1 → 停 E3 扩展；
* 任何一条线停，写“为什么停、排除了什么、还剩什么”。

### C.6 分析计划

1. 先跑 Gate 0/1，填基线与 ceiling；
2. 每条线先 1 seed 看方向，再 3 seeds，再 5 seeds；
3. 主表只放预注册对照；placebo/机制/分层放附录；
4. 每个 Δ 都报 5-seed mean/std、种子符号数、两个 CI；
5. 零结果报功效；操纵检查失败不报零结果；
6. 所有 run 的 config/git rev/ckpt md5 入 manifest。

### C.7 签名

| 项 | 填写 |
|---|---|
| 计划版本 | 2026-11-24 |
| 主实验开始日期 | __________ |
| 负责人 | __________ |
| 预注册冻结日期 | __________ |
| 变更记录 | __________ |

---

## 附录 D：一页纸执行清单（打印版）

**第 1 周**

- [ ] P0.1 判定侧表程序 + 测试
- [ ] P0.2 1k/10k/randommerge 侧表 + union + pairbest join
- [ ] P0.3 H4 5 seeds + 互斥校验
- [ ] P0.4 D2V checkpoint 确认 + 10 图对齐
- [ ] P0.5 训练器 smoke + throughput
- [ ] P0.6 H4 基线（Sunny/MinaCalc/star/冻结 D2V/CM3P+ext）

**第 2–3 周**

- [ ] P1.1 三臂 pilot（500k views）
- [ ] P1.2 三臂 seed 1（2M views）
- [ ] P1.3 seeds 2–3；必要时 4–5
- [ ] P1.4 C1/C2（仅在方向不明确时）
- [ ] 第 1 条决策门：解冻/冻结/随机谁赢

**第 3–5 周**

- [ ] P2.1 冻结探针 + 操纵检查 + λ 选择
- [ ] P2.2 A0/A1 seed 1 → seeds 2–3 → A2/A3/A4
- [ ] P2.3 H4/P4-full 复核
- [ ] 第 2 条决策门：H2 是否成立；placebo 是否无效

**第 4–6 周**

- [ ] P3.1 教师 + 参考玩家 + 签名 QA
- [ ] P3.2 冻结探针 B0–B3
- [ ] P3.3 B0–B3 seed 1 → seeds 2–3 → B4
- [ ] 第 3 条决策门：B2 < B1 是否成立

**第 6–7 周**

- [ ] P4.1 P4-full 复核
- [ ] P4.2 P4-1k 对齐
- [ ] P4.3 VSRG benchmark
- [ ] P4.4 分层/稳健性
- [ ] P4.5 写作 + 决策备忘录

---

## 附录 E：术语

| 术语 | 含义 |
|---|---|
| rank-NLL | 同一玩家两张图的成对 Bradley–Terry NLL；越低越好 |
| ceiling | 每图自由 $b_m$ 的 cross-fitted BT NLL；参照，不是硬上界 |
| strict-pairs | signal NLL 只在 ceiling 覆盖的 pair 上算 |
| H4 | 固定 D2V-unseen 1,363 张 test 图 + 40% test 玩家的双冷启动划分 |
| P4-full | full 20,832 图内随机 60/40 chart + 60/40 player |
| P4-1k | 1k 6,828 图内的 P4，与已有数字对齐 |
| top-4 | 最后 4 个 Transformer block + final norm |
| 六档 | MAX/300/200/100/50/miss |
| 判定比例 | 一局六档 counts / N |
| 参考玩家 | 教师签名用的 K 个固定 train 玩家 |
| 签名 | 参考玩家相对表现的 K 维向量，图内中心化 |
| LFLP | learning from label proportions |
| placebo | 打乱辅助标签/签名，保留头/参数/预算的对照 |

---

*本文是计划，不是结果。所有 TBD 由 Phase 0–4 填入；任何结论以 run 目录和
配对表为准。*
