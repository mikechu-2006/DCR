# 三个后续实验：具体方法论与执行计划

> **状态：草案 v3（2026-11-23，按你的最新回复更新）。**
> 本文件不包含新的正式实验结果。
> 目的：把旗舰模型提出的三个方向转成可执行、可预注册、可复现的实验；先钉死评价口径、
> 数据构造、代码改动、判据与风险，再决定跑哪些格子。
>
> 本文件里的“预实验 / 只读核查”只用于验证口径和估算样本量，**不是结论**。
> 已确认：四个数据集分别跑；加相关性；VSRG=benchmark；6×6 训练侧；
> MLP+ext 与 linear_extra、P2/P4 都不分主次；全部 CPU 任务提交 HPC；
> **Q3 用 A1（全部 pair）作字面实现，A2/A3 只作敏感性；Q7 因网络/磁盘限制不补
> randommerge/full 音频，Experiment 2 只汇报 1k 结果。**
>
> **2026-11-24 执行更新：Experiment 1 已在 1k + randommerge 的 P4 5 seeds 上完成；
> A1 没有稳定增益，机制是 n_plays=2 的重复标签/加权；10k/full 未跑。
> 结果见 [docs/first2_aux_results.md](first2_aux_results.md)。**
>
> **2026-11-24 执行更新：Experiment 2 已完成（1k 6,819 图）。2a 与 2b 都通过：
> 整 beatmapset 留出后 audio 仍 5/5 为负，external 线性 −0.01091、external MLP
> −0.00581；跨 set 严格子集 −0.01138 / −0.00631。结论：audio 可迁移到新 set，
> 不是主要同 set 泄漏；randommerge/full 因数据条件未补音频。
> 结果见 [docs/audio_newset_results.md](audio_newset_results.md)。**
>
> **2026-11-24 执行更新：Experiment 3 已完成。P4-holdout 上 D2V 残差在 MLP 基线
> 5/5 为负、5/5 CI<0（overall −0.00891，near −0.00932，LN −0.00862）；线性基线
> 5/5 为负但 3/5 CI<0。P2 chart-disjoint 版本只有 1.07 万次比较，未确认（线性
> −0.0037，MLP +0.0004）。结论：directional support；先做小规模列感知探针，
> 不直接投入完整四列编码器。结果见 [docs/d2v_residual_results.md](d2v_residual_results.md)。**

---

## 执行摘要

1. **Experiment 1（first2 辅助训练）**：只改训练信号，不改 test 标签/测试对/ceiling；
   主指标仍是历史最佳 rank-NLL。**1k / 10k / randommerge / full 四个数据集分别跑**，
   每个都做 P2/P4 5 seeds；相关性与 ΔNLL 一起报，VSRG benchmark 作 second gate。
   已复现 17.2% 的标签反转，且反转率随练习机会上升。
2. **Experiment 2（整 set 留出）**：先用现成 audio/legacy 预测做 2a（只读），再做按
   `beatmapset_id` 分组的 P4（2b）。2a 预实验显示新 set 子集的 audio 增益没有系统性
   缩小（覆盖率 10.8–18.2%，5/5 seed 为负）；2b 是干净检验，模拟有 ~92 万次 test 比较。
   只有 2b 通过，才值得为 randommerge/full 补 ~1.4 万张图的音频。
3. **Experiment 3（D2V 残差）**：在 1,365 张编码器未见图上重训 legacy 强基线；**MLP+ext
   与 linear_extra 同等对待，P2-holdout 与 P4-holdout 同等对待**，不预设主次。用训练内
   5 折 OOF 基线分数拟合带正则的 D2V 线性修正；并排看整体、近分对、长押图，随机/打乱
   512d 作负对照。P2-holdout 54,721 次比较、P4-holdout 31,962 次；只能可靠检测
   ~0.005–0.01 的效应。
4. **执行方式**：所有 CPU 任务（包括标签构造、split 生成、训练、评测、汇总）全部提交
   HPC 并行；本地只做代码修改和结果阅读。只有 Experiment 2 通过后的音频编码需要
   GPU/抓音频工程。
5. **已确认**：Q1 保持历史最佳主指标 + 加相关性；Q2 四个数据集分别跑；Q4/Q5 VSRG =
   benchmark；Q6 训练侧只算 6×6；Q8 MLP+ext 与 linear_extra 都跑、P2/P4 不分主次。
   **仍需确认**：Q3（辅助损失范围）、Q7（audio 阈值）；其余按默认。

---

## 0. 共同口径与先决问题

### 0.1 主指标保持不变

三个实验都沿用仓库现有的主口径：

* 观测单位：step1/pairbest 的 `(player_seg, beatmap)` 一行；
* 评价：同一玩家段内的成对 Bradley–Terry NLL（`docs/metrics.md` 的 rank-NLL），
  `d_m` 越高越难；评测入口 `scripts/step2_rank_eval.py`；
* 模型比较：同一 test pair 集上的 **paired ΔNLL**，按 `player_id_base` 聚类做
  cluster bootstrap（`scripts/paired_delta_nll.py`）；
* **相关性系数（与 ΔNLL 并列报）**：Pearson(d,b)、Spearman(d,b)、逐玩家
  lw/macro Spearman；每个数据集 × 协议 × 方法都要有这张表；
* 划分：P2（新 chart，5 折 chart-disjoint）与 P4（新玩家 + 新 chart，6×6 / 4×4），
  各 5 个 split seed；模型 seed 固定 0；
* test 标签：历史最佳 `loss = log(1 - ACC_best)`，不因任何新实验改标签、改筛选、改 ceiling。

**绝对值不跨数据集、不跨 P2/P4、不跨 ceiling 比较；只比同一 pair 集上的 Δ。**

### 0.2 “评价指标变了”到底指什么

先把最容易混淆的一点钉死：

> **Experiment 1 不改变正式评价指标。** 正式 test 仍然是历史最佳 ACC 的 rank-NLL；
> 前两条记录的最佳 ACC 只作为**训练时的辅助标签**，不进入 test 标签、不进入 ceiling、
> 不参与“用 test 选模型”。模型的输出仍然是 per-chart 难度 `d_m`；评价仍然问
> “同一玩家在两张图里哪张表现更好”。

因此 Experiment 1 是**训练信号干预**，不是评价口径变更。对照组和实验组共享：

1. 同一批 train/test pair；
2. 同一 test 标签（历史最佳）；
3. 同一 ceiling 与 link 拟合方式；
4. 同一 bootstrap 聚类。

如果要做的是“把 early-best 也变成主评价目标”，那是另一个实验：需要重新定义 test
pair 的标签、重新拟合 ceiling，并说明项目到底关心“历史最佳”还是“早期可达到”。
**本计划默认不做这个 target change；这一点需要你确认（见 §6 Q1）。**

### 0.3 VSRG 的位置

VSRG Dan Benchmark 是**第二个、外部的检查**，不是 DCR 主指标的替代：

* DCR 主指标：historical-best rank-NLL，回答“ACC 序”；
* VSRG benchmark：5 折 chart-group OOF isotonic 的 Dan MAE，回答“外部 Dan 标签”；
* 两者 construct 不同，数字不可互比；只比同一 benchmark 协议下的 paired MAE。

Experiment 1 如果只在 DCR 上赢、VSRG 上明显变差，不能声称“提升了 VSRG 泛化”。
反过来，VSRG 变好而 DCR 不变，也不能替代 DCR 主结论。两个都报。

### 0.4 只读核查：已复现的事实

以下数字来自本机只读核查，用于确认口径和估算样本量；**不是正式实验结果**。

#### (a) randommerge P4 s1 的 first2 标签敏感性

按 step1 的规则（先玩家门槛 ≥20 条，再 365 天空窗拆分；见 §1.3）重取每对前两条记录的
最好成绩，在“至少有 10 张训练图的玩家段”内比较历史最佳序与前两条最佳序：

| 比较 | 数量 |
|---|---:|
| 两种取法都能分出高低的比较 | **9,415,648** |
| 顺序反转 | **1,623,767** |
| 反转率 | **17.245%** |

与旗舰模型报告的 941.6 万 / 162.4 万 / 17.2% 一致（差异来自舍入/过滤细节）。按两张图中
较小的 `n_plays` 分桶：

| min(n_plays) | 比较数 | 反转数 | 反转率 |
|---:|---:|---:|---:|
| 2 | 6,535,638 | 867,853 | 13.28% |
| 3 | 2,200,006 | 548,130 | 24.91% |
| 4–5 | 611,550 | 184,712 | 30.20% |
| 6–10 | 62,997 | 20,896 | 33.17% |
| 11–20 | 5,457 | 2,176 | 39.88% |
| >20 | 0 | 0 | — |

**读法**：标签敏感性随练习机会单调上升。两张图的 `n_plays` 都等于 2 时不会反转；
`min(n_plays)=2` 桶里另一张图可能 >2，所以仍有 13.3% 反转。这正是“历史最佳受
best-of-k 选择偏差影响”的直接证据。

#### (b) first2 标签的支撑与覆盖（randommerge P4 s1，train+test 全体 pair）

| 项 | 值 |
|---|---:|
| `n_plays == 2` 的 pair 占比 | 49.66% |
| `n_plays > 2` 的 pair 占比 | 50.34% |
| first2 最佳 ACC ≤ 0.90 | 5.54% |
| first2 最佳 ACC > 0.998 | 0% |
| 历史最佳 loss 与 first2 loss 的相关系数 | 0.865 |
| first2 loss − 历史最佳 loss 的均值 | +0.262（first2 更差） |

**含义**：约一半 pair 的辅助标签与主标签完全相同；另一半才携带“去掉后续练习”的新信息。
first2 有 5.5% 落到主标签支撑之外（ACC ≤ 0.90），需要决定保留、mask 还是截断（§6 Q3）。

#### (c) audio-on/off 在“新谱面集”测试对上的预实验

用现成的 `results/p4_multiseed_audio/{audio,legacy}/p4/s*/signals.parquet` 和
`step1_1k_pairbest_p4_nogap_s*.parquet`，只保留“两张测试图各自所属的 beatmapset_id
都未出现在 6×6 训练图”的 test 比较：

| seed | 保留比较 / 全部 test 比较 | 覆盖率 | ΔNLL（linear_extra） | ΔNLL（MLP_extra） |
|---:|---:|---:|---:|---:|
| 1 | 137,692 / 754,962 | 18.2% | −0.0168 | −0.0098 |
| 2 | 126,903 / 704,387 | 18.0% | −0.0100 | −0.0072 |
| 3 | 148,465 / 866,704 | 17.1% | −0.0237 | −0.0221 |
| 4 | 86,223 / 799,097 | 10.8% | −0.0107 | −0.0091 |
| 5 | 110,766 / 718,566 | 15.4% | −0.0265 | −0.0082 |

Δ = audio − legacy，负值 = audio 更好；完整 test 上这两个方法分别是 −0.0083…−0.0132
（linear_extra）和 −0.0045…−0.0110（MLP_extra）。**在这个条件子集上，audio 的增益没有
系统性缩小，部分 seed 反而更大。** 这是“同曲泄漏不是全部”的初步信号，但它仍是标准 P4 上的
条件子集；真正的检验是 §2.3 的整 set 留出 P4。

#### (d) 整 beatmapset 分组的 P4 可行性（模拟，不落盘）

用 1k 的 6,819 图 / 3,806 个 set，玩家侧沿用 P4 的 60/40 分层，谱面侧按 set 分组：

| 项 | 值 |
|---|---:|
| set 数 | 3,806 |
| 单图 set 数 | 2,456 |
| 最大 set 图数 | 35 |
| 模拟 s1：分配到 train / test 的图 | 4,045 / 2,774（59.3% / 40.7%） |
| 模拟 s1：实际有 train / test 行的图 | 3,509 / 2,058 |
| 模拟 s1：train / test 行 | 37,418 / 18,902 |
| 模拟 s1：test 比较数 | 924,384 |
| 其中跨 set 比较 | 913,061（98.8%） |
| test 图中 set 出现在 train 的数量 | 0 |

**含义**：整 set 留出后仍有 ~92 万次 test 比较，样本量足够。但“分配图数”与“实际有行的
图数”不是一回事：536 张 train-set 图没有 train 玩家玩过，716 张 test-set 图没有 test 玩家
玩过；它们留在 parquet 里但不进模型/评测。简单按 set 随机 60/40 时，实际有行的图是
3,509/2,058（63/37）。正式实现要在 popularity 分层内做图数平衡，并**同时报分配图数和
实际有行的图数**（§2.3.1）。

#### (e) D2V holdout 的样本量

| 口径 | test 行 | test 图 | test 比较 | `ln_ratio ≥ 0.15` 的图 | 至少一张 LN 的比较 |
|---|---:|---:|---:|---:|---:|
| P2-holdout（单划分） | 7,276 | 955 | 54,721 | 347 / 955（36.3%） | 62.9% |
| P4-holdout s1 | 3,556 | 392 | 31,962 | 131 / 392（33.4%） | 51.4%（both-LN 10.0%） |

P2 的比较数约为 P4 的 1.7 倍；近分对按 25% 估算分别约 1.37 万 / 0.8 万次。**这决定了
Experiment 3 只能可靠检测 ~0.005–0.01 nats 量级的近分对效应**，更小的效应需要更多图或
更多 seed（§3.6）。

#### (f) 为 randommerge / full 补音频的覆盖率

| 图集 | 图数 | 已有 audio 向量 | 需要新编码 | 覆盖率 |
|---|---:|---:|---:|---:|
| randommerge | 20,388 | 6,710 | 13,678 | 32.9% |
| full | 20,832 | 6,819 | 14,013 | 32.7% |

**数据边界（已确认）**：randommerge/full 补音频在现条件下不可行（网络/磁盘限制），
所以 Experiment 2 **只覆盖 1k 的 6,819 图**，不把“补 randommerge/full 音频”作为行动。
这张表只用于说明为什么不能把 1k 结论外推到 full。

### 0.5 四个数据集的 split 现状（Experiment 1 要分别跑）

| 数据集 | pair 行 | 图 | P2 5 seeds | P4 5 seeds | first2 的 step0 |
|---|---:|---:|---|---|---|
| 1k | 109,706 | 6,828 | 已有 | 已有 | `step0_1k.csv` |
| 10k | 1,513,746 | 16,061 | **需生成** | **需生成** | `step0_10k.parquet` |
| randommerge | 453,406 | 20,388 | 已有 | 已有 | `step0_randommerge.parquet` |
| full | 1,898,076 | 20,832 | **需生成** | **需生成** | `step0_full.parquet` |

* 1k 用 canonical 6,828（`step1_1k_pairbest.parquet`）；**不要**和 audio 的 6,819
  `_nogap` 表混用。两条线的结论分开写。
* 10k / full 的 P2/P4 split 用现有脚本生成：`step1_pairbest_resplit.py`（P2）、
  `step1_p4_split.py`（P4），seeds 1–5，与 1k/randommerge 同协议。
* 10k / full 的 P4 行数分别是 ~1.5M / ~1.9M；一次只跑一个 seed，避免单进程内存峰值。

---

## 1. Experiment 1：用“前两条记录的最好 ACC”做辅助训练

### 1.1 问题与假设

**问题**：在训练时额外拟合“每对前两条记录的最好成绩”这一辅助序，能否改善模型对
**历史最佳 ACC 序**的预测，并进一步改善 VSRG Dan 泛化？

**机制假设（H1：best-of-k 去偏）**：

* 对玩家 `p`、谱面 `m`，历史最佳是 `k_m` 条记录的最小 loss；
  `E[min loss]` 随 `k_m` 增大而下降；
* 如果 `k_m` 与谱面难度/热度相关，历史最佳序会被“练得多”的图系统性压低（显得更简单）；
* first2 把每条图的练习机会截到 2 条（或 `min(2,k)`），去掉了大部分 best-of-k 选择偏差；
  代价是标签更噪、且更接近“初见/视奏”难度。

**竞争假设（H0）**：

* first2 太噪，或它学到的“视奏难度”与项目关心的“可达到难度”不一致；
* 测试集 `n_plays` 分布与训练不同，辅助任务会把模型拉向错误的方向；
* 任何增益只是“多了一个辅助损失”的正则化，换成 random-2 或打乱标签也一样。

**可证伪点**：如果 A1（first2 辅助）相对 A0 的 paired ΔNLL 不为负，或 A4（random-2）/
A6（打乱 first2）与 A1 同样好，则 H1 不成立。

**判据**：按原计划，A1 是主实验；A2/A3 是敏感性；A4/A5/A6 是控制。最终看 A1 相对
A0 的 paired ΔNLL 与相关性是否提升；如果没有提升，就是否定的答案，不需要额外实验。

### 1.2 为什么不是“换标签评价”

模型的输出仍然是 per-chart `d_m`；test 仍然是历史最佳序。first2 只改变训练时的第二个
似然项。对照组和实验组共享 test pair、test label、ceiling 和 link 拟合方式，所以
`ΔNLL = NLL(A1) − NLL(A0)` 可以直接读成“辅助训练是否让历史最佳序更好”。

### 1.3 标签构造（必须先做对）

**输入**：

* step0 play 表：`player_id, beatmap_id, timestamp, playcount_cur, loss`；
* step1 pairbest 表：`player_id, player_id_base, player_seg, beatmap_id, split, n_plays, loss`。

**步骤**（必须与 step1 的规则完全同序）：

1. **玩家门槛先于空窗拆分**：删除 step0 记录数 < 20 的玩家；不要先按内容表过滤，
   否则空窗拆分边界会变。randommerge 上若在错误的段边界上匹配，只能对上
   354,653 / 453,406 对（我们已实测），剩下的 pair 会静默错配。
2. 按 `(player_id, timestamp, beatmap_id, playcount_cur)` 排序，做 365 天空窗拆分，
   得到新的段 `player_id`（段 0 保留原 id，后续段用 > max(step0 id) 的新 id）。
3. 对每个 `(player_id, beatmap_id)`，按 `(timestamp, playcount_cur)` 排序，取前 2 条；
   `loss_first2 = min(loss of first 2)`。
4. 与 step1 表按 `(player_id, beatmap_id)` 左连接；只保留 step1 原有 pair。
5. QC（任一失败即停止）：
   * 训练 + 测试 pair 的 first2 覆盖率 = 100%；
   * `n_plays` 与 step0 的实际条数一致；
   * `loss_hist <= loss_first2` 对所有 pair 成立（历史最佳不可能更差）；
   * `n_plays == 2` 时 `loss_hist == loss_first2`；
   * 不因 first2 的 ACC 重新过滤 pair（A1 主实验保留全部；A3 只 mask 辅助损失）。

**数据选择：四个数据集分别跑，不设“主数据集”**。

| 数据集 | step0 | step1 | P2/P4 | 备注 |
|---|---|---|---|---|
| 1k | `data/processed/step0_1k.csv` | `step1_1k_pairbest.parquet` | 已有 | canonical 6,828；不要用 6,819 的 `_nogap` |
| 10k | `data/processed/step0_10k.parquet` | `step1_10k_pairbest.parquet` | **需生成** | 16,061 图 / 1.5M 对 |
| randommerge | `data/processed/step0_randommerge.parquet` | `step1_randommerge_pairbest_content.parquet` | 已有 | 20,388 图 / 453k 对 |
| full | `data/processed/step0_full.parquet` | `step1_full_pairbest_content.parquet` | **需生成** | 20,832 图 / 1.9M 对 |

* 四个数据集各出各的 P2/P4 表；**不合并、不跨数据集比较绝对 NLL**。
* 10k/full 的 split 生成是 Stage 0 的一部分（§4）。
* first2 标签脚本对四个数据集用同一实现；只换 `--step0` / `--pairs`。

**产物**：`data/processed/step1_{tag}_first2.parquet`，在 step1 原列上增加：

* `loss_first2`（float32）
* `n_first2`（int8/int16，通常 2）
* `first2_ts`（第二条记录的时间；用于诊断）
* `first2_vs_hist`（`loss_first2 - loss`，诊断用）

### 1.4 模型与训练目标

**基线**：`linear_extra`，内容表 legacy CM3P 512d，外部信号统一为
`msd_overall, sunny_star, log_n`：

* `log_n` 在四个数据集上都是 100% 覆盖（`rank_signals_randommerge_od.parquet`）；
* 1k/10k 的主线表没有用 `log_n`，所以本实验的 A0 是**重训的 matched baseline**，
  不能直接拿旧表当 A0；
* 另跑一个 `msd_overall, sunny_star`（无 `log_n`）的 feature-set 敏感性，确认辅助效应
  不是 `log_n` 的交互。主表仍用三特征。

`d_m = w' C_m + v' S_m`，L2 `λ_ridge = 1e-3`，模型 seed 0。

**辅助目标**：

~~~text
L_hist(w)   = BT_NLL(train pairs, historical-best labels) / N_hist
L_first2(w) = BT_NLL(train pairs, first-two-best labels)  / N_first2
L(w)        = L_hist(w) + λ_aux * L_first2(w) + λ_ridge * ||w||^2
~~~

* 两个似然共享同一个 `d`；不引入 per-chart 自由参数；
* 各自按自己的 `N` 归一化，`λ_aux` 才是无量纲权重；
* 主实验固定两个任务的 logit 系数为 1；如果结果对 `λ_aux` 很敏感，再加
  “per-task temperature” 敏感性：`sigmoid(β_hist Δd)`、`sigmoid(β_aux Δd)`，
  `β_hist=1`、`β_aux=exp(γ)`，多估一个 `γ`。
* 本计划只用口径 A（`label_shift=0`）；若以后用 C/D 口径，first2 也要按同一 shift
  变换，不能只 shift 主标签。

**伪代码（P4 单 seed；P2 把“训练行”换成 chart fold 的训练行即可）**：

~~~python
# 输入：train rows (6x6), test rows (4x4), X=[C,S], loss_hist, loss_first2
fit_players, val_players = split_players_by_cluster(train.player_id_base, 0.2, seed=0)
tab_h_fit = pair_table(train[fit_players], loss_hist)
tab_f_fit = pair_table(train[fit_players], loss_first2)
tab_h_val = pair_table(train[val_players], loss_hist)

best = None
for lam_aux in [0, 0.05, 0.1, 0.2, 0.5, 1.0]:
    w = fit_linear_aux(X, tab_h_fit, tab_f_fit, lam_aux, lam_ridge=1e-3)
    val_nll = nll(tab_h_val, X @ w)          # 只看历史最佳验证
    best = min(best, (val_nll, lam_aux, w))

w = fit_linear_aux(X, pair_table(train, loss_hist),
                   pair_table(train, loss_first2), best.lam_aux, 1e-3)
d = X @ w
# test 上只算历史最佳 nll / paired delta；first2 test 只作诊断
~~~

**λ_aux 的选择**：

* 只按**训练侧历史最佳验证 NLL**选，不看 test；
* P4：每个 seed 内，用 20% 的 `player_id_base` 做验证（`--val-frac 0.2`），
  在 6×6 训练行上拟合、在验证玩家上算历史最佳 NLL；选完再在全部 6×6 上重拟合；
* P2：在每个 chart fold 内做同样的 20% 玩家验证；**不要**在全部 train 图上选
  `λ_aux` 后再套到 chart fold（那会让 held-out chart 的标签进入超参）；
* grid：`0, 0.05, 0.1, 0.2, 0.5, 1.0`（0 就是 baseline；先粗后细）。

**实验臂**：

| 臂 | 辅助标签 | 辅助损失范围 | 目的 |
|---|---|---|---|
| A0 | 无 | — | baseline |
| A1 | first2 | 全部 train pair | **主实验** |
| A2 | first2 | 仅 `n_plays > 2` | 只加“有信息”的 pair |
| A3 | first2 | 仅 first2 ACC ∈ (0.90, 0.998] | 目标支撑匹配 |
| A4 | random-2（随机两条） | 全部 | 控制“去噪/正则化”，不是“早” |
| A5 | last-2（最后两条） | 全部 | 控制“早 vs 晚” |
| A6 | first2 标签在**训练侧** pair 间打乱（不跨 train/test） | 全部 | 控制“第二个似然本身” |

**分阶段**：

1. **1a（pilot）**：randommerge P4 s1，A0–A5；只检查实现、效应量、λ 曲线，不做结论。
   通过后四数据集并行铺开。
2. **1b（确认）**：1k / 10k / randommerge / full × P2/P4 × 5 seeds，A0/A1/A2/A3；
   若 A1 为正，再加 A4/A6。按 (dataset, protocol, seed) 拆 HPC job array。
3. **1c（外部）**：VSRG benchmark，四个数据集 × 2 臂（§1.5.3）。
4. **汇总**：四数据集 × 两协议 × 两指标的 4×2 表；不做跨数据集合并。

### 1.5 评价

#### 1.5.1 主评价

* 同一 test pair 集上的 `ΔNLL = NLL(A1) − NLL(A0)`；
* 每个 seed 用 `scripts/paired_delta_nll.py` 按 `player_id_base` 做 400 次 cluster
  bootstrap，报点估计与 95% CI；
* 5 个 split seed 报 mean ± std、负值 seed 数（5/5 等）；
* **相关性（用户要求，与 ΔNLL 并列）**：Pearson(d,b)、Spearman(d,b)、逐玩家
  lw/macro Spearman，每个数据集 × 协议 × seed 都报；另给 5-seed 的 mean ± std；
* 同时报 link β、ceiling 覆盖、test pair 数。

**注意**：5 个 split seed 不是独立样本（同一批图/玩家，只是划分不同）。不要把 5 个
seed 的 std 当成总体标准误；主证据是“每个 seed 的配对 CI + 5/5 方向”，不是跨 seed 的
t 检验。相关性系数同理：报逐 seed 与均值，不用它单独做显著性检验。

#### 1.5.2 机制分层

* **按练习机会**：在 raw within-player pair 上按 `min(n_plays_a, n_plays_b)` 分层
  （2 / 3 / 4–5 / 6–10 / 11–20），报每层的 ΔNLL。若 H1 成立，增益应更多出现在
  `min(n_plays) ≥ 3` 的层。`min=2` 的层里至少一张图的标签被截到 2 条；只有两张都
  `n_plays=2` 时辅助标签才等于主标签，所以该层仍可能因另一张图 `n_plays>2` 而有小增益。
* **按难度**：msd_overall / sunny_star 四分位；
* **按玩家活跃度**：每个 player cluster 的 train pair 数分层；
* **诊断**：在 first2 test 标签上评估 A1 vs A0。如果 A1 只改善 first2、不改善历史最佳，
  说明目标不匹配，不能算成功。

> 现有 `scripts/stratified_delta_nll.py` 只支持 chart-level 分层；`n_plays` 是
> player–chart 级变量，需要新脚本 `scripts/eval_pair_strata.py`，在 raw pair 上分层
> 后再按 player cluster bootstrap。

#### 1.5.3 VSRG benchmark（外部检查）

VSRG 指仓库里的 **VSRG Dan Benchmark**（1,382 行，5 折 chart-group OOF isotonic）。
“四个数据集分别跑”也延伸到外部检查：四个数据集各训一对 baseline/aux，做 8 个模型：

| 训练集 | baseline | aux | 备注 |
|---|---|---|---|
| 1k | B0 | B1 | 与 1k 主表同数据 |
| 10k | B0 | B1 | 更大、更偏 top 玩家 |
| randommerge | B0 | B1 | **与 RM-L（0.4216）同训练集** |
| full | B0 | B1 | 最大、覆盖最广 |

* 特征集用 `msd_overall, sunny_star`（与 RM-L 可比；benchmark 的 `log_n` 需要从
  `benchmark_content_legacy.parquet` 的 `n_windows` 另算；主表先不加，避免多一个变量）；
* 训练：`linear_extra`，λ=1e-3，seed 0，`--fit-all`，保存模型；
* 推理 + 5 折 chart-group OOF isotonic，四个训练集共用同一套 benchmark 折/校准；
* 报整体 MAE、Spearman、RC/LN/course、expected 分带、排除 201 个训练重叠 bid 的敏感性；
* 主对照是**同训练集的 baseline vs aux**（B1−B0）；RM-L（0.4216）只作 randommerge 行的
  外部参照。若四个数据集的 DCR 效应与 VSRG 效应方向不一致，单独报，不强行合并。

### 1.6 判据（预注册）

**四个数据集分别判定，不合并 NLL**；每个数据集内部看 P2/P4 5 seeds 的 paired ΔNLL
与相关性。下面的“≥3/4 数据集”只用于生产建议，不是跨数据集统计检验。

| 结果 | 判据 | 行动 |
|---|---|---|
| **强成功** | ≥3/4 数据集 mean ΔNLL ≤ −0.004；这些数据集 5/5 seed 为负、≥4/5 CI 排除 0；Pearson/Spearman 不回退；VSRG 至少 3/4 数据集同向 | 进入生产候选 |
| **成功** | ≥3/4 数据集 mean ΔNLL ≤ −0.002；这些数据集 ≥4/5 为负、≥3/5 CI 排除 0；相关性不回退；VSRG 不明显变差 | 值得复现/扩展 |
| **零结果** | 四个数据集都在 ±0.001 内，或方向混合 | 停止，不投入 first2 标签 |
| **负结果** | ≥3/4 数据集 mean ΔNLL > +0.001 | 停止，记录负面结果 |

补充规则：

* **相关性单独看**：如果 ΔNLL 变好但 Pearson/Spearman 明显回退（例如 ΔPearson < −0.01），
  标为“不一致”，不进入生产；不能只凭 NLL 一个数下结论。
* 如果 A1 为正，但 A4（random-2）或 A6（打乱）与 A1 同样好 → 只能声称“辅助损失正则化”，
  不能声称“练习不变性”；不进入生产。
* 如果 DCR 赢但 VSRG 在 ≥3/4 数据集变差 > 0.02 → 不声称“提升 VSRG 泛化”，只报 DCR
  结果，并把这个冲突作为后续分析对象。
* 如果 A2（仅 n_plays>2）明显好于 A1 → 说明一半“重复标签”在稀释辅助信号，后续默认 A2。
* P2 与 P4 结果不一致时分开报：P4 是双重冷启动，不能和 P2 平均成一个数。

### 1.7 代码改动与产物

**新脚本**：

* `scripts/step1_first2_labels.py`
  * `--step0`, `--pairs`, `--out`, `--gap-days 365`, `--min-plays-per-player 20`
  * 输出 `loss_first2` 等列 + QC JSON；
* `scripts/step2_cm3p_rank_aux.py`（或给 `step2_cm3p_rank.py` 加 `--aux-label-col`）
  * 复用 `fit_linear` 的 PairTable 结构，构建第二个 `PairTable`；
  * 目标 `L_hist + λ_aux L_first2`；
  * `--aux-weight-grid`、`--aux-scope {all, n_plays_gt2, support}`；
* `scripts/run_first2_aux.py`
  * 类似 `run_od_legacy.py`：train / eval / summarize；
  * 复用 `run_p2_p4_multiseed.py` 的 split 路径和 `paired_delta_nll.py`；
* `scripts/eval_pair_strata.py`
  * 按 `min(n_plays)` 等 raw-pair 变量分层，输出每层 ΔNLL + CI。

**产物**：

| 路径 | 内容 |
|---|---|
| `data/processed/step1_{1k,10k,randommerge,full}_first2.parquet` | 四个数据集的 first2 标签 |
| `data/processed/step1_10k_pairbest_p{2,4}_s{1..5}.parquet` | 10k 的新 split |
| `data/processed/step1_full_pairbest_p{2,4}_s{1..5}.parquet` | full 的新 split |
| `results/first2_aux/pilot/…` | 1a 的 λ 曲线与烟测 |
| `results/first2_aux/{dataset}/{p2,p4}/s*/…` | 1b 的信号、rank_eval、paired_delta |
| `results/first2_aux/summary.md` | 四数据集主表 + 相关性 + 分层 + 判据 |
| `results/first2_aux/vsrg/…` | 1c 的 4 数据集 × 2 臂 = 8 个模型、OOF、MAE |

### 1.8 计算量（全部提交 HPC）

* **标签构造**：1k 1.4M play、10k 13.4M、randommerge 4.5M、full 17.3M；每个数据集一个
  HPC CPU 作业（full 约几分钟到十几分钟，取决于 I/O）。
* **10k/full split 生成**：P2/P4 各 5 seeds；10k ~1.5M 行、full ~1.9M 行，一个 seed
  一个作业，避免单进程内存峰值。
* **训练**：每个数据集 × P2/P4 × 5 seeds × 臂。randommerge P4 单次 `linear_extra`
  约 438 s（8 线程）；10k/full 的 P4 训练行更多，单次可能 10–30 min；HPC 上按
  (dataset, protocol, seed) 拆成 job array，可同时跑几十个。
* **VSRG**：8 个 `--fit-all` 模型（4 数据集 × baseline/aux）；randommerge 约
  15–35 min，10k/full 约 1–3 h；推理/校准几分钟。拆成 8 个作业。
* 全部 CPU；无 GPU（音频编码除外，见 §2）。

### 1.9 风险与限制

1. **标签噪声**：first2 只有两条记录，比历史最佳噪；λ_aux 验证可能选到 0。
2. **支撑偏移**：5.5% 的 first2 ACC ≤ 0.90；A1 保留，A3 做 mask 敏感性。
3. **重复标签**：约一半 pair 的 first2 = 历史最佳；A1 会重复主损失，A2 是更干净的对照。
4. **早 vs 晚**：first2 也受日历时间/早期水平影响；A5（last-2）用来区分“早”与“晚”。
5. **测试集 n_plays 分布**：P4 的 test 玩家/图与 train 不同；分层报告，不要只报总体。
6. **四个数据集的切片不同**：1k 是 top-1000、10k 是 top-10000、randommerge 是随机快照、
   full 是三集并集；结论分别写，不能把四个数据集的 NLL 合并成一个数。
7. **VSRG 构造不同**：Dan 标签不是 ACC；VSRG 结果只作外部检查。
8. **10k/full 的 P4 规模**：单次 fit 的行数/内存明显更大；按 seed 拆作业，先跑一个
   smoke seed 确认时间和内存，再铺 5 seeds。

---

## 2. Experiment 2：整 beatmapset 留出，查音频增益能迁移多远

### 2.1 问题与假设

**问题**：audio-on 相对 audio-off 的增益，有多少来自“同一首歌的其他难度”被监督训练见过？
如果把整个 `beatmapset_id` 留出，audio 还能帮到“新谱面集”吗？

**两种解释**：

* **H_song**：audio 向量主要编码歌曲身份/音频风格；同 set 的其他难度在训练侧时，模型可以
  通过音频“认出这首歌”，把难度从同曲图迁移过来。整 set 留出后增益应大幅缩小。
* **H_content**：audio 编码了 tempo、密度、和弦/长条等跨歌曲可迁移的难度线索；整 set 留出后
  增益仍在。若是后者，audio 对 randommerge/full 的“新歌”也有价值。

这与 StepMania 难度估计里“按整首歌留出”的做法同源：同一首歌的多个难度共享音频/编曲，
不能当成独立样本；否则模型可以靠“认出这首歌”而不是“读懂这张谱”来得分。

### 2.2 Step 2a：用现成预测重算“新 set”配对差（只读）

**输入**：

* `data/processed/step1_1k_pairbest_p4_nogap_s{1..5}.parquet`（6,819 图）；
* `data/processed/beatmap_meta_1k.csv`（`beatmap_id, beatmapset_id`）；
* `results/p4_multiseed_audio/{audio,legacy}/p4/s{1..5}/signals.parquet`
  （4 个 embedding 方法 + fixed 信号）。

**定义**：

~~~text
train_charts = unique(beatmap_id where split == "train")     # 6×6
test_charts  = unique(beatmap_id where split == "test")      # 4×4
train_sets   = { beatmapset_id(m) : m in train_charts }
new_set(m)   = beatmapset_id(m) not in train_sets
~~~

* 对每个 test player，取所有 test chart 的无序对；
* **新 set 子集**：两张图都满足 `new_set`；
* **对照子集**：至少一张图 `not new_set`；再分 **mixed**（一张 new、一张 seen）与
  **both-seen**（两张都 seen）；
* 在每个子集内部再分：同 set vs 跨 set。

**指标**：

* audio vs legacy 的 paired ΔNLL（每个方法单独拟合 link，子集内同 pair 集）；
* 若子集内 link 拟合不稳（β 的 seed 间波动大），用全 test 上拟合的 β 固定后重算 NLL
  作为敏感性；两种口径都报；
* 按 `player_id_base` cluster bootstrap 400 次；
* 覆盖率：新 set test 图 / 全部 test 图；新 set 比较 / 全部 test 比较；
  有 ≥1 个可用 pair 的玩家数；
* 分层：msd/sunny 四分位、`ln_ratio`、`min(n_plays)`；报告子集构成，避免“新 set 子集
  恰好更难/更简单”被误读。

**预实验已跑**（§0.4c）：覆盖率 10.8–18.2%，Δ 在 5/5 seed 为负，且没有系统性小于
完整 test。
这支持“audio 不是纯同曲泄漏”，但仍是条件子集；Step 2b 才是干净检验。

**脚本**：`scripts/eval_newset_audio.py`，输出
`results/audio_newset/subset_s{seed}.json` 与 `summary.md`。

### 2.3 Step 2b：整 beatmapset 分组的 P4

#### 2.3.1 划分算法

在 `scripts/step1_p4_split.py` 增加 `--group-col beatmapset_id`（或新脚本
`scripts/step1_p4_split_sets.py`）：

1. 读 step1 pairbest + meta，给每张图附 `beatmapset_id`；缺失 set 的图按单图组处理并报告。
2. **玩家侧不变**：按 `player_id_base` 的活跃度（pair 行数）分 5 层，层内 60/40；
3. **谱面侧按 set 分组**：
   * 以 set 为单位，计算 set popularity = 该 set 内 distinct `player_id_base` 数；
   * 按 popularity 分 5 层；
   * 每层内把 set 分到 train/test，目标是**图数** 60/40（不是 set 数 60/40）。
     建议：层内按 set 图数从大到小排序，依次分给当前图数缺口更大的一侧；同图数随机
     打破。这样避免一个 35 图的 set 把比例拉偏。
4. 输出：与原 P4 相同 schema，加 `group_id`（set id）与 `p4_block`；
   `split ∈ {train,test,dropped}`。
5. QC：
   * 没有任何 set 同时出现在 train 和 test；
   * **分配图数**接近 60/40（55–65%）；**实际有 train/test 行的图数**另行报告
     （受玩家-图稀疏影响，通常低于分配数）；
   * 每个 block 的行数、test pair 数、ceiling 覆盖；
   * train/test 的 difficulty / popularity / LN 分布不要出现大偏移（报告，不强制匹配）。

**模拟结果**（§0.4d）：3,806 set；分配图 4,045/2,774（59.3/40.7%），实际有行图
3,509/2,058；37,418/18,902 行；924,384 test 比较；跨 set 98.8%。正式实现用图数平衡后，
分配图比例应更接近 60/40；实际有行图比例仍会因稀疏而偏离。

#### 2.3.2 训练与评测

* 数据：`step1_1k_pairbest_nogap.parquet`（6,819 图）→
  `step1_1k_pairbest_p4sets_s{1..5}.parquet`；
* 两臂：audio-on（`chart_content_cm3p_audio_on.parquet`）vs legacy audio-off
  （`chart_content_cm3p_local.parquet`）；
* 方法：4 个（linear、linear_extra、MLP、MLP_extra），与现有多 seed 表一致；
* 评测：`step2_rank_eval.py --split test --ceiling-fit all --common-maps --strict-pairs`；
* 主指标：test 上的 paired ΔNLL（audio − legacy），每 seed 的 cluster bootstrap；
* 次指标：Pearson/Spearman、ceiling、test pair 数；
* **严格子集**：跨 set 的 test 比较（两张图 set 不同）。整 set 留出后同 set 的 test 对
  仍共享音频；跨 set 子集是最干净的“新歌”检验。模拟中跨 set 占 98.8%，所以主结果
  基本就是它；仍分开报。

#### 2.3.3 与标准 P4 的比较

* 标准 P4 的 Δ（同方法、同 5 seeds）来自 `results/p4_multiseed_audio`；
* 只比较 **Δ**，不比较绝对 NLL（test 集/ceiling 不同）；
* 报 `Δ_setP4 − Δ_stdP4` 的描述性差值，以及各自 5/5、CI；
* 若 set-P4 的 ceiling 明显更高（test 更难/更噪），同时报相对 ceiling gap 的归一化
  效应，避免把“NLL 压缩”误读成“增益消失”。

### 2.4 判据

| 结果 | 判据 | 行动 |
|---|---|---|
| **通过** | Δ_setP4 ≤ −0.004，≥4/5 seed 为负，≥4/5 CI 排除 0；跨 set 子集 ≤ −0.003 | 认为 audio 在 **1k 新 set** 上可迁移；报告 1k 结果；**不补** randommerge/full 音频 |
| **弱通过** | −0.004 < Δ_setP4 < 0，≥4/5 为负 | 方向对但幅度不足；报告 1k；不补音频 |
| **不通过** | Δ_setP4 ≥ 0，或 abs(Δ) < 0.001 | 现有 1k 增益主要是同 set 泄漏；报告 1k；不补音频 |
| **方法反转** | MLP_extra 与 linear_extra 方向相反 | 先查优化/容量；报告 1k；不补音频 |

**数据边界（已确认）**：randommerge/full 音频**不补**（网络/磁盘限制），因此
Experiment 2 的结论只覆盖 1k 的 6,819 图；不能外推到 randommerge/full。若未来音频
条件允许，再用同一套 set-P4 判据决定是否扩展。

### 2.5 代码与产物

**脚本**：

* `scripts/eval_newset_audio.py`（Step 2a）；
* `scripts/step1_p4_split_sets.py` 或 `step1_p4_split.py --group-col`（Step 2b）；
* 复用 `scripts/run_p2_p4_multiseed.py`（`--split-p4 … --content-path …`）；
* 复用 `scripts/paired_delta_nll.py`（子集/跨 set 的 Δ）。

**产物**：

| 路径 | 内容 |
|---|---|
| `data/processed/step1_1k_pairbest_p4sets_s{1..5}.parquet` | 整 set 留出 P4 |
| `results/audio_newset/subset_s*.json` | Step 2a 的条件子集 Δ + 覆盖率 |
| `results/audio_newset/p4sets/…` | 两臂 × 4 方法 × 5 seeds 的信号与 rank_eval |
| `results/audio_newset/summary.md` | 主表 + 跨 set + 与标准 P4 对照 |

### 2.6 计算量（全部提交 HPC）

* Step 2a：纯 CPU 读表 + pair 表，一个 HPC 作业几分钟；五个 seed 可一个作业顺序跑。
* Step 2b：1k 6,819 图 / 109k 行；4 方法 × 2 臂 × 5 seeds。按 (method, arm, seed)
  拆 job array；单 job 约几分钟（linear）到十几分钟（MLP），HPC 可同时跑 40 个。
* **不补音频**：randommerge/full 的音频补齐受网络/磁盘限制，不在本计划内；2b 只用
  1k 现成的 6,819 张 audio 向量。

### 2.7 风险与限制

1. **整 set 划分改变难度/热度平衡**：按 set popularity 分层 + 图数平衡；报告分布。
2. **set 内图数差异大**：单图 set 2,456 个，最大 35 图；图数平衡算法必须显式处理。
3. **同 set 的 test 对**：整 set 留出后，test 内部仍有同 set 对；跨 set 子集是更严格的检验。
4. **编码器预训练暴露**：CM3P 的 audio 编码器可能在大语料上见过这些歌；Step 2b 只能排除
   “监督训练侧的同 set 图”，不能排除编码器预训练暴露。结论要写成“超出监督同 set 的迁移”。
5. **音频来源混合**：published 4,037 + local 2,782；local 子集更难/更热门。set-P4 仍用混合表；
   若通过，只报告 1k 的混合来源结果，不拆分来源。
6. **不能外推到 randommerge/full**：音频补齐受网络/磁盘限制；Experiment 2 只回答
   1k 的 6,819 图，不能据此说 randommerge/full 也会受益。
7. **ceiling 噪声**：set-P4 的 test 图更少/更集中，ceiling 覆盖可能下降；报 ceiling 与
   coverage，captured 只作参考。

---

## 3. Experiment 3：用 data2vec 残差修正强基线最容易判错的图

### 3.1 问题与假设

**问题**：在编码器没见过的图上，当 legacy 强基线（CM3P + external）给两张图近似难度时，
现成的 D2V 512d 向量能否分出哪张更难？长押图是否更明显？

**假设**：

* H1：D2V 含有 CM3P+MSD/Sunny 之外的列/长条信息；在基线近分对上有独立增益。
* H2：增益集中在 LN 图（hold 通道真正有用）；RC 上没有独立增益。
* H0：D2V 的增益只是额外容量/正则化，随机 512d 控制也一样。

**注意**：这不是“D2V 整体比 CM3P 强”的复测。D2V+ext 在 P4-holdout 为 0.6028；
它与 legacy MLP+ext 的**同子集**干净对照尚未补（0.5734 是 1k 全量 P4，不是同一批图）。
本实验只问“残差有没有独立信息”，不回答“D2V 是否整体更好”。

### 3.2 数据与基线

**编码器未见的图**：D2V holdout 的 1,365 张（编码器预训练 20,584 张，holdout 1,365）。

| 口径 | 文件 | 行 / 图 / 玩家 | test 比较 |
|---|---|---:|---:|
| P2-holdout | `step1_1k_pairbest_d2v_holdout.parquet` | 23,905 / 1,365 / 933 | 54,721（7,276 test 行） |
| P4-holdout | `step1_1k_pairbest_p4_d2vhold_s{1..5}.parquet` | 23,902 / 1,365 / 932 | 31,962（3,556 test 行） |

**内容**：

* CM3P legacy：`chart_content_cm3p_local.parquet`（512d）；
* D2V：`chart_content_d2v_holdout.parquet`（512d，L2=1）；
* 外部信号：`rank_signals_1k.parquet` 的 `msd_overall, sunny_star`；
* LN/pattern：`ln_ratio, patt_ln_percent, patt_mode_tag, n_holds`。

**强基线（两个同等对待，不预设主/备）**：

* `mlp_extra`：CM3P legacy + [msd, sunny]，h64×2、dropout 0.1、98 epoch、seed 0
  （与 1k P4 最强基线同配置）；
* `linear_extra`：λ=1e-3（小样本更稳）。

两个都跑 **P2-holdout 与 P4-holdout**；最后是 2×2 表（model × protocol），不先验地选
“主口径”。如果两个基线结论不一致，分开写，不取平均。

**必须重训基线**：现有 1k P4 的 MLP+ext 是在 6,828 图上训练的，holdout 图可能出现在它的
训练侧；不能直接拿它的预测当“编码器未见图”的基线。要在 1,365 图的 split 上重新训练
（P2 5 折 chart-disjoint；P4 6×6），保持同超参。

**小样本风险**：P4-holdout 只有 8,882 行 train / 3,556 行 test；MLP 可能过拟合。报告
train/val NLL、早停/固定 epoch 的差异；如果 MLP 不稳定，仍然报它，但把 `linear_extra`
作为稳定对照；两者方向不一致时以“不一致”记录，不强行选一个。

### 3.3 训练内折外预测的线性修正

**目标**：`d_m = d_base(m) + w' z_m`，其中 `z_m` 是 D2V 向量在训练图上标准化后的版本。

**为什么折外**：如果 `d_base` 是 in-sample 预测，修正项会学到基线的过拟合残差，而不是
可迁移的 D2V 信息。所以基线分数必须在训练侧折外生成。

**P2-holdout**：

* 直接复用 5 折 chart-disjoint：每张图的 `d_base` 来自没看过该图的折模型；
* 现有 `results/d2v_main/p2_hold_d2v_signals.parquet` 是 D2V 的 OOF 分数，但**不是**
  legacy 基线的；需要为 `mlp_extra` 与 `linear_extra` **各**跑一次 5 折。

**P4-holdout**：

* 在每个 seed 的 6×6 训练块内，把训练图分成 5 折（按 popularity / 难度分层）；
* 对每折，用其余 4 折的训练行拟合基线，预测该折图的 `d_base`；
* 得到所有 6×6 图的 OOF 基线分数；两个基线各做一遍；
* test 图的 `d_base` 用全部 6×6 行拟合的基线（或 5 折模型平均；主表默认全量拟合）。

**修正拟合**：

~~~text
z_m = (D2V_m - mean_train) / sd_train
d_m = d_base_m + w' z_m
L(w) = BT_NLL(train pairs; d) + λ_w * ||w||^2
~~~

* `λ_w` grid：`1e-3, 1e-2, 1e-1, 1, 10`（D2V 512d，样本小，强正则很可能赢）；
* 用训练侧 20% player clusters 的 historical-best NLL 选 `λ_w`；不看 test；
* 可选敏感性：`d = a * d_base + w' z`（a 自由）、PCA top-{16,32,64}、只对
  `ln_ratio ≥ 0.15` 的图加修正。

**对照**：

| 对照 | 做法 | 目的 |
|---|---|---|
| C1 random-512 | 用同维随机高斯特征替代 D2V | 控制“额外容量” |
| C2 shuffled-D2V | 把 D2V 行在图上打乱 | 控制“图级特异信息” |
| C3 D2V-only | 不用 `d_base`，只用 D2V 线性 | 参考 |
| C4 concat | `[CM3P, D2V]` 直接线性 | 检查“残差”是否不如简单拼接 |
| C5 baseline | 只用 `d_base` | 主对照 |

### 3.4 近分对与长押图

#### 3.4.1 近分阈值（训练集定）

* 对每个训练比较，算基线标准化差：
  `Δz = (d_base(b) - d_base(a)) / sd_train(d_base)`；
* 取 `τ = quantile(|Δz|, 0.25)`（训练比较的 bottom 25%）作为“近分对”；
* test 用同一个 `τ`，不重新调；
* 敏感性：10% / 33% 分位；以及“基线在近分对上的符号准确率 ≈ 55%”作为另一种定义。

**读法**：

* 近分对上的 NLL 和“修正后排序翻转的正确率”直接回答“D2V 能否在基线犹豫时分开”；
* 如果只在近分对上有增益、整体没有，仍然支持“四列局部编码器”的方向，但不要声称整体
  模型更好。

#### 3.4.2 长押图

* 主定义：`ln_ratio ≥ 0.15`（仓库 Mix/LN 阈值），pair 中至少一张图满足；
* 次定义：`patt_mode_tag ∈ {Mix, LN}`；
* 严格子集：两张图都 `ln_ratio ≥ 0.15`（P4-holdout 约 10% 比较；样本够就报）；
* 也报 `|ln_ratio_a - ln_ratio_b|` 的分层：一张 LN、一张 RC 的对比最能检验“长押期间的
  换指”信息。

### 3.5 评价与判据

**评价（2 基线 × 2 口径，不分主次）**：

* 整体 test 上的 paired ΔNLL（corrected − baseline），对
  `mlp_extra`/`linear_extra` × P2-holdout/P4-holdout 的 2×2 表分别报；
* 近分对 ΔNLL；
* LN 子集 ΔNLL；
* 每个都按 `player_id_base` cluster bootstrap；P4 报 5/5，P2 报单划分 CI；
* 四列编码器的投入门槛：至少 3/4 格同向，且近分/LN 的结论不被某一个基线单独驱动。

**判据树**：

| 观察 | 解释 | 行动 |
|---|---|---|
| 整体 Δ ≤ −0.002，近分 Δ ≤ −0.005，C1/C2 不显著（abs(Δ) < 0.002 或 CI 跨 0），LN 不变差 | D2V 有独立整体信息 | 投入四列局部编码器 |
| 整体 ≈ 0，但 LN Δ ≤ −0.005 且近分-LN Δ ≤ −0.005 | 信息集中在长押 | 做 LN-aware 局部编码器，不做通用大模型 |
| 近分 Δ ≤ −0.005，整体 ≈ 0 | 只在基线犹豫处有用 | 做“近分对重排”专用头/局部编码器 |
| C1 或 C2 与 D2V 一样好 | 增益是容量/正则化 | 拒绝，不投入 |
| 整体/近分/LN 都不显著 | 现成 D2V 没有独立信息 | 停止这条线 |

**多重比较**：整体、近分、LN 是三个预先指定的子集；不要把更多子集的结果当确认性证据。
P2 与 P4 不可互比，各自报。

**2×2 一致性**：上面的 Δ 要在 `mlp_extra`/`linear_extra` × P2/P4 的 2×2 表中至少
3/4 格同向；若只有 MLP 或只有 P4 有，标为“directional”，不直接投入四列编码器。

### 3.6 计算量与功效（全部提交 HPC）

* 基线：2 个模型 ×（P2 5 折 + P4 5 seeds × 5 折）= **60 次拟合**（小数据，CPU
  分钟级到十几分钟）；按 (model, protocol, seed/fold) 拆 job array。
* 修正：凸问题，512d × ~1–4 万 pair，秒级到分钟级；λ grid × seeds 也很快。
* 无 GPU。

**功效**：P2-holdout 54,721 比较，P4-holdout 31,962；近分 25% 约 13.7k / 8.0k；LN 至少一张
约 34.4k / 16.4k。按现有 cluster bootstrap 的经验，近分对的 95% CI 半宽约 0.01–0.02；
**只能可靠检测 ≥0.005–0.01 的效应**。如果 pilot 的效应在 0.002–0.005，需要：
(a) 把 P2 和 P4 的结论分开报；(b) 增加 D2V holdout 图数（重做编码器 holdout 划分）或
(c) 用更多 seed/更细的 pair 采样。不要在功效不足时宣称“没有增益”。

### 3.7 风险与限制

1. **样本小**：1,365 图 / 3.5k–7.3k test 行；近分/LN 子集更小。
2. **基线不对称**：CM3P 可能在预训练中见过这些图；D2V 没有。D2V 的残差增益是保守估计；
   若 D2V 失败，不能直接说“D2V 没信息”，可能是基线太强。
3. **MLP 过拟合**：小训练集上 98 epoch 可能过拟合；用 validation/早停，保留 `linear_extra`
   作为稳定对照。不因为 MLP 过拟合就只报 linear；两个都报，方向不一致标为“不一致”。
4. **近分阈值选择**：25% 是预注册的默认；如果换阈值结果翻转，说明效应不稳健。
5. **LN 定义**：`ln_ratio ≥ 0.15` 包含 Mix；纯 LN 只有 128 张，单独评测 pair 不够。
6. **“四列”不是直接可识别**：D2V 是 8 通道 piano roll 的 512d 表示，不是显式的四列 token；
   本实验只证明“有独立信息”，列特异性要靠后续局部编码器的消融。

---

## 4. 执行顺序、依赖与时间

| 阶段 | 内容 | 依赖 | 计算 | 产物 |
|---|---|---|---|---|
| 0a | 写脚本 + smoke test | 无 | HPC debug | 单元测试通过 |
| 0b | 10k/full P2/P4 splits + 四数据集 first2 标签 | 0a | HPC CPU，每数据集一作业 | `data/processed/step1_*_first2`、`*_p{2,4}_s*` |
| 1 | Experiment 2a：新 set 子集重算 | 无 | HPC CPU，分钟 | `results/audio_newset/subset_*` |
| 2 | Experiment 1a：randommerge P4 s1 pilot | 0b | HPC CPU，小时 | `results/first2_aux/pilot` |
| 3 | Experiment 1b：四数据集 × P2/P4 × 5 seeds | 0b/2 | HPC CPU，job array | `results/first2_aux/{dataset}` |
| 4 | Experiment 1c：VSRG 8 模型 | ≥3/4 数据集 DCR 为正 | HPC CPU | `results/first2_aux/vsrg` |
| 5 | Experiment 2b：整 set P4 | split 脚本 | HPC CPU，job array | `results/audio_newset/p4sets` |
| 6 | Experiment 3：两基线 OOF（60 fits） | D2V holdout | HPC CPU，job array | `results/d2v_residual/baseline` |
| 7 | Experiment 3：残差 + 对照 | 6 | HPC CPU，小时 | `results/d2v_residual` |
| 8 | 汇总报告 | 1–7 | HPC summarize / 本地读 | 3 篇 docs + 总表 |

**并行性**：0b、2a、Exp3 基线可以立即提交；Exp1 的四个数据集、P2/P4、5 seeds 都是独立
作业；Exp2b 不依赖 Exp1；Exp3 不依赖 Exp1/2。只有“汇总/报告”依赖前面完成。

### 4.1 全部 CPU 任务提交 HPC

按你的要求，**包括 CPU 任务在内，全部提交 HPC**，本地只改代码、读结果：

* **环境**：`$HOME/.conda/envs/DCR/bin/python`，`PYTHONNOUSERSITE=1`；仓库在
  `$HOME/DCR`，`data/processed` 已同步；每个 job `OMP_NUM_THREADS=4`（或 8），
  避免 40 个 job 同时抢核。
* **作业粒度**：一个 job 只做 (dataset, protocol, seed) 或 (dataset, stage)；
  10k/full 的 P4 **一次只跑一个 seed**，避免单进程内存峰值。
* **job array**：
  * Exp1：4 数据集 × 2 协议 × 5 seeds × 4 臂（A0/A1/A2/A3）= 160 个训练 job；
    建议按臂拆 4 个 array（每个 0–39），或一个 `--array=0-159`；
  * Exp2b：4 方法 × 2 臂 × 5 seeds = 40 个 job（`--array=0-39`）；
  * Exp3：2 基线 × 30 fits = 60 个 job（`--array=0-59`）；
  * 用 `%` 限制并发（例如 `--array=0-39%12`），先跑 smoke 再放开。
* **队列/资源**：CPU 任务用 `i64m512u` 或 `emergency_cpu`，8–16 核 / 64–128G / 12h；
  短作业用 `debug` 做 smoke。10k/full 的 P4 fit 若超过 12h，拆成 per-seed 或 per-fold。
* **数据交换**：HPC 上生成 labels/splits/signals；只把 JSON/MD/小 CSV `rsync` 回本地；
  大 parquet 留在 HPC，本地通过 `sshfs`/按需拷贝读取。
* **日志**：每个 job 写 `results/.../logs/<stage>_<dataset>_<protocol>_s<seed>.log`；
  失败先看 `tail -50`，不要重跑整个 array。
* **新增 slurm 脚本**（建议）：
  `scripts/slurm/first2_labels.slurm`、`first2_aux_train.slurm`、
  `first2_aux_eval.slurm`、`audio_newset.slurm`、`d2v_residual.slurm`。

### 4.2 不能说的话

* **Experiment 1**：不能说“first2 让模型更懂谱面难度”；只能说“在这四个数据集的
  historical-best ACC 序上，辅助训练使 paired ΔNLL / 相关性变好或不变”。也不能把 DCR
  的改善直接说成 VSRG 泛化改善；VSRG 必须单独报。
* **Experiment 2**：set-P4 不通过时，不能说“audio 没有信息”；只能说“现有增益主要来自
  监督训练侧的同 set 图”。set-P4 通过时，也不能说“编码器没在预训练中见过这些歌”；
  只能排除“监督同 set 泄漏”。
* **Experiment 3**：残差为 0 时，不能说“D2V 没有列信息”；只能说“在这 1,365 张图上，
  冻结点积/线性残差没有超过强基线”。残差为正时，也不能直接说“四列编码器一定更好”；
  列特异性要由后续消融证明。
* 三个实验都不能跨数据集/协议比较绝对 NLL；都不能把 captured 当作跨口径可比量。

---

## 5. 代码与文件清单

**新增脚本（建议）**：

| 脚本 | 作用 |
|---|---|
| `scripts/step1_first2_labels.py` | 从 step0 + step1 生成 first2 标签 + QC |
| `scripts/step2_cm3p_rank_aux.py` | 共享 d 的双似然线性拟合 |
| `scripts/run_first2_aux.py` | 1a/1b 的 train/eval/summarize |
| `scripts/eval_pair_strata.py` | 按 `min(n_plays)` 等 raw-pair 变量分层 ΔNLL |
| `scripts/eval_newset_audio.py` | 2a 的新 set 子集重算 |
| `scripts/step1_p4_split_sets.py` | 2b 的整 set P4（或扩展现有脚本） |
| `scripts/fit_d2v_residual.py` | 3 的 OOF 基线 + 残差拟合 |
| `scripts/eval_d2v_residual.py` | 3 的整体/近分/LN/对照评测 |
| `scripts/slurm/first2_*.slurm` | 0b/1a/1b/1c 的 HPC job array |
| `scripts/slurm/audio_newset.slurm` | 2a/2b 的 HPC job |
| `scripts/slurm/d2v_residual.slurm` | 3 的 HPC job array |

**数据产物**：

* `data/processed/step1_{1k,10k,randommerge,full}_first2.parquet`
* `data/processed/step1_10k_pairbest_p{2,4}_s{1..5}.parquet`
* `data/processed/step1_full_pairbest_p{2,4}_s{1..5}.parquet`
* `data/processed/step1_1k_pairbest_p4sets_s{1..5}.parquet`
* `data/processed/d2v_residual/{p2,p4}/…`（OOF baseline、修正权重、信号）

**结果目录**：

* `results/first2_aux/`
* `results/audio_newset/`
* `results/d2v_residual/`

---

## 6. 已确认与仍需确认

### 已确认（本轮）

1. **Q1 评价口径**：历史最佳 rank-NLL 是主指标；first2 只作训练辅助。**另加相关性系数**：
   Pearson/Spearman/逐玩家 Spearman，与 ΔNLL 并列报。
2. **Q2 数据集**：1k / 10k / randommerge / full **四个数据集分别跑**，不设主数据集，
   不合并 NLL。
3. **Q3 辅助损失范围**：**A1（全部 pair）作字面实现**；A2/A3 只作敏感性；first2 ACC ≤0.90
   的 5.5% 保留、不截断。最终看 A1 相对 A0 的 ΔNLL 与相关性是否提升。
4. **Q4/Q5 VSRG**：指仓库里的 VSRG Dan Benchmark；作为 second gate。
5. **Q6 Experiment 2 训练侧**：只算 6×6 监督训练图，不含 6×4/4×6 的 dropped 行；
   ceiling 仍可用全部行。
6. **Q7 音频数据边界**：randommerge/full 音频**不补**（网络/磁盘限制）；Experiment 2
   只汇报 1k 的 6,819 图结果，不外推到 randommerge/full。
7. **Q8 Experiment 3**：`mlp_extra` 与 `linear_extra` **同等对待**；P2-holdout 与
   P4-holdout **同等对待**，不预设主次。
8. **执行**：所有 CPU 任务提交 HPC 并行；本地只改代码、读结果。

### 仍需确认（不阻塞，先按默认）

* **Q9 set-P4 图数平衡**：popularity 分层内按图数贪心 60/40；分配图数 59.3/40.7，
  实际有行图 63/37 需要修。默认。
* **Q10 近分阈值**：训练集 bottom 25% 的 abs(Δz)；10%/33% 敏感性。默认。
* **Q11 长押定义**：`ln_ratio ≥ 0.15` 至少一张图；`patt_mode_tag ∈ {Mix,LN}` 为次；
  both-LN 样本够就报。默认。
* **Q12 四列编码器门槛**：近分 Δ ≤ −0.005 + 对照为 0 才投入；整体增益是加分项。
  默认。

Q9–Q12 不改，我就按这些默认实现。

---

## 附录 A：预实验复现要点

### A.1 first2 标签

四个数据集用同一实现，只换 `--step0` / `--pairs`。以 randommerge 为例：

1. 读 `step0_randommerge.parquet`；
2. 先删 `player_id` 记录数 < 20 的玩家；
3. 用 `scripts/step1_pairbest.py::assign_epochs(df, 365.0)` 做空窗拆分；
4. 按 `(player_id, beatmap_id, timestamp, playcount_cur)` 排序，取每组前 2 条，
   `min(loss)` 即 `loss_first2`；
5. 与 `step1_randommerge_c_p4_s1.parquet` 按 `(player_id, beatmap_id)` 合并；
6. 只在 `split=='train'`、每玩家 ≥10 张训练图的段内，枚举 chart 对，统计
   “两种标签都无 tie 且顺序相反”的比例。

复现数字：9,415,648 / 1,623,767 / 17.245%；`n_plays` 分桶见 §0.4a。
1k / 10k / full 的 step0 分别是 `step0_1k.csv` / `step0_10k.parquet` /
`step0_full.parquet`；同一脚本在 HPC 上按数据集拆作业。

### A.2 新 set 子集

关键点：用 `beatmap_meta_1k.csv` 的 `beatmapset_id`；train charts 的 set 集合；
test pair 要求两张图的 set 都不在 train_sets；在子集内用 `rank_metrics.build_pair_table`
建 pair 表，分别对 audio/legacy 信号拟合 link，按 `player_id_base` bootstrap。

### A.3 整 set P4 模拟

关键点：set popularity = set 内 distinct `player_id_base`；玩家侧 `_stratified` 60/40；
set 侧先按 popularity 分层，再按图数平衡。模拟 s1 得到分配图 4,045/2,774、实际有行图
3,509/2,058、924,384 test 比较、跨 set 98.8%。

---

## 附录 B：相关文档与代码

* 指标：[docs/metrics.md](../metrics.md)
* P2/P4 协议：[docs/p4_multiseed.md](../p4_multiseed.md)
* audio 实验：[docs/p4_multiseed_audio.md](../p4_multiseed_audio.md)
* 练习 vs 时间：[legacy/docs/archive/practice_vs_time.md](../../legacy/docs/archive/practice_vs_time.md)
* D2V：[data2vec_main/README.md](../../data2vec_main/README.md)、
  [docs/experiments.md](../experiments.md) §2.5
* 列 token 探针：[results/cm3p_column_probe/README.md](../../results/cm3p_column_probe/README.md)
* 现有 runner：`scripts/step2_cm3p_rank.py`、`scripts/run_p2_p4_multiseed.py`、
  `scripts/step2_rank_eval.py`、`scripts/paired_delta_nll.py`、
  `scripts/step1_p4_split.py`
