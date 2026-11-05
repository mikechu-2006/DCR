# 实验：P2 / P4 多划分种子（新 chart vs 新玩家 + 新 chart）

本文件记录 **P2（新 chart）** 与 **P4（新玩家 + 新 chart，双重冷启动）** 在
5 个 **train/test 划分种子** 上的重复实验。P2/P4 的绝对 NLL 不可互比（test 集、
ceiling、pair 集都不同）；这里比较的是同一口径内的方法与跨种子的稳定性。

模型初始化和训练种子固定为 0；5 个种子只改变 **train/test 划分**。

---

## 0. 摘要

5 个 **train/test 划分种子**（1–5）上，7 个方法在 P2 和 P4 上的 test NLL / Pearson /
Spearman 见 §5。核心：

* **CM3P + external MLP 在两个口径上都是最好**：P2 **0.5660 ± 0.0031**、P4
  **0.5734 ± 0.0043**；比 CM3P + external 线性好 **-0.0118 / -0.0114 nats**
  （5/5 个种子为负）。
* **CM3P-only 的 MLP 仍没有增益**：P2 0.6166 vs linear 0.6162、P4 0.6254 vs 0.6241；
  Δ 的符号在种子上是混合的。**能跨新图/新玩家泛化的是 external 特征 + 非线性**，
  不是 CM3P-only 的非线性容量。
* 旧 P2 表里「external 线性最好」是因为当时没有跑 external MLP；本实验补上了它，
  并把它放到 5 个划分种子上复核。P2 的 5-seed 均值与已发布的单划分数字在 ~1σ 内
  （external 线性 0.5779 vs 0.5756，CM3P 线性 0.6162 vs 0.6148）。
* 双重冷启动 P4 只比 P2 差 **~0.007–0.009 nats**（内容模型；fixed 信号幅度
  0.001–0.003，star 甚至略好），方法排序不变。但 P2/P4 的 test 集、ceiling 和 pair 数
  不同，**绝对 NLL 不可跨口径比较**。
* Pearson/Spearman 与 NLL 同序：external MLP 最高（P2 0.628 / 0.585；P4 0.588 / 0.543），
  fixed 信号最低（~0.28–0.35 / 0.31–0.38）；P4 的相关 std 更大（test block 更小、`b` 更噪）。

## 1. 两个口径

### 1.1 P2：新 chart（pair-random 划分重采样）

- 输入是 step1/pairbest 的 109,706 个 (玩家段, 谱面) 对。
- step1 的 split 是清洗的**最后一步**（[docs/data.md](data.md) §3.1），因此用
  `scripts/step1_pairbest_resplit.py` 在幸存对上重新做 pair-random 70/30，
  等价于用新的 `--split-seed` 重跑 step1/pairbest。
- 每个划分上重跑 P2：5 折 chart-disjoint（`--chart-folds 5 --chart-seed 0`），
  每张图由没看过该图的折模型打分。
- 模型 `--seed 0`；fixed 信号不训练。
- P2 的 5 个种子 = 5 个 pair-random split seed（1–5），原始表使用的 20260928 不包含在内。

### 1.2 P4：新玩家 + 新 chart（双重冷启动）

`scripts/step1_p4_split.py` 把玩家 cluster 和谱面**分别**分层抽样成 60/40
（各 5 个分位层，层内随机 60/40）：

~~~text
players: 60/40, 按每个 cluster 的 pair 行数（活跃度）分层
charts : 60/40, 按每张图的 distinct cluster 数（流行度）分层
train  : train players x train charts        (6 x 6)
test   : test players  x test charts         (4 x 4)
dropped: 6x4 / 4x6 两个 off-diagonal block   （不参与训练）
~~~

- 模型只在 6×6 的行上拟合；test 行（4×4）从不参与训练、早停或超参数选择。
- off-diagonal 行保留在 parquet 里，仅用于拟合 BT ceiling 的 `b`（ceiling 是参照，
  不是模型；见 §4）。模型脚本只读 `split == "train"`；信号评测只读 `split == "test"`，
  但 ceiling 的 `b` 可以用全部行（这正是 off-diagonal 的用途）。
- 模型 `--seed 0`；5 个种子 = 5 个 player/chart 分层划分种子（1–5）。
- P4 的每个种子同时改变玩家和谱面两侧，因此 fixed 信号（Sunny/Daniel/star）的
  test pair 集与 `b` 也随种子变化。

### 1.3 种子语义

| 量 | P2 | P4 |
|---|---|---|
| 5 个种子控制 | pair-random 70/30 划分 | 玩家 60/40 × 谱面 60/40 分层划分 |
| 模型初始化/训练种子 | 固定 0 | 固定 0 |
| chart fold seed | 固定 0（5 折） | —（没有 chart fold） |
| ceiling 的玩家 cross-fit seed | 固定 0 | 固定 0 |

因此 mean/std 反映的是**划分不确定性**，不是模型初始化噪声。固定模型种子让
方法之间的差异不被训练噪声掩盖；代价是 §5 的 std 不能外推为「换模型种子也会这么大」。

---

## 2. 数据与划分

### 2.1 P2 的 5 个划分

| seed | train rows | test rows | test charts | actual train frac |
|---|---|---|---|---|
| 1 | 76,735 | 32,971 | 4,582 | 0.6995 |
| 2 | 76,791 | 32,915 | 4,578 | 0.7000 |
| 3 | 76,763 | 32,943 | 4,625 | 0.6997 |
| 4 | 76,761 | 32,945 | 4,592 | 0.6997 |
| 5 | 76,832 | 32,874 | 4,564 | 0.7003 |

原始表（split seed 20260928）是 76,567 / 33,139，4,565 张 test 图；新种子在
1% 容差内复现了 70/30。

### 2.2 P4 的 5 个划分

玩家 cluster：942；谱面：6,828。每个种子都是 565 train / 377 test cluster，
4,098 train / 2,730 test charts；具体行数：

| seed | train rows (6×6) | test rows (4×4) | dropped (6×4 + 4×6) |
|---|---|---|---|
| 1 | 39,825 | 17,569 | 52,312 |
| 2 | 40,017 | 17,307 | 52,382 |
| 3 | 38,949 | 17,936 | 52,821 |
| 4 | 39,718 | 17,582 | 52,406 |
| 5 | 39,665 | 17,404 | 52,637 |

4×4 test 只占全部行的 ~16%，所以 P4 的 test pair 更少、ceiling 更噪；这是双重
冷启动的代价，不是实现差异。

---

## 3. 方法

| 方法 | 训练 | 配置 |
|---|---|---|
| CM3P + external 线性 | `linear_extra` | λ=1e-3；S = [msd_overall, sunny_star] |
| CM3P + external MLP | `mlp_extra` | h64、2 层、dropout 0.1、98 epoch、λ=1e-2 初始化；输入 [C,S] |
| CM3P 线性 (λ1e-3) | `linear` | λ=1e-3 |
| CM3P MLP h64 L2 | `mlp` | h64、2 层、dropout 0.1、98 epoch、λ=1e-2 初始化 |
| Sunny | fixed | `sunny_star`（osummania_map_analyser） |
| Daniel | fixed | `daniel_star` |
| star | fixed | `star`（osu! star rating） |

- 所有模型都是 content-only（谱面内容 + 可选的两个外部内容信号），没有玩家参数。
- 外部信号在 **train charts** 上 z-score；超参数不调（linear 的 λ 固定，MLP 固定
  98 epoch、无 early stop），以保证 P2/P4、种子之间的可比性。
- 所有训练/评测都在 1k pairbest 上，CPU、多线程。

---

## 4. 指标与 ceiling

- **test NLL**：`step2_rank_eval.py` 的 `nll_per_instance`，`--strict-pairs`，即
  与 ceiling 覆盖的 pair 同集。
- **Pearson(d,b) / Spearman(d,b)**：逐谱面信号 `d` 与 `b_full` 的相关（JSON 的
  `pearson_bhat` / `rho_bhat`）。`b_full` 是 `--ceiling-fit all` 的自由 per-chart
  BT MLE，用全部行拟合。
- **ceiling**：按 `player_id_base` 2 折 cross-fit 的无约束 BT；每折用 fold-train 玩家
  的全部行拟合 `b`，只在当前 split 的 fold-test 玩家 pair 上算 NLL，只保留两张图都在
  train 折出现过的 pair。P2 与 P4 都用 `--ceiling-fit all`。
- **P4 的 ceiling**：test chart 的行只存在于 4×4（test players）和 6×4（train players
  × test charts）两个 block。ceiling 使用全部行里 fold-train 玩家的行，因此对 test
  chart 的 `b` 来说，4×4 的 fold-test 玩家被 cross-fit 排除，6×4 的训练玩家和另一半
  test 玩家可以贡献 `b`。**模型从不见 6×4/4×6，ceiling 见；这是 ceiling 作为参照的
  定义，不是模型泄漏。**
- **captured** 仍在 `summary.csv` 里给出，但它不是本次的主指标；P2/P4 的 ceiling
  覆盖、test 集不同，captured 的绝对值不可跨口径比较。
- **mean/std**：对 5 个划分种子取均值与样本标准差（ddof=1）。P2 的 fixed 信号也随
  划分种子变化（test pair 集不同），所以 std 不再是 0。

---

## 5. 结果

### 5.1 P2（5 个 pair-random 划分种子）

每个 seed 内，7 个方法共享同一 test pair 集与 ceiling；表里是 5 个 seed 的
mean ± std（ddof=1，n=5）。ceiling NLL = **0.56126 ± 0.00348**。

| 方法 | test NLL | Pearson(d,b) | Spearman(d,b) |
|---|---|---|---|
| **CM3P + external MLP** | **0.56602 ± 0.00307** | **0.6281 ± 0.0081** | **0.5853 ± 0.0101** |
| CM3P + external 线性 | 0.57786 ± 0.00335 | 0.5822 ± 0.0122 | 0.5523 ± 0.0086 |
| CM3P 线性 (λ1e-3) | 0.61618 ± 0.00293 | 0.5412 ± 0.0082 | 0.5070 ± 0.0079 |
| CM3P MLP h64 L2 | 0.61664 ± 0.00265 | 0.5475 ± 0.0073 | 0.5146 ± 0.0084 |
| Sunny | 0.62868 ± 0.00207 | 0.3455 ± 0.0081 | 0.3834 ± 0.0019 |
| Daniel | 0.62768 ± 0.00212 | 0.3454 ± 0.0080 | 0.3837 ± 0.0031 |
| star | 0.66153 ± 0.00137 | 0.3088 ± 0.0074 | 0.3369 ± 0.0021 |

captured 均值（仅参考）：external MLP 0.964、external 线性 0.874、CM3P 线性 0.584、
CM3P MLP 0.580、Sunny 0.489、Daniel 0.496、star 0.240。5 个种子里 P2 的 captured
都 <1（最大 0.990）。

### 5.2 P4（5 个 6×6 / 4×4 划分种子）

test 是 4×4 block；ceiling NLL = **0.56540 ± 0.00859**。

| 方法 | test NLL | Pearson(d,b) | Spearman(d,b) |
|---|---|---|---|
| **CM3P + external MLP** | **0.57343 ± 0.00433** | **0.5881 ± 0.0264** | **0.5426 ± 0.0232** |
| CM3P + external 线性 | 0.58479 ± 0.00494 | 0.5374 ± 0.0391 | 0.5094 ± 0.0326 |
| CM3P 线性 (λ1e-3) | 0.62411 ± 0.00345 | 0.5095 ± 0.0219 | 0.4767 ± 0.0211 |
| CM3P MLP h64 L2 | 0.62540 ± 0.00653 | 0.5116 ± 0.0225 | 0.4763 ± 0.0325 |
| Sunny | 0.63111 ± 0.00777 | 0.3098 ± 0.0608 | 0.3480 ± 0.0492 |
| Daniel | 0.63102 ± 0.00601 | 0.3077 ± 0.0614 | 0.3440 ± 0.0497 |
| star | 0.66021 ± 0.00779 | 0.2813 ± 0.0626 | 0.3090 ± 0.0509 |

captured 均值：external MLP 0.940（有一个 seed 1.0175，见 §6.2）、external 线性 0.850、
CM3P 线性 0.544、CM3P MLP 0.534、Sunny 0.488、Daniel 0.488、star 0.260。

### 5.3 配对 ΔNLL（同一划分种子）

负值 = A 更好；括号里是"A 更好的种子数 / 5"。

| A : B | P2 ΔNLL | P4 ΔNLL |
|---|---|---|
| external MLP : external 线性 | **-0.01183 ± 0.00051**（5/5） | **-0.01137 ± 0.00540**（5/5） |
| external 线性 : CM3P 线性 | **-0.03832 ± 0.00098**（5/5） | **-0.03932 ± 0.00741**（5/5） |
| CM3P MLP : CM3P 线性 | +0.00046 ± 0.00190（3/5） | +0.00128 ± 0.00480（2/5） |
| external MLP : Sunny | **-0.06266 ± 0.00262**（5/5） | **-0.05769 ± 0.00787**（5/5） |

### 5.4 与已发布 P2 单划分的对照

已发布 P2 用的是 split seed 20260928、5 折 chart-disjoint、单模型种子；本实验的
5 个新划分种子的均值与它在 ~1σ 内（external MLP 当时没有正式的 P2 数字）：

| 方法 | 已发布 P2（seed 20260928） | 本实验 5-seed mean ± std |
|---|---|---|
| CM3P + external 线性 | 0.57563 | 0.57786 ± 0.00335 |
| CM3P 线性 (λ1e-3) | 0.61484 | 0.61618 ± 0.00293 |
| CM3P MLP h64 L2 | 0.61522 | 0.61664 ± 0.00265 |
| Sunny | 0.62718 | 0.62868 ± 0.00207 |
| Daniel | 0.62595 | 0.62768 ± 0.00212 |

---

## 6. 讨论与局限

### 6.1 主要发现

1. **external MLP 是唯一在两个口径上都稳定胜出的方法**。P2 0.5660、P4 0.5734；
   对 external 线性的 paired Δ 分别为 -0.0118 / -0.0114 nats，5/5 个划分种子为负。
   这说明 **external 特征 + 非线性**能跨新 chart、跨新玩家泛化；P1 上 external MLP
   对 external 线性的优势（val 0.5476 vs 0.5767）不是 seen-chart 的假象。
2. **CM3P-only 的 MLP 仍然没有增益**。P2 0.6166 vs 0.6162、P4 0.6254 vs 0.6241，
   Δ 的符号在种子上混合。非线性本身不能把纯 CM3P 内容推广到新图；旧结论只对
   CM3P-only 成立，不能扩展到 external MLP。
3. **P4 的双重冷启动只贵了 ~0.007–0.009 nats**（external MLP +0.0074、external 线性
   +0.0069、CM3P 线性 +0.0079、CM3P MLP +0.0088）；fixed 信号的差异只有 0.001–0.003
   （Sunny +0.0024、Daniel +0.0033、star -0.0013）。
   方法排序在两个口径上完全一致。但 P4 的 test 集更小（~16% 行）、ceiling 更高，
   **绝对 NLL 不可跨口径比较**，只能看同一口径内的排序和配对差。
4. **external 特征的价值远大于非线性**：external 线性比 CM3P 线性好 ~0.039 nats
   （两个口径都是 5/5 负）；CM3P-only MLP 相对线性 ~0。也就是说，跨图/跨玩家泛化的
   主力是 MSD+Sunny 这两个外部内容信号，CM3P 提供互补的 content 信息。
5. **Pearson/Spearman 与 NLL 同序**。external MLP 最高（P2 0.628/0.585，P4 0.588/0.543），
   fixed 信号最低。训练模型的 Pearson > Spearman（linear link 拟合 BT 似然），fixed
   信号相反（Sunny P2 0.345 < 0.383）；与 [docs/cm3p_rank.md](cm3p_rank.md) §5.3 的
   解释一致。
6. **fixed 基线**：Sunny/Daniel ~0.631、star ~0.660；external MLP 比 Sunny 好
   ~0.058–0.063 nats。CM3P 线性只比 Sunny 好 0.007–0.012 nats，external 特征才是
   主要增量。

### 6.2 这些数字不能怎么读

* **P2 与 P4 的 NLL / Pearson / Spearman 不可直接比**：test 行、谱面集、pair 数、
  ceiling 都不同。P4 只改的是划分协议；跨口径只能看"方法排序是否稳定"。
* **P4 的一个 seed 上 external MLP 的 captured = 1.0175 > 1**：P4 的 test block 只有
  ~16% 行，ceiling 估计更噪/欠拟合；captured 不是硬上界，主表用 NLL，captured 只
  放在 summary.csv 里参考。
* **模型种子固定为 0**：5 个种子的 std 是**划分方差**，不是训练方差。换模型初始化
  还会额外增加方差；不要把它当成总不确定性。
* **MLP 没有 early stop**：P2/P4 都用固定 98 epoch、无验证集，以保持协议可比。P4 的
  6×6 只有 ~40k 行，MLP 可能过拟合；P4 的 CM3P-only MLP 比线性略差可能就是因此。
  这不是"MLP 一定不行"，而是"在这个固定预算下不行"。
* **P4 的分层变量是活跃度/流行度**，不是难度；它保证 train/test 的玩家活跃度和谱面
  流行度分布接近，但不保证难度分布完全一致。
* **1k top-1000 切片**：结论不外推到全玩家/全谱面；ACC 截断、n_plays≥2 仍在。
* **external 是 stacking**：external MLP/线性 = CM3P + MSD/Sunny；0.566/0.573 是组合
  的功劳，CM3P 的净贡献看 CM3P-only 两行。
* **测试集只有 5 个划分种子**：mean/std 是这 5 个划分的经验值，不是总体置信区间；
  要做正式检验，用同一划分内的 paired Δ（§5.3）或增加种子。

---

## 7. 复现

~~~bash
# 1) P2：5 个新的 pair-random 70/30 划分
python scripts/step1_pairbest_resplit.py \
    --in data/processed/step1_1k_pairbest.parquet \
    --out-prefix data/processed/step1_1k_pairbest_p2 \
    --seeds 1,2,3,4,5

# 2) P4：5 个分层 6:4 玩家 × 6:4 谱面划分
python scripts/step1_p4_split.py \
    --in data/processed/step1_1k_pairbest.parquet \
    --out-prefix data/processed/step1_1k_pairbest_p4 \
    --seeds 1,2,3,4,5

# 3) 训练 + 官方评测 + 汇总（单进程 6 线程约 1.5–2 h；两个 4 线程进程约 1 h wall）
python scripts/run_p2_p4_multiseed.py --protocols p2,p4 --seeds 1,2,3,4,5 --threads 6

# 或分两个进程
python scripts/run_p2_p4_multiseed.py --protocols p4,p2 --seeds 1,3,5 --threads 4
python scripts/run_p2_p4_multiseed.py --protocols p4,p2 --seeds 2,4   --threads 4
python scripts/run_p2_p4_multiseed.py --aggregate-only --seeds 1,2,3,4,5
~~~

单种子/烟测：

~~~bash
python scripts/run_p2_p4_multiseed.py --quick --seeds 99
~~~

---

## 8. 产物

| 文件 | 内容 |
|---|---|
| `results/p4_multiseed/summary.csv` | 逐 seed × 方法的 NLL / Pearson / Spearman / captured / ceiling |
| `results/p4_multiseed/summary_mean_std.csv` | 5 seed 的 mean/std |
| `results/p4_multiseed/summary.md` | §5 表格的生成版本 |
| `results/p4_multiseed/p2/s<seed>/rank_eval.json` | P2 官方评测（strict，7 个信号） |
| `results/p4_multiseed/p4/s<seed>/rank_eval.json` | P4 官方评测 |
| `results/p4_multiseed/<p>/s<seed>/signals/<name>.json` | 每个模型/种子的训练元数据（配置、耗时、train/test NLL） |
| `results/p4_multiseed/<p>/s<seed>/signals.parquet` | 7 个信号合并表（可重建；parquet 不入库） |
| `data/processed/step1_1k_pairbest_p2_s<seed>.parquet` | P2 重采样划分（可重建） |
| `data/processed/step1_1k_pairbest_p4_s<seed>.parquet` | P4 划分（可重建） |
