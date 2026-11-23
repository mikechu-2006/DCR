# Random 合并集：P2 / P4 五种子结果（2026-11-19）

> 数据集：4 个 random 快照合并、内容过滤后的 453,406 对 / 12,758 玩家 / 20,388 图。
> 口径：P2 = pair-random 重采样 + 5 折 chart-disjoint；P4 = 玩家×谱面双重冷启动。
> 内容表：**legacy（audio-off，512 维）** `chart_content_cm3p_local.parquet`。
> 模型种子固定 0；5 个种子只切分。**不加权。**

---

## 0. 摘要

1. **P2 和 P4 在合并集上没有差别**：external 线性 P2 0.58326、P4 0.58271，
   配对 Δ(P2−P4) = **+0.00054 ± 0.00529**，远小于种子噪声。
   1k 上 P4 比 P2 差 0.007–0.009 nats；合并集（12× 玩家、3× 谱面）上这个代价消失。
2. **MLP 叠 external 信号反而更差**（与 1k 相反）：
   mlp_extra − linear_extra = P2 **+0.00498 ± 0.00090**、P4 **+0.00907 ± 0.00347**，
   5/5 个 seed 同向；Pearson/Spearman 也从 0.89/0.88 掉到 0.86/0.84。
3. **只用 CM3P 时 MLP 更好**：mlp − linear = P2 **−0.00551 ± 0.00072**（5/5）、
   P4 −0.00185 ± 0.00302（4/5）。说明 external 信号已经把线性可提取的部分拿走，
   额外非线性容量在这份更杂的数据上是在拟合噪声。
4. **方法排序**：external 线性 > Sunny > CM3P 线性 > CM3P MLP > Daniel > star
   （P2；P4 里 CM3P MLP 略好于 CM3P 线性，但都远落后于 external 线性）。
5. **ceiling 在这个数据集上不是上界**：captured 1.10–1.23，先不要引用 captured；
   NLL / Pearson / Spearman 的方法间比较不受影响（同一 test 集、同一 ceiling）。

---

## 1. 设置

| 项 | 值 |
|---|---|
| 数据 | `data/processed/step1_randommerge_pairbest_content.parquet` |
| P2 split | `data/processed/step1_randommerge_c_p2_s{1..5}.parquet`（train ~317k / test ~136k） |
| P4 split | `data/processed/step1_randommerge_c_p4_s{1..5}.parquet`（6×6 train / 4×4 test） |
| 内容 | `chart_content_cm3p_local.parquet`（legacy, audio off, 512-d） |
| 外部信号 | `msd_overall`, `sunny_star`（在 train charts 上 z-score） |
| 模型 | `linear_extra` / `mlp_extra`（h64×2, dropout 0.1, λ1e-2, 98 epoch）/ `linear` / `mlp`；fixed: Sunny / Daniel / star |
| 评测 | `step2_rank_eval.py --split test --ceiling-fit all --common-maps --strict-pairs --folds 2` |
| 计算 | HPC `i64m512u`：线性 array 14021385、MLP array 14022821、eval 修复 14023289 |
| 结果 | `results/randommerge_p2p4_hpc/{p2,p4}/s{1..5}/rank_eval.json` |

**过程记录**：MLP array 训练完成后，`run_p2_p4_multiseed.py` 的 `evaluate()` 因为
`rank_eval.json` 已存在而跳过评测，且 `signals.parquet` 只保留最后一个模型的列。
已用 `scripts/slurm/randommerge_eval_fix.slurm` 合并 `signals/<model>.parquet` 后
按 7 个信号重跑 eval。后续波次必须带 `--force` 或先删 `rank_eval.json`。

---

## 2. P2（新 chart，5 seeds）

ceiling NLL = 0.59779 ± 0.00141；test 17,030–17,075 图 / 3.46–3.52M pairwise。

| 方法 | test NLL | Pearson(d,b) | Spearman(d,b) |
|---|---:|---:|---:|
| **CM3P + external 线性** | **0.58326 ± 0.00145** | 0.8896 | 0.8829 |
| CM3P + external MLP | 0.58824 ± 0.00153 | 0.8604 | 0.8398 |
| Sunny | 0.60448 ± 0.00126 | 0.8614 | 0.8601 |
| CM3P MLP h64 L2 | 0.61752 ± 0.00104 | 0.8374 | 0.8349 |
| CM3P 线性 (λ1e-3) | 0.62303 ± 0.00139 | 0.8390 | 0.8432 |
| Daniel | 0.63361 ± 0.00061 | 0.8172 | 0.7991 |
| star | 0.64494 ± 0.00061 | 0.8114 | 0.8047 |

## 3. P4（新玩家 + 新 chart，5 seeds）

ceiling NLL = 0.59769 ± 0.00743；test 7,216–7,334 图 / 2.15–2.89M pairwise。

| 方法 | test NLL | Pearson(d,b) | Spearman(d,b) |
|---|---:|---:|---:|
| **CM3P + external 线性** | **0.58271 ± 0.00514** | 0.8824 | 0.8756 |
| CM3P + external MLP | 0.59178 ± 0.00502 | 0.8411 | 0.8115 |
| Sunny | 0.60390 ± 0.00666 | 0.8547 | 0.8551 |
| CM3P MLP h64 L2 | 0.62243 ± 0.00217 | 0.8218 | 0.8167 |
| CM3P 线性 (λ1e-3) | 0.62428 ± 0.00264 | 0.8329 | 0.8349 |
| Daniel | 0.63294 ± 0.00535 | 0.8105 | 0.7929 |
| star | 0.64467 ± 0.00395 | 0.8054 | 0.7985 |

## 4. 配对差（逐 seed）

| 对比 | P2 | P4 |
|---|---:|---:|
| external 线性：P2 − P4 | +0.00054 ± 0.00529 | |
| **mlp_extra − linear_extra** | **+0.00498 ± 0.00090**（5/5 更差） | **+0.00907 ± 0.00347**（5/5 更差） |
| mlp − linear（CM3P-only） | −0.00551 ± 0.00072（5/5 更好） | −0.00185 ± 0.00302（4/5 更好） |

---

## 5. ceiling 异常（必须处理）

- 所有方法在两个口径下都**优于 ceiling**（NLL 更小），captured 1.10–1.23。
- 1k 上 captured 最高 ~0.47，ceiling 是真正的上界。
- 最可能的原因：`--folds 2` 的玩家 cross-fit 把每张图的 `b` 只拟合在一半玩家的行上，
  合并集里每图 test 行更少、休闲玩家噪声更大，自由 per-chart BT 的 MLE 不再是好的上界。
- 影响范围：NLL 的方法间比较**不受影响**（同一 test 集、同一 ceiling）；
  受影响的只有 `captured` 和「离 ceiling 还差多少」的叙事。

---

## 6. 与 1k 的对照

| 项 | 1k（`docs/p4_multiseed.md`） | 合并集 |
|---|---:|---:|
| external 线性 P2 | 0.5779 | 0.5833 |
| external 线性 P4 | 0.5848 | 0.5827 |
| P2 − P4 | +0.007 ~ +0.009 | **+0.0005** |
| external MLP − external 线性 | **−0.011** | **+0.005 / +0.009** |
| test 图（P2 / P4） | 4,565 / 2,730 | 17,050 / 7,250 |

读法：合并集让 **P4 追平 P2**（内容函数见到的图 4.1k → 12.2k）；
但 external MLP 的相对优势反转成劣势——更大的图集 + 更杂的玩家总体下，
线性头已经够用，MLP 的额外容量只在去掉 external 信号时才体现价值。

---

## 7. 待办

1. 修 ceiling 口径后再决定 captured 是否可引用。
2. 若要 audio-on 臂：需为合并集的另外 13.7k 张图补编码（HPC GPU 任务线），
   1k 上 audio-on 一致更好（ΔNLL −0.008…−0.022）。
3. CM3P 分段聚合（A0–A3）已移交另一个 session，规格见
   [randommerge_mlp_pooling_plan.md](../legacy/docs/plans/randommerge_mlp_pooling_plan.md)（已归档）§2。
