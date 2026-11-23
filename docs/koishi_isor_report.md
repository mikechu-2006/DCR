# Koishi / ISOR 在 DCR 10K 与 randommerge 上的 rank-NLL 与相关性

> **状态：结果（2026-11-20）。** 执行口径见 [method_overfit_plan.md](method_overfit_plan.md)。
> Benchmark 快照 0bdbe23；DCR 数据与 rank 管线同 docs/metrics.md。
> 本轮没有外部 Dan 留出集，结论只到 DCR 的 ACC 序，不能证明“对 Dan 标签过拟合”。

---

## 0. 先回答：0.549 不是 10K 全量

上一条消息里的 **0.54964 是 Koishi-native 子集**，不是 10K 全量：

| 表 | 图 | 行 | pairs | Sunny NLL | 说明 |
|---|---:|---:|---:|---:|---|
| **10K ALL** | 16,061 | 1,513,746 | 225.64M | **0.606729** | 全部 4K（含 LN） |
| **10K RC20** | 9,928 | 1,001,538 | 99.35M | **0.617203** | LN≤0.18、notes≥20 |
| **10K LN** | 6,133 | 512,208 | 30.36M | 0.574111 | LN 子集 |
| **10K Koishi-native** | **1,056** | 323,675 | 13.94M | 0.549636 | Koishi 只覆盖 11–17 的图 |
| **RM ALL** | 20,388 | 453,406 | 39.65M | 0.605102 | randommerge 全部 4K |
| **RM RC20** | 12,563 | 343,715 | 20.96M | 0.609464 | randommerge RC |

results/p4_multiseed/summary.md 是 **1k**（6,828 图 / 109,706 行）的 P2/P4，
不是 10K；1k 上 Sunny 0.6287 与 10K 全量 0.6067 不可直接比——图集、玩家总体、
pair 集和 ceiling 都不同。

**为什么子集 NLL 更低**：Koishi-native 只有 1,056 张高难图，pair 集是 13.94M；
它的难度分布窄、玩家更集中于高难，NLL 自然低于包含大量简单图和更多玩家的 10K 全量。
表内比较看 ΔNLL，不跨图集比绝对值。

---

## 1. 口径

| 项 | 内容 |
|---|---|
| 方法 | ISOR v3.1（10K + RM）、Koishi v5.1（仅 10K）、Sunny（基线） |
| 10K | step1_10k_pairbest.parquet：1,513,746 行 / 16,061 图 / 225.64M pairs |
| randommerge | step1_randommerge_pairbest_content.parquet：453,406 行 / 20,388 图 / 39.65M pairs |
| RC 口径 | ln_ratio ≤ 0.18；RC20 = notes≥20（ISOR native），RC80 = notes≥80（两者） |
| ALL 口径 | 全部 4K；**LN 参与** |
| ISOR 信号 | ISOR v3.1 + DCR MinaCalc 505 注入 MSD（不是 benchmark 原 runner） |
| rank 管线 | step2_rank_eval.py --split all --ceiling-fit all --folds 1 --bootstrap 0 --min-l 10 |
| 相关性 | b_full 是 in-sample BT，仅作相关性诊断；NLL 不受 ceiling 影响；captured 不报 |
| 重叠图 | 不排除；10K 205 图 / 97,321 行（6.43%），RM 205 图 / 6,375 行（1.41%） |

**ISOR 在 LN 上的输出是 out-of-domain**：ISOR 文档只支持 4K rice（LN≤0.18），
但 analyze_beatmap_isor 对 LN>0.18 仍会返回数值。ALL 表的 LN 部分只作
coverage/鲁棒性，不能当作“ISOR 会评 LN”。

---

## 2. 结果

### 2.1 10K 全量 / RC / LN：Sunny vs ISOR

| 图集 | method | n_maps | n_rows | n_pairs | NLL | ΔNLL vs Sunny | Pearson(d,b) | Spearman(d,b) | lw ρ | macro ρ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **ALL** | Sunny | 16,061 | 1,513,746 | 225.64M | **0.606729** | — | 0.343 | 0.291 | 0.550 | 0.551 |
| **ALL** | ISOR | 16,061 | 1,513,746 | 225.64M | **0.605273** | **−0.001456** | 0.413 | 0.284 | 0.530 | 0.526 |
| **RC20** | Sunny | 9,928 | 1,001,538 | 99.35M | **0.617203** | — | 0.384 | 0.302 | 0.533 | 0.538 |
| **RC20** | ISOR | 9,928 | 1,001,538 | 99.35M | **0.614592** | **−0.002611** | 0.445 | 0.297 | 0.527 | 0.533 |
| **LN** | Sunny | 6,133 | 512,208 | 30.36M | **0.574111** | — | 0.308 | 0.289 | 0.604 | 0.592 |
| **LN** | ISOR | 6,133 | 512,208 | 30.36M | **0.578572** | **+0.004461** | 0.377 | 0.289 | 0.549 | 0.518 |

读法：

* **10K 全量：ISOR 的 NLL 低 0.00146 nats，但 Spearman/lw 也更低**（0.284 vs 0.291；
  0.530 vs 0.550），Pearson 更高（0.413 vs 0.343）。这是“NLL 略好、序相关没有更好”的
  组合，不能说是全面胜出。
* **RC20：ISOR NLL 低 0.00261，方向一致**；但 Spearman/lw 仍略低（0.297 vs 0.302；
  0.527 vs 0.533）。差异很小，不能把 −0.0026 说成实质提升。
* **LN：ISOR 差 0.00446**，符合 LN out-of-domain；ALL 的轻微 NLL 优势不是 LN 带来的。
* Pearson 对 Dan/star 尺度非线性敏感，ISOR 的 Pearson 更高不代表序更好；主看 NLL/Δ 与
  Spearman/lw。

### 2.2 10K Koishi-native：Sunny vs Koishi vs ISOR

1,056 图 / 323,675 行 / 13.94M pairs；**所有 Koishi-valid 图都在 HARD-RC 内**，
所以 hardrc_common 与 koishi_native 是同一 pair 集（NLL 相同，ceiling 因 --in 不同
而略不同；下表用 koishi_native 的 ceiling）。

| method | NLL | ΔNLL vs Sunny | Pearson(d,b) | Spearman(d,b) | lw ρ | macro ρ |
|---|---:|---:|---:|---:|---:|---:|
| **Koishi** | **0.545176** | **−0.004460** | 0.903 | **0.919** | **0.619** | **0.592** |
| Sunny | 0.549636 | — | 0.906 | 0.903 | 0.589 | 0.553 |
| ISOR | 0.561139 | +0.011504 | 0.875 | 0.896 | 0.578 | 0.537 |

读法：

* **Koishi 在自己的高难子集上赢了**：NLL −0.00446，lw +0.030，Spearman +0.016；
  Pearson 略低（0.903 vs 0.906）。
* 但覆盖只有 **1,056 / 16,061 图（6.57%）**、**323,675 / 1,513,746 行（21.38%）**、
  **13.94M / 225.64M pairs（6.18%）**；不能把它当作覆盖全 10K 的方法。
* ISOR 在同一高难子集上比 Sunny 差 0.01150，比 Koishi 差 0.01596。

### 2.3 randommerge：Sunny vs ISOR

| 图集 | method | n_maps | n_rows | n_pairs | NLL | ΔNLL vs Sunny | Pearson(d,b) | Spearman(d,b) | lw ρ | macro ρ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **ALL** | Sunny | 20,388 | 453,406 | 39.65M | **0.605102** | — | 0.838 | 0.836 | 0.547 | 0.550 |
| **ALL** | ISOR | 20,388 | 453,406 | 39.65M | **0.620411** | **+0.015309** | 0.801 | 0.803 | 0.483 | 0.474 |
| **RC20** | Sunny | 12,563 | 343,715 | 20.96M | **0.609464** | — | 0.854 | 0.847 | 0.542 | 0.550 |
| **RC20** | ISOR | 12,563 | 343,715 | 20.96M | **0.617070** | **+0.007606** | 0.842 | 0.837 | 0.509 | 0.507 |
| **LN** | Sunny | 7,824 | 109,689 | 3.78M | **0.593012** | — | 0.791 | 0.796 | 0.533 | 0.510 |
| **LN** | ISOR | 7,824 | 109,689 | 3.78M | **0.611908** | **+0.018895** | 0.745 | 0.761 | 0.454 | 0.425 |

读法：

* **randommerge 上 ISOR 全面差于 Sunny**：ALL +0.01531、RC20 +0.00761、LN +0.01890；
  Pearson/Spearman/lw 三项也一致更低。
* 这不是 LN 单一原因：**RC20 也差 0.00761**，所以是 broader player/chart 分布上的
  迁移失败，而不是“LN out-of-domain”一个因素。
* 与 10K 的 RC20 −0.00261 对比：同一个 ISOR，在 10K 略好、在 RM 明显差；
  说明其优势对数据集/玩家总体敏感。

---

## 3. 综合读法

1. **Koishi 的高难优势有方向性迁移，但幅度很小、覆盖很窄。**
   benchmark RC11–17 MAE 0.151（Sunny 0.526），DCR 10K 高难子集 ΔNLL 只有 −0.00446；
   lw 提升 +0.030 更明显。它仍是“高难专用”，不是全数据集解法。

2. **ISOR 的“全 RC 覆盖”不等于“更准”。**
   10K 全量/RC20 的 NLL 略低（−0.0015/−0.0026），但 Spearman/lw 没有更好；
   randommerge 上 ALL/RC/LN 全面更差（+0.0076 … +0.0153）。覆盖率 100% 是事实，
   但 supported RC 的排序优势不稳定。

3. **benchmark 的 Dan-MAE 优势没有等比例转成 DCR 的 ACC 序优势。**
   benchmark 上 Koishi/ISOR 比 Sunny 好 0.30–0.37 MAE，DCR 上 ΔNLL 只有 0.001–0.004
   量级；RM 上 ISOR 还反向。这不是“谁抄谁”的问题，而是 Dan 标签与 ACC 序两个 construct
   的差异；没有外部 Dan 数据，不能进一步断言过拟合。

4. **0.5496 与 1k 的 0.6287 不可比。**
   前者是 1,056 图高难子集，后者是 1k 全量；图集、pair 集、ceiling 不同。
   10K 全量 Sunny 是 0.606729，位于两者之间；这是数据集差异，不是代码错误。

---

## 4. 限制

1. **没有 benchmark 之外的 Dan 人工标注**：只能回答“DCR ACC 序上谁更好”，不能证明
   “对 benchmark 的 Dan 标签过拟合”。
2. **ISOR 信号是 ISOR v3.1 + DCR MinaCalc 505**，不是 benchmark 原 runner；Ankoku
   Butoukai 上 14.77 vs benchmark 14.27，差异来自 MSD 版本。绝对 NLL 会受此影响，
   但同一信号在 10K/RM 内部可比。
3. **ISOR 在 LN 上是 out-of-domain**：ALL/LN 表的 LN 部分只作 coverage/鲁棒性。
4. **--folds 1**：b_full 是 in-sample，仅用于 Pearson/Spearman；NLL 与 ΔNLL 不受
   ceiling 影响。captured 未报，避免把 in-sample ceiling 当 cross-fitted。
5. **Koishi 只测 10K**：randommerge 未测；其 10K 覆盖只有 6.57% 图 / 21.38% 行。
6. **不排除 benchmark 重叠图**：10K 6.43% 行、RM 1.41% 行；未单独筛。
7. **无 bootstrap CI**：ΔNLL 是点估计；0.001–0.004 的差异可能落在噪声内，报告按
   “量级很小/方向一致/不一致”描述，不写成显著胜负。

---

## 5. 结论

* **Koishi**：在 10K 的 1,056 张高难图上，NLL 和逐玩家排序都略好于 Sunny
  （ΔNLL −0.00446，lw +0.030）；但覆盖只有 6.57% 图 / 21.38% 行，**不能作为全数据集方法**。
* **ISOR**：覆盖率最高（10K/RM 全图），但 **10K 只是微弱 NLL 优势、排序相关没有更好；
  randommerge 全面更差**。在这份数据上，**没有证据支持用 ISOR 替换 Sunny**。
* **“更好”取决于子集与指标**：高难看 Koishi，全 RC 覆盖看 ISOR 的覆盖率而不是它的 NLL；
  两者都不是对 Sunny 的全面胜出。
* 若目标是“benchmark 上的 Dan MAE 更好”，本报告不能回答；需要 benchmark 之外的
  Dan 标注数据，本仓库没有。

---

## 6. 产物

| 路径 | 内容 |
|---|---|
| data/processed/signals_10k.parquet | 10K 宽表：sunny_star, isor_v31_dp, koishi_v5_dan |
| data/processed/signals_randommerge.parquet | RM 宽表 |
| data/processed/eval_subsets/step1_10k_*.parquet | ALL/RC20/RC80/HARD-RC/LN/KOISHI-NATIVE |
| data/processed/eval_subsets/step1_randommerge_*.parquet | ALL/RC20/RC80/HARD-RC/LN |
| results/ki_eval/10k/*.json | 10K 7 个 rank eval |
| results/ki_eval/rm/*.json | RM 3 个 rank eval |
| results/ki_eval/summary.csv / .md | 汇总表 |
| scripts/run_isor_signals.py / run_koishi_signals.mjs | 信号 runner |
| scripts/slurm/ki_eval.slurm | HPC 任务 |

---

## 7. 复现

~~~bash
# 1) 本地信号（已完成）
bash scripts/run_signals_local.sh
# 2) 组装 + 子集 + 同步 + 提交 HPC
bash scripts/run_ki_after_signals.sh
# 3) 等 HPC 完成后汇总
python3 scripts/summarize_ki.py --results 'results/ki_eval/*/*.json' \
  --out-csv results/ki_eval/summary.csv --out-md results/ki_eval/summary.md
~~~
