# DCR 评测 v3：ISOR（10K + randommerge）与 Koishi（10K），RC / LN-inclusive 两口径

> **状态：已完成（2026-11-20）。** 本文件是执行口径与评测矩阵；结果与结论见
> [koishi_isor_report.md](koishi_isor_report.md)。
>
> **Benchmark 快照**：[VSRG-DanEstimation-Benchmark](https://github.com/LeoBlackMT/VSRG-DanEstimation-Benchmark)
> commit `0bdbe23`（2026-11-19；`results/` 与 `e15f89b` 相同）。
>
> **按你的最新澄清**：ISOR 两个数据集都测；Koishi 只测 10K；LN 参与与否分两个口径；
> 不排除 benchmark 重叠图；不做外部 Dan 留出集（没有数据）。HPC 已确认可用。

---

## 0. 执行摘要

1. **方法**：**ISOR v3.1**（全 RC；benchmark RC11–17 MAE 0.222）与 **Koishi v5.1**
   （高难 11–17；benchmark RC11–17 MAE 0.151）；Sunny 作基线。
2. **数据集**：
   * **10K**：1,513,746 行 / 16,061 图 / 225.64M pairs；ISOR + Koishi 都测。
   * **randommerge**：453,406 行 / 20,388 图 / 39.65M pairs；**只测 ISOR**（Koishi 不测）。
3. **两个 LN 口径**：
   * **RC**：LN 排除（`ln_ratio ≤ 0.18`；ISOR native）；10K RC20 1,001,538 行 /
     9,928 图 / 99.35M pairs；RM RC20 343,715 行 / 12,563 图 / 20.96M pairs。
   * **ALL**：LN 参与（全部 4K 图）；10K 225.64M pairs，RM 39.65M pairs。
     ISOR 在 LN 图上会输出数值，但 **LN 是 ISOR 的 out-of-domain**；ALL 表只作
     coverage/鲁棒性，不声称“ISOR 会评 LN”。
4. **指标**：NLL、ΔNLL(vs Sunny)、Pearson/Spearman(d,b)、逐玩家 Spearman、coverage。
   用 `--folds 1` 的 in-sample `b_full` 只做相关性诊断；NLL 与 ΔNLL 不受 ceiling 影响；
   `captured` 不报（避免把 in-sample ceiling 当 cross-fitted）。
5. **HPC**：10 个 `step2_rank_eval` 任务，`i64m512u`，8 核 / 128G / 12h；先 1k smoke
   已在 HPC 跑通（Sunny NLL 0.63165，captured 0.4727；与本地 0.4721 差 0.0006，属 fold/数值差异）。
6. **结论边界**：没有 benchmark 之外的 Dan 人工标注，所以只能回答“在 DCR 的 ACC 序上谁更好”，
   不能证明“对 Dan 标签过拟合”。ISOR 在 LN 上的输出也不能当成 LN 难度结论。

---

## 1. 方法与 benchmark 位置

| 方法 | 版本 | 原生输出 | benchmark 总 MAE / 覆盖 | benchmark RC11–17 MAE / 覆盖 | 本轮数据集 |
|---|---|---|---:|---:|---|
| **ISOR** | v3.1 `ec6ab5b` | DP（Dan） | 0.267 / 1,280（92.6%） | 0.222 / 1,102（100%） | 10K + RM |
| **Koishi** | v5.1 `6828d61` | Dan 11–17 | 0.154 / 1,106（80.0%） | 0.151 / 1,084（98.4%） | 仅 10K |
| Sunny | `b6c1d89` | star | 0.576 / 1,363（98.6%） | 0.526 / 1,102（100%） | 10K + RM（基线） |

* ISOR 的“全覆盖”指 benchmark RC 1,246/1,246；不是 LN。
* Koishi 的 0.151 是 11–17 窄带成绩；10K 上 native 覆盖率会低，这是结果的一部分。
* 不把 benchmark MAE 与 DCR NLL 混为一谈；两者 construct 不同。

---

## 2. 数据与图集

### 2.1 规模

| 图集 | 10K 行 / 图 / pairs | randommerge 行 / 图 / pairs |
|---|---:|---:|
| **ALL** | 1,513,746 / 16,061 / **225.64M** | 453,406 / 20,388 / **39.65M** |
| **RC20**（LN≤0.18，notes≥20） | 1,001,538 / 9,928 / 99.35M | 343,715 / 12,563 / 20.96M |
| **RC80**（LN≤0.18，notes≥80） | 1,000,872 / 9,900 / 99.24M | 343,404 / 12,535 / 20.94M |
| **HARD-RC**（RC80 top 30% Sunny star） | 784,781 / 2,970 / 59.83M | 128,110 / 3,761 / 6.91M |
| **LN**（ln_ratio>0.18） | 512,208 / 6,133 / 30.36M | 109,689 / 7,824 / 3.78M |

* 10K HARD-RC 阈值 **Sunny star 5.0318**；RM 阈值 **4.5834**；先固定，再看方法结果。
* benchmark 重叠：10K 205 图 / 97,321 行（6.43%）；RM 205 图 / 6,375 行（1.41%）。
  **不排除**；只在报告里注明。
* 本地 `data/osu_cache/` 覆盖全部图；不需要重新下载。

### 2.2 缺失

* scope 外/解析失败 = `valid=false`，宽表 NaN；不插补。
* ISOR 在 LN 图上会返回数值（不报错），但它是 out-of-domain；报告用 `family` /
  `ln_ratio` 标注，不把 LN 子集当 supported 结果。
* Koishi 的 `BelowScope`/`AboveScope` 原样记录；10K ALL 里 Koishi 只覆盖一小部分。

---

## 3. 指标

### 3.1 NLL

~~~text
q_ij = sigmoid(g(d_j) - g(d_i))
L(d) = -sum_{i<j} [ n_ij log q_ij + n_ji log(1-q_ij) ]
NLL  = L(d) / N
~~~

* 主口径 linear（Sunny 现有表一致）；`d` 是原生分数：ISOR DP、Koishi Dan、Sunny star。
* 每个图集内所有信号同 pair 集；不同图集/不同数据集不能比绝对 NLL。
* ΔNLL = method − Sunny；负值更好。

### 3.2 相关性

* `pearson_bhat`：Pearson(d, b_full)；`rho_bhat`：Spearman(d, b_full)。
* 逐玩家 `lw` / `macro` Spearman。
* 本轮用 `--folds 1`：`b_full` 是 in-sample BT，仅用于相关性诊断；NLL 不受影响。
* 不报 `captured`：in-sample ceiling 不能冒充 cross-fitted。

### 3.3 coverage

* 每个表带 `n_rows / n_maps / n_pairs / coverage_rows / coverage_maps`。
* ALL 表里 ISOR 的 coverage 接近 100%，但这包含 LN out-of-domain；RC 表才是 supported coverage。

---

## 4. 评测矩阵（10 个 HPC 任务）

| # | 数据集 | 图集 | 信号 | 回答 |
|---|---|---|---|---|
| 0 | 10K | ALL | Sunny + ISOR | ISOR 在含 LN 的全分布上是否优于 Sunny（LN 部分为 out-of-domain） |
| 1 | 10K | ALL | Sunny + ISOR + Koishi（common） | 三者共同覆盖子集上的对比 |
| 2 | 10K | RC20 | Sunny + ISOR | ISOR native RC 全量 |
| 3 | 10K | RC80 | Sunny + ISOR + Koishi（common） | 两者都支持的 RC 子集 |
| 4 | 10K | HARD-RC | Sunny + ISOR + Koishi（common） | 高难 RC 上 Koishi 是否仍赢 |
| 5 | 10K | KOISHI-NATIVE | Sunny + ISOR + Koishi（common） | 只看 Koishi 覆盖的图 |
| 6 | 10K | LN | Sunny + ISOR | LN 子集；ISOR out-of-domain，描述性 |
| 7 | RM | ALL | Sunny + ISOR | ISOR 在全分布 |
| 8 | RM | RC20 | Sunny + ISOR | ISOR native RC |
| 9 | RM | LN | Sunny + ISOR | LN 子集；描述性 |

* 任务 1/3/4/5 用 `--common-maps`，实际 n 由 Koishi valid 决定。
* 任务 0/2/6/7/8/9 只有 Sunny+ISOR，coverage 由 ISOR 决定。
* HPC 脚本：`scripts/slurm/ki_eval.slurm`，`--folds 1 --bootstrap 0`。

---

## 5. 执行状态

| 阶段 | 状态 |
|---|---|
| benchmark 快照/版本 pin | 完成 |
| 1k HPC smoke（Sunny） | 完成：NLL 0.63165，captured 0.4727 |
| ISOR 10K 信号（16,061 图，DCR MSD 505 注入） | 完成 |
| Koishi 10K 信号（16,061 图） | 完成（1,056 valid） |
| ISOR randommerge 信号（20,388 图） | 完成 |
| 10K/RM 子集 parquet | 完成 |
| HPC 10 个 rank eval | 完成（job 14031088，10/10 COMPLETED） |
| 报告 | 完成：[koishi_isor_report.md](koishi_isor_report.md) |

### 5.1 ISOR 的可复现性

* 6 个方法在 benchmark 样本上逐位一致；ISOR 的 public 入口在 Linux 上缺 `msd.exe`，
  给出 16.07；注入 DCR MSD 505 后 Ankoku Butoukai 为 14.77，benchmark CSV 为 14.27。
* 本轮信号是 **ISOR v3.1 + DCR MinaCalc 505**，不是 benchmark 的原始 MSD 版本；
  报告会标注这一点，不声称逐位复现 benchmark。
* 若 HPC 上能跑 Wine/`msd.exe`，可另做一小批对账；不阻塞本轮。

---

## 6. 判据

| 观察 | 解释 |
|---|---|
| 10K HARD-RC 上 Koishi ΔNLL<0 | benchmark 高难优势迁移到 DCR ACC 序 |
| 10K HARD-RC 上 Koishi ΔNLL≥0 | 高难优势没有迁移；窄带拟合或 Dan/ACC construct 不同 |
| 10K/RM RC 上 ISOR ΔNLL<0 | ISOR 在 supported RC 域优于 Sunny |
| ALL 上 ISOR 变差、RC 上仍好 | 差异来自 LN out-of-domain，不是 RC 能力 |
| LN 子集上 ISOR 明显差于 Sunny | 符合 ISOR 文档的“LN unsupported”；不据此否定 RC 结果 |
| Koishi 在 10K 覆盖很低 | 高难方法的 coverage 限制；不能把它和全 RC 方法按总 NLL 排 |

**不能说的话**：

* 不能把 ISOR 在 LN 上的输出当成 LN 难度结论。
* 不能把 benchmark MAE 与 DCR NLL 混为一谈。
* 没有外部 Dan 标注，不能证明“Dan 标签过拟合”。
* 不能跨图集/数据集比较绝对 NLL。

---

## 7. 产物

| 路径 | 内容 |
|---|---|
| `data/processed/signals_raw/isor_10k_s*.ndjson` | ISOR 10K 原始输出 |
| `data/processed/signals_raw/koishi_10k_s*.ndjson` | Koishi 10K 原始输出 |
| `data/processed/signals_raw/isor_rm_s*.ndjson` | ISOR RM 原始输出 |
| `data/processed/signals_10k.parquet` | 宽表：sunny_star, isor_v31_dp, koishi_v5_dan |
| `data/processed/signals_randommerge.parquet` | 宽表：sunny_star, isor_v31_dp |
| `data/processed/eval_subsets/step1_*.parquet` | ALL/RC20/RC80/HARD-RC/LN/KOISHI-NATIVE |
| `results/ki_eval/10k/*.json` | 10K 7 个任务 |
| `results/ki_eval/rm/*.json` | RM 3 个任务 |
| `results/ki_eval/summary.md` / `.csv` | 汇总表 |
| `docs/koishi_isor_report.md` | 最终报告 |

---

## 8. 只剩的确认项（不阻塞执行）

1. **HARD-RC 阈值**：默认 top 30%（10K 5.0318；RM 4.5834）；若要更接近 11–17，
   可用 top 20%。本轮先 30%，报告里给 20/40 敏感性（若时间允许）。
2. **ISOR 的 MSD 版本**：本轮用 DCR MinaCalc 505；若你要求严格 benchmark 复现，
   需要在 HPC 上跑 Windows/Wine + `msd.exe` 另做一批，不阻塞本轮。
3. **Koishi 是否也要 RM**：你说 Koishi 只能测 10K，本轮不测 RM；如果之后要试，
   它的 coverage 会很低，只作描述。

---

## 9. 附：已核对事实

* benchmark 快照 `0bdbe23`；10 个算法；CSV `bid` 为空。
* Koishi benchmark：总 MAE 0.154（1,106 行）；RC11–17 0.151（1,084 行）；
  Death Melody 14.6339。
* ISOR benchmark：总 MAE 0.267（1,280 行）；RC11–17 0.222（1,102 行）；
  Ankoku Butoukai 14.27。
* ISOR 无 MSD：16.07（Ankoku）；注入 DCR MSD 505：14.77。
* 10K：1,513,746 行 / 16,061 图 / 225.64M pairs；RC20 99.35M；RC80 99.24M；
  HARD-RC 59.83M；LN 30.36M。
* RM：453,406 行 / 20,388 图 / 39.65M pairs；RC20 20.96M；LN 3.78M。
* benchmark 重叠：10K 205 图 / 97,321 行（6.43%）；RM 205 图 / 6,375 行（1.41%）。
