# randommerge linear_extra 的 VSRG Dan Benchmark 评测（2026-11-20）

> **状态：已完成。** 训练在 HPC CPU（job 14030754）完成；benchmark 推理、5 折校准和分层诊断在本机完成。
> 前序：[benchmark_plan.md](../legacy/docs/plans/benchmark_plan.md)（已归档）、[benchmark_vsrg.md](benchmark_vsrg.md)。
> 新数据集结果：[randommerge_p2p4_results.md](randommerge_p2p4_results.md)。
> 执行计划（含预注册读法）：[benchmark_linear_extra_plan.md](../legacy/docs/plans/benchmark_linear_extra_plan.md)（已归档）。
> **后续（2026-11-23）**：D2V 内容表示的对照见 [benchmark_d2v.md](benchmark_d2v.md)
> （D2V-L overall MAE 0.4527，比 RM-L 差 0.0311）。
> Benchmark 快照：VSRG-DanEstimation-Benchmark@e15f89b，samples/data.csv sha256 2b43165a…；
> samples 未更新，1,382 行不变。

---

## 0. 摘要

在 benchmark 的 **1,382 张 4K 谱面**上，评测在 randommerge 合并集上训练的
**linear_extra**（CM3P legacy + msd_overall + sunny_star，λ=1e-3）：

| 方法 | MAE | RMSE | bias | Spearman | Kendall | MAE 95% CI |
|---|---:|---:|---:|---:|---:|---|
| **RM-L（新）** | **0.4216** | 0.6440 | −0.0003 | 0.8652 | 0.7332 | [0.3965, 0.4481] |
| M1-L（旧） | 0.7411 | 1.4485 | +0.0069 | 0.8437 | 0.6957 | [0.6810, 0.8054] |
| Sunny isotonic | **0.3367** | 0.5376 | +0.0001 | 0.9060 | 0.7914 | [0.3155, 0.3587] |
| MSD isotonic | 0.6286 | 1.0077 | +0.0012 | 0.7849 | 0.6403 | [0.5882, 0.6697] |
| [Sunny, MSD] 线性 | 0.6568 | 0.8642 | −0.0006 | 0.9021 | 0.7856 | [0.6297, 0.6883] |
| 常数 median | 1.9195 | 2.8995 | 0.0000 | −0.0094 | −0.0073 | [1.8151, 2.0377] |

**主结论：**

1. **RM-L 比旧的 M1-L 大幅改善**：MAE 0.7411 → **0.4216**，
   配对 Δ = **−0.3195**（95% CI [−0.3802, −0.2658]）；排除 201 个训练重叠 bid / 210 行后
   0.7964 → **0.4419**，Δ = **−0.3545**（CI [−0.4201, −0.2843]）。改善不是重叠造成的。
2. **改善集中在低/中难度**：
   expected < 11 的 192 行，RM-L 0.9299 vs M1-L 2.4528（Δ −1.523）；
   expected 11–15 的 1,006 行，0.3058 vs 0.4485（Δ −0.143）；
   expected > 15 的 184 行，0.5241 vs 0.5548（Δ −0.031，**CI 跨 0**）。
   按 msd 分带同样：Q1 Δ −1.148、Q2 −0.109、Q3 −0.018（CI 跨 0）、
   Q4 −0.001（CI 跨 0）、msd≥32 −0.008（CI 跨 0）。
   **合并集在 benchmark 最难的一档没有带来显著增益**，这与它的“休闲化”分布一致。
3. **仍不如 isotonic sunny_star**：overall Δ = +0.0849（CI [0.0672, 0.1038]），
   head-to-head Sunny 赢 865 行 / RM-L 赢 517 行；RC 上 +0.0757，LN 上 +0.2103。
   但在 course 上 0.5227 vs Sunny 0.4763，Δ +0.0465（CI [−0.1033, +0.2104]，**不可区分**）。
4. **旧 M1-L 的 LN/course 失败被大幅修复**：LN 2.1115 → **1.0733**，
   course 2.2455 → **0.5227**；RC 0.5878 → **0.3655**。
5. **isotonic 仍然必要**：RM-L 的线性校准 MAE 0.8095，远差于 isotonic 0.4216；
   raw d_m 的 Spearman 0.8669，但 raw MAE 没有意义（d_m 是 BT 潜变量，只有仿射尺度）。
6. **与 published 算法相比**：RM-L 0.4216 优于 published Sunny（0.5763）、
   Companella（0.4934）、P1h（0.8314），与 published Daniel（0.4224，覆盖 83.3%）同量级；
   仍落后于 XDC1（0.2096）、Koishi（0.1540）、Azusa（0.2628）、ISOR（0.2672）、
   Roxy（0.2125）、Mixed（0.2879）。注意 published 算法的覆盖率不同（见 §3.6）。

---

## 1. 契约与数据边界

| 项 | 实际内容 |
|---|---|
| 输入 | samples/data.csv 1,382 行 + 1,382 个 .osu（4K、Mode 3） |
| 构成 | RC 1,246、LN 102、course 34 |
| 有 bid | 1,146 行 / 1,116 唯一 bid |
| 输出 | results/{Algorithm}.csv，8 列；只覆盖 got/delta/deltaAbs |
| 标签 | expected 1.0–18.3；25–75% 分位 13.4–14.2 |
| 内容 | benchmark_content_legacy.parquet：1,382 × 512，crop=legacy、audio=off、CM3P@26cc694 |
| 外部信号 | benchmark_signals.parquet：sunny_star / msd_overall，1,382 行无缺失 |

- **不做 audio-on**：benchmark 的 .osu 不带音频；本评测也不需要音频。
- **不重新编码**：benchmark legacy 内容表是 2026-11-19 那一轮编码的，本次原样复用；
  与 randommerge 训练表 chart_content_cm3p_local.parquet 的 CM3P 配置相同。
- **继承 b1321 修补**：上一轮编码时删掉了 Stay Alive x1.4 的 Bookmarks 尾逗号；
  本次没有重新编码，因此沿用该修补。hit object 未改。

---

## 2. 方法

### 2.1 模型

linear_extra：

~~~text
d_m = w'[C, msd_overall, sunny_star]
~~~

- C：CM3P legacy 512 维内容向量，按**训练图**逐维标准化；
- 外部信号：msd_overall / sunny_star，按**训练图**标准化；
- w：在玩家内 ACC 比较的聚合 pair counts 上，用 L2 正则（λ=1e-3）的
  Bradley–Terry 逻辑回归拟合；
- 与 MLP 不同，linear_extra 没有 epoch、没有 dropout、没有模型 seed 的随机性；
  它是一个凸问题，L-BFGS 从零初始化。

### 2.2 训练

| 项 | 值 |
|---|---|
| 训练表 | data/processed/step1_randommerge_pairbest_content.parquet |
| 行 / 图 / 玩家 | 453,406 / 20,388 / 12,758 |
| 内容表 | data/processed/chart_content_cm3p_local.parquet（21,949 × 512，覆盖 20,388/20,388） |
| 信号表 | data/processed/rank_signals_1k.parquet（21,949 图，两个信号无缺失） |
| 拟合口径 | --fit-all：用全部 453,406 行（含 pairbest 里原本标 test 的 136,060 行） |
| 唯一 pair / comparisons | 20,158,819 / 39,654,821 |
| λ / seed | 1e-3 / 0（linear 对 seed 不敏感） |
| 训练 NLL（in-sample） | **0.57954** |
| 计算 | HPC i64m512u，8 核，job 14030754；**1,095.2 s（18.25 min）** |
| 产物 | results/benchmark/models/RM_L_full.pt（195 KB） |

训练命令（HPC）：

~~~bash
python3 scripts/step2_cm3p_rank.py \
  --in data/processed/step1_randommerge_pairbest_content.parquet \
  --content-path data/processed/chart_content_cm3p_local.parquet \
  --signal-table data/processed/rank_signals_1k.parquet \
  --extra-signals msd_overall,sunny_star \
  --model linear_extra --lam 1e-3 \
  --val-frac 0 --seed 0 --threads 8 \
  --name cm3p_linear_extra --fit-all \
  --save-model results/benchmark/models/RM_L_full.pt \
  --out-signals results/benchmark/rm_linear/train/RM_L_signals.parquet \
  --out-json results/benchmark/rm_linear/train/RM_L.json
~~~

Slurm 脚本：[scripts/slurm/benchmark_rm_linear_train.slurm](../scripts/slurm/benchmark_rm_linear_train.slurm)。

### 2.3 推理与校准

- 推理：scripts/benchmark_infer.py 加载保存的 mu/sd、extra_mu/extra_sd、w；
  **不**按 benchmark 重新标准化（尺度冻结在训练集上）。
- 校准：与 2026-11-19 完全相同 ——
  expected ~ isotonic(d_m)，5 折 chart-group（同 bid 的 rate 变体同组），
  4 折拟合、留出折预测，拼成 OOF；线性校准 expected ~ a + b·d_m 作为对照；
  按 chart_group bootstrap 1,000 次给 MAE 95% CI。
- 折分配与旧评测逐行一致（G3 断言，见 §2.4）。

### 2.4 执行后的自检

| Gate | 结果 |
|---|---|
| G0 契约 | 1,382 行；legacy 内容 1,382×512；两个信号无缺失；快照 sha 未变 |
| G1 训练覆盖 | n_maps=20,388；n_train_rows=453,406；w 长度 514；extra_names=[msd_overall,sunny_star]；content_md5 与表一致 |
| G2 推理自检 | 20,388 个训练图，推理 d_m 与训练信号 **max abs Δ = 3.031e-07**（< 1e-6） |
| G3 折与旧结果 | 1,382/1,382 行 fold 相同；M1_L_isotonic 与 M1_L_d 的 max abs Δ = **0.0**；旧 4 个组合的 MAE/Spearman 逐位复现 |
| G4 主结果完整 | isotonic/linear/raw、CI、RC/family、重叠、难度带、范围、配对差、head-to-head 全部写出 |

G2 说明：benchmark_infer.py 要求 row_id 列，而训练内容表用的是 beatmap_id；
自检时在 /tmp 建了 row_id=beatmap_id 的临时视图，没有改表、没有改 infer 代码。

---

## 3. 结果

### 3.1 主表（5 折 chart-group OOF，全部 1,382 行）

| 方法 | MAE | RMSE | bias | Spearman | Kendall | MAE 95% CI |
|---|---:|---:|---:|---:|---:|---|
| **RM-L isotonic** | **0.4216** | 0.6440 | −0.0003 | 0.8652 | 0.7332 | [0.3965, 0.4481] |
| M1-L isotonic | 0.7411 | 1.4485 | +0.0069 | 0.8437 | 0.6957 | [0.6810, 0.8054] |
| M1-C isotonic | 0.7918 | 1.6234 | −0.0015 | 0.8450 | 0.6963 | [0.7249, 0.8688] |
| M2-L isotonic | 1.1489 | 1.9669 | −0.0022 | 0.6609 | 0.5069 | [1.0728, 1.2388] |
| M2-C isotonic | 1.1248 | 2.0585 | −0.0056 | 0.6823 | 0.5272 | [1.0442, 1.2185] |
| Sunny isotonic | 0.3367 | 0.5376 | +0.0001 | 0.9060 | 0.7914 | [0.3155, 0.3587] |
| MSD isotonic | 0.6286 | 1.0077 | +0.0012 | 0.7849 | 0.6403 | [0.5882, 0.6697] |
| [Sunny, MSD] 线性 | 0.6568 | 0.8642 | −0.0006 | 0.9021 | 0.7856 | [0.6297, 0.6883] |
| 常数 median | 1.9195 | 2.8995 | 0.0000 | −0.0094 | −0.0073 | [1.8151, 2.0377] |

线性校准对照：

| 方法 | MAE | Spearman |
|---|---:|---:|
| RM-L linear | 0.8095 | 0.8670 |
| M1-L linear | 1.2026 | 0.8524 |

RM-L 的 isotonic 比 linear 好 **0.388**，说明 d_m 与 Dan 的关系强非线性；
只用线性校准会严重低估这个方法。

raw d_m（未校准）的 Spearman 是 **0.8669**；raw MAE 不能报（d_m 与 Dan 差一个仿射偏移，
raw MAE 9.25 只是偏移的反映）。

### 3.2 RC / family / 重叠 / 干净子集

| 子集 | n | RM-L MAE | M1-L MAE | Sunny MAE | MSD MAE | RM-L rho |
|---|---:|---:|---:|---:|---:|---:|
| all | 1,382 | 0.4216 | 0.7411 | 0.3367 | 0.6286 | 0.8652 |
| clean（排除 210 行） | 1,172 | 0.4419 | 0.7964 | 0.3502 | 0.6571 | 0.8716 |
| seen（训练重叠 210 行） | 210 | 0.3084 | 0.4324 | 0.2616 | 0.4699 | 0.8307 |
| RC | 1,246 | 0.3655 | 0.5878 | 0.2898 | 0.4877 | 0.8363 |
| clean RC | 1,038 | 0.3767 | 0.6188 | 0.2955 | 0.4916 | 0.8384 |
| LN | 102 | 1.0733 | 2.1115 | 0.8630 | 2.1378 | 0.9711 |
| course | 34 | 0.5227 | 2.2455 | 0.4763 | 1.2658 | 0.9969 |
| msd≥32 | 669 | 0.2857 | 0.2939 | 0.2194 | 0.3564 | 0.7598 |

配对的 95% CI（按 chart_group bootstrap 1,000 次）：

| 子集 | n | vs M1-L ΔMAE [CI] | vs Sunny ΔMAE [CI] | vs MSD ΔMAE [CI] |
|---|---:|---|---|---|
| all | 1,382 | **−0.3195 [−0.3802, −0.2658]** | +0.0849 [+0.0672, +0.1038] | −0.2070 [−0.2449, −0.1711] |
| clean | 1,172 | **−0.3545 [−0.4201, −0.2843]** | +0.0917 [+0.0709, +0.1120] | −0.2152 [−0.2595, −0.1720] |
| seen | 210 | −0.1240 [−0.1895, −0.0613] | +0.0468 [−0.0014, +0.0964] | — |
| RC | 1,246 | −0.2223 [−0.2720, −0.1798] | +0.0757 [+0.0576, +0.0944] | −0.1222 [−0.1536, −0.0919] |
| LN | 102 | −1.0382 [−1.3869, −0.7087] | +0.2103 [+0.1006, +0.3110] | −1.0645 [−1.3751, −0.7391] |
| course | 34 | −1.7227 [−2.8107, −0.8386] | +0.0465 [−0.1033, +0.2104] | −0.7431 [−0.9765, −0.5303] |
| expected low | 192 | −1.5229 [−1.8296, −1.2267] | +0.1397 [+0.0487, +0.2228] | −0.6265 [−0.8058, −0.4274] |
| expected mid | 1,006 | −0.1426 [−0.1790, −0.1083] | +0.0722 [+0.0561, +0.0887] | −0.1170 [−0.1516, −0.0869] |
| expected high | 184 | −0.0307 [−0.0745, +0.0149] | +0.0970 [+0.0524, +0.1440] | −0.2617 [−0.3416, −0.1846] |
| msd Q1 | 346 | −1.1483 [−1.3431, −0.9455] | +0.1440 [+0.0818, +0.2028] | −0.5788 [−0.7113, −0.4470] |
| msd Q2 | 345 | −0.1094 [−0.1621, −0.0613] | +0.0654 [+0.0427, +0.0903] | −0.1065 [−0.1474, −0.0693] |
| msd Q3 | 346 | −0.0179 [−0.0389, +0.0044] | +0.0388 [+0.0221, +0.0568] | −0.0248 [−0.0509, +0.0012] |
| msd Q4 | 345 | −0.0007 [−0.0338, +0.0299] | +0.0914 [+0.0609, +0.1206] | −0.1175 [−0.1591, −0.0776] |
| msd≥32 | 669 | −0.0082 [−0.0267, +0.0104] | +0.0663 [+0.0492, +0.0840] | −0.0707 [−0.0954, −0.0458] |

（所有配对差均为 1,000 次 chart-group bootstrap。）

**读法**：RM-L 相对 M1-L 的增益在 clean 子集上更大（−0.3545 vs −0.3195），
说明不是“训练时见过 210 行”造成的；seen 子集本身的 Δ 只有 −0.1240。

### 3.3 难度分层

expected 三分：

| 带 | n | RM-L | M1-L | Sunny | Δ(RM-L−M1-L) [CI] |
|---|---:|---:|---:|---:|---|
| expected < 11 | 192 | 0.9299 | 2.4528 | 0.7902 | **−1.5229 [−1.8296, −1.2267]** |
| expected 11–15 | 1,006 | 0.3058 | 0.4485 | 0.2336 | **−0.1426 [−0.1790, −0.1083]** |
| expected > 15 | 184 | 0.5241 | 0.5548 | 0.4271 | −0.0307 [−0.0745, +0.0149] |

msd 四分位：

| 带 | n | RM-L | M1-L | Sunny | Δ(RM-L−M1-L) [CI] |
|---|---:|---:|---:|---:|---|
| Q1（最低） | 346 | 0.8141 | 1.9624 | 0.6701 | **−1.1483 [−1.3431, −0.9455]** |
| Q2 | 345 | 0.3066 | 0.4160 | 0.2413 | −0.1094 [−0.1621, −0.0613] |
| Q3 | 346 | 0.2165 | 0.2345 | 0.1777 | −0.0179 [−0.0389, +0.0044] |
| Q4（最高） | 345 | 0.3486 | 0.3494 | 0.2573 | −0.0007 [−0.0338, +0.0299] |
| msd≥32 | 669 | 0.2857 | 0.2939 | 0.2194 | −0.0082 [−0.0267, +0.0104] |

expected < 11 的 192 行里，RC 126 / LN 47 / course 19；
所以低段的改善不只是 LN/course，也包含 126 行 RC。
expected > 15 的 184 行里，RC 162 / LN 17 / course 5；
这一档 RM-L 与 M1-L 不可区分，且仍差于 Sunny（0.4271）。

### 3.4 范围 / 外推诊断

raw d_m 的分位数：

| 序列 | p1 | p5 | p50 | p95 | p99 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 训练 20,388 图 | −2.811 | −2.175 | −0.214 | 2.980 | 4.309 | −3.623 | 6.452 |
| benchmark 1,382 行 | −0.450 | 0.928 | 4.045 | 5.656 | 6.862 | −1.821 | 8.265 |

- benchmark 的 d_m 整体比训练集高：35.6%（492 行）高于训练 p99；
  1.66%（23 行）高于训练 max；没有行低于训练 min。
- 高于 p99 的 492 行 MAE = 0.353；高于 max 的 23 行 MAE = 0.591；
  低于 p99 的行 MAE 略高于 overall（因为低段本来更难拟合）。
- 结论：isotonic 的 out_of_bounds=clip 只作用于 23 行，范围外推**不是**主结果的主要来源；
  但这 23 行的 MAE 偏高，报告里应保留这个 caveat。

### 3.5 head-to-head

| 对比 | 子集 | n | RM-L wins | 对手 wins | ties | RM-L MAE | 对手 MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| RM-L vs M1-L | all | 1,382 | 828 | 553 | 1 | 0.4216 | 0.7411 |
| RM-L vs M1-L | clean | 1,172 | 708 | 463 | 1 | 0.4419 | 0.7964 |
| RM-L vs M1-L | RC | 1,246 | 720 | 525 | 1 | 0.3655 | 0.5878 |
| RM-L vs M1-L | course | 34 | 29 | 5 | 0 | 0.5227 | 2.2455 |
| RM-L vs M1-L | seen | 210 | 120 | 90 | 0 | 0.3084 | 0.4324 |
| RM-L vs M1-L | msd≥32 | 669 | 365 | 303 | 1 | 0.2857 | 0.2939 |
| RM-L vs Sunny | all | 1,382 | 517 | 865 | 0 | 0.4216 | 0.3367 |
| RM-L vs Sunny | course | 34 | 18 | 16 | 0 | 0.5227 | 0.4763 |
| RM-L vs Sunny | seen | 210 | 85 | 125 | 0 | 0.3084 | 0.2616 |

### 3.6 与 published benchmark 算法

同一 1,382 行、published CSV 原样：

| 算法 | n | 覆盖 | MAE | RMSE | bias | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Sunny | 1,363 | 98.6% | 0.5763 | 0.7829 | +0.5251 | 0.9057 |
| Daniel | 1,151 | 83.3% | 0.4224 | 0.5030 | −0.3478 | 0.8662 |
| XDC1 | 1,208 | 87.4% | 0.2096 | 0.3271 | +0.0329 | 0.8948 |
| Koishi | 1,106 | 80.0% | 0.1540 | 0.2046 | +0.0275 | 0.8954 |
| Azusa | 1,278 | 92.5% | 0.2628 | 0.3869 | +0.0093 | 0.9114 |
| ISOR | 1,280 | 92.6% | 0.2672 | 0.3850 | +0.0057 | 0.8972 |
| Roxy | 1,130 | 81.8% | 0.2125 | 0.3078 | +0.0232 | 0.8697 |
| Companella | 1,382 | 100% | 0.4934 | 0.7540 | +0.2927 | 0.8847 |
| Mixed | 1,382 | 100% | 0.2879 | 0.5126 | +0.0757 | 0.9202 |
| P1h | 1,382 | 100% | 0.8314 | 1.0107 | +0.7658 | 0.8472 |
| **RM-L（本报告）** | **1,382** | **100%** | **0.4216** | **0.6440** | **−0.0003** | **0.8652** |

说明：

- 与 published Daniel（0.4224）同量级，但 Daniel 只覆盖 83.3%；不能算“赢”。
- 优于 published Sunny（0.5763）、Companella（0.4934）、P1h（0.8314）。
- 落后于 XDC1、Koishi、Azusa、ISOR、Roxy、Mixed；
  其中 Mixed 是 100% 覆盖的 0.2879，和 RM-L 的差距是明确的。
- **不要**把 published Sunny（0.5763）和 §3.1 的 “Sunny isotonic”（0.3367）混为一谈：
  后者是本报告用 benchmark 标签做的 5 折 OOF 校准，数值更低。

### 3.7 可视化

| 图 | 内容 |
|---|---|
| [RM_L_vs_M1L_ALL.png](../results/benchmark/viz/RM_L_vs_M1L_ALL.png) | RM-L vs M1-L，全部 1,382 行 |
| [RM_L_vs_Sunny_ALL.png](../results/benchmark/viz/RM_L_vs_Sunny_ALL.png) | RM-L vs published Sunny，全部 1,382 行 |
| [RM_L_vs_Sunny_RC.png](../results/benchmark/viz/RM_L_vs_Sunny_RC.png) | RM-L vs published Sunny，RC 子集 |

生成脚本：scripts/render_benchmark_dashboard.py（按 benchmark 网页 stats.js 口径）。

---

## 4. 解读

1. **这是 DCR 在 benchmark 上目前最好的结果**：0.4216，比 M1-L 的 0.7411 低 0.32 MAE；
   在 1,382 行里 RM-L 赢 828 行、输 553 行（ties 1）。clean 子集上改善更大，
   说明不是训练重叠造成的。

2. **改善是“分布相关”的，不是全面变好**：
   低/中难度（expected < 15 的 1,198 行，占 86.7%）贡献了几乎全部增益；
   最难的一档（expected > 15、msd Q4、msd≥32）RM-L 与 M1-L **统计上不可区分**。
   这与合并集的分布完全一致：它有 20,388 图 / 12,758 玩家，但 msd≥30 的训练行
   只有 5,368（1.2%），而旧 1k 有 26,608（24.3%）。
   **“更多数据”在这里买到的是低/中段覆盖，不是高段覆盖。**

3. **family 的旧失败模式被修复**：
   LN 2.1115 → 1.0733，course 2.2455 → 0.5227，RC 0.5878 → 0.3655。
   course 上已经和 isotonic Sunny 打平（0.5227 vs 0.4763，CI 跨 0）；
   但 LN（1.0733 vs 0.8630）和 RC（0.3655 vs 0.2898）仍落后于 Sunny。

4. **仍不如 Sunny，但差距从 0.40 缩到 0.08**：
   M1-L 比 Sunny 差 +0.404；RM-L 只差 +0.0849。
   raw d_m 与 sunny_star 的 Spearman 是 0.947（M1-L 是 0.918），
   说明 RM-L 的内容头已经相当接近 Sunny 的排序；它输在 Dan 尺度上的条件均值，
   而不是排序。Sunny 的 isotonic 是 benchmark 标签上的灵活单调拟合，
   RM-L 的映射是在 randommerge 的 ACC 序上学的，两者目标不同。
   另外，RM-L 的输入本来就包含 sunny_star；这不是“独立信号打败融合方法”，
   而是“融合后的 Dan 条件均值仍不如单信号在 benchmark 标签上的灵活校准”。

5. **isotonic 是必要的**：RM-L 线性校准 0.8095 vs isotonic 0.4216。
   如果只报线性校准，会误判这个方法“没用”；如果只报 raw Spearman 0.8669，
   又会高估它的 Dan 尺度可用性。两套指标必须分开报。

6. **重叠不是驱动因素**：全部 0.4216 / clean 0.4419；Δ vs M1-L 全部 −0.3195 /
   clean −0.3545。seen 子集只有 −0.1240。训练重叠让两边都变容易，
   但没有制造 RM-L 的优势。

7. **范围外推不是主因，但要保留 caveat**：
   35.6% 的 benchmark 行高于训练 p99，1.66% 高于训练 max；
   23 行被 isotonic clip，MAE 0.591。整体结果不是由这 23 行决定的，
   但高段的校准依赖外推，这会影响“高段不可区分”的解释强度。

---

## 5. 局限

1. **校准用了 benchmark 标签**：主表是 5 折 OOF isotonic 后的 MAE，
   不是完全 label-free 的结果；raw Spearman/Kendall 另行报告。
2. **训练目标不同**：randommerge 的 d_m 来自玩家 ACC 序的 BT 潜变量；
   benchmark 的 expected 是社区 Dan 标注。两者相关但不同构造。
3. **模型族与数据同时变了**：RM-L 是 linear_extra + randommerge；
   M1-L 是 mlp_extra + 1k。0.7411 → 0.4216 不能全部归因于数据。
   randommerge 文档在同一数据上给出 linear_extra 优于 mlp_extra
   （P2 −0.0050、P4 −0.0091 NLL），所以模型族也贡献了一部分；
   要严格分离，需要再训一个 1k 的 linear_extra（本次未做）。
4. **难度覆盖不匹配**：合并集在 benchmark 高段的训练行远少于旧 1k；
   高段结果不可区分，不应外推成“新数据集在所有段位都更好”。
5. **谱面重叠**：201 bid / 210 行出现在 randommerge 训练图中；
   主表 + clean 敏感性都报了，clean 上改善更大。
6. **内容跨表差异**：38 个 rate=1.0 重叠行里 37 个 cosine ≥ 0.99999，
   1 个（Volcanic(delta)，bid 3353155）0.896；与上一轮相同的跨表设置。
   本次没有重新编码，个别 .osu 版本差异保留。
7. **published 算法覆盖率不同**：0.1540–0.4934 的 published 数字对应
   80.0%–100% 覆盖；不能只看 MAE 排名。
8. **单一模型、单一 split**：linear 对模型 seed 确定，但训练行/图的选择
   （P2/P4 split seed）没有在 benchmark 上做 5-seed 稳健性臂。
9. **HPC 环境**：训练在 HPC 的 c29431f 仓库、Python 3.11、pandas 2.3.3 下完成；
   step2_cm3p_rank.py 与 dcr/rank_metrics.py 的 sha256 与本地一致
   （4b1634e… / 41084b3…），但 checkpoint 的 git_commit 记的是 HPC HEAD。
10. **不涉及 audio-on / canon**：本轮只做 legacy audio-off；
    audio-on 和 canon 的结论仍以旧报告为准。

---

## 6. 产物

| 路径 | 内容 |
|---|---|
| results/benchmark/models/RM_L_full.pt | 全量 linear_extra（mu/sd/extra_mu/extra_sd/w/train_chart_ids） |
| results/benchmark/rm_linear/train/RM_L_signals.parquet | 20,388 个训练图的 d_m |
| results/benchmark/rm_linear/train/RM_L.json | 训练 JSON（NLL、行数、pair 数、args） |
| results/benchmark/rm_linear/train/RM_L_selfcheck.parquet | G2 自检推理 |
| results/benchmark/pred/RM_L.parquet | benchmark 1,382 行的 d_m |
| results/benchmark/eval_rm/summary.md / summary.csv | 5 个 run + baseline 的主表 |
| results/benchmark/eval_rm/metrics.json | 全部指标、CI、配对差 |
| results/benchmark/eval_rm/oof.parquet | 逐行 OOF got、fold、原始 d_m |
| results/benchmark/eval_rm/per_group.csv | family/pattern/difficulty 分层 |
| results/benchmark/eval_rm/published_baselines.csv | published 算法同表 |
| results/benchmark/eval_rm/candidates/DCR-CM3P-RM-Linear.csv | 8 列候选提交文件（OOF isotonic） |
| results/benchmark/eval_rm/breakdown/*.csv / summary.md | RC/family/重叠/难度带/范围/配对/head-to-head |
| results/benchmark/eval_rm/breakdown/paired_delta_by_band.csv | expected/msd 分带的配对 ΔMAE + CI |
| results/benchmark/viz/RM_L_vs_M1L_ALL.png | RM-L vs M1-L 仪表盘 |
| results/benchmark/viz/RM_L_vs_Sunny_ALL.png | RM-L vs published Sunny 仪表盘 |
| results/benchmark/viz/RM_L_vs_Sunny_RC.png | RC 子集 |
| results/benchmark/logs/slurm_rm_bench_linear_14030754.out / .err | HPC 训练日志 |
| scripts/slurm/benchmark_rm_linear_train.slurm | HPC 训练脚本 |
| scripts/benchmark_breakdown.py | 分层/敏感性/诊断脚本 |
| scripts/benchmark_calibrate_eval.py | 新增 RM_L 与 --runs（默认仍是旧 4 个） |

**候选 CSV 默认没有复制到 benchmark 仓库、没有改 results/index.json、没有 push。**

---

## 7. 复现

### 7.1 HPC 训练

~~~bash
# 在 HPC 登录节点，仓库在 ~/DCR
cd ~/DCR
/opt/slurm/bin/sbatch scripts/slurm/benchmark_rm_linear_train.slurm
~~~

或直接跑脚本里的 python 命令（见 §2.2）。训练完成后再执行一次 G2 自检；
benchmark_rm_linear_train.slurm 已包含自检和断言。

### 7.2 本机推理 / 评测 / 诊断

~~~bash
python3 scripts/benchmark_infer.py \
  --model results/benchmark/models/RM_L_full.pt \
  --content data/processed/benchmark_content_legacy.parquet \
  --signals data/processed/benchmark_signals.parquet \
  --out results/benchmark/pred/RM_L.parquet

python3 scripts/benchmark_calibrate_eval.py \
  --runs RM_L,M1_L,M1_C,M2_L,M2_C \
  --pred-dir results/benchmark/pred \
  --out-dir results/benchmark/eval_rm

python3 scripts/benchmark_breakdown.py \
  --oof results/benchmark/eval_rm/oof.parquet \
  --model results/benchmark/models/RM_L_full.pt \
  --train-signals results/benchmark/rm_linear/train/RM_L_signals.parquet \
  --out-dir results/benchmark/eval_rm/breakdown
~~~

### 7.3 关键复现断言

- benchmark 内容表 1,382 行、512 维；signals 1,382 行；快照 e15f89b。
- checkpoint：n_train_rows=453,406、n_maps=20,388、w.shape=(514,)、
  extra_names=[msd_overall,sunny_star]、content_md5=0c2d5da5…
- G2：自检 max abs Δ = 3.031e-07。
- G3：fold 1,382/1,382 相同；M1_L_isotonic max abs Δ = 0.0。
- 主表：RM-L 0.4216；M1-L 0.7411；Sunny 0.3367。

---

## 8. 一句话结论

**randommerge 合并集上的 linear_extra 把 benchmark MAE 从 0.7411 降到 0.4216，
成为 DCR 目前最好的结果；但增益集中在低/中难度，最难的一档与旧模型不可区分，
仍落后于 isotonic Sunny 0.085 MAE——这是一次“覆盖换来的进步”，不是“高段变强”。**
