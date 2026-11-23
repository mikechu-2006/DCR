# VSRG Dan Benchmark：4 个 DCR CM3P 组合的 5 折 chart-group LOCV 评测

> **状态：已完成（2026-11-19）。** Benchmark 快照：
> [VSRG-DanEstimation-Benchmark](https://github.com/LeoBlackMT/VSRG-DanEstimation-Benchmark)
> commit `e15f89b834ad6cdb055480039aa583baff802a43`。
>
> 本文只报告本仓库的 4 个 CM3P 组合；**不包含 audio-on 臂**，也不修改 benchmark 仓库。
>
> **后续（2026-11-20 / 2026-11-23）**：randommerge 合并集 linear_extra 的评测见
> [benchmark_linear_extra.md](benchmark_linear_extra.md)（CM3P RM-L overall MAE 0.4216）
> 和 [benchmark_d2v.md](benchmark_d2v.md)（D2V-L overall MAE 0.4527）。

---

## 0. 摘要

在 benchmark 的 **1,382 张 4K 谱面**上，把 DCR 的两种 CM3P 模型各配 legacy / canon 两个内容表：

| 组合 | 方法 | 内容表 |
|---|---|---|
| **M1-L** | CM3P + external MLP | legacy |
| **M1-C** | CM3P + external MLP | canon |
| **M2-L** | CM3P MLP h64 L2 | legacy |
| **M2-C** | CM3P MLP h64 L2 | canon |

**主结论：4 个组合都明显差于单信号 `sunny_star` 的 isotonic 校准，也没有超过 `msd_overall` 单信号；最好的组合是 M1-L，5 折 out-of-fold MAE 0.7411，而 `sunny_star` 是 0.3367、`msd_overall` 是 0.6286。**

具体读法：

- **external 信号有用，但不足以弥补域差**：M1（含 `msd_overall,sunny_star`）的 MAE 0.74–0.79，
  比 M2（只用 CM3P）的 1.12–1.15 好；但都差于只用一个 `sunny_star` 的校准。
- **legacy 略好于 canon（仅 M1）**：M1-C − M1-L = **+0.0507** MAE
  （95% CI [+0.0090, +0.0936]），即 canon 显著更差；M2 上 canon 略好但两者都很差，
  不足以说明 canon 更优。
- **主要失败来自 course/LN**：M1-L 在 RC 上 MAE **0.5878**，在 LN 上 **2.111**、
  course 上 **2.245**。136 个 course/LN 行贡献了总 MAE 的约 0.21。
- **不是训练子集噪声**：P4 5-seed 集成给出同量级结果（M1-L 0.7348、M1-C 0.8126、
  M2-L 1.0632、M2-C 0.9963），结论不依赖单一 6×6 训练块。
- **不是编码 bug**：单图对账 legacy/canon 与 DCR 内容表的 cosine 分别为
  0.9999907 / 0.9999931；benchmark 内容几何见 §3。
- **按 `expected ≥ 12` 重做（post-hoc）**：在 1,142 行上把 isotonic 只在子集内重拟合，
  M1-L **0.3194**、M1-C **0.3108**，M1-L 胜 `msd_overall` 0.0796、负 `sunny_star` 0.0798；
  M1-L 与 M1-C 的差 −0.0086（95% CI −0.0250…+0.0068）不显著。全量 MAE 0.7411 主要由
  240 行 `expected < 12` 拉动（该段 MAE 2.3815）；详见 §5.7。

**结论的性质**：M1/M2 的 `d_m` 是从 DCR top-1000 玩家的 ACC 序学到的
**相对难度潜变量**，benchmark 的 `expected` 是社区 Dan 标注；两者相关但不同构造。
本结果说明这个 DCR 难度分数**不能直接迁移**成 Dan 估计，而不是说明 CM3P 本身没有信息
（M1 的 Spearman 仍有 0.844，高于 `msd_overall` 的 0.785，但 MAE 更差）。

---

## 1. 契约与数据边界

| 项 | 实际内容 |
|---|---|
| 输入 | `samples/data.csv` 1,382 行 + 1,382 个 `.osu`（4K、Mode 3） |
| 构成 | RC 1,246（jack/speed/stamina/tech）、LN 102、course 34 |
| 有 `bid` | 1,146 行 / 1,116 唯一 bid |
| 输出 | `results/{Algorithm}.csv`，8 列；只覆盖 `got/delta/deltaAbs` |
| `got` | Reform Dan 数值化难度，与 `expected` 同尺度 |
| 标签 | `expected` 1.0–18.3；25–75% 分位 13.4–14.2 |

**不做 audio-on**：benchmark 的 `.osu` 不带音频；本评测也不需要音频。

**一个数据修补**：`b1321`（`Stay Alive x1.4`）的 `.osu` 在 `Bookmarks` 行末尾多了一个逗号，
CM3P 的 slider parser 报 `should be an int list`。删掉该尾逗号后重新编码；
只改这个字段，不改任何 hit object。

---

## 2. 方法：从 `d_m` 到 `got`

### 2.1 原始输出是实数，但尺度是 BT 潜变量

- 线性：`d_m = w'C`
- MLP：`d_m = w'C + MLP(C)`
- M1 的 `C` 换成 `[C, msd_overall, sunny_star]`

`d_m` 是每张谱面的一个实数难度分数，但只定义到仿射变换；它不是 Dan，也不是段位标签。

### 2.2 校准

- **主口径**：`expected ~ isotonic(d_m)`，5 折按 `chart_group`
  （同 bid 的 rate 变体同组；无 bid 用去 rate 后缀的 name）交叉拟合；
  每个 `got` 都是 out-of-fold。
- **对照**：线性 `expected ~ a + b·d_m`。
- **不假设 Dan 等距**：isotonic 只要求单调；线性校准会强制等距，只作对照。
- 校准只使用 benchmark 的 `expected`，不参与训练、特征工程或模型选择；
  这是本报告最大的解释限制。

---

## 3. 数据与内容表

### 3.1 Manifest

`scripts/benchmark_manifest.py` → `data/processed/benchmark_manifest.parquet`：

- 1,382/1,382 个 `.osu` 精确匹配；
- 压缩包里多出的 8 个嵌套 `.osu` 忽略；
- 所有谱面 Mode=3、CircleSize=4；
- 记录 `row_id, bid, pattern, family, expected, rate, osu_path, osu_md5, chart_group`。

### 3.2 外部信号

`scripts/benchmark_signals.py`：

- `sunny_star`：`osumania_map_analyser` commit `448ed7c`，Node 无头；
- `msd_overall`：MinaCalc standalone calc_version 505、goal 0.93、rate 1.0。

**对账**：32 张 DCR 图上，Sunny max |Δ| = **8.9e-16**，MinaCalc max |Δ| = **0**；
benchmark 侧 1,382/1,382 两个信号都有值。

### 3.3 CM3P 内容表

`scripts/benchmark_encode.py` 按 `row_id` 编码（不是 bid；rate 变体共用 bid）：

- legacy：crop=legacy、audio=off、max_length=2000、pad=longest、fp32；
- canon：crop=canon、audio=off，其余同上。

编码在 HPC CPU 节点完成。最终两臂各 **1,382 行**。

几何检查（400 张随机子样本的跨图 cosine）：

| 臂 | cos mean | cos median |
|---|---:|---:|
| legacy | 0.8641 | 0.8630 |
| canon | 0.9650 | 0.9880 |

canon 在 benchmark 上比 legacy 更同质；单图对账（DCR 193127）canon cosine 0.9999931，
说明不是编码错误，而是这批高难 4K 谱面在 canon 裁剪下差别更小。
逐行 legacy–canon cosine 均值 0.8340。

---

## 4. 训练

### 4.1 全量模型（主模型）

每个组合在 109,569 行上训练 98 epoch，模型 seed 0；超参与
[`docs/p4_multiseed_audio.md`](p4_multiseed_audio.md) 一致（h64、depth 2、
dropout 0.1、lr 3e-3、lam 1e-2）。训练 NLL：

| 组合 | train NLL |
|---|---:|
| M1-L | 0.5313 |
| M1-C | 0.5446 |
| M2-L | 0.5524 |
| M2-C | 0.5694 |

canon 的训练 NLL 比 legacy 差，与 DCR 上的既有结论一致。

### 4.2 P4 5 seed 稳健性

同一组合在 P4 的 5 个划分种子上各训练一次（每次只用 6×6 训练块），
共 20 个模型；推理时对 5 个模型的 z-score 后 `d_m` 取平均。
主结果用全量模型；P4 用来检查训练子集方差。

---

## 5. 结果

### 5.1 主表：5 折 chart-group out-of-fold（全部 1,382 行）

| 组合 | MAE | RMSE | bias | Spearman | Kendall | MAE 95% CI |
|---|---:|---:|---:|---:|---:|---|
| **M1-L** | **0.7411** | 1.4485 | +0.0069 | 0.8437 | 0.6957 | [0.6810, 0.8054] |
| M1-C | 0.7918 | 1.6234 | −0.0015 | 0.8450 | 0.6963 | [0.7249, 0.8688] |
| M2-L | 1.1489 | 1.9669 | −0.0022 | 0.6609 | 0.5069 | [1.0728, 1.2388] |
| M2-C | 1.1248 | 2.0585 | −0.0056 | 0.6823 | 0.5272 | [1.0442, 1.2185] |

线性校准（对照，强制等距）明显更差：
M1-L 1.2026、M1-C 1.1995、M2-L 1.4535、M2-C 1.4240。
这说明 `d_m` 与 Dan 的关系强非线性，isotonic 是必要的。

### 5.2 对照与已发布算法

| 方法 | MAE | RMSE | bias | Spearman |
|---|---:|---:|---:|---:|
| 常数 median | 1.9195 | 2.8995 | 0.0000 | −0.0094 |
| `sunny_star` -> isotonic (label-fit) | **0.3367** | 0.5376 | +0.0001 | 0.9060 |
| `msd_overall` -> isotonic (label-fit) | 0.6286 | 1.0077 | +0.0012 | 0.7849 |
| `[sunny_star, msd_overall]` 线性 | 0.6568 | 0.8642 | −0.0006 | 0.9021 |

> **命名很重要：这里的 Sunny isotonic 不是网站上的 Sunny 算法。**
> 网站 results/Sunny.csv 是 Sunny 算法自己输出的 Dan（未用 benchmark 标签再拟合），
> 在网站 scope=RC 上 MAE **0.5720**（用户看到的 0.57），在全量 1,382 行上 0.5763。
> 本表的 sunny_star -> isotonic 是：取 sunny_star 星数（不是 Dan），用 benchmark 的
> expected 做 **5 折 out-of-fold 单调校准**；它使用了 benchmark 标签，是一个
> **label-fit diagnostic / ceiling**，不是可提交的 Sunny 成绩。同理 msd_overall -> isotonic。
> 因为 M1/M2 的 got 也用了同样的 5 折 OOF isotonic，**M1 vs Sunny-isotonic 是
> “都用标签校准”的公平对比；M1 vs 网站 Sunny 则不是**（网站 Sunny 不用这些标签）。

benchmark 已发布算法（同 1,382 行，原样）：

| 算法 | n | 覆盖 | MAE | RMSE | Spearman |
|---|---:|---:|---:|---:|---:|
| Sunny | 1,363 | 98.6% | 0.5763 | 0.7829 | 0.9057 |
| Daniel | 1,151 | 83.3% | 0.4224 | 0.5030 | 0.8662 |
| XDC1 | 1,208 | 87.4% | 0.2096 | 0.3271 | 0.8948 |
| Koishi | 1,106 | 80.0% | 0.1540 | 0.2046 | 0.8954 |
| Azusa | 1,278 | 92.5% | 0.2628 | 0.3869 | 0.9114 |
| ISOR | 1,280 | 92.6% | 0.2672 | 0.3850 | 0.8972 |
| Roxy | 1,130 | 81.8% | 0.2125 | 0.3078 | 0.8697 |
| Companella | 1,382 | 100% | 0.4934 | 0.7540 | 0.8847 |
| Mixed | 1,382 | 100% | 0.2879 | 0.5126 | 0.9202 |
| P1h | 1,382 | 100% | 0.8314 | 1.0107 | 0.8472 |

### 5.3 只看向 RC（严格 RC：不含 course/LN）

| 方法 | RC n | MAE | 95% CI | Spearman |
|---|---:|---:|---|---:|
| **M1-L** | 1,246 | **0.5878** | [0.5427, 0.6453] | 0.829 |
| M1-C | 1,246 | 0.6247 | [0.5677, 0.6901] | 0.828 |
| M2-L | 1,246 | 1.0102 | [0.9391, 1.0905] | 0.619 |
| M2-C | 1,246 | 0.9682 | [0.8912, 1.0498] | 0.661 |
| Sunny isotonic | 1,246 | 0.2898 | [0.2699, 0.3114] | — |
| MSD isotonic | 1,246 | 0.4877 | [0.4568, 0.5190] | — |

**网站 scope=RC 的定义不同**：网站的 RC 是 **non-LN**（1,280 行，含 34 个 course），
不是本表的严格 RC（1,246 行，只含 jack/speed/stamina/tech）。这正解释了
网站 Sunny 的 0.57 与本表 strict-RC Sunny 0.5805 的差异：

| 方法（网站 RC，1,280 行） | MAE |
|---|---:|
| M1-L（全量校准） | 0.6319 |
| M1-C（全量校准） | 0.6651 |
| Sunny（网站原值，未用 benchmark 标签） | **0.5720** |
| Sunny-star -> isotonic（label-fit） | 0.2948 |
| MSD -> isotonic（label-fit） | 0.5084 |
| 常数 | 1.7526 |

网站 RC 下再看 `expected >= 12`（1,095 行，子集内重拟合）：

| 方法（网站 RC，12+） | MAE |
|---|---:|
| M1-L | 0.2982 |
| M1-C | 0.2909 |
| M2-L | 0.5066 |
| M2-C | 0.4604 |
| Sunny（网站原值） | 0.5013 |
| Sunny-star -> isotonic（label-fit） | **0.2193** |
| MSD -> isotonic（label-fit） | 0.3656 |

所以：在 12+ 的网站 RC 口径下，M1-L 的 **0.2982** 已经好于网站 Sunny 的 **0.5013**，
但仍差于用 benchmark 标签校准的 Sunny 星数 **0.2193**。

### 5.4 分 family

| 组合 | RC MAE | LN MAE | course MAE |
|---|---:|---:|---:|
| M1-L | 0.588 | 2.111 | 2.245 |
| M1-C | 0.625 | 2.382 | 2.145 |
| M2-L | 1.010 | 2.377 | 2.546 |
| M2-C | 0.968 | 2.592 | 2.463 |
| Sunny isotonic | 0.290 | 0.863 | 0.476 |
| MSD isotonic | 0.488 | 2.138 | 1.266 |

course/LN 的 within-family Spearman 并不低（M1-L：course 0.71、LN 0.82），
但全局 isotonic 的 **尺度映射**在这些 family 上失败；per-family isotonic 也没有救回来
（M1-L course 2.09、LN 2.14），说明是 `d_m` 的分布/尺度本身不匹配，而不是折分问题。

### 5.5 P4 5 seed 集成（稳健性）

| 组合 | MAE | RMSE | Spearman | MAE 95% CI |
|---|---:|---:|---:|---|
| M1-L | 0.7348 | 1.4214 | 0.8432 | [0.6733, 0.8006] |
| M1-C | 0.8126 | 1.6293 | 0.8447 | [0.7444, 0.8861] |
| M2-L | 1.0632 | 1.8089 | 0.6746 | [0.9945, 1.1406] |
| M2-C | 0.9963 | 1.7649 | 0.7103 | [0.9234, 1.0809] |

与全量模型同量级；M1 的 legacy/canon 差异仍存在，M2 仍明显更差。

### 5.6 网页口径可视化

按 benchmark 网页的 `stats.js` 口径复刻（exact≤0.2、close≤0.5、moderate≤1.0、miss>1.0；
MAE/RMSE/bias/median/P90/coverage、doughnut、expected-got scatter、delta 分布、
trend、pattern MAE、head-to-head |delta|）：

| 图 | 内容 |
|---|---|
| [M1L_vs_Sunny_ALL.png](../results/benchmark/viz/M1L_vs_Sunny_ALL.png) | M1-L vs published Sunny，全部 1,382 行 |
| [M1L_vs_M1C_ALL.png](../results/benchmark/viz/M1L_vs_M1C_ALL.png) | legacy vs canon（M1） |
| [M1L_vs_M2L_ALL.png](../results/benchmark/viz/M1L_vs_M2L_ALL.png) | external MLP vs CM3P-only MLP |
| [M1L_vs_Sunny_RC.png](../results/benchmark/viz/M1L_vs_Sunny_RC.png) | RC 子集（1,246 行） |
| [M1L_vs_Sunny_GE12.png](../results/benchmark/viz/M1L_vs_Sunny_GE12.png) | expected ≥ 12 子集（1,142 行），子集内重拟合 isotonic |
| [M1L_vs_M1C_GE12.png](../results/benchmark/viz/M1L_vs_M1C_GE12.png) | expected ≥ 12：legacy vs canon |
| [summary_mae_ge12.png](../results/benchmark/viz/summary_mae_ge12.png) | expected ≥ 12 的 4 组合 + 基线 MAE，以及全量低/高段误差分解 |
| [summary_mae.png](../results/benchmark/viz/summary_mae.png) | 4 组合 + 单信号基线的 overall / RC / family MAE |

生成脚本：[scripts/render_benchmark_dashboard.py](../scripts/render_benchmark_dashboard.py)。

### 5.7 expected ≥ 12 子集（post-hoc，子集内重拟合）

这组数字直接回应“下半区数据还没收集到”的观察：只用 `expected >= 12` 的 **1,142 行**
（83%），并在子集内重新做 5 折 chart-group isotonic 校准。**这是事后子集选择，
不能和 §5.1 的全量 MAE 直接比。**

| 方法 | n | MAE | 95% CI | RMSE | bias | Spearman |
|---|---:|---:|---|---:|---:|---:|
| **M1-L** | 1,142 | **0.3194** | [0.3005, 0.3402] | 0.4541 | −0.0013 | 0.7362 |
| **M1-C** | 1,142 | **0.3108** | [0.2931, 0.3298] | 0.4436 | −0.0024 | 0.7415 |
| M2-L | 1,142 | 0.5304 | [0.5007, 0.5622] | 0.7573 | +0.0023 | 0.4538 |
| M2-C | 1,142 | 0.4841 | [0.4565, 0.5142] | 0.6899 | +0.0004 | 0.5161 |
| Sunny-star -> isotonic（label-fit） | 1,142 | **0.2396** | [0.2248, 0.2546] | 0.3551 | −0.0007 | 0.8303 |
| MSD -> isotonic（label-fit） | 1,142 | 0.3990 | [0.3750, 0.4267] | 0.5793 | +0.0030 | 0.6287 |
| 常数（折内 mean） | 1,142 | 0.5830 | — | 0.9141 | 0.0000 | −0.0264 |

**配对 ΔMAE（同一 5 折、同一批 1,142 行）：**

| 对比 | ΔMAE | 95% CI | 读法 |
|---|---:|---|---|
| M1-C − M1-L | **−0.0086** | [−0.0250, +0.0068] | 不显著；12+ 上 legacy/canon 打平 |
| M1-L − Sunny-star isotonic（label-fit） | **+0.0798** | [+0.0640, +0.0949] | M1-L 仍差于 label-fit Sunny |
| M1-L − MSD isotonic（label-fit） | **−0.0796** | [−0.1007, −0.0585] | M1-L 明显好于 MSD |
| M2-C − M2-L | −0.0463 | — | M2 上 canon 略好，但两者都差 |

**分 family（子集内重拟合）：**

| 方法 | RC（1,083） | LN（47） | course（12） |
|---|---:|---:|---:|
| M1-L | 0.298 | 0.769 | 0.466 |
| M1-C | 0.290 | 0.744 | 0.449 |
| M2-L | 0.505 | 1.023 | 0.887 |
| M2-C | 0.457 | 1.004 | 0.898 |
| Sunny isotonic | 0.220 | 0.628 | 0.511 |
| MSD isotonic | 0.362 | 1.167 | 0.711 |

**分 expected 段（M1-L）：**

| 段 | n | MAE |
|---|---:|---:|
| 12–13 | 72 | 0.611 |
| 13–14 | 264 | 0.311 |
| 14–15 | 622 | **0.223** |
| 15+ | 184 | 0.543 |

**为什么全量是 0.7411，而 12+ 是 0.3194？**

- 全量校准下，`expected < 12` 的 240 行 MAE **2.3815**，`expected >= 12` 的 1,142 行
  MAE 0.3963；低段贡献了总误差的绝大部分。
- 只把全量校准的 `got` 截到 12+ 仍有 bias +0.1152；在子集内重拟合后 bias 降到 −0.0013，
  MAE 从 0.3963 降到 0.3194。因此上表用“子集内重拟合”，而不是直接截取全量 `got`。
- 12–13 段仍有 0.611，说明 12 附近是过渡带；14–15 段最好（0.223）。
- 结论：**在 12+ 主区间，M1 与 M2 的差距缩小，M1 已经超过 `msd_overall`，但仍落后
  `sunny_star` 约 0.08；legacy/canon 不再可分。** 全量结果的主要问题是低段分布外推，
  而不是 12+ 区间完全不可用。

---

## 6. 解读

1. **最强的信号不是 CM3P**：`sunny_star` 单信号经 label-fit isotonic 的 MAE 0.337，远好于 4 个组合；
   `msd_overall` 单独也有 0.629。M1 虽然把两者作为输入，但 `d_m` 是 DCR 玩家 ACC 序的
   潜变量，不能直接变成 Dan。**在严格 RC 子集上，M1-L 与未校准的 published Sunny 几乎打平
   （0.5878 vs 0.5805，head-to-head base wins 749 / Sunny wins 497）；网站 RC 口径
   （non-LN，含 course）下是 0.6319 vs 0.5720**；overall 的差距主要来自 course/LN 的 2.1–2.2 MAE。
2. **external 信号显著改善 M1 相对 M2**：M1 Spearman 0.844 vs M2 0.66–0.68，
   MAE 0.74–0.79 vs 1.12–1.15；但 M1 仍差于单信号，说明“加入 external”不等于
   “把 external 当 Dan 用”。
3. **canon 的差异只在全量上显著**：全量 M1-C − M1-L = +0.0507 MAE（CI 不含 0）；
   但在 12+ 子集内是 −0.0086（CI 跨 0），legacy/canon 不再可分。canon 的劣势主要来自
   低段/分布外样本，而不是 12+ 主区间。
4. **course/LN 是全量失败来源，12+ 内仍是最差但量级下降**：全量 course/LN MAE 2.1–2.6；
   12+ 子集内 M1-L 为 course 0.466、LN 0.769（RC 0.298），仍高于 RC 但已不是灾难。
5. **排序与 MAE 不同序**：M1-L 的 Spearman 0.844 高于 `msd_overall` 的 0.785，
   但全量 MAE 0.741 差于 `msd_overall` 的 0.629；12+ 重拟合后 M1-L 反超 MSD
   （0.319 vs 0.399），说明全量差距主要是低段尺度外推。
6. **12+ 子集是更公平的主区间**：在 1,142 行上 M1-L/M1-C 0.319/0.311，已接近
   `sunny_star` 的 0.240；但 12–13 过渡带仍有 0.611，15+ 为 0.543，不能只看 14–15 的 0.223。

---

## 7. 局限

1. **校准用了 benchmark 标签**：主表的 MAE 是“5 折校准后的 out-of-fold”，
   不是完全 label-free 的结果；无标签的 Spearman/Kendall 另行报告。
2. **训练/测试构造不同**：DCR 的 `d_m` 来自 top-1000 玩家 ACC 序；
   benchmark 是社区 Dan 标注。两者相关但不相同。
3. **course/LN 覆盖弱**：DCR 训练集中 LN 很少、course 基本没有；分 family 结果必须单列。
4. **P4 5 seed 只改训练子集**，模型 seed 固定 0；std 是训练子集方差，不是训练随机性。
5. **内容编码为 local no-audio**：本评测不使用 audio-on，不能外推到音频融合表示。
6. **一个谱面做了字段修补**：`b1321` 的 `Bookmarks` 尾逗号；不影响 hit object，但应在复现时记录。

---

## 8. 产物

| 路径 | 内容 |
|---|---|
| `data/processed/benchmark_manifest.parquet` | 1,382 行 manifest |
| `data/processed/benchmark_signals.parquet` | `sunny_star` / `msd_overall` |
| `data/processed/benchmark_content_{legacy,canon}.parquet` | 1,382 × 512 内容表 |
| `results/benchmark/models/*_full.pt` | 4 个全量模型 |
| `results/benchmark/models/*_p4_s{1..5}.pt` | 20 个 P4 模型 |
| `results/benchmark/pred/*.parquet` | 4 个全量模型预测 |
| `results/benchmark/pred_p4/*.parquet` | 4 个 P4 集成预测 |
| `results/benchmark/eval_full/` | 主校准/评测、候选 CSV |
| `results/benchmark/eval_full/rc_summary.csv` | 只看 RC 的 MAE + bootstrap CI |
| `results/benchmark/eval_full/family_summary.csv` | 按 family 的 MAE/RMSE |
| `results/benchmark/eval_full/subset_ge12_oof.parquet` | expected ≥ 12 的逐行 refit got / 原始 d |
| `results/benchmark/eval_full/subset_ge12_metrics.csv` | expected ≥ 12 的 refit/全量校准指标 |
| `results/benchmark/eval_full/candidates_ge12/` | expected ≥ 12 的候选 CSV（含 Sunny/MSD isotonic 对照） |
| `results/benchmark/viz/*_GE12.png` | expected ≥ 12 的网页口径 dashboard |
| `results/benchmark/eval_p4/` | P4 稳健性评测 |
| `scripts/benchmark_*.py` | manifest / signals / encode / infer / calibrate_eval |

**候选提交 CSV** 在 `results/benchmark/eval_full/candidates/`，共 4 份；
默认**没有**复制到 benchmark 仓库、没有改 `results/index.json`、没有 push。

---

## 9. 复现

~~~bash
python scripts/benchmark_manifest.py
python scripts/benchmark_signals.py --force
# 内容编码：HPC 上 scripts/slurm/benchmark_encode_array16.slurm（CPU，16 shards × 2 crops）
python scripts/run_benchmark_predict.py
python scripts/benchmark_calibrate_eval.py \
    --pred-dir results/benchmark/pred --out-dir results/benchmark/eval_full
python scripts/benchmark_calibrate_eval.py \
    --pred-dir results/benchmark/pred_p4 --out-dir results/benchmark/eval_p4
~~~
