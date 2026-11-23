# D2V-L 在 VSRG Dan Benchmark 上的评测（2026-11-23）

> **状态：已完成。** D2V 编码在 HPC GPU（job 14077897，A40）完成；D2V linear_extra
> 训练在 HPC CPU（job 14077896）完成；benchmark 推理、5 折校准和诊断在本机完成。
> 前序：[benchmark_linear_extra.md](benchmark_linear_extra.md)（CM3P RM-L）、
> [data2vec_main/README.md](../data2vec_main/README.md)（D2V 方法）、
> [deploy/README.md](../deploy/README.md)（两个函数的封装接口）。
> Benchmark 快照：VSRG-DanEstimation-Benchmark@e15f89b，1,382 行不变。

---

## 0. 摘要

用 **D2V**（data2vec audio-style，512-d）替换 CM3P 内容向量，其余条件与 RM-L 完全相同
（randommerge 453,406 行 / 20,388 图、linear_extra、λ=1e-3、同一套 5 折 chart-group OOF：

| 方法 | 内容 | MAE | RMSE | bias | Spearman | Kendall | MAE 95% CI |
|---|---|---:|---:|---:|---:|---:|---|
| RM-L（CM3P） | 512-d CM3P legacy | **0.4216** | 0.6440 | −0.0003 | 0.8652 | 0.7332 | [0.3965, 0.4481] |
| **D2V-L（新）** | **512-d D2V audio-style** | **0.4527** | 0.6983 | −0.0013 | 0.8534 | 0.7185 | [0.4270, 0.4821] |
| M1-L（旧） | 512-d CM3P legacy + MLP | 0.7411 | 1.4485 | +0.0069 | 0.8437 | 0.6957 | [0.6810, 0.8054] |
| Sunny isotonic | 单信号 | **0.3367** | 0.5376 | +0.0001 | 0.9060 | 0.7914 | [0.3155, 0.3587] |
| MSD isotonic | 单信号 | 0.6286 | 1.0077 | +0.0012 | 0.7849 | 0.6403 | [0.5882, 0.6697] |

**主结论：**

1. **D2V-L 比 RM-L 差 0.0311 MAE**：0.4527 vs 0.4216，
   配对 Δ = **+0.0311**（95% CI [+0.0076, +0.0531]，**显著**）；
   排除 210 行训练重叠后 0.4735 vs 0.4419，Δ = **+0.0316**（CI [+0.0054, +0.0578]）。
   这不是重叠造成的。
2. **D2V-L 比旧 M1-L 好很多**：0.7411 → 0.4527，Δ = **−0.2883**
   （CI [−0.3505, −0.2311]）；低/中段、LN、course 都明显改善。
3. **但高段反而不如 M1-L**：expected > 15 的 184 行，D2V 0.6232 vs M1-L 0.5548
   （Δ = +0.0684，CI [+0.0057, +0.1312]）；msd≥32 的 669 行，D2V 0.3158 vs M1-L 0.2939
   （Δ = +0.0219，CI [−0.0030, +0.0474]，接近显著）。
   D2V 是“低/中段换高段”的表示，不是全面优于旧模型。
4. **仍不如 isotonic Sunny**：overall Δ = +0.1160（CI [+0.0914, +0.1395]）；
   course 0.5174 vs 0.4763（Δ +0.0411，CI 跨 0，打平）；RC/LN 仍落后。
5. **D2V 与 RM-L 在 LN/course 上打平**：LN 1.0927 vs 1.0733（Δ +0.0194，CI 跨 0）；
   course 0.5174 vs 0.5227（Δ −0.0053，CI 跨 0）；RC 0.3986 vs 0.3655（Δ +0.0331，显著）。
6. **isotonic 仍必要**：D2V 线性校准 0.8649，远差于 isotonic 0.4527；
   raw d_m 的 Spearman 0.8571。

---

## 1. 契约与数据

| 项 | 内容 |
|---|---|
| benchmark | 1,382 行，RC 1,246 / LN 102 / course 34；快照 e15f89b |
| D2V 内容表 | 21,949 图 × 512-d，L2-normalised；source=d2v_main_holdout，d2v_rev=c29431f |
| benchmark D2V 编码 | 1,382/1,382 行，0 失败；n_windows 均值 11.2、范围 4–58 |
| 外部信号 | benchmark_signals.parquet：msd_overall（MinaCalc 505）、sunny_star（448ed7c） |
| 输出 | DCR-D2V-Linear.csv，8 列，1,382 行，无 Missing |

- D2V 的 .osu 解析和 piano roll 与 CM3P 完全独立；benchmark 的 1,382 个 .osu 全部
  解析成功（b1321 的 Bookmarks 尾逗号在 D2V parser 下不报错）。
- benchmark 的 D2V 内容表由 **同一 D2V holdout 编码器** 重新编码，不复用 CM3P 表。
- 与 RM-L 的对照是**单变量对照**：只有内容表示不同；训练行/图、外部信号、λ、
  校准折、评测脚本都相同。

---

## 2. 方法

### 2.1 D2V 编码器

来自 data2vec_main（route B，from-scratch audio-style）：

| 项 | 设置 |
|---|---|
| 输入 | 100 Hz × 8 通道 piano roll（4 onset + 4 hold） |
| 前端 | Conv1d(8→256) → Conv1d(256→256) → Conv1d(256→512, stride 2) → GELU → LayerNorm |
| 主干 | 12 层 / 512-d / 8 heads / FFN 2048；span mask（p=0.065，长度 10 帧） |
| 目标 | EMA teacher 顶层 K=6 层 LayerNorm 平均，只回归 masked 帧 |
| 图向量 | 窗口内有效帧均值，跨窗口按有效帧数加权，最后 L2 normalize |
| checkpoint | data2vec_main/runs/main_holdout/checkpoints/last.pt（HPC） |

benchmark 编码命令（HPC A40）：

~~~bash
PY=$HOME/.conda/envs/DCR/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH="$HOME/DCR/data2vec_main"
$PY -m d2v_main.extract \
  --config data2vec_main/configs/audio_style_base.yaml \
  --ckpt data2vec_main/runs/main_holdout/checkpoints/last.pt \
  --osu-dir data/benchmark_d2v_osu \
  --ids-file data/processed/benchmark_d2v_ids.txt \
  --out data/processed/benchmark_content_d2v.parquet \
  --device cuda --batch-size 32 --num-workers 6 --tag benchmark_d2v
python scripts/finalize_benchmark_d2v.py
~~~

编码日志：1,382 图，0 bad，索引 56.1 s；A40 前向约数分钟。
映射：data/benchmark_d2v_osu/{0..1381}.osu → benchmark_d2v_map.parquet → row_id。

### 2.2 训练

| 项 | 值 |
|---|---|
| 训练表 | step1_randommerge_pairbest_content.parquet |
| 行 / 图 / 玩家 | 453,406 / 20,388 / 12,758 |
| 内容 | chart_content_d2v_holdout.parquet（21,949 × 512） |
| 模型 | linear_extra，λ=1e-3，--fit-all，seed 0 |
| pair / comparisons | 20,158,819 / 39,654,821 |
| 训练 NLL（in-sample） | **0.58152**（CM3P RM-L 是 0.57954） |
| 计算 | HPC emergency_cpu，job 14077896，8 核；**1,128.5 s** |
| 产物 | results/benchmark/models/D2V_L_full.pt（195 KB） |

命令：

~~~bash
python3 scripts/step2_cm3p_rank.py \
  --in data/processed/step1_randommerge_pairbest_content.parquet \
  --content-path data/processed/chart_content_d2v_holdout.parquet \
  --signal-table data/processed/rank_signals_1k.parquet \
  --extra-signals msd_overall,sunny_star \
  --model linear_extra --lam 1e-3 \
  --val-frac 0 --seed 0 --threads 8 \
  --name cm3p_linear_extra --fit-all \
  --save-model results/benchmark/models/D2V_L_full.pt \
  --out-signals results/benchmark/d2v_linear/train/D2V_L_signals.parquet \
  --out-json results/benchmark/d2v_linear/train/D2V_L.json
~~~

### 2.3 校准与自检

- 评测：与 RM-L 完全相同的 5 折 chart-group OOF isotonic；同一 manifest、seed=0、
  bootstrap=1,000。
- G2 自检：训练内容表 20,388 图，推理 d_m 与训练信号 max abs Δ = **2.77e-07**。
- G3：1,382/1,382 行 fold 相同；M1_L 与 RM_L 的 OOF 指标与旧文件逐值一致（max Δ = 0.0）。
- 部署校准：在全 1,382 行上重新拟合 isotonic（固定权重），in-sample MAE 0.4132，
  OOF 0.4527；导出到 deploy/models/d2v_linear_extra.json。

---

## 3. 结果

### 3.1 主表

| 方法 | MAE | RMSE | bias | Spearman | Kendall | MAE 95% CI |
|---|---:|---:|---:|---:|---:|---|
| **D2V-L isotonic** | **0.4527** | 0.6983 | −0.0013 | 0.8534 | 0.7185 | [0.4270, 0.4821] |
| RM-L isotonic | **0.4216** | 0.6440 | −0.0003 | 0.8652 | 0.7332 | [0.3965, 0.4481] |
| M1-L isotonic | 0.7411 | 1.4485 | +0.0069 | 0.8437 | 0.6957 | [0.6810, 0.8054] |
| Sunny isotonic | 0.3367 | 0.5376 | +0.0001 | 0.9060 | 0.7914 | [0.3155, 0.3587] |
| MSD isotonic | 0.6286 | 1.0077 | +0.0012 | 0.7849 | 0.6403 | [0.5882, 0.6697] |
| 常数 median | 1.9195 | 2.8995 | 0.0000 | −0.0094 | −0.0073 | [1.8151, 2.0377] |

线性校准对照：D2V-L linear 0.8649（rho 0.8570）；isotonic 0.4527。
raw d_m 的 Spearman 0.8571；raw MAE 不能报（d_m 只有仿射尺度）。

### 3.2 与 RM-L / M1-L / Sunny / MSD 的配对差

| 子集 | n | vs RM-L ΔMAE [CI] | vs M1-L ΔMAE [CI] | vs Sunny ΔMAE [CI] | vs MSD ΔMAE [CI] |
|---|---:|---|---|---|---|
| all | 1,382 | **+0.0311 [+0.0076, +0.0531]** | **−0.2883 [−0.3505, −0.2311]** | +0.1160 [+0.0914, +0.1395] | −0.1759 [−0.2148, −0.1370] |
| clean | 1,172 | **+0.0316 [+0.0054, +0.0578]** | −0.3229 [−0.3918, −0.2498] | +0.1233 [+0.0973, +0.1473] | −0.1836 [−0.2297, −0.1394] |
| RC | 1,246 | +0.0331 [+0.0092, +0.0568] | −0.1893 [−0.2425, −0.1436] | +0.1088 [+0.0858, +0.1332] | −0.0891 [−0.1174, −0.0601] |
| LN | 102 | +0.0194 [−0.1192, +0.1646] | −1.0188 [−1.3681, −0.6931] | +0.2297 [+0.0914, +0.3783] | −1.0451 [−1.4073, −0.6767] |
| course | 34 | −0.0053 [−0.1553, +0.1490] | −1.7281 [−2.8155, −0.8344] | +0.0411 [−0.1005, +0.1858] | −0.7484 [−0.9773, −0.5190] |
| expected < 11 | 192 | +0.1011 [−0.0107, +0.2206] | −1.4217 [−1.7362, −1.1186] | +0.2408 [+0.1341, +0.3579] | −0.5254 [−0.7193, −0.3344] |
| expected 11–15 | 1,006 | +0.0054 [−0.0139, +0.0260] | −0.1373 [−0.1767, −0.0975] | +0.0776 [+0.0601, +0.0983] | −0.1116 [−0.1477, −0.0786] |
| expected > 15 | 184 | **+0.0990 [+0.0304, +0.1605]** | **+0.0684 [+0.0057, +0.1312]** | +0.1961 [+0.1310, +0.2569] | −0.1627 [−0.2473, −0.0806] |
| msd≥32 | 669 | **+0.0301 [+0.0085, +0.0526]** | +0.0219 [−0.0030, +0.0474] | +0.0965 [+0.0739, +0.1160] | −0.0406 [−0.0668, −0.0159] |

（正数 = D2V-L 更差；全部为 1,000 次 chart-group bootstrap。）

### 3.3 RC / family

| 子集 | n | D2V-L | RM-L | M1-L | Sunny | MSD |
|---|---:|---:|---:|---:|---:|---:|
| all | 1,382 | 0.4527 | 0.4216 | 0.7411 | 0.3367 | 0.6286 |
| clean | 1,172 | 0.4735 | 0.4419 | 0.7964 | 0.3502 | 0.6571 |
| RC | 1,246 | 0.3986 | 0.3655 | 0.5878 | 0.2898 | 0.4877 |
| LN | 102 | 1.0927 | 1.0733 | 2.1115 | 0.8630 | 2.1378 |
| course | 34 | 0.5174 | 0.5227 | 2.2455 | 0.4763 | 1.2658 |
| msd≥32 | 669 | 0.3158 | 0.2857 | 0.2939 | 0.2194 | 0.3564 |
| expected < 11 | 192 | 1.0311 | 0.9299 | 2.4528 | 0.7902 | 1.5564 |
| expected 11–15 | 1,006 | 0.3112 | 0.3058 | 0.4485 | 0.2336 | 0.4228 |
| expected > 15 | 184 | 0.6232 | 0.5241 | 0.5548 | 0.4271 | 0.7858 |

D2V-L 的分 family Spearman：RC 0.8289、LN 0.9738、course 0.9911。
D2V-L 在 expected > 15 的 r2 是 **−0.0093**，bias +0.408（系统性低估）；
这是它整体输给 RM-L 的主要来源。

### 3.4 head-to-head

| 对比 | 子集 | n | D2V wins | 对手 wins | ties | D2V MAE | 对手 MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| D2V vs RM-L | all | 1,382 | 637 | 745 | 0 | 0.4527 | 0.4216 |
| D2V vs RM-L | clean | 1,172 | 534 | 638 | 0 | 0.4735 | 0.4419 |
| D2V vs RM-L | RC | 1,246 | 580 | 666 | 0 | 0.3986 | 0.3655 |
| D2V vs RM-L | expected > 15 | 184 | 80 | 104 | 0 | 0.6232 | 0.5241 |
| D2V vs M1-L | all | 1,382 | 779 | 603 | 0 | 0.4527 | 0.7411 |
| D2V vs M1-L | expected > 15 | 184 | 90 | 94 | 0 | 0.6232 | 0.5548 |
| D2V vs Sunny | all | 1,382 | 483 | 898 | 1 | 0.4527 | 0.3367 |
| D2V vs MSD | all | 1,382 | 851 | 529 | 2 | 0.4527 | 0.6286 |

### 3.5 范围 / 外推

| 序列 | p1 | p5 | p50 | p95 | p99 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 训练 20,388 图 | −2.465 | −1.990 | −0.251 | 2.902 | 4.577 | −3.851 | 6.730 |
| benchmark 1,382 行 | −0.496 | 0.919 | 4.306 | 6.090 | 7.181 | −1.407 | 8.548 |

- 36.5%（504 行）高于训练 p99；2.03%（28 行）高于训练 max；没有行低于 min。
- 高于 p99 的 504 行 MAE 0.376；高于 max 的 28 行 MAE 0.322；
  范围外样本的 MAE 并不比 overall 差，所以范围截断不是 D2V 落后的主因。
- 但 D2V 的 raw d_m 整体更靠上；高段的 isotonic 仍依赖外推。

### 3.6 可视化

| 图 | 内容 |
|---|---|
| [D2V_L_vs_RM_L_ALL.png](../results/benchmark/viz/D2V_L_vs_RM_L_ALL.png) | D2V-L vs RM-L，全部 1,382 行 |
| [D2V_L_vs_Sunny_ALL.png](../results/benchmark/viz/D2V_L_vs_Sunny_ALL.png) | D2V-L vs published Sunny，全部 1,382 行 |

---

## 4. 解读

1. **D2V 是一个可用的 content provider，但不是 benchmark 上最好的**：
   0.4527 比旧 M1-L 好 0.288，但比 CM3P RM-L 差 0.031（显著）。
   这是一个干净的单变量对照——同一批训练行、同一组外部信号、同一个 λ、同一套折；
   差距只能来自内容表示本身。

2. **D2V 的增益和损失按难度分开**：
   低段（<11）1.0311 vs M1-L 2.4528、LN 1.0927 vs 2.1115、course 0.5174 vs 2.2455，
   都大幅改善；但高段（>15）0.6232 vs M1-L 0.5548，**反而更差**；
   msd≥32 也略差（0.3158 vs 0.2939，CI 接近 0）。
   D2V 的 8 通道 piano roll 可能更适合低/中段的节奏/长条结构，在最高段失去了
   CM3P 文本 token 能提供的信息；这只是一个假设，本报告没有做消融证明。

3. **D2V 与 RM-L 在 LN/course 上打平，输在 RC 和高段**：
   LN Δ +0.019（CI 跨 0）、course Δ −0.005（CI 跨 0）；
   RC Δ +0.033（显著）、expected>15 Δ +0.099（显著）。
   如果目标人群以 RC 高段为主，CM3P 更合适。

4. **仍然不如 Sunny**：overall +0.116；course 打平；RC/LN 落后。
   D2V 的 raw d_m 与 sunny_star 的相关性也高（内容/外部信号融合），但它仍然
   输在 Dan 尺度上的条件均值，而不是排序。

5. **isotonic 仍然必要**：0.8649 → 0.4527。只报线性校准会严重低估。

6. **范围不是主因**：36.5% 的 benchmark 行高于训练 p99，但范围外样本的 MAE 不更差；
   D2V 的问题是高段条件均值（bias +0.408），不是简单的端点截断。

---

## 5. 局限

1. **校准用了 benchmark 标签**：主表是 5 折 OOF isotonic；部署函数用全量 1,382 行
   固定拟合（in-sample 0.4132，OOF 0.4527）。
2. **训练目标不同**：D2V d_m 来自玩家 ACC 序的 BT 潜变量，benchmark expected 是
   社区 Dan 标注；两者相关但不同构造。
3. **D2V 预训练/编码的 transductive 成分**：编码器在 1k 的 80% 图上预训练；
   benchmark 的 201 个重叠 bid 可能被预训练见过，其余大部分没有。
   本报告没有把 D2V 的 benchmark 编码图从预训练中完全隔离。
4. **高段失败**：expected>15 的 r2 为负、bias +0.408；不要把 D2V 用于高段人群。
5. **单一模型/单一 split**：linear_extra 对模型 seed 确定，但没有做 P2/P4
   split-seed 的 benchmark 稳健性臂。
6. **内容跨表**：D2V 表 21,949 图、benchmark 单独编码 1,382 图；两者同一编码器/rev，
   但 benchmark 编码是独立的一次前向，不是训练表的子集。
7. **published 算法覆盖率不同**：D2V 0.4527 优于 published Sunny 0.5763、
   Companella 0.4934、P1h 0.8314，但落后 published Daniel 0.4224（83.3% 覆盖）
   和 XDC1/Koishi/Azusa/ISOR/Roxy/Mixed；不要把不同覆盖率的 MAE 直接排名。
8. **不涉及 audio-on**：D2V 是 audio-style 的谱面 piano roll，不是真实音频；
   不能外推到 audio-on 的 CM3P 臂。

---

## 6. 产物

| 路径 | 内容 |
|---|---|
| data/benchmark_d2v_osu/{0..1381}.osu | benchmark .osu 的 D2V 编码 staging（按 row 序号） |
| data/processed/benchmark_d2v_map.parquet | d2v_id → row_id/bid/osu_path/osu_md5 |
| data/processed/benchmark_content_d2v.parquet | 1,382 × 512 benchmark D2V 内容表 |
| results/benchmark/models/D2V_L_full.pt | D2V linear_extra 全量模型（195 KB） |
| results/benchmark/d2v_linear/train/D2V_L_signals.parquet | 20,388 个训练图的 d_m |
| results/benchmark/d2v_linear/train/D2V_L.json | 训练 JSON（NLL 0.58152、1,128.5 s） |
| results/benchmark/d2v_linear/train/D2V_L_selfcheck.parquet | G2 自检（max Δ 2.77e-07） |
| results/benchmark/pred/D2V_L.parquet | benchmark 1,382 行的 d_m |
| results/benchmark/eval_d2v/summary.md / summary.csv | 6 run 的同折 5 折评测 |
| results/benchmark/eval_d2v/oof.parquet | 逐行 OOF got / fold / d_m |
| results/benchmark/eval_d2v/compare/ | D2V vs RM-L/M1-L/Sunny/MSD 的 subset/paired/head-to-head |
| results/benchmark/eval_d2v/candidates/DCR-D2V-Linear.csv | 8 列候选 CSV |
| results/benchmark/viz/D2V_L_vs_RM_L_ALL.png | D2V vs RM-L 仪表盘 |
| results/benchmark/viz/D2V_L_vs_Sunny_ALL.png | D2V vs published Sunny 仪表盘 |
| deploy/models/d2v_linear_extra.json | 封装用的 D2V 权重 + 全量 isotonic 校准 |
| scripts/slurm/benchmark_d2v_extract.slurm | HPC GPU 编码 |
| scripts/slurm/benchmark_d2v_train.slurm | HPC CPU 训练 |
| scripts/finalize_benchmark_d2v.py | 数值 id → row_id 映射 |
| scripts/benchmark_d2v_compare.py | D2V 对比分析 |
| results/benchmark/logs/slurm_bench_d2v_* | HPC 日志 |

**候选 CSV 默认没有复制到 benchmark 仓库、没有改 results/index.json、没有 push。**

---

## 7. 复现

~~~bash
# 0) staging（已做；本地）
#    data/benchmark_d2v_osu/{0..1381}.osu
#    data/processed/benchmark_d2v_ids.txt / benchmark_d2v_map.parquet

# 1) HPC GPU 编码
sbatch scripts/slurm/benchmark_d2v_extract.slurm
python scripts/finalize_benchmark_d2v.py

# 2) HPC CPU 训练
sbatch scripts/slurm/benchmark_d2v_train.slurm

# 3) 本机推理 / 5 折评测 / 对比
python scripts/benchmark_infer.py \
  --model results/benchmark/models/D2V_L_full.pt \
  --content data/processed/benchmark_content_d2v.parquet \
  --signals data/processed/benchmark_signals.parquet \
  --out results/benchmark/pred/D2V_L.parquet

python scripts/benchmark_calibrate_eval.py \
  --runs D2V_L,RM_L,M1_L,M1_C,M2_L,M2_C \
  --pred-dir results/benchmark/pred \
  --out-dir results/benchmark/eval_d2v

python scripts/benchmark_d2v_compare.py \
  --oof results/benchmark/eval_d2v/oof.parquet \
  --model results/benchmark/models/D2V_L_full.pt \
  --train-signals results/benchmark/d2v_linear/train/D2V_L_signals.parquet \
  --out-dir results/benchmark/eval_d2v/compare

# 4) 封装 JSON + parity
python scripts/export_dan_deploy.py \
  --model results/benchmark/models/D2V_L_full.pt \
  --pred results/benchmark/pred/D2V_L.parquet \
  --content-type d2v \
  --out deploy/models/d2v_linear_extra.json \
  --oof results/benchmark/eval_d2v/oof.parquet --oof-col D2V_L_isotonic
python deploy/test_parity.py
~~~

---

## 8. 一句话结论

**D2V-L 在 benchmark 上 0.4527，明显优于旧 M1-L（0.7411），但比 CM3P RM-L（0.4216）
差 0.031 MAE；它的增益在低/中段和 LN/course，高段反而比 M1-L 更差。
D2V 适合作为第二路 content provider，不适合取代 CM3P 做高段模型。**
