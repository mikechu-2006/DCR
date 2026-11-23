# 实验：canon 裁剪内容表在 P2 / P4 上的复跑（2026-11-09）

**问题**：HPC 作业 `13613023`（canon 裁剪 + 窗口导出，见
[cm3p_crop_windows.md](cm3p_crop_windows.md) §6）产出的
`chart_content_cm3p_local_canon.parquet`，在**完全相同的 P2 / P4 协议**下，
与旧 legacy 表 `chart_content_cm3p_local.parquet` 相比，难度排序是更好还是更差？

**结论**：canon 表在 4 个 embedding 方法、2 个口径、5 个划分种子上**一致更差**
（ΔNLL +0.0047 … +0.0208 nats，P2 0/5、P4 0–1/5 个种子变好）。
训练 NLL 也同时变差，所以不是过拟合/正则噪声，而是 canon 表示里**可用的难度信息变少**。
方法排序与既有结论不变（external MLP 仍最好）。

---

## 0. 摘要

* **唯一变量是 content 表**：5 个划分种子文件、`rank_signals_1k.parquet`、ceiling
  缓存、模型超参、模型 seed（0）、评测参数全部沿用
  [p4_multiseed.md](p4_multiseed.md)；fixed 三个信号（Sunny / Daniel / star）不重新
  训练，逐 seed 与旧运行**逐位相同**（最大 |Δ| = 0.00e+00），这同时验证了划分文件没变。
* **canon 一致变差**：P2 上 CM3P-only 线性 +0.0078、MLP +0.0055；external 线性
  +0.0197、external MLP +0.0197。P4 上 +0.0064 / +0.0047 / +0.0186 / +0.0208。
  Pearson / Spearman 同向下降 0.05–0.08。
* **不是训练不充分**：P2 的 train NLL（5 折 × 5 seed 平均）同样变差
  +0.0069（linear）…+0.0172（mlp_extra），说明 canon 向量本身对标签的解释力更弱。
* **方法排序不变**：external MLP 仍然第一；它对 CM3P-only 的优势从 0.0502 缩到
  0.0383（P2）、从 0.0507 缩到 0.0363（P4）。external MLP 相对 external 线性 的
  增益几乎不变（P2 −0.0118 → −0.0118，即两者一起变差同幅度）。
* **解释**：canon 删掉的 hitsound/volume、kiai、SV 三类 token 并不是纯"载荷噪声"，
  它们承载了对难度排序有用、且与 msd/Sunny 互补的信息。canon 与 legacy 的图级向量
  余弦均值只有 0.881（中位 0.884，p5 0.817），差异足够大。
* **归因限度**：canon 表由新一版导出代码生成（除 crop 外还有 4 张图的窗口数不同，
  217,110 vs 217,214 窗）；要严格把"crop"与"代码版本"分开，需要一份
  **用同一版代码跑的 legacy-crop 对照表**（未做，见 §5）。

---

## 1. 设置

| 项 | legacy（对照） | canon（本次） |
|---|---|---|
| 内容表 | `data/processed/chart_content_cm3p_local.parquet` | `data/processed/chart_content_cm3p_local_canon.parquet` |
| 生成 | 2026-10-14，旧版 step0c | 2026-11-06，HPC `13613023`（v6 修复后的 full run） |
| 行 / 维度 | 21,949 × 512，单位范数 | 21,949 × 512，单位范数 |
| md5 | — | `c5d24c20119bcbaef688634fffb212e2` |
| crop | legacy（含 hitsound/volume、kiai、mania SV） | canon（只保留 note + timing，snapping 关闭） |
| 窗口 | 16 s / 16 s，max_length=2000 | 同上 |
| 两表图级余弦 | — | mean 0.8814 / median 0.8838 / p5 0.8165 / min 0.7408 |

协议（与 [p4_multiseed.md](p4_multiseed.md) 完全一致）：

* **P2**：`step1_1k_pairbest_p2_s{1..5}.parquet`，5 折 chart-disjoint，`--chart-seed 0`；
* **P4**：`step1_1k_pairbest_p4_s{1..5}.parquet`，6×6 训练 / 4×4 测试；
* 方法：CM3P 线性（λ=1e-3）、CM3P MLP h64 L2（98 epoch）、CM3P+external 线性、
  CM3P+external MLP（输入 `[C, msd_overall, sunny_star]`）；
* 模型 `--seed 0`、`--val-frac 0`、6 线程；评测 `step2_rank_eval.py --ceiling-fit all
  --common-maps --strict-pairs`；ceiling 逐 seed 与旧运行相同（脚本内断言通过）。

命令（新增 `--content-path`，其余同旧运行；见 [scripts/run_p2_p4_multiseed.py](../scripts/run_p2_p4_multiseed.py)）：

~~~bash
python scripts/run_p2_p4_multiseed.py --protocols p2,p4 --seeds 1,2,3,4,5 --threads 6 --content-path data/processed/chart_content_cm3p_local_canon.parquet --out-dir results/p4_multiseed_canon

python scripts/compare_multiseed_runs.py --baseline results/p4_multiseed --variant results/p4_multiseed_canon --out results/p4_multiseed_canon/vs_legacy.md
~~~

---

## 2. 结果（5 个划分种子的 mean ± std）

Δ = canon − legacy，**负值才是变好**。

### 2.1 P2（新 chart；ceiling NLL = 0.56126 ± 0.00348）

| 方法 | legacy NLL | canon NLL | ΔNLL | seeds 变好 | ΔPearson | ΔSpearman |
|---|---|---|---|---|---|---|
| CM3P + external MLP | 0.56602 ± 0.00307 | 0.58574 ± 0.00153 | **+0.01972 ± 0.00300** | 0/5 | −0.0753 | −0.0779 |
| CM3P + external 线性 | 0.57786 ± 0.00335 | 0.59758 ± 0.00171 | +0.01972 ± 0.00218 | 0/5 | −0.0534 | −0.0666 |
| CM3P 线性 (λ1e-3) | 0.61618 ± 0.00293 | 0.62399 ± 0.00184 | +0.00781 ± 0.00146 | 0/5 | −0.0558 | −0.0687 |
| CM3P MLP h64 L2 | 0.61664 ± 0.00265 | 0.62213 ± 0.00307 | +0.00549 ± 0.00301 | 0/5 | −0.0621 | −0.0679 |

### 2.2 P4（新玩家 + 新 chart；ceiling NLL = 0.56540 ± 0.00859）

| 方法 | legacy NLL | canon NLL | ΔNLL | seeds 变好 | ΔPearson | ΔSpearman |
|---|---|---|---|---|---|---|
| CM3P + external MLP | 0.57343 ± 0.00433 | 0.59425 ± 0.00413 | **+0.02082 ± 0.00356** | 0/5 | −0.0680 | −0.0725 |
| CM3P + external 线性 | 0.58479 ± 0.00494 | 0.60335 ± 0.00515 | +0.01855 ± 0.00385 | 0/5 | −0.0481 | −0.0629 |
| CM3P 线性 (λ1e-3) | 0.62411 ± 0.00345 | 0.63050 ± 0.00270 | +0.00639 ± 0.00552 | 0/5 | −0.0576 | −0.0683 |
| CM3P MLP h64 L2 | 0.62540 ± 0.00653 | 0.63013 ± 0.00266 | +0.00473 ± 0.00500 | 1/5 | −0.0589 | −0.0658 |

### 2.3 逐种子配对 ΔNLL（canon − legacy）

| 方法 | P2 s1 | s2 | s3 | s4 | s5 | P4 s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|---|---|---|---|
| CM3P + external 线性 | +.0197 | +.0181 | +.0174 | +.0204 | +.0230 | +.0152 | +.0155 | +.0242 | +.0171 | +.0207 |
| CM3P + external MLP | +.0198 | +.0184 | +.0178 | +.0177 | +.0249 | +.0149 | +.0207 | +.0232 | +.0238 | +.0215 |
| CM3P 线性 | +.0087 | +.0066 | +.0062 | +.0078 | +.0097 | +.0002 | +.0104 | +.0137 | +.0049 | +.0028 |
| CM3P MLP | +.0074 | +.0003 | +.0073 | +.0055 | +.0069 | −.0010 | +.0076 | +.0116 | +.0042 | +.0013 |

唯一负值是 P4 s1 的 CM3P MLP（−0.0010），其 std 与 0 不可区分。

### 2.4 captured（仅供参考）

| 方法 | P2 legacy | P2 canon | P4 legacy | P4 canon |
|---|---|---|---|---|
| CM3P + external MLP | 0.964 | **0.815** | 0.940 | **0.775** |
| CM3P + external 线性 | 0.874 | 0.725 | 0.850 | 0.704 |
| CM3P 线性 | 0.584 | 0.524 | 0.544 | 0.491 |
| CM3P MLP | 0.580 | 0.539 | 0.534 | 0.495 |

fixed 三个信号在两份运行里逐位相同（0.00e+00），所以 captured 的差完全来自内容表。

### 2.5 训练侧证据（P2，5 折 × 5 seed 的 train NLL 平均）

| 方法 | legacy train | canon train | Δ |
|---|---|---|---|
| CM3P 线性 | 0.59124 | 0.59816 | +0.00691 |
| CM3P + external 线性 | 0.56533 | 0.58032 | +0.01499 |
| CM3P MLP h64 L2 | 0.54516 | 0.55961 | +0.01445 |
| CM3P + external MLP | 0.52610 | 0.54326 | +0.01716 |

训练集上同样变差 → canon 向量的**线性可解码难度信息**变少，不是泛化噪声。

---

## 3. 解读

1. **删掉的三类 token 不是"难度无关载荷"**。T1 只证明了 canon 的裁剪是"只删掉了
   hitsound/volume、kiai、mania SV，note/timing 流不变"，**从没说过删掉之后下游更好**。
   本实验给出下游代价：CM3P-only 退化 0.005–0.008 nats，叠 external 后 0.019–0.021 nats。
2. **canon 损害的主要是与 external 互补的那部分**。用"external 栈 − CM3P-only"衡量
   C 与 S 的互补性：P2 从 −0.0383 降到 −0.0264，P4 从 −0.0393 降到 −0.0272
   （约 −0.012 nats）。这与"kiai/SV/hitsound 编码了段落结构、强度变化"的直觉一致；
   这些信息恰好是 msd/star 类分数看不到的。
3. **既有结论不受影响**：方法排序、external MLP 相对 external 线性的优势
   （P2 −0.0118 nats、5/5 种子为负）在 canon 下都复现；CM3P-only 的 MLP 与线性
   仍基本打平（canon 下 P2 −0.0019、P4 −0.0004，量级远小于 external 的 −0.027），
   只是符号从 legacy 的略微为正变成略微为负，不构成新结论。
   T1 的价值在于口径干净（可解释、可缓存、token 少 43.7%），不是分数更高。
4. **这不是"canon 实现有 bug"的证据**。core audit 64/64 通过、窗口自检 cosine ≥
   0.99999988、覆盖率 100%，表本身自洽；分数差是内容信息量的差。
5. **如果要为下游选表**：当前 rank 目标下应继续用 legacy 表；若要继续 canon 路线，
   T3 的 A1–A6 池化或"canon + 保留 kiai/SV"的消融才有意义。

---

## 4. 局限

* **crop 与代码版本没有完全分离**：canon 表由 11-06 的导出代码生成，除 crop 外还有
  4 张图的窗口数不同（217,110 vs 217,214 窗，0.05%）。严格归因需要同版本代码的
  legacy-crop 对照表（HPC 上约 35 min GPU，未做）。
* 只有 5 个划分种子、模型 seed 固定 0：std 是划分方差；但配对的 40 个
  seed×方法组合里 39 个 ΔNLL 为正（唯一例外是 P4 s1 的 CM3P MLP，−0.0010），
  结论不依赖单个种子。
* 1k top-1000 切片、`n_plays ≥ 2`、ACC 截断等既有口径限制不变。
* P4 s5 的 `CM3P MLP` 单步耗时 1,740 s（其它 seed 约 40–70 s，机器抖动），
  数值与其它种子一致，不影响结论。
* fixed 信号没有重训（本来就不依赖内容表），只在评测里作一致性检查。

---

## 4b. 子集复验（2026-11-09）

在官方发布向量覆盖的 4,042 图子集上（同一 P2/P4 协议、重新划分 5 个种子），
canon − legacy 仍为 **+0.0038 … +0.0187**（8 个方法组合里 6 个 0/5、2 个 1/5 变好），
与本文件全量结论同向。详见 [p4_multiseed_pub.md](p4_multiseed_pub.md) §3.3。

---

## 5. 后续可选

1. **同版本代码的 legacy-crop 对照**（HPC）：`CROP=legacy WINDOWS=on` 跑到
   `chart_content_cm3p_local_legacy_v6.parquet`，再跑本文件同一条命令，就能把
   "crop 效应"与"代码版本效应"分开。
2. **分层检查**：用 `scripts/stratified_delta_nll.py` 按 `count_total` 十层看
   canon 的退化是否集中在高/低流行度段（`cm3p_crop_windows.md` §4.5 的工具）。
3. **kiai/SV 单独消融**：分别保留 kiai 或 SV，看是哪一类 token 在贡献。

---

## 6. 产物

| 文件 | 内容 |
|---|---|
| [`results/p4_multiseed_canon/summary.csv`](../results/p4_multiseed_canon/summary.csv) | 逐 seed × 方法的 NLL / Pearson / Spearman / captured / ceiling |
| [`results/p4_multiseed_canon/summary_mean_std.csv`](../results/p4_multiseed_canon/summary_mean_std.csv) | 5 seed 的 mean/std |
| [`results/p4_multiseed_canon/summary.md`](../results/p4_multiseed_canon/summary.md) | 生成版摘要表 |
| [`results/p4_multiseed_canon/vs_legacy.md`](../results/p4_multiseed_canon/vs_legacy.md) / [`.csv`](../results/p4_multiseed_canon/vs_legacy.csv) | canon vs legacy 的配对 Δ（本文件 §2 的表） |
| `results/p4_multiseed_canon/<p>/s<seed>/rank_eval.json` | 官方评测（7 个信号） |
| `results/p4_multiseed_canon/<p>/s<seed>/signals/` | 每个模型/种子的训练元数据与信号列 |
| `data/processed/chart_content_cm3p_local_canon.parquet` | canon 内容表（md5 `c5d24c20…`），已回传本地 |
| `.cache/cm3p/index.parquet` / `index.meta.json` | 窗口索引与导出统计（21,949 图 / 217,110 窗） |
