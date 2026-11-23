# Random 合并集：MLP 波次 + CM3P 分段聚合特征（计划 v2）

> 2026-11-20。两件事：(A) MLP 档的结论；(B) CM3P 窗口聚合加特征
> （max / std / 分位数），看能不能在 P2/P4 上再降 NLL。
> 基线：P2 0.58326、P4 0.58271（linear_extra，5 split seeds，legacy 内容表）。
>
> v2 相对 v1 的变化：Part A 已跑完并确认 MLP 负结果；Part B 改为
> **per-seed A0 head + canon 内部配对**，去掉绝对门限；A0L 作为
> `log n`/`log mean_norm` 的控制；P4 单独拟合 train-only head。

---

## 0. 摘要

1. **Part A（MLP）已结束，结论是负的，MLP 关闭**：合并集上
   `mlp_extra` 比 `linear_extra` 差 **+0.0050（P2）/ +0.0091（P4）**，
   5/5 seed 同向；`mlp` 比 `linear` 在 P2 好 0.005 左右、P4 混合。
   混合 random10k 后 MLP 没有收益，**后续 Part B 只用 `linear_extra`，
   不再跑 MLP**。MLP 的 eval 曾因 runner 的 skip 逻辑被静默跳过，
   已用 `signals/<model>.parquet` 重新合并并重跑（`signals_all.parquet`）。
2. **窗口库的原料已经在了**：HPC `.cache/cm3p/win` 有 21,949 张图的窗口向量，
   合并集 20,388 张图 **100% 覆盖**，共 **201,766** 个窗口
   （float32 ≈ 0.413 GB，float16 ≈ 0.207 GB）。A0/A0L/A1 不需要 head，
   一次算完；A2/A2b/A2c/A3 的 z 统计量需要 per-seed head。
3. **crop 约束**：窗口库是 canon；当前主线内容表是 legacy。
   B 部分只在 **canon 内部**比较（A2b vs A0 / A0L），不拿 legacy 的绝对
   NLL 当门槛；如果 canon 内部有稳定增益，再讨论是否重编码 legacy 窗口。
4. **B 部分结果（2026-11-20，已跑完）**：A0L（log n + log mean_norm）在
   P2/P4 都稳定小赢（−0.00076 / −0.00079，5/5）；A2b 只在 P2 有
   −0.00182（vs A0L，5/5），P4 为 +0.00069（0/5，更差）；A2c/A3 无额外收益，
   A1 更差。canon 本身的代价（+0.0125 / +0.0142）大于池化增益，
   **不触发 B4 的 legacy 重编码**。完整表格见
   [randommerge_pool_results.md](randommerge_pool_results.md)。

---

## 1. Part A：MLP 波次（已完成，负结果）

| 项 | 值 |
|---|---|
| job | 14022821（`scripts/slurm/randommerge_mlp.slurm`，array 0–19） |
| 映射 | ID 0–9: `mlp_extra`；10–19: `mlp`；ID%10 0–4 → P2 seed 1–5，5–9 → P4 seed 1–5 |
| 超参 | h64 ×2 层、dropout 0.1、λ1e-2、lr 3e-3、98 epoch、5 折、模型 seed 0 |
| 实测墙钟 | P2 单模型 558–812 s（~10–13 min）；P4 88–111 s |

**官方 eval（5 seeds，linear_extra 为对照）**

| 协议 | mlp_extra | linear_extra | Δ(mlp_extra−linear_extra) | 逐 seed |
|---|---:|---:|---:|---|
| P2 | 0.58824 | 0.58326 | **+0.0050** | +0.0066, +0.0048, +0.0045, +0.0045, +0.0046 |
| P4 | 0.59178 | 0.58271 | **+0.0091** | +0.0063, +0.0063, +0.0148, +0.0095, +0.0085 |

**门 A：不通过，MLP 关闭。** `mlp_extra` 在两个协议上都更差，不能用
"1k 上 −0.011"外推。结论：**在 random10k 混合集上，MLP 相对于线性没有收益；
Part B 的特征工程一律用 `linear_extra`，不再为 MLP 排任务。**
（记录：MLP 用 `--val-frac 0` 固定 98 epoch；P2 s1 的 train NLL 到第 98 epoch
仍在下行而 test 为 0.5895，若以后复查，先做 player-cluster 验证/early-stop 诊断。）

**runner 的 eval bug（已修）**：`run_p2_p4_multiseed.py` 的
`assemble()` 会用本次 `--models` 覆盖 `signals.parquet`，
`evaluate()` 看到 `rank_eval.json` 已存在就 skip；MLP 波每个任务只跑一个模型，
因此 eval 被跳过、`summary.csv` 不会自动合并。修法见 §5：
`--assemble-only` / `--force-eval` / `--content-col`，并把四个模型的
`signals/<name>.parquet` 一起 merge 后再 eval。

---

## 2. Part B：CM3P 分段聚合特征

### 2.1 现在怎么聚合的

每张图的 512 维向量 = 所有 16 s 窗口向量的**平均**再归一化（A0）。
仓库里已有 T3 变体（`dcr/cm3p_pool.py`），但 1k 上从未跑过；本计划在
合并集上第一次跑。

| 变体 | 内容 | 宽度 | 依赖 head |
|---|---|---:|---|
| A0 | 窗口均值再归一化（对照） | 512 | 否 |
| A0L | A0 + log mean_norm + log n（元数据对照） | 514 | 否 |
| A1 | A0 + 逐维 std | 1024 | 否 |
| A2 | A0 + mean/sd/q90/q99/max(z) + logs | 519 | 是 |
| A2b | A0 + mean/sd/q75/q95/max(z) + logs | 519 | 是 |
| A2c | A0 + mean/sd/q75/q90/q95/q99/max(z) + logs | 521 | 是 |
| A3 | A2 + 最长高压段 / 最长休息段 | 521 | 是 |
| A4 | 按难度 softmax 加权的窗口均值 | 512 | 是 |
| A5 | minacalc 的区间聚合 | 512 | 否 |
| A6 | 窗口序列上的小 TCN（输出一个信号） | 512 | 是 |

A2b/A2c/A3 的 z 统计量用 A0 的 linear head：
`z_t = w · (W_t − μ)/σ`。**μ/σ 是 content-only，可以跨 seed 共享；
真正 label-dependent 的是 w**。A0/A0L/A1/A5 只依赖窗口矩阵，label-free，
一次算完；A2/A2b/A2c/A3/A4 是 (窗口, head) 的函数，per-seed 重算。
没有 q50：它和 mean/median 高度冗余；A2 vs A2b 宽度相同（519），
干净地比较 q90/q99 和 q75/q95 的位置；A2c 把两组都加上。

### 2.2 head 口径（v2 的关键修正）

**P2**：每个 split seed 单独拟合 5 折 heads，使用与该 seed 下游模型**同一个**
`--in` split 和 `--chart-seed 0`：

```
step2_cm3p_rank.py --in step1_randommerge_c_p2_s{seed}.parquet \
  --content-path .../a0.parquet --content-col features \
  --signal-table data/processed/rank_signals_1k.parquet \
  --model linear --chart-only --chart-folds 5 --chart-seed 0 \
  --val-frac 0 --lam 1e-3 --name a0_heads_p2_s{seed} \
  --dump-heads .../a0_heads_p2_s{seed}.npz \
  --out-signals .../a0_signal_p2_s{seed}.parquet \
  --out-json .../a0_signal_p2_s{seed}.json
```

这样评测图 b 用自己 fold 的 `w_k`，`w_k` 排除了整个 fold k；
剩下的 train-feature 泄漏（训练图 c 的 `w_j` 见过 fold k 的 train 行）
是 target-encoding 的已知问题，先用诊断量化，不默认上 nested。

**P4**：P4 的 train（6×6）和 test（4×4）不相交。**不能复用 P2 的 head**：
P2 的 train 行覆盖 P4 test 图，b 的 `w` 会包含其他 P4 test 图的标签。
每个 P4 seed 单独在 **P4 train 行**上拟合，特征用 `--head-mode full`
（所有图用 `w_full`，只在 P4 train 行上拟合）：

```
step2_cm3p_rank.py --in step1_randommerge_c_p4_s{seed}.parquet \
  --content-path .../a0.parquet --content-col features \
  --model linear --chart-only --chart-folds 5 --chart-seed 0 \
  --val-frac 0 --lam 1e-3 --name a0_heads_p4_s{seed} \
  --dump-heads .../a0_heads_p4_s{seed}.npz ...
```

`a0_heads*.npz` 现在会记录 `inp/chart_seed/lam/val_frac/model`，便于事后核对。

**已知残余泄漏**：P2 用 `w_j` 给训练图算 z，而 `w_j` 在 folds≠j 的 train 行上
拟合，包含评测 fold k 的 train 行。计划里不再声称"P2 完全干净"，而是：
**self-label-free（评测图自己的 fold 被排除）；train-feature 对 held-out fold
的依赖用诊断量化**。诊断：对一个 fold k，比较 `w_j`（含 k）和 `w_j`（去掉 k）
算出的 z 特征；若相关性 >0.999、下游 NLL 差 <1e-4，则可忽略；否则对 A2b 做
nested（5× 成本，只对最终胜者）。

### 2.3 实验矩阵（canon 内部，linear_extra，5 split seeds）

| 阶段 | 内容 | 成本 |
|---|---|---|
| B0 | 共享 A0/A0L/A1 特征（label-free） | ~10 min |
| B1 | 每 seed heads + A2/A2b/A2c/A3 特征（10 个 seed） | P2 ~20 min/seed，P4 ~7 min/seed；并行 |
| B2 | 每 seed 7 个变体 × `linear_extra`（P2 5 折 chart-disjoint，P4 单次） | P2 ~22 min/变体，P4 ~7 min；70 任务并行 |
| B3 | 每 seed 一次 rank eval + paired delta（A2b/A2 等 vs A0/A0L） | ~16 min/seed |
| B4（条件） | 若 canon 内 A2b 有稳定增益，再重编码 legacy 窗口并跑 A2b | **未触发**：P2 A2b−A0L −0.00182 但 P4 +0.00069；canon 代价 0.0125/0.0142 远大于增益 |

**不做绝对门限**。主判定是 canon 内部、同一 seed、同一 test pair 集上的
**paired ΔNLL**：A2b−A0、A2b−A0L、A1−A0、A2−A0L、A2c−A0L、A3−A0L。
报告 mean ± sd、逐 seed 符号（≥4/5）和 95% CI。A0L 用来隔离
`log n`/`log mean_norm`；若 A0L−A0 ≈ 0，在表里注一句即可。
`log n` 是 crop-independent 的，所以如果它有用，不需要 B4 的 GPU 重编码，
直接给 legacy 表加一列即可。

### 2.4 代码与驱动

- `dcr/cm3p_pool.py`：新增 `a0l`/`a2b`/`a2c`；`HeadBundle.weight_for/z`
  支持 `mode="fold"|"full"`；`pool_chart/build_feature_table` 透传 `head_mode`。
- `scripts/cm3p_pool.py`：`--head-mode {fold,full}`；a2b/a2c 的 head 检查。
- `scripts/step2_cm3p_rank.py`：`--dump-heads` 记录 `inp/chart_seed/lam/val_frac/model`。
- `scripts/run_p2_p4_multiseed.py`：`--content-col`、`--force-eval`、
  `--assemble-only`（修好 `--skip-train`）、`--signal-table`、`--cache-dir`。
- `scripts/run_pool_randommerge.py`：Part B 的 per-seed 驱动，四个 stage：
  `shared`（A0/A0L/A1）、`features`（heads + A2/A2b/A2c/A3）、
  `train`（linear_extra）、`eval`（rank eval + paired）、`summarize`。
- slurm：`randommerge_pool_shared.slurm`、`randommerge_pool_features.slurm`
  （10 任务）、`randommerge_pool_train.slurm`（70 任务，`%12`）、
  `randommerge_pool_eval.slurm`（10 任务），按 afterok 依赖串起来。

---

## 3. 时间线（已完成，2026-11-20 10:35 CST）

1. 本地改代码 + 单元测试 + 合成数据 smoke（不碰真实窗口库）。
2. 同步到 HPC，先跑 `shared`（A0/A0L/A1）和 P2 seed 1 的
   `features` 做真实数据冒烟，核对宽度、head 元数据、feature 覆盖。
3. 通过后提交 `features`（10）+ `train`（70）+ `eval`（10）全量 array；
   Part A 的 eval 已经重跑过，只需跑一次 `--aggregate-only` 更新 summary。
4. 汇总 canon 内部 paired Δ；B4 未触发（见 §0.4 / 结果文档）。

---

## 4. 执行记录（2026-11-20）

| 阶段 | HPC job | 说明 |
|---|---|---|
| Part A eval | 已完成 | `signals_all.parquet` + 四模型 eval；summary 已更新；MLP 负结果 |
| shared | 14024151 | A0/A0L/A1（label-free） |
| features smoke | 14024152[0] | P2 seed 1，真实窗口库冒烟 |
| features full | 14024174（0-9） | 10 个 (protocol, seed) |
| train | 14024175（0-69%12） | 70 个 (protocol, seed, variant)，linear_extra only |
| eval | 14024176（0-9） | rank eval + paired delta |
| summarize | 14024177 | summary.csv / summary_pairs.csv / summary.md |

依赖顺序：shared → features smoke → features full → train → eval → summarize。
smoke 失败时后续任务会被 Slurm 的 afterok 依赖挡住。
**全部 COMPLETED（2026-11-20 10:35 CST）**；smoke 的 P2 s1 头拟合 1548 s，
真实数据宽度核对通过（a2/a2b 519，a2c/a3 521）。

---

## 5. 风险

| 风险 | 处理 |
|---|---|
| P2 head 的 train-feature 泄漏 | 每 seed 对齐 5 折 head；诊断 `w_j` 去掉 fold k 的敏感性；必要时只对 A2b 做 nested |
| P4 复用 P2 head 会泄漏 test 标签 | P4 单独在 train 行上拟合，`--head-mode full`；不复用 |
| A0L/log n 混淆 | A0L 实测 −0.00076/−0.00079（5/5），量小但一致；log n 是 crop-independent，可单独加到 legacy 做 513 维验证（无 GPU） |
| canon/legacy 混淆 | B 部分只在 canon 内部比较；组合结论必须等 legacy 重编码 |
| 窗口库只在 HPC | `shared`/features 都在 HPC 跑；特征表（20k × 519 float32 ≈ 42 MB/变体）可拉回本地 |
| MLP 负结果 | 主模型只跑 linear_extra；MLP 关闭，不再排 HPC 任务 |
| 70 任务队列 | `%12` 限流；P4 任务很短；必要时先跑 P2 的 5 个 seed，再跑 P4 |
| 池化只是线性变换的重复 | A1/A2b 是新的统计量；A4/A6 才是非线性的真正尝试 |
