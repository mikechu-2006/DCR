# Random 合并集：canon 窗口池化结果（2026-11-20）

> 数据：20,388 图，P2/P4 各 5 个 split seed（模型 seed 固定 0）。
> 模型：`linear_extra`（λ1e-3）；主线的 legacy 基线：P2 0.58326、P4 0.58271。
> 特征：canon 窗口库；A0/A0L/A1 label-free；A2/A2b/A2c/A3 用 per-seed A0 head
> （P2 5 折 chart-disjoint；P4 只用 P4 train 行，`--head-mode full`）。
> 评测：`step2_rank_eval.py --split test --ceiling-fit all --common-maps
> --strict-pairs --folds 2`；NLL 越小越好。

## 1. 绝对 NLL

| 变体 | P2 | P4 | 说明 |
|---|---:|---:|---|
| legacy linear_extra | 0.58326 ± 0.00145 | 0.58271 ± 0.00514 | 主线，非 canon |
| A0 | 0.59580 ± 0.00146 | 0.59689 ± 0.00592 | canon 窗口均值（对照） |
| A0L | 0.59504 ± 0.00153 | 0.59611 ± 0.00596 | A0 + log mean_norm + log n |
| A1 | 0.59628 ± 0.00133 | 0.59933 ± 0.00505 | A0 + 逐维 std |
| A2 | 0.59341 ± 0.00160 | 0.59675 ± 0.00581 | mean/sd/q90/q99/max(z) + logs |
| A2b | **0.59322 ± 0.00157** | 0.59680 ± 0.00577 | mean/sd/q75/q95/max(z) + logs |
| A2c | 0.59320 ± 0.00159 | 0.59680 ± 0.00576 | A2 ∪ A2b 的分位数 |
| A3 | 0.59345 ± 0.00161 | 0.59674 ± 0.00578 | A2 + 最长高压/休息段 |

**canon 本身的代价**：A0 − legacy = **+0.01254（P2）/ +0.01418（P4）**。
即使 A2b 在 canon 内赢了 0.0026，绝对 NLL 仍比 legacy 差 **+0.00996（P2）/
+0.01409（P4）**。

## 2. 配对 ΔNLL（同 seed、同 test pair 集；Δ < 0 表示前者更好）

| pair | P2 mean ± sd | P2 seeds better | P4 mean ± sd | P4 seeds better |
|---|---:|---:|---:|---:|
| A0L − A0 | **−0.00076 ± 0.00010** | 5/5 | **−0.00079 ± 0.00019** | 5/5 |
| A2b − A0 | −0.00258 ± 0.00021 | 5/5 | −0.00010 ± 0.00023 | 3/5 |
| A2b − A0L | −0.00182 ± 0.00017 | 5/5 | +0.00069 ± 0.00020 | 0/5 |
| A1 − A0 | +0.00048 ± 0.00033 | 0/5 | +0.00243 ± 0.00192 | 0/5 |
| A2 − A0L | −0.00163 ± 0.00015 | 5/5 | +0.00064 ± 0.00018 | 0/5 |
| A2b − A2 | −0.00020 ± 0.00005 | 5/5 | +0.00005 ± 0.00008 | 1/5 |
| A2c − A2b | −0.00001 ± 0.00002 | 3/5 | −0.00000 | 3/5 |
| A3 − A2 | +0.00004 ± 0.00002 | 0/5 | −0.00001 ± 0.00004 | 3/5 |

## 3. 结论

1. **唯一在两个协议上都稳定的是 A0L**（log mean_norm + log n）：
   −0.00076 / −0.00079，5/5 seed 同向。幅度很小，但方向一致。
2. **z 分布统计量只在 P2 有帮助，P4 没有**：A2b 相对 A0L 在 P2 是
   −0.00182（5/5），在 P4 是 **+0.00069（0/5，更差）**。cross-chart 冷启动下
   这些统计量不迁移。
3. **A2b 是 P2 最好的变体**；q75/q95 比 q90/q99 好 −0.00020（5/5），
   但 A2c（两组分位数都加）没有额外收益（−0.00001，3/5），A3 的最长段
   也没有收益（+0.00004，0/5）。
4. **A1（逐维 std）是负结果**：P2 +0.00048、P4 +0.00243，0/5；1024 维的
   二阶矩没有提供可泛化的难度信息。
5. **canon 的代价（+0.0125/+0.0142）远大于池化增益（P2 0.0018）**；
   P4 更是零增益。**不触发 B4（legacy 窗口重编码）**。
6. 如果只想要一个便宜的生产增益：`log n` 是 crop-independent 的，可以给
   legacy 内容表直接加一列（513 维，不用 GPU，~20 min/seed）测一下
   A0 + log n；A0L 的 −0.0008 里有多少来自 `log n` 还没拆开。
   `log mean_norm` 和 z 统计量需要窗口，代价高、收益小，不建议。

## 4. 已知限制

- P2 的 per-seed head 消除了"评测图自己 fold"的泄漏，但训练图的 z 仍然来自
  见过 held-out fold train 行的 `w_j`（target-encoding 的依赖）。P2 的
  −0.0018 可能有少量乐观成分；P4 的零增益说明即使有泄漏也没有强效果。
  计划里的 nested 诊断没有跑；如果要把 P2 的 A2b 写进论文，建议先做一次
  `w_j` 去掉 held-out fold 的敏感性检查。
- P4 test 图的 head 是在 P4 train 行上单独拟合的，没有复用 P2 head；
  A0L 的 P4 −0.0008 是干净的。
- MLP 已关闭；所有结果都是 `linear_extra`。

## 5. 复现

```bash
# 本地冒烟（合成数据）
python3 scripts/test_cm3p_pool.py
python3 scripts/smoke_pool_randommerge.py

# HPC 全量（2026-11-20，全部 COMPLETED）
bash scripts/slurm/randommerge_pool_submit.sh
# 14024151 shared -> 14024152 smoke -> 14024174 features ->
# 14024175 train (0-69%12) -> 14024176 eval -> 14024177 summarize
```

产物：`results/randommerge_pool/summary.md`、`summary_pairs.csv`、
`summary_mean_std.csv`、`{p2,p4}/s{seed}/rank_eval.json`、`paired_delta.json`、
`data/processed/chart_pool_features_randommerge/`。

## 6. 后续

下一轮把 A2b + log + OD 应用到 `linear_extra`，在 P2/P4 下同时看 NLL 和
Pearson/Spearman；计划见 [randommerge_a2b_log_od_plan.md](../legacy/docs/plans/randommerge_a2b_log_od_plan.md)（已归档）。
注意 A2b 已经包含 log，主臂是 `A2b + OD`；OD 在 1k rank eval 里是 0 信号，
先验很低。

**L0–L3 已跑完（2026-11-20）**：`log_n` 在 legacy 上 P2 −0.00079 / P4 −0.00069
（5/5），Pearson/Spearman +0.0005–0.0007；OD 近噪声（控制 `log_n` 后
−0.00013/−0.00006）。结果见 [randommerge_od_legacy_results.md](randommerge_od_legacy_results.md)。
生产建议：legacy `linear_extra` 加 `log_n`，不加 OD。
