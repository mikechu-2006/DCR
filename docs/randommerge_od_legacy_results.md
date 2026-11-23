# Legacy L0–L3：log_n / OD 在 linear_extra 上的结果（2026-11-20）

> 数据：randommerge 20,388 图；P2/P4 各 5 个 split seed；模型 `linear_extra`
> （λ1e-3，模型 seed 0）。内容表是主线 legacy `chart_content_cm3p_local.parquet`
> （512，图级均值）；只加 chart-level extra signals，无 GPU、无窗口重编码。
>
> L0 = `msd_overall,sunny_star`（对照）；L1 = + `log_n`；L2 = + `od`；
> L3 = + `log_n, od`。`log_n = log(n_win)`（crop-independent）；
> `od` 来自 `rank_signals_1k.parquet`（0–10）。
> 官方 eval：`--split test --ceiling-fit all --common-maps --strict-pairs --folds 2`。

## 1. 绝对指标

| arm | P2 NLL | P2 Pearson | P2 Spearman | P4 NLL | P4 Pearson | P4 Spearman |
|---|---:|---:|---:|---:|---:|---:|
| L0 | 0.58326 ± 0.00145 | 0.8896 | 0.8829 | 0.58271 ± 0.00514 | 0.8824 | 0.8756 |
| L1 | **0.58247 ± 0.00153** | 0.8902 | 0.8835 | **0.58202 ± 0.00513** | 0.8831 | 0.8761 |
| L2 | 0.58310 ± 0.00140 | 0.8898 | 0.8834 | 0.58263 ± 0.00520 | 0.8826 | 0.8760 |
| L3 | **0.58234 ± 0.00148** | 0.8905 | 0.8839 | **0.58196 ± 0.00518** | 0.8832 | 0.8764 |

**L0 精确复现主线**（P2 0.58326 / P4 0.58271），说明新信号表和评测链路一致。

## 2. 配对 Δ（ΔNLL < 0 更好；ΔPearson/ΔSpearman > 0 更好）

### P2

| pair | ΔNLL | NLL seeds | ΔPearson | P seeds | ΔSpearman | S seeds |
|---|---:|---:|---:|---:|---:|---:|
| L1 − L0（log_n） | **−0.00079 ± 0.00011** | **5/5** | +0.0007 | 5/5 | +0.0005 | 5/5 |
| L2 − L0（OD） | −0.00016 ± 0.00007 | 5/5 | +0.0002 | 5/5 | +0.0005 | 5/5 |
| L3 − L1（OD + log_n） | −0.00013 ± 0.00008 | 5/5 | +0.0002 | 5/5 | +0.0004 | 5/5 |
| L3 − L2（log_n + OD） | −0.00076 ± 0.00010 | 5/5 | +0.0006 | 5/5 | +0.0005 | 5/5 |
| L3 − L0（combined） | **−0.00092 ± 0.00012** | **5/5** | +0.0009 | 5/5 | +0.0009 | 5/5 |

### P4

| pair | ΔNLL | NLL seeds | ΔPearson | P seeds | ΔSpearman | S seeds |
|---|---:|---:|---:|---:|---:|---:|
| L1 − L0（log_n） | **−0.00069 ± 0.00019** | **5/5** | +0.0007 | 5/5 | +0.0005 | 5/5 |
| L2 − L0（OD） | −0.00008 ± 0.00014 | 4/5 | +0.0002 | 5/5 | +0.0004 | 5/5 |
| L3 − L1（OD + log_n） | −0.00006 ± 0.00014 | 3/5 | +0.0001 | 5/5 | +0.0004 | 5/5 |
| L3 − L2（log_n + OD） | −0.00067 ± 0.00019 | 5/5 | +0.0006 | 5/5 | +0.0005 | 5/5 |
| L3 − L0（combined） | **−0.00076 ± 0.00013** | **5/5** | +0.0008 | 5/5 | +0.0009 | 5/5 |

## 3. 结论

1. **`log_n` 是唯一清晰、免费的增益**：L1−L0 = P2 −0.00079（5/5）、
   P4 −0.00069（5/5）；Pearson/Spearman 都 +0.0005–0.0007（5/5），没有
   相关性回退。`log_n` 是 crop-independent 的，直接作为
   `linear_extra` 的 extra signal 即可，不需要 GPU、不需要改内容表。
2. **OD 基本是噪声**：L2−L0 只有 P2 −0.00016、P4 −0.00008；控制 `log_n`
   后 L3−L1 只有 P2 −0.00013（5/5）、P4 −0.00006（3/5）；和 1k rank eval
   里 `od` 是 0 信号的先验一致。**不建议把 OD 加进生产**。
3. **L3（log_n + OD）绝对最好**（P2 0.58234、P4 0.58196），但比 L1 只多
   0.00013/0.00006；这点收益不值得多一个 OD 列和解释成本。**推荐 L1**。
4. 相关性指标和 NLL 同向：L1/L3 的 Pearson/Spearman 都小幅变好；没有
   "NLL 变好但序变差"的冲突。
5. **A2b 部分没有在这次跑**：canon C 臂（A2b+OD）和 legacy L4（legacy A2b）
   都未运行。canon A2b 上一轮 P2 −0.0018（vs A0L）、P4 +0.0007（更差）；
   legacy A2b 还需要 GPU 窗口重编码。基于这次 L0–L3，最划算的下一步是
   **把 `log_n` 加进 legacy 生产**，A2b 的 GPU 重编码暂时不值得。

## 4. 生产建议

在 legacy 的 `linear_extra` 里加 `log_n`：

```
--extra-signals msd_overall,sunny_star,log_n
--signal-table data/processed/rank_signals_randommerge_od.parquet
```

- 内容表保持 `chart_content_cm3p_local.parquet`（512）；
- `log_n` 从 `.cache/cm3p/index.parquet` 的 `n_win` 计算；
- 不需要 GPU、不需要窗口重编码；
- 预期：P2 −0.0008、P4 −0.0007，5/5 seed；Pearson/Spearman 小幅 +0.0005–0.0007。
- OD 不加；L4（legacy A2b）暂缓。

## 5. 复现

```bash
# 新信号表
# data/processed/rank_signals_randommerge_od.parquet
# 训练/评测/汇总
bash scripts/slurm/randommerge_od_legacy_submit.sh
# train 14031717 (0-39%12) -> eval 14031718 (0-9) -> summarize 14031719
```

产物：`results/randommerge_od_legacy/summary.md`、`summary_pairs.csv`、
`summary_mean_std.csv`、`{p2,p4}/s{seed}/rank_eval.json`、`paired_delta.json`。
