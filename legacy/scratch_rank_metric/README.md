# legacy/scratch_rank_metric —— rank-NLL 开发期的临时探针

2026-10-31 开发 [`dcr/rank_metrics.py`](../../dcr/rank_metrics.py) 与
[`scripts/step2_rank_eval.py`](../../scripts/step2_rank_eval.py) 时用的一次性脚本和中间结果。
方法论与最终数字已经沉淀到 [docs/metrics.md](../../docs/metrics.md) 与
[docs/rank_eval.md](../../docs/rank_eval.md)，本目录只留档，**不再维护，也不保证可运行**。

## 内容

| 文件 | 当时的用途 |
|---|---|
| `probe1.py` | 第一版：把 pair 聚合成 BT counts，比较 in-sample 与 split 的 b 拟合 |
| `probe2.py` / `probe3.py` | 逐玩家 Spearman、random/star/msd/notes/bpm 的 NLL 与 bhat 相关 |
| `probe_bt.py` | 直接用 `b` 的 BT 拟合（ceiling）与 per-player 口径对照，产出 `bhat*.npy` |
| `probe_diag*.py` | 诊断 ceiling 欠拟合、fold 覆盖与 `captured>1` |
| `probe_side.py` | 副产物：按玩家分组的统计 |
| `small*.json` / `train_rank*/` | 小样本（`step1_small.parquet`）上的训练与评测烟测 |
| `rank_eval_1k_msd_insample.json` | 旧口径（in-sample ceiling）下 MSD 的评测，已被 `--ceiling-fit all` 版本取代 |
| `step1_dup.parquet` / `step1_small.parquet` | 当时的输入切片（大文件由 .gitignore 忽略） |

> 这些脚本读 `data/processed/step1_1k_pairbest.parquet` 等路径，按迁移前的仓库根书写；
> 现在直接运行可能因输入缺失而失败。要复现结论请用：
>
> ```bash
> python scripts/step2_rank_eval.py --help
> python scripts/test_rank_metrics.py
> ```
