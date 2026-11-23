# CM3P mania-column probe（canon, audio-off, P4）

> 2026-11-19。问题：把上游从未发出的 `[MANIA_COLUMN_*]` 打开（opt-in 1-based patch），
> 冻结 CM3P 的 512 维向量能否在 P4 难度序上带来增益？
>
> **结论：没有增益。** 主实验四个方法全部为正 ΔNLL；线性方法 5/5 个划分种子变差。
> 排除截断图的敏感性上四个方法仍然 5/5 变差，幅度更大，说明不是截断造成的。

## 产物

| 文件 | 内容 |
|---|---|
| `../../data/processed/chart_content_cm3p_local_canon_columns_on.parquet` | column-on 内容表，6,828 图，source=1，audio-off，canon |
| `../../data/processed/chart_content_cm3p_local_canon_columns_on.parquet.meta.json` | cfg_hash / crop / columns / 截断统计 |
| `../../.cache/cm3p/index_canon_columns_on.parquet` | 窗口索引 |
| `../p4_multiseed_canon_columns/` | P4 5-seed 结果 |
| `../p4_multiseed_canon_columns/vs_canon.md` | 与 `results/p4_multiseed_canon` 的配对 Δ |
| `../p4_multiseed_canon_columns/sensitivity/truncation_sensitivity.csv` | 排除截断图的敏感性 |
| `hpc_job_14007057.out` | HPC 作业日志 |

## 设置

- HPC job `14007057`：`gpu1-12`，2026-11-18 23:53:56 → 2026-11-19 00:16:29（约 22.5 min）。
- 命令：`step0c_chart_content.py --source osu-dir --ids-file data/processed/cm3p_ids_step1_6828.csv
  --audio off --crop canon --windows on --mania-columns on`。
- checkpoint 冻结；4K mania；max_length=2000；16 s / 16 s 窗口。
- P4：`step1_1k_pairbest_p4_s{1..5}.parquet`，6,828 图；5 个划分种子，模型种子 0；
  4 个 CM3P 方法与 3 个 fixed 信号；评测 `--ceiling-fit all --common-maps --strict-pairs`。
- baseline：`results/p4_multiseed_canon`（同 canon 表、同划分、同 ceiling）。

## Embedding QC

- 6,828/6,828 图，79,274 窗，覆盖率 100%；`n_failed=0`，`n_no_windows=0`。
- `columns_emitted=true`；column token 合计 19,725,526，note token 合计 19,725,804。
- 逐图 off/on cosine：mean **0.9822**，median 0.9820，p05 0.9753，min 0.9686。
- 跨图 cosine：off mean 0.9573 / median 0.9723；on mean 0.9683 / median 0.9792。
- PCA participation ratio：2.386 → 2.564；PCA1 0.613 → 0.592。表示被改变但没有塌缩。
- 截断：**150 图 / 434 窗**，丢 27,919 个 core note token；`n_windows` 与 canon off 逐图完全一致。
- `checksum_ok` 全 True。

## 与已有 checkpoint row 探针一致

`experiments/cm3p_column_probe/summary.json`（2026-11-05）直接检查了 checkpoint 的
token embedding 行：`[MANIA_COLUMN_1..18]` 的 embedding norm 均值 1.042，
与 660 个 known-unused token 的 1.040 统计上不可区分（KS p=0.891，MW p=0.559）；
与 622 个 known-used token 的 1.368 则显著不同（p≈1e-33）。decoder 行和 bias 也
与 unused 行接近（KS p=0.141 / 0.396）。

结论：**published checkpoint 从未训练过 column 行**。这与本次 P4 的负结果完全一致，
也说明“直接打开 column token”本质上是在喂未训练的 OOV 行，而不是在注入可用特征。

## P4 主结果（5 seeds，Δ = on − off；负值才更好）

| 方法 | canon off | canon+column | ΔNLL | seeds 变好 | ΔPearson | ΔSpearman |
|---|---:|---:|---:|---:|---:|---:|
| CM3P + external 线性 | 0.60335 | 0.60581 | +0.00246 ± 0.00146 | 0/5 | −0.0155 | −0.0205 |
| CM3P + external MLP | 0.59425 | 0.59567 | +0.00142 ± 0.00394 | 2/5 | −0.0127 | −0.0109 |
| CM3P 线性 (λ1e-3) | 0.63050 | 0.63244 | +0.00194 ± 0.00132 | 0/5 | −0.0119 | −0.0130 |
| CM3P MLP h64 L2 | 0.63013 | 0.63246 | +0.00232 ± 0.00498 | 2/5 | −0.0083 | −0.0042 |

fixed 三个信号逐位相同；ceiling NLL 完全一致（0.56540 ± 0.00859）。

## 截断敏感性

排除在 column-on 索引里 `n_core_tokens_dropped > 0` 的 147 张图（以及 canon off 的 2 张），
对同一批已训练信号在过滤后的 test 子集上重算 ceiling / NLL。**这是评测子集敏感性，
不是重新训练**；绝对 NLL 与主表不可直接比，只看配对 Δ。

| 方法 | ΔNLL（排除截断图） | seeds 变好 |
|---|---:|---:|
| CM3P + external 线性 | +0.00373 ± 0.00093 | 0/5 |
| CM3P + external MLP | +0.00322 ± 0.00356 | 0/5 |
| CM3P 线性 | +0.00350 ± 0.00119 | 0/5 |
| CM3P MLP h64 L2 | +0.00562 ± 0.00443 | 0/5 |

截断不是主因；排除后负向更一致。

## 解读与局限

1. 冻结 checkpoint 在 canon/audio-off 下没有从 mania column token 获得可泛化的难度信息；
   线性 probe 一致变差，MLP 的种子间符号混合且均值仍为正。
2. 逐图 cosine 0.982 说明 column token 改变了表示，但幅度不大；下游的小幅退化更像是
   OOD token 引入的噪声/分布偏移，而不是“列信息不够强”。
3. 这是**表示探针**，不是“column 无用”的最终结论：没有测试 column embedding 初始化、
   微调、只给 note-start 发 token、7K、audio-on 或更大 max_length。
4. 因为主结果没有正增益，未做 constant/shuffled-column 控制；如果以后要证明
   “是列值而不是额外 token”，再做该控制。
5. P4 的 5 个种子只变划分；模型种子固定 0。std 是划分方差，不是训练方差。
