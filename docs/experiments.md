# DCR 实验总表（2026-11-23 汇总）

> **范围**：把仓库里已有的实验产物按统一口径汇总；**没有训练、推理或重新校准任何模型**。
> 缺失的格子写“—”，并在 §5 列出缺口；没有用新实验去补。
>
> **数据源**：results/ 下各实验目录的 summary*.csv / summary*.md / rank_eval*.json，
> 以及 docs/ 里已写好的结论。主要表格都给了来源链接。
>
> **口径提醒**：1k / 6,819 audio-on / randommerge / 4,042 published 子集 / 10K / VSRG benchmark
> 的绝对 NLL、RMSE、MAE **不可互比**；P2 与 P4 的 test 集和 ceiling 也不同。表内只做同列比较。

## 0. 指标与协议

### 0.1 sign-NLL

仓库里没有单独命名为 sign-NLL 的列；本项目实际使用的是**成对 Bradley–Terry NLL**
（docs/metrics.md 中的 **rank-NLL**）：对同一玩家玩过的两张图，用系统难度差预测
“哪张图表现更好”的符号，取对数损失，单位是 nats/comparison，越低越好。
本表把 nll_mean 直接作为 **sign-NLL** 汇报；如果你指的是另一种 sign 指标，
需要重新定义口径，当前数据不能直接套用。

### 0.2 Pearson / Spearman

表中 Pearson / Spearman 是 **per-chart 难度 d_m 与 cross-fitted BT 尺度 b 的相关**
（同一 test 图集），不是逐玩家相关；越高越好。定义见 [metrics.md](metrics.md) §4–§5。

### 0.3 P2 / P4

| 协议 | 含义 | 本表地位 |
|---|---|---|
| **P2** | 新 chart（chart-disjoint），玩家仍在训练集；pair-random 70/30 + 5 折 chart-disjoint | **主口径** |
| **P4** | 新玩家 + 新 chart 双重冷启动（6×6 / 4×4） | 副口径，有则另附 |
| 10K ALL / RC20 / LN | 10K 全量或子集，没有 P2/P4 划分 | 单独表，不与 P2 比 |
| VSRG benchmark | 外部 Dan 标签，5 折 chart-group LOCV | 单独 benchmark |

表里的 mean ± std 默认是 **5 个划分种子**（模型种子固定 0）的均值/标准差；
单划分结果不标 ±，并在备注里注明。每个种子的 train/test 划分不同，所以 std 是划分方差，
不是训练方差。

## 1. 方法索引

| 方法族 | 实现 / 入口 | 类型 | 出现在 |
|---|---|---|---|
| **bias / mf_dot** | step2/predict | loss 回归 oracle；无 rank 信号；mirt 已退役 | 仅 §4（RMSE） |
| **CM3P 线性** | step2/predict + 内容表 | 冻结 512d 内容向量，PCA 后点积 | 1k / audio / canon / published 子集 / randommerge |
| **CM3P 内容锚定 MLP** | step2/cm3p_mlp | 内容塔 + 玩家向量点积 | 同上 |
| **CM3P + external** | 内容表 + msd_overall,sunny_star | 线性 / MLP 两种头 | 1k / audio / canon / randommerge / D2V |
| **data2vec（route B）** | [data2vec_main/](../data2vec_main/README.md) | from-scratch audio-style 自监督 512d 内容 | 1k P2/P4、P4-holdout |
| **randommerge linear_extra** | scripts/step2_cm3p_rank.py | CM3P + msd/sunny(+log_n/OD) 线性 | randommerge P2/P4、benchmark RM-L |
| **窗口池化 A0–A6** | scripts/cm3p_pool.py | 16s 窗口均值 / 分位 / TCN | randommerge P2/P4（A0–A3 已测） |
| **固定信号** | rank_signals_1k.parquet | Sunny / Daniel / star / MSD / MinaCalc | 全部 P2/P4 表 |
| **ISOR / Koishi** | scripts/run_isor_signals.py 等 | 外部 Dan 估计器，映射到 DCR 序 | 10K / randommerge 全量 |
| **published Dan 算法** | benchmark results/*.csv | 外部算法原样预测 | VSRG benchmark 部分覆盖表 |
| **MinaCalc / analyser** | minacalc/、外部 .osu 工具 | 逐图固定信号 | 1k / 10K / benchmark |

## 2. 玩家行为数据集（P2 主口径）

### 2.1 1k pairbest：legacy 内容表，5 seeds

**6,828 图 / 109,706 对**；P2/P4 各 5 个划分种子；内容表 chart_content_cm3p_local.parquet
（audio-off legacy）。P2 为主口径，P4 另附在同一行。

| 方法 | P2 sign-NLL | P2 Pearson | P2 Spearman | P4 sign-NLL | P4 Pearson | P4 Spearman |
|---|---:|---:|---:|---:|---:|---:|
| **CM3P + external MLP** | **0.5660 ± 0.0031** | **0.6281 ± 0.0081** | **0.5853 ± 0.0101** | **0.5734 ± 0.0043** | **0.5881 ± 0.0264** | **0.5426 ± 0.0232** |
| CM3P + external 线性 | 0.5779 ± 0.0033 | 0.5822 ± 0.0122 | 0.5523 ± 0.0086 | 0.5848 ± 0.0049 | 0.5374 ± 0.0391 | 0.5094 ± 0.0326 |
| CM3P 线性 (λ1e-3) | 0.6162 ± 0.0029 | 0.5412 ± 0.0082 | 0.5070 ± 0.0079 | 0.6241 ± 0.0035 | 0.5095 ± 0.0219 | 0.4767 ± 0.0211 |
| CM3P MLP h64 L2 | 0.6166 ± 0.0026 | 0.5475 ± 0.0073 | 0.5146 ± 0.0084 | 0.6254 ± 0.0065 | 0.5116 ± 0.0225 | 0.4763 ± 0.0325 |
| Sunny | 0.6287 ± 0.0021 | 0.3455 ± 0.0081 | 0.3834 ± 0.0019 | 0.6311 ± 0.0078 | 0.3098 ± 0.0608 | 0.3480 ± 0.0492 |
| Daniel | 0.6277 ± 0.0021 | 0.3454 ± 0.0080 | 0.3837 ± 0.0031 | 0.6310 ± 0.0060 | 0.3077 ± 0.0614 | 0.3440 ± 0.0497 |
| star | 0.6615 ± 0.0014 | 0.3088 ± 0.0074 | 0.3369 ± 0.0021 | 0.6602 ± 0.0078 | 0.2813 ± 0.0626 | 0.3090 ± 0.0509 |

来源：[results/p4_multiseed/summary_mean_std.csv](../results/p4_multiseed/summary_mean_std.csv)、
[summary.md](../results/p4_multiseed/summary.md)。ceiling：P2 0.56126 ± 0.00348、P4 0.56540 ± 0.00859。

**读法**：P2 上 external MLP 比 external 线性低 **0.0118**，P4 低 **0.0114**（paired，5/5）；
CM3P-only MLP 与线性基本打平（P2 +0.0005、P4 +0.0013，符号随种子混合）。
**1k 上 MLP 的增益来自 external 特征，不是纯 CM3P 内容。**

### 2.2 内容表变体 I：canon（1k，5 seeds）

同一批 6,828 图、同一批划分，只把内容表换成 canon（裁剪 hitsound/volume、kiai、mania SV）。
固定信号 Sunny/Daniel/star 与 §2.1 逐位相同，不重复列。

| 方法 | P2 sign-NLL | P2 Pearson | P2 Spearman | P4 sign-NLL | P4 Pearson | P4 Spearman | ΔNLL P2 vs legacy | ΔNLL P4 vs legacy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **CM3P + external MLP** | 0.5857 ± 0.0015 | 0.5528 ± 0.0143 | 0.5074 ± 0.0097 | 0.5943 ± 0.0041 | 0.5201 ± 0.0272 | 0.4701 ± 0.0282 | +0.0197 | +0.0208 |
| CM3P + external 线性 | 0.5976 ± 0.0017 | 0.5288 ± 0.0094 | 0.4857 ± 0.0108 | 0.6033 ± 0.0051 | 0.4892 ± 0.0247 | 0.4464 ± 0.0215 | +0.0197 | +0.0186 |
| CM3P 线性 (λ1e-3) | 0.6240 ± 0.0018 | 0.4853 ± 0.0085 | 0.4383 ± 0.0101 | 0.6305 ± 0.0027 | 0.4520 ± 0.0219 | 0.4084 ± 0.0174 | +0.0078 | +0.0064 |
| CM3P MLP h64 L2 | 0.6221 ± 0.0031 | 0.4855 ± 0.0129 | 0.4467 ± 0.0113 | 0.6301 ± 0.0027 | 0.4527 ± 0.0268 | 0.4105 ± 0.0209 | +0.0055 | +0.0047 |

来源：[results/p4_multiseed_canon/summary_mean_std.csv](../results/p4_multiseed_canon/summary_mean_std.csv)。
**canon 在 P2/P4 上一致变差**：4 个方法 × 2 协议中 39/40 个 seed×方法组合为正；
Pearson/Spearman 也全部变差。canon 的价值是 token −43.7% 和可缓存，不是分数。

### 2.3 内容表变体 II：完整 audio-on（6,819 图，新划分，5 seeds）

published 4,037 + HPC 新算 2,782 = **6,819 图**；这份表与 §2.1 的旧 6,828 图/旧划分
**不是同一 test 集**，所以只能看同新划分的 audio − legacy 配对 Δ。

| 方法 | P2 sign-NLL | P2 Pearson | P2 Spearman | P4 sign-NLL | P4 Pearson | P4 Spearman | ΔNLL P2 vs legacy | ΔNLL P4 vs legacy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **CM3P + external MLP** | **0.5585 ± 0.0021** | **0.6551 ± 0.0082** | **0.6260 ± 0.0086** | **0.5673 ± 0.0099** | **0.6057 ± 0.0233** | **0.5757 ± 0.0144** | **−0.0083** | **−0.0076** |
| CM3P + external 线性 | 0.5686 ± 0.0012 | 0.6293 ± 0.0116 | 0.6022 ± 0.0076 | 0.5740 ± 0.0091 | 0.5867 ± 0.0276 | 0.5669 ± 0.0226 | −0.0100 | −0.0108 |
| CM3P 线性 (λ1e-3) | 0.6053 ± 0.0031 | 0.5881 ± 0.0062 | 0.5489 ± 0.0065 | 0.6073 ± 0.0103 | 0.5475 ± 0.0303 | 0.5152 ± 0.0227 | −0.0127 | −0.0123 |
| CM3P MLP h64 L2 | 0.5967 ± 0.0037 | 0.6091 ± 0.0074 | 0.5673 ± 0.0089 | 0.5981 ± 0.0074 | 0.5691 ± 0.0292 | 0.5335 ± 0.0185 | −0.0208 | −0.0218 |

固定信号（同新划分，完整指标）：

| 信号 | P2 sign-NLL | P2 Pearson | P2 Spearman | P4 sign-NLL | P4 Pearson | P4 Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Sunny | 0.6265 | 0.3452 | 0.3810 | 0.6293 | 0.3274 | 0.3544 |
| Daniel | 0.6265 | 0.3458 | 0.3814 | 0.6297 | 0.3260 | 0.3554 |
| star | 0.6600 | 0.3103 | 0.3370 | 0.6595 | 0.2915 | 0.3129 |

来源：results/p4_multiseed_audio/audio/summary_mean_std.csv。
来源：[combined_summary.md](../results/p4_multiseed_audio/combined_summary.md)、
[audio/summary_mean_std.csv](../results/p4_multiseed_audio/audio/summary_mean_std.csv)、
[legacy/summary_mean_std.csv](../results/p4_multiseed_audio/legacy/summary_mean_std.csv)。

**读法**：4 个 embedding 方法在 P2/P4 上都是 **5/5 种子变好**（8/8 个 protocol×method），
audio 表的 CM3P + external MLP 仍是最好；增益在 CM3P-only MLP 上最大（−0.021），
叠 external 后收窄（−0.008）。

### 2.4 randommerge：20,388 图，5 seeds

4 个 random 快照在 step0 合并 → 453,406 对 / 20,388 图 / 12,758 玩家。
内容表仍是 legacy 512 维；linear_extra 的 external 特征是 msd_overall,sunny_star。
P4 比 P2 低 0.00054（基本持平）。

| 方法 | P2 sign-NLL | P2 Pearson | P2 Spearman | P4 sign-NLL | P4 Pearson | P4 Spearman |
|---|---:|---:|---:|---:|---:|---:|
| CM3P + external MLP | 0.5882 ± 0.0015 | 0.8604 ± 0.0014 | 0.8398 ± 0.0032 | 0.5918 ± 0.0050 | 0.8411 ± 0.0120 | 0.8115 ± 0.0199 |
| **CM3P + external 线性** | **0.5833 ± 0.0014** | **0.8896 ± 0.0009** | **0.8829 ± 0.0015** | **0.5827 ± 0.0051** | **0.8824 ± 0.0021** | **0.8756 ± 0.0023** |
| CM3P 线性 (λ1e-3) | 0.6230 ± 0.0014 | 0.8390 ± 0.0015 | 0.8432 ± 0.0022 | 0.6243 ± 0.0026 | 0.8329 ± 0.0036 | 0.8349 ± 0.0049 |
| CM3P MLP h64 L2 | 0.6175 ± 0.0010 | 0.8374 ± 0.0016 | 0.8349 ± 0.0020 | 0.6224 ± 0.0022 | 0.8218 ± 0.0023 | 0.8167 ± 0.0026 |
| Sunny | 0.6045 ± 0.0013 | 0.8614 ± 0.0008 | 0.8601 ± 0.0011 | 0.6039 ± 0.0067 | 0.8547 ± 0.0041 | 0.8551 ± 0.0035 |
| Daniel | 0.6336 ± 0.0006 | 0.8172 ± 0.0008 | 0.7991 ± 0.0007 | 0.6329 ± 0.0053 | 0.8105 ± 0.0065 | 0.7929 ± 0.0070 |
| star | 0.6449 ± 0.0006 | 0.8114 ± 0.0006 | 0.8047 ± 0.0008 | 0.6447 ± 0.0039 | 0.8054 ± 0.0064 | 0.7985 ± 0.0065 |

来源：[results/randommerge_p2p4_hpc/summary_mean_std.csv](../results/randommerge_p2p4_hpc/summary_mean_std.csv)、
[docs/randommerge_p2p4_results.md](randommerge_p2p4_results.md)。
**注意**：randommerge 的 ceiling 欠拟合（captured 1.10–1.23），本表只报 NLL/Pearson/Spearman，
不引用 captured。1k 上 external MLP 的优势没有迁移：这里 external 线性更好，MLP 反而
+0.0050 / +0.0091（5/5 更差）。

#### 2.4.1 randommerge：log_n / OD（linear_extra）

| arm | 特征 | P2 sign-NLL | P2 Pearson | P2 Spearman | P4 sign-NLL | P4 Pearson | P4 Spearman |
|---|---|---|---:|---:|---:|---:|---:|
| L0 | msd,sunny | 0.5833 ± 0.0014 | 0.8896 ± 0.0009 | 0.8829 ± 0.0015 | 0.5827 ± 0.0051 | 0.8824 ± 0.0021 | 0.8756 ± 0.0023 |
| L1 | +log_n | **0.5825 ± 0.0015** | **0.8902 ± 0.0009** | **0.8835 ± 0.0015** | **0.5820 ± 0.0051** | **0.8831 ± 0.0022** | **0.8761 ± 0.0023** |
| L2 | +od | 0.5831 ± 0.0014 | 0.8898 ± 0.0009 | 0.8834 ± 0.0015 | 0.5826 ± 0.0052 | 0.8826 ± 0.0022 | 0.8760 ± 0.0023 |
| L3 | +log_n,+od | 0.5823 ± 0.0015 | 0.8905 ± 0.0009 | 0.8839 ± 0.0015 | 0.5820 ± 0.0052 | 0.8832 ± 0.0023 | 0.8764 ± 0.0023 |

来源：[results/randommerge_od_legacy/summary_mean_std.csv](../results/randommerge_od_legacy/summary_mean_std.csv)。
log_n 的 ΔNLL 是 −0.00079 / −0.00069（5/5）；OD 只有 −0.00016 / −0.00008，接近噪声；
L3 比 L1 只再多 −0.00013 / −0.00006。现行推荐仍是加 log_n，不把 OD 当主特征。

#### 2.4.2 randommerge：canon 窗口池化 A0–A3

全部是 canon 窗口特征 + linear_extra；A0 是窗口均值，A0L 加 log n / log mean_norm，
A1 加逐维 std，A2/A2b/A2c 加 z 分位，A3 加最长高压/休息段。A4–A6（softmax/minacalc/TCN）未跑。

| 变体 | P2 sign-NLL | P2 Pearson | P2 Spearman | P4 sign-NLL | P4 Pearson | P4 Spearman |
|---|---:|---:|---:|---:|---:|---:|
| a0 | 0.5958 ± 0.0015 | 0.8727 ± 0.0011 | 0.8697 ± 0.0016 | 0.5969 ± 0.0059 | 0.8645 ± 0.0045 | 0.8634 ± 0.0039 |
| **a0l** | **0.5950 ± 0.0015** | **0.8735 ± 0.0010** | **0.8705 ± 0.0016** | **0.5961 ± 0.0060** | **0.8651 ± 0.0045** | **0.8641 ± 0.0039** |
| a1 | 0.5963 ± 0.0013 | 0.8708 ± 0.0013 | 0.8656 ± 0.0017 | 0.5993 ± 0.0050 | 0.8592 ± 0.0041 | 0.8556 ± 0.0039 |
| a2 | 0.5934 ± 0.0016 | 0.8746 ± 0.0009 | 0.8718 ± 0.0015 | 0.5967 ± 0.0058 | 0.8647 ± 0.0047 | 0.8635 ± 0.0039 |
| **a2b** | **0.5932 ± 0.0016** | **0.8746 ± 0.0009** | **0.8719 ± 0.0015** | 0.5968 ± 0.0058 | 0.8647 ± 0.0047 | 0.8635 ± 0.0039 |
| a2c | 0.5932 ± 0.0016 | 0.8747 ± 0.0009 | 0.8719 ± 0.0015 | 0.5968 ± 0.0058 | 0.8647 ± 0.0047 | 0.8635 ± 0.0039 |
| a3 | 0.5935 ± 0.0016 | 0.8746 ± 0.0008 | 0.8718 ± 0.0014 | 0.5967 ± 0.0058 | 0.8647 ± 0.0047 | 0.8635 ± 0.0039 |

来源：[results/randommerge_pool/summary_mean_std.csv](../results/randommerge_pool/summary_mean_std.csv)、
[docs/randommerge_pool_results.md](randommerge_pool_results.md)。
A0L 比 A0 好 −0.00076/−0.00079；A2b 在 P2 再 −0.00182（5/5），P4 +0.00069（5/5 更差）；
**全部池化变体仍比 legacy 线性差 0.00996/0.01409 以上**，canon 本身的代价是
+0.0125/+0.0142（A0 vs legacy）。池化只作“窗口信息是否值得”的诊断，不是生产方案。

### 2.5 data2vec（route B）

data2vec 是 from-scratch audio-style 自监督编码器：100 Hz×8 通道 piano roll →
50 Hz/512d → span mask → 12L/512d Transformer + EMA teacher。P2 是 **单划分 + 5 折
chart-disjoint**（不是 5 个划分种子）；P4 是 5 个划分种子。

| 内容表 / 方法 | P2 sign-NLL | P2 Pearson | P2 Spearman | P4 sign-NLL | P4 Pearson | P4 Spearman | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| D2V 线性 | **0.5863** | **0.5829** | **0.5314** | 0.5930 ± 0.0042 | 0.5373 | 0.4840 | P2 单划分 |
| **D2V + external 线性** | **0.5810** | **0.5915** | **0.5417** | **0.5856 ± 0.0046** | **0.5411** | **0.4901** | P4 最好 D2V |
| D2V MLP | — | — | — | 0.5968 ± 0.0028 | 0.5283 | 0.4753 | P2 只跑完 3/5 折 |
| D2V MLP + external | — | — | — | 0.5861 ± 0.0046 | 0.5310 | 0.4784 | P2 只跑完 3/5 折 |
| CM3P canon 线性 | 0.6240 | 0.4902 | 0.4448 | 0.6305 ± 0.0027 | 0.4520 | 0.4084 | 对照 |
| CM3P canon + external | 0.6009 | 0.5347 | 0.4892 | 0.6034 ± 0.0052 | 0.4892 | 0.4464 | 对照 |
| **CM3P legacy MLP + external** | **0.5660 ± 0.0031** | **0.6281 ± 0.0081** | **0.5853 ± 0.0101** | **0.5734 ± 0.0043** | **0.5881 ± 0.0264** | **0.5426 ± 0.0232** | 1k 最强基线 |

来源：[data2vec_main/README.md](../data2vec_main/README.md)、
[results/d2v_main/eval/](../results/d2v_main/eval/)、
[p4_d2v/summary.md](../results/d2v_main/p4_d2v/summary.md)、
[p4_d2v_mlp/summary.md](../results/d2v_main/p4_d2v_mlp/summary.md)。

**读法**：D2V 在 P2 比 canon+线性低 0.0377、P4 低 0.0375（P4 paired 5/5），
但它**只赢 canon + 线性**；换成 legacy + external MLP，P2/P4 仍差
0.0150/0.0122 左右（D2V+ext 0.5810/0.5856 vs 0.5660/0.5734）。
VSRG benchmark（2026-11-23，randommerge linear_extra）：D2V-L **0.4527**，
优于旧 M1-L 0.7411，但比 CM3P RM-L 0.4216 差 0.031；高段 expected>15
D2V-L 0.6232 甚至比 M1-L 0.5548 差。见 [benchmark_d2v.md](benchmark_d2v.md)。
D2V MLP 没有增益。P4 主口径还有 **transductive 成分**：编码器预训练见过
6,828 图中的 80%（5,462 张）；下面 holdout 是干净口径。

#### 2.5.1 data2vec：P4-holdout / P2-holdout

6,828 张中的 80%（5,462 张）参与预训练，剩下 1,365 张是 holdout；这些图编码器没见过。P2-holdout 是
单划分 5 折，P4-holdout 是 5 seeds。

| 内容表 / 方法 | P2 sign-NLL | P2 Pearson | P2 Spearman | P4 sign-NLL | P4 Pearson | P4 Spearman |
|---|---:|---:|---:|---:|---:|---:|
| D2V 线性 | **0.5971** | **0.4675** | **0.4394** | **0.6047 ± 0.0099** | **0.4591** | **0.4326** |
| D2V + external 线性 | **0.5939** | **0.4838** | **0.4558** | **0.6028 ± 0.0122** | **0.4601** | **0.4398** |
| CM3P canon 线性 | 0.6370 | 0.3551 | 0.3423 | 0.6521 ± 0.0075 | 0.3482 | 0.3354 |
| CM3P canon + external | 0.6167 | 0.4146 | 0.4181 | 0.6285 ± 0.0105 | 0.3919 | 0.3978 |

来源：[results/d2v_main/eval/p2_hold_*.json](../results/d2v_main/eval/)、
[p4_hold_d2v/summary.md](../results/d2v_main/p4_hold_d2v/summary.md)、
[p4_hold_cm3p/summary.md](../results/d2v_main/p4_hold_cm3p/summary.md)。
P2 holdout ceiling 0.56145；P4 holdout ceiling 0.57888 ± 0.01712。
**D2V 在干净口径仍领先 canon+linear，但没有与 legacy MLP+ext 的同口径对照**，所以
不能把这里的“领先 canon”读成“超过 1k 最强基线”。

### 2.6 P4 速览（各数据集 / 条件的最好方法）

| 数据集 / 条件 | 方法 | P4 sign-NLL | P4 Pearson | P4 Spearman | 备注 |
|---|---|---:|---:|---:|---|
| 1k legacy | CM3P + external MLP | **0.5734 ± 0.0043** | **0.5881** | **0.5426** | §2.1 |
| 1k audio-on | CM3P + external MLP | **0.5673 ± 0.0099** | **0.6057** | **0.5757** | 新 6,819 划分 |
| 1k canon | CM3P + external MLP | 0.5943 ± 0.0041 | 0.5201 | 0.4701 | canon 代价 |
| randommerge | CM3P + external 线性 | **0.5827 ± 0.0051** | **0.8824** | **0.8756** | 与 P2 持平 |
| D2V（P4 主口径） | D2V + external 线性 | **0.5856 ± 0.0046** | **0.5411** | **0.4901** | 含 transductive |
| D2V（P4-holdout） | D2V + external 线性 | **0.6028 ± 0.0122** | **0.4601** | **0.4398** | 干净口径 |
| 4,042 published 子集 | published + external MLP | **0.5952 ± 0.0119** | **0.5048** | **0.4850** | 不可与全量比 |
| canon + column | canon + external MLP | 0.5957 ± 0.0029 | 0.5074 | 0.4593 | 无增益 |

来源：分别见 §2.1–§2.5.1、§2.7.2 和 §2.7.3 的对应表；P4-holdout 见 §2.5.1。

### 2.7 历史单划分 / 子集实验

#### 2.7.1 CM3P per-chart 难度（2026-11-03 单划分）

P1 = 同玩家、同 chart 的新 pair（不是泛化口径）；P2 = chart-disjoint；
new-player val = 20% held-out player cluster。captured 只作参考，val 的 ceiling 本身欠拟合。

| 口径 | 方法 | sign-NLL | Pearson | Spearman | captured |
|---|---|---:|---:|---:|---:|
| P1 | CM3P 线性 | 0.5899 | 0.5834 | 0.5467 | 0.7920 |
| P1 | CM3P + external 线性 | 0.5711 | 0.5950 | 0.5566 | 0.9358 |
| P1 | CM3P MLP h64 L2 | 0.5525 | 0.6430 | 0.6207 | 1.0788 |
| P1 | CM3P MLP + external | 0.5317 | 0.6897 | 0.6689 | 1.2380 |
| P2 | CM3P 线性 (chart-disjoint) | 0.6148 | 0.5546 | 0.5192 | 0.6006 |
| P2 | CM3P + external 线性 | 0.5756 | 0.6025 | 0.5623 | 0.9013 |
| P2 | CM3P MLP h64 L2 | 0.6152 | 0.5613 | 0.5291 | 0.5976 |
| new-player val | CM3P 线性 | 0.6021 | — | — | 0.8754 |
| new-player val | CM3P + external 线性 | 0.5870 | — | — | 1.0205 |
| new-player val | CM3P MLP h64 L2 | 0.5680 | — | — | 1.2030 |
| new-player val | CM3P MLP + external | 0.5476 | — | — | 1.3987 |
| new-player val | Sunny | 0.6367 | — | — | 0.5429 |

来源：[results/cm3p_rank_1k/rank_eval_p1_final.json](../results/cm3p_rank_1k/rank_eval_p1_final.json)、
[rank_eval_p2_final.json](../results/cm3p_rank_1k/rank_eval_p2_final.json)、
[player_disjoint_val.json](../results/cm3p_rank_1k/player_disjoint_val.json)、
[docs/cm3p_rank.md](cm3p_rank.md)。P1 的 captured >1 是 ceiling 欠拟合；
new-player val 的 Pearson/Spearman 没有落盘，标“—”。

#### 2.7.2 4,042 图 published 子集：published / legacy / canon 三臂

官方发布 audio-on 512d 向量与 step1 的重叠子集（4,042 图 / 60,039 行），三臂在同一子集、
同一 5 个划分上重跑。**子集 NLL 不可与 §2.1 的全量 6,828 图 NLL 比较**；captured 最高
1.71，也不引用。

| 臂 | 方法 | P2 sign-NLL | P2 Pearson | P2 Spearman | P4 sign-NLL | P4 Pearson | P4 Spearman |
|---|---|---:|---:|---:|---:|---:|---:|
| published | CM3P + external MLP | **0.5886 ± 0.0016** | **0.5466 ± 0.0148** | **0.5322 ± 0.0159** | **0.5952 ± 0.0119** | **0.5048 ± 0.0334** | **0.4850 ± 0.0297** |
| published | CM3P + external 线性 | 0.5903 ± 0.0025 | 0.5416 ± 0.0212 | 0.5241 ± 0.0170 | 0.5968 ± 0.0083 | 0.5154 ± 0.0299 | 0.4850 ± 0.0189 |
| published | CM3P 线性 | 0.6139 ± 0.0025 | 0.5074 ± 0.0125 | 0.4777 ± 0.0102 | 0.6175 ± 0.0071 | 0.4761 ± 0.0173 | 0.4315 ± 0.0143 |
| published | CM3P MLP h64 L2 | 0.6085 ± 0.0029 | 0.5249 ± 0.0065 | 0.5008 ± 0.0110 | 0.6132 ± 0.0112 | 0.4881 ± 0.0277 | 0.4545 ± 0.0247 |
| legacy | CM3P + external MLP | 0.6016 ± 0.0025 | 0.5028 ± 0.0233 | 0.4903 ± 0.0133 | 0.6101 ± 0.0116 | 0.4534 ± 0.0393 | 0.4349 ± 0.0247 |
| legacy | CM3P + external 线性 | 0.6017 ± 0.0020 | 0.4907 ± 0.0256 | 0.4705 ± 0.0177 | 0.6119 ± 0.0125 | 0.4621 ± 0.0411 | 0.4287 ± 0.0248 |
| legacy | CM3P 线性 | 0.6305 ± 0.0015 | 0.4483 ± 0.0161 | 0.4274 ± 0.0103 | 0.6386 ± 0.0085 | 0.4065 ± 0.0223 | 0.3730 ± 0.0199 |
| legacy | CM3P MLP h64 L2 | 0.6326 ± 0.0025 | 0.4421 ± 0.0211 | 0.4397 ± 0.0120 | 0.6422 ± 0.0057 | 0.3977 ± 0.0341 | 0.3904 ± 0.0320 |
| canon | CM3P + external MLP | 0.6202 ± 0.0040 | 0.4090 ± 0.0124 | 0.4004 ± 0.0088 | 0.6281 ± 0.0093 | 0.3415 ± 0.0309 | 0.3365 ± 0.0307 |
| canon | CM3P + external 线性 | 0.6184 ± 0.0032 | 0.4197 ± 0.0226 | 0.4054 ± 0.0148 | 0.6267 ± 0.0090 | 0.3894 ± 0.0452 | 0.3635 ± 0.0345 |
| canon | CM3P 线性 | 0.6384 ± 0.0035 | 0.3754 ± 0.0189 | 0.3571 ± 0.0140 | 0.6434 ± 0.0049 | 0.3471 ± 0.0346 | 0.3153 ± 0.0316 |
| canon | CM3P MLP h64 L2 | 0.6433 ± 0.0030 | 0.3770 ± 0.0226 | 0.3664 ± 0.0158 | 0.6460 ± 0.0061 | 0.3064 ± 0.0340 | 0.2944 ± 0.0347 |

固定信号（三臂相同，完整指标）：

| 信号 | P2 sign-NLL | P2 Pearson | P2 Spearman | P4 sign-NLL | P4 Pearson | P4 Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Sunny | 0.6373 | 0.2874 | 0.3244 | 0.6382 | 0.2619 | 0.2911 |
| Daniel | 0.6369 | 0.2799 | 0.3199 | 0.6371 | 0.2550 | 0.2872 |
| star | 0.6601 | 0.2517 | 0.2973 | 0.6606 | 0.2278 | 0.2646 |

来源：[results/p2p4_pub/](../results/p2p4_pub/)、
[docs/p4_multiseed_pub.md](p4_multiseed_pub.md)。**published 子集内最好**，但全量最强仍是
§2.1 的 legacy external MLP（0.5660/0.5734）；子集只作方向复验。

#### 2.7.3 CM3P 打开 mania column（canon，P4 only）

| 方法 | canon off | canon + column | ΔNLL | seeds 变好 | ΔPearson | ΔSpearman |
|---|---:|---:|---:|---:|---:|---:|
| CM3P + external 线性 | 0.6033 | 0.6058 | +0.0025 | 0/5 | −0.0155 | −0.0205 |
| CM3P + external MLP | 0.5943 | 0.5957 | +0.0014 | 2/5 | −0.0127 | −0.0109 |
| CM3P 线性 | 0.6305 | 0.6324 | +0.0019 | 0/5 | −0.0119 | −0.0130 |
| CM3P MLP h64 L2 | 0.6301 | 0.6325 | +0.0023 | 2/5 | −0.0083 | −0.0042 |

来源：[results/cm3p_column_probe/README.md](../results/cm3p_column_probe/README.md)、
[p4_multiseed_canon_columns/summary_mean_std.csv](../results/p4_multiseed_canon_columns/summary_mean_std.csv)。
结论：published checkpoint 没训练过 [MANIA_COLUMN_*]，直接打开是 OOV 行；**没有增益**。
排除截断图后 4 个方法仍 5/5 变差。

### 2.8 10K / randommerge 全量：ISOR / Koishi（非 P2）

10K 没有 P2/P4 划分；ISOR 只在 10K/randommerge 全量或 RC/LN 子集上测。
因此这张表**不能**和 §2 的 P2/P4 表直接比。

| 图集 | 图数 | 方法 | sign-NLL | Pearson | Spearman | ΔNLL vs Sunny |
|---|---:|---|---:|---:|---:|---:|
| 10K ALL | 16,061 | Sunny | 0.6067 | 0.3429 | 0.2913 | — |
| 10K ALL | 16,061 | ISOR v3.1 | **0.6053** | **0.4127** | 0.2839 | **−0.00146** |
| 10K RC20 | 9,928 | Sunny | 0.6172 | 0.3839 | 0.3023 | — |
| 10K RC20 | 9,928 | ISOR v3.1 | **0.6146** | **0.4447** | 0.2974 | **−0.00261** |
| 10K LN | 6,133 | Sunny | **0.5741** | 0.3077 | 0.2893 | — |
| 10K LN | 6,133 | ISOR v3.1 | 0.5786 | 0.3771 | 0.2889 | **+0.00446** |
| 10K Koishi-native | 1,056 | Sunny | 0.5496 | 0.9063 | 0.9035 | — |
| 10K Koishi-native | 1,056 | Koishi v5.1 | **0.5452** | 0.9032 | **0.9185** | **−0.00446** |
| 10K Koishi-native | 1,056 | ISOR v3.1 | 0.5611 | 0.8746 | 0.8962 | +0.01150 |
| RM ALL | 20,388 | Sunny | **0.6051** | **0.8383** | **0.8364** | — |
| RM ALL | 20,388 | ISOR v3.1 | 0.6204 | 0.8013 | 0.8031 | **+0.01531** |
| RM RC20 | 12,563 | Sunny | **0.6095** | **0.8543** | **0.8467** | — |
| RM RC20 | 12,563 | ISOR v3.1 | 0.6171 | 0.8420 | 0.8374 | **+0.00761** |
| RM LN | 7,824 | Sunny | **0.5930** | **0.7913** | **0.7963** | — |
| RM LN | 7,824 | ISOR v3.1 | 0.6119 | 0.7449 | 0.7606 | **+0.01890** |

来源：[results/ki_eval/summary.csv](../results/ki_eval/summary.csv)、
[docs/koishi_isor_report.md](koishi_isor_report.md)。10K ALL/RC20 上 ISOR 的 NLL 略低，
但 Spearman/逐玩家相关也略低；LN 是 out-of-domain；randommerge 上 ISOR 全面更差。
Koishi 只覆盖 1,056/16,061 图（6.57%）。**benchmark 上的 Dan-MAE 优势没有迁移到
DCR 的 ACC 序**；不能据此证明“对 Dan 标签过拟合”。

## 3. VSRG Dan Benchmark（外部）

### 3.1 全表（n=1,382，out-of-fold isotonic）

benchmark 快照 e15f89b；1,382 张 4K（RC 1,246 / LN 102 / course 34）。
**isotonic 是主校准**；RM-L 与旧 4 组合在 eval_rm 同折重评，P4 是 5-seed 集成。

| 方法 | 训练 / 来源 | n | MAE ↓ | RMSE | Spearman ↑ |
|---|---|---:|---:|---:|---:|
| **RM-L** | randommerge linear_extra | 1,382 | **0.4216** | 0.6440 | 0.8652 |
| M1-L（P4） | CM3P+ext MLP, P4 5-seed | 1,382 | 0.7348 | 1.4214 | 0.8432 |
| M1-L（1k） | CM3P+ext MLP, 1k | 1,382 | 0.7411 | 1.4485 | 0.8437 |
| M1-C（1k） | CM3P+ext MLP, canon | 1,382 | 0.7918 | 1.6234 | 0.8450 |
| M1-C（P4） | CM3P+ext MLP, canon P4 | 1,382 | 0.8126 | 1.6293 | 0.8447 |
| M2-C（P4） | CM3P MLP, canon P4 | 1,382 | 0.9963 | 1.7649 | 0.7103 |
| M2-L（P4） | CM3P MLP, P4 | 1,382 | 1.0632 | 1.8089 | 0.6746 |
| M2-C（1k） | CM3P MLP, canon | 1,382 | 1.1248 | 2.0585 | 0.6823 |
| M2-L（1k） | CM3P MLP, 1k | 1,382 | 1.1489 | 1.9669 | 0.6609 |
| sunny_star isotonic | 固定信号 | 1,382 | **0.3367** | 0.5376 | 0.9060 |
| msd_overall isotonic | 固定信号 | 1,382 | 0.6286 | 1.0077 | 0.7849 |
| [Sunny, MSD] 线性 | 固定信号 | 1,382 | 0.6568 | 0.8642 | 0.9021 |
| 常数 median | — | 1,382 | 1.9195 | 2.8995 | −0.0094 |

来源：[eval_full/summary.csv](../results/benchmark/eval_full/summary.csv)、
[eval_p4/summary.md](../results/benchmark/eval_p4/summary.md)、
[eval_rm/summary.md](../results/benchmark/eval_rm/summary.md)、
[docs/benchmark_linear_extra.md](benchmark_linear_extra.md)。
**所有 CM3P 系组合都没有超过 isotonic Sunny；RM-L 是 DCR 系最好，但仍差 +0.0849
[0.0672, 0.1038]。** published 算法另有覆盖问题，见 §3.3。

### 3.2 高难子集：expected ≥ 12（n=1,142）

子集内重拟合 isotonic（post-hoc），**不能与 §3.1 的全量 MAE 混比**。

| 方法 | GE12 MAE | 95% CI | n | 备注 |
|---|---:|---|---:|---|
| M1-L | **0.3194** | [0.3005, 0.3402] | 1,142 | 1k |
| M1-C | **0.3108** | [0.2931, 0.3298] | 1,142 | canon |
| M2-L | 0.5304 | [0.5007, 0.5622] | 1,142 | 1k |
| M2-C | 0.4841 | [0.4565, 0.5142] | 1,142 | canon |
| sunny_star | **0.2396** | [0.2248, 0.2546] | 1,142 | 固定信号 |
| msd_overall | 0.3990 | [0.3750, 0.4267] | 1,142 | 固定信号 |
| RM-L | 0.3187 | — | 1,142 | **全量校准，未在子集内重拟合** |

来源：[eval_full/subset_ge12_metrics.csv](../results/benchmark/eval_full/subset_ge12_metrics.csv)、
[eval_rm/oof.parquet](../results/benchmark/eval_rm/oof.parquet)。
M1-L 在全量是 0.7411、GE12 是 0.3194，差距主要由 expected<12 的 240 行拉动
（该段 MAE 2.3815）。RM-L 的 GE12 没有重拟合，0.3187 只是全量 isotonic 的直接切分，
**不能和 refit 的 0.3194 当作同口径胜负**。P4 组合的 GE12 没有跑，标缺失。

### 3.3 部分覆盖 / published 算法

published CSV 原样，**没有做 isotonic 重校准**；覆盖行上的 MAE 与 GE12（expected≥12）
都是从逐行 got 直接汇总，不是新模型。**GE12 n 小于 1,142 表示该算法在部分高难图上
没有输出；不能把覆盖行当成统一高难集。** 如果要 published 的 isotonic 版本，需要用
benchmark expected 做 OOF 校准，属于新的校准实验，本轮没做。

| 算法 | n | 覆盖率 | MAE（覆盖行） | GE12 MAE | GE12 n |
|---|---:|---:|---:|---:|---:|
| Sunny | 1,363 | 98.6% | 0.5763 | 0.5011 | 1,142 |
| Daniel | 1,151 | 83.3% | 0.4224 | 0.4089 | 1,095 |
| XDC1 | 1,208 | 87.4% | **0.2096** | 0.1975 | 1,140 |
| **Koishi** | 1,106 | 80.0% | **0.1540** | **0.1508** | 1,072 |
| Azusa | 1,278 | 92.5% | 0.2628 | 0.2129 | 1,093 |
| ISOR | 1,280 | 92.6% | 0.2672 | 0.2202 | 1,095 |
| Roxy | 1,130 | 81.8% | 0.2125 | 0.1982 | 1,077 |
| Companella | 1,382 | 100% | 0.4934 | 0.4473 | 1,142 |
| Mixed | 1,382 | 100% | 0.2879 | 0.2090 | 1,142 |
| P1h | 1,382 | 100% | 0.8314 | 0.8270 | 1,142 |

来源：[eval_full/published_baselines.csv](../results/benchmark/eval_full/published_baselines.csv)、
benchmark 仓库 results/*.csv（.cache/benchmark/...）；GE12 汇总为本次直接聚合。
**Koishi 仍是全场最低（0.1508，覆盖 1,072/1,142 高难行）；RM-L 0.3187 只与
M1-L refit 0.3194 接近，仍远落后 Koishi/Mixed/XDC1/Azusa/ISOR/Roxy。**

### 3.4 RC / LN / course 分层

| 方法 | RC（1,246） | LN（102） | course（34） |
|---|---:|---:|---:|
| RM-L | **0.3655** | 1.0733 | 0.5227 |
| M1-L | 0.5878 | 2.1115 | 2.2455 |
| M1-C | 0.6247 | 2.3824 | 2.1447 |
| M2-L | 1.0102 | 2.3770 | 2.5460 |
| M2-C | 0.9682 | 2.5918 | 2.4629 |
| Sunny | **0.2898** | **0.8630** | **0.4763** |
| MSD | 0.4877 | 2.1378 | 1.2658 |

来源：[eval_full/rc_summary.csv](../results/benchmark/eval_full/rc_summary.csv)、
[eval_rm/breakdown/subset_metrics.csv](../results/benchmark/eval_rm/breakdown/subset_metrics.csv)。
旧 M1-L 的 LN/course 失败（2.11/2.25）被 RM-L 大幅修复（1.07/0.52），但 RM-L 在 LN 上
仍明显落后 Sunny（1.07 vs 0.86）。

### 3.5 重叠 / clean / 高难带的 RM-L 敏感性

randommerge 训练集与 benchmark 有 201 个重叠 bid / 210 行；下表是排除重叠后的 clean 结果。

| 方法 | all | clean | rc | clean_rc | msd_hard |
|---|---:|---:|---:|---:|---:|
| RM-L | 0.4216 | 0.4419 | **0.3655** | **0.3767** | 0.2857 |
| M1-L | 0.7411 | 0.7964 | 0.5878 | 0.6188 | 0.2939 |
| Sunny | **0.3367** | **0.3502** | **0.2898** | **0.2955** | **0.2194** |
| MSD | 0.6286 | 0.6571 | 0.4877 | 0.4916 | 0.3564 |

来源：[eval_rm/breakdown/summary.md](../results/benchmark/eval_rm/breakdown/summary.md)。
RM-L 相对 M1-L 的 paired Δ：all −0.3195 [−0.3802, −0.2658]、clean −0.3545、
rc −0.2223、clean_rc −0.2422；相对 Sunny：all +0.0849（RM-L 更差）。
msd_hard（msd≥32，n=669）上 RM-L 与 M1-L 基本打平（−0.0082，CI 跨 0），
说明合并集的增益集中在低/中难度，不是“最高难段变强”。

### 3.6 isotonic vs linear（对照）

| 方法 | isotonic MAE | linear MAE |
|---|---:|---:|
| RM-L | **0.4216** | 0.8095 |
| M1-L | **0.7411** | 1.2026 |
| M1-C | **0.7918** | 1.1995 |
| M2-L | **1.1489** | 1.4535 |
| M2-C | **1.1248** | 1.4240 |

isotonic 是本 benchmark 的主校准；linear 作对照，差距很大（RM-L 0.42 vs 0.81）。
published 算法没有做 isotonic 重校准，见 §3.3。

## 4. 旧 loss 回归（附录；不是 sign-NLL）

bias / mf_dot 是 loss 回归基线，仓库没有把它们转成 rank 信号；因此**没有
sign-NLL/Pearson(d,b)**，不补跑。最初的稀疏矩阵填充 / MSE 支线（mirt、mirt_1、
mirt_exp、msd 特征、F1/F2、漂移）已于 2026-11-23 退役，只保留 bias/mf_dot 作为参考。

| 模型 | 1k RMSE | 1k Spearman | 10k RMSE | 10k Spearman |
|---|---:|---:|---:|---:|
| bias | 1.5236 | 0.7330 | 1.3727 | 0.6874 |
| mf_dot | 1.4548 | 0.7632 | 1.3327 | 0.7079 |

pairbest 口径 A 的 mf_dot dim=8：1k RMSE 0.7015 / Spearman 0.6749；
10k 0.6844 / 0.6966。CM3P 线性（clean、覆盖子集）RMSE 1.57595。
来源：[docs/models.md](models.md) §2–§3、[legacy/results/initial_mse/step2_1k/](../legacy/results/initial_mse/step2_1k/)。

## 5. 缺失与边界

### 5.1 覆盖矩阵

| 方法族 | 1k P2/P4 | audio | canon | randommerge | 4,042 子集 | 10K | VSRG |
|---|---|---|---|---|---|---|---|
| bias / mf_dot | —（仅 loss） | — | — | — | — | — | — |
| CM3P 线性 / MLP | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| CM3P + external 线性 / MLP | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| data2vec | ✓（P2 单划分） | — | — | —（仅有日志） | — | — | — |
| randommerge linear_extra | — | — | — | ✓ | — | — | ✓（RM-L） |
| 窗口池化 A0–A3 | — | — | — | ✓ | — | — | — |
| 固定信号（Sunny/Daniel/star/MSD） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| ISOR / Koishi | — | — | — | ✓（全量） | — | ✓ | published 原样 |
| published Dan 算法 | — | — | — | — | — | — | ✓（部分覆盖） |

### 5.2 缺口

* **sign-NLL 名称**：仓库没有独立的 sign-NLL 字段；本表按 rank-NLL（成对 BT NLL）汇总。
  如果指的是另一种指标，需要另行定义，现有结果不能直接换算。
* **10K 没有 P2/P4**：10K 只有 ALL/RC20/LN/Koishi-native 等全量或子集结果；不能把
  10K 的 NLL 与 1k/randommerge 的 P2/P4 比。
* **ISOR/Koishi 没有 P2/P4**：只有 10K/randommerge 全量；Koishi 覆盖 6.57% 图。
* **bias/mf_dot 没有 rank 信号**：只有 RMSE/Spearman（§4）；mirt 已退役。
* **D2V P2 MLP 不完整**：只跑完 3/5 折，没有 5 折汇总；D2V randommerge 只有日志，
  没有结果。HPC 批量 14076958/14076959 截至 2026-11-23 20:25 仍 pending。
* **D2V P4-holdout 没有 legacy MLP+ext 对照**：只与 canon 对照，不能写“干净口径超过
  1k 最强基线”。
* **T3 A4–A6 未跑**：A0–A3 的池化结果在 §2.4.2；A4（softmax）/A5（minacalc）/A6（TCN）
  没有结果，不要引用。
* **randommerge captured 不可用**：ceiling 欠拟合，captured 1.10–1.23；本表不引用。
* **p2p4_pub captured >1**：4,042 子集的 ceiling 更噪，captured 最高 1.71；只作方向复验。
* **benchmark GE12**：M1/M2/Sunny/MSD 是子集内 refit；RM-L 只有全量校准切分
  （0.3187），没有 refit；P4 组合的 GE12 没跑。published 的 GE12 是从逐行 CSV 直接汇总，
  未做 isotonic 重校准；覆盖率不同，不能与全表 MAE 混排。
* **benchmark P4 臂**：P4 是 5-seed 集成，不是 5 个种子的 mean±std；报告里不把它当
  另一种“模型方差”。
* **P1 不是泛化口径**：同玩家、同 chart 的新 pair；只看作历史诊断。
* **未重复的已有表**：1k 全量固定信号（12.48M comparisons）与 LN/RC 子集在
  [docs/rank_eval.md](rank_eval.md) / [results/rank_eval_1k/](../results/rank_eval_1k/)；
  它们不是 P2/P4，本文不重复。`results/player_vector_fit/`、`results/minacalc/whitelist_stats/`、
  `results/cm3p_probe_official/` 是技能拟合 / 分布 / 官方向量复现的非排名诊断，也不在本表。
  `legacy/` 的退役实验只在需要历史对照时引用。

## 6. 产物索引

| 内容 | 文件 |
|---|---|
| 1k P2/P4 5 seeds | [results/p4_multiseed/](../results/p4_multiseed/) |
| 1k audio-on | [results/p4_multiseed_audio/](../results/p4_multiseed_audio/) |
| 1k canon | [results/p4_multiseed_canon/](../results/p4_multiseed_canon/) |
| 4,042 published 子集 | [results/p2p4_pub/](../results/p2p4_pub/) |
| randommerge P2/P4 | [results/randommerge_p2p4_hpc/](../results/randommerge_p2p4_hpc/) |
| randommerge 池化 / OD | [results/randommerge_pool/](../results/randommerge_pool/)、[results/randommerge_od_legacy/](../results/randommerge_od_legacy/) |
| data2vec | [data2vec_main/README.md](../data2vec_main/README.md)、[results/d2v_main/](../results/d2v_main/) |
| CM3P per-chart 单划分 | [results/cm3p_rank_1k/](../results/cm3p_rank_1k/) |
| column probe | [results/cm3p_column_probe/](../results/cm3p_column_probe/) |
| 10K ISOR/Koishi | [results/ki_eval/](../results/ki_eval/) |
| VSRG benchmark | [results/benchmark/](../results/benchmark/)、[docs/benchmark_vsrg.md](benchmark_vsrg.md)、[docs/benchmark_linear_extra.md](benchmark_linear_extra.md)、[docs/benchmark_d2v.md](benchmark_d2v.md) |
| 旧 loss 回归（bias/mf_dot） | [legacy/results/initial_mse/step2_1k/](../legacy/results/initial_mse/step2_1k/)、[docs/models.md](models.md) |

---

**一句话结论**：在 P2 主口径上，1k 的最强结果是 CM3P + external MLP（0.5660；
audio-on 0.5585），randommerge 的最强结果是 external 线性（0.5833）；data2vec
（0.5863）只赢 canon+线性，没有超过 1k 的 legacy+external。外部 benchmark 上，
RM-L 把全表 MAE 从 0.7411 降到 0.4216，但仍落后 isotonic Sunny（0.3367）和
Koishi（0.1540，80% 覆盖）；高难子集（expected≥12）上 Sunny 0.2396、M1-L refit
0.3194、RM-L 全量校准切分 0.3187，DCR 系没有超过专用 Dan 估计器。
