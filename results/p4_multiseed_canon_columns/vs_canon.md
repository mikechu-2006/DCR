# content ablation: p4_multiseed_canon_columns vs p4_multiseed_canon

Paired over the same train/test split seeds (model seed fixed at 0). Δ < 0 means the variant has the lower (better) NLL.

## P4（新玩家 + 新 chart）

| 方法 | baseline NLL | variant NLL | ΔNLL (variant−baseline) | seeds 变好 | ΔPearson | ΔSpearman |
|---|---|---|---|---|---|---|
| CM3P + external 线性 | 0.60335 ± 0.00515 | 0.60581 ± 0.00478 | +0.00246 ± 0.00146 | 0/5 | -0.0155 | -0.0205 |
| CM3P + external MLP | 0.59425 ± 0.00413 | 0.59567 ± 0.00293 | +0.00142 ± 0.00394 | 2/5 | -0.0127 | -0.0109 |
| CM3P 线性 (λ1e-3) | 0.63050 ± 0.00270 | 0.63244 ± 0.00268 | +0.00194 ± 0.00132 | 0/5 | -0.0119 | -0.0130 |
| CM3P MLP h64 L2 | 0.63013 ± 0.00266 | 0.63246 ± 0.00459 | +0.00232 ± 0.00498 | 2/5 | -0.0083 | -0.0042 |

fixed signals（不依赖 content 表，只作一致性检查）:

| 信号 | baseline | variant | 最大 |Δ| |
|---|---|---|---|
| Sunny | 0.63111 ± 0.00777 | 0.63111 ± 0.00777 | 0.00e+00 |
| Daniel | 0.63102 ± 0.00601 | 0.63102 ± 0.00601 | 0.00e+00 |
| star | 0.66021 ± 0.00779 | 0.66021 ± 0.00779 | 0.00e+00 |

