# content ablation: audio vs legacy

Paired over the same train/test split seeds (model seed fixed at 0). Δ < 0 means the variant has the lower (better) NLL.

## P4（新玩家 + 新 chart）

| 方法 | baseline NLL | variant NLL | ΔNLL (variant−baseline) | seeds 变好 | ΔPearson | ΔSpearman |
|---|---|---|---|---|---|---|
| CM3P + external 线性 | 0.59376 ± 0.00564 | 0.58285 ± 0.00787 | -0.01091 ± 0.00366 | 5/5 | +0.0583 | +0.0566 |
| CM3P + external MLP | 0.58592 ± 0.00649 | 0.58012 ± 0.00730 | -0.00581 ± 0.00425 | 5/5 | +0.0337 | +0.0437 |
| CM3P 线性 (λ1e-3) | 0.62912 ± 0.00468 | 0.62110 ± 0.00732 | -0.00801 ± 0.00309 | 5/5 | +0.0480 | +0.0329 |
| CM3P MLP h64 L2 | 0.63476 ± 0.00662 | 0.62112 ± 0.00582 | -0.01364 ± 0.00261 | 5/5 | +0.0603 | +0.0582 |

fixed signals（不依赖 content 表，只作一致性检查）:

| 信号 | baseline | variant | 最大 |Δ| |
|---|---|---|---|
| Sunny | 0.63284 ± 0.00768 | 0.63284 ± 0.00768 | 0.00e+00 |
| Daniel | 0.63094 ± 0.00691 | 0.63094 ± 0.00691 | 0.00e+00 |
| star | 0.66922 ± 0.00668 | 0.66922 ± 0.00668 | 0.00e+00 |

