# P2 / P4 multi-seed summary

Seeds are train/test split seeds; model seed is fixed at 0.

## P4（新玩家 + 新 chart，6×6 / 4×4）

ceiling NLL = 0.57888 ± 0.01712 （5 seeds）

| 方法 | test NLL | Pearson(d,b) | Spearman(d,b) |
|---|---|---|---|
| CM3P + external 线性 | 0.62848 ± 0.01047 | 0.3919 ± 0.0401 | 0.3978 ± 0.0365 |
| CM3P 线性 (λ1e-3) | 0.65211 ± 0.00754 | 0.3482 ± 0.0575 | 0.3354 ± 0.0457 |
| Sunny | 0.63931 ± 0.01208 | 0.2511 ± 0.0494 | 0.3238 ± 0.0259 |
| Daniel | 0.63855 ± 0.01007 | 0.2431 ± 0.0596 | 0.3191 ± 0.0370 |
| star | 0.66451 ± 0.00947 | 0.2186 ± 0.0572 | 0.2871 ± 0.0309 |

