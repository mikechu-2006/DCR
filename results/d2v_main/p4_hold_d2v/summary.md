# P2 / P4 multi-seed summary

Seeds are train/test split seeds; model seed is fixed at 0.

## P4（新玩家 + 新 chart，6×6 / 4×4）

ceiling NLL = 0.57888 ± 0.01712 （5 seeds）

| 方法 | test NLL | Pearson(d,b) | Spearman(d,b) |
|---|---|---|---|
| CM3P + external 线性 | 0.60280 ± 0.01215 | 0.4601 ± 0.0590 | 0.4398 ± 0.0441 |
| CM3P 线性 (λ1e-3) | 0.60471 ± 0.00994 | 0.4591 ± 0.0606 | 0.4326 ± 0.0416 |
| Sunny | 0.63931 ± 0.01208 | 0.2511 ± 0.0494 | 0.3238 ± 0.0259 |
| Daniel | 0.63855 ± 0.01007 | 0.2431 ± 0.0596 | 0.3191 ± 0.0370 |
| star | 0.66451 ± 0.00947 | 0.2186 ± 0.0572 | 0.2871 ± 0.0309 |

