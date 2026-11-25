# P2 / P4 multi-seed summary

Seeds are train/test split seeds; model seed is fixed at 0.

## P4（新玩家 + 新 chart，6×6 / 4×4）

ceiling NLL = 0.52449 ± 0.00000 （1 seeds）

| 方法 | test NLL | Pearson(d,b) | Spearman(d,b) |
|---|---|---|---|
| CM3P + external 线性 | 0.56327 ± 0.00000 | 0.8023 ± 0.0000 | 0.6505 ± 0.0000 |
| CM3P 线性 (λ1e-3) | 0.57559 ± 0.00000 | 0.7780 ± 0.0000 | 0.6104 ± 0.0000 |
| Sunny | 0.61090 ± 0.00000 | 0.6651 ± 0.0000 | 0.5949 ± 0.0000 |
| Daniel | 0.62272 ± 0.00000 | 0.6317 ± 0.0000 | 0.5442 ± 0.0000 |
| star | 0.64639 ± 0.00000 | 0.6174 ± 0.0000 | 0.5440 ± 0.0000 |

