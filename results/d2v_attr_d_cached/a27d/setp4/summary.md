# P2 / P4 multi-seed summary

Seeds are train/test split seeds; model seed is fixed at 0.

## P4（新玩家 + 新 chart，6×6 / 4×4）

ceiling NLL = 0.52448 ± 0.00000 （1 seeds）

| 方法 | test NLL | Pearson(d,b) | Spearman(d,b) |
|---|---|---|---|
| CM3P + external 线性 | 0.57352 ± 0.00000 | 0.7999 ± 0.0000 | 0.6625 ± 0.0000 |
| CM3P 线性 (λ1e-3) | 0.59092 ± 0.00000 | 0.7654 ± 0.0000 | 0.5900 ± 0.0000 |
| Sunny | 0.61090 ± 0.00000 | 0.6652 ± 0.0000 | 0.5950 ± 0.0000 |
| Daniel | 0.62272 ± 0.00000 | 0.6317 ± 0.0000 | 0.5443 ± 0.0000 |
| star | 0.64639 ± 0.00000 | 0.6175 ± 0.0000 | 0.5442 ± 0.0000 |

