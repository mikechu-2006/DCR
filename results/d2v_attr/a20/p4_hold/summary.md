# P2 / P4 multi-seed summary

Seeds are train/test split seeds; model seed is fixed at 0.

## P4（新玩家 + 新 chart，6×6 / 4×4）

ceiling NLL = 0.57794 ± 0.00000 （1 seeds）

| 方法 | test NLL | Pearson(d,b) | Spearman(d,b) |
|---|---|---|---|
| CM3P + external 线性 | 0.59306 ± 0.00000 | 0.4647 ± 0.0000 | 0.4208 ± 0.0000 |
| CM3P 线性 (λ1e-3) | 0.59395 ± 0.00000 | 0.4706 ± 0.0000 | 0.4260 ± 0.0000 |
| Sunny | 0.64162 ± 0.00000 | 0.3070 ± 0.0000 | 0.3498 ± 0.0000 |
| Daniel | 0.63593 ± 0.00000 | 0.3123 ± 0.0000 | 0.3608 ± 0.0000 |
| star | 0.65852 ± 0.00000 | 0.2784 ± 0.0000 | 0.3163 ± 0.0000 |

