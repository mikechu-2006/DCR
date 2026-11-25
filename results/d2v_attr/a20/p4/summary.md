# P2 / P4 multi-seed summary

Seeds are train/test split seeds; model seed is fixed at 0.

## P4（新玩家 + 新 chart，6×6 / 4×4）

ceiling NLL = 0.55443 ± 0.00000 （1 seeds）

| 方法 | test NLL | Pearson(d,b) | Spearman(d,b) |
|---|---|---|---|
| CM3P + external 线性 | 0.57957 ± 0.00000 | 0.5711 ± 0.0000 | 0.4984 ± 0.0000 |
| CM3P 线性 (λ1e-3) | 0.58839 ± 0.00000 | 0.5709 ± 0.0000 | 0.5054 ± 0.0000 |
| Sunny | 0.63168 ± 0.00000 | 0.3297 ± 0.0000 | 0.3350 ± 0.0000 |
| Daniel | 0.63034 ± 0.00000 | 0.3255 ± 0.0000 | 0.3347 ± 0.0000 |
| star | 0.66221 ± 0.00000 | 0.2973 ± 0.0000 | 0.2956 ± 0.0000 |

