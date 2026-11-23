# P2 / P4 multi-seed summary

Seeds are train/test split seeds; model seed is fixed at 0.

## P4（新玩家 + 新 chart，6×6 / 4×4）

ceiling NLL = 0.56540 ± 0.00859 （5 seeds）

| 方法 | test NLL | Pearson(d,b) | Spearman(d,b) |
|---|---|---|---|
| CM3P + external 线性 | 0.60335 ± 0.00515 | 0.4892 ± 0.0247 | 0.4464 ± 0.0215 |
| CM3P 线性 (λ1e-3) | 0.63050 ± 0.00270 | 0.4520 ± 0.0219 | 0.4084 ± 0.0174 |
| Sunny | 0.63111 ± 0.00777 | 0.3098 ± 0.0608 | 0.3480 ± 0.0492 |
| Daniel | 0.63102 ± 0.00601 | 0.3077 ± 0.0614 | 0.3440 ± 0.0497 |
| star | 0.66021 ± 0.00779 | 0.2813 ± 0.0626 | 0.3090 ± 0.0509 |

