# P2 / P4 multi-seed summary

Seeds are train/test split seeds; model seed is fixed at 0.

## P4（新玩家 + 新 chart，6×6 / 4×4）

ceiling NLL = 0.56540 ± 0.00859 （5 seeds）

| 方法 | test NLL | Pearson(d,b) | Spearman(d,b) |
|---|---|---|---|
| CM3P + external 线性 | 0.58561 ± 0.00459 | 0.5411 ± 0.0149 | 0.4901 ± 0.0192 |
| CM3P 线性 (λ1e-3) | 0.59301 ± 0.00417 | 0.5373 ± 0.0154 | 0.4840 ± 0.0207 |
| Sunny | 0.63111 ± 0.00777 | 0.3098 ± 0.0608 | 0.3480 ± 0.0492 |
| Daniel | 0.63102 ± 0.00601 | 0.3077 ± 0.0614 | 0.3440 ± 0.0497 |
| star | 0.66021 ± 0.00779 | 0.2813 ± 0.0626 | 0.3090 ± 0.0509 |

