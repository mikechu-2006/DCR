# P2 / P4 multi-seed summary

Seeds are train/test split seeds; model seed is fixed at 0.

## P4（新玩家 + 新 chart，6×6 / 4×4）

ceiling NLL = 0.56540 ± 0.00859 （5 seeds）

| 方法 | test NLL | Pearson(d,b) | Spearman(d,b) |
|---|---|---|---|
| CM3P + external MLP | 0.57373 ± 0.00382 | 0.5888 ± 0.0264 | 0.5437 ± 0.0246 |
| CM3P MLP h64 L2 | 0.62540 ± 0.00653 | 0.5116 ± 0.0225 | 0.4763 ± 0.0325 |
| Sunny | 0.63111 ± 0.00777 | 0.3098 ± 0.0608 | 0.3480 ± 0.0492 |
| Daniel | 0.63102 ± 0.00601 | 0.3077 ± 0.0614 | 0.3440 ± 0.0497 |
| star | 0.66021 ± 0.00779 | 0.2813 ± 0.0626 | 0.3090 ± 0.0509 |

