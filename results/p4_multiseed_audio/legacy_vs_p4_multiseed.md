# unpaired comparison: legacy vs p4_multiseed

This script never pairs seeds and does not require equal ceilings.  It is intended for runs on different split sets; if the ceilings match, the two runs share splits and the mean difference equals the paired mean difference, but the table is still descriptive.  If the ceilings differ, only the difference of the 5-seed means is meaningful.

## P2（新 chart）

- ceiling NLL: p4_multiseed **0.56126 ± 0.00348**; legacy **0.56104 ± 0.00394** （ceilings differ; different split files）

| 方法 | p4_multiseed NLL | legacy NLL | Δmean | ΔPearson | ΔSpearman |
|---|---|---|---|---|---|
| CM3P + external 线性 | 0.57786 ± 0.00335 | 0.57859 ± 0.00185 | +0.00073 | -0.0057 | -0.0063 |
| CM3P + external MLP | 0.56602 ± 0.00307 | 0.56677 ± 0.00117 | +0.00075 | -0.0025 | -0.0018 |
| CM3P 线性 (λ1e-3) | 0.61618 ± 0.00293 | 0.61796 ± 0.00286 | +0.00178 | -0.0014 | +0.0007 |
| CM3P MLP h64 L2 | 0.61664 ± 0.00265 | 0.61741 ± 0.00389 | +0.00077 | -0.0013 | +0.0027 |
| Sunny | 0.62868 ± 0.00207 | 0.62651 ± 0.00231 | -0.00217 | -0.0003 | -0.0024 |
| Daniel | 0.62768 ± 0.00212 | 0.62647 ± 0.00239 | -0.00122 | +0.0005 | -0.0023 |
| star | 0.66153 ± 0.00137 | 0.65996 ± 0.00184 | -0.00157 | +0.0015 | +0.0001 |

## P4（新玩家 + 新 chart）

- ceiling NLL: p4_multiseed **0.56540 ± 0.00859**; legacy **0.56618 ± 0.01099** （ceilings differ; different split files）

| 方法 | p4_multiseed NLL | legacy NLL | Δmean | ΔPearson | ΔSpearman |
|---|---|---|---|---|---|
| CM3P + external 线性 | 0.58479 ± 0.00494 | 0.58478 ± 0.00771 | -0.00001 | +0.0010 | +0.0041 |
| CM3P + external MLP | 0.57343 ± 0.00433 | 0.57488 ± 0.00867 | +0.00145 | -0.0098 | -0.0060 |
| CM3P 线性 (λ1e-3) | 0.62411 ± 0.00345 | 0.61962 ± 0.00959 | -0.00449 | -0.0065 | -0.0042 |
| CM3P MLP h64 L2 | 0.62540 ± 0.00653 | 0.61988 ± 0.00713 | -0.00552 | -0.0040 | +0.0035 |
| Sunny | 0.63111 ± 0.00777 | 0.62934 ± 0.00805 | -0.00177 | +0.0175 | +0.0064 |
| Daniel | 0.63102 ± 0.00601 | 0.62967 ± 0.00583 | -0.00135 | +0.0183 | +0.0114 |
| star | 0.66021 ± 0.00779 | 0.65949 ± 0.00604 | -0.00072 | +0.0102 | +0.0039 |

