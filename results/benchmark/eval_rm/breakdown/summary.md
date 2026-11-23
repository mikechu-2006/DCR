# RM-L breakdown / sensitivity

OOF: results/benchmark/eval_rm/oof.parquet

Rows: 1382; seen in randommerge training: 210; clean: 1172

## Subset metrics

| subset | method | n | mae | rmse | bias | spearman |
|---|---|---|---|---|---|---|
| all | RM-L | 1382 | 0.4216 | 0.6440 | -0.0003 | 0.8652 |
| all | M1-L | 1382 | 0.7411 | 1.4485 | 0.0069 | 0.8437 |
| all | Sunny | 1382 | 0.3367 | 0.5376 | 0.0001 | 0.9060 |
| all | MSD | 1382 | 0.6286 | 1.0077 | 0.0012 | 0.7849 |
| clean | RM-L | 1172 | 0.4419 | 0.6703 | -0.0192 | 0.8716 |
| clean | M1-L | 1172 | 0.7964 | 1.5410 | -0.0264 | 0.8488 |
| clean | Sunny | 1172 | 0.3502 | 0.5565 | -0.0066 | 0.9105 |
| clean | MSD | 1172 | 0.6571 | 1.0542 | -0.0112 | 0.7990 |
| rc | RM-L | 1246 | 0.3655 | 0.5457 | 0.0387 | 0.8363 |
| rc | M1-L | 1246 | 0.5878 | 1.1252 | 0.0432 | 0.8186 |
| rc | Sunny | 1246 | 0.2898 | 0.4563 | 0.0205 | 0.8856 |
| rc | MSD | 1246 | 0.4877 | 0.7380 | -0.1294 | 0.7792 |
| clean_rc | RM-L | 1038 | 0.3767 | 0.5591 | 0.0250 | 0.8384 |
| clean_rc | M1-L | 1038 | 0.6188 | 1.1865 | 0.0130 | 0.8199 |
| clean_rc | Sunny | 1038 | 0.2955 | 0.4637 | 0.0171 | 0.8866 |
| clean_rc | MSD | 1038 | 0.4916 | 0.7471 | -0.1684 | 0.7940 |
| ln | RM-L | 102 | 1.0733 | 1.3528 | -0.3556 | 0.9711 |
| ln | M1-L | 102 | 2.1115 | 2.9167 | 0.1114 | 0.8243 |
| ln | Sunny | 102 | 0.8630 | 1.1171 | -0.1172 | 0.9798 |
| ln | MSD | 102 | 2.1378 | 2.5199 | 1.9284 | 0.9535 |
| course | RM-L | 34 | 0.5227 | 0.6757 | -0.3625 | 0.9969 |
| course | M1-L | 34 | 2.2455 | 3.6562 | -1.6376 | 0.7123 |
| course | Sunny | 34 | 0.4763 | 0.6113 | -0.3928 | 0.9963 |
| course | MSD | 34 | 1.2658 | 1.5048 | -0.9934 | 0.9950 |
| msd_hard | RM-L | 669 | 0.2857 | 0.4025 | -0.0021 | 0.7598 |
| msd_hard | M1-L | 669 | 0.2939 | 0.4169 | 0.0167 | 0.7371 |
| msd_hard | Sunny | 669 | 0.2194 | 0.3171 | -0.0171 | 0.8313 |
| msd_hard | MSD | 669 | 0.3564 | 0.5070 | -0.0056 | 0.6353 |

## RC

| subset | method | n | mae | mae_ci_lo | mae_ci_hi | spearman |
|---|---|---|---|---|---|---|
| rc | RM-L | 1246 | 0.3655 | 0.3415 | 0.3895 | 0.8363 |
| rc | M1-L | 1246 | 0.5878 | 0.5427 | 0.6453 | 0.8186 |
| rc | Sunny | 1246 | 0.2898 | 0.2699 | 0.3114 | 0.8856 |
| rc | MSD | 1246 | 0.4877 | 0.4568 | 0.5190 | 0.7792 |
| clean_rc | RM-L | 1038 | 0.3767 | 0.3513 | 0.4046 | 0.8384 |
| clean_rc | M1-L | 1038 | 0.6188 | 0.5612 | 0.6809 | 0.8199 |
| clean_rc | Sunny | 1038 | 0.2955 | 0.2751 | 0.3177 | 0.8866 |
| clean_rc | MSD | 1038 | 0.4916 | 0.4574 | 0.5289 | 0.7940 |

## Family

| subset | method | n | mae | rmse | spearman |
|---|---|---|---|---|---|
| rc | RM-L | 1246 | 0.3655 | 0.5457 | 0.8363 |
| rc | M1-L | 1246 | 0.5878 | 1.1252 | 0.8186 |
| rc | Sunny | 1246 | 0.2898 | 0.4563 | 0.8856 |
| rc | MSD | 1246 | 0.4877 | 0.7380 | 0.7792 |
| ln | RM-L | 102 | 1.0733 | 1.3528 | 0.9711 |
| ln | M1-L | 102 | 2.1115 | 2.9167 | 0.8243 |
| ln | Sunny | 102 | 0.8630 | 1.1171 | 0.9798 |
| ln | MSD | 102 | 2.1378 | 2.5199 | 0.9535 |
| course | RM-L | 34 | 0.5227 | 0.6757 | 0.9969 |
| course | M1-L | 34 | 2.2455 | 3.6562 | 0.7123 |
| course | Sunny | 34 | 0.4763 | 0.6113 | 0.9963 |
| course | MSD | 34 | 1.2658 | 1.5048 | 0.9950 |
| clean | RM-L | 1172 | 0.4419 | 0.6703 | 0.8716 |
| clean | M1-L | 1172 | 0.7964 | 1.5410 | 0.8488 |
| clean | Sunny | 1172 | 0.3502 | 0.5565 | 0.9105 |
| clean | MSD | 1172 | 0.6571 | 1.0542 | 0.7990 |

## Overlap sensitivity

| subset | method | n | mae | mae_ci_lo | mae_ci_hi | delta_vs_M1_L | delta_ci_lo | delta_ci_hi |
|---|---|---|---|---|---|---|---|---|
| all | RM-L | 1382 | 0.4216 | 0.3965 | 0.4481 | -0.3195 | -0.3802 | -0.2658 |
| all | M1-L | 1382 | 0.7411 | 0.6810 | 0.8054 | 0.0000 | nan | nan |
| all | Sunny | 1382 | 0.3367 | 0.3155 | 0.3587 | -0.4044 | nan | nan |
| all | MSD | 1382 | 0.6286 | 0.5882 | 0.6697 | -0.1124 | nan | nan |
| clean | RM-L | 1172 | 0.4419 | 0.4109 | 0.4695 | -0.3545 | -0.4201 | -0.2843 |
| clean | M1-L | 1172 | 0.7964 | 0.7170 | 0.8760 | 0.0000 | nan | nan |
| clean | Sunny | 1172 | 0.3502 | 0.3262 | 0.3759 | -0.4462 | nan | nan |
| clean | MSD | 1172 | 0.6571 | 0.6092 | 0.7064 | -0.1393 | nan | nan |

## Difficulty bands

| band_type | band | method | n | mae | spearman |
|---|---|---|---|---|---|
| expected | high | RM-L | 184 | 0.5241 | 0.7363 |
| expected | high | M1-L | 184 | 0.5548 | 0.6966 |
| expected | high | Sunny | 184 | 0.4271 | 0.7264 |
| expected | high | MSD | 184 | 0.7858 | 0.5615 |
| expected | mid | RM-L | 1006 | 0.3058 | 0.6773 |
| expected | mid | M1-L | 1006 | 0.4485 | 0.6487 |
| expected | mid | Sunny | 1006 | 0.2336 | 0.7752 |
| expected | mid | MSD | 1006 | 0.4228 | 0.5623 |
| expected | low | RM-L | 192 | 0.9299 | 0.9177 |
| expected | low | M1-L | 192 | 2.4528 | 0.2832 |
| expected | low | Sunny | 192 | 0.7902 | 0.9294 |
| expected | low | MSD | 192 | 1.5564 | 0.8179 |
| msd | Q4 | RM-L | 345 | 0.3486 | 0.8449 |
| msd | Q4 | M1-L | 345 | 0.3494 | 0.8243 |
| msd | Q4 | Sunny | 345 | 0.2573 | 0.9096 |
| msd | Q4 | MSD | 345 | 0.4661 | 0.7207 |
| msd | Q3 | RM-L | 346 | 0.2165 | 0.4045 |
| msd | Q3 | M1-L | 346 | 0.2345 | 0.3836 |
| msd | Q3 | Sunny | 346 | 0.1777 | 0.5791 |
| msd | Q3 | MSD | 346 | 0.2414 | 0.0419 |
| msd | Q2 | RM-L | 345 | 0.3066 | 0.5529 |
| msd | Q2 | M1-L | 345 | 0.4160 | 0.5014 |
| msd | Q2 | Sunny | 345 | 0.2413 | 0.7191 |
| msd | Q2 | MSD | 345 | 0.4131 | 0.0501 |
| msd | Q1 | RM-L | 346 | 0.8141 | 0.9498 |
| msd | Q1 | M1-L | 346 | 1.9624 | 0.7540 |
| msd | Q1 | Sunny | 346 | 0.6701 | 0.9644 |
| msd | Q1 | MSD | 346 | 1.3929 | 0.8464 |
| msd_fixed | >=32 | RM-L | 669 | 0.2857 | 0.7598 |
| msd_fixed | >=32 | M1-L | 669 | 0.2939 | 0.7371 |
| msd_fixed | >=32 | Sunny | 669 | 0.2194 | 0.8313 |
| msd_fixed | >=32 | MSD | 669 | 0.3564 | 0.6353 |

## Paired deltas (positive = RM-L worse)

| comparison | subset | n | delta_mae | delta_ci_lo | delta_ci_hi |
|---|---|---|---|---|---|
| RM-L vs M1-L | all | 1382 | -0.3195 | -0.3802 | -0.2658 |
| RM-L vs M1-L | clean | 1172 | -0.3545 | -0.4201 | -0.2843 |
| RM-L vs M1-L | rc | 1246 | -0.2223 | -0.2720 | -0.1798 |
| RM-L vs M1-L | clean_rc | 1038 | -0.2422 | -0.2984 | -0.1891 |
| RM-L vs M1-L | msd_hard | 669 | -0.0082 | -0.0267 | 0.0104 |
| RM-L vs Sunny | all | 1382 | 0.0849 | 0.0672 | 0.1038 |
| RM-L vs Sunny | clean | 1172 | 0.0917 | 0.0709 | 0.1120 |
| RM-L vs Sunny | rc | 1246 | 0.0757 | 0.0576 | 0.0944 |
| RM-L vs Sunny | clean_rc | 1038 | 0.0811 | 0.0619 | 0.1021 |
| RM-L vs Sunny | msd_hard | 669 | 0.0663 | 0.0492 | 0.0840 |
| RM-L vs MSD | all | 1382 | -0.2070 | -0.2449 | -0.1711 |
| RM-L vs MSD | clean | 1172 | -0.2152 | -0.2595 | -0.1720 |
| RM-L vs MSD | rc | 1246 | -0.1222 | -0.1536 | -0.0919 |
| RM-L vs MSD | clean_rc | 1038 | -0.1149 | -0.1491 | -0.0793 |
| RM-L vs MSD | msd_hard | 669 | -0.0707 | -0.0954 | -0.0458 |

## Paired deltas by band

| band_type | band | comparison | n | delta_mae | delta_ci_lo | delta_ci_hi |
|---|---|---|---|---|---|---|
| expected | high | RM-L vs M1-L | 184 | -0.0307 | -0.0745 | 0.0149 |
| expected | high | RM-L vs Sunny | 184 | 0.0970 | 0.0524 | 0.1440 |
| expected | high | RM-L vs MSD | 184 | -0.2617 | -0.3416 | -0.1846 |
| expected | mid | RM-L vs M1-L | 1006 | -0.1426 | -0.1790 | -0.1083 |
| expected | mid | RM-L vs Sunny | 1006 | 0.0722 | 0.0561 | 0.0887 |
| expected | mid | RM-L vs MSD | 1006 | -0.1170 | -0.1516 | -0.0869 |
| expected | low | RM-L vs M1-L | 192 | -1.5229 | -1.8296 | -1.2267 |
| expected | low | RM-L vs Sunny | 192 | 0.1397 | 0.0487 | 0.2228 |
| expected | low | RM-L vs MSD | 192 | -0.6265 | -0.8058 | -0.4274 |
| msd | Q4 | RM-L vs M1-L | 345 | -0.0007 | -0.0338 | 0.0299 |
| msd | Q4 | RM-L vs Sunny | 345 | 0.0914 | 0.0609 | 0.1206 |
| msd | Q4 | RM-L vs MSD | 345 | -0.1175 | -0.1591 | -0.0776 |
| msd | Q3 | RM-L vs M1-L | 346 | -0.0179 | -0.0389 | 0.0044 |
| msd | Q3 | RM-L vs Sunny | 346 | 0.0388 | 0.0221 | 0.0568 |
| msd | Q3 | RM-L vs MSD | 346 | -0.0248 | -0.0509 | 0.0012 |
| msd | Q2 | RM-L vs M1-L | 345 | -0.1094 | -0.1621 | -0.0613 |
| msd | Q2 | RM-L vs Sunny | 345 | 0.0654 | 0.0427 | 0.0903 |
| msd | Q2 | RM-L vs MSD | 345 | -0.1065 | -0.1474 | -0.0693 |
| msd | Q1 | RM-L vs M1-L | 346 | -1.1483 | -1.3431 | -0.9455 |
| msd | Q1 | RM-L vs Sunny | 346 | 0.1440 | 0.0818 | 0.2028 |
| msd | Q1 | RM-L vs MSD | 346 | -0.5788 | -0.7113 | -0.4470 |
| msd_fixed | >=32 | RM-L vs M1-L | 669 | -0.0082 | -0.0267 | 0.0104 |
| msd_fixed | >=32 | RM-L vs Sunny | 669 | 0.0663 | 0.0492 | 0.0840 |
| msd_fixed | >=32 | RM-L vs MSD | 669 | -0.0707 | -0.0954 | -0.0458 |

## Head to head

| comparison | subset | n | a_wins | b_wins | ties | a_mae | b_mae |
|---|---|---|---|---|---|---|---|
| RM-L vs M1-L | all | 1382 | 828 | 553 | 1 | 0.4216 | 0.7411 |
| RM-L vs M1-L | clean | 1172 | 708 | 463 | 1 | 0.4419 | 0.7964 |
| RM-L vs M1-L | rc | 1246 | 720 | 525 | 1 | 0.3655 | 0.5878 |
| RM-L vs M1-L | clean_rc | 1038 | 602 | 435 | 1 | 0.3767 | 0.6188 |
| RM-L vs M1-L | msd_hard | 669 | 365 | 303 | 1 | 0.2857 | 0.2939 |
| RM-L vs Sunny | all | 1382 | 517 | 865 | 0 | 0.4216 | 0.3367 |
| RM-L vs Sunny | clean | 1172 | 432 | 740 | 0 | 0.4419 | 0.3502 |
| RM-L vs Sunny | rc | 1246 | 465 | 781 | 0 | 0.3655 | 0.2898 |
| RM-L vs Sunny | clean_rc | 1038 | 381 | 657 | 0 | 0.3767 | 0.2955 |
| RM-L vs Sunny | msd_hard | 669 | 261 | 408 | 0 | 0.2857 | 0.2194 |
| RM-L vs MSD | all | 1382 | 868 | 514 | 0 | 0.4216 | 0.6286 |
| RM-L vs MSD | clean | 1172 | 735 | 437 | 0 | 0.4419 | 0.6571 |
| RM-L vs MSD | rc | 1246 | 760 | 486 | 0 | 0.3655 | 0.4877 |
| RM-L vs MSD | clean_rc | 1038 | 628 | 410 | 0 | 0.3767 | 0.4916 |
| RM-L vs MSD | msd_hard | 669 | 404 | 265 | 0 | 0.2857 | 0.3564 |

## d_m range diagnostic

| series | stat | value |
|---|---|---|
| train | p1 | -2.811249 |
| train | p5 | -2.175022 |
| train | p50 | -0.213913 |
| train | p95 | 2.980289 |
| train | p99 | 4.308507 |
| train | min | -3.622861 |
| train | max | 6.451692 |
| benchmark | p1 | -0.449910 |
| benchmark | p5 | 0.928431 |
| benchmark | p50 | 4.044757 |
| benchmark | p95 | 5.655812 |
| benchmark | p99 | 6.862269 |
| benchmark | min | -1.821149 |
| benchmark | max | 8.264525 |
| benchmark | share_below_min | 0.000000 |
| benchmark | share_above_max | 0.016643 |
| benchmark | mae_above_max | 0.591421 |
| benchmark | n_above_max | 23.000000 |
| benchmark | share_outside_minmax | 0.016643 |
| benchmark | mae_outside_minmax | 0.591421 |
| benchmark | n_outside_minmax | 23.000000 |
| benchmark | share_below_p1 | 0.000000 |
| benchmark | share_above_p99 | 0.356006 |
| benchmark | mae_above_p99 | 0.353350 |
| benchmark | n_above_p99 | 492.000000 |
| benchmark | share_outside_p1p99 | 0.356006 |
| benchmark | mae_outside_p1p99 | 0.353350 |
| benchmark | n_outside_p1p99 | 492.000000 |

