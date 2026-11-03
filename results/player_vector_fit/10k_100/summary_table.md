| Method | Params | Train rows | Test rows | RMSE | Spearman |
|---|---:|---:|---:|---:|---:|
| MiniCalc per-player (6D LM) | 600 | 3,969 | 5,015 | 1.6761 | 0.3574 |
| CM3P+MLP (local 21,949, d16/h64/L2) | 40,364 | 3,969 | 5,015 | 1.6285 | 0.1186 |
| Baseline x=20 | 0 | - | 5,015 | 2.0049 | 0.1753 |

Paired cluster bootstrap (2,000 resamples by player):

- Delta RMSE (MLP - MiniCalc) = -0.0464, 95% CI [-0.1558, +0.0667]
- Delta Spearman (MLP - MiniCalc) = -0.2393, 95% CI [-0.3650, -0.1001]
- MLP lower RMSE on 55/100 players; MiniCalc higher Spearman on 79/100 players

Notes:
- Both methods use the exact same 3,969 train / 5,015 test rows and 100 stratified players.
- MiniCalc: 40 train charts max, 6 LM iterations, predictions clipped to score [0.90, 0.998].
- CM3P+MLP: local no-audio 21,949-chart table, 60 epochs, content-eval-heldout to score all 5,015 test rows.
- MiniCalc wall time: 1,755.8 s. MLP training wall time: about 4 s.
