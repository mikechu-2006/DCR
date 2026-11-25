# a17 ablation (a20d minus SV/kiai) — Wed Nov 25 07:52:23 PM CST 2026

jobs:
train: 14125021
extract: 14125022
probe: 14125023

sacct:
~~~
14125021|d2v_attr_train|COMPLETED|0:0|00:52:16
14125021.batch|batch|COMPLETED|0:0|00:52:16
14125021.extern|extern|COMPLETED|0:0|00:52:16
14125022|d2v_attr_extract|COMPLETED|0:0|00:07:36
14125022.batch|batch|COMPLETED|0:0|00:07:36
14125022.extern|extern|COMPLETED|0:0|00:07:36
14125023|a17-probe|COMPLETED|0:0|00:08:21
14125023.batch|batch|COMPLETED|0:0|00:08:21
14125023.extern|extern|COMPLETED|0:0|00:08:21
~~~

summaries:
/home/mikechu/Documents/Projects/DCR/results/d2v_attr_d_cached/a17d/setp4/summary.csv
/home/mikechu/Documents/Projects/DCR/results/d2v_attr_d_cached/a20d_reprobe/setp4/summary.csv

## Analysis (2026-11-25 20:00)

set-P4 seed 1 (cached split, 5,204 charts, 1,159,146 comparisons):

| arm | channels | NLL linear | NLL +ext | vs a0d (lin/ext) | vs a20d (lin/ext) |
|---|---:|---:|---:|---:|---:|
| a0d | 8 | 0.60023 | 0.59242 | - | - |
| a17d | 17 | 0.59892 | 0.58704 | -0.00131 / -0.00538 | +0.02334 / +0.02377 |
| a20d (reprobe) | 20 | 0.57558 | 0.56327 | -0.02464 / -0.02915 | - |

Interpretation: SV+kiai carry almost the entire a20d gain; tempo/beat/bar alone are
near-zero. a17d has the LOWEST masked loss (0.0263 < a20d 0.0352 < a0d 0.0588), so
self-supervised loss is not a proxy for difficulty usefulness.
C3 mapper top-1: a0d 0.1340 / a17d 0.1447 / a20d 0.1532; C4 kiai R2: 0.047 / 0.033 /
0.675 -- removing SV+kiai zeroes the kiai signal and the rank gain together.
Single seed, no CI; style-vs-difficulty remains unresolved without mapper-disjoint /
shuffle controls. See docs/d2v_attr_results.md section 3.4.
