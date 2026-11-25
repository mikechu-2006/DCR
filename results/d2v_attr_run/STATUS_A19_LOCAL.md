# a19 ablation — local CPU probe fallback (Thu Nov 26 12:01:48 AM CST 2026)

Content tables: a17d/a20d produced earlier on HPC; a19d produced on HPC job 14128340/14128341 (train/extract COMPLETED).
Probe (linear/linear_extra, set-P4 seed 1) run locally because the HPC CPU probe job 14128342 was stuck at Priority and was cancelled.

## summaries
/home/mikechu/Documents/Projects/DCR/results/d2v_attr_d_cached/a17d_rerun/setp4/summary.csv
/home/mikechu/Documents/Projects/DCR/results/d2v_attr_d_cached/a19d/setp4/summary.csv
/home/mikechu/Documents/Projects/DCR/results/d2v_attr_d_cached/a20d_rerun/setp4/summary.csv

## Analysis (2026-11-26 00:10)

Three-point decomposition (set-P4 seed 1, local CPU probe, same runner):

| arm | channels | NLL linear | NLL +ext | step (lin/ext) | vs a0d (lin/ext) |
|---|---:|---:|---:|---:|---:|
| a0d | 8 | 0.60023 | 0.59242 | - | - |
| a17d | 17 | 0.59891 | 0.58704 | - | -0.00132 / -0.00538 |
| a19d | 19 | 0.58350 | 0.57018 | add SV: -0.01541 / -0.01686 | -0.01673 / -0.02223 |
| a20d | 20 | 0.57559 | 0.56327 | add kiai: -0.00791 / -0.00691 | -0.02464 / -0.02915 |

SV is the larger contributor (~2/3); kiai ~1/3. C3 mapper top-1: a0d .1340,
a17d .1447, a19d .1191, a20d .1532. C4: a19d SV-change R2 .743 / kiai R2 .060;
a20d .730 / .675. So SV looks more like reading/visual difficulty, kiai more
style/song-section-like, but single seed and C3 n=470 test charts are caveats.
HPC CPU probe job 14128342 was cancelled after being stuck at Priority; the probe
was re-run locally (a17d/a20d recompute differ from HPC by <=1.5e-5).
