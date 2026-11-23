# results/benchmark/

VSRG-DanEstimation-Benchmark 外部评测产物。方法、覆盖率、限制与结论见
[docs/benchmark_vsrg.md](../../docs/benchmark_vsrg.md)（旧 4 组合）和
[docs/benchmark_linear_extra.md](../../docs/benchmark_linear_extra.md)（RM-L）。

## 目录

| 路径 | 内容 |
|---|---|
| `pred/{RM_L,M1_L,M1_C,M2_L,M2_C}.parquet` | 5 个全量模型的 benchmark `d_m`（RM_L = randommerge linear_extra） |
| `pred_p4/*.parquet` | 4 个 P4 5-seed 集成的 `d_m` |
| `eval_full/summary.md` / `summary.csv` | 主评测：5 折 chart-group LOCV、isotonic/linear |
| `eval_full/oof.parquet` | 逐行 out-of-fold `got`、fold、原始 `d_m` |
| `eval_full/rc_summary.csv` | 只看向 RC 的 MAE + bootstrap 95% CI |
| `eval_full/family_summary.csv` | 按 family 的 MAE/RMSE |
| `eval_full/candidates/*.csv` | 4 份按 benchmark 8 列契约生成的候选提交文件（OOF isotonic） |
| `eval_p4/` | P4 5-seed 集成的同口径评测 |
| `eval_rm/` | RM-L + 旧 4 组合的同折 5 折评测、OOF、候选 CSV |
| `eval_rm/breakdown/` | RC / family / 重叠 / 难度带 / 范围 / 配对 / head-to-head |
| `viz/` | dashboard 图（M1-L vs Sunny/M1-C、RM-L vs M1-L、RM-L vs Sunny、RC、summary） |
| `models/` | 5 个全量（含 RM_L）+ 20 个 P4 `.pt` 模型（不入库，可由命令重建） |
| `logs/` | 编码、训练、推理、评测日志 |

## 关键数字（全量模型，out-of-fold isotonic）

| 组合 | MAE | Spearman |
|---|---:|---:|
| **RM-L（randommerge linear_extra）** | **0.4216** | **0.8652** |
| M1-L | 0.7411 | 0.8437 |
| M1-C | 0.7918 | 0.8450 |
| M2-L | 1.1489 | 0.6609 |
| M2-C | 1.1248 | 0.6823 |
| Sunny isotonic | 0.3367 | 0.9060 |
| MSD isotonic | 0.6286 | 0.7849 |

只看向 RC：RM-L **0.3655** [0.3415, 0.3895]；M1-L **0.5878** [0.5427, 0.6453]；Sunny 0.2898、MSD 0.4877。
RM-L 在 LN 1.0733、course 0.5227；M1-L 仍是 LN 2.1115、course 2.2455。
**RM-L 相对 M1-L 的增益集中在低/中难度；expected>15 和 msd Q4 上不显著。**
细节与限制见 [docs/benchmark_linear_extra.md](../../docs/benchmark_linear_extra.md)。

注意：RM-L 的 expected≥12 子集重拟合未在本轮做；下面的 GE12 表仍是旧 4 组合。

**expected ≥ 12 子集（1,142 行，子集内重拟合 isotonic）：**
M1-L **0.3194** [0.3005, 0.3402]、M1-C **0.3108** [0.2931, 0.3298]、
Sunny **0.2396**、MSD **0.3990**。M1-L 胜 MSD 0.0796，负 Sunny 0.0798；
M1-L/M1-C 差 −0.0086（CI 跨 0）。全量 0.7411 主要由 `expected<12` 的 240 行拉动
（该段 MAE 2.3815）。图见 `viz/*_GE12.png`，细节见 [docs/benchmark_vsrg.md §5.7](../../docs/benchmark_vsrg.md#57-expected--12-子集post-hoc子集内重拟合)。

## 复现

~~~bash
# RM-L：HPC CPU 训练（job 14030754），产物见 results/benchmark/models/RM_L_full.pt
# sbatch scripts/slurm/benchmark_rm_linear_train.slurm
python scripts/benchmark_infer.py \
    --model results/benchmark/models/RM_L_full.pt \
    --content data/processed/benchmark_content_legacy.parquet \
    --signals data/processed/benchmark_signals.parquet \
    --out results/benchmark/pred/RM_L.parquet
python scripts/benchmark_calibrate_eval.py \
    --runs RM_L,M1_L,M1_C,M2_L,M2_C \
    --pred-dir results/benchmark/pred --out-dir results/benchmark/eval_rm
python scripts/benchmark_breakdown.py \
    --oof results/benchmark/eval_rm/oof.parquet \
    --model results/benchmark/models/RM_L_full.pt \
    --train-signals results/benchmark/rm_linear/train/RM_L_signals.parquet \
    --out-dir results/benchmark/eval_rm/breakdown

# 旧 4 组合
python scripts/benchmark_manifest.py
python scripts/benchmark_signals.py --force
# 内容编码：HPC CPU，scripts/slurm/benchmark_encode_array16.slurm
python scripts/run_benchmark_predict.py
python scripts/benchmark_calibrate_eval.py \
    --pred-dir results/benchmark/pred --out-dir results/benchmark/eval_full
python scripts/benchmark_calibrate_eval.py \
    --pred-dir results/benchmark/pred_p4 --out-dir results/benchmark/eval_p4
~~~
