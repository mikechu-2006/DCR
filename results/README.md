# results/ —— 实验结果索引

本目录只放**能被文档直接引用的小体积产物**（JSON / CSV / Markdown / PNG），以及
可重建的大产物子目录。原始 dump、训练中间件与 `experiments/` 不入库；每个数字都应
能对应到下表的「生成」命令与「解读」文档。

> 2026-11 整理：`rank_eval_1k_*` 收进 `rank_eval_1k/`；`minacalc_whitelist_stats/`
> 与 `verify_native_vs_modified.*` 收进 `minacalc/`；旧 clean 口径的 `step2_1k_*`
> 收进 `step2_1k/`（原 `results/step2_1k_summary_loss.json` 现为
> `results/step2_1k/summary_loss.json`）。

## 目录

| 路径 | 内容 | 生成 | 解读 |
|---|---|---|---|
| [`rank_eval_1k/`](rank_eval_1k/) | 1k pairbest 上 MinaCalc / osumania_map_analyser 的 rank-NLL（full / msd / LN / RC） | [`scripts/step2_rank_eval.py`](../scripts/step2_rank_eval.py) | [docs/rank_eval.md](../docs/rank_eval.md) |
| [`cm3p_rank_1k/`](cm3p_rank_1k/) | CM3P 内容向量 → per-chart 难度的训练结果、P1/P2/published 评测、新玩家 val、paired ΔNLL | [`scripts/run_cm3p_rank_experiments.py`](../scripts/run_cm3p_rank_experiments.py) | [docs/cm3p_rank.md](../docs/cm3p_rank.md) |
| [`p4_multiseed/`](p4_multiseed/) | P2/P4 5 个**划分种子** × 7 方法的 NLL / Pearson / Spearman、mean/std、配对 Δ；逐 seed 官方评测 JSON | [`scripts/run_p2_p4_multiseed.py`](../scripts/run_p2_p4_multiseed.py) | [docs/p4_multiseed.md](../docs/p4_multiseed.md) |
| [`p4_multiseed_canon/`](p4_multiseed_canon/) | 同一协议的 canon 裁剪内容表复跑（只换 content 表），4 个 embedding 方法 vs legacy 的配对 Δ | [`scripts/run_p2_p4_multiseed.py`](../scripts/run_p2_p4_multiseed.py) `--content-path …`、[`scripts/compare_multiseed_runs.py`](../scripts/compare_multiseed_runs.py) | [docs/p4_multiseed_canon.md](../docs/p4_multiseed_canon.md) |
| [`p2p4_pub/`](p2p4_pub/) | 官方发布 CM3P 向量的重叠子集（4,042 图）上 published / legacy / canon 三臂 × P2/P4 × 5 种子的配对复测 | 同上的 runner（`--split-p2/--split-p4` 指向子集划分） | [docs/p4_multiseed_pub.md](../docs/p4_multiseed_pub.md) |
| [`p4_multiseed_audio/`](p4_multiseed_audio/) | 完整 audio-on 内容表（published 4,037 + local audio 2,782 = 6,819 图）在 P2/P4 上与 audio-off legacy 的同划分配对复测；含数据变更检查 | [`scripts/run_p2_p4_multiseed.py`](../scripts/run_p2_p4_multiseed.py) `--content-path … --split-*`、[`scripts/compare_multiseed_runs.py`](../scripts/compare_multiseed_runs.py)、[`scripts/compare_multiseed_unpaired.py`](../scripts/compare_multiseed_unpaired.py) | [docs/p4_multiseed_audio.md](../docs/p4_multiseed_audio.md) |
| [`benchmark/`](benchmark/) | VSRG Dan Benchmark 外部评测：4 个 CM3P 组合的 1,382 图预测、5 折 chart-group isotonic 校准、OOF 指标与候选 CSV | [`scripts/benchmark_manifest.py`](../scripts/benchmark_manifest.py)、[`scripts/benchmark_signals.py`](../scripts/benchmark_signals.py)、[`scripts/benchmark_encode.py`](../scripts/benchmark_encode.py)、[`scripts/run_benchmark_predict.py`](../scripts/run_benchmark_predict.py)、[`scripts/benchmark_calibrate_eval.py`](../scripts/benchmark_calibrate_eval.py) | [docs/benchmark_vsrg.md](../docs/benchmark_vsrg.md) |
| [`minacalc/whitelist_stats/`](minacalc/whitelist_stats/) | 白名单 21,949 张图的 MinaCalc 全曲/分段难度分布 | [`scripts/minacalc_whitelist_stats.py`](../scripts/minacalc_whitelist_stats.py) | [report.md](minacalc/whitelist_stats/report.md)、[docs/models.md §4](../docs/models.md) |
| [`minacalc/verify_native_vs_modified.*`](minacalc/) | C++ 原生 MinaCalc 505 与本地 Python 逐区间打分的 120 图核对 | 一次性核对（脚本未入库） | [minacalc/README.md](../minacalc/README.md) |
| [`minacalc/inspect_score_example.txt`](minacalc/inspect_score_example.txt) | `inspect_score.py` 的终端示例（7 维难度） | [`scripts/inspect_score.py`](../scripts/inspect_score.py) | [minacalc/README.md](../minacalc/README.md) |
| [`player_vector_fit/`](player_vector_fit/) | 100 名玩家的 MiniCalc 技能拟合 vs CM3P+MLP 对照 | [`scripts/fit_player_vectors.py`](../scripts/fit_player_vectors.py) | [summary_table.md](player_vector_fit/10k_100/summary_table.md) |
| [`mirt_cnonneg_1k/`](mirt_cnonneg_1k/) | mirt 的 D 非负约束分析（旧 clean 口径） | `scripts/step2_predict.py`（历史运行） | [legacy/docs/step2_prediction.md](../legacy/docs/step2_prediction.md) |
| [`step2_1k/`](step2_1k/) | 2026-10 旧 clean 口径的 step2 潜向量、指标与维度扫描笔记 | `scripts/step2_predict.py`（历史运行） | [docs/models.md §2](../docs/models.md) |

## 主结果对应到哪些文件

| 结论 | 文件 | 详见 |
|---|---|---|
| legacy clean 的 bias / mf_dot / mirt RMSE | [`step2_1k/summary_loss.json`](step2_1k/summary_loss.json) | [docs/models.md §2.1](../docs/models.md) |
| pairbest 口径 A 的 mf_dot dim=8 | [`step2_1k/pair_distinct_dimsweep.txt`](step2_1k/pair_distinct_dimsweep.txt) | [docs/models.md §2.2](../docs/models.md) |
| MinaCalc `msd_overall`：captured 0.4067 | [`rank_eval_1k/msd.json`](rank_eval_1k/msd.json)、[`full.json`](rank_eval_1k/full.json) | [docs/rank_eval.md §3](../docs/rank_eval.md) |
| analyser Sunny / Daniel：captured 0.472 | [`rank_eval_1k/full.json`](rank_eval_1k/full.json)、[`summary.csv`](rank_eval_1k/summary.csv) | [docs/rank_eval.md §4](../docs/rank_eval.md) |
| LN / RC 子集 | [`ln_mix.json`](rank_eval_1k/ln_mix.json)、[`ln15.json`](rank_eval_1k/ln15.json)、[`rc.json`](rank_eval_1k/rc.json) | [docs/rank_eval.md §4.3–4.4](../docs/rank_eval.md) |
| CM3P 的 P1 / P2 / published 评测 | [`cm3p_rank_1k/rank_eval_p1_final.json`](cm3p_rank_1k/rank_eval_p1_final.json) 等 | [docs/cm3p_rank.md §5](../docs/cm3p_rank.md) |
| 新玩家（player-disjoint）val | [`cm3p_rank_1k/player_disjoint_val.json`](cm3p_rank_1k/player_disjoint_val.json) | [docs/cm3p_rank.md §5.2](../docs/cm3p_rank.md) |
| 配对 ΔNLL 与 95% CI | [`cm3p_rank_1k/paired_delta.json`](cm3p_rank_1k/paired_delta.json)、[`paired_delta_p2.json`](cm3p_rank_1k/paired_delta_p2.json) | [docs/cm3p_rank.md §5.5](../docs/cm3p_rank.md) |
| P2/P4 5 个划分种子：external MLP 0.5660 / 0.5734 | [`p4_multiseed/summary_mean_std.csv`](p4_multiseed/summary_mean_std.csv)、[`summary.md`](p4_multiseed/summary.md) | [docs/p4_multiseed.md §5](../docs/p4_multiseed.md) |
| 完整 audio-on 表：8/8 组合 5/5 种子变好，ΔNLL −0.0076…−0.0218 | [`p4_multiseed_audio/combined_summary.md`](p4_multiseed_audio/combined_summary.md)、[`audio_vs_legacy.md`](p4_multiseed_audio/audio_vs_legacy.md) | [docs/p4_multiseed_audio.md §4.3](../docs/p4_multiseed_audio.md) |
| VSRG benchmark：M1-L OOF MAE 0.7411，低于 Sunny 0.3367；RC 0.5878 | [`benchmark/eval_full/summary.md`](benchmark/eval_full/summary.md)、[`summary.csv`](benchmark/eval_full/summary.csv)、[`oof.parquet`](benchmark/eval_full/oof.parquet) | [docs/benchmark_vsrg.md §5](../docs/benchmark_vsrg.md) |
| 白名单 MinaCalc 难度分布 | [`minacalc/whitelist_stats/report.md`](minacalc/whitelist_stats/report.md) | [docs/models.md §4](../docs/models.md) |

## 命名与口径

* **`_1k` / `_10k`**：`data/raw` 里对应的 dump tag；数字只在同一 tag 内比较。
* **`_final.json`**：`run_cm3p_rank_experiments.py` 之后用官方 runner 重跑得到的评测；
  同目录下未带 `_final` 的 JSON 是训练时的 quick-eval，**不要引用**。
* **`*.parquet`**：逐谱面信号表（`beatmap_id + <signal>`），是官方评测的输入，可重建。
* **`*_pp.csv`**：逐玩家 Spearman，用于画分布 / 诊断，不参与主表。
* **`captured`** 的分母是 cross-fitted BT ceiling；**不同 map 集 / 不同子集 / 不同
  ceiling 的 captured 不能直接比**（见 [docs/metrics.md §4](../docs/metrics.md)）。
* **RMSE 与 rank-NLL 是两套口径**：clean 的 RMSE、pairbest 的 RMSE、rank-NLL 的 NLL
  三者互不可比，详见 [docs/rank_eval.md §6](../docs/rank_eval.md)。

## 复现

~~~bash
# rank-NLL：MinaCalc / analyser（ceiling 会缓存到 data/processed/rank_eval_cache/）
python scripts/step2_rank_eval.py \
    --in data/processed/step1_1k_pairbest.parquet \
    --signal-table data/processed/rank_signals_1k.parquet \
    --signals msd_overall --split all --min-l 10 --folds 2 --bootstrap 300 \
    --out results/rank_eval_1k/msd.json

# CM3P → per-chart 难度（约 30 min CPU；--quick 只做烟测）
python scripts/run_cm3p_rank_experiments.py --quick

# P2/P4 5 个划分种子 × 7 方法（CPU；两个进程各 4 线程约 1 h）
python scripts/run_p2_p4_multiseed.py --protocols p2,p4 --seeds 1,2,3,4,5 --threads 6

# 白名单 MinaCalc 分布
python scripts/minacalc_whitelist_stats.py --jobs 8

# 玩家向量拟合
python scripts/fit_player_vectors.py --dataset 10k --players 100 --seed 20261023

# experiments/ 索引（本机，不入库）
python scripts/index_experiments.py
~~~

## 版本控制

* **入库**：JSON / CSV / Markdown / PNG / TXT（小体积、可引用）。
* **不入库**：`*.npz` / `*.parquet` / `*.npy`（由上述命令重建）；
  `experiments/` 整个目录也不入库。
* 删除某个结果目录前，先确认没有 docs 或脚本引用它：`grep -rn "results/<dir>" docs scripts`。
