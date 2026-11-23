# legacy/ —— 退役代码与归档文档

这里只放主线不再需要、但需要留档的内容。主线代码、配置和文档都在仓库根；
每次运行的实验目录在 [`experiments/`](../experiments/)，**不**在本目录。

整理原则：主线只保留已接入 `main.py` 并被验证的步骤（step0/step1 预处理、
bias / mf_dot 等 oracle、CM3P 内容向量）以及 [`minacalc/`](../minacalc/) 包本身；
其余代码、文档、历史日志与旧结果都移到这里。

## 目录

| 路径 | 文件数 | 内容 |
|---|---|---|
| [`scripts/`](scripts/) | 124 | 退役脚本：MinaCalc MSD/fit-skills、旧 v2 管线、学习曲线/遗忘/玩家分析、PCA、step2 容量诊断、map_analyser、fetch_tags、`_tmp_*` |
| [`docs/`](docs/) | 73 | 归档文档：step0/step1 长文与清洗分析、CM3P 设计/结果/MLP/远程与 HPC runbook、MinaCalc、map_analyser、[archive/](docs/archive/)、[figs/](docs/figs/)、[plans/](docs/plans/)、[reviews/](docs/reviews/)、[ops/](docs/ops/)、[research/](docs/research/) |
| [`configs/`](configs/) | 4 | 旧维度扫描、map_analyser、MinaCalc L1/L2 配置 |
| [`logs/`](logs/) | 32 | 2026-09 ~ 2026-10 的历史运行日志（新日志在各实验目录的 `logs/`） |
| [`results/`](results/) | 5 个目录 | 已退役分析（PCA）与初始 MSE 支线（[initial_mse/](results/initial_mse/)：step2_1k、mirt_cnonneg_1k） |
| [`scratch_rank_metric/`](scratch_rank_metric/) | 30 | 2026-10-31 开发 rank-NLL 时的临时探针、小样本烟测与中间结果（含 [说明](scratch_rank_metric/README.md)） |
| `local/`（仅本机） | 2 | `DCR.bundle`、`upload_to_github.sh`（被 .gitignore 忽略，不随仓库分发） |
| [`MANIFEST.tsv`](MANIFEST.tsv) | — | 五轮整理的全部 `old → new` 路径（303 条，含删除的缓存） |

## 第一轮：退役分析 + 实验目录（2026-10-20）

* **step 注册表精简**：`step0/fetch_tags`、`step0/map_analyser`、
  `step2/map_analyser` 从 `dcr/steps.py` 移除，脚本与文档进本目录。
* **退役分析脚本**：`diag_*`、`compare_*`、`pca_analysis`、`player_plays`、
  `fetch_player_scores`、`play_order_curve`、`learning_rate_variance`、
  `forgetting_curve`、`practice_vs_time`、`plot_chart_*`、`probe_*`、
  `_tmp_*` 全部归档。
* **旧 v2 管线**：原 `scripts/legacy/` 整体迁到 `legacy/scripts/`。
  其中 `parse_dump.py` 仍被 `scripts/step0_preprocess.py` 使用，已提升为
  [`scripts/parse_dump.py`](../scripts/parse_dump.py)，不在本目录。
* **实验目录统一**：根目录的 `experiments_<时间戳>_<name>/` 全部移到
  [`experiments/`](../experiments/)，去掉冗余前缀（`experiments/20261020_193449_pairbest_1k`）。
  历史实验里的 `manifest.json` / `run.sh` 仍保留迁移前的绝对路径（作为运行记录原样保存）。

## 第二轮：MinaCalc 与文档精简（2026-10-20）

* **MinaCalc 只保留包**：`step0h_minacalc_msd.py`、`step2_minacalc_as_feature.py`、
  `step2_minacalc_L2.py`、fit-skills 系列（`step2_fit_skills.py` / `step2_bench.py` /
  `step2_player_step.py` / `step2_export_charts.py` / `step2_fit.slurm`）以及
  L1/L2 配置和 3 份 MinaCalc 文档归档。主线只留 [`minacalc/`](../minacalc/README.md)。
  注册表同步删除了 `step0/minacalc_msd`、`step2/minacalc_as_feature`、
  `step2/minacalc_L2`。
* **文档压缩**：14 份 `docs/*.md` 合并为 3 份主线文档
  （[`procedure.md`](../docs/procedure.md)、[`data.md`](../docs/data.md)、
  [`models.md`](../docs/models.md)）；step0/step1 长文、CM3P 设计与实测、
  MinaCalc、HPC runbook 原文全部归档到这里。
* **step1/clean 仍留在主线**：MinaCalc/CM3P 的旧配置曾显式使用它，且
  `step1_pairbest.py` 复用了它的工具函数。

## 第三轮：rank-NLL 开发产物（2026-11）

* **临时目录归档**：仓库根的 `scratch_rank_metric/` 整体移到这里
  （[`scratch_rank_metric/`](scratch_rank_metric/)）。里面的 `probe*.py` 是开发
  `dcr/rank_metrics.py` 时的探索脚本，结论已经写进
  [docs/metrics.md](../docs/metrics.md) 与 [docs/rank_eval.md](../docs/rank_eval.md)；
  正式实现和自检在 `dcr/rank_metrics.py`、`scripts/step2_rank_eval.py`、
  `scripts/test_rank_metrics.py`。
* **results/ 分组**：主线结果按主题收进子目录（`rank_eval_1k/`、`minacalc/`、
  `step2_1k/`），索引见 [results/README.md](../results/README.md)。这一步只搬运行产物，
  不改任何数字；legacy 文档里 `results/step2_1k_summary_loss.json` 等旧路径按迁移前
  位置书写。第五轮起 `step2_1k/` 本身也退役，见 [initial_mse/](results/initial_mse/)。

## 第四轮：docs 精简（2026-11-23）

为了降低后续 agent 的默认阅读量，把**已执行完的计划、方法评审、一次性运维快照与产品调研**
移出 `docs/`。主线 `docs/` 只保留当前口径、结果和 [总表](../docs/experiments.md)。

| 新位置 | 文件 | 说明 |
|---|---|---|
| [plans/](docs/plans/) | `benchmark_plan.md`、`benchmark_linear_extra_plan.md`、`method_overfit_plan.md` | benchmark 执行计划，结果已在 docs/benchmark_*.md |
| [plans/](docs/plans/) | `random_merge_plan.md`、`randommerge_mlp_pooling_plan.md`、`randommerge_a2b_log_od_plan.md` | randommerge 计划；结果在 docs/randommerge_*_results.md |
| [plans/](docs/plans/) | `p4_multiseed_audio_plan.md`、`data2vec_pretrain_plan.md`、`cm3p_crop_windows_v4_plan.md` | audio / data2vec / canon 的旧计划；当前口径在对应结果文档 |
| [reviews/](docs/reviews/) | `mlm_continued_pretraining_review.md` | MLM 继续预训练的方法论反思，非当前结果 |
| [ops/](docs/ops/) | `hpc_queue_probe.md` | 2026-11-05 的一次性队列快照 |
| [research/](docs/research/) | `plugin_packaging.md` | 游戏内封装调研，未实现 |

移入后 `docs/` 从 33 个 .md 降到 21 个；被移动文档里的相对链接按迁移前位置书写，
需要时按 [MANIFEST.tsv](MANIFEST.tsv) 放回。

## 第五轮：初始 MSE 支线退役（2026-11-23）

* **范围**：最初的稀疏矩阵填充 / MSE 支线（`mirt` / `mirt_1` / `mirt_exp`、
  D 约束、msd 特征、F1/F2、漂移）不再维护。`step2/predict` 只保留
  **bias / mf_dot** 作为参考基线；`--models mirt...` 会被 CLI 拒绝并指向本目录。
* **结果归档**：旧 clean 口径的 `results/step2_1k/` 与 `results/mirt_cnonneg_1k/`
  移到 [results/initial_mse/](results/initial_mse/)。
* **代码保留**：`step2_predict.py` 里的 mirt 分支没有删（仍可被历史脚本 import），
  但 CLI 禁用；历史设计与消融在
  [legacy/docs/step2_prediction.md](docs/step2_prediction.md)、
  [legacy/docs/step2_fit_skills.md](docs/step2_fit_skills.md) 等旧文档。

## 使用说明

* 归档脚本与文档里的**相对路径按迁移前的位置书写**，直接运行/点击可能失效；
  需要时请按 `MANIFEST.tsv` 把文件放回去，或对照主线对应物修改路径。
* 恢复一个退役脚本不会自动恢复它的 step 注册；要接回 `main.py` 需在
  `dcr/steps.py` 里重新 `register(StepDef(...))`。
* 主线仍然可用的同类能力：
  * 谱面内容特征：`step2/cm3p_mlp`（`step2/predict` 也接受 CM3P 内容表）；
  * 旧 play 级清洗：`step1/clean`；
  * 基线：`step2/predict` 的 `bias` / `mf_dot`（mirt 已退役）。
