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
| [`docs/`](docs/) | 61 | 归档文档：step0/step1 长文与清洗分析、CM3P 设计/结果/MLP/远程与 HPC runbook、MinaCalc、map_analyser、[archive/](docs/archive/)、[figs/](docs/figs/) |
| [`configs/`](configs/) | 4 | 旧维度扫描、map_analyser、MinaCalc L1/L2 配置 |
| [`logs/`](logs/) | 32 | 2026-09 ~ 2026-10 的历史运行日志（新日志在各实验目录的 `logs/`） |
| [`results/`](results/) | 6 | 已退役分析（PCA）的旧结果目录 |
| `local/`（仅本机） | 2 | `DCR.bundle`、`upload_to_github.sh`（被 .gitignore 忽略，不随仓库分发） |
| [`MANIFEST.tsv`](MANIFEST.tsv) | — | 两轮整理的全部 `old → new` 路径（260 条，含删除的缓存） |

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

## 使用说明

* 归档脚本与文档里的**相对路径按迁移前的位置书写**，直接运行/点击可能失效；
  需要时请按 `MANIFEST.tsv` 把文件放回去，或对照主线对应物修改路径。
* 恢复一个退役脚本不会自动恢复它的 step 注册；要接回 `main.py` 需在
  `dcr/steps.py` 里重新 `register(StepDef(...))`。
* 主线仍然可用的同类能力：
  * 谱面内容特征：`step2/cm3p_mlp`（`step2/predict` 也接受 CM3P 内容表）；
  * 旧 play 级清洗：`step1/clean`；
  * 基线：`step2/predict` 的 `bias` / `mf_dot` / `mirt`。
