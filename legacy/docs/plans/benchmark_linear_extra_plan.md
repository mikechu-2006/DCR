# 实验计划：randommerge 合并集 linear_extra 的 VSRG Dan Benchmark 评测

> **状态：已执行（2026-11-20）。** 结果见 [benchmark_linear_extra.md](benchmark_linear_extra.md)。
> 训练在 HPC CPU（job 14030754，1,095.2 s）完成；benchmark 推理、5 折校准和诊断在本机完成。
> 预注册的 H1 只在最高难度段成立：overall 0.4216 vs M1-L 0.7411，
> 但 expected>15 / msd Q4 上与 M1-L 不显著。
> 前序计划与报告：[benchmark_plan.md](benchmark_plan.md)、[benchmark_vsrg.md](benchmark_vsrg.md)。
> 新数据集结果：[randommerge_p2p4_results.md](randommerge_p2p4_results.md)。
> Benchmark 快照：VSRG-DanEstimation-Benchmark@e15f89b，samples/data.csv sha256 2b43165a…；
> 本地克隆在 .cache/benchmark/VSRG-DanEstimation-Benchmark/，与上次评测一致，未更新。

---

## 0. 目标与范围

在 benchmark 的 **1,382 张 4K 谱面**上，评测在 **randommerge 合并集**上训练的 linear_extra：

- **模型**：linear_extra，d_m = w'[C, msd_overall, sunny_star]，L2 λ = 1e-3；
- **内容**：CM3P legacy（audio-off，512-d，OliBomby/CM3P@26cc694）；
- **训练**：data/processed/step1_randommerge_pairbest_content.parquet，
  453,406 行 / 20,388 图 / 12,758 玩家；用 --fit-all 在全量行上拟合一个部署模型并保存权重；
- **评测**：沿用 2026-11-19 的管线 —— benchmark manifest / signals / legacy 内容表不变，
  5 折 chart-group out-of-fold 校准（isotonic 主口径、linear 对照），同一套 baseline、bootstrap 和候选 CSV。

要回答三个问题：

1. 与旧的全量 MLP + external（M1-L，OOF isotonic MAE **0.7411**）相比，
   换成更大但更“休闲”的合并集后，Dan 尺度上是变好还是变差？
2. 与 sunny_star isotonic（0.3367）、msd_overall isotonic（0.6286）的关系是否仍与上次一致？
3. 失败是否仍集中在 course/LN？RC（1,246 行）上是否仍接近 published Sunny？

**明确不做**：audio-on 臂、canon 臂、修改 benchmark 仓库、改 results/index.json、push、
按 benchmark 标签调超参或选模型。

---

## 1. 与上次的差异（只看变化项）

| 项 | 上次（M1/M2） | 本次（RM-L） |
|---|---|---|
| 训练表 | step1_1k_pairbest_nogap.parquet | step1_randommerge_pairbest_content.parquet |
| 行 / 图 / 玩家 | 109,569 / 6,819 / 1,039 | 453,406 / 20,388 / 12,758 |
| 模型 | mlp_extra（h64×2, dropout 0.1, 98 ep, λ=1e-2） | linear_extra（凸优化，λ=1e-3，无 epoch / 模型 seed） |
| 内容臂 | legacy + canon | legacy（合并集实验只跑 legacy） |
| 外部信号 | msd_overall, sunny_star，训练图 z-score | 同 |
| 校准 | 5 折 chart-group，OOF isotonic（主）/ linear（对照） | 同（同 manifest、seed=0、bootstrap=1000） |
| 稳健性 | P4 5 个划分 seed 的模型集成 | 默认无（linear 对模型 seed 确定；split-seed 臂可选，见 §12 Q6） |
| 主对照 | — | 旧 M1-L，**同一 5 折、同一 OOF 口径** |

**关键差异不是“更多数据”这么简单。** 合并集图更多（20,388 vs 6,819），但玩家总体更休闲；
benchmark 的谱面非常难（msd_overall 中位数 **31.86**），而合并集只有 **1.23%** 的图 ≥ 31.86。
按行算：

| 子集 | 旧 1k（109,706 行） | randommerge（453,406 行） |
|---|---:|---:|
| msd_overall ≥ 24 的行 | 81,993（74.7%） | 52,144（11.5%） |
| msd_overall ≥ 30 的行 | 26,608（24.3%） | 5,368（1.2%） |
| 图数 | 6,828 | 20,388 |
| 玩家数 | 1,039 | 12,758 |

（旧 1k 的行数用 step1_1k_pairbest.parquet 的 109,706 行；报告里的训练表过滤后是 109,569 行，差异 137 行不影响比例。）

也就是说，**合并集在 benchmark 难度段的训练比较预算反而更少**：msd ≥ 30 上从约 2.7 万行掉到约 0.5 万行。
这是本次实验最重要的先验风险，见 §10 与 §13。

---

## 2. 执行前已核对的事实

### 2.1 没有可直接加载的 randommerge checkpoint

- scripts/run_p2_p4_multiseed.py 的 train() 没有传 --save-model；
  P2 用 --chart-only --chart-folds 5，P4 用 6×6 / 4×4 分块。
- results/randommerge_p2p4_hpc/{p2,p4}/s{1..5}/signals/*.json 里 save_model: null、
  chart_only: true、fit_all: false。
- results/ 下除 benchmark 的 26 个 .pt 外，没有 randommerge 权重。

**结论**：文档里的 linear_extra 数字来自“5 折图不相交 / P4 分块”的**协议模型**，
不是可部署的单一 checkpoint。要在 benchmark 上推理，必须用同一超参做一次
--fit-all --save-model 的受控重训。

### 2.2 linear_extra 的定义与保存内容

- scripts/step2_cm3p_rank.py：
  d = w'[C, S]；C 用**训练图**逐维标准化，S=[msd_overall, sunny_star] 用**训练图**标准化；
  fit_linear 是带 L2 的逻辑回归（聚合 pair counts 上的精确 BT NLL）。
- scripts/run_p2_p4_multiseed.py 中 linear_extra 的参数是 --model linear_extra --lam 1e-3。
- --save-model 保存 mu/sd（512 维内容）、extra_mu/extra_sd（2 维外部信号）、w、
  content_md5、train_chart_ids、git_commit 等；linear 的 d_mu_raw/d_sd_raw = 0/1。
- scripts/benchmark_infer.py **已经支持** linear_extra：
  读 extra_names，用保存的 extra_mu/extra_sd 标准化，d_raw = X @ w，
  不重新按 benchmark 标准化；这正是我们要的“冻结尺度”。

### 2.3 数据、内容表、信号覆盖

| 项 | 事实 | 证据 |
|---|---|---|
| 合并集 pairbest | 453,406 行；20,388 图；12,758 玩家；split = 317,346 train / 136,060 test | step1_randommerge_pairbest_content.parquet |
| 内容表 | chart_content_cm3p_local.parquet：21,949 × 512 | 读表确认 |
| 内容覆盖 | P2/P4 训练时 content_coverage_maps = 1.0，n_maps = 20,388 | results/randommerge_p2p4_hpc/p2/s1/signals/cm3p_linear_extra.json |
| 外部信号表 | rank_signals_1k.parquet：21,949 图；msd_overall / sunny_star 在 20,388 图上均无 NaN | 读表确认 |
| benchmark 内容 | benchmark_content_legacy.parquet：1,382 × 512；crop=legacy、audio=off、cm3p_rev=OliBomby/CM3P@26cc694；n_windows 4–58 | 读表确认 |
| benchmark 信号 | benchmark_signals.parquet：1,382 行；sunny_star / msd_overall 无缺失 | 读表确认 |

**编码兼容性抽查**（不改任何东西，只读）：
benchmark 与训练表都覆盖的 rate=1.0 行共 38 行；独立编码的 cosine
37/38 ≥ 0.99999，最低 Volcanic(delta)（bid 3353155）0.896；n_windows 38/38 完全一致。
这说明两张表是同一 CM3P 配置，可以互用；个别图的 .osu 版本差异与上次评测时的跨表设置相同。
**本次不需要重新编码 benchmark 内容**（上一轮已经做过 1,382 行两臂编码）。

### 2.4 benchmark 快照与折

- 快照 commit、samples/data.csv sha256 与上次一致（见文首）。
- 旧 results/benchmark/eval_full/oof.parquet 的 5 折计数是 275 / 279 / 276 / 273 / 279；
  折由 benchmark_calibrate_eval.py 的 make_folds() 从 manifest 的
  family + expected 分箱和 chart_group 决定，**与预测列无关**。
- 因此把 RM_L 加进同一 pred 目录后，只要 manifest/seed 不变，折应与旧评测逐行一致；
  §9 的 G3 会做逐行断言，不靠假设。

### 2.5 与训练图的重叠

- benchmark 的 **201 个 unique bid / 210 行（15.2%）** 出现在 randommerge 的训练图中
  （208 行 RC + 2 行 LN；course 0）。
- 这 201 个 bid 与上次报告 §6 所说的“与 DCR whitelist 重叠的 201 张”是**同一组**
  （benchmark bid ∩ chart_content_cm3p_local.parquet = 201 bids / 210 rows）。
- 旧 1k 覆盖其中 199 个，合并集覆盖 201 个；多出的 2 个是合并集新增的图。
- 这不是 Dan 标签泄漏（训练目标是玩家 ACC 序，不是 benchmark 的 expected），
  但确实是**谱面内容重叠**。主表仍用全部 1,382 行，另做“排除 201 bid / 210 行”的敏感性。

### 2.6 运行时间参考

| 运行 | 规模 | 时间 |
|---|---|---:|
| randommerge p2/s1 linear_extra（5 个 chart fold + 全量 fit） | 317k 行，12.1M pair / 19.3M comparisons | 1,314 s（HPC 8 线程） |
| randommerge p4/s1 linear_extra | 166k 行 | 438 s |
| 本次 --fit-all（预估） | 453k 行，约 17.3M pair / 27.7M comparisons | 约 15–35 min（8 线程，取决于 pair 表构建） |

---

## 3. 训练 RM-L（全量部署模型）

### 3.1 命令

在仓库根目录：

~~~bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python scripts/step2_cm3p_rank.py \
  --in data/processed/step1_randommerge_pairbest_content.parquet \
  --content-path data/processed/chart_content_cm3p_local.parquet \
  --signal-table data/processed/rank_signals_1k.parquet \
  --extra-signals msd_overall,sunny_star \
  --model linear_extra --lam 1e-3 \
  --val-frac 0 --seed 0 --threads 8 \
  --name cm3p_linear_extra \
  --fit-all \
  --save-model results/benchmark/models/RM_L_full.pt \
  --out-signals results/benchmark/rm_linear/train/RM_L_signals.parquet \
  --out-json results/benchmark/rm_linear/train/RM_L.json
~~~

要点：

- --fit-all 忽略 split 列，**使用全部 453,406 行**（包括 pairbest 里原本标 test 的 136,060 行）。
  这是部署口径，与旧 benchmark 的 109,569 行全量模型一致。若你要求“只用 train 317k 行”，见 §12 Q2。
- --val-frac 0：不选 λ，固定 1e-3，与 randommerge_p2p4_results.md 的 linear_extra 完全一致。
- --name cm3p_linear_extra：输出信号列名，便于和 P2/P4 的已知列对齐。
- 该模型是凸问题，L-BFGS 从零初始化，**对模型 seed 不敏感**；--seed 0 只是保持命令行一致。
- 不传 --chart-folds、不传 --chart-only；--save-model 与这两个互斥条件都不会触发。

### 3.2 预期日志（用来自检，不是结果）

~~~text
[data] maps=20,388 (content coverage 100.0% of step1 maps) train_rows=453,406 test_rows=0
[pairs] train ~17.3M ( ~27.7M comparisons) | test skipped (--fit-all)
[val] 12758 fit players / 0 val players; fit pairs ...
[save-model] results/benchmark/models/RM_L_full.pt (linear_extra, fit_all=True, train_rows=453,406)
[done] train_nll=0.5xx (fit_all) -> RM_L_signals.parquet / RM_L.json
~~~

train_nll 是**全量 in-sample** 的 BT NLL，和 P2/P4 文档里的 test NLL 不可直接比较；
它只用来确认拟合没有发散（应落在 0.57–0.59 附近）。

### 3.3 训练后自检（G2）

用 benchmark_infer.py 在**训练内容表**上重算一遍，和刚写出的训练信号逐图对比：

~~~bash
python scripts/benchmark_infer.py \
  --model results/benchmark/models/RM_L_full.pt \
  --content data/processed/chart_content_cm3p_local.parquet \
  --signals data/processed/rank_signals_1k.parquet \
  --out results/benchmark/rm_linear/train/RM_L_selfcheck.parquet
~~~

按 row_id/beatmap_id 对齐 20,388 个训练图，比较 d_m 与 cm3p_linear_extra：
**max |Δ| 应 < 1e-6**（线性头 + 冻结标准化，理论上只差浮点舍入）。
若失败，先修保存/加载逻辑，不进入 benchmark 推理。

---

## 4. Benchmark 推理

~~~bash
python scripts/benchmark_infer.py \
  --model results/benchmark/models/RM_L_full.pt \
  --content data/processed/benchmark_content_legacy.parquet \
  --signals data/processed/benchmark_signals.parquet \
  --out results/benchmark/pred/RM_L.parquet
~~~

检查：

- 1,382 行，row_id 与 manifest 完全一致；
- d_m 无 NaN；打印 mean/sd；
- 与训练信号的 d_m 分布对比（范围诊断见 §6.4）。

---

## 5. 校准与评测（与上次同一套）

### 5.1 同一 5 折、同一 OOF 口径

- 主口径：expected ~ isotonic(d_m)，在 4 折上拟合、留出折上预测，5 折拼成 OOF；
- 对照：expected ~ a + b·d_m（线性，强制等距）；
- 折：StratifiedGroupKFold(5, shuffle=True, random_state=0)，
  strata = family + expected 五分位，groups = chart_group（同 bid 的 rate 变体同组）；
- bootstrap：按 chart_group 重采样 1,000 次，给 MAE 95% CI；
- baseline：常数 median、sunny_star isotonic/linear、msd_overall isotonic/linear、
  [sunny_star, msd_overall] 线性；
- published 算法：Sunny、Daniel、XDC1、Koishi、Azusa、ISOR、Roxy、Companella、Mixed、P1h（同 1,382 行，原样）。

### 5.2 代码改动：给 benchmark_calibrate_eval.py 加一个 run

scripts/benchmark_calibrate_eval.py 目前把 4 个组合硬编码在 RUNS 里。
最小、可回退的改法：

1. 在 RUNS 中增加：

~~~python
"RM_L": {"display": "DCR-CM3P-RM-Linear",
         "method": "CM3P + external linear (randommerge)",
         "arm": "legacy"},
~~~

2. 增加 --runs（逗号分隔，默认 M1_L,M1_C,M2_L,M2_C，保持旧命令行为不变）；
   需要新结果时传 --runs RM_L,M1_L,M1_C,M2_L,M2_C。
3. 把原来遍历 RUNS 的循环改成遍历 run_keys；delta_vs_M1_L 的 base 在
   M1_L 存在时保持不变，不存在则跳过。

**不改**：fit_oof、make_folds、bootstrap、候选 CSV 的 8 列契约。
这样旧 4 个组合在同一折上会被重算一遍，可用于断言与 eval_full 一致。

### 5.3 运行

~~~bash
python scripts/benchmark_calibrate_eval.py \
  --runs RM_L,M1_L,M1_C,M2_L,M2_C \
  --pred-dir results/benchmark/pred \
  --out-dir results/benchmark/eval_rm
~~~

results/benchmark/pred/RM_L.parquet 与旧 4 个预测放在同一目录；旧文件不修改。
eval_rm/ 是新增目录，eval_full/ 与 eval_p4/ 保持原样。

### 5.4 结果表模板（数字待填）

| 方法 | MAE | RMSE | bias | Spearman | Kendall | MAE 95% CI |
|---|---:|---:|---:|---:|---:|---|
| **RM-L isotonic（新）** | — | — | — | — | — | — |
| M1-L isotonic（旧） | 0.7411 | 1.4485 | +0.0069 | 0.8437 | 0.6957 | [0.6810, 0.8054] |
| M1-C isotonic | 0.7918 | 1.6234 | −0.0015 | 0.8450 | 0.6963 | [0.7249, 0.8688] |
| M2-L isotonic | 1.1489 | 1.9669 | −0.0022 | 0.6609 | 0.5069 | [1.0728, 1.2388] |
| M2-C isotonic | 1.1248 | 2.0585 | −0.0056 | 0.6823 | 0.5272 | [1.0442, 1.2185] |
| Sunny isotonic | 0.3367 | 0.5376 | +0.0001 | 0.9060 | — | — |
| MSD isotonic | 0.6286 | 1.0077 | +0.0012 | 0.7849 | — | — |
| [Sunny, MSD] 线性 | 0.6568 | 0.8642 | −0.0006 | 0.9021 | — | — |
| 常数 median | 1.9195 | 2.8995 | 0.0000 | −0.0094 | — | — |

（旧 4 行的数字来自 benchmark_vsrg.md；新表会用同一脚本重算，G3 断言它们不变。）

---

## 6. 分层、敏感性与诊断

### 6.1 必须报的分层

- **RC**（1,246 行）：RM-L、M1-L、Sunny isotonic、MSD isotonic 的 MAE + 95% CI；
- **family**：RC / LN / course 的 MAE、RMSE、n；
- **difficulty**：expected < 11 / 11–15 / > 15（沿用旧 per_group 口径）。

### 6.2 重叠敏感性（新增）

用 checkpoint 里的 train_chart_ids（20,388 个训练图）定义重叠：manifest 的 bid 在其中 → “seen”。

- 主表：全部 1,382 行；
- 干净子集：排除 201 bid / **210 行**（RC 1,038 + LN 100 + course 34 = **1,172 行**）；
- 报告两者的 RM-L、M1-L、Sunny、MSD MAE（配对 ΔMAE + chart-group bootstrap CI）。

**注意**：这里不重新拟合校准曲线。OOF 预测已经是在全部 1,382 行上交叉拟合的；
敏感性只在**评估子集**上取子集指标，保持同一折，避免“换折后不可比”。

### 6.3 难度带诊断（新增，针对 §1 的风险）

按 benchmark 的 msd_overall 四分位（或固定 <24 / 24–30 / 30–34 / ≥34）分带，报：

- n、RM-L / M1-L / Sunny 的 MAE 与 Spearman；
- 每带的 d_m 均值/标准差；
- 目的：看 RM-L 是否在 msd 高端（最接近训练分布边缘）明显崩掉。

### 6.4 范围/外推诊断（新增）

- 训练 20,388 图的 d_m 分位数（p1、p5、p50、p95、p99、min、max）；
- benchmark 1,382 行的 d_m 分位数；
- benchmark 行落在训练 [min, max] 之外、[p1, p99] 之外的比例；
- 分“范围内 / 范围外”各报 MAE。

理由：isotonic 的 out_of_bounds="clip" 会把超出训练范围的 d_m 全部压到端点；
如果 benchmark 大部分行在训练范围之外，MAE 会被端点常数支配。这是解释“为什么变好/变差”的关键诊断。

### 6.5 配对与 head-to-head

- RM-L vs M1-L：逐行 |delta| 的 wins/ties/losses、配对 ΔMAE 的 95% CI；
- RM-L vs Sunny isotonic：同上；
- RM-L vs M1-L 的 Spearman 差（同一批行）。

### 6.6 汇总脚本

新增 scripts/benchmark_breakdown.py，输入：

- --oof results/benchmark/eval_rm/oof.parquet（含 RM_L / M1_L / sunny_iso / msd_iso 等列）；
- --manifest data/processed/benchmark_manifest.parquet；
- --signals data/processed/benchmark_signals.parquet；
- --model results/benchmark/models/RM_L_full.pt（读 train_chart_ids）；
- --train-signals results/benchmark/rm_linear/train/RM_L_signals.parquet（读训练 d_m）；
- --out-dir results/benchmark/eval_rm/breakdown。

输出：

| 文件 | 内容 |
|---|---|
| rc_summary.csv | RC 子集 MAE + bootstrap CI（RM-L / M1-L / sunny / msd） |
| family_summary.csv | 按 family 的 MAE/RMSE |
| overlap_sensitivity.csv | 全部 vs 排除 210 行 |
| difficulty_band_summary.csv | 按 msd 四分位 / expected 带的指标 |
| range_diagnostic.csv | 训练 vs benchmark 的 d_m 分位数、范围外比例、范围内外 MAE |
| paired_delta.csv | RM-L vs M1-L / Sunny 的配对 ΔMAE + CI |
| head_to_head.csv | 逐行 wins/ties/losses |
| summary.md | 上述表的可读汇总 |

实现上直接复用 benchmark_calibrate_eval.py 的 metrics() 与按 chart_group 的 bootstrap 逻辑
（import 或复制 20 行），不重写指标定义。

---

## 7. 代码改动清单

| 文件 | 改动 | 目的 | 风险 |
|---|---|---|---|
| scripts/benchmark_calibrate_eval.py | RUNS 增加 RM_L；新增 --runs（默认旧 4 个）；循环改用 run_keys | 同一脚本评测新臂 + 旧 4 臂，折和校准逻辑不变 | 低；默认路径保持旧行为 |
| scripts/benchmark_breakdown.py | 新增 | RC / family / 重叠 / 难度带 / 范围 / 配对汇总 | 低；只读 oof.parquet 和 manifest |
| scripts/run_benchmark_predict.py | **不改** | 旧的 4 个模型推理命令保持原样 | — |
| scripts/benchmark_infer.py | **不改** | 已支持 linear_extra | — |
| scripts/step2_cm3p_rank.py | **不改** | 已支持 --fit-all --save-model | — |
| results/benchmark/README.md | 增加 RM-L 条目 | 产物索引 | — |
| docs/benchmark_linear_extra.md | 新增 | 本实验报告 | — |

如果不想动 benchmark_calibrate_eval.py，替代方案是复制一份
benchmark_calibrate_eval_rm.py 只改 RUNS；但那样会把校准代码分叉，不推荐。

---

## 8. 产物

| 路径 | 内容 |
|---|---|
| results/benchmark/models/RM_L_full.pt | 全量 linear_extra（含 mu/sd/extra_mu/extra_sd/w/train_chart_ids） |
| results/benchmark/rm_linear/train/RM_L_signals.parquet | 20,388 个训练图的 d_m（训练信号） |
| results/benchmark/rm_linear/train/RM_L.json | 训练 JSON（行数、pair 数、NLL、args、content_md5） |
| results/benchmark/rm_linear/train/RM_L_selfcheck.parquet | 自检推理输出（G2） |
| results/benchmark/pred/RM_L.parquet | benchmark 1,382 行的 d_m |
| results/benchmark/eval_rm/summary.md / summary.csv | 5 个 run + baseline 的主表 |
| results/benchmark/eval_rm/metrics.json | 全部指标、CI、配对差 |
| results/benchmark/eval_rm/oof.parquet | 逐行 OOF got、fold、原始 d_m |
| results/benchmark/eval_rm/per_group.csv | family/pattern/difficulty 分层 |
| results/benchmark/eval_rm/published_baselines.csv | published 算法同表 |
| results/benchmark/eval_rm/candidates/DCR-CM3P-RM-Linear.csv | 8 列候选提交文件（OOF isotonic） |
| results/benchmark/eval_rm/breakdown/* | §6 的分层/敏感性/诊断表 |
| results/benchmark/viz/RM_L_*.png | dashboard 图（RM-L vs Sunny、RM-L vs M1-L、RC） |
| docs/benchmark_linear_extra.md | 方法、契约、结果、限制、复现 |
| results/benchmark/logs/ | 训练、推理、评测日志 |

默认**不**复制到 benchmark 仓库、**不**改 results/index.json、**不** push。

---

## 9. Gates

| Gate | 条件 | 不通过时 |
|---|---|---|
| **G0 契约** | benchmark commit / data.csv sha256 未变；1,382 行；legacy 内容 1,382×512；两个信号 1,382 行无缺失 | 停，先查数据 |
| **G1 训练覆盖** | n_maps=20,388、n_train_rows=453,406、content coverage 100%、w 长度 514、extra_names=[msd_overall,sunny_star]、content_md5 与表一致 | 停，先查输入表/过滤 |
| **G2 推理自检** | 训练内容表上 benchmark_infer 的 d_m 与训练信号 max abs Δ < 1e-6；benchmark 推理 1,382 行无 NaN | 停，修保存/加载 |
| **G3 折与旧结果一致** | 新 eval_rm/oof.parquet 的 fold 与 eval_full/oof.parquet 逐行相同；M1_L_isotonic 的 MAE / Spearman 与旧值差 < 1e-9 | 停，查 manifest/环境版本 |
| **G4 主结果完整** | RM-L 的 isotonic/linear/raw 三套指标、CI、RC/family、重叠敏感性、范围诊断都写出 | 不写结论，先补齐 |
| **G5 负结果也报** | 若 RM-L 不优于常数 baseline 或不优于 M1-L，照实报告，不调参、不换折、不删臂 | — |

---

## 10. 风险与应对

| 风险 | 触发条件 | 应对 |
|---|---|---|
| **训练难度覆盖不足**（本次最大风险） | 合并集 msd≥30 只有 5,368 行（1.2%），benchmark 中位数 31.9；RM-L 在 msd 高端 MAE 明显差 | §6.3 难度带 + §6.4 范围诊断；结论限定“在该数据分布下”；可选 hard-chart 臂见 Q4 |
| **谱面重叠** | 201 bid / 210 行在训练图中 | 主表 + 排除 210 行的敏感性；不把重叠当标签泄漏，但明确写 |
| **目标错配** | BT 潜变量 vs 社区 Dan 标注 | 与上次相同的解释限制；报 raw Spearman/Kendall 与校准后 MAE 两套 |
| **isotonic 端点截断** | benchmark d_m 大量落在训练范围外 | §6.4 统计范围外比例；报告“范围内/外”分别的 MAE；不把截断后的 MAE 当成模型排序能力 |
| **校准用了 benchmark 标签** | 主表是 OOF 校准后的 MAE | 与上次一致，明确写；raw 指标单列 |
| **内容跨表小差异** | 38 个 rate=1.0 重叠行中 1 行 cosine 0.896 | 与上次同一设置；不重新编码；在限制里记录 |
| **本地内存/时间** | 453k 行、约 17M pair；本机 11 GB（可用约 6 GB） | 先本地；OOM/超时切 HPC i64m512u；见 Q8 |
| **版本漂移** | sklearn isotonic 版本变化导致旧数字不能复现 | G3 断言；不通过则固定环境或记录版本 |
| **benchmark 更新** | 上游 commit 变化 | 固定 e15f89b；不自动 pull |

---

## 11. 算力与时间线

| 阶段 | 工作 | 预计 |
|---|---|---|
| T0 | 数据/内容/信号/快照自检（G0），训练 RM-L（G1） | 30–45 min（本地） |
| T1 | 自检推理（G2）+ benchmark 推理 | 5 min |
| T2 | 5 折校准评测（G3） | 5–10 min |
| T3 | 分层/敏感性/诊断（G4）+ dashboard | 10 min |
| T4 | 写 docs/benchmark_linear_extra.md，更新 README | 30–60 min |

总时长约 1.5–2 h（不含 HPC 排队）。训练和评测都是 CPU，不需要 GPU。

---

## 12. 需要你确认 / 我不明白的地方

下面每一条都给了默认值；如果你不特别说，我按默认执行。

**Q1（最关键）：“新数据集上的模型”具体指哪一个训练实例？**
randommerge 的文档只保存了每个 split 的 signals/*.parquet 和 JSON，**没有保存 .pt**；
P2 的 linear_extra 是 5 个 chart-fold 模型，P4 是 6×6 块模型，都不是单一可部署模型。
我默认理解为：

> (a) 用全部 453,406 行 --fit-all 重训一个 linear_extra，保存
> results/benchmark/models/RM_L_full.pt，再在 benchmark 上推理。

如果你指的是 (b) P2 某个 fold 的模型、(c) P4 某个 seed / 5-seed 集成、
(d) 你手上已有的某个 .pt（请给路径），请告诉我。不同选择会改变训练行数和“模型见过的图”集合。

**Q2：训练行范围用全部 453,406 行，还是只用 317,346 行 train？**
--fit-all 会用全部行（包括 pairbest 里原本标 test 的 136,060 行）。
默认：**全部**（部署口径，与旧 benchmark 的全量模型一致）。
如果你要严格复现 P2 的“只用 train 行”口径，我改成 317,346 行，但那个模型不是最终部署模型。

**Q3：201 bid / 210 行的重叠怎么处理？**
默认：**主表用全部 1,382 行**，另报排除这 210 行的敏感性（1,172 行）。
与上次报告 §6 的“201 张 DCR whitelist”是同一组；旧 1k 覆盖 199 个，合并集覆盖 201 个。
如果你希望主表就用“干净”的 1,172 行，请说；那样与上次主表的口径会不一致。

**Q4：合并集更“休闲”，是否还要加一个 hard-chart 诊断臂？**
事实：benchmark 的 msd_overall 中位数 31.9；合并集只有 1.2% 的图 ≥ 31.9，
msd≥30 的训练行只有 5,368（旧 1k 是 26,608）。
默认：**只跑全量主臂**，把难度覆盖作为诊断和解释写进报告；
如果你要，我可以额外加一个“只用 msd≥24 图 / 或按难度加权”的 linear_extra 诊断臂
（同一管线，多一次训练 + 推理），但**不放进主表**。

**Q5：对照范围只做 RM-L + 旧 4 个 + baseline，还是也在合并集上补跑其他方法？**
默认：**只 RM-L**；旧的 4 个组合复用已有预测（同折重算以确认 G3），不在合并集上重训
mlp_extra/mlp/linear。如果你要“合并集 4 方法全套”，需要额外 3 次训练 + 推理。

**Q6：要不要 P2 split-seed 稳健性臂？**
旧报告有 P4 5-seed 集成；linear_extra 对模型 seed 是确定性的，所以真正的方差来自
“训练行/图的选择”。默认：**不跑** 5 个 split-seed 的 benchmark 迁移（那要 5 次训练）；
如果你要，我可以在主结果出来后加跑，报告 benchmark MAE 的 seed 间 mean±std。

**Q7：命名、候选 CSV、可视化？**
默认：

- run key：RM_L；
- 候选 CSV：DCR-CM3P-RM-Linear.csv；
- 生成 dashboard 图（RM-L vs Sunny、RM-L vs M1-L、RC）；
- **不**复制到 benchmark 仓库、**不**改 results/index.json、**不** push。

名字不合意可以直接改，不影响模型。

**Q8：本地还是 HPC？**
本地：8 核、11 GB（当前可用约 6 GB），预估 20–40 min；HPC i64m512u 也可（现有 slurm 模板）。
默认：**先本地**，OOM/超时再切 HPC。如果必须 HPC，请说。

**Q9：报告放哪里？**
默认：新建 docs/benchmark_linear_extra.md，在 docs/benchmark_vsrg.md 和
results/benchmark/README.md 加链接；不覆盖旧报告。

**Q10：本次是否只做“第 1 项”？**
你的消息以“1.”开头，我按“本次只做这一项”理解；如果后面还有第 2、3 项，
请一并给我，我在计划里预留接口（例如 audio-on / canon / hard-chart 臂）。

---

## 13. 预注册的读法（避免结果出来后换叙事）

**主比较**

1. RM_L_isotonic 与 M1_L_isotonic 在全部 1,382 行上的 MAE 差（配对、按 chart_group bootstrap 95% CI）。
   - 这是唯一的主假设检验；其他都是描述性/敏感性。
2. RC 子集（1,246 行）上的同一比较。
3. 干净子集（1,172 行）上的同一比较，作为重叠敏感性。

**预期（写在这里，结果出来对照）**

- **H1**：由于合并集在 msd≥30 上的训练行只有旧 1k 的约 1/5，RM-L 在 benchmark 高端
  （msd 高、expected 高）很可能**不优于** M1-L；overall MAE 可能持平或更差。
  如果 RM-L 反而更好，说明“更多玩家/图”的覆盖收益超过了难度覆盖的损失，这会是更强的结论。
- **H2**：RM-L 与 M1-L 都可能仍明显差于 sunny_star isotonic（0.3367）；
  但 RM-L 至少应优于常数 baseline（1.9195），否则说明 d_m 在 Dan 尺度上没有可用信号。
- **H3**：course/LN 仍是主要失败来源；RC 上的差距小于 overall。
- **H4**：raw Spearman 可能不低，但 MAE 受尺度/端点截断影响；两套指标分开报，不互相替代。

**不做的事**

- 不因为 RM-L 结果不好而改 λ、改校准、换折、删臂；
- 不把“OOF isotonic 后的 MAE”说成 label-free；
- 不把 P2/P4 的 NLL 优势直接外推成 Dan MAE 优势；
- 不在同一份报告里混合“全量模型”和“P4 5-seed 集成”的口径而不标注。

---

## 14. 复现命令（汇总）

~~~bash
# 0) 只读自检
python - <<'PY'
import pandas as pd, torch
# 见 §2.3；确认 1,382 / 20,388 / 453,406 与信号完整性
PY

# 1) 训练（G1）
python scripts/step2_cm3p_rank.py \
  --in data/processed/step1_randommerge_pairbest_content.parquet \
  --content-path data/processed/chart_content_cm3p_local.parquet \
  --signal-table data/processed/rank_signals_1k.parquet \
  --extra-signals msd_overall,sunny_star \
  --model linear_extra --lam 1e-3 --val-frac 0 --seed 0 --threads 8 \
  --name cm3p_linear_extra --fit-all \
  --save-model results/benchmark/models/RM_L_full.pt \
  --out-signals results/benchmark/rm_linear/train/RM_L_signals.parquet \
  --out-json results/benchmark/rm_linear/train/RM_L.json

# 2) 自检（G2）
python scripts/benchmark_infer.py \
  --model results/benchmark/models/RM_L_full.pt \
  --content data/processed/chart_content_cm3p_local.parquet \
  --signals data/processed/rank_signals_1k.parquet \
  --out results/benchmark/rm_linear/train/RM_L_selfcheck.parquet

# 3) benchmark 推理
python scripts/benchmark_infer.py \
  --model results/benchmark/models/RM_L_full.pt \
  --content data/processed/benchmark_content_legacy.parquet \
  --signals data/processed/benchmark_signals.parquet \
  --out results/benchmark/pred/RM_L.parquet

# 4) 同一 5 折评测（G3）
python scripts/benchmark_calibrate_eval.py \
  --runs RM_L,M1_L,M1_C,M2_L,M2_C \
  --pred-dir results/benchmark/pred \
  --out-dir results/benchmark/eval_rm

# 5) 分层/敏感性/诊断
python scripts/benchmark_breakdown.py \
  --oof results/benchmark/eval_rm/oof.parquet \
  --manifest data/processed/benchmark_manifest.parquet \
  --signals data/processed/benchmark_signals.parquet \
  --model results/benchmark/models/RM_L_full.pt \
  --train-signals results/benchmark/rm_linear/train/RM_L_signals.parquet \
  --out-dir results/benchmark/eval_rm/breakdown

# 6) 可视化（可选）
python scripts/render_benchmark_dashboard.py \
  --base results/benchmark/eval_rm/candidates/DCR-CM3P-RM-Linear.csv \
  --compare results/benchmark/eval_rm/candidates/DCR-CM3P-XMLP-Legacy.csv \
  --scope ALL --out results/benchmark/viz/RM_L_vs_M1L_ALL.png
~~~

---

## 15. 一句话总结

**这不是“把旧管线原样再跑一遍”：新数据集更大，但在 benchmark 难度段的训练行反而更少；
本次的主要价值是把“数据规模 vs 难度覆盖”的权衡量化出来，并用与上次逐行可比的 5 折口径
给出 RM-L 与 M1-L / Sunny / MSD 的差距。**
