# 模型：oracle 基线、CM3P、MinaCalc

本文件只保留主线模型的口径、命令与主结果。设计论证、消融与逐项检查在归档文档：

* [legacy/docs/step2_prediction.md](../legacy/docs/step2_prediction.md)
* [legacy/docs/cm3p_step2_integration_plan.md](../legacy/docs/cm3p_step2_integration_plan.md)
* [legacy/docs/cm3p_step2_results.md](../legacy/docs/cm3p_step2_results.md)
* [legacy/docs/cm3p_step2_mlp.md](../legacy/docs/cm3p_step2_mlp.md)

## 1. 接口与指标

step2 只读 step1 的输出表，按 `split` 列取数：

* `split == "train"` 的行只用于训练；
* `split == "test"` 的行只用于算指标；
* `split == "dropped"` 忽略。

指标都在 test 行上算：

| 指标 | 含义 |
|---|---|
| RMSE | loss 上的均方根误差，越小越好 |
| Spearman | 预测 loss 与真实 loss 的秩相关，越大越好 |
| n_scored | 参与打分的 test 行数 |
| MAE / pp_p50 | 换回 ACC 后的平均绝对误差 / 绝对误差中位数（百分点）；只在 pairbest 分析里报告 |

观测单位由 step1 决定：默认 pairbest 是「一行一个 (玩家段, 谱面) 对」；
legacy clean 是「一行一个 play」。两套的数字不可直接比较。

## 2. oracle 基线：`step2/predict`

~~~text
L_hat = b0 + bu[u] + bm[m] + interaction
  bias   : interaction = 0
  mf_dot : interaction = <P[u], C[m]>
~~~

| 参数 | 默认 | 说明 |
|---|---|---|
| `--models` | `bias,mf_dot,mirt` | 逗号分隔，一次跑多个变体 |
| `--dim` | 16 | 潜向量维度 |
| `--epochs` / `--lr` / `--wd` | 60 / 0.02 / 1e-4 | 训练 |
| `--features` | F0 | F1 加 `ln(playcount_cur)`；F2 加自然年实体/漂移 |
| `--p-drift` | none | `logpc` / `time`：玩家向量的练习量/时间漂移 |
| `--chart-content` | 无 | `cm3p`：用冻结内容向量替换 `C` |
| `--d-constraint` | none | mirt 的 `D` 约束：`nonneg` / `simplex` |

运行时会跑 9 项自检（划分互斥、loss 范围、标签可逆等），只有全部通过才落盘。
产物：`results/step2_{tag}_{model}_{features}[_dimD]_{target}.npz`（潜向量 + 预测）
与累积的 `results/step2_{tag}_summary_{target}.json`。

### 主结果（legacy play 级 clean）

| 模型 | 1k RMSE | 1k Spearman | 10k RMSE | 10k Spearman |
|---|---|---|---|---|
| bias | 1.5236 | 0.7330 | 1.3727 | 0.6874 |
| mf_dot | 1.4548 | 0.7632 | 1.3327 | 0.7079 |
| **mirt** | **1.4582** | **0.7633** | **1.3331** | **0.7096** |

### 默认口径 A（pairbest）

观测单位与标签支撑不同，RMSE 不能与上表直接比。mf_dot dim=8：

| tag | RMSE | Spearman | pp_p50 |
|---|---|---|---|
| 1k | 0.7015 | 0.6749 | 0.4327 |
| 10k | 0.6844 | 0.6966 | 0.5607 |

## 3. CM3P 内容向量

CM3P（[OliBomby/CM3P](https://github.com/OliBomby/CM3P)）给每张谱面一个 **512 维冻结向量**。
两条用法：

### 3.1 线性点积（`step2/predict` + 内容表）

`--chart-content cm3p --content-path <表> --content-dim 64 --content-reduce pca`：

* `C[m]` 是冻结的内容向量（buffer，不是参数），`P` 与它同空间；
* 每张谱面只剩 `bm` 一个自由参数（1-hot 的 `C` 是 `dim` 个）；
* 没有内容行的谱面会从训练与评估两张表一起删掉，所以数字只在「有内容向量」的子集上可比。

配置：[configs/exp_cm3p_1k.yaml](../configs/exp_cm3p_1k.yaml)。

### 3.2 内容锚定 MLP（`step2/cm3p_mlp`）

冻结的 512 维向量先过一个小 MLP（`--hid 64 --depth 2`，LayerNorm）投影到交互空间
`--dim 16`，再与玩家向量做点积；`--content-reduce raw` 表示不做 PCA。门控 `w`
默认为 1，`w=0` 就退化成加性模型。自检：

~~~bash
python scripts/test_step2_cm3p_mlp.py
~~~

配置：[configs/exp_cm3p_mlp_1k.yaml](../configs/exp_cm3p_mlp_1k.yaml)。

### 3.3 主结果（clean 口径、覆盖子集）

CM3P 覆盖约一半谱面，下列 RMSE 只在「有内容向量」的子集上、且 test 仍按旧 clean
口径取，不要和 pairbest 的数字混比：

| 模型 | dim | test RMSE |
|---|---|---|
| 1-hot | 8 | 1.58147 |
| 1-hot | 16 | 1.58314 |
| **CM3P 线性** | **32** | **1.57595** |
| CM3P MLP d16 | 16 | 不如线性（过拟合，见归档文档） |

### 3.4 CM3P 数据准备

~~~bash
# 1) 抓 .osu 正文（默认整个白名单，可断点续传）
python scripts/step0d_fetch_osu.py --ids-file manifests/whitelist_1k.csv --out-dir data/osu_cache

# 2) 无音频推理 → 本地内容表
python scripts/step0c_chart_content.py --source osu-dir --ids-file manifests/whitelist_1k.csv \
    --osu-dir data/osu_cache --audio off --out data/processed/chart_content_cm3p_local.parquet

# 3) 混用探针（可选）：与已发布表的重叠图余弦是否够高
python scripts/step0e_validate_mix.py --computed data/processed/chart_content_cm3p_probe.parquet

# 4) 合并 + 重建白名单，然后让 step0 剔除抓不到正文的谱面
python scripts/step0c_chart_content.py --source table --in data/processed/chart_content_cm3p_local.parquet --tag 1k
python scripts/step0g_whitelist.py --tag 1k --meta data/processed/beatmap_meta_1k.csv \
    --cache data/osu_cache --out data/processed/whitelist_1k.csv
~~~

远程机器与 Slurm 的完整流程归档在
[legacy/docs/cm3p_remote_runbook.md](../legacy/docs/cm3p_remote_runbook.md) 与
[legacy/docs/cm3p_hpc_slurm.md](../legacy/docs/cm3p_hpc_slurm.md)；
HPC 依赖见 `requirements-cm3p.txt`、`vendor/` 与 `scripts/slurm/*.slurm`。

## 4. MinaCalc

MinaCalc 相关只保留 [minacalc/](../minacalc/) 这个包本身（逐区间可微打分、C++ 区间导出、
玩家技能求解）；用法见 [minacalc/README.md](../minacalc/README.md)。

原先的 MSD 特征路线（`step0h_minacalc_msd.py`、`step2_minacalc_as_feature.py`、
`step2_minacalc_L2.py`）与 fit-skills 脚本、配置、设计文档都已归档到 `legacy/`，
不再注册为 step。
