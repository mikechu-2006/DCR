# scripts/ —— 脚本索引

本目录是**实现层**：一个 step 一个程序，都可以脱离 `main.py` 单独运行。
统一入口是仓库根的 `main.py`（读 `configs/*.yaml`，按顺序调用这里的程序，
把产物收进 `experiments_{时间戳}/`）；流程与配置说明见
[docs/procedure.md](../docs/procedure.md)。

## 管线 step（由 main.py 编排）

| 文件 | step 版本 | 说明 |
|---|---|---|
| `step0_preprocess.py` | `step0/preprocess` | 原始 dump → play 表（5 列）+ 谱面元数据（17 列） |
| `step0c_chart_content.py` | `step0/chart_content` | beatmap_id → CM3P 512 维内容向量（HF 预计算或本地推理） |
| `step0d_fetch_osu.py` | `step0/fetch_osu` | 按 id 清单抓 `.osu` **正文**（默认全量白名单；无音频/图片） |
| `step0g_whitelist.py` | `step0/whitelist` | 白名单表 + 缓存状态（`cacheable`），step0 据此剔除抓不到的谱面 |
| `step0e_validate_mix.py` | `step0/validate_mix` | 校验不同来源的内容向量能否混用 |
| `step0f_fetch_tags.py` | `step0/fetch_tags` | 爬谱面页拿玩家投票 tag（`chart_user_tags_{tag}.parquet`） |
| `step1_clean.py` | `step1/clean` | train/test 划分（`split` 列）+ 清洗：ACC 门槛 + 每玩家热身截断 |
| `step2_predict.py` | `step2/predict` | 按 `split` 列取 train/test，训练低秩回归 + 推理 + 指标 + 自检 |

显式路径参数（不传时保持旧行为）：

* `step0_preprocess.py`：`--dump-dir --beatmap-src --whitelist --out --meta-out --work-dir`
* `step0c_chart_content.py`：`--tag --meta-dir --ids-file --in --out`
* `step0d_fetch_osu.py`：`--ids-file --out-dir --state`（默认 `manifests/whitelist_1k.csv` → `data/osu_cache`）
* `step0g_whitelist.py`：`--tag --meta --cache --out --manifest-out --dropped-out --json`
* `step0f_fetch_tags.py`：`--tag --meta --out-dir --work-dir`
* `step1_clean.py`：`--in --out [--no-csv|--no-parquet]` + 划分 `--train-frac --split-method --split-seed`
* `step2_predict.py`：`--train-path --proc-dir --out-dir [--content-path] [--split-col]`

## 依赖与约定

* `scripts/legacy/` 是退役脚本（旧 v2 管线）。当前脚本引用其中的模块时，
  需要把 `scripts/legacy` 加进 `sys.path`——`step0_preprocess.py` 及 `probe_*.py`
  依赖 `legacy/parse_dump.py`，文件开头已处理。
* 需要 `parse_dump` 的脚本必须能 import 到 `scripts/legacy/parse_dump.py`；
  直接 `python scripts/step0_preprocess.py ...` 即可运行。

## 其余脚本（历史分析 / 诊断，未纳入编排）

| 分组 | 文件 | 说明 |
|---|---|---|
| 学习曲线 | `play_order_curve.py` `learning_rate_variance.py` | 依赖旧 v2 表（`charts_v2_*` / `plays_v2_*`） |
| 遗忘/时间 | `forgetting_curve.py` `practice_vs_time.py` | 同上；结论已归档到 docs/archive/ |
| 谱面分布 | `plot_chart_players.py` `plot_chart_sparsity.py` | 谱面玩家数 / 稀疏度分布图 |
| step2 诊断 | `diag_step2_*.py` | 容量、可辨识性、锚定、内容交互等专项诊断 |
| 数据探针 | `probe_*.py` | mod 策略、NF/CL 口径、现代成绩表覆盖等一次性核对 |
| 临时 | `_tmp_*.py` | 一次性计算，保留备查 |

这些脚本同样可以挂进统一编排（不改代码）：

```yaml
extra_steps:
  - name: my_analysis
    script: scripts/xxx.py
    params: {tag: "{tag}", out-dir: "{results}"}
    outputs: ["{results}/xxx_*.csv"]
```

## 运维脚本

| 文件 | 说明 |
|---|---|
| `remote_cm3p_run.sh` | 远程机器上补齐 CM3P 向量的一键流程（见 docs/cm3p_remote_runbook.md） |
| `slurm/*.slurm` | HPC / Slurm 提交模板（见 docs/cm3p_hpc_slurm.md） |
