# 流程与编排

## 0. 一条命令

~~~bash
python main.py -c configs/exp_pairbest_1k.yaml
~~~

`main.py` 是唯一入口：按 YAML 依次调用 `scripts/` 下的 step 程序，把本次实验的
配置快照、日志、中间产物与结果全部放进 `experiments/<时间戳>[_{name}]/`，并把每一步的
命令行、耗时、输入输出指纹与指标写进 `manifest.json`。

## 1. 配置（configs/*.yaml）

一个 YAML = 一次实验。字段全部可省略，缺省用 `dcr/config.py` 的 `DEFAULTS`：

| 字段 | 含义 | 默认 |
|---|---|---|
| `name` | 实验名，拼在实验目录名后 | 空 |
| `tag` | 数据集标签，决定 `step0_{tag}` 等文件名 | `1k` |
| `python` | 解释器 | 与 `main.py` 相同 |
| `cache_dir` | 上游产物缓存（没跑的 step 从这里取） | `data/processed` |
| `osu_cache_dir` | `.osu` 正文缓存（跨实验复用） | `data/osu_cache` |
| `env` | 追加的环境变量 | `{}` |
| `io.out_format` | play 表格式：parquet / csv | `parquet` |
| `io.link_cache` | 复用缓存时在 `artifacts/` 建符号链接 | `true` |
| `io.hash` / `hash_max_mb` | 记录 sha256 / 超过该大小跳过 | `true` / 512 |
| `steps` | 要跑的 step，没写的不跑 | `{}` |
| `extra_steps` | 任意脚本挂进同一套编排 | `[]` |

~~~yaml
steps:
  step0:
    version: preprocess
    params:
      dump_dir: data/raw/extracted/2026_09_01_performance_mania_top_1000
  step1:
    version: pairbest              # 不写就是默认 pairbest；旧 play 级写 clean
    params:
      train_frac: 0.70
      split_method: pair-random
      split_seed: 20260928
      min_plays_per_player: 20
      gap_days: 365
      min_plays_per_pair: 2
      acc_floor: 0.90
      max_best_acc: 0.998
  step2:
    version: predict
    params:
      models: [bias, mf_dot, mirt]
      dim: 16
      epochs: 60
~~~

参数规则：

* 键名 = 脚本长选项，下划线换横线（`acc_floor` → `--acc-floor`）；
* `null` / `false` 表示不传；列表默认逗号拼接（`models: [bias, mf_dot]`）；
* 字符串可用占位符 `{tag}` `{exp}` `{artifacts}` `{results}` `{logs}`
  `{figures}` `{cache}`；
* 路径（包括 `--exp-root` / `--exp-dir`）都按**仓库根**解析；
* 由编排层接管的输出路径与 `tag` 不能写进 `params`，写了会直接报错。

## 2. 实验目录

默认落在仓库根的 `experiments/` 下（`dcr/paths.py` 的 `DEFAULT_EXPERIMENT_ROOT`）：

~~~text
experiments/20261020_193449_pairbest_1k/
  config.yaml      解析后的完整配置
  manifest.json    每步的命令行 / 参数 / 输入输出指纹 / 耗时 / 状态 / 指标
  run.sh           可重放的命令序列（自动 cd 到仓库根）
  logs/            pipeline.log + 每步一份 NN_step_version.log
  artifacts/       step0 play 表、beatmap_meta、step1 表、内容向量…
  results/         step2 的 npz 与 summary json
  figures/         图（分析类步骤）
~~~

`--exp-root` 换父目录，`--exp-dir` 指定已有目录（配合 `--resume` 续跑）。
`experiments/` 整体不进 git；参数与结论请落到 `configs/` 与 `docs/`。

## 3. step 版本

| step | version | 程序 | 说明 |
|---|---|---|---|
| step0 | `preprocess` | `scripts/step0_preprocess.py` | dump → play 表（5 列）+ 谱面元数据（17 列） |
| step0 | `chart_content` | `scripts/step0c_chart_content.py` | beatmap_id → CM3P 512 维内容向量 |
| step0 | `fetch_osu` | `scripts/step0d_fetch_osu.py` | 抓 `.osu` **正文**到 `osu_cache_dir` |
| step0 | `whitelist` | `scripts/step0g_whitelist.py` | 白名单表 + 缓存状态（`cacheable`） |
| step0 | `validate_mix` | `scripts/step0e_validate_mix.py` | 校验不同来源的 CM3P 向量能否混用 |
| step1 | `pairbest` | `scripts/step1_pairbest.py` | **默认 A**：一对一行，规则见 [data.md](data.md) §3 |
| step1 | `clean` | `scripts/step1_clean.py` | legacy play 级：划分后清洗 train 侧，需显式 `version: clean` |
| step2 | `predict` | `scripts/step2_predict.py` | bias / mf_dot / mirt（+ CM3P 内容向量） |
| step2 | `cm3p_mlp` | `scripts/step2_cm3p_mlp.py` | CM3P 内容锚定 MLP |

`python main.py --list-steps` 随时可查。

## 4. 输入怎么接上

每个 step 的输入按优先级解析：

1. 配置里写死的 `steps.<step>.inputs.<名字>`；
2. 本次运行上游 step 刚产出的路径；
3. `cache_dir` 里的习惯文件名。

| 输入 | 上游 | cache 候选 |
|---|---|---|
| step1 `plays` | step0 `plays` | `step0_{tag}.parquet` → `.csv` |
| step2 `train` | step1 `plays` | pairbest：`step1_{tag}_pairbest.parquet`；clean：`step1_{tag}.parquet` |
| step2 `content`（可选） | step0 `content` | `chart_content_cm3p.parquet` |
| step0/chart_content 的 `meta` | step0 `meta` | `beatmap_meta_{tag}.csv` |
| step0/preprocess 的 `whitelist`（可选） | step0 `whitelist` | `whitelist_{tag}.csv` |
| step0/whitelist 的 `.osu` 缓存 | — | `osu_cache_dir` |

命中缓存时会在实验目录的 `artifacts/` 里留一个相对符号链接，实验目录仍然自描述。
找不到输入时会明确报出缺哪个、可以怎么补。

## 5. 常用命令

~~~bash
# 默认口径 A
python main.py -c configs/exp_pairbest_1k.yaml
python main.py -c configs/exp_pairbest_10k.yaml

# CM3P
python main.py -c configs/exp_cm3p_1k.yaml
python main.py -c configs/exp_cm3p_mlp_1k.yaml

# 旧 clean 口径
python main.py -c configs/exp_std_1k.yaml
python main.py -c configs/exp_full_1k.yaml

# 参数扫描
for d in 4 8 16 32; do
  python main.py -c configs/exp_pairbest_1k.yaml --set name=dim$d --set steps.step2.params.dim=$d
done

# 只看计划 / 续跑 / 只跑部分 step
python main.py -c configs/exp_full_1k.yaml --dry-run
python main.py -c configs/exp_full_1k.yaml --exp-dir experiments/20261020_193449_pairbest_1k --resume
python main.py -c configs/exp_full_1k.yaml --only step0,step1
~~~

退出码：0 全部成功；1 某步失败（默认停止，`--keep-going` 继续）；2 配置/参数错误。

## 6. 扩展

新增一个 step 版本：在 `dcr/steps.py` 里 `register(StepDef(...))`，声明脚本、
输入、cache 候选名和 `plan()`（params → argv）。

任意脚本挂进来（不改代码）：

~~~yaml
extra_steps:
  - name: my_analysis
    script: scripts/xxx.py          # 或 cmd: [bash, scripts/xxx.sh]
    after: [step2]
    params: {tag: "{tag}", out-dir: "{results}"}
    outputs: ["{results}/xxx_*.csv"]
~~~

退役脚本、归档文档与历史日志在 [legacy/](../legacy/README.md)；
逐文件去向见 [legacy/MANIFEST.tsv](../legacy/MANIFEST.tsv)。
