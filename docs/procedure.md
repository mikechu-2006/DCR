# 流程

# main

由main.py调用程序运行。每个step可能存在多种版本，或允许使用多种不同参数。通过yaml启动以处理调用的step版本以及参数。指定实验目录，保存在experiments_{时间戳}，将实验参数，中间文件，运行结果集中存放在该目录文件夹中。

# step0

（默认step0是做好了不动的，只有step1会发生变化）

负责将源dump预处理为表格形式，包含格式转换，以及仅最基础的数据清理。
可以接受输入路径参数，输出路径参数。

# step1

输入：step0整理得到的表格

首先step1需要对数据进行train/test split 

然后对训练数据进行清洗

可以接受输入路径参数，输出路径参数。

# step2

在清洗训练数据上训练得到模型，然后在test数据集上进行推理，计算指标。

可以接受输入路径参数

---

# 实现（当前仓库）

上面是需求，下面是它在本仓库里的落地方式。

## 0. 一条命令

```bash
python main.py -c configs/exp_std_1k.yaml     # 一次实验 = 一个 YAML
```

`main.py` 是唯一入口：按 YAML 依次调用各 step 的程序，把这次实验的**参数、
中间文件、结果**全部放进新目录 `experiments_{时间戳}[_{name}]/`，并把每一步的
完整命令行、耗时、输入输出指纹与指标写进 `manifest.json`。

## 1. 配置文件（configs/*.yaml）

| 字段 | 含义 | 默认 |
|---|---|---|
| `name` | 实验名，拼在目录名之后 | 空 |
| `tag` | 数据集标签，决定 `step0_{tag}` / `step1_{tag}` 等文件名 | `1k` |
| `python` | 解释器；不写则与 `main.py` 相同 | 同 main.py |
| `cache_dir` | 上游产物缓存目录（没跑的 step 从这里取产物） | `data/processed` |
| `osu_cache_dir` | `.osu` 谱面缓存（**只有正文**，无音频/图片；跨实验复用，所以不在 `cache_dir` 下） | `data/osu_cache` |
| `env` | 追加的环境变量，如 `{OMP_NUM_THREADS: "4"}` | `{}` |
| `io.out_format` | step0 play 表格式 | `parquet` |
| `io.link_cache` | 复用缓存产物时在 `artifacts/` 建相对符号链接 | `true` |
| `io.hash` / `io.hash_max_mb` | 记录产物 sha256 / 超过该大小跳过 | `true` / 512 |
| `steps` | 要跑的 step（**没写的不跑**） | `{}` |
| `extra_steps` | 任意脚本挂进同一套编排 | `[]` |

每个 step 的写法：

```yaml
steps:
  step0:
    version: preprocess          # 版本；不写用该 step 的默认版本
    params:                      # 传给这个版本程序的参数
      dump_dir: data/raw/extracted/2026_09_01_performance_mania_top_1000
    # 可选：inputs.whitelist 显式指定白名单表（§4.3）；不写时自动取
    # cache_dir/whitelist_{tag}.csv —— 有它就在 step0 里剔除不可缓存的谱面
  step1:
    version: clean               # train/test 划分 + 清洗，输出带 split 列
    inputs:                      # 可选：显式指定输入（默认自动解析，见 §4）
      plays: data/processed/step0_1k.parquet
    params:
      train_frac: 0.70           # train 比例；test = 1 - 该值
      split_method: pair-random  # pair-random（默认）| beatmap-random
      split_seed: 20260928
      acc_floor: 0.90            # 只作用于划入 train 的行
      drop_frac: 0.30            # 同上
  step2:
    version: predict
    params:
      models: [bias, mf_dot, mirt]
      dim: 16
      epochs: 60
```

`params` 的规则：

* 名字 = 脚本的长选项，**下划线换横线**：`acc_floor` → `--acc-floor`；
* `null` / `false` → 不传该参数（用脚本自己的默认值）；
* 列表 → 逗号拼成一个参数（`models: [bias, mirt]` → `--models bias,mirt`）；
  声明为 repeat 的参数（如 step0c 的 `--osu-dir`）改写成重复 flag；
* 字符串里可用占位符：`{tag}` `{exp}` `{artifacts}` `{results}` `{logs}`
  `{figures}` `{cache}`，以及本次解析出的输入名；
* 由编排层接管的参数（输出路径、`tag` 等）不能写进 `params`，写了会直接报错
  并告诉你去哪写。

## 2. 实验目录

```
experiments_20261012_153948_full_1k/
  config.yaml      解析后的完整配置（含默认值），可直接再传给 main.py
  manifest.json    每步：命令行 / 参数 / 输入输出指纹 / 耗时 / 状态 / 指标
  run.sh           本次执行过的等价命令序列（手工重放用）
  logs/            pipeline.log + 每步一份 NN_step_version.log
  artifacts/       中间文件：step0 play 表、beatmap_meta、step1 清洗表、内容向量…
  results/         结果：step2 的 npz 与 summary json
  figures/         图（分析类步骤的输出）
```

`manifest.json` 中每一步都记录：

* `cmd_line`：可直接复制执行的完整命令；`cmd`：同上的数组形式；
* `inputs`：路径 + 来源（`config` / `step:<name>` / `cache`）；
* `outputs`：路径 + 大小 + sha256（大文件按 `io.hash_max_mb` 跳过）；
* `seconds` / `status` / `exit_code` / `log`；
* `report`：机器可读报告（step1 的漏斗与校验、step2 的指标）。

## 3. step 版本

| step | version | 程序 | 说明 |
|---|---|---|---|
| step0 | `preprocess` | `scripts/step0_preprocess.py` | dump → play 表（5 列）+ 谱面元数据 |
| step0 | `chart_content` | `scripts/step0c_chart_content.py` | beatmap_id → CM3P 内容向量 |
| step0 | `fetch_osu` | `scripts/step0d_fetch_osu.py` | 抓 `.osu` **正文**到 `osu_cache_dir`（默认整个白名单；自算向量的前置） |
| step0 | `whitelist` | `scripts/step0g_whitelist.py` | 白名单表 + 缓存状态；step0/preprocess 用它剔除不可缓存的谱面（§4.3） |
| step0 | `fetch_tags` | `scripts/step0f_fetch_tags.py` | 谱面标签：爬 osu! 谱面页的玩家投票 tag |
| step0 | `validate_mix` | `scripts/step0e_validate_mix.py` | 校验不同来源向量能否混用 |
| step1 | `clean` | `scripts/step1_clean.py` | **train/test 划分**（`split` 列）+ ACC 门槛 + 每玩家热身截断 |
| step2 | `predict` | `scripts/step2_predict.py` | 按 `split` 列取 train/test，训练 + 推理 + 指标 |

`python main.py --list-steps` 随时可查。

## 4. 划分与输入接续

### 4.1 train/test 划分在 step1（`split` 列）

划分**由 step1 负责**（2026-10-12 从 step2 移过来）：step2 默认只读 `split` 列，自己不划分。

| 参数（step1） | 默认 | 含义 |
|---|---|---|
| `train_frac` | `0.70` | 划入 train 的比例；test = 1 − 该值 |
| `split_method` | `pair-random` | `pair-random`：对 `(player_id, beatmap_id)` **有序对**均匀随机，同一对的行必在同一侧；`beatmap-random`：随机挑出 test 比例的 `beatmap_id`，这些谱面的**所有**行都进 test（谱面级外推，测试谱面在训练里完全不可见） |
| `split_seed` | `20260928` | 与 step2 旧的 `--split-seed` 相同，因此 `pair-random` 与旧流程的划分**逐对一致** |

step1 的输出 = step0 的全部行（顺序、数值不变）+ 一列 `split`：

| 值 | 含义 |
|---|---|
| `"train"` | 划入 train，且通过了清洗 |
| `"test"` | 划入 test，**未被清洗**（清洗是 train 侧的处理） |
| `"dropped"` | 划入 train，但被清洗删掉（行仍留在表里，可做消融） |

step2 的取数：`split=="train"` 的行**只用于训练**、`split=="test"` 的行**只用于算指标**、
`split=="dropped"` 忽略。step2 不再自己划分，也没有第二个测试口径（旧版的 `--split-from self` /
`--pair-universe` / `--test-pair-frac` 已随划分一起移除；历史数字见
[docs/step2_prediction.md](step2_prediction.md) §7 的口径提示）。

### 4.2 输入怎么接上

每个 step 的输入按优先级解析：

1. 配置里写死的 `steps.<step>.inputs.<名字>`；
2. 本次运行中上游 step 刚产出的路径；
3. `cache_dir` 里的习惯文件名：

| 输入 | 上游 | cache 里的候选文件名 |
|---|---|---|
| step1 `plays` | step0 `plays` | `step0_{tag}.parquet` → `step0_{tag}.csv` |
| step2 `train` | step1 `plays` | `step1_{tag}.parquet` → `step1_{tag}.csv`（**要有 `split` 列**；train/test 都从它取） |
| step2 `content`（可选） | step0 `content` | `chart_content_cm3p.parquet` |
| step0/chart_content、step0/fetch_tags 的 `meta`（可选） | step0 `meta` | `beatmap_meta_{tag}.csv` |
| step0/preprocess 的 `whitelist`（可选） | step0 `whitelist` | `whitelist_{tag}.csv`（§4.3） |
| step0/whitelist 的 `.osu` 缓存 | — | `osu_cache_dir`（默认 `data/osu_cache`，**不在** `cache_dir` 下） |

于是「只改 step2 参数」的实验配置写 **step1 + step2** 就够（step0 自动从 `cache_dir` 取）：
上游产物会在 `artifacts/` 里留一个相对符号链接，实验目录仍然自描述。
三处都找不到时会报出「缺哪个输入、可以怎么补」，而不是一个含糊的报错；
拿旧的无 `split` 列的 step1 表当输入时也会明确提示（加 `--force` 重跑 step1，几秒钟的事）。

step2 只声明了一个输入 `train`（= step1 的输出表）：旧版那条「step2 自己划分 + 再读一张 step0 表
当测试源」的路径已随划分迁到 step1 一起删除。

### 4.3 白名单表：抓不到 `.osu` 的谱面在 step0 剔除

CM3P 内容向量需要谱面正文，而 dump 不带正文。缓存方案是**整个白名单都尝试抓一遍**
（`step0/fetch_osu` → `osu_cache_dir`，只有 `.osu` 正文），已删除/受限的谱面（HTTP
404/403/451）永远抓不到。`step0/whitelist` 把结果落成一张表：

* `data/processed/whitelist_{tag}.csv`：全部白名单 + `cached / valid / md5_ok / cacheable`
  与 `fetch_status`，另写 `whitelist_{tag}_dropped.csv`（抓不到的清单及原因）和
  `results/whitelist_{tag}_report.json`；
* `manifests/whitelist_{tag}.csv`（`--manifest-out`，入库）：与缓存无关的全量 id 清单，
  新 clone / 远端抓取就用它。

`step0/preprocess` 声明了可选输入 `whitelist`，于是 **cache_dir 里有这张表就会自动取用**：
`cacheable=False` 的谱面连同它们的 play 一起从 step0 的两个产物里消失（一份 id 集合同时
驱动元数据表和两条 play 流，所以不可能对不上）。没有这张表时行为不变，仍是 dump 的全量 4K
白名单；想显式跑全量就把不带 `cacheable` 列的 `manifests/whitelist_{tag}.csv` 写进
`inputs.whitelist`。规则、计数与理由见
[docs/step0_preprocessing.md](step0_preprocessing.md) §2.4。

## 5. 常用命令

```bash
# 只跑 step2，复用 data/processed 里的 step0/step1（最常用）
python main.py -c configs/exp_std_1k.yaml

# 从 raw dump 全流程
python main.py -c configs/exp_full_1k.yaml

# 参数扫描：不改脚本、不记长参数，用 --set 覆盖任意字段
for d in 4 8 16 32; do
  python main.py -c configs/exp_std_1k.yaml --set name=dim${d} --set steps.step2.params.dim=${d}
done

# 先看要跑什么（不建目录、不执行）
python main.py -c configs/exp_full_1k.yaml --dry-run
python main.py -c configs/exp_full_1k.yaml --skip step0 --dry-run

# 中断后续跑同一个实验目录（产物已存在的 step 直接跳过）
python main.py -c configs/exp_full_1k.yaml \
    --exp-dir experiments_20261012_153948_full_1k --resume

# 只跑/跳过某些 step
python main.py -c configs/exp_full_1k.yaml --only step0,step1
python main.py -c configs/exp_full_1k.yaml --skip step0

# 查看解析后的完整配置 / 内置 step / 版本
python main.py -c configs/exp_std_1k.yaml --print-config
python main.py --list-steps
```

退出码：0 = 全部成功；1 = 某个 step 失败（默认立即停止，`--keep-going` 可继续）；
2 = 配置或参数错误（会打印一句可直接照做的说明）。

## 6. 扩展

* **新增一个 step 版本**：在 `dcr/steps.py` 里 `register(StepDef(...))`，
  声明脚本、输入、cache 候选名与 `plan()`（把 params 变成 argv）。
  之后任何 YAML 都能引用 `version: <新版本>`。
* **任意脚本挂进来**（不改代码）：

```yaml
extra_steps:
  - name: my_analysis
    script: scripts/xxx.py          # 或 cmd: [bash, scripts/xxx.sh]
    after: [step2]                  # 只影响 extra_steps 之间的顺序
    params: {tag: "{tag}", out-dir: "{results}"}
    outputs: ["{results}/xxx_*.csv"]   # 记进 manifest（支持通配符）
```

## 7. 与脚本的关系

step 的实现仍然是 `scripts/` 下的独立程序，可以脱离 `main.py` 单独运行；
编排层只按 YAML 组装参数、按顺序调用、把产物收进实验目录。为此给几个脚本补了
显式路径参数（不传时行为与以前完全一致）：

* step0：`--dump-dir / --beatmap-src / --out / --meta-out / --work-dir`（已有），
  新增 `--whitelist`（§4.3）；
* step0d：默认清单改成全量白名单 `manifests/whitelist_1k.csv`、默认缓存目录
  `data/osu_cache`（`--ids-file / --out-dir / --state` 都可覆盖）；
* step0g：`--tag --meta --cache --out --manifest-out --dropped-out --json`（新增）；
* step1：`--in / --out`（已有），新增 `--no-csv`（10k 只留 parquet）；
* step2：新增 `--train-path / --proc-dir / --out-dir / --split-col`（只读 step1 的一张表）；
* step0c：新增 `--meta-dir`（`--out` / `--ids-file` 不变）；
* step0f：`--meta --out-dir --work-dir`（页面缓存默认落在 `cache_dir/step0f_{tag}`，换实验目录也能断点续爬）。

`scripts/legacy/` 是退役脚本；当前脚本引用它时需要在 `sys.path` 里加上
`scripts/legacy`（`scripts/step0_preprocess.py` 依赖其中的 `parse_dump.py`）。
