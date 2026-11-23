# 数据：step0 + step1

本文件是预处理的口径说明；完整论证、逐项实测与坑在归档文档：

* [legacy/docs/step0_preprocessing.md](../legacy/docs/step0_preprocessing.md)
* [legacy/docs/step1_pairbest.md](../legacy/docs/step1_pairbest.md)
* [legacy/docs/step1_cleaning.md](../legacy/docs/step1_cleaning.md)
* [legacy/docs/step1_cleaning_analysis.md](../legacy/docs/step1_cleaning_analysis.md)

## 1. 数据来源

来自 [data.ppy.sh](https://data.ppy.sh/) 的 osu!mania 4K 成绩 dump：

| tag | dump | 可用表 |
|---|---|---|
| `1k` | `2026_09_01_performance_mania_top_1000` | `scores`（modern）+ `osu_scores_mania_high`（legacy）+ `osu_beatmaps` |
| `10k` | `2026_09_01_performance_mania_top_10000` | 只有 legacy；谱面元数据借用 1k 的 `osu_beatmaps` |

数据放在 `data/raw/extracted*/...`，约 14 GB，不入库。下载命令见根 [README](../README.md)。

## 2. step0/preprocess

原始 SQL → 两张表。只做「什么算一条有效观测」，不做聚合、不做玩家门槛、不加派生特征。

### 2.1 输出契约

**play 表**（`step0_{tag}.{parquet,csv}`，5 列）：

| 列 | 含义 |
|---|---|
| `player_id` | osu! user id |
| `beatmap_id` | 谱面 id（1.0× NM/CL，所以谱面 id 就是曲目，没有 rate 列） |
| `timestamp` | 游玩时间 |
| `playcount_cur` | 这是该玩家在该谱面的第几次记录（step0 原值，下游不重算） |
| `loss` | `log(1 - ACC)`，float32；ACC 由 300/100/50/miss 与 classic 权重算出 |

**谱面元数据**（`beatmap_meta_{tag}.csv`，17 列）：
`beatmap_id, beatmapset_id, mapper_id, version, star, bpm, max_combo, count_total,
diff_overall, diff_drain, hit_length, total_length, playcount, passcount, approved,
last_update, checksum`。

### 2.2 保留规则

~~~text
4K:      playmode == 3 (mania) 且 diff_size == 4
mod:     canon = mods - {MR, SD, PF};  keep <=> canon == {CL}
         MR/SD/PF 不改变谱面，折叠成"无"；NF 显式拒绝（它会补出本来会被放弃的记录）
         其它 mod（DT/NC/HT/HD/FL/EZ/HR/...）一律拒绝
去重:    modern 的 legacy_score_id 指向 legacy 行；两表重叠部分只留 modern
~~~

1k 另有逐 mod 的漏斗与反事实计数；这里只保留规则，细节见归档的 step0 文档。

### 2.3 规模

| | 1k | 10k |
|---|---|---|
| play 行 | 1,399,615 | 13,359,355 |
| 玩家 | 990 | 9,812 |
| 谱面 | 20,655 | 21,899 |

### 2.4 可选资产：`.osu` 缓存与白名单

CM3P 需要谱面正文，dump 不带正文。`step0/fetch_osu` 把白名单里的
`.osu` **正文**抓到 `data/osu_cache`（无音频/图片）；`step0/whitelist` 生成
`whitelist_{tag}.csv`（`cacheable` 列）。把该表放进 `cache_dir` 或在
`steps.step0.inputs.whitelist` 指定后，step0 会剔除抓不到正文的谱面。

## 3. step1/pairbest（默认，口径 A）

### 3.1 规则顺序

~~~text
0. 玩家门槛：step0 记录数 < 20 的玩家整人剔除
1. 空窗拆分：同一玩家连续 >= 365 天没记录 -> 新玩家段
2. 每对最高：每个 (玩家段, 谱面) 只留 loss 最小的一条
3. 单次对剔除：n_plays < 2 的对整对剔除
4. 高 ACC 剔除：最高 ACC > 0.998 的对整对剔除
5. 低 ACC 剔除：最高 ACC <= 0.90 的对整对剔除
6. 划分：在幸存对上 pair-random 70/30（也可 beatmap-random，或 play-quota）
~~~

清洗在划分**之前**，所以 test 也经过了同样的 pair 过滤；这正是「删除所有单独的
（玩家，谱面）对」的实现。标签支撑因此被截断为
`loss ∈ [log 0.002, log 0.1) ≈ [-6.22, -2.30)`。

### 3.2 输出

`step1_{tag}_pairbest.{parquet,csv,json}`，9 列：
step0 的 5 列 + `split` + `player_id_base`（原玩家 id）+ `player_seg`（空窗段号）
+ `n_plays`（该对在 step0 里的记录数）。`split ∈ {train, test}`。

### 3.3 规模

| | 1k | 10k |
|---|---|---|
| 保留对 = 输出行 | 109,706（step0 行的 7.84%） | 1,513,746（11.33%） |
| train / test | 76,567 / 33,139 | 1,059,890 / 453,856 |

### 3.4 消融开关

* `--max-best-acc 1.0`：不做高 ACC 上限（B 口径）；
* `--label-shift 0.001`：标签换成 `log(1.001-ACC)`（C 口径；同时给 step2 传同样的参数；
  没有可辩护的依据，只作比较）；
* 两者同时打开 = D 口径。

四个口径的 1k/10k 结果见归档文档；对应配置是
`configs/exp_pairbest_1k.yaml` / `exp_pairbest_shift_1k.yaml` 等。

划分变体（不是标签消融）：

* `--split-method beatmap-random`：按**谱面**划分，一张图的所有对必在同一侧；
  配合 `content_eval_heldout` 可做冷启动，配置
  [configs/exp_pairbest_coldstart_cm3p_mlp_1k.yaml](../configs/exp_pairbest_coldstart_cm3p_mlp_1k.yaml)。
* `--split-method play-quota`（step 版本 `step1/pairbest_playquota`）：打乱白名单后
  整张谱面依次进 test，直到 test 的 play 数达到 `1-train_frac`；test 与 train 的谱面
  集合互斥，实际比例略高于 `train_frac`（整图不可拆），`split.actual_test_frac` 是实际值。
  配置 [configs/exp_pairbest_playquota_cm3p_1k.yaml](../configs/exp_pairbest_playquota_cm3p_1k.yaml)。
* **P4（双重冷启动）**：玩家和谱面各分层 60/40，train 只用 6×6、test 只用 4×4；
  off-diagonal 行不训练，只留给 BT ceiling。实现 `scripts/step1_p4_split.py`，
  协议与结果见 [docs/p4_multiseed.md](p4_multiseed.md)。P2 的多种子划分重采样见
  `scripts/step1_pairbest_resplit.py`。

### 3.5 full：三集并集（1k + 10k + randommerge）

`full` 不是新的 dump，而是 1k / 10k / randommerge 三个 step0 play 表的并集。合并发生在
**play 行**上（step 版本 `step0/union`，实现 `scripts/step0_union_datasets.py`）：按
(玩家, 谱面, 秒) 去重，同 key 的 loss 不一致时保留最早来源 / 最低 loss，
`playcount_cur` 在合并后的时间线上重算。之后 step1/pairbest 用与单集完全相同的规则
重建对（seed 20260928），所以 n_plays、空窗拆分和 ACC 过滤都按并集时间线计算。

为什么不在 step1 层直接 concat：1k 的 play 有 **99.99%**（1,399,421/1,399,615）已经在
10k 里，pair 级拼接会重复计数；step1 还会给空窗段分配跨数据集会碰撞的新 player_id；
pair-random 的划分也依赖对集合本身。randommerge 的 4 快照合并用的是同一套 play 级
逻辑，见 [random_merge_report.md](random_merge_report.md)。

| 项 | 值 |
|---|---:|
| 输入 play（三集合计） | 19,295,078 |
| 参与重复 key 的 play 行 | 3,956,457（1,954,668 个 key；删除 2,001,789 行） |
| 其中 loss 冲突 | 11（全部来自 10k 内部的重复行；保留最早来源 / 最低 loss） |
| 合并后 play / 玩家 / 谱面 / 对 | 17,293,289 / 34,444 / 22,247 / 10,428,838 |
| n_plays>=2 的对 | 2,610,107 |
| step1 输出 | 1,898,076 对 / 22,535 玩家段（20,681 基础玩家）/ 20,945 谱面 |
| train / test | 1,329,090 / 568,986（29.98%） |
| n_plays 中位数 / 均值 | 2 / 3.71 |
| 每玩家对数 p10/p50/p90 | 2 / 35 / 259 |
| 内容表覆盖 | 20,832 / 20,945 图（99.46%）；过滤版 1,897,881 行 / 20,832 图 |
| star 中位数 / <=3 星行 / <=4 星行 | 5.25 / 6.31% / 18.22% |

**并集没有制造新的对。** 每个来源里 n_plays=1、并集后 >=2 的对分别是 29 / 699 / 1,266
（1k / 10k / randommerge），但每一个都至少在一个其他来源里已经有 >=2 条 play，所以
“两个来源各贡献一条单次 play 才凑成的对”是 **0 个**。

**来源 step1 对在并集里的覆盖**（按 (基础玩家, 谱面) 匹配）：

| 来源 | 覆盖 | 丢失 | 合并后 n_plays<2 | best ACC>0.998 | 其他 |
|---|---:|---:|---:|---:|---:|
| 1k | 99.9991% | 1 | 0 | 1 | 0 |
| 10k | 99.9922% | 118 | 8 | 110 | 0 |
| randommerge | 99.9625% | 170 | 0 | 170 | 0 |

来源对消失只有两种原因：并集把 best ACC 推过 0.998（简单图上多一条高 ACC play），
或者 **重复 play**——10k 里有 8 对在 step0 中其实是同一次 play 被记录两次（同秒、
`playcount_cur` 不同），step1 的 n_plays=2 来自重复行；按秒去重后只剩 1 条，这对
因此退出。这是合并对源数据的修正，不是丢数据。

去重也作用于**单个来源内部**：1k 有 3 个重复 key（6 行，loss 一致）、10k 有 37 个
（74 行，11 个 loss 不一致）、randommerge 已是去重表（0 个）；跨来源的两两重复 play
的 loss 冲突都是 0。`source_first` / `source_last` 按 **--play 的合并顺序**记录（不是
时间顺序），`n_sources` 才是“被几个来源看到”。

**口径边界**：full 的绝对 NLL / RMSE 不能和 1k、10k、randommerge 的单集数字互比；
来源的 train/test 标签也不复用（并集对新的对集合重新划分）。randommerge 行在 README
里的 453,406 对 / 20,388 图是**内容过滤后**的数字；full 的主表未过滤，CM3P 训练用
`step1_full_pairbest_content.parquet`。

产物（`data/processed/`）：

* `step0_full.parquet`：8 列（5 列 play + `source_first` / `source_last` / `n_sources`）。
* `step0_full_union.json`：合并报告（来源重叠、重复、复活、meta 覆盖）。
* `beatmap_meta_full.csv`：22,419 图，并集谱面 100% 覆盖。
* `step1_full_pairbest.parquet` / `.json`：主表与漏斗（16/16 校验通过）。
* `step1_full_pairbest_content.parquet` / `.json`：按内容表过滤的版本；20,832 图在
  `rank_signals_1k.parquet`（21,949 图）里 100% 覆盖，`msd_overall` / `sunny_star` /
  `ett_068_overall` / `daniel_star` 均无 NaN，可直接跑 `linear_extra` 一类实验。

复现：`python main.py -c configs/exp_pairbest_full.yaml`，需要三个上游 step0 表；
randommerge 的 step0 由 `scripts/step0_merge_snapshots.py` 从 4 个 random 快照合并。
完整数据报告见 [full_union_report.md](full_union_report.md)。

## 4. step1/clean（legacy play 级）

输出仍然是 play 行，只多一列 `split`：

~~~text
0. 玩家门槛：step0 记录数 < 200 的玩家整人标 dropped（先于划分）
1. 划分：pair-random 或 beatmap-random
2. 只对 train 行清洗：
   - ACC <= 0.90 删除
   - 每个玩家按时间删除最早的 floor(drop_frac * n) 条（热身截断）
~~~

test 行**原样保留**，`dropped` 行仍留在表里（可做消融）。必须显式写
`steps.step1.version: clean`。它的 test 是未清洗的 play，不能和 pairbest 的 test 混用。

## 5. 常用命令

~~~bash
python main.py -c configs/exp_pairbest_1k.yaml      # 默认 A + bias/mf_dot
python main.py -c configs/exp_pairbest_10k.yaml    # 默认 A，10k
python main.py -c configs/exp_pairbest_full.yaml      # 1k+10k+randommerge 并集（需 3 个 step0 表）
python main.py -c configs/exp_full_1k.yaml         # raw dump 全流程（legacy clean）
python main.py -c configs/exp_10k.yaml             # 10k 全流程（legacy clean）

# 单独跑预处理
python scripts/step0_preprocess.py --tag 1k \
    --dump-dir data/raw/extracted/2026_09_01_performance_mania_top_1000 \
    --out data/processed/step0_1k.parquet --meta-out data/processed/beatmap_meta_1k.csv
python scripts/step1_pairbest.py --tag 1k \
    --in data/processed/step0_1k.parquet --out data/processed/step1_1k_pairbest.parquet
~~~

可运行的最小实现（去掉 I/O 与校验之后的核心逻辑）在 [preprocess/](../preprocess/)。
