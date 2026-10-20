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
6. 划分：在幸存对上 pair-random 70/30（也可 beatmap-random）
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
