# 低难度数据：诊断、来源与可行性

> 2026-11-19。本文回答三件事：低难度为什么拟合差、去哪里补数据、
> 以及能不能用 osu! API 批量爬玩家 best 成绩。
>
> 结论先行：**低难度数据不是"没有"，是被抽样和清洗规则一起删掉了**；
> `performance_mania_random_10000` 就是现成解药（实测 13 张表）；
> **不要**用 API 批量爬 best——违反使用条款，而且拿到的数据进不了本管线。
>
> **勘误（2026-11-19 晚）**：本文第 1 节的 star 分桶只是**描述性统计**。
> star 是差预测，不能作为分层/加权/评测口径；后续以
> [random_merge_plan.md](random_merge_plan.md) 的 `bm`（每图截距）口径为准。

---

## 1. 现状：低难度样本有多少，怎么被删的

### 1.1 现在的 10k 集（`step1_10k_pairbest`）

| 项 | 值 |
|---|---:|
| 对（行） | 1,513,746 |
| 玩家段 / 谱面 | 10,266 / 16,061 |
| star 中位数 | **5.53** |
| ≤3★ 对 | 20,582（1.36%） |
| ≤4★ 对 | 112,558（7.44%） |

按 star 分桶，问题在**每图样本量**而不只是总量：

| star | 对 | 谱面 | 单玩家谱面占比 | 中位玩家数/图 |
|---|---:|---:|---:|---:|
| ≤2 | 3,755 | 1,354 | — | — |
| 2–3 | 16,827 | 3,293 | — | — |
| ≤3 合计 | 20,582 | 4,647 | **42.4%** | **2** |
| 3–4 | 91,976 | 4,586 | 7.2% | 9 |
| 4–5 | 349,792 | 3,933 | 1.1% | 50 |
| 5–6 | 533,772 | 1,851 | 0.2% | 164 |

≤3★ 的图有 42% 只有 1 个玩家。每图一个 `d_m` 潜变量，1 个观测时只能靠内容
外推——这正是"低难度拟合差"的结构性来源。

### 1.2 清洗规则在做"难度选择性删失"

`max_best_acc = 0.998`：顶级玩家打简单图几乎都是 99.8%+，于是整对被删。
用 `step0_10k.parquet` 复算（每对取最好成绩，n≥2）：

| star | 对 | 因 ACC>0.998 删除 | 占比 |
|---|---:|---:|---:|
| ≤2 | 20,967 | 16,558 | **79.0%** |
| 2–3 | 69,743 | 51,733 | **74.2%** |
| 3–4 | 244,627 | 149,045 | 60.9% |
| 4–5 | 584,768 | 229,894 | 39.3% |
| 5–6 | 663,736 | 124,954 | 18.8% |
| 6–7 | 393,331 | 29,403 | 7.5% |
| 7–8 | 114,260 | 3,608 | 3.2% |
| 8+ | 51,374 | 1,025 | 2.0% |
| 合计 | 2,142,806 | 606,240 | 28.3% |

留下的低星对，是"顶级玩家在简单图上翻车"的那部分——一个**被选择的尾部**，
不是简单图上的正常表现。标签支撑也被截断在 `loss ∈ [-6.22, -2.30)`
（ACC 0.90–0.998）。

### 1.3 指标把问题藏起来了

- test（30%）里 ≤3★ 只有 6,193 对，全局 RMSE / Spearman 由 4–6★ 支配。
- `nohighacc` 消融（10k，mf_dot）：RMSE **0.6844 → 1.0798**，Spearman
  **0.6966 → 0.8350**。把简单样本放回来，序变好、RMSE 变差——因为
  `log(1-ACC)` 在 99.8% 附近不可识别（一条 100 就跨越 0.3 个 log 单位），
  MSE 会把预测拉向均值。
- 外部旁证（VSRG Dan benchmark，`results/benchmark/eval_full/oof.parquet`，
  M1-L isotonic）：≤8 Dan MAE **3.06**（n=114，bias **+2.56**），8–10 是 1.67，
  13.5–14 是 **0.25**。低端不是"略差"，是十倍误差；isotonic 的尾部没有任何锚点。

### 1.4 诊断

三条同时成立：

1. **抽样偏差**：top-10k 玩家不怎么打简单图（≤3★ 仅 1.36% 的对）。
2. **选择性删失**：0.998 上限把简单图上的正常发挥删掉，只留下失败。
3. **标签不可识别**：接近满分时 `log(1-ACC)` 的信息量趋零，RMSE 又把它放大。

所以"补低难度数据"的本质是：**换一批水平更低的玩家**，让他们在低星图上的
中高 ACC 进入标签窗口。

---

## 2. `random_10000`：有用，而且是最省事的一条路

### 2.1 先更正一条旧结论

`legacy/docs/step0_preprocessing.md` §2.1 写的是：random dump「只有
`osu_user_beatmap_playcount` 和 `osu_user_stats_mania` 两张表，没有成绩表，
无法算 loss，已删除」。**这条是错的。**

实测 `2026_09_01_performance_mania_random_10000.tar.bz2`（447,544,515 B）是
**13 张表**的完整 dump：

| 表 | 大小 |
|---|---:|
| `scores.sql` | 563,214,806 |
| `osu_scores_mania_high.sql` | 190,376,279 |
| `osu_user_beatmap_playcount.sql` | 160,772,165 |
| `osu_beatmap_failtimes.sql` | 142,480,259 |
| `osu_beatmap_difficulty.sql` | 1,029,953,623 |
| `osu_beatmap_difficulty_attribs.sql` | 1,176,127,933 |
| `osu_beatmaps.sql` / `osu_beatmapsets.sql` | 56,561,867 / 46,211,131 |
| `osu_user_stats_mania.sql` | 1,827,303 |
| `sample_users.sql` | 271,657 |
| 其余（counts / attribs / blacklist） | < 3 KB |

旧结论大概率来自**一次被截断的解压**：它记的 161 MB + 1.8 MB 正好是
playcount + stats 两张表，总量 ~300 MB 也正好是那次残缺解压的体积。
`2026_11_05` 的 random dump（454,450,707 B）同样是 13 张表。

### 2.2 实测：跑完 step0 + step1 的结果

本机实测（2026-09-01 快照，seed 20260928，规则同默认）：

| 项 | top-10k | **random-10k** |
|---|---:|---:|
| 抽样玩家 | 10,000 | 10,000 |
| 有成绩的玩家 | 9,812 | 6,349 |
| 过玩家门槛（≥20 记录） | 9,368 | **3,194** |
| 对（行） | 1,513,746 | 109,787 |
| 谱面 | 16,061 | **16,130** |
| star 中位数 | 5.53 | **3.78** |
| ACC 中位数 | 0.9853 | **0.9723** |
| ≤3★ 对占比 | 1.36% | **23.4%** |
| ≤3★ 谱面 | 4,647 | **6,335** |
| 因 ACC>0.998 删除（≤2★） | 79.0% | **9.9%** |
| 因 ACC>0.998 删除（2–3★） | 74.2% | **6.1%** |

两个 dump 合并：

| 项 | 值 |
|---|---:|
| 总对 | 1,623,533 |
| ≤3★ 对 | 46,319（2.85%） |
| ≤3★ 谱面 | **7,264**（random 新贡献 2,617，+56%） |
| ≤3★ 每图中位玩家数 | **3**（只用 top 时是 2） |
| ≤3★ 中 ≥5 玩家的图占比 | **34.3%**（只用 top 时是 20.9%） |

一句话：**一个 random 快照贡献的低星对（25,737）比整个 top-10k（20,582）还多**，
且这些对的标签落在窗口内，不是删失尾部。

### 2.3 代价

- 下载 447 MB（压缩）/ ~2.5 GB（解压；只解 6 张表约 0.9 GB）。
- step0 + step1 在本机各 ~1–2 分钟。
- 单个快照只有 ~11 万对；要上百万对，叠多个快照或与 top-10k 合并。

---

## 3. 其他来源

| 来源 | 能给什么 | 限制 |
|---|---|---|
| `2026_07_13 / 08_01 / 09_01 / 11_05` 的 random_10000 | 每份 ~3.2k 玩家 / 11 万对，低星占 23% | 快照不同，玩家可能重叠，需去重 |
| 同快照的 `osu_beatmap_failtimes` | 全玩家的失败位置分布，**不依赖抽样** | 只是谱面侧信号，可作低星 `d_m` 的正则/辅助目标 |
| 1k dump 的 `osu_beatmap_difficulty*` | 全谱面逐 mod 难度（全局表，按 beatmap_id join 即可） | 与 top/random 快照同日，可直接复用 |
| 玩家客户端 `client.realm` / `scores.db` | **失败局、低 ACC、非通过**——唯一能看到删失前的数据 | 志愿者规模，工具见 `legacy/docs/player_plays.md` |
| 管线改造 | 0.998 改成删失似然（Tobit）而不是删除；低星分层采样；按 star 分桶评测 | 不改数据也能缓解 1.3 的指标问题 |

---

## 4. 能不能批量爬玩家的 "B200"（best 200）？

**结论：不要用它建数据集。**

### 4.1 政策

osu! API 文档明确把下面的用法列为禁止：

> Harvesting mass score/user/beatmap data (consider using data.ppy.sh instead ...)

并要求：

> Please limit your usage to no more than 60 requests per minute (generally 1 request per second).

10,000 玩家 × 2 页（`/users/{id}/scores/best`, limit=100）= 20,000 请求，
按 1 rps 是 5.5 小时"理论下限"，但已经属于条款禁止的 mass harvesting，
token 可能被吊销。

### 4.2 数据本身也不对

- `best` 是**通过局**、每个 (玩家, 谱面) 只有 1 条 → `min_plays_per_pair=2`
  会把所有对删光（step0 的 pairbest 口径要求同一对至少 2 次记录）。
- 没有失败局、没有低 ACC 尾部；对顶级玩家，best 200 反而是他们**最难**的图 →
  低难度覆盖比 dump 更差。
- 对低排名玩家，best 200 是他们的天花板（约 3–4★），确实落在低难度区，但要先能
  批量枚举低排名玩家——API 的 rankings 只给排行榜，且同样受 60 rpm 限制。

**同样一次下载能拿到的东西严格更多**：random dump 直接给完整 `scores` +
`osu_scores_mania_high` + 谱面侧表，还是官方允许的批量数据通道。

如果只是补某个具体玩家，用 `legacy/docs/player_plays.md` 的 scores.db /
client.realm / recent 路线，单人取数，不触发批量条款。

---

## 5. 复现命令

~~~bash
cd /home/mikechu/Documents/Projects/DCR
cd data/raw
curl -LO https://data.ppy.sh/2026_09_01_performance_mania_random_10000.tar.bz2
mkdir -p extracted_random
tar -xjf 2026_09_01_performance_mania_random_10000.tar.bz2 -C extracted_random \
  --wildcards '*/osu_beatmaps.sql' '*/scores.sql' '*/osu_scores_mania_high.sql' \
  '*/osu_user_stats_mania.sql' '*/sample_users.sql'
cd ../..

python3 scripts/step0_preprocess.py \
  --dump-dir data/raw/extracted_random/2026_09_01_performance_mania_random_10000 \
  --tag random --out data/processed/step0_random.parquet \
  --meta-out data/processed/beatmap_meta_random.csv
python3 scripts/step1_pairbest.py --tag random \
  --in data/processed/step0_random.parquet \
  --out data/processed/step1_random_pairbest.parquet
~~~

产物：`data/processed/step1_random_pairbest.parquet`（109,787 行）、
`data/processed/step1_random_pairbest.json`（漏斗）、
`data/processed/step0_random.parquet`（1,110,238 行 play）。

## 6. 下一步

1. 把 random 与 top-10k 合并（注意同一玩家跨快照去重），重跑 step2。
2. 评测改成 **按 star 分桶** 报 RMSE / Spearman / captured，别再只看全局 RMSE。
3. 低星样本按删失处理（0.998 以上不进 loss 而是进 censoring indicator），
   或在低星段单独训练/加权。
4. `legacy/docs/step0_preprocessing.md` §2.1 的 random 结论需要按本文更正。
