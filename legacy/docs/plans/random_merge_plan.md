# Random 快照合并计划（step0 级重建全部对）

> 2026-11-19。目标：把多个 performance_mania_random_10000 快照合并成**一个
> play 级数据集**，再用旧管线（step1/pairbest → step2/predict）重建全部
> (玩家段, 谱面) 对。先量数据量，再决定加权。**不使用 star 作为分层口径。**

---

## 0. 结论式摘要

1. **合并在 step0（play 行）做，不在 step1（对）做。** 对是由 play 重新算出来的：
   n_plays、playcount_cur、空窗分段都必须在合并后的时间线上重算。
   在 pair 级合并会把「快照内单次、跨快照两次」的对永远丢掉。
   （实测这个收益是 0——抽样几乎不放回，见 §2.5；step0 级合并仍因口径统一而必要。）
2. 现有 4 个 mania random 快照：**2026-07-13 / 08-01 / 09-01 / 11-05**，各 1 万玩家。
3. 去重键：modern 用全局 score id；legacy 用自然键（user, beatmap, date, 判定, mods）。
   跨快照重复率必须实测（§3.2），不能假设。
4. 合并后的输出保持 step1_*_pairbest.parquet 的 **9 列 schema 不变**，旧管线直接吃。
5. 数据量报告已出（§2.5）；**评测口径固定为 P2 + P4（§7），默认不加权（§6）。**

---

## 1. 数据源

| snapshot | 文件 | 大小 (B) | 状态 |
|---|---|---:|---|
| 2026-07-13 | performance_mania_random_10000.tar.bz2 | 473,361,585 | 下载中 |
| 2026-08-01 | performance_mania_random_10000.tar.bz2 | 452,993,533 | 下载中 |
| 2026-09-01 | performance_mania_random_10000.tar.bz2 | 447,544,515 | 已有 |
| 2026-11-05 | performance_mania_random_10000.tar.bz2 | 454,450,707 | 下载中 |

每个都是 13 张表的完整 dump；只解压管线需要的 5 张：

    osu_beatmaps.sql  scores.sql  osu_scores_mania_high.sql
    osu_user_stats_mania.sql  sample_users.sql

sample_users.sql 只用于核对抽样与玩家重叠，不进模型。

---

## 2. 为什么必须在 step0 合并

step1/pairbest 的对定义是 (player_seg, beatmap_id)，输入三样东西都由**全量 play 行**决定：

| 字段 | 现在怎么来 | 跨快照合并后 |
|---|---|---|
| n_plays | 该对在 step0 的记录数 | 必须合并后重数（决定 min_plays≥2） |
| playcount_cur | 该 play 在 (player, beatmap) 内的 1-based 序号 | 必须按合并后时间线重排 |
| player_seg | 365 天空窗切段 | 必须看合并后的时间线 |
| best ACC | 对内 min loss | 必须在合并后取 |

典型收益：某玩家 7 月打一次、9 月再打一次，两个快照各自只有 1 条 → 两次都被
min_plays_per_pair=2 整对删掉；合并后 n_plays=2，这一对复活。
这正是「重建所有的对」的实际含义，也是 §5 要量的指标。

---

## 2.5 实测（4 个快照，2026-11-19 已完成）

先合并两个快照做探针，结果改变了对「重建」的预期；4 个快照跑完后结论不变：

| 项 | 值 |
|---|---:|
| 4 快照 play / 玩家 / 谱面 | 4,536,108 / 25,105 / 22,200 |
| 合并后对（player, beatmap） | 2,999,619 |
| 两两玩家交集 | 53–72（Jaccard **0.0043–0.0056**） |
| 出现在 1 / 2 / 3 个快照的玩家 | 24,744 / 356 / 5 |
| 跨快照重复 play | 126,596 条（2.8%） |
| **复活对（单快照都是 1 次、合并后 ≥2）** | **0** |
| step1 输出 | **453,601 对 / 12,759 玩家 / 20,501 谱面** |

**结论：random 抽样几乎不放回——4 次 10k 抽样两两只重 50–70 个人。**
所以合并的收益是**更多玩家 / 更多图**，而不是「把单次对复活成两次对」：
复活对是 0，重复 play 只有 2.8%（且集中在 356 个双快照玩家上）。
`step0` 级合并仍然必要（口径统一、playcount/空窗重算），但不要指望它把
单次对救回来。真正的增量在规模（4× 单快照）和覆盖（20,501 张图）上。

## 3. 管线设计

### 3.1 流程

    每个 snapshot:
      step0_preprocess.py --dump-dir ... --tag rand_<snap>
            -> step0_rand_<snap>.parquet  (+ beatmap_meta_<snap>.csv)
            -> 加 snapshot / source 两列

    合并:
      union by (player_id, beatmap_id, timestamp, loss, score_key)
      dedupe: modern 按全局 score id；legacy 按自然键
      重算 playcount_cur（按 timestamp, score_key 排序）
      -> step0_randommerge.parquet

    step1:
      step1_pairbest.py --tag randommerge --in step0_randommerge.parquet \
          --out step1_randommerge_pairbest.parquet
      -> 与旧管线同 schema 的 9 列 + 同 seed 20260928

### 3.2 去重与一致性（必须先做探针）

1. **modern scores**：id 是全局自增，同一 play 在不同快照里应同 id。
   探针：取 09-01 与 11-05 的共同玩家，统计 id 交集 / 并集，应接近 1.0。
2. **legacy osu_scores_mania_high**：score_id 可能被 osu-web 重建，不能当主键。
   用自然键 (user_id, beatmap_id, date, count300, count100, count50, countmiss,
   countgeki, countkatu, enabled_mods, maxcombo, score)；同一秒完全相同的两局
   几乎不可能。
3. **dump 内 modern↔legacy**：沿用现有 legacy_score_id 去重，再跨快照去重。
4. **探针不通过**（例如 id 漂移 >1%）：改用自然键做全集去重，并在报告里写明。

### 3.3 口径冻结（与旧管线一致）

- 4K mania（playmode=3 & diff_size=4）；mods 归一到 CL，NF 拒绝；
- classic ACC 公式与 loss=log(1-ACC) 不变；
- 玩家门槛 ≥20 条、空窗 365 天、n_plays≥2、ACC ∈ (0.90, 0.998]；
- pair-random 70/30，seed 20260928；
- 输出列与 step1_{tag}_pairbest.parquet 完全一致（9 列），step2/CM3P/rank-eval 不改代码。

### 3.4 谱面元数据

- beatmap_meta 按 beatmap_id 取并集；可变字段（star/checksum/last_update/approved）
  **以最新快照为准**，另存 meta_snapshot 列；
- 只在晚快照出现的谱面（新图）自然进入；只在早快照出现的图保留；
- star 仅作元数据保留，**不作为分层/评测轴**（见 §5.3）。

### 3.5 目录与命名

    data/raw/extracted_random/<snapshot>/          # 已解压
    data/processed/step0_rand_20260713.parquet
    data/processed/step0_rand_20260801.parquet
    data/processed/step0_rand_20260901.parquet
    data/processed/step0_rand_20261105.parquet
    data/processed/step0_randommerge.parquet       # 合并后（+ snapshot_first/last）
    data/processed/step1_randommerge_pairbest.parquet
    data/processed/step1_randommerge_pairbest.json # 漏斗

---

## 4. 执行步骤

| # | 步骤 | 产物 | 预计 |
|---|---|---|---|
| P0 | 冻结本计划；确认 4 快照清单 | 本文 | 已完成 |
| P1 | 下载 3 个快照（8 连接 + bzip2 -t 校验）并解压 5 表 | data/raw/... | **完成**（~25 min） |
| P2 | 每快照跑 step0，加 snapshot/source 列 | 4 × step0_rand_* | **完成**（每个 ~45 s） |
| P3 | 一致性探针（§3.2）+ 玩家重叠统计 | step0_randommerge.json | **完成** |
| P4 | 合并 + 去重 + 重算 playcount_cur | step0_randommerge.parquet | **完成** |
| P5 | 跑 step1/pairbest（旧规则、旧 seed） | step1_randommerge_* | **完成**（13 s） |
| P6 | 出「数据量报告」§5；**决策门** | docs/random_merge_report.md | **完成** |
| P7 | P2 / P4 两口径 × 5 种子评测（§7） | [randommerge_p2p4_results.md](randommerge_p2p4_results.md) | **完成（线性档）**；MLP / audio 待补 |

P1 起由后台任务执行，失败可续传（脚本保留 .parts）。

---

## 5. 决策门：合并后先量什么（不用 star）

### 5.1 规模

- 玩家（base / segments）、play 行、对、谱面；
- 每玩家 play 数、每玩家对数的分布（p10/p50/p90）；
- 4 个快照的玩家重叠矩阵（两两 Jaccard、出现在 k 个快照的玩家数）。

### 5.2 「重建」到底复活了多少对

这是本计划的核心指标：

| 指标 | 定义 |
|---|---|
| 单快照单次对 | 在所有单快照里 n_plays=1 的对 |
| **复活对** | 单快照里都是 1 次，合并后 n_plays≥2 |
| 新增对 | 合并后存在、任何单快照的 pairbest 里都没有 |
| 掉出对 | 单快照里在窗口内、合并后 best ACC 出窗口 |
| 窗口移动对数 | best ACC 因合并跨过 0.90 / 0.998 的对 |

### 5.3 难度覆盖（替代 star）

用**数据侧难度**而不是 star：

- 在合并 train 上拟合 bias 模型，取每图截距 bm（含每图自由截距的难度潜变量）；
- 按 bm 十分位报告：对计数、谱面数、每图玩家数、残差 RMSE；
- 需要外部对照时用内容信号（minacalc / sunny_star / benchmark signals），
  它们是谱面侧、与玩家抽样独立；
- star 只作为附录里的「劣质参考」，不参与任何切分或加权决策。

### 5.4 标签质量

- ACC/loss 分布；0.998 上限处堆积的对数（删失量）；
- 0.90 下限处堆积；
- 合并后 ACC 中位数 vs 单快照（预期向窗口中部移动）。

---

## 6. 加权：默认不加权（只列触发条件）

### 6.0 先问估计量：默认不该加权

加权改变的是**估计量**，不是数据量。先看实际的不平衡有多大（合并集，按 bm 把谱面
五等分；bm 越低越简单）：

| 谱面五分位 | 谱面 | 对 | 对/图 | n≥2 的对 | 因 ACC>0.998 删 | 因 ACC≤0.90 删 |
|---|---:|---:|---:|---:|---:|---:|
| Q1 最简单 | 3,951 | 45,936 | 11.6 | 67,617 | 10.9% | 13.9% |
| Q2 | 3,952 | 96,802 | 24.5 | 126,202 | 8.4% | 9.8% |
| Q3 | 3,952 | 109,676 | 27.8 | 135,545 | 9.0% | 6.9% |
| Q4 | 3,952 | 110,827 | 28.0 | 131,413 | 8.4% | 4.9% |
| Q5 最难 | 3,952 | 89,366 | 22.6 | 99,686 | 5.6% | 3.7% |

（注：n≥2 一列是 ACC 窗口**之前**的对，所以比「对」列大；删失比例以它为分母。）

结论：**难度段的总条数确实不均，但只差约 2.4×（46k vs 111k），不是 20×。**
Q1 的问题是**每图支持低**（11.6 对/图，中位 3 个玩家）和**两端删失都重**
（0.998 删 10.9%，0.90 删 13.9%）；这两件事加权都解决不了，只能补数据或改删失处理。

因此默认口径应当是：

- 若目标是「对一个随机抽到的 (玩家, 谱面) 对做预测」——**均匀权重（W0）就是对的**，
  加权反而把估计量改成别的总体；
- 真正需要加权只有三种情况：① 部署/评测目标分布与采样分布不同；② 要输出**每张谱面
  等权**的难度评级；③ 某些玩家被系统性过采样；
- 现在数据里真正的不平衡是玩家/谱面**集中度**（下节），它只影响方差（ESS），
  不影响无偏性。

更该先做的是 **split**：现在 pair-random 下，**98.66% 的 test 对，其玩家和谱面都在
train 里**（玩家 99.37%、谱面 99.27%）。这比加权更能改变结论——先换 player-disjoint
或 chart-disjoint（repo 已有 `scripts/player_disjoint_val.py`、P2/P4 设计）。

| 方案 | 定义 | 解决什么 | 风险 |
|---|---|---|---|
| W0 不加权 | 每对等权 | 基线 | 重叠玩家/热门图被放大 |
| W1 每玩家封顶 | 每玩家段最多 K 对 | 活跃玩家支配 | 丢长尾 |
| W2 每谱面封顶 | 每图最多 K 对 | 热门图支配 | 低难度图仍稀疏 |
| W3 支持度加权 | w = 1/sqrt(n_players_chart) | 让稀疏图有梯度 | 噪声放大 |
| W4 快照分层 | 按 snapshot_first 分层/加权 | 快照重复 | 无明显收益时不做 |

### 6.1 实测后的判断依据（2026-11-19）

合并集 453,601 对 / 12,759 玩家 / 20,501 图：

| 集中度 | 值 |
|---|---:|
| 前 1% 玩家贡献的对 | 13.4% |
| 前 10% 玩家贡献的对 | 56.3% |
| 前 1% 谱面贡献的对 | 15.3% |
| 前 10% 谱面贡献的对 | 52.0% |
| 只有 1 个玩家的谱面 | 9.2%（占对 3.2%） |
| ≥50 玩家的谱面 | 10.1% |

有效样本量（ESS）：

| 口径 | ESS |
|---|---:|
| W0 均匀 | 453,601 |
| 1/sqrt(每玩家对数) | 255,579 |
| 1/sqrt(每图支持) | 292,384 |
| 两者同时 | 138,639 |

**建议：默认 W0，不加权。** 只有在选定了「每玩家等权」或「每图等权」的估计量、
或部署分布与采样分布确实不同时，才按上表加权重：玩家侧用 1/sqrt(对数) 或封顶 100，
谱面侧用 1/sqrt(支持) 连续降权；W2 硬封顶不做（会把 9.2% 的单玩家图清零）。
评测口径已固定为 P2 / P4（§7），其中 P4 就是 player+chart 双重冷启动；
加权只在上面的三种情况成立时才引入，且必须与口径一起报告。

---

## 7. 评测口径：P2 / P4（都切在 pair_best 上）

两个口径都作用在 `step1_randommerge_pairbest.parquet`（幸存对表）上：清洗
（玩家门槛 / 空窗 / pair-best / ACC 窗口）已经完成，split 是最后一步
（[docs/data.md](data.md) §3.1），所以两种切法都不改变清洗口径。**默认不加权。**

已生成（2026-11-19）：

    data/processed/step1_randommerge_p2_s{1..5}.parquet  + _p2_summary.json
    data/processed/step1_randommerge_p4_s{1..5}.parquet  + _p4_summary.json

### 7.1 P2：pair-random 重采样（新 chart 口径）

- 5 个划分种子（1–5），pair-random 70/30，重切幸存对，不重跑清洗。
- 实测：train 317,251–318,076 / test 135,525–136,350（约 70/30）。
- 内容模型额外做 5 折 chart-disjoint 打分（`--chart-folds 5 --chart-seed 0`），
  每张图由没见过它的折打分——这是 P2 的「新 chart」部分。
- **bias / mf_dot 只有 P2 的 pair-random 部分适用**：它们有自由的 per-chart 截距，
  chart-disjoint 部分只对内容模型（CM3P）有意义。

### 7.2 P4：双重冷启动（新玩家 + 新 chart）

- 同一张 pair_best 上：玩家按活跃度（pair 行数）分层 60/40，谱面按流行度
  （distinct 玩家数）分层 60/40。
- train = 6×6（train players × train charts），test = 4×4；off-diagonal（6×4 / 4×6）
  标为 `dropped`，只留给 BT ceiling，模型从不训练。
- 实测 5 个种子：玩家（`player_id_base` 口径，共 11,763）7,058 / 4,705，谱面（共 20,501）12,301 / 8,200；
  train 161,507–166,403 行，test 70,838–73,847 行，dropped 216,360–218,307 行。
- 代价：只有约 16% 的行进 test、48% 被丢弃，test 更小、ceiling 更噪；
  这是双重冷启动的定义，不是实现差异。
- **P4 只适合内容模型**：bias / mf_dot 对没见过的谱面没有 `bm`，无法在 P4 上评测；
  P4 一律用 CM3P 内容模型（± external 信号）。

### 7.3 共同约定

- 模型初始化 / 训练种子固定 0；5 个种子只切分 → mean±std 反映**划分不确定性**。
- 指标：loss 轴 RMSE / MAE / Spearman + rank 轴 NLL / captured；按 `bm` 五分位报，
  **不用 star**。
- ceiling 按 [docs/p4_multiseed.md](p4_multiseed.md) §4 的 cross-fit BT 口径。
- 报告必须写清 P2 还是 P4：两者 test 集、ceiling、pair 数都不同，**绝对 NLL 不可互比**。

### 7.4 命令

    # P2：5 个 pair-random 划分种子
    python3 scripts/step1_pairbest_resplit.py \
        --in data/processed/step1_randommerge_pairbest.parquet \
        --out-prefix data/processed/step1_randommerge_p2 --seeds 1,2,3,4,5

    # P4：5 个玩家×谱面双重分层种子
    python3 scripts/step1_p4_split.py \
        --in data/processed/step1_randommerge_pairbest.parquet \
        --out-prefix data/processed/step1_randommerge_p4 --seeds 1,2,3,4,5

## 8. 风险与回退

| 风险 | 处理 |
|---|---|
| 并行下载拼包损坏 | fetch_dump_parallel.sh 先校验总大小 + bzip2 -t；失败保留分片续传 |
| 跨快照 id 漂移 | 探针 §3.2；不行就全自然键去重 |
| legacy 表被重建导致 id 变化 | 同上，自然键兜底 |
| 谱面元数据漂移 | 最新快照为准 + meta_snapshot 列 |
| 磁盘（当前 12 G 可用，91%） | 4 档 ≈ 1.8 G + 解压 4×0.9 G；合并验证后删 archive |
| 合并后口径与旧结果不可比 | 保留 random（09-01 单快照）作对照，报告两套漏斗 |

---

## 9. 待定问题（需要拍板）

1. **是否把 top-10k 也并进来**：本计划只做 random；top-10k 作为独立分层，
   后续可选 union（高难度端靠它）。默认**不并**，避免改变现有对照。
2. **是否合并 top-10k 的 4 个快照**：同理，本期不做。
3. playcount_cur 的语义：沿用「1-based 序号」，合并后重排；下游若把它当特征需确认。
4. 加权：**已定，默认 W0 不加权**；只有 §6.0 的三种情况成立才引入，且与 P2/P4 口径一起报告。
5. 训练预算：P2/P4 各 5 个划分种子，是否全部跑 5 个还是先跑 1 个（seed 1）出基线。
