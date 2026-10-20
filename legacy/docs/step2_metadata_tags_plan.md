# step2 × beatmap metadata + user tags 接入方案

**目标.** 给 step2 增加一条**谱面侧完全由内容定义**的预测路径：把
（a）`beatmap_meta_{tag}.csv` 的基础信息（时长 / 物量 / BPM / OD / HP / star / 流行度…）
与（b）已爬好的 **osu! user tags**（玩家投票的风格 / 技能标签 + 票数）
拼成谱面特征向量 `x_m`，用一层**可学线性投影** `C_m = W^T x_m` 取代（或补充）
`mf_dot/mirt` 里那张自由的 per-chart 表 `C`。

**一句话.** 这不是新模型族，而是把既有的 `--chart-content` 机制从"冻结向量（CM3P）"
推广到"**冻结特征 + 可学投影**"；`diag_step2_fm.py` 已经用 7 个 metadata 特征验证过这条路
（§2.3），本方案是把特征扩到 `metadata + tags + 派生量` 并把**冷启动**补上。

**状态.** 方案。**尚未实现**。本文所有现状数字都已核对到文件（2026-10-14）。

---

## 0. TL;DR

1. **机制已经存在，缺的是特征构造器。** step2 已有 `--chart-content {none,cm3p}`：
   把 `C` 换成冻结矩阵（`register_buffer`，不进 Adam），`P` 与它同维，交互仍是点积
   （`scripts/step2_predict.py:231-237, 299, 319-323`）。metadata/tags 需要的只是
   `C_m = W^T x_m` 这一步——**`W` 可学，所以它正好等价于 `diag_step2_fm.py` 的 FM 形式**
   （`⟨P_u, W^T x_m⟩ = Σ_f ⟨P_u, W[:,f]⟩ x_mf`）。
   **外部调研后又往前推了一步**：主模型取**分层收缩** `C_m = W^T x_m + s_m E_m`（§4.5），
   纯内容与纯自由是它的两个极限——文献一致认为不该二选一。
2. **user tags 已经爬好了，但没有任何模型用过它。**
   `data/processed/chart_user_tags_1k.parquet`（19,158 行 / 6,438 张谱面 / 88 个 tag），
   配套 `chart_user_tag_vocab_1k.csv`、`chart_user_tag_coverage_1k.csv`。全仓库 grep 只有
   `step0f_fetch_tags.py` 自己引用它们。
3. **两个免费但值钱的派生特征已经验证**：
   `max_combo − count_total` **恒 ≥ 0**（min 0，8.4% 恰为 0）= **长条数量**，
   于是 `ln_ratio = max_combo/count_total − 1` 是不用解 `.osu` 就能拿到的 **LN 密度**；
   它和 `LN` 系 tag 对得上（§3.2）。另一条是 `nps = count_total/hit_length`（密度），
   和 `streams/speed` tag 对得上。
4. **真正的独家能力是冷启动，而它现在跑不了。** `item_ids` 只从 **train** 谱面建
   （`:625`），`chart_feats` 也对齐到 train 谱面（`:636-639`）；`--split-method beatmap-random`
   下测试谱面 100% 不在词表里 → 覆盖率≈0。要测冷启动**必须改代码**（§5.3），
   这正是 `cm3p_step2_results.md` §5 列的未做项 CS1。
5. **必须先重跑 step1。** `data/processed/step1_1k.parquet` 还是 2026-10-03 的旧格式
   （964,417 行、**没有 `split` 列**），而现在的 step2 只按 `split` 列取数。
6. **评测铁律不变**：metadata 覆盖 100% → 可与主表直接比；tags 只覆盖
   **6,438 / 21,949 = 29.3%** → **必须在同一子集上重跑 1-hot 与 metadata-only 对照**，
   跨覆盖率比 RMSE 没有意义（`cm3p_step2_results.md` §3.2 的既有结论）。
7. **需要拍板的 12 件事**见 §10，最重要的是 **D1 要不要用 `star`**（用了 step3 就失去动机）
   与 **D9 主模型取纯内容还是分层收缩**（外部文献推荐后者）。
8. **外部文献（§13）改变了三处选型**：主模型改为分层收缩、加 DropoutNet 式 dropout、
   判据从"只看 RMSE"改成"同时读 `σ²` 与 tagged/untagged 分层"。

---

## 1. 这次要做什么：与现有四条路径的关系

| 路径 | 谱面侧表示 | per-chart 自由参数 | 冷启动 | 状态 |
|---|---|---|---|---|
| `--chart-content none`（主表） | 自由表 `C`（`mirt` 另有 `D`） | `K+1`（`mirt`：`2K+1`） | ✘ | 已跑（mf_dot 1.4548 / mirt 1.4582） |
| `--chart-content cm3p` | 冻结 CM3P 向量（PCA 白化到 `d`） | 1（只剩 `b_m`） | 机制支持，**未测** | 已跑（覆盖子集 d32 1.57595） |
| **本方案 `--chart-content feat`** | **`W^T x_m`（metadata+tags+派生）** | **1（只剩 `b_m`）；`--drop-bias m` 后 0** | **✔（需 `--drop-bias m`+`--chart-vocab all`）** | **待实现** |
| `diag_step2_fm.py`（诊断） | `Σ_f ⟨P_u,W[:,f]⟩x_mf` | 1 | ✘（没走主脚本） | 已跑（1.5028） |

**与 CM3P 的关系：互补，不是替代。**
`chart_content_cm3p.parquet` 只覆盖 11,877 / 21,949（**54.11%**），
`chart_content_cm3p_local.parquet` 覆盖 21,949（**100%**）但 `cm3p_mix_report.json` 判定
**DO NOT MIX**（无音频编码是另一种表示，`recall@1 = 0.285`）。
→ metadata/tags 是**免费、100% 覆盖、可解释**的一条独立路线；
"metadata+tags vs CM3P-local" 与 "metadata+tags **+** CM3P-local" 两个问题值得各测一次。

---

## 2. 现状盘点（全部已核对）

### 2.1 metadata：21,949 行 × 17 列，覆盖 100%

`data/processed/beatmap_meta_1k.csv`（与 `beatmap_meta_10k.csv` **md5 相同**，10k dump 无该表）：

`beatmap_id, beatmapset_id, mapper_id, version, star, bpm, max_combo, count_total,
diff_overall(OD), diff_drain(HP), hit_length, total_length, playcount, passcount,
approved, last_update, checksum`

实测分布：

| 列 | 中位 | 范围 | 备注 |
|---|---|---|---|
| `hit_length` | 125 s | 2 – 5,029 s | **时长**；max ≈ 83.8 min |
| `total_length` | 128 s | 25 – 5,092 s | `total − hit` = 前奏/尾奏 |
| `count_total` | 1,098 | 2 – 63,907 | 物量 |
| `bpm` | 175 | 31.75 – 666 | |
| `max_combo` | 1,514 | 2 – 87,813 | `max_combo ≥ count_total` **恒成立** |
| `star` | 3.14 | 0.054 – 10.37 | 2026-09-01 快照 |
| `diff_overall` / `diff_drain` | 7.5 / 7.8 | 0 – 10 | OD / HP |
| `playcount` / `passcount` | 17,497 / 5,989 | 8 – 1.28e7 / 3 – 2.19e6 | 快照，时间穿越 |
| `mapper_id` | — | 1,399 个不同值 | 类别特征 |
| `approved` | — | 1: 19,187 / 4: 2,703 / 3: 59 | ranked / loved / qualified |
| `version` | — | 2 个缺失 | 难度名，**混乱**（`Normal/Hard/Easy/Insane/…` + `HD/NM/MX` + 自定义）→ v1 不用 |

### 2.2 user tags：已爬好，未被使用

| 文件 | 规模 | 内容 |
|---|---|---|
| `chart_user_tags_1k.parquet` | 19,158 行 | `beatmap_id, tag_id, tag_name, count`（长表） |
| `chart_user_tag_vocab_1k.csv` | 106 行 | `tag_id, name, ruleset_id, description` |
| `chart_user_tag_coverage_1k.csv` | 21,949 行 | `beatmap_id, beatmapset_id, set_fetched, beatmap_present, n_tags, n_votes` |

- 有 ≥1 tag 的谱面 **6,438 / 21,949 = 29.3%**（但**行层面覆盖 58%**，见 §5.1——
  有 tag 的多是热门图）；tag_id 出现 88 个；
  每张有 tag 的谱面平均 **2.98** 个 tag；谱面×tag 满矩阵的非零率 **0.82%**（极稀疏）。
- **两条缺失来源必须分开**（这是特征工程的关键，§3.3）：
  1. `set_fetched = 0`：**我们从没抓到过这张图**。21,949 中 **4,318 张（19.7%）**，
     来自 **1,473 个 beatmapset**；
  2. `set_fetched = 1, n_tags = 0`：抓到了，**确实还没有人投票**（新图/冷门图）。
     在抓到的 17,631 张里占 **63.5%**。
- **爬取可续**：断点缓存还在（`data/interim/step0f_1k/pages.jsonl` 5,513 行 +
  `missing_sets.txt` 1,473 行）。按"已抓到的图里 36.5% 有 tag"同比例外推，
  续爬完可把有 tag 的谱面从 **6,438 → 约 8,000**（+24%），**这是最便宜的一步增量**。
- **流行度混淆已实测**：`n_votes` 与 `playcount` Spearman **0.307**、与 `star` Spearman **0.382**。
  tag 票数同时编码了"这张图有名"和"这张图难" → 票数不能裸用（§3.3、§6）。
- tag 语义是**真的**（不是噪声）：见 §3.2 的两组验证。

### 2.3 已有的直接证据：内容锚定 > 自由 1-hot（但要注意口径）

`scripts/diag_step2_fm.py` + `docs/step2_capacity_dim_sweep.md` §5，1k、同一 split、同一优化器、
7 个 metadata 特征（`star, bpm, max_combo, count_total, diff_overall, diff_drain, hit_length`）、
y 标准化后反算回 loss 单位：

| 配置 | 参数量 | 训练 RMSE | 测试 RMSE |
|---|---|---|---|
| 自由低秩 K=1 | 20,623 | 0.4517 | 1.5501 |
| 自由低秩 K=4 | 82,492 | 0.3894 | 1.5614 |
| 自由低秩 K=16 | 329,968 | 0.3153 | 1.6957 |
| **FM dim=4** | **3,976** | 0.4594 | **1.5028** |
| **FM dim=16** | **15,904** | 0.4561 | **1.5028** |
| FM dim=16（**去掉 star**） | 15,888 | 0.4624 | 1.5132 |

三条读数：

1. FM 用 1/20 的参数比该诊断里最好的自由低秩（K=1，1.5501）低 **3.1%**；
2. **`dim=4` 与 `dim=16` 完全相同（1.5028）** → 7 个特征已经张满谱面空间，
   "想再提升必须**加特征**，不是加 dim"——这正是本方案加 tags 的理由；
3. **去掉 `star` 只退化 0.69%**（1.5028 → 1.5132）：`max_combo / count_total / hit_length`
   已经承担了大部分工作 → §10 的 D1 有底气把 `star` 默认关掉。

> ⚠️ **口径警告（必须写进文档）**：上表的"自由低秩"最好只有 1.5501，而**主表**
> `mf_dot/mirt` 是 1.4548/1.4582 —— 两者**训练配方不同**（diag 用 ALS 初始化、
> 分组 lr 0.5/0.03、cosine 120 epoch、无 wd；主脚本用 Adam lr 0.02 + wd 1e-4 + 60 epoch）。
> 所以**不能**说"FM 比主表的 mirt 好"；能说的是"在 diag 这一套配方里，参数化换成内容锚定更好"。
> 正式实现必须落回主脚本，与主表在**同一配方**下对照。

### 2.4 操作事实：缓存里的 step1 是旧格式

| 文件 | 行数 | 列 |
|---|---|---|
| `data/processed/step0_1k.csv` | 1,399,615 | `player_id, beatmap_id, timestamp, playcount_cur, loss` |
| `data/processed/step1_1k.parquet` | **964,417** | 同上 —— **没有 `split` 列**（2026-10-03 产物） |

新 `step1/clean` 的输出 = **step0 的全部行 + `split` 列**（`train`/`test`/`dropped`）
（`scripts/step1_clean.py:17-32`）。所以**任何本方案的实验都要先跑 step1**：

```bash
python scripts/step1_clean.py --tag 1k          # 或走 main.py -c configs/exp_std_1k.yaml
```

---

## 3. 特征工程规格

统一记一张表：`chart_features_{tag}_{spec}.parquet`，键 `beatmap_id`，
列 `beatmap_id, f000, f001, …` + 一份 `chart_features_{tag}_{spec}.json` schema
（特征名、块归属、标准化统计、tag 词表、**train 谱面 id 集合的 sha256**）。

**统计量一律只在 train 谱面上拟合**（沿用 `align_content` 的 "PCA basis fitted on TRAINING
charts only" 规则，`scripts/step2_predict.py:455-470`），否则是同一种泄漏。

### 3.1 块 M：数值 metadata（覆盖 100%）

| 特征 | 变换 | 理由 |
|---|---|---|
| `log(hit_length)` | log | **时长**（用户点名） |
| `total_length − hit_length` | 原值，标准化 | 前奏/尾奏 |
| `log1p(count_total)` | log1p | 物量 |
| `nps = count_total / hit_length` | log | **密度**（已验证与 `streams/speed` 对齐） |
| `ln_ratio = max_combo/count_total − 1` | 原值 | **长条比例**（§3.2 新发现） |
| `bpm` | 原值 | |
| `diff_overall`, `diff_drain` | 原值 | OD / HP |
| `star` | 原值 | **默认排除**，见 §10 D1 |
| `log1p(playcount)`, `log1p(passcount)`, `passcount/playcount` | log/log/原值 | 流行度、通过率 → **默认关**，见 §10 D2 |
| `approved` one-hot (1/3/4) | | ranked / loved / qualified |
| `mapper_id` | 嵌入或 target-encode | **默认关**，见 §10 D3 |

**默认块 M ≈ 9–10 维**（不含 star / 流行度 / mapper）。

### 3.2 派生特征：免费但值钱（new，已实测）

**(a) 长条数量 = `max_combo − count_total`。** osu!mania 里长条头尾各给 1 combo，
所以 `max_combo = count_total + n_holds`。实测在全部 21,949 张图上：

```
max_combo - count_total:  min = 0     frac < 0 = 0.0000     frac == 0 = 0.0839     mean = 508.2
```

**恒不为负**，与公式一致 → `ln_ratio = n_holds / count_total` 是可直接用的 **LN 密度**。
`docs/step2_capacity_dim_sweep.md` §5.1 说"更丰富的特征（密度 / chord / jack / **长条** / 手数）
需解 `.osu`"——**长条这一项不用解，它就在 metadata 里。**

**与 crowd tag 对齐验证：**

| 分组 | n | `ln_ratio` 均值 | 中位 |
|---|---|---|---|
| 有 LN 系 tag（`style/LN density` / `LN coordination` / `LN release` / `LN mixed` / `gimmick/LN inverse` / `style/longjack`） | 1,699 | **0.642** | **0.474** |
| 无 | 20,250 | 0.485 | 0.346 |

**(b) 密度 `nps` 与 `streams/speed` tag 对齐验证：**

| 分组 | n | `nps` 中位 |
|---|---|---|
| 有 `streams/speed` tag | 868 | **12.87** |
| 无 | 21,081 | 8.81 |

→ 结论：**派生量的方向与人群标签一致**，说明 (i) 派生特征有语义，
(ii) 后面用 tag 去解释学到的 `D` 维是可行的（§7 v2）。

### 3.3 块 T：user tags（覆盖 29.3%）

**词表**：只从 **train 谱面**建；只保留在 train 谱面上出现 ≥ `--tag-min-charts`（默认 20）张的
tag。88 个 tag_id 里预计留 40–60 列（长尾 `tag_*` 单票标签会被丢掉，这是想要的）。

**取值**（默认 + 消融）：

| 方案 | 公式 | 优点 | 缺点 |
|---|---|---|---|
| **二进制（默认）** | `1[count ≥ 1]` | 抗票数长尾；与"有没有这个属性"语义一致 | 丢强度 |
| `log1p(count)` | 标准化 | 保留强度 | **票数 ∝ 流行度**（与 playcount Spearman 0.307）→ 引入混淆 |
| 份额 | `count / n_votes` | 去掉总量 | 单票图上极不稳（25% 的有 tag 图只有 1 个 tag） |

**两条缺失指示（必须分开，默认都开）**：

1. `is_unfetched = 1[set_fetched = 0]`（19.7%）：**我们没数据**；
2. `is_no_votes = 1[set_fetched = 1 ∧ n_tags = 0]`（占 63.5%）：**数据说没有**。

tag 列本身**零填充，不做均值插补**——均值插补会把"平均风格"注入一张可能极端的图。

**可选标量** `log1p(n_votes)`：默认**不加**（它就是流行度的另一种写法）。

**外部文献补充的四条做法（§13 Q4，建议全部采纳）：**

1. **tag 块要单独一组惩罚，不要与稠密数值特征共用一个 `wd`。**
   `W` 拆成 `[W_M | W_T]` 两组，`W_T` 用自己的惩罚倍率（group penalty / 稀疏组 lasso 的思路）。
   理由：58 个二进制列与 10 个标准化连续列的**有效尺度差一个量级**，共用一个 `wd`
   等于给 tag 块施加了完全不同的有效收缩。
2. **IDF 式降权近普遍 tag。** 像 `streams/stamina` 这种在 1,325 张图上出现的标签，
   信息量远低于只出现 119 次的 `expression/high contrast`。可对每列乘 `log(N/n_j)` 再标准化。
3. **对流行度残差化。** 票数部分是"注意力代理"（与 `playcount` Spearman 0.307）。
   做法：先把 tag 列对 `log1p(playcount)` 回归、取残差再入模。
   **默认建议只做 (1)，(2)(3) 放消融**——避免 v1 一次改太多东西。
4. **缺失不是干扰项，是一等分层。** 必须**分别报告 tagged / untagged 谱面上的指标**：
   tag 特征只能影响 29% 的谱面（58% 的行），而两个缺失指示会**整体平移**另外那部分行，
   **一个总量上的增益完全可能掩盖另外 71% 谱面上的损失**。
   → 这条直接进 §5.2 的 B 组与 §6 的清单。

### 3.4 一份默认特征表长什么样

**已按本节规格原型化（in-memory，未落盘）实测**：

```
块 M（12）：log_hit_length, total_minus_hit, log1p_count, log_nps, ln_ratio, bpm,
            od, hp, approved_q, approved_loved, is_unfetched, is_no_votes
块 T（58）：tag_<id>（二进制）   ← 阈值"train 谱面上 ≥20 张"从 88 个 tag_id 里筛出 58 个
默认 F = 70           --star on 时 +1；--popularity on 时 +3；--mapper on 时 +8

实测：21,949 行 × 70 列，NaN 0、inf 0；
      tag 块密度 1.46%（58 列全矩阵），6,246 张谱面至少有一个保留的 tag
      （6,438 - 6,246 = 192 张只有长尾单票 tag，被阈值丢掉 —— 这是想要的）
      is_unfetched = 19.67%，is_no_votes = 51.00%
      M 块连续列在 train 谱面上 mean 0 / sd 1；两个缺失指示保持 0/1 原值不做标准化
```

**参数账**：`F=70, K=16` → `W` 只有 **1,120** 个参数，而自由 `C` 表是 `19,640 × 16 = 314,240`。
**per-chart 自由参数从 314k 降到 19,640（只剩 `b_m`），冷启动模式下再降到 0。**

### 3.5 预实验（已实测）：tags 里有多少是 metadata 解释不掉的

**这是本文支持"tags 值得试"的最强一条证据。** 做法：对每个有 ≥100 张谱面的 tag，
用 **10 个 metadata 特征**（`star, bpm, log_hit_length, log1p_count_total, log_nps, ln_ratio,
diff_overall, diff_drain, log1p_playcount, pass_rate`，标准化）做 **5-fold CV 逻辑回归**，看 AUC。
AUC 高 = tag 与 metadata 冗余；AUC 低 = tag 带正交信息。36 个 tag 的实测结果：

```
AUC: min 0.700   median 0.800   max 0.893        AUC < 0.9 的个数: 36 / 36
```

| 最正交（AUC 低） | n | AUC | | 最冗余（AUC 高） | n | AUC |
|---|---|---|---|---|---|---|
| `additions/keysounds` | 263 | 0.700 | | `streams/stamina` | 1,325 | 0.893 |
| `expression/high contrast` | 119 | 0.742 | | `style/handstream` | 370 | 0.882 |
| `expression/simple` | 671 | 0.742 | | `style/dump` | 385 | 0.865 |
| `style/LN coordination` | 792 | 0.752 | | `skillset/wristjack` | 400 | 0.863 |
| `skillset/reading` | 541 | 0.767 | | `style/chordstream` | 378 | 0.856 |

**读法（三条）：**

1. **metadata 只能解释 tag 的一部分**：即使最"物理"的 tag（`streams/stamina`）也只到 0.893；
   而 `expression/simple`（"设计直白、好上手"）这种**手感类** tag 只有 0.742 ——
   它说的是"这张图打起来是什么感觉"，而 `hit_length` / `bpm` / `nps` / `ln_ratio` 说不出来。
2. **"tags 可能有用"的必要条件已满足**：如果 metadata 能以 0.97 AUC 认出 tag，
   那加 tag 基本等于加噪声。现在 **36/36 都 < 0.9** → 存在正交信息。
3. **但这不是充分条件**：AUC 说的是"能否从 metadata 认出 tag"，
   **不是**"tag 能否降低 loss 预测误差"。真正的判据仍是 §5.2 的 **B3 vs B2**。
   这条预实验的作用是**把"大概率白干"的先验改掉** —— 值得认真跑，而不是顺手一测。

> 复现：§11 之前的 python 片段（标准化 + `LogisticRegression` + `cross_val_predict`，
> `StratifiedKFold(5, shuffle=True, random_state=0)`）。该预实验**未写入任何文件**。

---

## 4. 模型规格

### 4.1 式子：内容锚定 = "冻结特征 + 可学线性投影"（≡ FM）

```
free（现状）   ℓ̂ = b0 + b_u + b_m + ⟨P_u, C_m⟩              C ∈ R^{n_item × K}  自由
本方案         C_m = W^T x_m,  W ∈ R^{F × K}                x_m 冻结、W 可学
               ℓ̂ = b0 + b_u + b_m + ⟨P_u, W^T x_m⟩
             = b0 + b_u + b_m + Σ_f ⟨P_u, W[:,f]⟩ · x_mf     ← diag_step2_fm.py 的 FM 形式
冷启动         ℓ̂ = b0 + b_u + ⟨P_u, W^T x_m⟩                b_m 也去掉 → per-chart 自由参数 = 0
```

**关键恒等式**：FM 的 `Σ_f ⟨P_u, W[:,f]⟩ x_mf` **就是**"把谱面特征线性投影到 `P` 的空间再做点积"。
所以不需要新模型族，只需要让 `C` 由冻结特征经一个可学矩阵生成。

**参数量与谱面数脱钩**：`K·F`（K=16, F=73 → 1,168）对 `n_item·K`（20k×16 = 320k）。
per-chart 只剩 `b_m`（20k 个），冷启动模式下去掉。

**规范（gauge）**：`W` 可学 + `P` 自由 → 只有 `P W^T` 可辨识（`GL(K)` 旋转 + 尺度），
与现状同构。**但 span 被钉死在 `x` 的 F 维里**——这正是 `dim=4 == dim=16` 的原因，
也正是"加特征才有用"的原因。对 step3 的意义：给出去的谱面向量是 `C_m = W^T x_m`，
**对任意有 `x` 的谱面都有定义**，且有 schema 可复现。

### 4.2 与 `--chart-content cm3p` 的机制关系：代码增量很小

现在的代码（`scripts/step2_predict.py`）：

```python
# :231-237   建参数
if content == "none":
    self.p["C"] = nn.Parameter(draw((n_item, dim)))
else:
    self.register_buffer("C_feat", chart_feats.to(device=device, dtype=torch.float32))
# :256-269   只把用到的参数交给 Adam
if content == "none": self.active += ["C"]
# :299       前向
C = self.C_feat[m] if self.content != "none" else p["C"][m]
# :319-323   导出（给 step3）
def chart_matrix(self): return self.C_feat... if self.content != "none" else self.np_array("C")
```

**要改的就是把"冻结向量"与"冻结特征"分开**：

```python
# 新增：feature 模式（content == "feat"）
self.register_buffer("X_feat", chart_feats)          # (n_item, F) 冻结
self.p["W"] = nn.Parameter(draw((F, dim), std=0.01)) # 可学 → 进 Adam、吃 --wd
self.active += ["W"]
# 前向
if self.content == "none":    C = p["C"][m]
elif self.content == "cm3p":  C = self.C_feat[m]
else:                         C = self.X_feat[m] @ p["W"]     # (B, F) @ (F, K)
# 导出
def chart_matrix(self): return (self.X_feat @ self.p["W"]).detach() if self.content == "feat" else ...
```

`W` **必须进 `active`**（要 wd 正则化 tag 系数），但**不要**给它单独调 lr/wd——
除非 §5.2 的 B5 消融显示需要（参考仓库已有教训：`--d-logit-wd` 的对称点陷阱）。


**实现地雷（照抄 CM3P 路径会挂的地方，逐条核对过）：**

| 位置 | 现状 | feature 模式必须改 |
|---|---|---|
| `:448` `load_content_table` | 硬编码 `columns=["beatmap_id", "embedding"]`，把每个元素 `np.stack` 成矩阵 | 特征表是**扁平的 `f000…f0NN` 列**：要么让新表沿用 `embedding` 列名（一行特征塞进一个 list），要么加 `--content-format {vector,table}` 分支 |
| `:467-470` + `:638` `--content-reduce pca` | 默认 PCA 白化到 `--content-dim`，并**把 `--dim` 钉成 content dim** | feature 模式必须 `--content-reduce none`，且**不要**改 `--dim`（`K` 是自由超参，`F` 是特征数，两者无关） |
| `:692-700` check `content_aligned_with_item_ids` | 断言 `chart_feats` 每行 **L2 范数 ≈ 1**（白化的产物） | 特征模式没有这个性质 → 断言改成"行数 == `len(item_ids)` 且 id 探针全部解析成功" |
| `:455-466` `align_content` | 训练谱面缺一行就 `raise SystemExit` | **保留**这个 fail-fast（特征表覆盖 100%，缺行一定是 bug） |
| `:794` variant 命名 | `_cm3p{dim}` | 新增 `_feat{spec}`（如 `_featMT16`），避免覆盖已有产物 |

### 4.3 冷启动模式

```python
# 需要 --drop-bias m（复用已有开关）与 --chart-vocab all（新增，见 §5.3）
ℓ̂ = b0 + b_u + ⟨P_u, W^T x_m⟩          # 零 per-chart 自由参数
```

这是**内容表示唯一不可替代的能力**，也是本方案相对于"再加一个 1-hot 变体"的全部意义。
注意 `b_u` 仍是自由的：本方案的冷启动是 **cold-chart / warm-player**，
cold-player 不在范围内（要另做玩家侧内容，见 §10 D7）。

### 4.4 `mirt` 版本（v2）：把 `D` 也内容锚定

`mirt` 的 `D ∈ R^{n_item × K}` 仍是自由的，所以 **`mirt` 默认不能冷启动**。v2：

```
D_m = softplus(V^T x_m)   或   softmax(V^T x_m)      # V ∈ R^{F × K}
```

语义上这是本方案最漂亮的一步：`D` 是"这张图考哪几维"，而 **tags 正好就是这件事的人群标注**
（`skillset/streams`、`style/chordjack`、`streams/stamina`…）。
于是 v2 直接给出一个可证伪的分析：**学到的 `D` 维与 tag 是否对齐**（相关矩阵 / 每维 top tag）。
若对齐，说明 `D` 有语义、可作为 step3 的表示；若不对齐，说明现有 `D` 只是一个自由旋转的基。

> **外部调研给这一节加了一条更硬的动机**（§13.3）：osu!mania 的难度是**真多维**的
> （stamina / jack / stream / chordjack / technical），社区技能系统全部按 skillset 建模。
> 因此**单一标量 `b_m` 本身就是设定错误**——把 `D` 内容锚定不只是"省参数"，
> 而是让模型能表达"这张图考哪几种技能"，并且**tags 恰好就是这个 skillset 标注**。
> 这条把 §4.4 从"看情况做"提升为"结构性需要"。

### 4.5 分层收缩：**外部文献推荐的主模型**（原 v3 → 提升为主模型）

> **这一节因为外部调研（§13）从"可选 v3"提升为主线。** 文献的一致结论是：**不要二选一**。
> 内容/侧特征模型在稀疏与冷启动区制占优，自由 ID 嵌入在交互充足时占优，
> 因此推荐的形式是**两者之间的收缩谱**，而不是"纯内容 vs 纯自由"的等容量对拍。
> 这条恰好也是 `docs/step2_capacity_dim_sweep.md` §7.3/§8-3 独立得出的收口。

**两种等价写法（文献里各有名字）：**

```
# (a) 贝叶斯先验式（CTR / HFT / Bayesian MF with side info 的写法）
C_m ~ N( W^T x_m , σ_m² )         σ_m² 可学、随样本量收缩
# (b) 残差式（等价的重参数化，更好实现）
C_m = W^T x_m + s_m · E_m         s_m = sqrt(n_m / (n_m + λ))
D_m = softplus(V^T x_m + s_m · F_m)    # mirt 的 D 同理
```

`n_m` = 该谱面在训练侧的 play 数：play 多的图自己学（`s→1`，退回现状的自由 `C`），
稀疏图退回内容先验（`s→0`，退回 §4.1 的纯内容模型）。**两个极限正好就是 A 组与 B 组的两个端点。**

**为什么这是主模型而不是加分项**：本项目的稀疏度是"**处处接近冷启动**"——
pair 中位只有 1 条 play、谱面中位只有 7 个玩家。小样本 IRT 的标准结论是
**自由 per-item 参数在这个样本量下不可估**，层次先验收缩是公认补救
（[10.1177/0146621619893786](https://doi.org/10.1177/0146621619893786)）。
换句话说，**现在的 `mirt`/`mf_dot` 就在做一件统计上不该做的事**。

**σ²（或 λ）本身就是一个更好的实验读数**：与其只看 RMSE 的 A/B，不如直接看**学到的收缩强度**——
它回答"侧特征到底替代了多少自由参数"，并给出 per-chart 的"内容解释度"。
建议 summary 里输出 `σ²`/`λ` 的估计值，以及按 `n_m` 分箱的 `‖s_m·E_m‖ / ‖C_m‖` 曲线。

**实现注意（仓库已有的坑）**：不能靠"给 `E` 一个更大的 wd"来近似收缩——
Adam 的 wd 是每步固定幅度的拉回，**不是按样本量收缩的**
（仓库已经因为同类误解踩过 `--d-logit-wd` 的对称点坑，`step2_prediction.md` §4.4b）。必须显式写 `s_m`。

### 4.6 DropoutNet 式 dropout：让**一个**模型同时覆盖热启动与冷启动

问题：纯内容路径在热启动数据上会被自由残差"悄悄忽略"（内容分支拿不到梯度压力）。
文献的成熟解法（DropoutNet, NeurIPS 2017）是**训练时随机丢弃输入分支**：

```
--chart-dropout p        # 以概率 p 把该样本的内容项整体置零（只剩自由 C）
--content-dropout p      # 以概率 p 把自由残差置零（只剩 W^T x_m）—— 这条才是关键
```

这样**同一个 checkpoint** 在推理时有两种用法，不需要重训：

| 推理方式 | 等价于 | 对应实验 |
|---|---|---|
| 全量（内容 + 残差） | §4.5 主模型 | A2 / B3 |
| 只保留 `W^T x_m` | §4.1 纯内容 | A4 / A5 / C1 / C2 |
| 只保留自由 `C` | 现状 `mf_dot`/`mirt` | A1 |

**这是本方案对"冷启动要不要单独训一个模型"的回答**：不需要。
只要 `--content-dropout > 0`，内容分支就一定是被训练过的。
（若不做 dropout，冷启动必须**另跑**一次纯内容配置，否则 C 组结果可能是
"内容分支没训好"的假象而非真实能力。）

---

### 4.7 换个名字：这套模型在心理测量学里有 50 年历史（Explanatory IRT / LLTM）

外部调研指出：**"把 item 难度回归到 item 协变量上"不是新想法，而是有名字的标准模型**——
Linear Logistic Test Model（Fischer 1973），现代框架叫 **Explanatory IRT**：

```
item 难度 = 截距 + f(协变量) + item 残差项        ← 在 GLMM 里估
```

三条必须遵守的结论（文献口径，直接约束本方案）：

1. **必须保留截距**（或另行固定潜变量的度量）——否则不可辨识。
   本方案的 `b_m`（或 `W` 的常数列）就是这个截距。
2. **必须保留 item 残差**，除非你愿意**主张"协变量完全解释了难度"**。
   这个主张是**可检验的、且经常被拒绝**（这正是 random-weights LLTM 存在的原因）。
   → 直接支持 §4.5 把分层收缩当主模型，而不是纯内容模型。
3. **正则化要层次化/带惩罚，不能用平坦 L2**——与 §3.3 的组惩罚、§4.5 的显式收缩一致。

**最有价值的引用**：[Vie & Kashima, Knowledge Tracing Machines (AAAI 2019)](https://doi.org/10.1609/aaai.v33i01.3301750)
证明了 **FM 在数学上涵盖 additive factor model / PFA / 多维 IRT**，
且在"学生数据稀疏"时仍能准确估计能力并同时使用侧信息——
**这正是本项目要做的事，且已被证明可行。** 建议在文档/答辩里直接引用它作为方法依据。

**要坦白的两处不匹配**（文献类比弱的地方，写进结果里）：
- 教育测量里的 item 通常有**上百次作答**，而本项目谱面中位只有 **7 个玩家**——比层次 2PL 已验证的区制还要稀疏一个量级；
- IRT 假设**二值/多分类**响应，而 `loss = log(1−ACC)` 是连续量，**在心理测量学里没有先例**，
  所以"区分度"与"ACC 方差"会混在一起。

---

## 5. 评测协议

### 5.1 三个 universe，与"同子集对照"铁律

**已实测**（跑 `scripts/step1_clean.py --tag 1k` 到临时路径，2026-10-14：
train 674,890 / test 420,401 / dropped 304,324，与 `docs/step2_prediction.md` §7 口径提示里
"新口径比旧口径多 454 行"完全一致，说明管线自洽）：

| universe | 谱面 | train 行 | test 行 | 占 train | 占 test | 能比什么 |
|---|---|---|---|---|---|---|
| **U_all** | 21,949（metadata 100%） | 674,890 | 420,401 | 100% | 100% | 与**主表**（bias/mf_dot/mirt）直接比 |
| **U_tag** | 6,438（≥1 tag） | **387,923** | **244,257** | **57.5%** | **58.1%** | **只能**与"同一子集上重跑的 1-hot / metadata-only"比 |
| **U_fetched** | 17,631（`set_fetched=1`） | 585,279 | 369,911 | 86.7% | 88.0% | 分离"没抓到"与"没人投票"两种缺失 |

> **好消息**：谱面层面 tag 只覆盖 29.3%，但**行层面覆盖 58%**（有 tag 的图往往是热门图，
> play 更多）。所以 B 组（tags 的 A/B）有 **24.4 万条测试行**，统计上完全站得住，
> 不需要靠"续爬"来救。续爬（M0）仍然是便宜的加分项（U_tag 的谱面数 +24%），
> 但**不是 B 组可行性的前提**。

**铁律**（沿用 `cm3p_step2_results.md` §3.2）：**过滤必须在 split 之前**，
且任何子集上的 A/B 都必须包含该子集自己的 1-hot 基线。
实现上直接复用 `--restrict-to-content on`（`scripts/step2_predict.py:530-541`
已经把过滤放在 split 之前，并在 `:632-634` 打印"这是同子集的 1-hot 基线"）。

### 5.2 消融阶梯

**A 组 —— U_all（metadata，全部谱面；与主表可比）**

| # | 配置 | 回答 |
|---|---|---|
| A0 | `bias` | 下限 |
| A1 | `mf_dot` dim16（1-hot，主表复现） | 参照 **1.4548** |
| A2 | `mf_dot --chart-content feat` dim16 | metadata 锚定能否追平 1-hot？ |
| A3 | A2 `--drop-bias m` | 冷启动模式的代价 |
| A4 | A2 `--dim {4, 8, 16, 32}` | **最优 dim 会不会从 4 抬起来**（§9 的判据） |
| A5 | A2 `--feat-include star` / 不含 star | 时间穿越值多少（diag 说 0.69%） |
| A6 | A2 vs `--chart-content cm3p`（54% 覆盖，需同子集） | metadata vs 内容向量谁更值 |

**B 组 —— U_tag（同子集，同 split，行在 split 前已裁）**

| # | 配置 | 回答 |
|---|---|---|
| B0 | `bias` | 子集下限 |
| B1 | `mf_dot` 1-hot（`--restrict-to-content on`） | **同子集 1-hot 对照** |
| B2 | `mf_dot --chart-content feat`（不含 tag 块） | 同子集 metadata-only |
| B3 | `mf_dot --chart-content feat`（含 tag 块） | **主问题：tags 值多少** |
| B4 | B3 去掉两条缺失指示 | 缺失编码值不值 |
| B5 | B3 `--tag-encoding logcount` | 编码消融（票数 vs 有无） |
| B6 | B3 + `--dim {4,16,32}` | tags 是否抬高了最优 dim |
| B7 | B3 去掉 star / 流行度 | 泄漏敏感性 |

**C 组 —— 冷启动（step1 `--split-method beatmap-random`）**

**已实测规模**（`python scripts/step1_clean.py --tag 1k --split-method beatmap-random`，
写到临时路径）：**train 14,086 张 / test 6,385 张谱面，交集 = 0**（谱面级完全留出）；
train 656,234 行 / **test 447,399 行**；其中 **254,385 行（56.9%）落在有 tag 的谱面上**
→ C2（tags 对冷启动的价值）同样有足够样本。

| # | 配置 | 回答 |
|---|---|---|
| C0 | `bias --drop-bias m`（只有 b0+b_u） | 冷启动下限（**唯一能跑的 1-hot 对照**） |
| C1 | `mf_dot --chart-content feat --drop-bias m` | **metadata 冷启动 RMSE** |
| C2 | C1 + tag 块 | **tags 在冷启动上值不值**（理论上这里收益最大） |

**D 组 —— 时间敏感性（可后置）**：把测试行限制到 `timestamp < 2025-01-01` 重算 A2/B3 与各自对照，
看**差值**是否变化。tag / star / playcount 都是 2026 年快照，这是唯一能直接量化"时间穿越"的做法。

### 5.3 冷启动协议需要代码改动（不是加个开关）

现状的阻塞点，逐行：

| 位置 | 现状 | 冷启动需要 |
|---|---|---|
| `:624-627` | `item_ids = np.unique(train.beatmap_id)`；`items = {m:i}` | 内容模式下 `item_ids = np.unique(concat(train, test))` |
| `:636-639` | `align_content(content_ids, content_E, item_ids, …)` → `chart_feats` 只有 train 谱面行 | 特征表对齐到 **union**；学习型 per-chart 参数仍只给 train 谱面 |
| `:655-672` | `map_ids(df.beatmap_id, item_ids)` → 测试谱面得 `-1` → `te_ok=False` → 算作"未覆盖" | 内容模式下测试谱面可打分；覆盖率重新定义为"**玩家在 train 词表 ∧ 谱面有特征**" |
| `:683-686` | `chk.check("vocab_from_train_only", item_ids == unique(train))` | **契约变更**：内容模式下改成"学习参数的行 ⊆ train 谱面"，并在 summary 里显式标注 |

新增开关：`--chart-vocab {train, all}`（默认 `train` = 现状，`all` = 冷启动）。
**`pair-random` 下两者结果应逐位相同**（测试谱面本来就 99.6% 在 train 词表里），
这本身就是一条新校验：`pair_random_unchanged_by_chart_vocab`。

---

## 6. 泄漏与效度清单（新增校验项，进 `Checklist`）

| # | 断言 | 为什么 |
|---|---|---|
| N1 | `chart_feats_standardized_on_train_only`：标准化统计 / tag 词表 / 缺失率全部只由 train 谱面算出 | 与 `align_content` 的 PCA-on-train 同一条规则 |
| N2 | `tag_vocab_from_train_charts_only`：词表 == train 谱面里出现过的 tag 集合（阈值后） | 与既有 `vocab_from_train_only` 同构 |
| N3 | `no_test_chart_leak_in_features` 仅 `--chart-vocab all` 时放宽并**显式标注** | 冷启动就是要用测试谱面自己的 metadata（这不算泄漏，但要写清楚） |
| N4 | `pair_random_unchanged_by_chart_vocab`：`--chart-vocab all` 在 pair-random 下指标与 `train` 逐位相同 | 证明新代码路径没改变既有口径 |
| N5 | 报告 `star`/`playcount`/`passcount`/`n_votes` 是否入模 | 时间穿越可追溯 |

**必须写进结果的六条口径说明：**

1. `star` / `playcount` / `passcount` 是 **2026-09-01 快照**，贴在 **2013 年起**的游玩上；
2. tags 是 **2026-10 爬取**的累计投票，**投票发生在游玩之后**——这是标准的 time-travel；
   它是 `docs/step2_capacity_dim_sweep.md` §5-3 说的"star 的时间穿越效应可忽略（0.69%）"
   的同类问题，但**tag 从未被测过**，必须先量（D 组）。
   ⚠️ **文献里最强的补救手段我们用不了**：外部调研建议"只用游玩发生前的时间戳化投票事件"，
   但 `step0f` 从谱面页拿到的 `top_tag_ids` 是 `{tag_id, count}` **聚合计数，没有逐票时间戳**
   （osu-web 的 `beatmap_tags` 表本身也只有 `(beatmap_id, tag_id, user_id)`）。
   → **时间穿越无法消除，只能被界定**：剩下的手段是"禁运期"（embargo）与**前向时间评估**（D 组）。
   这条限制必须写进结果，不能含糊过去。
3. **`star` 是一个"版本化产物"**，不是固定属性：官方难度算法每次更新都会重算全部谱面的 star。
   所以一张图**在游玩发生的那个时刻并不存在现在这个 star 值**。
   → 这是 D1（默认不放 star）的又一条理由，比"时间穿越 0.69%"更本质。
4. **tag 的系数没有因果解释。** tags 是**同一批人**在游玩之后投的，是 post-treatment 变量；
   把系数读成"这种风格更难"会引入 post-treatment / collider 偏误。
   **用于预测是合法的，用于因果结论不合法**——写进结果口径。
5. `n_votes` 与流行度混淆（0.307 / 0.382）→ 默认不用票数强度、只用有无；
6. `--chart-vocab all` 下模型能看到测试谱面的 **metadata**（不包括任何 play）——
   这是"用内容外推"的定义，但**不能**说成"谱面级留出下完全无信息"。

---

## 7. 实现计划（文件级）

| # | 文件 | 改动 | 估计 |
|---|---|---|---|
| **M0** | 无（跑 `scripts/step0f_fetch_tags.py`） | 续爬 1,473 个缺失 set（断点缓存已在），刷新 tags 三件套 | 0.5 h |
| **M1** | `scripts/step1_clean.py`（跑，不改） | 生成带 `split` 列的 `step1_1k` | 0.2 h |
| **M2** | **新** `scripts/step0h_chart_features.py`（~200 行） | metadata + tags + 派生量 → `chart_features_{tag}_{spec}.parquet` + `json` schema；统计量只用 train 谱面；规格由 CLI 控制（`--blocks M,T --star on/off --tag-encoding binary --tag-min-charts 20`） | 0.5 d |
| **M3** | `dcr/steps.py` | 照抄 `step0/chart_content`（`:283-291`）注册 `step0/chart_features` | 0.2 h |
| **M4** | `scripts/step2_predict.py` | ① `--chart-content` 增 `feat`；② `W` 参数 + `active` + 前向 `X_feat[m] @ W`；③ `chart_matrix()` 导出 `W^T x`；④ `--chart-vocab {train,all}` + 覆盖率重定义（§5.3 四处）；⑤ npz/summary 增加 `W` / `feat_names` / `feat_schema_hash`；⑥ 新增 N1–N4 四条 check | 1 d |
| **M5** | `configs/exp_meta_1k.yaml`、`exp_meta_tags_1k.yaml`、`exp_meta_cold_1k.yaml` | A/B/C 三组 | 0.2 d |
| **M6** | 跑 A/B/C 阶梯 + `docs/step2_metadata_tags_results.md` + 更新 README 表格 | | 1 d |
| **M7** ⭐ | **分层收缩**（§4.5，文献推荐的主模型）：`C_m = W^T x_m + s_m E_m`，输出 `σ²/λ` 与收缩曲线 | | 1 d |
| **M8** | **DropoutNet 式 dropout**（§4.6）：`--content-dropout`，让一个 checkpoint 覆盖热/冷 | | 0.3 d |
| **M9**（v2） | `--d-content {none,linear}`：`D_m = softplus(V^T x_m)`，并出 `D`×tag 对齐分析 | | 1 d |
| **M10** | tag 块分组惩罚 + IDF 降权 + 对流行度残差化（§3.3 的四条） | | 0.5 d |

**总量**：`M0–M6`（可跑出 A/B/C 三组完整对照）≈ **3 人日**；
加上 `M7+M8`（文献推荐的主模型形态）≈ **4.3 人日**；再加 `M9+M10` ≈ **5.8 人日**。
真正的新代码 ≈ **350–400 行**，其余是接线与跑批。

> **排期建议**：不要跳过 `M7`。按 §13 的文献结论，**纯内容 vs 纯自由的对拍本身不是推荐模型**；
> 先做 `M7` 的收缩模型，A/B 两组的"端点"由它一个模型的两个极限给出，同时还能读出 `σ²` 这个更有信息量的量。

---

## 8. 与现有 step2 契约的兼容性

- **不新增 step**:先扩 `--chart-content`，跑通后再考虑提升为独立版本
  （`step2/predict_feat`）（理由：`--restrict-to-content`、10 条 check、
  npz/summary 导出、`--p-drift`、F1/F2 都能直接复用，重写脚本要复制约 1,200 行）。
- **主表口径零变化**：`--chart-content none`（默认）路径完全不动。
- **vc 产物名**：`step2_1k_mf_dot_F0_feat{W}^T…` 需要新增 variant 后缀（`:794` 附近），
  命名建议 `_feat{spec}`（如 `_featM` / `_featMT`）。
- **step3 接口**：`chart_matrix()` 返回 `W^T x`，npz 里同时存 `W`、`feat_names`、
  `feat_schema_hash`、`train_chart_ids` —— 这是 step2 第一次能给 step3 一个
  **对任意新谱面可复现**的谱面表示。

---

## 9. 预期与可证伪的判据

诚实的先验。**两条方向相反的既有证据**必须同时摆出来：

- **偏悲观**：`docs/step2_capacity_dim_sweep.md` §5 说"信号低维、7 个特征已张满谱面空间、
  `dim4 == dim16`"——暗示再加特征可能没用；
- **偏乐观**：本文 §3.5 的预实验说 **metadata 识别不出任何一个 tag（36/36 AUC < 0.9，
  中位 0.80）**——暗示 tags 带正交信息，**有东西可加**。

两者不矛盾：前者说的是"已有特征张满了**它们能表达的那部分**空间"，
后者说的是"tags 指向的空间比那部分大"。所以下表的分歧点就在 tags 上：

| 预期 | 判据（不满足就是假设错了） |
|---|---|
| metadata-only 在 U_all 上**接近但略差于** 1-hot 1.4548 | 差 > 5% → 特征块选错了（比如该保留 star/流行度） |
| tags 值 **0.3–1.5%**（上界受 §3.5 的乐观证据支持，下界受"信号低维"支持） | < 0.1% → tags 在 **loss 预测**上与 metadata 冗余，结论应是"不值"；**注意这与 §3.5 不矛盾**（tag 的可识别性 ≠ tag 的预测力），不要为了自洽去改判据 |
| **tags 会把最优 dim 从 4 抬起来** | 若加 tag 后 `dim4 == dim16` 依旧 → tags 只是重新参数化了已有维度，**没有新信息**。这是最干净的可证伪判据 |
| 冷启动（C 组）RMSE 明显差于热启动 | 但**必须显著好于 C0**，否则内容表示没有价值 |
| tags 在**冷启动**上的相对增益 > 在热启动上的 | tags 的主要价值应体现在新图上（那里没有 `b_m` 可依赖） |
| **分层收缩（M7）应当同时不差于两个端点**（纯内容 / 纯自由） | 若不优于**任何**端点 → 侧特征与自由参数**没有互补性**，这比"tags 没用"更强的否定 |
| **加 tag 后 `σ²`/`λ` 的估计值应当下降**（自由残差被内容替代得更彻底） | 若 `σ²` 不变或上升 → tag **没有替代任何自由参数**，即没有真正的信息（§4.5 的读数比 RMSE 灵敏） |
| **tagged / untagged 分层报告的差值应当稳定** | 若总量涨、untagged 子集明显跌 → 增益来自缺失指示的**水平位移**而非 tag 语义，必须改判为"不值" |

**不要做的事**：不要用"覆盖率更高"来解释 RMSE 变化（必须先对齐子集）；
不要因为 FM 在 diag 里 1.5028 就宣布它优于主表 mirt 1.4582（配方不同，§2.3）。

---

## 10. 未决问题（需要拍板）

| # | 问题 | 我的建议 |
|---|---|---|
| **D1** | **要不要把 `star` 放进特征？** | **默认不放**。放了 step2 就吃掉了 step3 的回归目标，"从行为学难度"的叙事失效；diag 实测代价只有 0.69%。保留 `--feat-include star` 作消融 |
| **D2** | 要不要 `playcount/passcount` 等流行度？ | **默认不放**（时间穿越 + 与 tag 票数混淆）；作消融 |
| **D3** | 要不要 `mapper_id` 嵌入？ | v1 **不放**；v2 作消融（1,399 个 mapper，是风格代理也是流行度代理） |
| **D4** | tag 编码默认？ | **二进制**；`logcount`/`share` 作消融 |
| **D5** | 要不要做冷启动（C 组）？ | **要，而且应该是本方案的主结果**——这是内容表示唯一不可替代的能力，也是 `cm3p_step2_results.md` §5 列的第一项未做 |
| **D6** | 先续爬 tag 还是先建特征？ | **先续爬**（0.5 h）。但注意：它**不是** B 组的前提——U_tag 已有 **24.4 万测试行（58%）**，统计上够用（§5.1）。续爬的收益是 U_tag 谱面数 +24%、并且少一档"没抓到"的缺失，属于加分项 |
| **D7** | 玩家侧要不要也做内容（cold player）？ | 不在本方案；但要注意现在所有评估都是 warm-player |
| **D8** | 要不要同时接 CM3P-local（100% 覆盖）做"metadata+tags+CM3P"？ | 值得，但**放在 A/B 之后**——先把免费特征的增益钉死，再加贵的 |
| **D9** | **主模型取纯内容还是分层收缩？** | **分层收缩**（§4.5）。外部文献一致推荐"不要二选一"，且本项目"处处接近冷启动"的稀疏度使自由 per-item 参数在统计上不可估。纯内容与纯自由作为它的两个极限保留为消融 |
| **D10** | 要不要上 DropoutNet 式 dropout？ | **要**（§4.6）。它让一个 checkpoint 同时给出热启动与冷启动结果，省掉一份独立训练；不做的话冷启动必须另跑纯内容配置 |
| **D11** | 元数据特征要不要用**惩罚式选择**（BTLLasso 思路）而不是手挑 7–12 个？ | v1 先手挑（可解释、好对照）；v2 用组 lasso 做特征选择，回答"到底哪几个字段有用"——现在手挑的清单没有任何数据支持 |
| **D12** | 要不要按"前向时间"重做一次评估口径？ | 建议**在 D 组做**（限制 test 到 `timestamp < cut`）。外部调研指出随机划分会实质改变推荐系统评估的结论（Meng et al., RecSys 2020）；本项目的主口径是 pair-random，值得至少量一次偏差 |

---

## 11. 复现命令（计划，尚未可运行）

```bash
# M0 续爬 tags（断点缓存已在 data/interim/step0f_1k/）
python scripts/step0f_fetch_tags.py --tag 1k --workers 6 --max-rps 4

# M1 划分 + 清洗（产出带 split 列的 step1）
python scripts/step1_clean.py --tag 1k

# M2 特征表（统计量只用 train 谱面）
python scripts/step0h_chart_features.py --tag 1k --blocks M,T \
    --star off --popularity off --mapper off --tag-encoding binary --tag-min-charts 20

# A 组：全谱面 metadata
python main.py -c configs/exp_meta_1k.yaml \
    --set steps.step2.params.models=mf_dot \
    --set steps.step2.params.chart_content=feat \
    --set steps.step2.params.content_path=data/processed/chart_features_1k_MT.parquet

# B 组：同子集 tags 对照（必须先跑 1-hot）
python main.py -c configs/exp_meta_tags_1k.yaml \
    --set steps.step2.params.restrict_to_content=on      # B1 同子集 1-hot

# C 组：冷启动
python scripts/step1_clean.py --tag 1k --split-method beatmap-random
python main.py -c configs/exp_meta_cold_1k.yaml \
    --set steps.step2.params.drop_bias=m --set steps.step2.params.chart_vocab=all
```

---

## 12. 一句话总结

**模型不用新写——step2 的 `--chart-content` 已经是"冻结谱面表示 + 自由玩家向量"的插座，
把插头从"CM3P 向量"换成"metadata+tags 特征经一个可学矩阵 `W`"即可；
真正的增量是特征工程（长条比例与密度是免费且已验证的）、同子集对照纪律，
以及为冷启动补上 `item_ids`/`chart_vocab` 这一处代码改动。**

> **外部调研后的修订（2026-10-14）**：模型形从"纯内容"改成"**分层收缩**"（§4.5），
> 并加 DropoutNet 式 dropout（§4.6）。理由见 §13。

---

## 13. 外部文献调研（2026-10-14）

由子代理检索 OpenAlex / arXiv / 会议论文集完成。**只列对本方案有直接约束作用的结论**，
每条给出处。无法独立核实的标识符已在该子代理报告中标注。

### 13.1 侧特征 vs 自由 item 嵌入（推荐系统）

**结论：文献不推荐二选一，推荐"加法混合"或"内容作先验"的收缩形式。**

| 形态 | 代表工作 |
|---|---|
| 内容 → 潜因子的线性映射（**本方案 §4.1 的直接祖先**） | Gantner et al., ICDM 2010, [10.1109/icdm.2010.129](https://doi.org/10.1109/icdm.2010.129) |
| 加法混合：item 因子 = 自由 ID 因子 + 内容的投影 | VBPR, [10.1609/aaai.v30i1.9973](https://doi.org/10.1609/aaai.v30i1.9973) |
| **内容作先验、方差可学（本方案 §4.5 的模板）** | CTR, [10.1145/2020408.2020480](https://doi.org/10.1145/2020408.2020480)；Collective MF, [10.1145/1401890.1401969](https://doi.org/10.1145/1401890.1401969) |
| **分支 dropout 防止内容支被忽略（本方案 §4.6）** | DropoutNet, NeurIPS 2017, [W2753686090](https://openalex.org/W2753686090) |
| 内容特征与学到的嵌入**不在同一空间**，需要对齐/蒸馏 | Aligning Distillation, SIGIR 2023, [10.1145/3539618.3591732](https://doi.org/10.1145/3539618.3591732) |
| **基线纪律**：自由因子基线必须同等调参 | Dacrema et al., RecSys 2019, [10.1145/3298689.3347058](https://doi.org/10.1145/3298689.3347058)；Rendle et al., RecSys 2020, [10.1145/3383313.3412488](https://doi.org/10.1145/3383313.3412488) |

**类比弱处**：多数冷启动论文假设 item **零交互**；本项目的"处处接近冷启动"（pair 中位 1 条 play、
谱面中位 7 个玩家）不是那类文献区分的区制。

### 13.2 IRT 用 item 特征预测 item 参数（**最贴近本项目的传统**）

- **Linear Logistic Test Model**（Fischer 1973）与 **Explanatory IRT** 就是这件事；
  见 §4.7。实践指南：[10.7275/8f33-hz58](https://doi.org/10.7275/8f33-hz58)；GLMM 写法 [10.18637/jss.v039.i12](https://doi.org/10.18637/jss.v039.i12)。
- **"协变量没有完全解释难度"是可检验且常被拒绝的** → 必须保留 item 残差：
  Random Weights LLTM, [10.1177/0146621602026003003](https://doi.org/10.1177/0146621602026003003)。
- **小样本 item 校准（对"中位 7 个玩家"最关键的一篇）**：层次先验是公认补救，
  [10.1177/0146621619893786](https://doi.org/10.1177/0146621619893786)、[10.1177/01466216241227547](https://doi.org/10.1177/01466216241227547)。
- **最有价值的方法学引用**：Vie & Kashima, *Knowledge Tracing Machines*, AAAI 2019,
  [10.1609/aaai.v33i01.3301750](https://doi.org/10.1609/aaai.v33i01.3301750) ——
  **证明 FM 涵盖 additive factor model / PFA / 多维 IRT，且在数据稀疏时仍能准确估计能力并使用侧信息。**
- **类比弱处（必须写进结果）**：教育 item 通常有上百次作答（本项目的 10 倍以上）；
  IRT 假设二值响应，而 `log(1−ACC)` 是连续的、**在心理测量学里没有先例**。

### 13.3 节奏游戏 / 体育的技能评分与侧信息

- **最接近的已发表工作**：BiRating（Beat Saber，2025），[arXiv:2502.19742](https://arxiv.org/abs/2502.19742) ——
  **就是本项目的模型类**（玩家×谱面二部图、只用分数迭代平均），并报告了假设失效的谱面族上难度估计退化，
  这正是"注入谱面内容"的标准动机。
- 带协变量的配对比较：BradleyTerry2 [10.18637/jss.v048.i09](https://doi.org/10.18637/jss.v048.i09)；
  **BTLLasso**（在自由能力项**之外**对协变量做惩罚选择）[10.18637/jss.v088.i09](https://doi.org/10.18637/jss.v088.i09)
  → 对应 §10 D11。
- osu!mania 社区技能系统（**非同行评议，仅作先例指针**）：
  [Mania Skills System](https://github.com/SK-la/Ez2Lazer/wiki/Mania-Skills-System-English)、
  [osu-mania-skills](https://github.com/mjh10042/osu-mania-skills)。
- **关键提示**：mania 难度是**真多维**（skillset），因此**单一标量 `b_m` 是设定错误**——
  这独立支持了 §4.4（把 `D` 也内容锚定）与 §10 D9。

### 13.4 极稀疏、人群投票 tag 的编码

1. **优先二进制，其次 `log1p`，`share` 只在带收缩时用，绝不用原始计数** ——
   稀疏词式特征上 Bernoulli 模型通常不劣于计数模型：[W1550206324](https://openalex.org/W1550206324)。
2. **IDF 式降权近普遍 tag + 对流行度残差化** → §3.3 第 2、3 条。
3. **不要静默零插补**：未标注是 MNAR。显式缺失指示是经典解法（Jones 1996, JASA,
   [10.1080/01621459.1996.10476680](https://doi.org/10.1080/01621459.1996.10476680)），
   但该法**对预测合法、对效应估计有偏**（Groenwold 2012, [10.1503/cmaj.110977](https://doi.org/10.1503/cmaj.110977)）
   → 与本项目"只做预测"的定位一致。
4. **tag 块用单独的组惩罚**（不要与稠密特征共用 `wd`）→ §3.3 第 1 条。
5. **已知失效模式**：稀疏、同义、噪声是 tag 型推荐系统的三大问题
   （survey, [10.1007/s11390-011-0176-1](https://doi.org/10.1007/s11390-011-0176-1)）。

### 13.5 泄漏与时间混淆

- **最强的补救手段我们做不到**（无逐票时间戳，见 §6 第 2 条）→ 只剩禁运期 + 前向时间评估。
- **随机划分会实质改变结论**：Meng et al., RecSys 2020, [10.1145/3383313.3418479](https://doi.org/10.1145/3383313.3418479)
  → §10 D12。
- **流行度/反馈环**：[10.1145/3340531.3412152](https://doi.org/10.1145/3340531.3412152)、
  [10.1145/3306618.3314288](https://doi.org/10.1145/3306618.3314288)。
- **因果陷阱**：tags 是 post-treatment 变量，条件在它上面会产生 collider 偏误
  （Elwert & Winship 2014, [10.1146/annurev-soc-071913-043455](https://doi.org/10.1146/annurev-soc-071913-043455)）
  → 只能用于预测，不能用于因果结论。

### 13.6 外部调研改变的三处技术选型（汇总）

| # | 原方案 | 修订后 | 依据 |
|---|---|---|---|
| 1 | 纯内容模型是主线，分层收缩是可选 v3 | **分层收缩是主模型**，纯内容/纯自由是它的两个极限 | §13.1 的 CTR/HFT/VBPR 一致结论 + §13.2 的小样本不可估 |
| 2 | 冷启动要另跑一次纯内容配置 | **加 `--content-dropout`**，一个 checkpoint 同时覆盖热/冷 | §13.1 的 DropoutNet |
| 3 | 只看 RMSE 的 A/B 判据 | **增加 `σ²`/`λ` 读数与 tagged/untagged 分层报告** | §13.1 的"σ² 回答侧特征有没有用" + §13.4 的缺失分层 |

**未被改变的部分**：特征工程规格（§3）、同子集对照铁律（§5.1）、
`--chart-vocab` 的冷启动代码改动（§5.3）、实现地雷清单（§4.2）——外部调研没有推翻这些，
只是把模型形态往"混合"推了一步。
