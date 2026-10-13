# Step 1 — train/test 划分 + 数据清洗

**目标.** 从 step0 的 play 表出发：

0. **划分**（2026-10-12 起在这里做）：把样本单位分成 train / test 两组，train 占 `--train-frac`
   （默认 0.70）。默认 `--split-method pair-random`：对所有 `(player_id, beatmap_id)` **有序对**
   均匀随机；`beatmap-random`：随机挑出 `1 - train_frac` 比例的 `beatmap_id`，这些谱面的
   **所有**行都进 test。划分结果写进新列 `split`（§2）。
1. **删除 ACC ≤ 0.90 的游玩**（**只作用于划入 train 的行**）；
2. **删除每个 `player_id` 在时间上最早的 `floor(0.3 · n)` 条游玩**（同样只作用于 train 行；
   `n` = 该玩家**划入 train 的行**在 (1) 之后剩下的条数）。

**状态.** 已实现并执行；`scripts/step1_clean.py` 产出带 `split` 列的
`step1_{tag}.{csv,parquet,json}`，全部 post-condition 通过（§6）。

**一句话口径.** 先划分（test 行**原样保留**，绝不参与清洗），再对 train 行按 ACC 砍掉脏记录、
按玩家砍掉「入门期」；`playcount_cur` 不重算，所以删行后会出现空洞。**输出 = 输入的全部行**
（顺序、数值都不变），只多一列 `split`。不做 cell 聚合、不做收敛判定、不做权重。

---

## 1. 这一步在哪里

| 步骤 | 名称 | 输入 | 输出 | 本文档 |
|---|---|---|---|---|
| 0 | 预处理 | 原始 dump | `step0_{tag}.csv`（play 表）+ `beatmap_meta_{tag}.csv` | [step0_preprocessing.md](step0_preprocessing.md) |
| **1** | **划分 + 清洗** | step0 的 play 表 | **`step1_{tag}.csv`**（5 列 + `split`，行数与 step0 相同） | **yes** |
| 2 | 建模 | step1 的表（按 `split` 取 train/test） | 潜向量 + 指标 | [step2_prediction.md](step2_prediction.md) |

step0 的 play 表是 `player_id, beatmap_id, timestamp, playcount_cur, loss`，只含
**1.0×（NM）的 classic 游玩**（`DT/NC/HT` 被拒、`NF` 被拒、`MR/SD/PF` 被 canonicalize 掉，
见 step0 §9 D1）。因此 `beatmap_id` 直接就是曲目，**没有 `rate` 列**。

### 1.1 train/test 划分（`split` 列）

划分**在 step1 完成**（2026-10-12 由 step2 移到这里）：step2 只按 `split` 列取数，自己不划分。

| 参数 | 默认 | 含义 |
|---|---|---|
| `--train-frac` | `0.70` | 划入 train 的比例；test = 1 − 该值 |
| `--split-method` | `pair-random` | `pair-random`：对 `(player_id, beatmap_id)` **有序对**均匀随机，同一对的行必在同一侧；`beatmap-random`：随机挑出 test 比例的 `beatmap_id`，这些谱面的所有行都进 test（谱面级外推，更严格） |
| `--split-seed` | `20260928` | 划分种子。与 step2 的旧 `--split-seed` 相同，因此 `pair-random` 的划分与旧流程（step2 自己划分）**逐对完全一致**（已实测：1k 的 259,500 个 test 对完全相同） |

输出新增一列 `split`（字符串，三选一）：

| 值 | 含义 |
|---|---|
| `"train"` | 划入 train，且通过了下面的两条清洗规则 |
| `"test"` | 划入 test。**原样保留**：清洗是 train 侧的处理，用它去筛 test 就是以处理结果为条件（[archive/cleaning_plan.md](archive/cleaning_plan.md) §3.2） |
| `"dropped"` | 划入 train，但被规则一或规则二删掉 |

两个方法都保证**划分单位不重叠**：test 侧的行绝不会出现在 train 里（脚本每次运行都校验）。
行的总数不变：`train + test + dropped = step0 行数`。

**为什么清洗只作用于 train。** 保留旧口径（train 用清洗后的标签、test 用未清洗的标签），
但把「哪些行被清洗掉」限定在训练侧：这样 test 集的定义不依赖清洗参数，换 `--acc-floor`
或 `--drop-frac` 时测试集不变，实验之间可比。

**与旧流程的差别（重要）。** 旧 step1 对**所有**行做清洗，再用 step2 自己划分 pair；
因此规则二的 `n` 里包含 test 行。现在 `n` 只数 train 行，`floor(0.3n)` 因此略小：
1k 的 train 从 674,436 行变成 **674,890 行**（+454）。划分本身不变（同一批 test 对），
变的是被清掉的行。

### 与旧 step1 的关系

本版**不在旧实现上修改**，是另起的一套。旧的那套（`scripts/legacy/build_v2.py` →
`charts_v2_*_v3.parquet`、`scripts/legacy/convergence_filter.py`）建在**已被 step0 取代的总体**上：
它读 `data/legacy/interim` 里带 mod 的中间表，自己分 `NM/DT/HT` 三档（`chart_id = beatmap_id·4 + rate`）、
保留 `NF`。两套的口径不可互换，旧的规模数字（primary 141,477 等）**不能**直接搬到本版上。

---

## 2. 清洗：两条规则的精确定义（只作用于 `split == "train"` 的行）

> 下面两条规则**只跑在划入 train 的行上**；`split == "test"` 的行原样保留，
> `split == "dropped"` 就是「划入 train 后被这两条之一删掉」的行。

### 2.1 规则一：ACC 门槛

```
删除 ACC <= 0.90 的游玩（仅 train 行）
```

`loss = log(1 − ACC)`（step0 §4.1），所以

```
ACC <= 0.90   <=>   1 - exp(loss) <= 0.90   <=>   exp(loss) >= 0.10   <=>   loss >= log(0.10)
```

**实现上直接比较 loss**：保留 `loss < log(0.10) = −2.302585092994046`，即删掉 `loss >= log(0.10)`。
这样做而不是先算 `ACC = 1 − exp(loss)`，是为了避开 float32 下 `exp` 在阈值附近的舍入：
loss 空间比较是单调等价的，且与 step0 输出的列直接对应。

**边界是严格的**：ACC 恰好等于 0.90 的游玩**被删除**（保留的是 `ACC > 0.90`）。
实测输出的 `min ACC = 0.9000000032`、`max loss = −2.3025851`，两端都刚好卡在阈值外侧。

**注意这不是旧版的口径。** 旧 `build_v2.py` 用的是 **cell 中位 ACC < 0.90**（先聚合再筛，
丢整个 cell）；本版是**逐条游玩**的 ACC，cell 只会在其记录被逐条删掉后才消失。

### 2.2 规则二：每个玩家的「最早 30%」

```
对每个 player_id（只看划入 train 的行）：
    把这些行按时间升序排列
    n = 条数（= 规则一筛完之后该玩家的 train 行数）
    删除前 floor(0.3 * n) 条，保留第 floor(0.3*n)+1 条及其后
```

- **是全局的 per-player 截断，不是 per-cell 的。** 键只有 `player_id`，跨谱面、跨年份一起排。
- **取整用 `floor`**（已确认）：`n = 3 → 删 0`，`n = 10 → 删 3`，`n = 100 → 删 30`。
  所以 **`n < 4` 的玩家一条都不删** —— 1k 有 6 个、10k 有 71 个这样的玩家（§4.2）。
  代价是删掉的略少于 30%（1k 实测 29.96%、10k 29.97% 的候选比例，见 §4.4 注）。
- **排序键** = `(player_id, timestamp, beatmap_id, playcount_cur)`。
  step0 内部的排序键是 `(user_id, beatmap_id, ts, sid)`，而 `sid`（score id）**不在** play 表里，
  所以同一秒内并列时的先后无法完全复原，用 `(beatmap_id, playcount_cur)` 作确定性 tie-break。
  这是全流程唯一的近似点（§5）。

### 2.3 顺序：先划分 → 先 ACC → 后时间

三步的先后是固定的：**先划分（§1.1），再筛 ACC，最后砍时间**。因此规则二里的 `n` 是
「该玩家的 train 行经过 ACC 筛之后」的行数，「最早 30%」是相对干净记录算的。

两种顺序的结果差异很小，但是有方向性的：

| tag | 先 ACC 后时间（本版） | 先时间后 ACC | 差 |
|---|---|---|---|
| 1k | **964,417** | 968,538 | −4,121（−0.43%） |
| 10k | **9,119,324** | 9,188,444 | −69,120（−0.75%） |

先筛 ACC 会**多删**一点，原因很直接：ACC 淘汰的脏记录有 **48.3%（1k）/ 50.4%（10k）**
本来落在「全量最早 30%」窗口内；先把它们删掉，剩下的干净记录重新算 30%，于是又多吃掉一批
原本靠脏记录「占位」而幸存的早期记录。差异 < 1%，但如果以后要跟别的口径对比，必须记住本版是哪一个。

（上表与 §5.4 的数字来自 2026-10-03 的**旧口径**运行：当时对所有行做清洗、划分在 step2。
新口径下规则二的分母只数 train 行，绝对行数会略变，比例结论不变。）

### 2.4 输出契约

| 项 | 值 |
|---|---|
| 列 | `player_id, beatmap_id, timestamp, playcount_cur, loss, split` —— step0 的 5 列（同顺序）**加一列 `split`** |
| 行 | **与 step0 行数相同、顺序相同**；`split` 之外每一列都与 step0 逐值相同 |
| `split` | 字符串 `"train"` / `"test"` / `"dropped"`（§1.1） |
| `timestamp` | `YYYY-MM-DD HH:MM:SS`（无引号），与 step0 的 CSV 逐字节同格式 |
| `loss` | `float32`（与 step0 的 `--float-dtype` 默认一致） |
| `playcount_cur` | **保留 step0 原值，不重算**（见下） |
| 文件 | `data/processed/step1_{tag}.csv`、`step1_{tag}.parquet`、`step1_{tag}.json` |
| 写盘 | 原子（`.part` + `replace`），中断不会留下半截文件 |

step2 的取数方式：`split == "train"` 的行训练、`split == "test"` 的行评估、
`split == "dropped"` 的行忽略（不改标签、不参与训练）。

**为什么不重算 `playcount_cur`。** 它的定义是「这是该 (player, beatmap) cell 的第 k 条**记录**」。
删行后它的取值会出现空洞（如 `5,6,7…`），但这正是我们想保留的信息：**能看出该 cell 前面被删了几条**。
重算成 `1..n′` 会把这个信息抹掉，而且会让「同一张谱面在第几次游玩时达到什么水平」失去可比性。
实测有空洞的 cell：1k **75,725 / 577,999 = 13.10%**，10k **707,815 / 5,145,934 = 13.75%**
（非零即证明没有被重编号，见 §6 的 `output_rows_are_unmodified_input_rows`）。

---

## 3. 刻意不做的事

这一版 step1 **只做上面两条**。以下都**不做**，留在后续步骤：

| 不做 | 旧实现里对应什么 |
|---|---|
| cell 聚合（`loss_med` / mean / max / min、`n_scores`） | `build_v2.py` 的核心产物 |
| `(player, beatmap)` 的 `attempts <= 2` 门槛（`osu_user_beatmap_playcount`） | `build_v2.py` 剔除② |
| cell 自身条数 `n_scores <= 2` 门槛 | `build_v2.py` 剔除③ |
| 收敛判定 / 分层 / 样本权重 | `convergence_filter.py`、`cleaning_plan.md` §2 |
| 谱面元数据派生（`star`、`count_total` 地板、density…） | step0 §1 非目标、§13 |
| 时间派生特征（`span`、gap、日历年） | step0 §13.5 |
| `playcount_cur` 归一化（除以 cell size） | step0 §13.1 |
| `rate` / `chart_id` 分档 | 已被 step0 取消（只留 1.0×） |

所以 `step1_{tag}.csv` **不是**一个可直接训练的 cell 表，它是**行级清洗后的 play 表**。

---

## 4. 实测（旧口径：2026-10-03 的运行，清洗作用于全部行）

> 下面 4.1–4.4 的数字来自划分还在 step2、清洗作用于全部行的旧口径，保留作为对照；
> 新口径（划分在 step1、清洗只作用于 train）的实测见 §4.5。

### 4.1 漏斗

| | 1k | 10k |
|---|---|---|
| step0 行数 | 1,399,615 | 13,359,355 |
| 规则一删（ACC ≤ 0.90） | −22,517（1.61%） | −338,041（2.53%） |
| 规则二删（最早 `floor(0.3n)`） | −412,681（29.96% 于剩余行） | −3,901,990（29.97%） |
| **step1 行数** | **964,417** | **9,119,324** |
| 保留比例 | **68.91%** | **68.26%** |

### 4.2 玩家与截断量

| | 1k | 10k |
|---|---|---|
| 玩家 in → out | 990 → **990** | 9,812 → **9,793** |
| 一条都没删的玩家（`floor(0.3n)=0`，即 `n ≤ 3`） | 6 | 71 |
| 每玩家删除条数：中位 / p75 / p90 / p99 / max | 300 / 563 / 922 / 1,907 / **3,517** | 309 / 528 / 825 / 1,662 / **5,894** |
| 每玩家保留条数：中位 / max | 701 / 8,209 | 724 / 13,754 |

10k 有 **19 个玩家整段消失**（他们的所有记录都被 ACC 门槛删光）；1k 没有这种玩家。

### 4.3 输出表性质

| | 1k | 10k |
|---|---|---|
| 谱面 | 20,376（step0: 20,655） | 21,899（step0: 21,899，**未减少**） |
| cell | 577,999（step0: 866,616） | 5,145,934（step0: 7,784,519） |
| 单记录 cell | 413,316 = **71.51%**（step0: 74.12%） | 3,581,082 = **69.59%**（step0: 72.19%） |
| `loss` mean / sd | −5.9760 / 2.1340 | −5.0216 / 1.9161 |
| `loss` min / max | −12.4663 / **−2.3026** | −12.4663 / **−2.3026** |
| `ACC` mean / min | 0.98801 / **0.9000000** | 0.98074 / **0.9000000** |
| `playcount_cur` mean / max | 2.624 / 125 | 3.052 / 527 |
| 时间范围 | 2015-02-13 … 2026-08-31 | 2014-02-14 … 2026-08-31 |
| 运行时间 / CSV 体积 | 16.3 s / 48.4 MB | 135.9 s / 459.3 MB |

两点值得注意：

- **`loss` 的 max 从 step0 的 −0.2669（1k）/ −0.1358（10k）变成了 −2.3026** —— 这正是规则一的痕迹：
  极差的那批（ACC 0.234 之类）全被删掉了，剩下的上界就是 ACC = 0.90。
- **单记录 cell 的比例反而略微下降**（1k 74.12% → 71.51%）。两个方向同时发生：
  只记录过一次的 cell 如果那条被删，整个 cell 消失；而记录多的 cell 被砍掉早期几条后，
  有可能掉到只剩 1 条、反而加入单记录集合。净效果是后者略占上风。

### 4.5 新口径实测（2026-10-12，划分在 step1、清洗只作用于 train）

运行方式（step0 用缓存，只跑 step1）：

```bash
python main.py -c configs/exp_std_1k.yaml --only step1 --set name=split_check       # 1k, pair-random
python main.py -c configs/exp_10k.yaml --only step1 --set name=check10k \
    --set steps.step1.params.formats=parquet                                        # 10k, pair-random
python main.py -c configs/exp_std_1k.yaml --only step1 --set name=chart_split \
    --set steps.step1.params.split_method=beatmap-random                            # 1k, 谱面级
```

| 项 | 1k（pair-random） | 10k（pair-random） | 1k（beatmap-random） |
|---|---|---|---|
| step0 行数 | 1,399,615 | 13,359,355 | 1,399,615 |
| 划分单位 | 866,616 对 | 7,784,519 对 | 20,655 张谱面 |
| train / test 单位 | 607,116 / 259,500 | 5,450,441 / 2,334,078 | 14,270 / 6,385 |
| 实际 train 比例 | 0.70056 | 0.70016 | 0.69073 |
| `split == "test"` 行 | 420,401 | 4,007,074 | 447,399 |
| train 候选行 | 979,214 | 9,352,281 | 952,216 |
| 规则一删（ACC ≤ 0.90） | −15,710 | −237,052 | −15,404 |
| 规则二删（最早 `floor(0.3n)`） | −288,614 | −2,730,189 | −280,578 |
| `split == "train"` 行 | **674,890** | **6,385,040** | **656,234** |
| `split == "dropped"` 行 | 304,324 | 2,967,241 | 295,982 |
| 输出行数 | 1,399,615（= step0） | 13,359,355（= step0） | 1,399,615（= step0） |
| 校验 | 12/12 | 12/12 | 12/12 |
| 运行时间 / 产物 | 4.7 s / 21 MB parquet | 94.4 s / 184 MB parquet | 4.7 s / 21 MB parquet |

`beatmap-random` 的 train 比例略低于 0.70（0.69073）：单位是谱面，而每张谱面的 play 数不同，
按谱面数 70% 划分不等于按行数 70%；`actual_train_frac` 记在报告的 `split` 块里。

与旧口径的对照（pair-random）：

| | 旧（划分在 step2，清洗全部行） | 新（划分在 step1，清洗 train） |
|---|---|---|
| test 对集合（1k） | 259,500 对 | **完全相同**（逐对比对通过） |
| test 对集合（10k） | 2,334,078 对 | **完全相同**（`n_pairs`/`n_test_pairs` 逐值一致） |
| test 行（1k / 10k） | 420,401 / 4,007,074 | 420,401 / 4,007,074 |
| train 行（1k） | 674,436 | **674,890**（+454） |
| train 行（10k） | 6,384,278 | **6,385,040**（+762） |
| 被删行 | 丢弃，输出里没有 | 保留并标为 `dropped`（1k 304,324 / 10k 2,967,241） |

test 集完全没变，train 只多出几百行 —— 因为规则二的分母从「该玩家全部行」变成
「该玩家 train 行」，`floor(0.3n)` 因此略小。

### 4.4 顺序选择的影响（§2.3 的展开）

| tag | 先 ACC 后时间 | 先时间后 ACC | 差 | ACC 淘汰行中落在「全量最早 30%」内的比例 |
|---|---|---|---|---|
| 1k | 964,417 | 968,538 | −4,121（−0.43%） | 10,871 / 22,517 = **48.3%** |
| 10k | 9,119,324 | 9,188,444 | −69,120（−0.75%） | 170,503 / 338,041 = **50.4%** |

---

## 5. 已知局限与近似

1. **同秒并列的先后不可复原。** 时间戳是秒级，step0 内部用 `sid` 破平局，而 `sid` 不在 play 表里。
   本版用 `(beatmap_id, playcount_cur)` 破平局。只有当「第 `k` 条和第 `k+1` 条落在同一秒」
   且恰好被 30% 切在中间时才会选错一条，影响可忽略但存在。
2. **「30%」是记录数的 30%，不是真实游玩数的 30%。** `playcount_cur` 只数**已记录的成功游玩**，
   覆盖率约为真实游玩次数的 1/3（step0 §3.2、§8.1）。失败/放弃/中途退出的游玩不在表里，
   而它们**按假设集中在这条时间轴的前段**。因此真实意义上的「入门期」比被删掉的部分更长，
   本版删掉的 30% 是「记录里的最早 30%」。
3. **`floor` 让小玩家免疫。** `n ≤ 3` 的玩家一条不删（1k 6 人 / 10k 71 人）。
   这是 `floor` 的直接后果，不是 bug；若要「至少删 30%」得改 `ceil`。
4. **规则一在 loss 空间比较**，与 `ACC > 0.90` 数学等价；float32 下 ACC 恰为 0.90 的记录
   可能落在保留侧，属测度为零的边界情形。
5. **`playcount_cur` 有空洞**（§2.4）。下游若假设它是 `1..n` 连续，会踩到；这是刻意的。
6. **被删的行现在有标记**：`split == "dropped"` 就是「划入 train 但被清洗删掉」的行，
   它们**仍在输出里**（旧版会丢）。可以据此做「清洗前 / 清洗后」的消融，不必重跑。
   代价：输出与 step0 同大小（1k 的 CSV 从 48 MB 变成 70 MB 量级），不再省磁盘。
7. **不做 cell 级聚合**，所以「同一 cell 内还有几条」「这个 cell 能不能用」都要下游自己算。

---

## 6. 校验（脚本每次运行都跑，失败即 `SystemExit`）

| # | 断言 | 说明 |
|---|---|---|
| 1 | `rows_preserved` | 输出行数 == step0 行数 |
| 2 | `split_labels_valid` | 标签只有 train/test/dropped，且三类之和 == 行数 |
| 3 | `split_units_disjoint` | train 侧（含 dropped）与 test 侧的划分单位（有序对或谱面）**无交集** |
| 4 | `actual_train_frac_within_1pct` | 实际 train 比例与 `--train-frac` 相差 ≤ 1% |
| 5 | `min_acc_strictly_above_floor_on_train` | **train 行**最小 ACC > 0.90（test 行不受限） |
| 6 | `loss_negative` | 所有 `loss < 0` |
| 7 | `per_player_train_kept_equals_n_minus_floor` | 每玩家 train 保留数 == `n − floor(0.3n)` |
| 8 | `no_dup_player_beatmap_playcount` | `(player, beatmap, playcount_cur)` 无重复 |
| 9 | `playcount_cur_ge_1` | — |
| 10 | `playcount_cur_monotonic_in_cell` | cell 内 `playcount_cur` 单调递增（用 `groupby.diff()` 向量化） |
| 11 | `output_rows_are_unmodified_input_rows` | 除 `split` 外**每一列、每一行**都与 step0 逐值相同、顺序相同 —— 最强的一条，同时排除「重排」与「重算列值」两类 bug |
| 12 | `no_time_overlap` | 每个玩家被删的最后一条 ≤ 保留的第一条 |

1k 的新口径运行：12/12 通过（§4.5）。

---

## 7. 复现

```bash
# 推荐：走统一入口（参数、产物、日志都进 experiments_{时间戳}/）
python main.py -c configs/exp_std_1k.yaml --only step1

# 也可以单独跑脚本（runbook 风格）
python scripts/step1_clean.py --tag 1k     # → step1_1k.{csv,parquet,json}
python scripts/step1_clean.py --tag 1k --split-method beatmap-random --train-frac 0.7
python scripts/step1_clean.py --tag 10k --no-csv   # 10k 只留 parquet
```

全部可配置（都有默认值，不硬编码）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--tag` | `1k` | 决定输入 `step0_{tag}.{parquet,csv}` 与输出文件名 |
| `--in` | 自动探测 | 显式指定 step0 play 表；`.parquet` 优先于 `.csv` |
| `--out` | `data/processed/step1_{tag}.csv` | 输出 CSV 路径（`.parquet` / `.json` 跟随同名） |
| `--train-frac` | `0.70` | 划入 train 的比例（test = 1 − 该值） |
| `--split-method` | `pair-random` | `pair-random` \| `beatmap-random`（§1.1） |
| `--split-seed` | `20260928` | 划分种子 |
| `--acc-floor` | `0.90` | 删除 `ACC <=` 此值的游玩（仅 train 行） |
| `--drop-frac` | `0.30` | 每个玩家删除最早的比例（仅 train 行） |
| `--no-parquet` | off | 只写 CSV |
| `--no-csv` | off | 只写 parquet（`--out` 仅作命名基准） |

---

## 8. 未决问题

1. **`floor` vs `ceil`。** 现在 `n ≤ 3` 的玩家完全不被截断。若认为「入门期必须至少砍掉 30%」，
   改 `ceil`（代价：小样本玩家损失更多）。
2. **要不要区分「被规则一删」和「被规则二删」。** 现在两者都是 `dropped`，要细分得再加列
   （如 `drop_reason`）；影响做消融时的解释粒度。
3. **「30%」是否该按 cell 而非按玩家算。** 本版按玩家全局算（已确认）；若建模关心的是
   「同一张谱面内部的顺序」，per-cell 的截断是另一种（不同）的口径。
4. **是否补一个 `n` 列（cell size）**，让下游不必自己 groupby 求 `max(playcount_cur)`。
5. **是否把 step0 的 `beatmap_meta` 派生特征并进来**（`star` 等）—— step0 §9 D5 明确把它划给
   step1，但本版没做（本版只筛行）。
