# Step 1（默认）— pair-best 清洗 + 玩家空窗拆分（2026-10-20）

**精简版.** [preprocess/README.md](../preprocess/README.md) 与
[preprocess/step1_pairbest.py](../preprocess/step1_pairbest.py) 是本方法的压缩参考；
1k/10k 输出与生产 A 表逐值相同。

**默认口径.** 从 2026-10-20 起，<code>step1</code> 不写 <code>version</code> 时使用本版本
（<code>dcr/config.py</code> 的 <code>STEP_DEFAULT_VERSION</code>）；默认参数就是下面的**口径 A**：
玩家记录数 &gt;= 20、1 年空窗拆分、每对只留最高成绩、单次对剔除、最高 ACC ∈ (0.90, 0.998]、
标签 <code>log(1-ACC)</code>。<code>step1/clean</code> 保留为旧的 play 级口径，必须显式写
<code>version: clean</code>；两套总体不可混用。

**目标.** 这是 step1 的**默认版本**（<code>step1/pairbest</code>，<code>scripts/step1_pairbest.py</code>），
与 [step1/clean](step1_cleaning.md) 并列，不修改旧版。它把观测单位从 **play** 换成
**(玩家段, 谱面) 对**，并做六件事：

0. step0 记录数 **< 20** 的玩家整人剔除（保留玩家门槛，阈值从 clean 的 200 降到 20）；
1. 同一玩家连续 **≥ 1 年**没有记录 → 拆成两个（或更多）不同的玩家；
2. 每个 (玩家段, 谱面) 对只保留**最高成绩**（loss 最小）；
3. 该对**只打过一次** → 整个对剔除；
4. 该对的最高 ACC **严格大于 99.8%** → 整个对剔除（<code>log(1-ACC)</code> 的负尾不友好）；
5. 该对的最高 ACC **≤ 0.90** → 整个对剔除（保留 clean 的 ACC 下限）。

清洗在**划分之前**作用于全表，然后在剩下的对上做 pair-random / beatmap-random 划分。
输出是一行一个对的紧凑表（<code>step1_{tag}_pairbest.parquet</code>），不是 step0 的全量行表。
**幸存对的最高 ACC 因此落在 (0.90, 0.998] 区间**。

**状态.** 已实现并在 1k / 10k 上跑通；step1 自检 16/16 通过。**默认口径是 A**
（0.998 上限 + <code>log(1-ACC)</code>），参数默认值就是 A。step2 的
<code>bias</code> / <code>mf_dot</code>（<code>dim=8</code>）结果见 §6；
**有/无 0.998 上限的 A/B 消融、标签 shift（log(1.001-ACC)）的 C/D 实验与
MAE/Spearman/pp_p50 解释见 [step1_cleaning_analysis.md](step1_cleaning_analysis.md)**。

**一句话口径.** 先剔掉记录数 < 20 的玩家，再按 1 年空窗把玩家拆段，然后在每个段内把每张
谱面压成「最高成绩」一行；只打过一次的对、最高 ACC > 99.8% 的对、最高 ACC ≤ 0.90 的对
都不要；最后在保留下来的对上按 70/30 划分。**输出一行 = 一个对**，被剔除的对不留在表里。

---

## 1. 这一步在哪里

| 步骤 | 版本 | 程序 | 产物 | 文档 |
|---|---|---|---|---|
| 0 | <code>preprocess</code> | <code>scripts/step0_preprocess.py</code> | play 表（5 列）+ 谱面元数据 | [step0_preprocessing.md](step0_preprocessing.md) |
| 1 | <code>clean</code> | <code>scripts/step1_clean.py</code> | <code>step1_{tag}.{csv,parquet,json}</code>：全部行 + <code>split</code> | [step1_cleaning.md](step1_cleaning.md) |
| **1（默认）** | **<code>pairbest</code>** | **<code>scripts/step1_pairbest.py</code>** | **<code>step1_{tag}_pairbest.{csv,parquet,json}</code>：一行一对** | **本文档** |
| 2 | <code>predict</code> | <code>scripts/step2_predict.py</code> | 潜向量 npz + 指标 json | [step2_prediction.md](step2_prediction.md) |

**两套 step1 不可混用。** <code>clean</code> 的输出保留 step0 的全部行（被清洗的行标成
<code>dropped</code>），行级清洗只作用于 train 侧，test 是未清洗的 play；<code>pairbest</code>
的输出在物理上只保留「每对最高成绩」的幸存对，test 也经过同样的 pair 过滤。两者的文件名、
行数、观测单位、test 集和标签支撑都不同，不能拿一边训练、另一边当基线。

### 1.1 为什么是独立版本，以及为什么设为默认

旧 <code>clean</code> 的契约是「输出 = 输入的全部行 + <code>split</code> 列」，所有下游分析都假设
<code>(player_id, beatmap_id, playcount_cur)</code> 唯一、一个 cell 里有多条 play。新方法把
一个 cell 压成一条、还会重编号玩家，物理上改变了表的粒度；把它塞进旧版本会破坏
<code>rows_preserved</code> / <code>output_rows_are_unmodified_input_rows</code> 等全部校验，也会让旧实验
在不知情的情况下读到另一套总体。因此新方法作为**独立 step 版本**实现，产物名带
<code>_pairbest</code> 后缀（<code>dcr/steps.py</code> 的 <code>_plan_step1_pairbest</code>）。

**为什么设为默认（2026-10-20）。** 口径 A 是「pair-best + 0.998 上限 + <code>log(1-ACC)</code>」：
它保留了 pair-best 的观测单位与 1 年空窗拆分，同时把 <code>log(1-ACC)</code> 的极端负尾截断在
ACC 0.998，避免 B 那种 RMSE 被少数高 ACC 对拉到 1.1 的问题；它也是最早跑通、与
[cleaning_analysis](step1_cleaning_analysis.md) 的 A 表逐项对应的基准。B 的 pp_p50 更低但
RMSE 最差，C/D 的标签 shift 没有可辩护的依据（<code>1.001</code> 是任意常数），所以都不作为默认。
详见 [cleaning_analysis](step1_cleaning_analysis.md) §4.7。

---

## 2. 规则的精确定义（按执行顺序）

### 2.0 玩家门槛（<code>--min-plays-per-player</code>，默认 20）

与 <code>clean</code> 的规则零同义：按玩家在 step0 全表里的记录数，低于门槛的玩家**整人剔除**。
<code>clean</code> 默认 200，本版本默认 **20**——pair 级规则已经通过「至少两次游玩」筛掉了大量
偶发对，玩家门槛只需再拦住几乎不玩这张表的账号，不必像 clean 那样高。它先于空窗拆分执行，
数的是 step0 全表记录数（不是 pair 数、也不是 ACC 筛后的行）。

实测：1k 剔除 **13 个玩家 / 81 行**；10k 剔除 **217 个玩家 / 1,543 行**。若要与 clean 的 200
对齐，显式传 <code>--min-plays-per-player 200</code>；<code>0</code> 关闭。

### 2.1 玩家空窗拆分（<code>--gap-days</code>，默认 365）

对每个 <code>player_id</code>，把它在 step0 的全部记录按
<code>(timestamp, beatmap_id, playcount_cur)</code> 排序（同秒并列时用后两者确定性破平局）；
相邻两条记录的间隔 **≥ <code>gap-days</code> 天**时，后一条开始一个新段：

~~~text
player 的段落 = 按时间排序后，被「≥ 1 年的空窗」切开的连续段
~~~

- **在 pair 过滤之前**做，用的是**原始 play 历史**。如果先压成「每对最高成绩」再拆，
  一段 2 年的空窗里可能只有被删掉的 play，最高的那条落在空窗一侧，拆分会漏掉；用全量
  记录才能还原「这个人在什么时候停止/恢复活动」。
- **每个段是一个新玩家**。段的编号从 0 开始；**段 0 保留原 <code>player_id</code>**，段 1、2、…
  分配 <code>max(step0 player_id) + 1</code>、<code>+2</code>、… 的全新 id（按 <code>(base, seg)</code> 排序分配，确定性）。
  因此没有空窗的玩家 id 不变；被拆玩家的第一段仍用原 id，后续段是新 id。
- 输出里同时保留 <code>player_id_base</code>（原 id）与 <code>player_seg</code>（段号），可以无损还原。
- **边界是 ≥**：恰好 365 天的间隔也拆。<code>--gap-days 0</code> 关闭（所有段号都是 0）。
- 口径限制：空窗是按 **step0 记录**算的，不是真实游玩次数（记录覆盖率约 1/3，见
  [step0_preprocessing.md](step0_preprocessing.md) §3.2）。一个真实只停玩 8 个月、但记录
  稀疏的玩家可能被误判为停玩 1 年；反之亦然。

### 2.2 每对只留最高成绩

对每个 <code>(player_id_new, beatmap_id)</code> 对：

~~~text
keep = argmin loss        loss = log(1 - ACC) 越小 = ACC 越高
tie-break = 最早 timestamp，然后最小 playcount_cur
~~~

保留行的 <code>timestamp</code>、<code>playcount_cur</code>、<code>loss</code> 都是**原始值**，不重算、不重排。
<code>n_plays</code> 记的是**压成一条之前**该对的记录数（含被丢掉的较低成绩），所以
<code>n_plays >= 2</code> 才可能留下；下游可以据此区分「打了 2 次就达到」与「打了 50 次才达到」。

**可选标签 shift（<code>--label-shift s</code>，默认 0）。** 默认标签是 step0 的
<code>log(1-ACC)</code>；设为 s 时换成

~~~text
loss_new = log(s + exp(loss_step0)) = log(1 + s - ACC)
~~~

ACC 过滤仍按 <code>ACC = 1 + s - exp(loss_new)</code> 判定，所以
(0.90, 0.998] 的窗口不变；变化的只是标签数值与 MSE 的加权。s=0.001 即
<code>log(1.001-ACC)</code>，值域下界是 <code>log(0.001) = -6.9078</code>。
**本节 §2.4/§2.5 的阈值公式是 s=0 的形式**；s>0 时把 <code>log(1-·)</code> 换成
<code>log(1+s-·)</code>。标签 shift 的 C/D 结果见
[step1_cleaning_analysis.md](step1_cleaning_analysis.md) §4，指标解释见 §5。

### 2.3 单次对剔除（<code>--min-plays-per-pair</code>，默认 2）

<code>n_plays < 2</code> 的对整对剔除。等价于用户要求的「若该对游玩次数 == 1 则剔除」。

为什么：一个对只打过一次时，「最高成绩」就是那一次，它既不能代表稳定水平，也没有
「多次尝试里挑最好」的降噪效果；它和「打了多次、取最好」在标签分布上不是同一个东西。
注意这里的「一次」是 **step0 记录数**，不是真实游玩次数（step0 只含约 1/3 的成功游玩）；
一个只记录了一次的对，真实可能打过多次，所以这条规则比字面「只玩过一次」更严格。
1k 里这类对占 **74.4%**（647,523 / 870,038），10k 里占 **72.8%**（5,689,791 / 7,816,887），
所以这条规则是漏斗里最粗的一刀。

### 2.4 最高 ACC > 99.8% 的对剔除（<code>--max-best-acc</code>，默认 0.998）

~~~text
ACC_best = 1 - exp(loss_best)
若 ACC_best > 0.998  ->  剔除整个对
~~~

实现上在 **loss 空间**比较（与 <code>clean</code> 的 ACC 门槛同一做法，避免 <code>exp</code> 在边界上的舍入）：

~~~text
keep  <=>  loss_best >= log(1 - 0.998) = log(0.002) = -6.214608098...
~~~

**边界是严格的**：ACC 恰好 0.998 的对保留，严格大于才删。1k 输出的最大 ACC 是
0.9979997（对应 loss = -6.2144570），10k 是 0.9979999，没有一行越过阈值。

为什么：<code>loss = log(1-ACC)</code> 在 ACC → 1 时趋于 <code>-∞</code>，高 ACC 的标签在 loss 空间被
<code>1/(1-ACC)</code> 放大。这些对既主导 MSE，又处在「再高一点就不可分」的饱和区；把它们留在
训练集里，模型会把容量花在标签的极端负尾上。剔掉后**标签支撑被截断**：输出的
<code>loss ∈ [-6.2146, ...]</code>、<code>ACC <= 0.998</code>。这是有意的口径，不是数据丢失 bug。

### 2.5 低 ACC 对剔除（<code>--acc-floor</code>，默认 0.90）

保留 <code>clean</code> 的规则一：最高 ACC **≤ 0.90** 的对整对剔除，保留 <code>ACC > 0.90</code>。
在 loss 空间：

~~~text
drop  <=>  loss_best >= log(1 - 0.90) = log(0.10) = -2.302585
keep  <=>  loss_best <  -2.302585
~~~

**作用对象是 pair-best 那条**，不是逐条 play：低分 play 在 §2.2 已经被丢掉了，这一条只问
「保留下来的最高成绩是否仍然高于 ACC 下限」。因此一个对即使有一条 0.85、一条 0.95，
最高成绩 0.95 > 0.90，仍会保留（<code>n_plays=2</code>），不会因为低分 play 被删而变成
「单次对」——这是与「先逐条过 ACC 门槛、再压 pair」不同的口径，本版选了前者。

实测剔除：1k **912 对**；10k **13,035 对**。<code>--acc-floor 0.0</code> 关闭。

### 2.6 train/test 划分

在**幸存的对**上调用与 <code>clean</code> 同一个 <code>split_rows()</code>：

| 参数 | 默认 | 含义 |
|---|---|---|
| <code>--train-frac</code> | 0.70 | train 比例；test = 1 − 该值 |
| <code>--split-method</code> | <code>pair-random</code> | 对 <code>(player_id, beatmap_id)</code> 有序对均匀随机；<code>beatmap-random</code> 按谱面整张留出 |
| <code>--split-seed</code> | 20260928 | 与 <code>clean</code> / 旧 step2 相同 |

**划分在清洗之后**，这是与 <code>clean</code> 最重要的语义差别：<code>clean</code> 的 test 行「原样保留、
不参与行级清洗」，所以 test 是未清洗的目标总体；<code>pairbest</code> 的观测单位已经变成对，
test 必须是同一套 pair-best 表，否则一个 test 对可能有多条 play、标签定义不一致。
代价是 test 集也经过了 ACC 窗口（0.90, 0.998] 与单次对过滤，§8 第 2 条把这个代价写清楚。

---

## 3. 输出契约

| 列 | 含义 |
|---|---|
| <code>player_id</code> | **新**实体 id：段 0 = 原 id；段 >0 = 全新 id（> max 原 id） |
| <code>beatmap_id</code> | 原值 |
| <code>timestamp</code> | 保留的**最高成绩**那条的时间 |
| <code>playcount_cur</code> | 保留那条的原始值（不重算） |
| <code>loss</code> | 保留那条的原始值（float32）；<code>--label-shift s>0</code> 时是新标签 <code>log(1+s-ACC)</code> |
| <code>split</code> | train / test（没有 dropped：被剔除的对物理上不写） |
| <code>player_id_base</code> | step0 的原 <code>player_id</code> |
| <code>player_seg</code> | 该玩家内部的段号（0 起） |
| <code>n_plays</code> | 该对压成一条前的记录数（≥ <code>--min-plays-per-pair</code>） |

- **行数 = 幸存对数**，不是 step0 行数；一行一对，<code>(player_id, beatmap_id)</code> 唯一。
- **幸存行的最高 ACC 严格落在 (0.90, 0.998] 区间**：<code>loss ∈ [log(0.002), log(0.10))</code>。
- 文件：<code>data/processed/step1_{tag}_pairbest.csv</code> / <code>.parquet</code> / <code>.json</code>（实验里在
  <code>artifacts/</code> 下，名字不变）。写盘原子（<code>.part</code> + <code>replace</code>）。
- 除 <code>player_id</code> 的重新编号外，每一条输出的测量列都是 step0 对应行的**逐值相同子集**
  （脚本每次运行都校验）。
- step2 的取数不变：<code>split == "train"</code> 训练、<code>split == "test"</code> 评估；<code>dropped</code> 为 0 行。

---

## 4. 参数

| 参数 | 默认 | 含义 |
|---|---|---|
| <code>--tag</code> | <code>1k</code> | 决定输入 <code>step0_{tag}</code> 与输出文件名 |
| <code>--in</code> | 自动探测 | step0 play 表；<code>.parquet</code> 优先于 <code>.csv</code> |
| <code>--out</code> | <code>data/processed/step1_{tag}_pairbest.csv</code> | 输出基准名（<code>.parquet</code> / <code>.json</code> 跟随） |
| <code>--train-frac</code> | 0.70 | train 比例 |
| <code>--split-method</code> | <code>pair-random</code> | <code>pair-random</code> 或 <code>beatmap-random</code> |
| <code>--split-seed</code> | 20260928 | 划分种子 |
| <code>--min-plays-per-pair</code> | 2 | 对记录数 < 此值时剔除（默认 = 单次对剔除） |
| <code>--max-best-acc</code> | 0.998 | 最高 ACC 严格大于此值时剔除；<code>1.0</code> 关闭 |
| <code>--acc-floor</code> | 0.90 | 最高 ACC ≤ 此值时剔除（保留 clean 规则一）；<code>0.0</code> 关闭 |
| <code>--label-shift</code> | 0.0 | 标签换成 <code>log(s + exp(loss_step0)) = log(1+s-ACC)</code>；<code>0.001</code> 即 <code>log(1.001-ACC)</code> |
| <code>--gap-days</code> | 365 | 连续 ≥ 此天数没记录 → 新玩家；<code>0</code> 关闭 |
| <code>--min-plays-per-player</code> | 20 | 玩家在 step0 全表里的记录数 < 此值时整人剔除；<code>0</code> 关闭 |
| <code>--no-parquet</code> / <code>--no-csv</code> | off | 只写另一种格式 |

<code>dcr/steps.py</code> 里还支持 <code>params.formats: both | parquet | csv</code>；10k 建议 <code>parquet</code>
（1300 万行的输入只产出约 150 万行，CSV 仍然没有必要）。

---

## 5. 实测漏斗

运行方式：

~~~bash
python main.py -c configs/exp_pairbest_1k.yaml
python main.py -c configs/exp_pairbest_10k.yaml
~~~

| 项 | 1k | 10k |
|---|---|---|
| step0 行 / 玩家 / 谱面 | 1,399,615 / 990 / 20,655 | 13,359,355 / 9,812 / 21,899 |
| step0 原始对 | 866,616 | 7,784,519 |
| 玩家门槛 < 20 条剔除（玩家 / 行） | **13 / 81** | **217 / 1,543** |
| 空窗拆分：段 / 新增段 / 受影响玩家 | **1,343 / +366 / 272** | **12,469 / +2,874 / 2,273** |
| 拆分后的对 | 870,038 | 7,816,887 |
| 单次对剔除（对 = 行） | **647,523**（74.4%） | **5,689,791**（72.8%） |
| ≥2 次的对 / 行 | 222,515 / 752,011 | 2,127,096 / 7,668,021 |
| 最高 ACC ≤ 0.90 剔除 | **912** | **13,035** |
| 最高 ACC > 0.998 剔除 | **111,897**（50.3% 于多玩对） | **600,315**（28.2%） |
| 保留对 = 输出行 | **109,706** | **1,513,746** |
| 保留占 step0 行 | **7.84%** | **11.33%** |
| 保留占拆分后的对 | 12.61% | 19.36% |
| 输出玩家 / 其中的基础玩家 | 1,039 / 942 | 10,266 / 9,368 |
| 输出谱面 | 6,828 | 16,061 |
| <code>n_plays</code> 中位 / 均值 / p90 / max | 2 / 3.47 / 6 / 111 | 2 / 3.67 / 7 / 527 |
| <code>loss</code> mean / sd / min / max | -4.5453 / 0.9413 / -6.2145 / -2.3034 | -4.2929 / 0.9167 / -6.2145 / -2.3026 |
| <code>ACC</code> min / median / mean / max | 0.9001 / 0.9897 / 0.9837 / 0.9980 | 0.9000 / 0.9853 / 0.9800 / 0.9980 |
| split train / test | 76,567 / 33,139（30.21% test） | 1,059,890 / 453,856（29.98% test） |
| 校验 / 运行 | **16/16** / 4.6 s | **16/16** / 38.3 s |

读法：

- **单次对是最大的一刀**：1k 的 870,038 个对里有 647,523 个只打过一次，占 74.4% 的对、
  46.3% 的行；10k 是 72.8% 的对。新方法的名义保留率（7.84% / 11.33% 于 step0 行）之所以
  这么低，主因是这条，而不是 ACC 窗口。
- **0.998 在多玩对里仍然删掉一半（1k）**，但 10k 只删 28.2%：10k 的玩家更多、每对练习
  更多，最高 ACC 的分布更靠下（更少「一打就 99.8%+」的对）。
- **ACC ≤ 0.90 的下限只删 0.4% / 0.6%**（912 / 222,515；13,035 / 2,127,096）：pair-best
  已经把成绩抬高，真正落在 0.90 以下的多玩对很少。它的作用主要是把输出的 ACC 下界钉在
  0.90 以上，而不是再大规模删行。
- **空窗拆分对行数影响很小**（1k 只多了 3,422 个对；10k 多了 32,368 个），但把 272 / 2,273
  个玩家拆成了更多实体；这正好是「同一个人隔了很久回来，水平不能当成一个潜向量」的
  建模需求。
- 1k 的基础玩家从 990 掉到 942（<20 门槛拿走 13 个，pair 过滤再拿走 35 个）；10k 从 9,812
  掉到 9,368（门槛 217 个，pair 过滤 227 个）。输出实体数（1,039 / 10,266）大于基础玩家数，
  是因为被拆玩家的后续段各自成为独立实体。

---

## 6. step2 结果：bias 与 mf_dot，dim=8

配置：<code>models=[bias, mf_dot]</code>、<code>dim=8</code>、<code>epochs=60</code>、<code>lr=0.02</code>、<code>wd=1e-4</code>、
<code>features=F0</code>、<code>entity=player</code>、<code>seed=0</code>、<code>device=auto</code>、<code>batch=8192</code>。
指标全部算在 <code>split=test</code> 的**对**上。

### 6.1 1k

| 模型 | RMSE | MAE | Spearman | pp_p50 | pp_p90 | pp_mean | n_scored | 覆盖率 |
|---|---|---|---|---|---|---|---|---|
| <code>bias</code> | 0.744262 | 0.593287 | 0.605454 | 0.486022 | 2.072224 | 0.844324 | 32,308 | 97.49% |
| <code>mf_dot</code> | **0.701488** | **0.529689** | **0.674891** | **0.432723** | **1.802366** | **0.746587** | 32,308 | 97.49% |

- 训练词表：**1,015 个实体 / 6,143 张谱面**；test 33,139 行，其中 32,308 行两端都在
  训练词表里（覆盖率 97.49%；实体覆盖 99.91%、谱面覆盖 97.58%）。
- <code>mf_dot</code> 相对 <code>bias</code>：RMSE **−0.0428（−5.75%）**、Spearman **+0.0694**、
  pp_mean **−0.0977**。
- <code>pp_*</code> 是把 loss 误差换算回 ACC 百分点：<code>|exp(pred)−exp(y)|·100</code>。pp_p50 = 0.43、
  pp_p90 = 1.80 表示一半的测试对预测误差在 0.43 个百分点内。

### 6.2 10k

| 模型 | RMSE | MAE | Spearman | pp_p50 | pp_p90 | pp_mean | n_scored | 覆盖率 |
|---|---|---|---|---|---|---|---|---|
| <code>bias</code> | 0.702022 | 0.567179 | 0.672263 | 0.584018 | 2.299756 | 0.953789 | 452,664 | 99.74% |
| <code>mf_dot</code> | **0.684441** | **0.550401** | **0.696620** | **0.560683** | **2.234631** | **0.925112** | 452,664 | 99.74% |

- 训练词表：**10,138 个实体 / 15,203 张谱面**；test 453,856 行，覆盖 452,664 行（99.74%；
  实体覆盖 99.97%、谱面覆盖 99.77%）。
- <code>mf_dot</code> 相对 <code>bias</code>：RMSE **−0.0176（−2.50%）**、Spearman **+0.0244**、
  pp_mean **−0.0287**。dim=8 在 10k 上的增益明显小于 1k（−5.75%）；一个自然的猜测是
  10k 的实体/谱面更多（10,138 / 15,203，1k 是 1,015 / 6,143），固定 8 维交互的容量更紧。
  这只是观察，没有做维度/容量消融，不能当作因果结论。
- 产物：<code>experiments/20261020_193513_pairbest_10k/results/step2_10k_summary_loss.json</code>、
  <code>step2_10k_{model}_F0_dim8_loss.npz</code>。

### 6.3 这些数字不能与旧 play 级数字直接比

旧 README / 文档的主表（1k bias 1.5236、mf_dot 1.4548；10k bias 1.3727、mf_dot 1.3327）
是在 **step0 的未清洗 play** 上评估的，标签支撑到 <code>loss = -12.47</code>、ACC 低至 0.23；
本节的 test 集是**每对最高成绩 + 单次对 + ACC ∈ (0.90, 0.998]** 的过滤总体，1k / 10k 的
test loss 只在 <code>[-6.2144, -2.3038]</code> / <code>[-6.2145, -2.3026]</code>。RMSE 的绝对尺度因此天然更小，
**不能用「0.75 < 1.52」说明新清洗让模型更好**。要做真正的对照，必须固定同一个 test 集
（例如都在 pair-best 的 test 上，或都回到 play 级），这不在本次运行范围内。

---

## 7. 校验（每次运行都跑，失败即 SystemExit）

| # | 断言 | 说明 |
|---|---|---|
| 1 | <code>rows_nonempty</code> | 至少保留一个对 |
| 2 | <code>one_row_per_player_beatmap</code> | 输出 <code>(player_id, beatmap_id)</code> 唯一 |
| 3 | <code>min_plays_per_pair_enforced</code> | 所有 <code>n_plays >= --min-plays-per-pair</code> |
| 4 | <code>pair_play_counts_match_input</code> | 用独立 groupby 从原始行重算 <code>n_plays</code> 并逐对比较（>2M 行时抽 5,000 对） |
| 5 | <code>kept_score_is_pair_min_loss</code> | 用独立 groupby 重算该对的最小 loss，确认输出就是最小那条 |
| 6 | <code>best_acc_at_or_below_max</code> | 所有行的 <code>loss >= log(1-max-best-acc)</code> |
| 7 | <code>min_acc_strictly_above_floor</code> | 所有行的 <code>loss < log(1-acc-floor)</code>（即 ACC 严格高于 0.90） |
| 8 | <code>player_ids_unique_per_base_segment</code> | 一个 <code>(player_id_base, player_seg)</code> 只映射到一个新 id，且新 id 两两不同 |
| 9 | <code>segment_zero_keeps_base_id</code> | 段 0 的 <code>player_id == player_id_base</code> |
| 10 | <code>later_segments_get_fresh_ids</code> | 段 >0 的新 id 全部大于 <code>max(step0 player_id)</code> |
| 11 | <code>gap_split_is_exact</code> | 段内相邻原始记录间隔全 < gap；同 base 的段边界间隔全 ≥ gap（双向） |
| 12 | <code>split_labels_valid</code> | 只有 train/test，且两者之和 = 输出行数 |
| 13 | <code>split_units_disjoint</code> | train 侧与 test 侧的划分单位（有序对或谱面）无交集 |
| 14 | <code>actual_train_frac_within_1pct</code> | 实际 train 比例与 <code>--train-frac</code> 相差 ≤ 1% |
| 15 | <code>loss_negative</code> | 所有 loss < 0 |
| 16 | <code>output_rows_are_unmodified_input_rows</code> | 输出是 step0 的逐值子集（除 <code>player_id</code> 重编号外） |

1k / 10k 都是 16/16 通过。10k 的第 4、5 项在 5,000 对上抽样复核（固定 seed）。

---

## 8. 已知局限与取舍

1. **空窗与 n_plays 都按记录算，不按真实游玩算。** step0 只含约 1/3 的真实游玩，所以
   「连续 1 年没记录」比「连续 1 年没玩」可能更严格或更宽松；单次对剔除是「只记录了一次」
   剔除，不是「只玩过一次」剔除。要按真实 playcount 判，需要 dump 的
   <code>osu_user_stats_mania.playcount</code> 或外部成绩表，本版没有做。
2. **ACC 两端过滤都用了标签本身，且作用在 test 上。** 这是「先看成绩再决定谁进 test」的
   选择（selection on the outcome）；<code>clean</code> 刻意避免这种做法（行级清洗只碰 train）。
   pairbest 无法沿用，因为观测单位已变成对、test 必须是同一种对表。后果：test 的标签
   分布被截断到 (0.90, 0.998]，模型既不在 ACC > 99.8% 的区域、也不在 ACC ≤ 0.90 的区域
   被评估；若要严格评估某条阈值本身，应在过滤前先留出对，再在留出对上看过滤/不过滤的差异。
3. **test 集不再是「未清洗的目标总体」。** 旧文档（[step2_prediction.md](step2_prediction.md)
   §2.3）把「test 用未清洗标签」当成口径；本版本的 test 是 pair-best + 单次对 +
   ACC ∈ (0.90, 0.998] 的总体。两套数字不可互换。
4. **单次对剔除有选择性。** 只打过一次的对占 74.4% / 72.8%，把它们删掉等于把总体偏向
   「反复打同一张谱面」的玩家/谱面；这些玩家/谱面通常更投入、更热门。幸存总体不是 step0
   的无偏样本。
5. **每对只留最高成绩会把「练习过程」压掉。** 这一点与旧版 play 级模型相反：旧版
   保留每条 play 的顺序与 <code>playcount_cur</code>，可以学「第几次达到什么水平」；本版只留最好
   那条，F1（<code>ln(playcount_cur)</code>）的语义只剩「最高成绩出现在第几次」，样本量也小得多。
6. **同一对跨空窗时可能被拆成两个对。** 拆分在 pair 过滤之前，所以「停 2 年、回来又打
   同一张图一次」会被认为是两个玩家各打一次，两个对都因 <code>n_plays=1</code> 被剔除。这是
   「先拆人、再看对」的直接后果；如果更希望保留这种跨期对，应把顺序改成「先压对、再拆人」，
   但那会让 pair 的「最高成绩」横跨技能断层。两种口径都说得通，本版选了前者。
7. **玩家 id 被重编号。** 段 0 保留原 id，后续段是新 id；下游若要对照 step0，用
   <code>player_id_base</code> + <code>player_seg</code>。
8. **玩家门槛是 20 条记录，不是 20 次真实游玩。** 它比 <code>clean</code> 的 200 低一个数量级，
   所以低活跃玩家仍可能贡献少量对；换 <code>--min-plays-per-player</code> 会改变幸存玩家与
   划分单位，从而改变 test 集。<code>0</code> 关闭。
9. **ACC 0.90 只作用于 pair-best 那条。** 一个对可以有低于 0.90 的历史 play，只要最高一条
   > 0.90 就保留（§2.5）。这与「先逐条过 ACC 门槛、再压 pair」不同：后者可能把该对删成
   单次对，再被单次对规则剔掉。本版选的是「保留下来的成绩必须过线」。
10. **10k 的逐对校验抽样。** 原始表 >2M 行时，<code>pair_play_counts_match_input</code> 与
    <code>kept_score_is_pair_min_loss</code> 用固定 seed 抽 5,000 对做独立 groupby 复核（报告里
    <code>verification_sample.sampled=true</code>），其余检查仍是全量。

---

## 9. 复现

~~~bash
# 走统一入口（推荐；产物进 experiments/<timestamp>_pairbest_*/）
python main.py -c configs/exp_pairbest_1k.yaml
python main.py -c configs/exp_pairbest_10k.yaml

# 单独跑 step1（输出到 data/processed/step1_{tag}_pairbest.*）
python scripts/step1_pairbest.py --tag 1k
python scripts/step1_pairbest.py --tag 10k --no-csv --min-plays-per-player 200

# 消融：关掉某条规则
python scripts/step1_pairbest.py --tag 1k --gap-days 0            # 不拆空窗
python scripts/step1_pairbest.py --tag 1k --max-best-acc 1.0      # 不删高 ACC
python scripts/step1_pairbest.py --tag 1k --acc-floor 0.0         # 不删低 ACC
python scripts/step1_pairbest.py --tag 1k --min-plays-per-pair 1  # 保留单次对
~~~

对应的实验目录（本次运行）：

| tag | 实验目录 | step1 产物 | step2 产物 |
|---|---|---|---|
| 1k | <code>experiments/20261020_193449_pairbest_1k</code> | <code>artifacts/step1_1k_pairbest.parquet</code> | <code>results/step2_1k_summary_loss.json</code> |
| 10k | <code>experiments/20261020_193513_pairbest_10k</code> | <code>artifacts/step1_10k_pairbest.parquet</code> | <code>results/step2_10k_summary_loss.json</code> |

---

## 10. 与 step1/clean 的对照

| 项 | <code>clean</code> | <code>pairbest</code> |
|---|---|---|
| 观测单位 | play | (玩家段, 谱面) 对 |
| 输出行 | = step0 全部行（含 dropped） | 只含幸存对（一行一对） |
| 玩家 id | 原样 | 按 1 年空窗重编号（段 0 保留原 id） |
| 每对保留 | 全部 play | 只留最高成绩 |
| 单次对 | 保留（只受行级规则影响） | 剔除 |
| 玩家门槛 | 200 条记录 | 20 条记录 |
| ACC 范围 | 行级 ACC > 0.90；无上限 | pair-best ACC ∈ (0.90, 0.998] |
| 清洗作用域 | train 侧行级；test 原样 | 全表总体过滤；test 同规则 |
| 划分 | 先划分，再清洗 train | 先清洗，再划分 |
| test 集 | 未清洗的 play | 过滤后的 pair-best |
| 产物 | <code>step1_{tag}.*</code> | <code>step1_{tag}_pairbest.*</code> |
| 典型规模（1k） | 1,399,615 行，train 671,105 | 109,706 行，train 76,567 |

**不要把两套的数字放进同一张「模型对比」表。** 它们的 test 集、标签支撑和观测单位都不同；
唯一安全的比较是固定其中一套总体（通常是 pairbest 的 test 对）再比较模型，或反过来把
旧模型在新 test 上重新打分。
