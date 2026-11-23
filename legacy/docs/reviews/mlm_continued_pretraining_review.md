# MLM 继续预训练的方法论反思与检验

> 2026-11-23。范围：CM3P beatmap tower 上的 MLM 继续预训练（mania 列 token 修复 + 下游难度）。
> 不讨论其它自监督目标。本文的“检验”是**离线结构检查**：parser / tokenizer / 数据统计 / 读出代码；
> 没有训练模型。模型层面的结论需要 HPC 实验验证。样本量在各节标注。

## 0. 结论

1. **MLM 适合做“工程修复”，不适合单独做“列/难度有用”的科学证明。** 它能把
   `[MANIA_COLUMN_*]` 从 OOV 行训到可用；但下游是否受益取决于 mask、数据划分、
   读出路径和对照，不是“训了列”自动成立。
2. **三个已经测到的硬问题：**
   - column-only / random masking 有严重同事件泄漏：49.6% 的被 mask note 处于 partial chord，
     47.9% 的 hold 只 mask 了 start/end 之一；
   - 20% chart holdout 里 59.6% 的图与预训练图共享 beatmapset（song-level 泄漏）；
   - `max_length=2000` 的截断集中在硬图（median star 7.03 vs 全体 4.70）。
3. **event-span 40% 干净得多**：partial chord 1.8%，hold split 0.2%。
4. **下游读出不匹配**：MLM 只监督 token；图表向量是 CLS + 冻结 projection。
   下游 null 不能直接解释为“列没用”，下游 positive 也可能来自 token→CLS 的间接传播。
5. **最小合法协议**：beatmapset-disjoint holdout、event-span mask、
   column-off / shuffled-column / column-ablation 三个控制、CLS 与 mean-pooled 两套读出、
   2–3 个 model seeds、2000/4000 都报。

### 检验一览

| 检验 | 样本 | 结果 | 影响 |
|---|---|---|---|
| POS_X 是否泄漏列 | parser 代码 | mania 不发位置 token | 排除该假设 |
| 和弦顺序的列信息 | 300 / 800 图 | 查表 32.7%（单 note 28.4%，4-note 45.8%） | 列部分可猜，单 note 真盲 |
| column-only 同事件泄漏 | 200 图 | partial chord 49.6%，hold 47.9% | 不能作为列测试 |
| event-span 泄漏 | 200 图 | partial chord 1.8%，hold 0.2% | 主 mask |
| song-level 泄漏 | 6,828 图 | 59.6% 共享 beatmapset | 用 set-disjoint |
| 截断偏差 | 2,000 图 | 0.53% 窗，硬图 star 7.03 vs 4.70 | 2000/4000 都报 |
| 训练池分布 | 250/组 | NPS 13.21 vs 7.43 | 硬图不足 |

## 1. 两个问题要分开

| 问题 | 需要的证据 | MLM 够不够 |
|---|---|---|
| 工程：OOV 列行能否训到可用 | 列行 norm、列 top-1、列 ablation | 够 |
| 科学：列/上下文对难度有没有用 | 下游 rank-NLL + 控制 + 读出匹配 | 不够 |

“列训练失败”和“列训练成功但下游没用上”是两种完全不同的失败；实验设计必须能区分它们。

## 2. CM3P 的 MLM 实际在优化什么

- 输入：16 s 窗口 token 流，`[CLS]` + `[BOS]` + groups。
- 随机 mask 25% 非特殊 token；80% `[MASK]` / 10% random / 10% keep；
  labels 原始 id，其余 -100（[mmrs_dataset.py:195](../external/cm3p/utils/mmrs_dataset.py#L195)、
  [default.yaml:154](../external/cm3p/configs/train/default.yaml#L154)）。
- head：Linear 768→768 + GELU + LayerNorm；decoder 768→3167，**与 input embedding tie**
  （[modeling_cm3p.py:1229](../external/cm3p/cm3p/modeling_cm3p.py#L1229)、
  [:1244](../external/cm3p/cm3p/modeling_cm3p.py#L1244)）。
- `sparse_prediction=True`：只在 masked 位置算 logits
  （[:1349](../external/cm3p/cm3p/modeling_cm3p.py#L1349)）。
- loss：token-level CE。**没有 CLS / chart-level 项。**
- CM3P 原预训练就是 contrastive + 0.5 × MLM（[modeling_cm3p.py:995-996](../external/cm3p/cm3p/modeling_cm3p.py#L995-L996)），
  encoder 已经见过 MLM；released checkpoint 只是移除了 decoder head。继续 MLM 不是新任务，
  真正的 OOV 是列行。

**一句话：MLM 优化的是“局部 token 好不好猜”，不是“图表向量好不好用”。**

## 3. 检验 1：列到底缺不缺？

### 3.1 排除 POS_X 泄漏

一个自然的怀疑是：mania 的 x 坐标就是列，tokenizer 会不会已经通过 `[POS_X_*]` 泄漏了列？

- [parsing_cm3p.py:497-501](../external/cm3p/cm3p/parsing_cm3p.py#L497-L501)：只有
  `beatmap.mode in [0, 2]` 才写 `group.x/y`；`mode == 3` 只写 `group.column`。
- tokenizer 只在 `group.x is not None and group.y is not None` 时发
  `[POS_X_*]` / `[POS_Y_*]`。
- 结论：mania **不会**发位置 token，POS_X 不泄漏列。这一假设可以排除。

### 3.2 和弦顺序的隐式信息

列 token 确实不存在，但“列信息”不一定完全不存在：和弦内的 note 顺序可能泄漏列。

300 张 pairbest 图（735,812 notes）的统计：

| 量 | 值 |
|---|---:|
| 和弦内 note 占比 | 58.0% |
| 单 note 占比 | 42.0% |
| 和弦顺序单调（asc + desc） | 90.8% |

再用一个只看 `(chord size, position in chord)` 的查表预测列
（800 张图，401,914 个 test notes）：

| 子集 | top-1 |
|---|---:|
| 全部 | 32.7% |
| 单 note | 28.4% |
| 和弦 note | 35.6% |
| chord size 2 | 33.2% |
| chord size 3 | 39.6% |
| chord size 4 | 45.8% |

chance 是 25%。结论：

- **单 note 基本无列信息**（28.4% ≈ chance）；这部分占 42%，是显式列 token 的真正价值所在。
- **和弦有部分隐式信息**，尤其 4-note chord（45.8%）；带 timing/pattern 上下文的 Transformer
  只会更高。
- 所以“column-blind”在输入层面成立，但“列完全不可得”不成立；列 token 的增量主要来自
  单 note 和部分和弦。

### 3.3 OOV 证据的强度

现有证据是 norm：column 行 1.042 ≈ unused 1.040（KS p=0.891）；decoder 行/bias 也接近
unused（p=0.141/0.396），见 [results/cm3p_column_probe/README.md](../results/cm3p_column_probe/README.md)。

这是“未训练”的强证据，但不是功能证据：norm 低不等于模型一定不用列。需要 §8.3 的
ablation / shuffle 才能下结论。

## 4. 检验 2：masking 是否真的在测列？

### 4.1 column-only / random 的同事件泄漏

200 张 pairbest 图、16 s 窗的模拟；hold 泄漏另在 200 张图上模拟（67,681 个 hold）：

| mask 方案 | 被 mask note 处于 partial chord | hold 只 mask 一端 |
|---|---:|---:|
| column-only 40% | **49.6%** | **47.9%** |
| random token 25% | ~42%（按 chord size 加权） | 37.5%（=2p(1-p)） |
| event-span 40% | **1.8%** | **0.2%** |

- 列 token 在 hold 的 start 和 end 各出现一次；只 mask 一个，另一个直接给出答案。
  column-only 40% 下 47.9% 的 hold 是这种情况。
- column-only 40% 下近一半的被 mask note 在和弦里有未 mask 的邻居；邻居的列直接泄漏。
- random 25% 同理，只是比例低一些。

**结论：column-only / random token 的“列 top-1”很大程度是抄同事件/邻居，不能作为
“列可学”的证据。**

### 4.2 event-span 的干净程度

event-span 40%（连续 40% 的 note event，整段所有 token mask）：

- masked event 39.8%；
- partial chord 1.8%；
- hold start/end 被 span 切开 0.2%。

这是目前唯一适合“列/上下文学习”主张的 mask。仍需把 span 边界对齐到 chord/hold，
或至少报告 partial 比例。

### 4.3 建议

- 主实验：event-span 40%，按 chord 边界切；hold 的两端要么都在 span 内、要么都在外
  （或整 hold mask）。
- column-only 只作为“能不能从上下文猜列”的诊断，且必须整 chord / 整 hold mask；
  否则不报。
- 报告 mask 覆盖率、partial chord / hold 比例。

## 5. 检验 3：下游读出匹配吗？

CM3P 的图表向量是 `beatmap_projection(CLS)`：

- `pooled_output = last_hidden_state[:, 0]`
  （[modeling_cm3p.py:632](../external/cm3p/cm3p/modeling_cm3p.py#L632)）；
- `beatmap_projection`（[:761](../external/cm3p/cm3p/modeling_cm3p.py#L761)）；
- `beatmap_embeds = projection(pooler_output)`
  （[:958-959](../external/cm3p/cm3p/modeling_cm3p.py#L958-L959)）。

而 `CM3PForMaskedLM` 只有 encoder + head/decoder；**没有 projection，也没有 CLS 目标**。
MLM 的梯度只从 token 位置来，CLS 只有间接梯度，`beatmap_projection` 完全冻结。

后果：

- 即使 MLM 把 token 表示训好了，CLS + 冻结 projection 也可能不反映；下游 null 不能直接
  解释为“列没用”。
- 反过来，下游 positive 也可能只是 token→CLS 的间接传播，不能证明列被用。
- 原 CM3P 的 512d 几何是 contrastive 目标塑造的；MLM-only 继续训练没有 chart-level 项，
  可能漂移。

建议同时报两套读出：

1. **CLS + frozen projection**（原口径，和现有 CM3P 表可比）；
2. **mean-pooled token states + 新 linear probe**（直接测 token 表示）；
3. 可选：fine-tune projection / CLS（测适配上限，但不是 frozen probe）。

如果 mean-pooled 有 gain 而 CLS 没有，问题在读出，不在列。

## 6. 检验 4：数据划分的 song-level 泄漏

- 6,828 张 pairbest 图只有 3,810 个 beatmapset（约 1.8 图/set）。
- 随机 20% chart holdout（1,365 图）：**59.6%** 的 holdout 图与预训练图共享 beatmapset；
  27.4% 的 holdout 图在 holdout 内有同 set 的兄弟图。
- 同一首歌的不同难度共享 BPM、节奏、结构；MLM 的 masked prediction 会变容易，下游也可能
  被抬高。

建议 **beatmapset-disjoint holdout**：

- 20% sets（762 / 3,810）→ 1,366 图（20.0%）；
- star 4.76 vs 全体 4.73，难度平衡；
- pairbest rows 22,220（20.3%）；
- rank 评测的 pairs 会降到约 4%（两张图都在 holdout），必须报告 coverage。

重复 checksum：21,949 / 6,828 都是 0 重复组；没有精确重复图，但 set 泄漏仍在。

## 7. 检验 5：截断与分布

2,000 张 pairbest、23,468 个 16 s 窗，column-on：

| max_length | 截断窗 | 截断图 | 备注 |
|---:|---:|---:|---|
| 2000 | 125（0.53%） | 37 | 截断图 median star **7.03**，全体 **4.70** |
| 4000 | 0 | 0 | maxlen p99 2,144，max 3,202 |

- 截断从尾部删 token（`longest_first`），硬图的尾段（往往最难）被删；这是对硬图的
  结构性偏差。
- 但 checkpoint 的 processor 配置是 `max_length=2000`（上游训练 yaml 写 4000；以真实 HF 配置为准）；
  用 4000 可能是位置 OOD（RoPE 可能外推，但没训练过）。
- 建议 2000 和 4000 都跑/都报；2000 下报告截断率与“排除截断图”的敏感性；不要静默用 4000。

训练池分布（250 张/组）：

| 组 | star 均值 | notes 中位 | NPS 均值 |
|---|---:|---:|---:|
| pairbest（评测） | 4.73 | 2,030 | 13.21 |
| 非 pairbest（补规模） | 2.66 | 817 | 7.43 |

MLM 在补规模池上会偏向简单图；硬图覆盖不足。要按 star 分层采样/加权，
或把 pairbest 的 80% 也算进训练池。

## 8. 检验 6：目标与优化的 confound

### 8.1 token 类型不平衡

200 张 pairbest 图的窗口：列 token 占 **28.7%**（p50 248 / 843）。
random 25% 下只有约 **7%** 的 token 是 masked column；event-span 40% 下约 **40%** 的列
被 mask。

如果列预测更难，它的梯度份额小；MLM 可能主要学时间/事件类型，列行学得慢。
建议按 token 类型分开报 loss/accuracy；必要时列加权，或 column-only（整 chord）。

### 8.2 80/10/10 与随机替换

- 10% keep-original：列 token 有 10% 直接可见；报列准确率时只报 80% `[MASK]` 子集，
  否则虚高。
- 10% random replacement：CM3P 的 `eligible_random_token_ids` 只排除了 `[AUDIO]`，
  没有排除 `[PAD]/[MASK]/[CLS]` 等特殊 token
  （[mmrs_dataset.py:174-182](../external/cm3p/utils/mmrs_dataset.py#L174-L182)）；
  而且可能把 `[CIRCLE]` 换成 `[TIME_SHIFT]`，构造非法谱面。
  建议 same-type replacement 或 100% `[MASK]` 做消融。

### 8.3 缺少的控制

| 控制 | 做法 | 回答 |
|---|---|---|
| MLM column-off | 同数据/同 mask/同预算，只是不发列 token | 增益是“列”还是“多训了 20k 图” |
| shuffled-column | 列值在窗口内随机打乱 | 模型是否在用列的**值** |
| column ablation | 推理时列 token 换 `[UNK]` / 固定列 | 模型是否真的用列 |
| model seeds | 2–3 个训练种子 | MLM 方差是否大于效应 |

现有 P4 只变划分种子、model seed 固定 0；MLM 训练方差可能大于效应。
没有 column-off，任何 column-on 的增益都不能归因于列。

### 8.4 灾难性遗忘

22 层、244K 预训练、20k 继续训练；低 lr + 冻底 12 层；早停看 held-out MLM 和下游。
不要用大 lr 把原表示洗掉。

## 9. 一个最小合法协议

| 项 | 设置 |
|---|---|
| 数据 | beatmapset-disjoint 20% sets holdout；训练池 = 15,121 补规模 + 80% pairbest sets |
| tokenizer | canon、column patch、max_length 2000 与 4000 两套（2000 可比，4000 覆盖敏感性）；列分布 1–4 各 ~25% 已验证 |
| mask | event-span 40%，chord/hold 边界对齐 |
| 控制 | column-off、shuffled-column、column ablation |
| 模型 | CM3P MLM head、tied decoder、列行重初始化、冻底 12 层、lr 1e-5/1e-4 |
| 读出 | CLS+frozen projection；mean-pooled + 新 linear probe |
| 指标 | MLM loss by token type；列 top-1（single/chord/hold、只算 [MASK] 子集）；查表 baseline；列 norm；下游 rank-NLL；coverage/truncation |
| seeds | 2–3 model × 5 split；paired Δ |
| 预算 | A800 1–2 天/run（冻底更快） |

## 10. 结论与开放问题

- MLM 继续预训练是合理的“工程修复”，但不是干净的“列/难度”科学实验。
- 在跑模型之前，先接受三个离线事实：set 泄漏、column-only 泄漏、截断偏差。
  它们会直接改变 mask / 数据 / 读出的设计。
- 如果目标只是“把 OOV 行训到可用”，event-span + column-off 控制 + 列 ablation 足够；
  如果目标是“列对难度有用”，还需要下游读出匹配，并接受 MLM 只是 token-level 目标。

开放问题：

1. 单 note 的列信息是否真的能提升难度？需要按单 note / 和弦分层看下游。
2. 是否需要一个 chart-level（CLS）目标来训练图表向量？MLM 本身不提供。
3. 2000 vs 4000 的 OOD 影响有多大？RoPE 外推是否可靠？
4. 80/10/10 vs 100% [MASK] vs same-type replacement，哪个更适合结构化谱面？
5. set-disjoint 后 pairs 覆盖约 4%，统计功效够不够？要不要 5 折 set-disjoint？

## 11. 这些检验没有覆盖什么

- **没有训练模型**：列到底能不能被 MLM 学到、学到后下游是否受益，仍未验证。
- 查表 baseline 是**下界**，不是 Transformer 的实际列准确率；模型有 timing/pattern 上下文，
  只会更高。
- mask 模拟只算**结构泄漏**，不算注意力能从上下文补多少；event-span 的 1.8% 是“同事件
  直接泄漏”，不是“列完全不可猜”。
- set / 截断 / 分布统计是**抽样**；全量结论要重跑。
- 读出 mismatch 是**代码层论证**，需要下游实验确认。
- 没有覆盖 OD/HP、7K、audio-on、其它 keycount；这些不在本文范围。

## 附：离线检查的复现

- 解析/统计：`dcr/cm3p_offline.py`（`parse_osu` / `chart_groups` /
  `window_token_lengths` / `mania_column`）。
- 数据：`data/osu_cache/<beatmap_id>.osu`、`data/processed/beatmap_meta_1k.csv`、
  `data/processed/step1_1k_pairbest.parquet`。
- 样本量：各节标注；随机种子固定，可复现。
- 未训练任何模型；所有结论是 parser / tokenizer / 数据 / 读出代码层面的结构或统计结果。
