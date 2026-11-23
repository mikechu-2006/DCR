# data2vec 预训练用于 DCR：可行性、tokenizer 与实验计划

> 2026-11-22，讨论稿（未实现）。依据：data2vec（arXiv:2202.03555）与 DCR 现有 CM3P 管线。
> 相关：[models.md](models.md)、[cm3p_rank.md](cm3p_rank.md)、[cm3p_crop_windows.md](cm3p_crop_windows.md)、
> [../results/cm3p_column_probe/README.md](../results/cm3p_column_probe/README.md)。

## 0. TL;DR

1. **可行，而且 DCR 有一个非常适合 data2vec 的缺口。** 冻结的 CM3P checkpoint 从未训练过
   `[MANIA_COLUMN_*]` 行：column 行 norm 1.042 ≈ 未使用行 1.040（KS p=0.891），
   所以“直接打开 column token”的 P4 探针里，线性方法 5/5 个种子变差（+0.00194 NLL）。这不等于“列没用”，
   而是“列的 embedding 没被训练过”。data2vec 的继续预训练正好可以训练它。
2. **两条路线**：A（CM3P 兼容：event tokens + ModernBERT + column patch + data2vec 继续预训练，
   低风险，先做）；B（mania 原生 **audio-style 连续帧特征**：piano roll → 1D conv → span mask，
   不是 8-bit 离散码；干净，主赌注）。
3. **核心实验是 2×2**：{column off, column on} × {MLM 继续预训练, data2vec}，
   外加 from-scratch audio-style。先把“列可训练”与“目标函数”两个因素分开。
4. **数据划分不能直接用 15,121。** 21,949 张白名单图里 6,828 张是 pairbest 评测图，
   但剩下的 15,121 张明显更简单（star 2.66 vs 4.73，notes 817 vs 2,030）——只在这批图上预训练
   会缺高难图。默认用“pairbest 分层 80/20 holdout + 15,121 补规模”，只在 20% holdout 上评测。
5. **成功标准不是打败 CM3P+external（P4 0.594）**，而是：P4 NLL 比 CM3P-only（0.6305）
   低 ≥0.005 且 4/5 个划分种子成立，并与 MSD/Sunny 互补。只打平就不值得继续。

## 1. 现状：CM3P 给了什么，缺什么

### 1.1 已有的内容塔

| 项 | 值 |
|---|---|
| 向量 | 冻结 512 维，单位化；逐窗 → 图均值 |
| backbone | ModernBERT 22 层 / 768d / 12 heads，vocab 3167，max 8192 |
| 预训练 | 244K 图，contrastive + metadata + 可选 MLM；发布 checkpoint `OliBomby/CM3P` |
| 本地口径 | audio off、canon crop、16 s / 16 s 窗口、max_length 2000 |
| 覆盖 | 21,949 张白名单图；pairbest 6,828 张 100% |

P4（新玩家 + 新图，5 seeds，model seed 0）基线：

| 方法 | test NLL | 备注 |
|---|---:|---|
| CM3P 线性 | 0.63050 ± 0.00270 | column-blind |
| CM3P MLP h64 L2 | 0.63013 ± 0.00266 | column-blind |
| CM3P + external 线性 | 0.60335 ± 0.00515 | MSD+Sunny |
| CM3P + external MLP | 0.59425 ± 0.00413 | 当前最强 |
| ceiling | 0.56540 ± 0.00859 | 5 seeds |

P2（chart-disjoint）：CM3P-only 线性 NLL 0.6148、captured 0.6006；CM3P+external 0.5756 / 0.9013。

### 1.2 缺什么

* **column-blind**：上游 parser 写 `Group.column`、tokenizer 读 `Group.mania_column`，
  从未发出列 token。P4 的 column-on 探针说明：冻结模型喂 OOV 列行只会引入噪声。
* **目标不是难度**：CM3P 的对比目标是“谱面 ↔ 元数据对齐”，MLM 预测的是低层离散 token
  （time/column/hitsound），不是“这段有多难”。
* **没有全局难度/耐力/长度**：16 s 窗口 + 均值池化，长曲的 stamina 只能靠下游聚合。

## 2. data2vec 与 DCR 的对应关系

| data2vec | DCR |
|---|---|
| 输入序列 | 一张图的 token 序列（16 s 窗口） |
| student / teacher | masked 窗口 / 完整窗口 |
| 教师参数 | 学生参数的 EMA（stop-grad） |
| 目标 | teacher 顶层 K 层 hidden state，逐 token LayerNorm 后平均 |
| loss | 只在 masked 位置算 Smooth-L1 |
| 下游 | 池化成图向量 → `d_m`（rank-NLL）或内容塔（loss 回归） |

**为什么可能优于 CM3P 的 MLM**：连续、上下文化的目标不受 3167 词表限制；不用 decoder head；
EMA teacher 提供稳定目标；不需要负样本，batch 可以小。**风险**：掩码重建学到的是“谱面长什么样”，
不一定是“有多难”；20k 图规模也远小于 CM3P 的 244K。

## 3. Tokenizer

### 3.1 路线 A：CM3P-compatible event tokens

* 直接复用 `CM3PBeatmapTokenizer` + `dcr/cm3p_windows.py::install_mania_column_patch()`
  （修 `Group.mania_column` 属性名和 0/1-based）。
* canon crop（hitsounds/kiai/SV 关）、`add_snapping=False`、16 s 窗口、max_length 4000
  （在 6,828 张 pairbest 图上，column-on + max_length 2000 截断了 150 图 / 434 窗；
  4000 更安全，模型 RoPE 支持 8192）。
* 每个 tap ≈ `[CIRCLE] [TIME_SHIFT_t] [MANIA_COLUMN_c]`（3–4 token）；hold 多一个
  `[HOLD_NOTE_END]`。**实测**（300 张随机图、3,004 个 16 s 窗、canon + add_snapping=False + CLS）：
  column off 每窗 p50 417 / p90 737 / p99 1,053 / max 1,377；column on 为 578 / 1,044 / 1,525 / 2,012。
  max_length 2000 时 column-on 只有 1/3,004 个窗截断，4000 时 0。
* **masking**：按时间选 2–4 段 0.5–2 s 的 span，覆盖 30–50% 的 note event；
  把 span 内所有 event 的所有 token 换成 `[MASK]`，loss 只算这些位置。
  * 副作用：3 token/note 会泄漏“这段有几颗 note”。可接受的 v1；
    更干净的做法是每个 event 压成一个 `[MASK]`，teacher 侧对 event 内 token 取均值当目标（v2）。
  * 也可以只 mask 属性 token（保留 `[CIRCLE]`/时间，mask 列），专门逼列预测；作为消融。
* **初始化**：加载 CM3P beatmap tower；`[MANIA_COLUMN_*]` 行重新初始化
  （used-token 均值 + 小噪声），因为它们在 checkpoint 里是 OOV。

优点：复用 checkpoint/processor，和现有 CM3P 表可直接对比；缺点：列是 OOV、token/event 数不定、
有 note 数泄漏。

### 3.2 路线 B：mania 原生连续帧特征（audio-style）

这条路线直接照搬 data2vec-audio 的输入侧，**不做离散词表**：

* **输入信号**：把谱面写成 100 Hz（10 ms）的 4 列 piano roll，8 个通道
  = 4 列 onset + 4 列 hold 状态；onset 用高斯核（σ≈8–10 ms）软化，避免 10 ms 网格
  把相邻 note 合并。可选再加 hold-end / 密度通道。
* **特征编码器**：学生/教师共享的 1D CNN（kernel 3–5，stride 2 → 50 Hz，512 维），
  对应 data2vec-audio 的 7 层 conv 把 16 kHz 波形压到 50 Hz。输出是连续向量，不是 256 个码。
* **masking**：在 50 Hz 特征上做 span mask（起点 p≈0.065、长度 10 帧 ≈200 ms，约 49% 帧；
  20–40 帧做消融），masked 帧替换成**可学习的 mask 向量**；teacher 看未 mask 的特征。
* **为什么不是 8-bit code**：4 列 × 2 bit 每帧只能给每列一个状态，装不下“同一列一帧内
  hold-end + tap”这类事件，也丢 velocity/hitsound 和 sub-frame timing；25 ms 会把高 BPM 的
  1/8、grace note、轻微错位的和弦合并。piano roll + conv 让模型自己学局部模式，没有词表瓶颈。
* 窗口 16 s = 800 个 50 Hz 帧（32 s = 1600）；整曲靠窗口 + 池化。
* 空帧问题：谱面比音频稀疏，但空帧就像静音；卷积仍能看到 onset 周围的局部结构。
  如果担心，按 note 密度采样 mask span，或 loss 只算非空帧（消融）。

与 data2vec-audio 的逐项对应：

| data2vec-audio | 谱面 |
|---|---|
| 16 kHz waveform | 100 Hz × 8 ch piano roll |
| 7×Conv1d，stride 320 → 50 Hz | Conv1d stack，stride 2 → 50 Hz |
| 512 维连续特征 | 512 维连续特征 |
| span mask，p=.065 × 10 帧 ≈49% | 同左，span 长度做消融 |
| 可学习 mask 向量 | 同左 |
| feature encoder 师生共享 | conv 师生共享 |
| target 用 instance norm | IN / LN 做消融 |

优点：与 data2vec-audio 同构、无 OOV/词表设计、列/和弦/hold/密度天然编码；
缺点：没有 CM3P 预训练初始化（可拿 CM3P 的 Transformer 权重热启动，conv 是新的）、
4K 专用、10 ms 网格仍有量化但比 25 ms/8-bit 轻得多。

### 3.3 建议

先做 A 的 2×2（复用 CM3P，最快回答“列能不能被训练出来”），同时做 B 的小规模 from-scratch
作为对照。如果 A 的 frozen probe 明显弱于 CM3P，再决定是否把 B 作为主跑。

## 4. 预训练目标与训练细节

* teacher = 学生的 EMA，`Δ ← τΔ + (1−τ)θ`；τ 0.999 → 0.9999，前 30k step 线性；
   teacher 前向 `no_grad`，fp32 EMA master weights。
* student 看 masked 窗口，teacher 看完整窗口；top-K = 4–8（22 层里取顶层）；
  每个 token 先 LayerNorm，再对 K 层取平均；Smooth-L1 β=1–2；loss 只在 masked 位置，
  按 `1/sqrt(dim)` 缩放（对齐 fairseq data2vec 的实现）。
* student 侧加一个 768→768 线性投影头（teacher 目标不经过它）——BYOL 式非对称。
* AdamW，lr 2e-5–1e-4（继续预训练取小），wd 0.01，warmup 5%，cosine，clip 1.0，bf16；
  batch 16–32 个窗口；22 层 × 2000 token × (student+teacher) 需要 gradient checkpointing。
* **防塌**：LayerNorm target + EMA + mask 视角；监控 `target_var` / `pred_var`，
  5k step 后低于 0.1 / 0.01 直接停（照抄 `data2vec_audio.py` 的做法）。
  谱面 token 熵低，塌缩不是主要风险；主要风险是“掩码太好猜”，所以 span 要够长、
  要覆盖 note 密集区。
* 实现：不必搬 fairseq；约 300 行 PyTorch 即可（EMA、mask collator、top-K 取层、Smooth-L1）。
  CM3P 的 remote model 需要能返回 hidden states；不能的话包一层 forward hook。

## 5. 数据与划分（最关键）

| 集合 | 数量 | star 均值 | notes 中位 | 用途 |
|---|---:|---:|---:|---|
| 白名单全部 | 21,949 | — | — | transductive 上界 |
| pairbest 评测图 | 6,828 | 4.73 | 2,030 | P2/P4 的 chart 全集 |
| 非 pairbest 图 | 15,121 | **2.66** | **817** | 只能补“简单图”，不能单独当预训练池 |

**分布不匹配**：非 pairbest 图明显更简单（star 2.66 vs 4.73，notes 817 vs 2,030）。
只在这 15,121 张上预训练，编码器会缺少高难图的密度/和弦/jack 模式，而 P4 评测的正是这批图。

推荐协议（按优先级）：

1. **固定 chart holdout（默认）**：从 6,828 张 pairbest 里按 star 分层抽 20% 作为
   encoder-eval（约 1,366 张）；预训练池 = 15,121 + 其余 80%（约 5,462 张），共约 20,583 张。
   只在 holdout 图上做 P2/P4（先在 step1 表上过滤 `beatmap_id in holdout` 再跑 rank runner；
   pairs 会变少，报告 coverage）。干净且覆盖高难图。预算允许就换 3 个 holdout 种子重复
   （每次重训 encoder）。15,121 只用来补规模；如果担心简单图淹没高难图，按 star 分层配额采样。
2. **transductive 上界**：全 21,949 预训练（评测图见过）。只用于判断
   “如果连见过都赢不了，就别做 inductive 了”。
3. **外部语料**：抓 osu! mania 4K 全量图，覆盖全难度段；最干净，但多一步数据工程。
4. **不推荐**：只用 15,121 预训练就声称 P4 泛化——难度分布不匹配。

- 只要预训练池包含 6,828 张 pairbest 图，P2/P4 的 test chart 就一定被见过；
  所以“全 21,949 预训练”只能当 transductive 上界，不能当泛化结论。
- 外部语料（244K 图的 .osu）目前没有；CM3P-Embeddings-244K 只有向量。要扩规模得另抓
  osu! mania 图，先不做。

## 6. 下游接入

1. **新 step**：`scripts/step0i_d2v_embed.py`（名字待定）输出与 CM3P 表同契约的 parquet：
   `beatmap_id, embedding, n_windows, checksum_ok, source, d2v_rev`。
2. **rank-NLL**：直接复用 `scripts/step2_cm3p_rank.py --content-path <new.parquet>`，
   先跑 frozen linear（M0）和 MLP（M1），P2 + P4，5 seeds。
3. **fine-tune**：新脚本，解冻后 4–6 层 + attention pooling，用同一套 pair counts 的 BT NLL；
   lr 1e-5；chart-disjoint 的 train 行训练，player-disjoint val 早停。
4. **loss 回归**：`step2_predict.py` 目前只接受 `--chart-content cm3p`；加一个 `d2v` 分支
   或把 `--content-path` 泛化。内容塔 + 玩家向量点积；OD/HP/keycount 作为标量特征接进去
   （CM3P 词表没有 OD，见 [models.md](models.md) §3.5）。
5. **互补性**：与 CM3P、MSD、Sunny 做线性 blend，报告 paired ΔNLL；这是最现实的收益点。

## 7. 实验矩阵

| ID | 预训练 | tokenizer | column | 下游 | 回答的问题 |
|---|---|---|---|---|---|
| E0 | 无（冻结 CM3P） | CM3P | off | frozen linear | 基线 0.6305 |
| E1 | 无（冻结 CM3P） | CM3P | on | frozen linear | 已有负结果 0.6324 |
| E2 | MLM 继续预训练 | CM3P | on | frozen linear | 训练列行能否修好 E1 |
| E3 | data2vec | CM3P | off | frozen linear | 目标函数单独的增益 |
| **E4** | **data2vec** | **CM3P** | **on** | **frozen linear** | **主实验** |
| E5 | data2vec | audio-style conv | on | frozen linear | 输入侧的影响 |
| E6 | E4 + 解冻 | CM3P | on | fine-tune rank-NLL | 适配后的上限 |
| E6b | 无 | CM3P | on | fine-tune rank-NLL | “不预训练只微调”对照 |
| E7 | E4 | CM3P | on | loss 回归 mf_dot | 第二条评价轴 |
| E8 | E4 + CM3P/external | — | — | blend | 互补性 |

核心是 E0–E4 的 2×2：**先把“列可训练”与“data2vec 目标”拆开**。如果 E2 已经追平 E4，
那主要贡献是列；如果 E2 没效果而 E4 有效，才是 data2vec 的功劳。
E2 需要给 CM3P 加一个 2 层 decoder head（发布 checkpoint 的 `has_decoder_head=False`）；
data2vec 不需要 decoder，这也是它工程上更轻的原因之一。

指标：P4 test NLL（主）、captured、Pearson(d,b)、P2 NLL；5 个划分种子、固定 model seed 0；
报告 paired ΔNLL 和 seeds 变好数。诊断：column token 频率/norm、target_var、截断率。

## 8. 里程碑与算力

| 阶段 | 内容 | 产出 | 量级 |
|---|---|---|---|
| M0 | token 导出 + 审计（列、截断、span 覆盖） | `data/processed/d2v_tokens/` | CPU 1 天 |
| M1 | 小规模子集（约 2,000 图）data2vec 烟测 | loss/target_var 曲线 | 1 GPU < 1 天 |
| M2 | §5 的 80/20 holdout 预训练 + frozen probe | E4 vs E0/E2 | A800 1–2 天 |
| M3 | 2×2 + audio-style + fine-tune | E0–E7 | 1–2 周 |
| M4 | blend / loss 回归 / 报告 | E8 | 3–5 天 |

## 9. 风险与 kill criteria

* **列训练不出来**：E2/E4 都不比 E0 好 → 冻结 CM3P 的 22 层对 15k 图的学习率/容量不够，
  转 from-scratch audio-style（B）或全量 21,949 transductive。
* **掩码太好猜**：loss 一开始就接近 0、target_var 正常但 probe 很差 → 加长 span、提高 mask ratio、
  按 note 密度采样，或改成 event 压缩。
* **重建≠难度**：probe 线性可解但 NLL 不降 → 需要 difficulty-aware 辅助（MinaCalc 区间难度
  蒸馏、rank-NLL 微调），而不是继续加 data2vec。
* **规模/分布**：15k 图对 22 层模型偏小，且非 pairbest 图更简单（star 2.66 vs 4.73）；
  只在这批图上训练会缺高难图。先按 §5 的 holdout 协议，不行再试 transductive 或外部语料。
* **评测噪声**：5 seeds 的 paired Δ 比 captured 绝对值可靠；不要用单 seed 下结论。

## 10. 开放问题

1. 16 s 窗口够不够表达 stamina/长度？要不要 32 s 或层次化（窗口级再做一层 data2vec）？
2. 要不要加一个 CLS 级 data2vec loss（student CLS 预测 teacher CLS），给全局难度一个目标？
3. data2vec 2.0 的“只前向未 mask token + 卷积 decoder”能否把 22 层模型的开销降下来？
4. 把 MinaCalc 的逐区间 `required_skill` 当 dense teacher，做“难度蒸馏 + data2vec”多任务，
   是否比纯 SSL 更贴近目标？
5. 7K/其它 keycount 是否值得纳入（当前 DCR 只做 4K）？
