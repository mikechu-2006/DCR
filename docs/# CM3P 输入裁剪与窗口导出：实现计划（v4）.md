# CM3P 输入裁剪与窗口导出：实现计划（v5）

> v5（2026-11-05 修订；文件名沿用 v4 以保持链接）：
> **T1 固定启用，T3 是唯一实验变量。** 所有 T3 实验共享同一套 canon 裁剪；
> 不再设 T1 开/关对照，也不要求 A0 复现旧表的 NLL。
> 旧 CM3P linear 0.61484 只作历史参考，不是验收门槛。
>
> **真实 HF checkpoint** = Hugging Face 上 `OliBomby/CM3P` 的权重 + `preprocessor_config.json`；
> 它不是上传到 HPC 的仓库副本。HPC 只是算力，本地仓库/数据/产物是唯一权威。
>
> v4 的 OD 决定不变（见 §4）。CM3P 侧只做三件事：删零信息 token、保存窗口、学聚合。
> 相对 v4：T1 从“实验变量”改为“固定预处理”；T2 只跑一份 canon 窗口库；
> A0 改为 T3 内部对照；`add_snapping=False`、`add_cls_token=True`、`max_length=2000`
> 是真实 checkpoint 的口径，不再按 v4 §2 的 4000/snapping 假设。

---

## 0. 口径

| 项 | 定义 |
|---|---|
| 主指标 | P2 chart-disjoint（5 折）test NLL，`--strict-pairs`，ceiling 0.56276 |
| 历史参考 | CM3P linear 0.61484（captured 0.6006）、CM3P MLP 0.61522。它们来自旧 legacy 表，**只作历史参考，不是 A0 的复现目标，也不是验收门槛** |
| T1 | **固定 canon**：`add_hitsounds/add_kiai/add_mania_sv=False`；其余 flag 从真实 checkpoint 继承（实际 `snapping=False`、`cls=True`）。T1 是预处理选择，不是实验变量 |
| T2 | **固定**：16 s / 16 s 窗口，真实 `max_length=2000`；只产出一份 canon 窗口库 |
| 实验变量 | **只有 T3**：池化 A0–A6。A0 是同一 canon 窗口库上的 T3 对照，不是旧表复现 |
| 次指标 | P1 NLL、`Pearson(d,b)`、`captured` |
| 代价 | tokens/note、tokens/窗、`pad_efficiency`、s/chart |
| 判定 | 同 pair 集、同 ceiling 上，**T3 变体 vs A0** 的 paired ΔNLL，cluster bootstrap，95% CI 不含 0 |

只认 efficacy。不做 invariance 证明，不做 PCA，512 维冻结，不做整谱单次前向。
P2 就是正确协议，玩家不隔离。

### 0.1 术语与权威来源

- **真实 HF checkpoint**：Hugging Face 上 `OliBomby/CM3P` 发布的**权重 + processor config**（`preprocessor_config.json`、tokenizer 配置）。`CM3PProcessor.from_pretrained` 读的是它，不是本仓库 `configs/train/*.yaml`。它不是上传到 HPC 的仓库副本。
- **OOD（out-of-distribution）输入**：模型预训练时**没有见过**的 token 类型、token 组合或序列长度。比如 `[MANIA_COLUMN_*]` 若从未被发射，对应 embedding 行就没收到过梯度；把列号打开就是把未训练的行喂进 attention。OOD 不等于“必然报错”，而是“输出不可信，必须用探针验证”。T1 裁剪本身也是一次输入分布变化，但它是**删 token**，且是已接受的固定口径；column 是**加模型没见过的 token**，风险不同。
- **HPC**：只提供 GPU 算力。代码、数据、legacy 表、最终产物都以**本地**为准；HPC 上的副本可以随时重传/丢弃，不能成为唯一状态。T0 在本地做离线审计；T2 在 HPC 跑完必须把表、`.npy`、index、meta 全部回传本地；T3 在本地 CPU 做。
- **checkpoint 快照**：已把 `processor_config.json`、`beatmap_parser/preprocessor_config.json`、`beatmap_tokenizer/tokenizer_config.json`、`config.json` 拷到 [configs/cm3p_hf/](../configs/cm3p_hf/README.md)。T0 把运行时 processor 的 snapshot 写进 index/meta，并用这份本地快照核对（当前实现已硬断言 window/stride/max_time，`max_length` 记录并使用）；不再依赖 HPC 上的副本。注意根目录文件名是 `processor_config.json`，不是 `preprocessor_config.json`（后者的实际路径在 `beatmap_parser/` 下）。

---

## 1. 源码级事实（全部逐行读过）

### 1.1 谱面塔（beatmap tower）

| 事实 | 位置 |
|---|---|
| mania 的列号**算出来了却从未发射**（见 §1.4） | 写 `parsing_cm3p.py:501`、读 `tokenization_cm3p.py:191` |
| 每个 mania tap：**真实 checkpoint 4 token**（`[CIRCLE]` `[TIME_SHIFT]` `[HITSOUND]` `[VOLUME]`）；上游若开 snapping 是 5（多 `[SNAPPING]`）；v4 假设的 6 多算了从未发射的 column | `tokenization_cm3p.py:178-204` |
| hold：**真实 6 token**；上游开 snapping 是 8；v4 假设的 10 同样多算了 column | 同上 |
| 真实 checkpoint 的 processor：`add_snapping=False`、`add_cls_token=True`、`max_length=2000`（每窗另有 CLS/BOS/EOS 3 个特殊 token） | §0.1、[cm3p_crop_windows.md](cm3p_crop_windows.md) §0.1 |
| 拍/小节/红线：每个 **2 token** | `parsing_cm3p.py:367-380` |
| kiai 每次切换 **2–3 token**（event + time (+snapping)） | `parsing_cm3p.py:334-339` |
| mania SV 每次变化 **3–4 token**（event + time (+snapping) + scroll） | `parsing_cm3p.py:311-317` |
| 窗口不重叠 16 s，归属 `start_ms <= t < end_ms` | `processing_cm3p.py:517-546` |
| 截断 `LONGEST_FIRST`，`max_length` 来自 preprocessor config | `tokenization_cm3p.py:230` |
| **beatmap 词表里没有 OD / HP / AR / 星级** | `tokenization_cm3p.py:81-127` |

### 1.2 metadata 塔（这次新增，回答 Q1）

| 事实 | 位置 |
|---|---|
| 谱面塔与 metadata 塔是**两个独立 encoder**，编码时零交互 | `modeling_cm3p.py:942-972` |
| 两者只在对比学习损失里以点积相遇：`logits = metadata_embeds @ beatmap_embeds.T` | `modeling_cm3p.py:976` |
| metadata 序列是**固定 14 个常数 token** + tag：`[DIFFICULTY][YEAR][MODE][STATUS][MAPPER][CS][HITSOUNDED][SONG_LENGTH][SONG_POSITION][GLOBAL_SV][MANIA_KEYCOUNT][HOLD_NOTE_RATIO][SCROLL_SPEED_RATIO]` | `tokenization_cm3p.py:632-654` |
| `[DIFFICULTY_x.x]` 就是**官方星级**（`StarRating` 按 speed 插值） | `processing_cm3p.py:80-87` |
| mania 在这里只拿到 `MANIA_KEYCOUNT` / `HOLD_NOTE_RATIO` / `SCROLL_SPEED_RATIO` | `processing_cm3p.py:111-113` |
| **OD 不在任何一处**：既不在 beatmap 词表，也不在 `CM3PMetadata` 的字段里 | 全仓 grep 无 `overall_difficulty` |

### 1.3 三条直接结论

1. **列号没有被送出去**——不是“无法进入 token 流”，而是上游写错字段 + 0/1-based 偏移；
   补丁能把它发出来（见 §1.4）。要不要开是 OOD/未训练 embedding 的问题，不是能不能。
2. **keycount 删不掉也不用删**：beatmap 流里没有它，只在 metadata 塔（而 metadata 塔是关的）。
3. **官方星级 `[DIFFICULTY_*]` 已经在 metadata 塔里，`include_metadata=False` 已经把它关了**——
   你要删的东西在 token 侧已经删掉了。剩下的只是"别让 `beatmap_meta.star` 混进内容表"。

### 1.4 已确认的上游 bug：键位（column）从来没有进过 token 流

三处代码对不上，而且**全静默**：

| 位置 | 内容 |
|---|---|
| `parsing_cm3p.py:57` | dataclass 字段叫 `mania_column` |
| `parsing_cm3p.py:501` | 写的是 **`group.column`**（不是 `group.mania_column`） |
| `tokenization_cm3p.py:191` | 读的是 `group.mania_column is not None` |

`Group` 是普通 dataclass（没有 `__slots__`），所以 `group.column = ...` 不报错，
只是给实例加了一个**没人读的新属性**；而被读的 `mania_column` 永远是 `None`，
tokenizer 的 `is not None` 守卫直接跳过 —— **不抛异常、不告警、不降级提示**。
全仓 grep 只有这一处写、一处读，没有别的地方补上。

**第二个 bug 叠在上面**：`_get_mania_column` 返回 **0-based**（`clip(int(x/512*K), 0, K-1)`，
`parsing_cm3p.py:464-466`），而 `_tokenize_mania_column` 把它 **clip 到 1..18**
（`tokenization_cm3p.py:151-153`）。所以就算只改属性名，4K 会得到 `1,1,2,3`——
第 0 列和第 1 列撞成同一个 token，`[MANIA_COLUMN_4]` 永不出现。

**正确修法**（opt-in）：

```python
group.mania_column = self._get_mania_column(pos, int(beatmap.circle_size)) + 1
```

**所以列号不是“进不了 token 流”，只是上游没写对。** 词表里已经有 `[MANIA_COLUMN_1..18]`，
tokenizer 也已经有发射分支（`tokenization_cm3p.py:191`）；补丁把 `mania_column` 设成 1-based 后，
列 token 就会正常出现。**能不能进 = 能；要不要开 = 另一个问题**：冻结 checkpoint 大概率没训练过
这 18 行 embedding，开了是 OOD 输入，必须用探针验证，不能默认“修好 bug 就更好”。

**后果**：冻结的 CM3P 谱面向量**结构性看不见按了哪个键**。真实配置下它能看到的每 note 信息只有
`[类型] [TIME_SHIFT]`（真实 checkpoint `add_snapping=False`，没有 `[SNAPPING]`），
T1 裁剪前还多 hitsound/volume。
mania 的难度主要由列型决定（jack / trill / roll / chord 形状），所以这是一个**表示层缺口**，
不是裁剪造成的，裁剪侧也补不回来。

**它解释了仓库里三个一直有点怪的结果**：

| 现象 | 在"列盲"下的解释 |
|---|---|
| CM3P-only MLP 在 P2（新图）上 0 增益 | 非线性只能记住时序统计，没有结构可学 |
| 外部 MSD 在 P2 上给 −0.037 | MSD 有真正的键型/列信息，CM3P 没有 |
| `msd_jackspeed` captured 0.37、`msd_stream` 只有 0.16 | jack 难度是纯列型概念 |

**能不能补**：把 `mania_column` 打开等于喂 `[MANIA_COLUMN_1..18]` 这 18 行 embedding，
而它们大概率**在预训练里从未收到梯度**（244K 表若由同一份代码产出，就从未发射过这些 token）——
未训练的行 ≈ 初始化噪声。所以"打开"很可能变差；真要用，得**微调那 18 行**（+ 顶部若干层），
那是 fine-tune，不是裁剪。判据只能靠探针（见 §7 第 3 步）。

**那能不能走 x 坐标？** 技术上可以，但当前代码不会自动这么做：

- 真实 parser 的 `add_positions=True`，tokenizer 也确实会发 `[POS_X_*]` / `[POS_Y_*]`；
- 但 `_add_group` 只在 `mode in [0, 2]` 时填 `group.x/y`，mania 分支只写 `group.column`；
- 要让它走 x，需要 patch mania 分支：`group.x = int(pos[0]); group.y = 0`。
  `position_range=[-256,768,-256,640]`、`position_step=4`，4K 的 0..512 完全在范围内；
- 代价是每个 note 多 **2 个 token**（`[POS_X_*]` + 常量 `[POS_Y_0]`）。canon 的 2.76 token/note
  会变成约 4.76，几乎回到 legacy 的 4.90——**T1 的 token 收益基本被抵消**；
- 语义上 `[POS_X_*]` 是 std/catch 的坐标词表，checkpoint 在 mania 上下文里没见过；
  它不是 `[MANIA_COLUMN_*]`，仍是 OOD，只是 embedding 行本身训练过（与未训练的 column 行不同）。

所以：**x 坐标能进，但不便宜，也不是“干净修好”**。如果只是想验证“列信息值多少”，
它只能作为与 column-on 并列的 side experiment，不能进主 T1/T3。
更省 token 的替代是 `[DISTANCE_*]`（1 token/note，但要打开 `add_distances` 并 patch mania 的
`group.distance`，语义同样错位），或者干脆修正 `mania_column` 后用 1 token 的
`[MANIA_COLUMN_*]`。三者都要探针，不能默认谁更好。

（顺带：本地 `external/cm3p` 是 shallow clone，`git log` 只有一个 commit `546f2e2`，
blame 全挂在 grafted root 上，**没法用 git 判断这个 bug 是何时引入的**。
能判断的只有两条：checkpoint 里这 18 行的范数/方差 vs 训练过的 `[SNAPPING_*]` 行；或 off/on 探针。）

---

## 2. token 预算（400 张抽样；真实 checkpoint 配置）

> **v4 的数字已作废。** v4 按“6 token/tap + 有 snapping + max_length=4000”算出 tokens/chart=10,591、
> tokens/note=7.47、2000 时截断 9.3%→0.8%。真实 checkpoint 是“4 token/tap + 无 snapping + 2000”，
> 所以绝对预算低约 1/3，裁剪收益也更大。下表是实现侧用真实配置离线复现的 400 图结果。

| 量 | legacy（真实 checkpoint 默认） | canon（T1 固定） |
|---|---:|---:|
| tokens / chart（mean） | 8,041 | 4,524 |
| tokens / note（pooled total/total） | 4.90 | 2.76 |
| tokens / window（mean） | 757 | 426 |
| tokens / window（p50 / p95） | 709 / 1,509 | 409 / 801 |
| 撞 `max_length=2000` 的窗口 | 18 / 4,251 = 0.42% | 0 / 4,251 = 0% |
| 排序分批 pad_efficiency（B=16） | — | 0.995 |
| 每窗 pad 到 2000 的 pad_efficiency | — | 0.213 |
| column token | 0 / 705,943 note events | 0（默认 off） |
| hold 跨 16 s 边界 | 978 / 49,801 = 1.96% | 同左 |

- 裁剪节省 **43.7%** token；attention 的 n² 约 **3.16×**（v4 写的 27% / 1.8× 是旧假设）。
- **截断不再是一个隐藏 confound**：真实 `max_length=2000` 下 canon 窗口 0 个越界；legacy 也只有 0.42%。
  HF 的 `truncation_side=right` 仍会砍窗口尾巴，T2 逐窗记账（`tok_len_full`、
  `n_truncated_windows`、`n_core_tokens_dropped`），不静默。
- 这些数仍然只是 400 图抽样；全量 21,949 由 `scripts/slurm/cm3p_budget.slurm` 生成
  `results/cm3p_rank_1k/token_budget_all.json`。**绝对预算以真实运行产物为准**，
  本表用于设计和验收，不当作已完成的实测。

（T0 仍要在每次运行时用 `processor.beatmap_kwargs` 断言 window=16 / stride=16 / max_time≥16000，
 并把真实 `max_length` 写进 index/meta；真实值来自 HF checkpoint 的 preprocessor config，
 不是 `processing_cm3p.py` 的 30 s / 8000 默认。）

---

## 3. T1：固定启用（不是实验变量）

**T1 在这一次正式实验里对所有 T3 实验都是同一套**：只改三个载荷开关，其余从真实 checkpoint 继承。
不设 T1 on/off 对照，也不把 T1 的 ΔNLL 当成 T3 的归因。

`CM3PBeatmapParser.__init__`（`parsing_cm3p.py:201-215`）已有全部开关，**上游零修改**：

| 开关 | 去掉/保留的 token | 决定 |
|---|---|---|
| `add_hitsounds=False` | `[HITSOUND_*]` + `[VOLUME_*]`（每 note 2 个） | **删** |
| `add_kiai=False` | `[KIAI_ON/OFF]`（每次切换 2–3 个） | **删** |
| `add_mania_sv=False` | `[SCROLL_SPEED_*]`（每次变化 3–4 个） | **删** |
| `add_timing=True` | `[BEAT]` `[MEASURE]` `[TIMING_POINT]` | **保留**（见下） |
| `add_snapping` | 真实 checkpoint 是 **False**；不强行打开 | **继承**（OOD，见 §0.1） |
| `add_cls_token` | 真实 checkpoint 是 **True**；池化读 CLS | **继承，不碰** |
| `add_timing_points` | 真实 checkpoint 是 **True** | **继承** |

**为什么保留拍/小节/红线网格**：不规则的 beat/measure 间隔本身携带信息——
切分音、变速、非整数小节正是 tech 类谱面的特征。删掉它只省约 15% token，
却把"节奏不规则性"这个维度整个抹掉。这与"删 hitsound"的性质不同：后者是**零信息**，前者是**信息编码方式较贵**。

**为什么不动 `add_snapping`**：v4 写“必须保持 True”是错的。真实 checkpoint 的 processor 就是
`add_snapping=False`；强行打开等于把模型没训练过的 `[SNAPPING_*]` 塞进去，是 OOD 输入，不是“保留信息”。
T1 只做删载荷，不做加 token。

实现：

```python
proc = CM3PProcessor.from_pretrained(rev)
proc.beatmap_parser.add_hitsounds = False
proc.beatmap_parser.add_kiai = False
proc.beatmap_parser.add_mania_sv = False
```

parser 在调用时读取这些 flag，改属性立即生效。

### 3.1 每个键在做什么、填什么

先看一个真谱面的 token 流（`data/osu_cache/3870997.osu`，4K，91 notes，BPM≈149）：

```text
真实 checkpoint 默认: [BOS] [CIRCLE] [TIME_SHIFT_47] [HITSOUND_0_2_2] [VOLUME_40] ...
T1 canon 之后:        [BOS] [CIRCLE] [TIME_SHIFT_47] [CIRCLE] [TIME_SHIFT_448] ...
```

每个 note event 4 → 2 token（CIRCLE/TIME_SHIFT 保留）；hold 6 → 4
（`[HOLD_NOTE]` `[TIME_SHIFT]` `[HITSOUND]` `[VOLUME]` `[HOLD_NOTE_END]` `[TIME_SHIFT]`
→ 去掉 hitsound/volume 后的 4 个）。注意真实配置没有 `[SNAPPING]` 和 `[MANIA_COLUMN]`；
v4 示例里的这两个 token 不存在。

#### A. 要改的三个（parser 开关，只影响"发不发 token"）

| 键 | 它在生产什么 token | 填 | 理由 |
|---|---|---|---|
| `add_hitsounds` | 每个 note 的 `[HITSOUND_{bits}_{set}_{add}]` + `[VOLUME_v]`（音效位 whisper/finish/clap、采样集、音量，都来自 timing point） | **False** | 与按键难度无关；每 note −2 token |
| `add_kiai` | 每次 kiai 开关的 `[KIAI_ON]`/`[KIAI_OFF]`（+时间 (+snapping)，2–3 token） | **False** | 高潮段标记，纯演出 |
| `add_mania_sv` | 每个 SV 变化点的 `[SCROLL_SPEED_CHANGE]` + 时间 (+snapping) + `[SCROLL_SPEED_x.xx]`（3–4 token） | **False** | 流速玩家可自调；数据清洗里视觉障碍模组也已剔除 |

#### B. 保留/继承的（删了会掉信息，但不能强行打开）

| 键 | 它在生产什么 | 填 | 理由 |
|---|---|---|---|
| `add_timing` | 沿 timing point 生成 `[TIMING_POINT]`/`[MEASURE]`/`[BEAT]` 网格事件（每个 2 token）。这是模型看 **BPM 变化与小节结构** 的唯一通道 | **True** | 不规则节奏本身是 tech 线索 |
| `add_snapping` | 每个 note 的 `[SNAPPING_k]`，k=1..16 表示落在几分音符上（k=0 = 不在网格）。是"节奏关系"的唯一编码 | **继承真实值（False）** | v4 写“必须 True”不成立：真实 checkpoint 没训练过 snapping；强行打开是 OOD，不是保留信息 |

#### C. 无所谓、别动的（对 mania 零影响）

| 键 | 说明 | 填 |
|---|---|---|
| `add_timing_points` | 只决定每个 timing point 后第一拍叫 `[TIMING_POINT]` 还是 `[MEASURE]`（token 数不变） | True（默认） |
| `add_positions` / `add_distances` / `add_sv` | std/catch 的坐标/距离/滑块 SV；mania 分支根本不写这些字段 | True（默认） |
| `slider_version` | std 滑块控制点编码方式 | 2（默认） |
| `mania_bpm_normalized_scroll_speed` | 只影响 SV 数值；`add_mania_sv=False` 后无关 | True（默认） |

#### D. 绝对不要碰（tokenizer：词表定义 + 池化位置）

| 键 | 为什么 |
|---|---|
| `add_cls_token` | 池化是 `last_hidden_state[..., 0, :]`（`modeling_cm3p.py:632`）。关掉 = 把池化读的东西从 `[CLS]` 换成 `[BOS]`，是换掉输出而不是少一个 token |
| `min_time` / `max_time` / `time_step` | 定义 `[TIME_SHIFT_t]` 的词表区间与步长（0–30000 ms，10 ms）。改了会落到 `[UNK]` |
| `max_distance` / `distance_step` / `position_range` / `position_step` / `position_split_axes` | 同上（且 mania 不用） |
| `separate_new_combo_token` | `[CIRCLE_NEW_COMBO]` vs `[CIRCLE]`+`[NEW_COMBO]` 的词表形态 |

#### E. Processor 级（不是 parser 属性，走 `beatmap_kwargs`）

| 键 | 填 | 说明 |
|---|---|---|
| `window_length_sec` / `window_stride_sec` | 保持 checkpoint 原值（文档记 16 / 16） | 已决定不改 stride |
| `min_window_length_sec` | 1.0 | 默认 |
| `max_length` | **使用真实值（2000）**，不要硬写成 4000 | 真实 checkpoint 是 2000；canon 0 个窗口越界，legacy 0.42% |

#### F. 怎么填（就这三行 + 继承）

```python
proc = CM3PProcessor.from_pretrained(rev)
pp = proc.beatmap_parser
assert pp.add_timing                              # 先断言；不要在 canon 里改 True/False
# add_snapping / add_cls_token 保持 checkpoint 的值（真实 False / True）
pp.add_hitsounds = False
pp.add_kiai = False
pp.add_mania_sv = False
# proc.beatmap_tokenizer / proc.metadata_tokenizer 一律不碰
```

这三个值要进缓存键（`cfg_hash`），否则新旧 token 流会静默混用。

两条自检：

1. **配置快照**：`print(proc.beatmap_kwargs)`，断言 `window_length_sec==16`、`stride==16`、
   `tokenizer.max_time>=16000`；`max_length` 记录真实值（当前 2000）并使用它。
2. **计数守恒**：同一张图，`add_hitsounds` True/False 的 token 数之差应恰等于 2 × note 数。

收益：**−43.7% token**，attention 成本 ∝ n² → 约 **3.16×**（v4 的 27% / 1.8× 是旧假设）。

---

## 4. 谱面常数（OD）：决定不注入 CM3P

### 4.1 决策

**OD 不进 CM3P 的输入，也不进 head。** CM3P 侧只负责"内容 → 难度"这一件事；
谱面常数（OD、HP、keycount…）由下游任务自己读——`step2_cm3p_rank.py` 已经有现成的
`--extra-signals` 通道，任何二次分析也可以直接 join。

本节的其余内容是**决策记录**，避免以后重开这个讨论。

### 4.2 依据（四条，都有行号）

**① CM3P 没有"常数 → 谱面塔"的通道。**
`beatmap_embeds` 是 chart token 的纯函数（`modeling_cm3p.py:942-959`）；
metadata 走另一条独立路径（`:962-972`），两者唯一交汇是
`logits = metadata_embeds @ beatmap_embeds.T`（`:976`）。
编码阶段没有 cross-attention / concat / FiLM。**所以"喂常数"在设计上就只能产生第二个向量，
而不是改变谱面向量。**

**② metadata 塔在你现在的语料上填不满。**
`_parse_osu_file`（`beatmap_files_dataset.py:61-164`）不填 `StarRating` / `SubmittedDate` /
`Status` / `UserId` / `TopTagIds`，全部 `pd.NA`；
`get_metadata` 里 `beatmap_metadata["SubmittedDate"].year` 会抛 `AttributeError`，
被 dataset `_iter` 的 `try/except` 吞掉 → `continue` → **静默丢图**。
要启用必须先改上游 + 补一张 MMRS 元数据表，而那张表的核心字段正是你要排除的官方星级。

**③ metadata 塔里没有 OD。**
14 个字段是 star / year / mode / status / mapper / cs / hitsounded / song_length /
song_position / global_sv / mania_keycount / hold_note_ratio / scroll_speed_ratio。
唯一量纲吻合的空槽位是 `[CS_x.x]`（0.0–10.0 step 0.1，`tokenization_cm3p.py:484`），
而 mania 里 `cs` 恒为 `None` 所以永远是 `[CS_UNK]`（`processing_cm3p.py:110`）。
用它就得微调 101 行 embedding——**成本与收益不对称**。

**④ 实测 OD 单独几乎没有排序信息。**
同玩家成对一致率：官方 star **0.6330**、HP drain 0.5258、**OD 0.5097**（se ≈ 0.005）。
也就是说 OD 唯一的理论收益只能来自"OD × 内容"的交互项，而那个交互项在 head 侧做只要 515 个参数——
**但那是下游的实验，不属于 CM3P 侧**，所以本计划不做。

### 4.3 下游怎么读 OD（现成，零改动）

`step2_cm3p_rank.py` 已支持：

```bash
python scripts/step2_cm3p_rank.py --model linear_extra --extra-signals od,msd_overall ...
```

即 `d = w'C + v'S`，`S` 可以是 `od`（`rank_signals_1k.parquet` 里已有该列，
rank_eval 的 summary 里也有 `od` 一行）。注意两点，写在这里以免下游重新踩：

- **只有 `od` 的 `S` 是加性的**，而实测 OD ≈ 0.51 一致率 → 预期增益 ≈ 0，不要期待。
- 想要"OD × 内容"的交互，得在下游自己加一列（如 `od × d` 或 `od × z`），
  那是下游特征工程，**不改 CM3P 侧的产出**。

### 4.4 一个必须记住的后果

**从此 `C` 与 `d` 都是 OD 盲的。** 任何"预测我会打多少 ACC"的下游任务
都必须自己把 OD 接进去；CM3P 这条路给不了。这一条写进 `docs/` 的模型说明里，
避免以后有人拿 `d` 直接当绝对难度用。

---

## 5. T2：窗口导出 + 分桶（一次全量 GPU 跑）

改 `step0c.from_osu_dir` 405–437 行：

```python
acc[bid] = {"win": np.stack(vecs).astype(np.float16),        # (n_win, 512) 时间序
            "mean_norm": float(np.linalg.norm(np.mean(np.stack(vecs), axis=0))),
            "tok_len": tok_len_per_window}
```

同时加 `BucketSampler`（Q2 已确认做）：

```python
class BucketSampler:
    """按 ceil(len/64)*64 分桶；桶内随机，桶序列随机；同 chart 的窗口保持时间序。"""
```

- 分桶会改变 batch 组成 → 数值微变（attention mask 正确时 1e-5 量级），用 T2 的自检兜住。
- 打印 `pad_efficiency = Σlen / (B × max_len)` 与 charts/s，作为这一项的验收。

旁路产物（主表 `COLUMNS` 契约不动）：

```text
.cache/cm3p/win/<beatmap_id>.npy       # float16；21,949 × 10.63 × 512 × 2B ≈ 240 MB
.cache/cm3p/index.parquet              # beatmap_id, n_win, mean_norm, tok_len[], content_md5, cfg_hash
data/processed/chart_pool_features_1k.parquet
```

唯一保留的自检（防导出代码写错）：**同一次 canon 运行**里，
`mean(win)/||mean(win)|| == 该次运行写出的图级 embedding`，cosine ≥ 1−1e-6。
这个自检只证明“窗口存储 → 图级向量”没写错，**不要求 canon 窗口均值等于旧 legacy 表**；
旧表 `chart_content_cm3p_local.parquet` 保留作历史参考，不是 T2 的验收对象。

主实验**只跑一份 canon 窗口库**；不需要为了 A0 再跑一份 legacy 窗口库。
如果以后想把 canon 与旧表做诊断性对比，再单独用 `--crop legacy` 跑 N 张，属于 side experiment。

**窗口 stride 不改（Q4）**：保持 16 s 不重叠。已知的边界效应——
跨 16 s 边界的 hold 会把 `[HOLD_NOTE_END]` 甩进下一个窗口——**作为已知局限记录，不修**。
只在 index 里多存一列 `n_holds_split`（落在边界 2 ms 内的 hold 数），供以后量化，成本为零。

---

## 6. T3：池化（纯 CPU；唯一实验变量）

**所有 T3 变体共享同一份 T1 canon 窗口库**，所以比较只在 T3 内部发生：
A0 是同一 canon 窗口库上的均值对照，A1–A6 是不同聚合方式。
**不要求 A0 复现旧 legacy 表的 NLL，也不要求它等于 0.61484**；旧数字只作历史参考。

先图级后窗口级：`z_t = w·W_t`，w 取 A0 线性头的解。

| 变体 | 特征 | 宽度 |
|---|---|---|
| A0 | canon 窗口的图级均值（**T3 内部对照**） | 512 |
| **A2** | 均值 + `mean(z_t), sd(z_t), q90, q99, max, log mean_norm, log n_win` | 519 |
| A3 | A2 + 最长高压连续段 + 休息段统计 | 521 |
| A4 | `LSE(z)=log Σ exp(βz_t)/β` 替换 mean | 512 |
| A5 | `minacalc.aggregate_rows(z_t)` | 512 |
| A6 | 窗口序列过 1–2 层 attention/TCN | 512 |
| A1 | 逐维 mean+std（**风险项**，1024 维 / 6,828 图） | 1024 |

pipeline B 零改动：`step2_cm3p_rank.py --content-path chart_pool_features_1k.parquet --content-col features`。

**主判据（paired，T3 内部）**：A2/A3/A4/A5/A6 每个变体 vs **A0**，
同 pair 集、同 ceiling、同 5 折；cluster bootstrap 的 95% CI 不含 0 才算有增益。
**旧 0.61484 不是门槛**；如果 A0 本身是 0.63 或 0.59，那是 T1/T2 的固定口径结果，
不影响“哪个 T3 变体比 A0 好”的配对比较。绝对 NLL 仍上报，但只作描述。

---

## 7. 执行顺序

**T1 已定，不再作为变量。** 列盲是已知表示层局限；它影响的是“CM3P 内容向量本身能解释多少”，
不影响“在固定 T1 下，哪个 T3 聚合更好”。所以主实验只变 T3，列探针降级为可选的 side experiment，
不阻塞 T2/T3。

| 序 | 工作 | 收益 | 备注 |
|---|---|---|---|
| 1 | T0 断言（`beatmap_kwargs` 快照 + 每类 token 计数） | — | 后面全部依赖；真实 `max_length=2000`，不是 4000 |
| 2 | T1 固定三个 flag + core audit + 计数自检 | 43.7% token；n²≈3.16×；截断 0.42%→0 | **所有 T3 实验共用**，不做 on/off 对照 |
| 3 | T2 窗口导出 + 分桶（一次 GPU 跑） | 3–4× 速度 | 只产出一份 canon 窗口库；打印 pad_efficiency |
| 4 | T3 池化 A0–A6 | NLL 主增益 | **唯一实验变量**；纯 CPU |
| 5 | （可选）列 off/on 探针 | 量化列盲值多少 NLL | 不阻塞主实验；如要开 column，必须作为单独固定的 T1 变体，不能混进主 T3 |
| 6 | （可选）把 canon 与旧表做描述性对比 | 历史参考 | 不进主判定；不要求 A0 复现 0.61484 |

可选的列探针结果只用于**未来**是否重跑，不影响本次 T3 的主结论：

| 探针结果 | 解读 | 下一步 |
|---|---|---|
| on 明显更好 | 那 18 行 embedding 里有可用信号 | 另开一轮把 column 固定进 T1，再重跑 T2/T3 |
| 差不多 | 行是噪声但被 attention 忽略 | 维持 off；列缺口靠微调/换 encoder |
| on 明显更差 | 行是初始化噪声、且干扰注意力 | 确认"冻结 encoder 列盲"；维持 off |

## 8. 验收矩阵

| 阶段 | 变更 | 判据 |
|---|---|---|
| T1 | 固定 3 flag（不是实验变量） | core stream 与 legacy 逐窗一致（只差 hitsound/volume/kiai/SV）；token 计数守恒；cfg_hash 固定 |
| T2 | 窗口导出 + 分桶 | **同一次运行**的窗口均值 vs 图级向量 cosine ≥ 1−1e-6；pad_efficiency 与 charts/s 上报 |
| T3 | A2/A3/A4/A5/A6 vs **A0** | **配对** ΔNLL 的 95% CI 不含 0；同一 T1 窗口库、同一 pair 集、同一 ceiling |

---

## 9. 混淆检查（只在做 canon vs legacy 描述性对比时）

T1 固定后，主实验的 A2–A6 vs A0 都在 canon 内部，不受"删 hitsound 是否只是掉了流行度代理"影响。
只有当你额外想把 canon 与旧 legacy 表做描述性对比时，才需要这一层：
hitsound 是"被认真做过"的代理，与流行度相关，而标签来自 top-1000 玩家的分布。
用 `beatmap_meta.count_total` 分十层，层内单独比。**这不是主判据。**

---

## 10. 风险

| 风险 | 信号 | 处置 |
|---|---|---|
| preprocessor 的 window/max_length 不是 16/2000 | T0 断言失败 | 记录真实值并按 §2 重算预算；window/stride/max_time 不匹配默认 SystemExit，`--allow-config-mismatch` 才继续 |
| 真实 checkpoint 与本地 yaml 不一致 | T0 快照与预期不同 | 以 checkpoint 的 `processor.default_kwargs` 为准；把快照写进 index/meta，不靠 HPC 记忆 |
| flag 没生效 | token 计数差 ≠ 2×note | 改为在构造 parser 时传参，并在 core audit 里逐窗比较 |
| 分桶改变数值 | 与 batch=1 的 cosine < 1−1e-5 | 记录容差，或固定 batch 组成；分桶参数不进 cfg_hash |
| fp16 存储 | 重跑一致性 1e-3 | 换 fp32（436 MB） |
| HPC 成为唯一状态 | 本地找不到 T2 产物 | HPC 只跑；表/.npy/index/meta 必须回传本地，T3 本地做 |
| column 被误当成“已修好” | 开了 on 但 NLL 变差 | 主实验固定 off；on 只作单独固定变体/探针，不混进 T3 |

---

## 11. 已定执行口径（原 Q1''/Q2'' 关闭）

- **执行位置**：本地仓库/数据/最终产物是权威；HPC 只提供 GPU。代码和数据从本地传上去；
  T2 产出的内容表、`.cache/cm3p/win/*.npy`、`index.parquet`、两份 `.meta.json` 全部回传；
  T3 在本机 CPU 跑。HPC 上的副本可以随时重传或删除。
- **旧表**：`chart_content_cm3p_local.parquet` 本地为准；HPC 上那份只是副本。
  旧 0.61484 / 0.61522 只作历史参考，**不是 A0 的复现目标，也不是验收门槛**。
- **列号**：可以进入 token 流（词表 + tokenizer 分支 + 1-based patch 都在）；但主实验固定
  `--mania-columns off`（或 auto 的“检测但不 patch”），因为冻结 checkpoint 大概率没训练过这些行。
  如果以后要把 on 作为正式口径，必须**作为固定的 T1 变体另开一轮**，再在那一轮内部只比较 T3。
- **T1**：固定 canon；所有 T3 实验共享；不做 T1 on/off 对照。
- **T3**：唯一实验变量；主判据是 A2–A6 vs A0 的 paired ΔNLL（95% CI 不含 0）。