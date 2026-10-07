# 谱面 Embedding（Transformer）调研与方案

> **目标**：给 `(player, beatmap) -> loss` 管线补上**内容侧**的谱面向量 `e_m`，
> 与 step2 学到的**行为侧**潜向量 `C_m` / `D_m` 互补，并为 step3（`C/D -> star`）提供内容特征与冷启动先验。
> 面向 osu!mania 4K。
>
> 调研结论全部标注了出处链接；本文档只描述方案，**不含任何已实现代码**。

---

## 0. TL;DR

1. 你记忆里的"osu mania 4K diffusion"最贴近的是 **[C-H-001/DeepMania](https://github.com/C-H-001/DeepMania)**
   （"AI-powered osu!mania 4K charting tool using Diffusion Models"，2025-12 建仓，WIP）。
   容易混的还有两个：**[OliBomby/osu-diffusion](https://github.com/OliBomby/osu-diffusion)**
   是 **osu!standard** 的坐标扩散（不是 mania），
   **[OliBomby/Mapperatorinator](https://github.com/OliBomby/Mapperatorinator)** 是全模式（含 mania）的
   spectrogram→beatmap 框架。
2. "像自然语言翻译"的那一半是 **[gyataro/osuT5](https://github.com/gyataro/osuT5)**：
   T5 encoder-decoder，mel 频谱 → 离散事件词表，思路抄自 Magenta 的 MT3。
3. **生成模型 ≠ 表示模型**。真正"用 Transformer 抽谱面 embedding"的现成开源实现只有一个：
   **[token03/bobert](https://github.com/token03/bobert)**（osu!standard，~500k 谱面预训练，384 维）。
   它的 mania 分支**没人做过**：`build-features` 会抽 taiko/catch/mania 特征，但"no model is trained on them yet"。
   —— 这正是你项目的空位。
4. **推荐主线**：BoBERT 复刻 + mania 4K 适配 + **行为蒸馏**（§4）。不建议直接把生成模型当 encoder（§3.1）。
5. **最重要的一条纪律**：若 step3 的科学主张是"**行为**潜向量 → star"，则 `e_m` 的预训练**不能拿 star 当监督**，
   否则 step3 变成同义反复（§4.6）。此时 `e_m` 的角色是**控制变量**，不是特征。

---

## 1. 先把三类东西分开

这是整件事最容易混的地方：三者都能叫"谱面模型"，但输入输出完全不同。

| 类别 | 输入 → 输出 | 代表 | 对你有用吗 |
|---|---|---|---|
| **生成模型** | 音频（+难度/风格条件）→ 谱面事件序列 | osuT5、Mapperatorinator、DeepMania、AutoOsu、BeatLearning、osu-diffusion | 模型**必须**有"谱面语言"的内部表示，但那是为生成下一个 note 服务的，不保证是好的整谱表示 |
| **表示模型（encoder）** | 谱面 → 定长向量 | **BoBERT**（standard）；Mapperatorinator `classifier/`（借生成模型做迁移） | ✅ 直接对标，是你想要的 |
| **行为共现 embedding** | 玩家成绩共现矩阵 → 谱面向量 | [Ameobea/osu-beatmap-atlas](https://github.com/Ameobea/osu-beatmap-atlas)（osu!track 高分共现 + UMAP/PyMDE） | 方法学上与你的 step2 `C` 同类（甚至你的 MIRT 口径更干净），是对照组不是工具 |

> 注：`osu-beatmap-atlas` 的做法是"同一批玩家的 top play 里共同出现的谱面 → 近邻"，
> 这与你 step2 用 `(player, beatmap) -> loss` 学出的 `C_m` 是同一个思想的两种实现。
> 它说明"行为 embedding"这条路在 osu! 社区已被验证可行，而**内容 embedding** 是没人补上的另一半。

---

## 2. 开源资产盘点

### 2.1 生成模型

| 项目 | 模式 | 输入 | 输出/关键设计 | 能否出 embedding |
|---|---|---|---|---|
| [osuT5](https://github.com/gyataro/osuT5) | std 为主 | mel 频谱（1 frame / 位置） | T5 encoder-decoder；decoder 每步 softmax over **离散事件词表**；输出稀疏（只在有 note 时出事件） | 只能取 encoder 的音频侧；谱面侧在 decoder，且是自回归前缀，不是整谱表示 |
| [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator) ★601 | **全模式（含 mania：`gamemode=3`, `keycount`, `hold_note_ratio`, `scroll_speed_ratio`）** | 频谱 + 可选 in-context 谱面 | 基于 osuT5 + osu-diffusion；`in_context=[TIMING,KIAI,MAP,GD,NO_HS]` 可把**已有谱面**喂进去当上下文；作者自述 ~5700 GPU 小时 / 261 runs | 有 `classifier/`：在 V22 上迁移学习，输出"high-level feature vectors for beatmaps"，用于谱面相似度 / FID 式评估 |
| [osu-diffusion](https://github.com/OliBomby/osu-diffusion) ★47 | **osu!standard** | — | "Generate osu! standard beatmap object coordinates using a diffusion model with a transformer backbone" | 否（且不是 mania） |
| [DeepMania](https://github.com/C-H-001/DeepMania) | **osu!mania 4K** | 81 通道 = 80 mel + 1 onset envelope | 条件扩散（目标 SR 为条件）；预测 **Gaussian heatmap** 而非二值网格；小节内 straight/swing 竞争量化；低 SR 屏蔽 1/6、1/8 | 否（但它的输入表示法与"难度条件"设计值得抄） |
| [AutoOsu](https://github.com/issyun/AutoOsu) | osu!mania | — | ISMIR 2023 LBD | 否 |
| [sedthh/BeatLearning](https://github.com/sedthh/BeatLearning) | 节奏游戏（"for acoustic people"） | — | 开源生成模型集合，HF 上有 [模型卡](https://huggingface.co/sedthh/BeatLearning) | 否 |
| [softchart-v15](https://huggingface.co/JacobLinCool/softchart-v15) | **taiko** | mel | 7.94M 参数统一模型，slot/time 双模式 + beat head；ALBERT 式因子化 embedding | 否，但它的**因子化 embedding** 值得抄 |

### 2.2 表示模型（重点）

**[token03/bobert — BoBERT: Bidirectional osu! Beatmap Encoder Representations from Transformers](https://github.com/token03/bobert)**
（仅 osu!standard，但整套设计可以直接搬到 mania 4K）

| 项 | 做法 |
|---|---|
| Tokenizer | **一个 hit object 一个 token**；[FT-Transformer](https://arxiv.org/abs/2106.11959) 式：28 个特征分成 6 个连续 + 5 个类别，各 16 维 embedding，拼接后线性投影到 384 |
| Encoder | [ModernBERT](https://arxiv.org/abs/2412.12663) 风格：9 层 / d=384 / 6 头 / pre-norm / SwiGLU，16.1M 参数；**±128 object 的局部注意力**，只在第 3、6、9 层做全局注意力（RoPE） |
| 池化 | 三个全局层的输出 **mean-pool** → 384 维谱面向量 |
| 预训练目标 1 | **Masked reconstruction**：30% hit object 按短 span 遮蔽（含边界邻居加噪），预测其全部特征，80/10/10 配方 |
| 预训练目标 2 | **Strain regression**：线性头从 mean-pool 向量回归 strain 值（用 [parsecore](https://github.com/Apart-Studio/parsecore) 的 SR / structural strain 因子），目的是让池化向量**直接携带难度信息** |
| 适配阶段 | 在"同一 collection 的谱面"上做**对比学习**（linear adapter，初始化为恒等） |
| 规模 | ~500,000 张谱面预训练；序列截断 4,096 objects |
| 数据来源 | [Beatconnect](https://beatconnect.io/) 的 `.osu` 文件 |
| 已知局限 | 不读 AR/CS/OD/HP；>4096 objects 截断；只有 standard；**mania 特征已抽但无模型** |
| 发布物 | [huggingface.co/token03/bobert](https://huggingface.co/token03/bobert)：`model.safetensors` + `embeddings.parquet` + 元数据目录 |

**Mapperatorinator `classifier/`** 的定位值得单独记一笔（[classifier/README](https://github.com/OliBomby/Mapperatorinator/blob/main/classifier/README.md)）：

- 任务：给定谱面的一段 **8 秒**，预测 3,731 个 ranked mapper 中的哪一个（top-1 val acc **12.5%**）。
- 真实目的：**产出谱面的高层特征向量**，用于算生成谱面与真实谱面的相似度（对标图像生成的 FID）。
- 作者结论：用它算出的 FID **与生成谱面的实际质量并不接近**。
- 教训：**(a)** "借生成模型做迁移 + 分类任务逼出特征"是可行套路；**(b)** 但特征好不好用，必须用下游任务验证，不能看它像不像 FID。

---

## 3. 关键结论与坑

### 3.1 别指望从生成模型里"顺便"拿到谱面 embedding

生成模型的 decoder 隐状态语义是"下一个事件"，`in_context=[MAP]` 喂进去的谱面虽然会被编码，但那是**条件前缀**，
没有人验证过它的池化表示是否携带难度/风格信息。硬要用，等于自己重新做一个 §4.6 的验证，成本比直接训 encoder 高。
**唯一值得试的复用**是：直接拿 Mapperatorinator 的 `classifier` 思路，在 mania 上重训一个 8 秒片段分类器（§6 路线 3）。

### 3.2 内容 embedding 很容易学到"歌"而不是"谱面"

`beatmap_id ⊆ beatmapset_id`（同一首歌的多张难度共享音频）。你的 README 也写了 "beatmap_id —— 就是「曲目」"。
如果 `e_m` 是在随机划分上训/评的，它很可能主要在编码 BPM/曲风/混音，而不是 note 布局。**必须**：

- 划分与评估一律按 `beatmapset_id` **分组**（GroupKFold），绝不按 beatmap 随机切；
- linear probe 里把 `bpm` / `hit_length` 当**竞争解释**报出来，而不是只报 star；
- 检索纯度按 `mapper_id`、skillset 分组看，而不只是"看起来像"。

### 3.3 你的 dump 里没有 `.osu` 正文

`osu_beatmaps` 只有元数据 17 列，**没有 note 布局**。要做内容 embedding 必须先补一份 `.osu` 语料（§4.1）。
`beatmap_meta_1k.csv` 里的 `checksum` 是 `.osu` 的 MD5 —— 这是你唯一的**版本一致性校验手段**：
osu! 谱面会被 mapper 更新，而 dump 是 **2026-09-01 快照**，现在下载到的多半是**更新后**的版本。
校验不上的谱面要么丢弃、要么单独标记（这是全流程里最容易被忽略的一个 silent leak）。

### 3.4 convert 谱要注意

若白名单从 `playmode=3 & diff_size=4` 来，理论上都是原生 mania；但 std→mania convert 的 key 数由 CS 决定
（CS=4 的 std 谱转 4K）。convert **没有自己的 `.osu` mania 布局**（`osu.ppy.sh/osu/{id}` 返回原始 std 文件，
`Mode:` 字段不是 3）。下载后解析 `Mode:` 字段即可零成本甄别 —— **内容模型必须在原生 mania 上训**，
否则学到的 note 序列根本不对应 4K 布局。

### 3.5 官方 mania SR 本身就是这些特征的确定性函数

mania 的 star 是从 note 序列算出来的手写公式。这意味着：**用内容特征预测 star 是个"作弊简单"的任务**，
梯度提升树 / 简单统计量就能到很高 Spearman。所以 §4.6 才会强调 `e_m` 在 step3 里的正确角色是**控制变量**。
反过来说，`e_m` 真正不可替代的价值在于**行为侧**：把 `C_m` 与谱面内容对齐。

---

## 4. 推荐方案：BoBERT 复刻 + mania 4K 适配 + 行为蒸馏

### 4.0 与现有管线的关系

```
.osu 语料 ──► [内容 encoder] ──► e_m (384-d)  ─┬─► 先验/正则 ──► C_m （step2）
                                              ├─► 控制变量 ──► step3 的增量检验
                                              └─► 冷启动/新 dump 的直接特征

step0/1/2 ──► C_m, D_m (行为侧)  ──► 蒸馏标签 ──► 回灌 [内容 encoder]（你的独有信号）
```

### 4.1 数据获取与对齐

| 来源 | 规模 | 形态 | 备注 |
|---|---|---|---|
| [project-riz/osu-beatmaps](https://huggingface.co/datasets/project-riz/osu-beatmaps) | **213,068 张谱面**，2,526 h 音频，2007-10-06 … 2025-12-31 | WebDataset（`opus`/mp3 + json，json 里内嵌 `.osu` 正文与 beatmap_id） | 全模式，**需自筛 mania 4K**；同时带音频，适合做多模态 |
| Kaggle [gernyataro/osu-beatmap-dataset](https://www.kaggle.com/datasets/gernyataro/osu-beatmap-dataset) | osuT5 官方训练集 | — | 已被 osuT5 验证过可用 |
| Beatconnect / osu! 镜像 | 任意 | 按 beatmap_id 取 `.osu` | BoBERT 的数据来源；适合只补你那 21,949 张 |
| TalTech [digikogu 数据集](https://digikogu.taltech.ee/et/Download/407038c2-72d4-40e0-890e-67ccf6d523c1) | 523 h 音频 / 2,129 h 谱面（学位论文数据，作者 Richard Nagyfi） | — | 需自己确认授权与字段格式 |

**对齐流程**：`beatmap_meta_1k.csv` 的 `beatmap_id` + `checksum` → 下载 → 解析 `Mode == 3` → 校验 MD5
→ 产出 `{beatmap_id, md5_ok, n_onsets, keycount, source}`。这一步建议独立成一个 step 脚本，产物落 `data/processed/`。

### 4.2 Tokenization（mania 4K 专属设计）

**基本单位 = onset（同刻所有列合并成一个 token）**，不是"每列一个 token"，也不是"每个 note 一个 token"。
理由：mania 的和弦（chord）是天然语义单元；按列拆成 4 路并行序列表在 LN（长条）上会错位。

每个 onset token 的特征（照 FT-Transformer 的做法：连续特征各一个 16 维 embedding，类别特征一个 embedding，拼接后投影到 d）：

| 特征 | 类型 | 设计 |
|---|---|---|
| `cols` | 类别（16 类） | 4-bit multi-hot 的整数编码（0000 不会出现） |
| `dt_beat` | 连续 + 类别 | 与上一 onset 的间隔（**拍**为单位，需要 timing point；退化方案用 ms + BPM 桶）；类别桶 `{1/1,1/2,1/3,1/4,1/6,1/8,其他}` |
| `grid_phase` | 类别 | 对齐到 1/1..1/8 的分数（straight/swing 的显式提示，DeepMania 的"竞争量化"说明这事很重要） |
| `ln` | 连续 ×4 | 每列 hold 长度（0 = 无），分桶 + 连续双路 |
| `chord_size` | 类别 | 1–4 |
| `local_density` | 连续 ×3 | 前 1s / 2s / 4s 内 note 数（归一化） |
| `jack / trill / roll` | 类别/连续 | 与前一 onset 的列重合度、列序是否交替、滑动窗内的重复模式 |
| `hand_balance` | 连续 | 左右手列数差的滑动均值（4K 左 2 右 2） |
| `bar_phase` / `kiai` | 连续/类别 | 小节内相位、是否 kiai（解析 `.osu` 的 TimingPoints / Events） |
| `abs_time` | 连续 | 绝对时间（ms 或小节序号）——**必须给**，因为 token index ≠ 时间 |

**位置编码**：RoPE 作用在 token index 上（BoBERT/ModernBERT 的做法）+ `abs_time` 作为一个输入特征。
不要只用 RoPE，否则 1/2 与 1/4 的密集段落会被压成同样的相对结构。

参考最小实现（伪代码，仅示意）：

```python
def parse_mania(path):
    """→ (mode, keycount, [(t_ms, col, hold_len_ms), ...])"""
    mode = cs = None; notes = []; section = None
    for raw in open(path, encoding="utf-8-sig"):
        line = raw.strip()
        if line.startswith("["): section = line; continue
        if section == "[General]" and line.startswith("Mode:"):      mode = int(line.split(":")[1])
        elif section == "[Difficulty]" and line.startswith("CircleSize:"): cs = float(line.split(":")[1])
        elif section == "[HitObjects]" and line and not line.startswith("//"):
            p = line.split(",")
            x, t, typ = int(p[0]), int(p[2]), int(p[3])
            col = min(3, x * 4 // 512)                      # mania: x∈[0,512) → 列
            end = int(p[5].split(":")[0]) if (typ & 128) and len(p) > 5 else t
            notes.append((t, col, end - t))
    return mode, cs, notes

def to_onsets(notes, bpm):
    """合并同刻 note → onset token 序列"""
    buckets = {}
    for t, col, ln in notes:
        o = buckets.setdefault(t, {"cols": 0, "ln": [0] * 4})
        o["cols"] |= 1 << col
        if ln: o["ln"][col] = ln
    beat = 60000.0 / bpm
    out, prev = [], None
    for t in sorted(buckets):
        o = buckets[t]
        dt = 0 if prev is None else t - prev
        out.append({**o, "dt_beat": dt / beat, "grid": snap(dt / beat),
                    "chord": bin(o["cols"]).count("1"), "t": t})
        prev = t
    return out
```

### 4.3 结构

直接复刻 BoBERT 的骨架（16.1M 参数，单卡可训）：

- 9 层 / d=384 / 6 头 / pre-norm / SwiGLU / RoPE；
- **±128 onset 的局部窗口注意力**，第 3、6、9 层做全局注意力；
- 池化：三个全局层输出 mean-pool（也可 concat 后线性降维）→ `e_m`；
- 变长批处理：按 onset 数分桶（mania 4K 一张谱通常 1–6k onset，远短于 BoBERT 的 4096 object 上限）。

如果后面要压到 `dim=8` 与 step2 的 `C` 对齐，加一个**线性 adapter** 出低维 `e_m^{8}`，而不是把 384 维直接塞进 step2。

### 4.4 预训练目标（按优先级）

1. **Masked feature reconstruction**（必做，唯一无标签信号）：按小节/短 span 遮蔽 30% onset，预测其全部特征，
   含边界邻居加噪，80/10/10。
2. **对比学习**：正样本 = 同一谱面的不同时间窗；同 `beatmapset_id` 不同难度可作为 **hard positive**
   （风格相同、难度不同 —— 注意这会把"难度"从表示里推开，做难度任务时要显式声明）。
   InfoNCE + 投影头，batch 内负样本。
3. **辅助回归**（可选，**与 step3 主张冲突时禁用**）：官方 mania star（`beatmap_meta` 已有）、
   [rosu-pp-py](https://github.com/ppy-sb/rosu-pp-py) 的 mania 难度/strain 输出、`max_combo`、
   note density、LN 比例。Etterna 的 MinaCalc（4K，按 skillset 分解）也可作为更细的回归目标。
4. **行为蒸馏**（你的独有信号，见 §4.6）。

### 4.5 规模与算力

- 你的白名单 21,949 张 × 平均 ~1.5–2k onset ≈ **4×10⁷ token**：16M 参数单卡几小时一个 epoch，完全可行。
- BoBERT 的 500k 谱面量级可以用 §4.1 的 HF 数据集补齐（213k 张全模式，筛出 mania 4K 后按经验会明显变少，
  实际能扩到多少需要先跑一遍数据统计 —— 这是 M1 的第一个验收项）。
- 结论：**先在 21,949 张上把 pipeline 跑通**，规模扩大是第二步。

### 4.6 与 step2 / step3 对接（本节是重点）

**先说禁忌**：你不能既用 star 训练 `e_m`，又用 `e_m` 证明"step3 从行为潜向量预测 star"。
那等于把答案喂进特征里。正确姿势是二选一：

- 若目标是**预测 star**：`e_m`（无 star 监督）作为**控制变量**。step3 的问题改成
  "在控制内容 `e_m` 之后，行为侧 `C/D` 对 star 还有多少增量 R²" —— 这才是行为潜向量的科学价值。
- 若目标是**表示质量**：允许用 star 监督（BoBERT 的 strain regression），但此时 step3 只能作为
  "复现官方 SR" 的 sanity check，不能当发现。

**三种用法（按推荐度）**：

1. **行为蒸馏（最推荐，也是你相对于 BoBERT 的唯一优势）**
   你有 21,899 张谱面的行为数据（10k 口径），BoBERT 没有。
   让 `e_m` 去预测 step2 的**可识别量**：
   ```
   L_distill = MSE( g(e_m),  C_m ⊙ D_m )  [+ cosine]
   ```
   ⚠️ **不要直接回归 `C_m`**：你的 README 已明确 `mirt` 里只有 `C·D` 与 `D` 可识别，`C` 本身有
   `(dim−1)` 维规范自由度 → 直接回归 `C` 在不同的 step2 run 之间没有可比性。
   若确实想要低维 `C` 形式，先跑 Procrustes/CCA 对齐到参考 run，再蒸馏，并在文档里写明这一步。
2. **作为 step2 的内容先验（最彻底）**
   把 `e_m` 直接写进似然：`C_m = f_θ(e_m) + Δ_m`，`Δ_m` 加 L2 惩罚，联合训练 step2。
   好处是验收口径天然就是你已有的主口径（step0 RMSE，1k dim8 = **1.4582**，10k = **1.3331**），
   坏处是要改 step2 脚本、破坏"每步一个脚本一个产物"的现有结构。
3. **作为即插即用特征**
   在 `--features` 里加 `F3 = w · e_m`（低维 adapter 输出）。实现最轻，但要小心 step2 的
   "test 读 step0、train 读 step1" 划分与 `beatmapset` 分组划分的相互作用（见 §3.2）。

### 4.7 评估协议

| 层级 | 指标 | 验收 |
|---|---|---|
| 表示质量 | Linear probe（**冻结** `e_m`）：star 的 Spearman、mapper top-1、上传年份、LN 比例、BPM | 与 §3.5 的"规则特征基线"对比，必须显著更好才有意义 |
| 表示质量 | 检索纯度：kNN 里同 `mapper_id` / 同 `beatmapset_id` / 同 pattern 标签的比例 | pattern 标签可用 [sed-i/mania-pattern-annotations](https://huggingface.co/datasets/sed-i/mania-pattern-annotations)（592 条人工 + 4403 条 agent 判定） |
| 划分纪律 | 上述所有指标都按 `beatmapset_id` **GroupKFold** 报 | 随机划分的结果一律不采纳 |
| **下游（主口径）** | 把 `e_m` 接进 step2 后 step0 RMSE 的变化；**冷启动子集**（每谱面 play 数 ≤ k）单独报 | 1k 目标 < 1.4582 / 1.4225（F1+Pc），10k 目标 < 1.3331 |

---

## 5. 里程碑

| # | 内容 | 验收标准 |
|---|---|---|
| **M0** | 规则特征基线：从 `.osu` 抽统计量（duration、note density、chord 分布、LN 比例、column entropy），线性/GBM 回归 star | 拿到 star 的 Spearman 上界参考值（预期很高，用于校准后续期望） |
| **M1** | 数据：21,949 张 `.osu` 下载 + `Mode==3` + MD5 校验 + onset 统计；统计可扩展的 mania 4K 语料规模 | 产出 `beatmap_content_1k.parquet`（含 `md5_ok` 列与缺失率），缺失率写进文档 |
| **M2** | 从零训 encoder（masked + 对比，**不用 star**），384 维 + 8 维 adapter | GroupKFold 下 linear probe 优于 M0 的规则基线；检索纯度显著高于随机 |
| **M3** | 行为蒸馏 + 接入 step2（§4.6 用法 1 或 2） | step0 RMSE 相对 1.4582 / 1.3331 有可复现下降；冷启动子集下降更明显 |
| **M4**（可选） | step3 的增量检验：控制 `e_m` 后 `C/D` 对 star 的增量 | 得到"行为潜向量是否携带内容之外的信息"的明确结论 |

---

## 6. 备选路线

| 路线 | 成本 | 说明 |
|---|---|---|
| **路线 3（复用生成模型）** | 中 | 学 Mapperatorinator `classifier`：拿它的 mania 权重，在 8 秒片段上做迁移分类/回归，逼出特征。省掉自己设计 tokenizer，但受制于它的许可证、依赖（CUDA 13 / torch 2.10）与"特征未验证"的风险 |
| **路线 4（多模态）** | 高 | 谱面 token + 音频（CLAP/MERT 或 [CLaMP 3](https://huggingface.co/sander-wood/clamp3)）双塔对比。理论上最强，但引入音频后 §3.2 的"学到歌"风险成倍放大，不建议作为第一版 |
| **路线 5（直接抄 BoBERT 权重）** | 低但无效 | BoBERT 只有 standard 权重，mania 特征维度/语义都不同，**不能直接用**；可复用的是它的代码结构与 `core/` 特征约定 |

---

## 7. 参考链接

**生成模型**
- Mapperatorinator — https://github.com/OliBomby/Mapperatorinator ｜ classifier 说明 — https://github.com/OliBomby/Mapperatorinator/blob/main/classifier/README.md
- osu-diffusion — https://github.com/OliBomby/osu-diffusion
- osuT5 — https://github.com/gyataro/osuT5
- DeepMania（mania 4K 扩散）— https://github.com/C-H-001/DeepMania
- AutoOsu — https://github.com/issyun/AutoOsu
- BeatLearning — https://github.com/sedthh/BeatLearning ｜ https://huggingface.co/sedthh/BeatLearning
- SoftChart v1.5（taiko）— https://huggingface.co/JacobLinCool/softchart-v15

**表示模型**
- BoBERT — https://github.com/token03/bobert ｜ 权重 — https://huggingface.co/token03/bobert
- osu-beatmap-atlas（行为共现 embedding）— https://github.com/Ameobea/osu-beatmap-atlas ｜ https://osu-atlas.ameo.dev/

**数据集**
- project-riz/osu-beatmaps（213k 谱面 + 音频 + `.osu` 正文）— https://huggingface.co/datasets/project-riz/osu-beatmaps
- Kaggle osu-beatmap-dataset（osuT5）— https://www.kaggle.com/datasets/gernyataro/osu-beatmap-dataset
- TalTech digikogu（523 h 音频 / 2,129 h 谱面）— https://digikogu.taltech.ee/et/Download/407038c2-72d4-40e0-890e-67ccf6d523c1
- mania pattern annotations — https://huggingface.co/datasets/sed-i/mania-pattern-annotations

**工具**
- beatmap-lens（mania 解析/渲染/标注）— https://github.com/Pulsefield/beatmap-lens
- rosu-pp-py（各模式难度/pp）— https://github.com/ppy-sb/rosu-pp-py
- parsecore（star + structural strain，BoBERT 的辅助目标）— https://github.com/Apart-Studio/parsecore
- CM3P（BoBERT 架构灵感来源）— https://github.com/OliBomby/CM3P

**方法学**
- MusicBERT: Symbolic Music Understanding with Large-Scale Pre-Training — https://arxiv.org/abs/2106.05630
- ModernBERT — https://arxiv.org/abs/2412.12663
- FT-Transformer (Revisiting Deep Learning Models for Tabular Data) — https://arxiv.org/abs/2106.11959
- Beat-Aligned Spectrogram-to-Sequence Generation of Rhythm-Game Charts (GOCT) — https://arxiv.org/abs/2311.13687
- Time-based Chart Partitioning: Improving Local Coherency in Rhythm Game Chart Generation (AIIDE) — https://ojs.aaai.org/index.php/AIIDE/article/download/36808/38946/40885
- MT3 (transcription with transformers) — https://magenta.tensorflow.org/transcription-with-transformers

---

## 8. 一句话总结

现成能抄的只有 **BoBERT**（standard），mania 4K 的谱面 encoder 是空位；
主线是"**BoBERT 骨架 + mania onset tokenizer + 无 star 监督的预训练 + 行为蒸馏**"，
下游验收就用你已有的 step0 RMSE 口径，而不是只看 linear probe 好不好看。
