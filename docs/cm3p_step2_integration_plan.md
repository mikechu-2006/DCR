# CM3P × step2 接入方案 v2：冻结内容向量 + 纯点积交互，两条独立管线

> **状态**：**已实现并实测**（2026-10-09）。实现清单与全部实跑数字见
> [docs/cm3p_step2_results.md](cm3p_step2_results.md)；本文保留设计与理由，不重复数字。
> **上游**：[docs/beatmap_embedding_plan.md](beatmap_embedding_plan.md)（谱面 encoder 调研）。
> **v1 → v2 的变更**（按 2026-10-09 的规格重写）：
> 1. **弃用 `mirt`**，只用点积 `mf_dot`：`L = b0 + bu[u] + bm[m] + <P_u, C_m>`；
>    丢掉 `D`、`-C·D`、以及围绕它的 GL(K) 规范讨论。
> 2. **`C_i` 是冻结的谱面 embedding**（不再有 `W_c` 投影层），`P_i` 改成与 `C_i` **同维**。
> 3. **两条互不依赖的管线**：管线 A（transformer 推理 → 内容表，可增量合并）、管线 B（step2 读表、只取有值的行）。
> 4. v1 的 §4.2（软先验被规范吃掉）在 v2 里**自动消失** —— 原因见 §4.2。

---

## 0. TL;DR

1. **模型**：`L_hat(u, m) = b0 + bu[u] + bm[m] + <P_u, C_m>`。
   - `b0 / bu / bm` = 3 个偏置单元，保持自由（`bm` 仍是每谱面一个自由参数）；
   - `C_m` = 谱面 m 的 CM3P 向量，**冻结、不训练**；
   - `P_u` = 玩家 u 在同维空间里的向量，**与 `C_m` 同维**，从零学起。
2. **"替换 1-hot"在 v2 里的准确含义**：交互项 `<P_u, C_m>` 里**不再有任何 per-chart 自由参数** ——
   `C_m` 完全由谱面内容决定。1-hot 只残留在 `bm` 这一个偏置上。
   谱面侧自由参数从 `O(M·K)`（1k 约 33 万）降到 `M@@（只有 `bm`）。
3. **`d`（= `C` 的维度）就是 `P` 的参数量，必须当超参扫，不能钉死 512**。
   `d=512` 时 `P` 有 **504,832** 个参数（1k 覆盖子集），而仓库已有的 `free_K16`（330k 参数）实测**过拟合**
   （test 0.7950 vs `free_K1` 的 0.7227，见 §1.1）。计划扫 `d ∈ {8,16,32,64,128,512}`。
4. **`C` 冻结之后，模型对参数是线性的、且完全可识别**（没有旋转规范、没有 `(dim−1)` 维自由度）。
   `P_u` 可以被逐维解释成"玩家 u 在 CM3P 语义空间里的偏好坐标"。
5. **数据可行性已实测**：HF [OliBomby/CM3P-Embeddings-244K](https://huggingface.co/datasets/OliBomby/CM3P-Embeddings-244K)
   与本仓库 21,949 张 4K mania 白名单**交集 11,877 张（54.1%）**，checksum 一致率 98.78%。
   ⇒ 管线 B 只在**有值的 54%** 上跑，1k 剩 541,251 plays / 986 玩家 / 11,520 图。
6. **验收线用本地 dim 扫描的最优值**：1k 全表 `mf_dot` dim16 = **1.454837**
   （§1.1 的 0.7091 是另一套标准化口径的旁证，**不作为门槛**）。
   **实测：真实时间漂移把它推到 1.37173（−5.7%）**，内容向量在同子集上与 1-hot 打平略优 ——
   见 [docs/cm3p_step2_results.md](cm3p_step2_results.md)。

---

## 1. 现状与"1-hot"指什么

`scripts/step2_predict.py` 的 `EmbeddingModel`（[L133-253](../scripts/step2_predict.py#L133-L253)）：

    L_hat = b0 + bu[u] + bm[m] + interaction
      mf_dot : interaction = <P[u], C[m]>
      mirt   : interaction = <P[u] - C[m], D[m]>      <- v2 弃用

| 参数 | 形状 | v2 处置 |
|---|---|---|
| `P` | `(n_ent, K)` | ✅ 保留，但维度改成 `d`（= `C` 的维度） |
| `C` | `(n_item, K)` | ❌ **删掉**，换成冻结的 `C_m = e_m` |
| `D` | `(n_item, K)` | ❌ 删掉（没有 `mirt` 了） |
| `bm` | `(n_item,)` | ✅ 保留（3 个偏置之一，仍是 1-hot 表） |
| `bu`, `b0` | `(n_ent,)`, `(1,)` | ✅ 保留 |

`nn.Embedding` 就是 `onehot(m) @ W`，所以旧的 `C`/`D` 就是 1-hot 编码。
v2 把这两张表整个拿掉，只留 `bm`。

### 1.1 仓库里已有的直接证据：内容锚定 > 自由 1-hot

[docs/step2_prediction.md](step2_prediction.md) 没写，但脚本已经跑过：
[scripts/diag_step2_fm.py](../scripts/diag_step2_fm.py) 在同一 split、同一训练循环上只改**交互参数化**：

* **自由低秩** `<P_u, D_m>`，参数量 `K·(n_ent+n_item)` —— 1-hot 那一套；
* **内容锚定（FM）** `Σ_f <P_u, V_f>·x_mf = <P_u, V·x_m>` —— `C_m = V x_m` 由内容线性生成，
  参数量 `dim·(n_ent+n_feat)`，**与谱面数量无关，新谱面免费拿向量**。

实测（`data/processed/diag_step2_fm_1k*.json`，标准化 loss 的 test play RMSE）：

| 模型 | 参数 | 7 列全用（含 star） | 去掉 star（6 列） |
|---|---|---|---|
| free_K1 | 20,623 | 0.7265 | 0.7227 |
| free_K4 | 82,492 | 0.7318 | 0.7318 |
| free_K16 | 329,968 | 0.7950 | 0.7950 |
| **fm_dim4** | **3,976** | **0.7042** | **0.7091** |
| fm_dim16 | 15,904 | 0.7042 | 0.7098 |

1. 自由 1-hot 因子**过参数化**：`free_K1 → free_K16` 参数 ×16，测试误差反而 **0.7265 → 0.7950**。
2. 内容锚定用 1/5 的参数、且不含 star，也优于最好的自由因子（**0.7091 vs 0.7227**）。
   free_K 两列的差（0.7265 / 0.7227）来自两次独立运行 ⇒ **这张表的噪声底约 ±0.004**。
3. ⇒ 这条证据说明"内容锚定"这条路本身是通的（**旁证，不是门槛**）。
   本方案的验收线改用**本地 dim 扫描的最优值 1.454837**，实测见
   [docs/cm3p_step2_results.md](cm3p_step2_results.md)。

---

## 2. CM3P 给出什么（已核对源码）

`external/cm3p/` 是 [OliBomby/CM3P](https://github.com/OliBomby/CM3P) 的克隆，**MIT 许可**。

| 项 | 事实 | 出处 |
|---|---|---|
| 输出 | `outputs.beatmap_embeds`，**每 16 秒窗一个 512 维向量，L2 归一化** | `configs/model/default.yaml: projection_dim 512`；`modeling_cm3p.py:958-960` |
| 窗口 | `window_length_sec = 16`，`window_stride_sec = 16`，`max_length = 4000` token/窗 | `configs/train/default.yaml:100-107` |
| 官方聚合 | 按窗求平均 → 再 L2 归一化 → 每谱面 **1×512** | `extract_beatmap_embeddings.py:243-266` |
| **mania 原生支持** | beatmap 侧词表含 `[MANIA_COLUMN_1..18]`；mode 3 额外产出 `mania_keycount` / `hold_note_ratio` / `scroll_speed_ratio` | `tokenization_cm3p.py:110,151-153`；`processing_cm3p.py:111-113` |
| 预训练是否含 mania | `gamemodes: [0,1,2,3]` | `configs/train/default.yaml:137` |
| **推理只需要 .osu** | `audio=None` 合法，`num_audio_tokens=0`，metadata 塔可完全不加载 | `processing_cm3p.py:459-476,520-529` |
| ⚠️ 音频陷阱 | `BeatmapFilesDataset` 在 `include_audio=True` 且音频加载失败时 **整张谱面 `continue`** | `utils/beatmap_files_dataset.py:236-248` |

**两条推论**：

1. 只要有 `.osu` 正文就能出 `e_m`，**不需要 metadata、不需要音频、不需要联网元数据**。
2. **"是否融合音频"必须固定成一个开关**：官方 244K 是带音频算的；我们补算只有 `.osu` 时只能 `include_audio=False`。
   两者**不可混用**，见 §9 R3。

---

## 3. 数据可行性（**已实测**，2026-10-09）

| 量 | 值 |
|---|---|
| 244K 集合里 mania（`ModeInt==3`） | 42,802 |
| 其中 4K（`Cs==4`） | **34,406** |
| 与 DSR 白名单 `beatmap_id` 交集 | **11,877 / 21,949 = 54.1%** |
| 交集内 `checksum` 一致率 | **98.78%**（余下 1.2% 是谱面被 mapper 更新过） |
| 覆盖的时间结构 | `last_update ≤ 2023` 几乎全覆盖；2024 / 2025 / 2026 = 0 / 48 / 97 张 |

**覆盖子集规模**（把 play 表限制到"有 embedding 的谱面"之后，重新数）：

| tag | step1 play | 玩家 | 图 | 有序对（step0 口径 = split 用的池子） | 训练 play |
|---|---|---|---|---|---|
| **1k** | 541,251 | 986 | 11,520 | **612,997**（测试对 183,663） | **378,750** |
| **10k** | 5,188,442 | 9,765 | 11,877 | （未测） | （未测） |

⚠️ 这两个数字要在 §4.3 一起看：**1k 覆盖子集的训练 play 只有 378,750 条**。
1k 每图训练 play 数 p10=2 / p25=5 / p50=13 / p75=42 / p90=114，`n ≤ 5` 的图占 **28.2%**（冷启动床位）；
10k 只有 0.6% ⇒ 稀疏冷启动只能在 1k 上做。

⚠️ 单文件 `beatmap_embeddings.parquet` 615 MB，本机网络实测**会中途断流**
（`curl: (18) end of response ... bytes missing`）—— 必须用**可续传/分块**方式取。

---

## 4. 目标模型（v2 规格）

### 4.1 式子

    L_hat(u, m) = b0 + bu[u] + bm[m] + <P_u, C_m>
      C_m : (d,)        冻结，= CM3P(e_m)（见 §5 管线 A）
      P_u : (n_ent, d)  自由，d 与 C_m 相同
      b0  : (1,)        自由
      bu  : (n_ent,)    自由
      bm  : (n_item,)   自由     <- 唯一的 per-chart 1-hot 残留

**参数量**（1k 覆盖子集，`n_ent = 986`，`n_item = 11,520`）：

| `d` | `P` 参数 | + 3 个偏置 ≈ 12,507 | 合计 | 对比 |
|---|---|---|---|---|
| 8 | 7,888 | | 20,395 | ≈ `free_K1` |
| 16 | 15,776 | | 28,283 | |
| 64 | 63,104 | | 75,611 | |
| 128 | 126,208 | | 138,715 | |
| **512（你的规格）** | **504,832** | | **517,339** | 比 `free_K16`（330k，0.7950）还大 |

这张表就是 §4.3 的全部理由。

### 4.2 为什么 v2 比 v1 干净：规范陷阱自动消失

v1 里最麻烦的一件事：`mirt` 的谱面因子有 **GL(K) 规范自由度**
（`P→PA^{-T}, C→CA^{-T}, D→DA`），所以写在 `C` 上的**软先验会被规范吃掉**
（[scripts/diag_step2_gauge.py](../scripts/diag_step2_gauge.py)）。

v2 里：

* **`C` 是常数，不是参数** ⇒ 没有任何"写在 `C` 上的先验"可以被吃掉；
* 平移规范 `C → C + a` 需要 `bu[u] → bu[u] - <P_u, a>` 补偿，
  但 `C` 被钉死 ⇒ 这个变换根本不允许；
* 剩下的 `P → P A^{-T}, C → C A` 要求 `C` 跟着变 ⇒ 同样不允许。

⇒ **模型对 `P` 是线性的、完全可识别的**。报告 `P_u` 时**不需要 Procrustes/CCA 对齐**，跨 run 直接可比。

### 4.3 `d` 就是 `P` 的参数量：必须扫，不能钉死 512

* 1k 覆盖子集的**训练 play 只有 378,750 条**，而 `d=512` 时 `P` 有 **504,832** 个参数 —— **参数比样本还多**（更别提还要和 12,507 个偏置竞争）；
* 仓库已有证据：**330k 自由参数的 `free_K16` 明显过拟合**（0.7950 vs 20k 参数的 0.7265，§1.1）；
* 但要注意：`C` 的 512 维**不是** 512 个自由度/谱面，它是 512 个**跨谱面共享**的坐标轴。
  过拟合压力落在 `P` 上，所以正则应加在 `P` 上（weight decay / ridge `λ`），而不是靠砍 `C`。

**处置**：`d` 作为第一类超参，扫 `{8, 16, 32, 64, 128, 512}`，每个 `d` 同时扫 `P` 的 weight decay。
降维方式两种都实现作为对照：**(a) PCA 白化到 `d`**（只在 train 谱面上拟合，防泄漏）、
**(b) 原始 512 维 + 强 wd**。若两条件结论一致，说明结论不是降维方式带来的。

### 4.4 冷启动时 `bm` 怎么办

`bm` 是 per-chart 自由参数，新谱面没有它。约定：

    bm[m] = 0        当 m 不在训练词表里（冷启动）

于是冷启动预测 = `b0 + bu[u] + <P_u, C_m>` —— **仍然是一个完整、有意义的预测**，
这正是"内容替换 1-hot"买到的东西（1-hot baseline 里这张图没有索引，**根本无法打分**）。
可选增强（不进主表）：把 `bm` 也内容化 `bm[m] = w_b·[C_m;1]`，但那就丢掉了"3 个偏置"的设定，作为附录实验。

---

## 5. 两条独立管线的接口契约

    ┌─ 管线 A（可在别的机器 / GPU 上跑）──────────────────────────┐
    │  .osu 语料 ──► CM3P beatmap tower ──► 每谱面 1×512          │
    │  HF 244K 预计算 ─────────────────────┐                     │
    │                                      ├─► 合并（按 id 去重） │
    │                                      ▼                     │
    │              data/processed/chart_content_cm3p.parquet     │
    └────────────────────────────────────────────────────────────┘
                                  │  只读；无 torch / transformers 依赖
    ┌─ 管线 B（step2）─────────────▼──────────────────────────────┐
    │  有值的谱面 ──► 过滤 play 表 ──► 30% 有序对划分 ──► 点积模型 │
    └────────────────────────────────────────────────────────────┘

**硬性要求：管线 B 不得 import `transformers`，也不得依赖 `external/cm3p`。**
step2 保持"纯 tabular 低秩拟合"，CM3P 只以一张 parquet 的形式出现。

### 5.1 管线 A：内容表

新脚本 `scripts/step0c_chart_content.py`，产物**一张表**（与 tag 无关，键是 `beatmap_id`）：

| 列 | 类型 | 说明 |
|---|---|---|
| `beatmap_id` | int64 | 主键 |
| `embedding` | list<float32>[512] | CM3P 输出，L2 归一化 |
| `n_windows` | int16 | 参与平均的 16 s 窗数（诊断用；`=1` 的谱面单独看） |
| `checksum_ok` | bool | 与 `beatmap_meta` 的 `checksum` 是否一致（False 单独标记，不静默丢） |
| `source` | int8 | 0 = HF 预计算 / 1 = 本地推理(无音频) / 2 = 本地推理(有音频) |
| `cm3p_rev` | string | 模型 revision，用于复现 |

两种输入模式：

* `--source hf-precomputed`：读 244K parquet，按 `ModeInt==3 & Cs==4` 过滤，与白名单求交；
* `--source osu-dir`：对本地 `.osu` 跑 CM3P（需要 `transformers` + HF checkpoint + GPU），
  **必须显式 `--audio {on,off}`**，并把结果写进 `source` 列。

### 5.2 合并与幂等

* **合并语义**：按 `beatmap_id` 求并集；冲突时**新行覆盖旧行**，但 `source` / `cm3p_rev` 保留，
  便于事后审计"这一行是哪种口径算出来的"。
* **幂等**：重复跑只补缺失的 id，不重算已有的（`--merge-with` 语义，参考
  `external/cm3p/extract_beatmap_embeddings.py:281-309`）。
* **原子写**：`.part` + `replace`（沿用 step0 / step1 的约定）。
* 产物**不入库**（`data/` 全在 `.gitignore` 里）。

### 5.3 管线 B：step2 侧

1. **先过滤、后划分**（顺序不能反）：把 step0 / step1 的 play 表先 inner join 到"有 embedding 的谱面"，
   **然后**再按 `(player_id, beatmap_id)` 有序对做 30% 划分（seed `20260928` 不变）。
   先划分后过滤会改变测试集构成，与 baseline 不可比。
2. **丢弃量必须报**：`n_rows_dropped_no_content`、`n_charts_dropped` 进 summary。
3. **行序断言**：`C[i]` 必须对应 `item_ids[i]`；加一条 checklist
   （这是整条链最容易静默错位的地方）。
4. `C` 作为 `register_buffer` 注册（不进 Adam、不参与梯度）。
5. **降维变换只在 train 谱面上拟合**，随 npz 一起存（`pca_mean` / `pca_components`），供复现。

CLI：

    --chart-content {none,cm3p}     # 默认 none，向后兼容
    --content-path PATH             # 默认 data/processed/chart_content_cm3p.parquet
    --content-dim N                 # 8/16/32/64/128/512；512 = 原始向量
    --content-reduce {pca,raw}      # 默认 pca（在 train 谱面上拟合）
    --models mf_dot                 # 本方案只用点积；mirt 保留但不在实验矩阵内

`EmbeddingModel` 的改动很小：删掉 `C`（和 `D`）的构造，
加一个 `register_buffer("C", ...)`，`forward` 里 `p["C"][m]` → `self.C[m]`：

    C = self.C[m] if self.content != "none" else self.p["C"][m]
    return out + (P * C).sum(1)

---

## 6. 消融阶梯与对照

全部在**覆盖子集 + 同一 split seed**上跑。

| # | 模型 | 谱面交互 | 玩家侧 | 目的 |
|---|---|---|---|---|
| **A0** | 1-hot 点积 `mf_dot` dim8 | `C` 自由（1-hot） | `P` (n_ent×8) | **新 baseline**（对应 §1.1 的 `free_K*`） |
| **A0b** | 现有内容锚定 `fm_dim4`（6 列去 star 元数据） | `C = V x_m` | `P` (n_ent×4) | 旁证（另一套标准化口径），**不作为门槛** |
| **A1** | **你的规格**：`C` 冻结 512 维原始向量 | 0 个自由参数 | `P` (n_ent×512) | 字面实现；风险见 §4.3 |
| **A2** | `C` = CM3P-PCA-`d`，`d ∈ {8,16,32,64,128}` | 0 | `P` (n_ent×d) | **主实验**：`d` 扫描 |
| **A3** | A1 + `P` 的 wd 扫描 | 0 | 同上 | 检查 A1 是否只是欠正则 |
| **A4** | `C = W_c e`（可训练投影） | `W_c` (d×512) | `P` (n_ent×d) | 对照：降维用"学"的而不是 PCA |
| **C1** | `C` 换随机高斯矩阵（同形状） | 0 | 同 A2 | 对照：不是"任意冻结向量都有用" |
| **C2** | `C` 按谱面随机置换 | 0 | 同 A2 | 对照：不是"容量 / 正则化"的功劳 |
| **C3** | `C` = 7 列元数据（复用 `fm` 口径） | 0 | 同 A2 | 同一脚本内的苹果对苹果对比 |
| **C4** | A2 但把 `star` 从 `e_m` 里回归掉 | 0 | 同 A2 | 泄漏控制，见 §9 R4 |

---

## 7. 冷启动评测协议（新增，必做）

现有 30% 有序对口径**测不出**内容表示的价值（每张图都在训练集里出现过）。加两个：

* **CS1（稀疏图）**：训练集里 play 数 `≤ {1,2,5,10}` 的图，单独报这些图的测试 play RMSE。
  1k 覆盖子集里 `n ≤ 5` 有 3,250 张图 / 9,388 plays。
* **CS2（整图留出）**：按 **`beatmapset_id`** 分组留出 20% 的图，训练集里一张都不出现。
  * 1-hot 模型（A0）：索引 `-1` ⇒ **覆盖率 0，RMSE 未定义**（这就是结论本身）；
  * v2 模型：`bm = 0`（§4.4），照常打分 ⇒ 报 RMSE / coverage。
  * 必须按 `beatmapset_id` 而不是 `beatmap_id` 留出，否则同曲的其他难度会漏答案
    （[docs/beatmap_embedding_plan.md](beatmap_embedding_plan.md) §3.2）。

新增 `--holdout-charts-by {none,beatmapset}`，并加一条 checklist：**留出谱面不出现在训练行里**。

---

## 8. 里程碑

| # | 内容 | 验收标准 | 成本 |
|---|---|---|---|
| ~~M0~~ | ~~独立诊断门（岭回归）~~ | **已跳过**：直接实现了 §4.1 的式子并全量实测，见 [docs/cm3p_step2_results.md](cm3p_step2_results.md) | — |
| **M1（管线 A）** | `scripts/step0c_chart_content.py` + `chart_content_cm3p.parquet`（先走 HF 预计算路线） | 覆盖 11,877 张；`checksum_ok` 与缺失率写进文档；重复运行幂等（第二次 0 新增） | 615 MB（需续传）+ 半小时 |
| **M2（管线 B）** | `step2_predict.py` 加 `--chart-content`，跑通 A0 / A1 / A2 | 先过滤后划分有断言；`C` 与 `item_ids` 行序断言；丢弃量进 summary；9 条 checklist 全过 | 1k 每模型分钟级 |
| **M3** | `d` 扫描（A1 / A2 / A3） | ✅ **已完成**：`d ∈ {8,16,32,64,128,512}`，内部最优 `d=32`（1.57595），512 最差（1.59759）；wd 未扫 | 6 次 1k 训练 |
| **M4** | CS1 / CS2 | CS2 上 A0 覆盖率 0、v2 > 0 且 RMSE 有限；CS1 分层表 | 需要 `--holdout-charts-by` |
| **M5（管线 A 补全）** | 下载缺的 10,072 张 `.osu` + CM3P 推理，合并进表 | 覆盖率 → ~100%；与 HF 重叠部分余弦 ≥ 0.99（否则音频开关没对齐，R3） | GPU 作业，不阻塞 M1–M4 |
| **M6** | 10k 复现 + 文档回填 | 10k 覆盖子集同结论；更新 `docs/step2_prediction.md` §7 与 `docs/overview.md` §8 | 小时级 |

---

## 9. 风险与未决问题

| # | 风险 | 影响 | 处置 |
|---|---|---|---|
| **R1** | `d=512` 时 `P` 有 504,832 个参数 | 过拟合（`free_K16` 的 330k 已经翻车） | §4.3：`d` 与 wd 双扫；A1 只作为"字面规格"报告 |
| **R2** | 覆盖率 54.1%，且全是 ≤2023 的图 | 外部效度：2024+ 的 9,217 张完全没有 | 写进文档；M5 补全；另报"2024+ 子集"对照 |
| **R3** | HF 预计算带音频、自算不带音频 | 两种 `C_m` 不可混用（等于注入与谱面无关的域偏移） | `source` 列 + M5 的重叠双算余弦 ≥ 0.99 |
| **R4** | CM3P 的 metadata 塔用 `DifficultyRating`（star）做对比学习 | `C_m` 可能携带 star ⇒ **step3 里只能当控制变量**，否则同义反复 | C4 对照；step3 表述按 [docs/beatmap_embedding_plan.md](beatmap_embedding_plan.md) §4.6 |
| **R5** | 每图 3–15 个 16 s 窗 → 官方脚本一律均值池化 | 丢掉难度剖面；`n_windows=1` 的谱面信息量尤低 | 先均值；再试 `[mean; std; max]` 拼接作为增量 |
| **R6** | 无 GPU，管线 A 的自算部分昂贵 | M5 阻塞 | 明确为独立 GPU 作业；M1 先走 HF 预计算 |
| **R7** | 换口径后与原 README 数字（1.4582 / 1.3331）不可比 | 误报进步 / 退步 | 所有表标 `subset = cm3p-covered`；A0 永远是同一张表的第一行 |
| **R8** | `bm` 与内容项可能部分共线（内容能解释的谱面均值会被 `bm` 吸走） | 低估内容的作用 | 报告 `--drop-bias m` 的对照（仓库已有该开关） |

**未决（需要拍板）**：

1. `C_i` 用**原始 512 维**还是**先 PCA 降维**？两种都满足"`P_i` 与 `C_i` 同维"，
   但 §4.3 的参数表说明 `d` 直接决定过拟合风险。
   *建议：两者都跑，主表用 `d` 扫描结果，512 作为其中一格。*
2. 是否接受"管线 B 先在有值的 54% 上跑"，其余等管线 A 补全？
3. `mirt` / `D` / `--d-constraint` 系列开关：**保留在代码里**（向后兼容、不删历史结论）还是清理掉？
   *建议保留*，只是不在本方案的实验矩阵内。
4. 是否现在就加 `--holdout-charts-by`（CS2 必需），还是先用现有口径出结论、冷启动留到第二轮？

---

## 10. 复现命令（计划，尚未可运行）

    # 管线 A（一次，产物与 tag 无关）
    python scripts/step0c_chart_content.py --source hf-precomputed \
        --merge-with data/processed/chart_content_cm3p.parquet \
        --out data/processed/chart_content_cm3p.parquet

    # M0 诊断门（不碰 step2）
    python scripts/diag_step2_content_inter.py --content cm3p --content-dim 64

    # 管线 B：A0（新 baseline，必须第一步）
    python scripts/step2_predict.py --tag 1k --models mf_dot --dim 8 --epochs 60

    # 管线 B：A1（你的规格，冻结 512 维）
    python scripts/step2_predict.py --tag 1k --models mf_dot --epochs 60 \
        --chart-content cm3p --content-dim 512 --content-reduce raw

    # 管线 B：A2（d 扫描）
    for d in 8 16 32 64 128; do
      python scripts/step2_predict.py --tag 1k --models mf_dot --epochs 60 \
        --chart-content cm3p --content-dim $d --content-reduce pca
    done

---

## 11. 一句话总结

`L = 3 个偏置 + <P_u, C_m>`，`C_m` 是冻结的 CM3P 向量、`P_u` 与它同维；
拿到这个式子之后模型对参数是线性的、完全可识别，v1 的规范陷阱自动消失；
唯一要小心的变成 `d` —— 它直接等于 `P` 的参数量（`d=512` 时 50 万），
而仓库已有的 `free_K16` 证据说明这个量级会过拟合，所以 `d` 要扫；
两条管线靠一张 `chart_content_cm3p.parquet` 解耦，step2 侧不 import transformers；
验收线是**本地 dim 扫描的最优值 1.454837**；实测结论（内容打平略优、真实时间漂移 −5.7%）见
[docs/cm3p_step2_results.md](cm3p_step2_results.md)。
