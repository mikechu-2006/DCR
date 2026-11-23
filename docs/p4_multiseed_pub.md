# 实验：官方发布 CM3P 向量在 P2 / P4 上的同协议复测（重叠子集，纯本地）

> 状态：**已执行**（2026-11-09 19:30 → 20:59，本地 8 核 CPU，无 HPC）。
> 对象是官方已发布、**音频在环**的 512 维图级向量（`hf-precomputed`），
> 只用它与我们 step1 谱面的**重叠子集**，协议与 [p4_multiseed.md](p4_multiseed.md) 相同。

---

## 0. 摘要

* **子集内，官方向量是最好的内容表**：与同一子集上的本地 audio-off legacy 表比，
  4 个方法在 P2/P4 上**全部变好**，ΔNLL = **−0.0114 … −0.0290**，
  40 个 seed×方法组合**全部为负**（5/5 × 8）。
  **注意**：这只在 4,042 图子集内成立；本项目目前最低的绝对 test NLL 仍是
  [p4_multiseed](p4_multiseed.md) 的全量（6,828 图）结果 **0.56602（P2）/ 0.57343（P4）**，
  两个口径的 ceiling（0.5613 vs 0.6169）和 test 集不同，**绝对 NLL 不可跨子集比较**。
* **增益结构**：CM3P-only 提升更大（P2 −0.0166/−0.0241，P4 −0.0210/−0.0290），
  叠 external 后收窄（P2 −0.0114/−0.0129，P4 −0.0152/−0.0149）；加 external 的增量
  （CM3P MLP → +external MLP）从 legacy 的 +0.031 降到 published 的 +0.020（P2）。
  这**不能**直接读成"官方向量与 msd/Sunny 冗余"：见 §4b，diminishing returns 与
  冗余在现有实验里不可区分，而且子集 ceiling 已失真。
* **canon 的代价在子集上复现**：canon − legacy = +0.0038 … +0.0187，
  8 个方法组合里 6 个 0/5、2 个 1/5 变好，与全量实验（+0.0047 … +0.0208）同向。
* **但子集复现不了"external MLP ≫ external 线性"**：两者差 −0.0001 … −0.0018
  （全量是 −0.0118 / −0.0114），且所有 external 栈的 captured > 1（最高 1.71）。
  子集更小、ceiling 更噪（P4 ceiling std 0.019），因此**子集只作方向复验**，
  主结论仍以 6,828 图的全量实验为准。
* 即便用官方向量，**CM3P-only 仍不如 legacy + external**（P2 0.6085 vs 0.6016）：
  外部内容信号的价值没有因为换更好的内容表而消失。

---

## 1. 背景与数据

### 1.1 官方表

| 项 | 值 |
|---|---|
| 文件 | `data/processed/chart_content_cm3p.parquet`（25 MB，11,877 行） |
| 维度 | 512，单位范数（0.9999999 … 1.0000001） |
| `source` | 0 = `hf-precomputed` |
| `n_windows` | 全部为空（Int16 NA）→ 无法核对窗口划分 |
| `checksum_ok=False` | 145 / 11,877（重叠子集内 66 / 4,042） |
| 与本地表关系 | mix report **DO NOT MIX**：同图余弦 mean 0.212 |
| 生成方式 | 官方脚本 `external/cm3p/extract_beatmap_embeddings.py`：`include_audio=True`、`include_beatmap=True`、`include_metadata=False`，处理器来自 checkpoint，bf16，逐窗归一化后取均值再归一化 |
| 官方 parser 配置 | [`configs/cm3p_hf/beatmap_parser/preprocessor_config.json`](../configs/cm3p_hf/beatmap_parser/preprocessor_config.json)：hitsounds/kiai/mania_sv = **true**、snapping = false、timing/timing_points = true；processor：max_length **2000**、window 16 s / stride 16 s |
| 跨图余弦（随机 2,000 图） | published **0.570**；本地 audio-off legacy 0.898；canon 0.963（audio-off 表示接近塌缩） |

### 1.2 重叠子集

| 项 | 全量 step1 | 官方重叠子集 | 占比 |
|---|---:|---:|---:|
| 谱面 | 6,828 | **4,042** | 59.2% |
| 行 | 109,706 | **60,039** | 54.7% |
| 玩家 id / cluster | 1,039 / 942 | 1,015 / 931 | — |

子集略简单、略冷门（msd 21.53→20.12、sunny 5.30→4.93、count_total 2,464→2,171）。
三臂都在同一子集上，所以臂间配对有效；子集 NLL 不能与全量 NLL 比。

### 1.3 历史锚点

[docs/cm3p_rank.md](cm3p_rank.md) §5.4 曾在同一 4,042 图上做过 P1 口径的一次性检查：
published linear strict 0.58799（subset ceiling 0.61258），同子集 local linear 0.61222；
方向与本实验一致（官方向量更好）。

---

## 2. 设计

| 臂 | content 表 | 作用 |
|---|---|---|
| **published** | `chart_content_cm3p.parquet`（官方，audio-on） | 主对象 |
| **legacy** | `chart_content_cm3p_local.parquet`（本地 audio-off） | 对照（主判据 published − legacy） |
| **canon** | `chart_content_cm3p_local_canon.parquet` | 复验 canon 结论在子集是否成立 |

协议与 [p4_multiseed.md](p4_multiseed.md) 一致，只换谱面集合：

* 子集表 `data/processed/step1_1k_pairbest_pub.parquet`（60,039 行）；
  在它上面**重新划分** 5 个种子（不能沿用原划分再筛行，否则 train 比例会漂到 ~55%）：
  P2 pair-random 70/30（train 41,962–42,057，frac 0.699–0.701；5 折 chart-disjoint）；
  P4 玩家 cluster 6:4 × 谱面 6:4（560/371 cluster、2,425/1,617 图；4×4 test 9,063–9,837 行）；
* 4 个方法：CM3P 线性 λ1e-3、CM3P MLP h64 L2（98 ep）、CM3P+external 线性、
  CM3P+external MLP（`[C, msd_overall, sunny_star]`）；模型 `--seed 0`、`--val-frac 0`、6 线程；
* 评测：`step2_rank_eval.py --ceiling-fit all --common-maps --strict-pairs`；
  三臂共享同一批划分文件，逐 (protocol, seed) 的 ceiling 完全相同（对比脚本内断言）。

**可比性边界**：只做同子集内的配对；不把子集 NLL 与全量 NLL 或其它子集比；
published 是"官方发布向量"整体，不能把 published − legacy 全算成"音频的功劳"。

### 2.5 子集 NLL 为什么整体高 ~0.04（不是模型变差）

把同一份 legacy 表分别放在两个"基准"上：

| 量（P2, seed 1） | 全量 6,828 | 子集 4,042（重划） | 子集 @ 原划分（诊断） |
|---|---:|---:|---:|
| test 行 | 32,971 | 17,982 | 17,831 |
| 评测比较数（strict 后） | 1,050,877 | 298,421 | 292,651 |
| test 每玩家图数（均值） | 33.8 | 19.7 | 19.1 |
| rows / chart | 16.1 | 14.9 | 14.9 |
| **ceiling NLL** | **0.5600** | **0.6166** | **0.6189** |
| legacy external MLP | 0.5657 | 0.5988 | —（训练集不同，不可比）|
| captured | 0.958 | 1.234 | — |

* **重新划分只贡献 ~0.002**：只把原划分限制到 4,042 图，ceiling 已是 0.6189，与新划分的
  0.6166 基本一致。**跳升来自图集/行数，不是划分种子。**
* **机制**：子集里每个 test 玩家平均只有 19.7 张图（全量 33.8），评测比较数从 1.05M 掉到
  0.30M，每张图的 rows 也从 16.1 降到 14.9；难度分布更窄（msd std 5.85→5.57）。
  每个比较的可预测性下降、per-chart 的 BT b 更噪 → ceiling 和模型 NLL 都上升。
* **模型升得比 ceiling 少**：+0.033 vs +0.057，所以 captured 从 0.958 变成 1.234
  （模型 NLL 低于 cross-fit ceiling）。**子集的 ceiling 已经不可信**。
* 同样的效应在 P4 更大（5 seed 均值）：ceiling 0.5654 → 0.6315（+0.066），
  legacy external MLP 0.5734 → 0.6101（+0.037）。

结论：跨子集只能看配对 Δ；绝对 NLL 只在同一基准内可比。全量
[p4_multiseed](p4_multiseed.md) 仍是唯一的主结果。

---

## 3. 结果（5 个划分种子 mean ± std）

### 3.1 三个内容表在同一子集上的 NLL

**P2（新 chart；ceiling 0.61687 ± 0.00272）**

| 方法 | published | legacy | canon |
|---|---|---|---|
| CM3P + external 线性 | **0.59032 ± 0.00248** | 0.60171 ± 0.00204 | 0.61841 ± 0.00323 |
| CM3P + external MLP | **0.58864 ± 0.00158** | 0.60158 ± 0.00249 | 0.62024 ± 0.00395 |
| CM3P 线性 (λ1e-3) | **0.61389 ± 0.00252** | 0.63050 ± 0.00148 | 0.63835 ± 0.00349 |
| CM3P MLP h64 L2 | **0.60852 ± 0.00294** | 0.63262 ± 0.00250 | 0.64326 ± 0.00301 |
| Sunny / Daniel / star | 0.63733 / 0.63686 / 0.66008 | 同左（不依赖内容表） | 同左 |

**P4（新玩家 + 新 chart；ceiling 0.63151 ± 0.01897）**

| 方法 | published | legacy | canon |
|---|---|---|---|
| CM3P + external 线性 | **0.59678 ± 0.00829** | 0.61194 ± 0.01253 | 0.62670 ± 0.00896 |
| CM3P + external MLP | **0.59521 ± 0.01193** | 0.61010 ± 0.01162 | 0.62806 ± 0.00927 |
| CM3P 线性 (λ1e-3) | **0.61751 ± 0.00709** | 0.63855 ± 0.00852 | 0.64341 ± 0.00492 |
| CM3P MLP h64 L2 | **0.61323 ± 0.01124** | 0.64218 ± 0.00568 | 0.64595 ± 0.00613 |
| Sunny / Daniel / star | 0.63822 / 0.63711 / 0.66061 | 同左 | 同左 |

### 3.2 主判据：published − legacy（同子集配对）

Δ < 0 = 官方向量更好；括号是 5 个种子的变好计数，"±" 是种子间 std。

| 方法 | P2 ΔNLL | P4 ΔNLL | ΔPearson (P2/P4) | ΔSpearman (P2/P4) |
|---|---|---|---|---|
| CM3P + external 线性 | **−0.01139 ± 0.00098**（5/5） | **−0.01516 ± 0.00454**（5/5） | +0.051 / +0.053 | +0.054 / +0.056 |
| CM3P + external MLP | **−0.01294 ± 0.00206**（5/5） | **−0.01489 ± 0.00483**（5/5） | +0.044 / +0.051 | +0.042 / +0.050 |
| CM3P 线性 (λ1e-3) | **−0.01661 ± 0.00184**（5/5） | **−0.02105 ± 0.00464**（5/5） | +0.059 / +0.070 | +0.050 / +0.058 |
| CM3P MLP h64 L2 | **−0.02409 ± 0.00384**（5/5） | **−0.02895 ± 0.00964**（5/5） | +0.083 / +0.090 | +0.061 / +0.064 |

逐种子 ΔNLL：

| 方法 | P2 s1 | s2 | s3 | s4 | s5 | P4 s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|---|---|---|---|
| CM3P + external 线性 | −.0116 | −.0119 | −.0109 | −.0125 | −.0100 | −.0113 | −.0172 | −.0093 | −.0185 | −.0194 |
| CM3P + external MLP | −.0113 | −.0128 | −.0121 | −.0165 | −.0120 | −.0094 | −.0194 | −.0178 | −.0099 | −.0180 |
| CM3P 线性 | −.0184 | −.0175 | −.0145 | −.0180 | −.0148 | −.0162 | −.0242 | −.0172 | −.0272 | −.0204 |
| CM3P MLP | −.0236 | −.0303 | −.0212 | −.0247 | −.0207 | −.0289 | −.0452 | −.0265 | −.0238 | −.0203 |

### 3.3 子集复验：canon − legacy

| 方法 | P2 ΔNLL | P4 ΔNLL |
|---|---|---|
| CM3P + external 线性 | +0.01670 ± 0.00298（0/5） | +0.01477 ± 0.00531（0/5） |
| CM3P + external MLP | +0.01865 ± 0.00250（0/5） | +0.01796 ± 0.00469（0/5） |
| CM3P 线性 (λ1e-3) | +0.00785 ± 0.00340（0/5） | +0.00485 ± 0.00376（1/5） |
| CM3P MLP h64 L2 | +0.01064 ± 0.00253（0/5） | +0.00377 ± 0.00577（1/5） |

方向与全量实验 [p4_multiseed_canon.md](p4_multiseed_canon.md) 一致（全量 +0.0047 … +0.0208），
说明"canon 去掉的 kiai/SV/hitsound 携带难度信息"这个结论不依赖 6,828 图这个特定集合。

### 3.4 参考：published − canon

P2 −0.0245 … −0.0347、P4 −0.0259 … −0.0329，全部 5/5 变好——三个内容表里 canon 最差。

### 3.5 captured 不能用在子集上

所有 external 栈的 captured 都 > 1（published 1.35–1.71、legacy 1.20–1.44、canon 0.96–1.17）。
子集只有 9–18k test 行、931 个 cluster，ceiling 系统性偏高（P4 ceiling std 0.019），
captured 失去上界含义。**本实验只看 strict NLL / Pearson / Spearman 的配对差**。

---

## 4. 解读

1. **在 4,042 图子集内，官方发布向量是三个内容表里最强的**，而且优势不是单点：
   4 个方法、2 个口径、40 个 seed×方法组合全部同向。这也意味着我们本地 audio-off
   重编码在这批图上丢掉了真实信息（与 §5.4 的 P1 一次性结果一致）。
   **但全局最优仍是全量 6,828 图的 legacy + external MLP（0.56602 / 0.57343）**——
   全量里另有 2,786 张没有官方向量的图，子集实验不覆盖它们，两者不可比绝对 NLL。
2. **增益结构只能描述，不能识别机制**：CM3P-only 提升 0.017–0.029，叠上 msd/Sunny
   后只剩 0.011–0.015；加 external 的增量（CM3P MLP → +external MLP）从 legacy 的
   +0.031 降到 published 的 +0.020（P2）。这至少与三种机制相容：
   (i) 官方向量与 msd/Sunny 信息重叠；(ii) 边际收益递减（起点 NLL 更低，同样提升
   换算成 nats 更小）；(iii) 模型容量/正则的差异。本实验没有设计去区分它们。
3. **但 external 仍然必要**：官方向量 + 外部特征（P2 0.5886）仍明显好于官方向量单用
   （0.6085–0.6139），也好于 legacy + external（0.6016）。
4. **canon 结论稳健**：在更小、更难的子集上复现同向代价，说明它不是全量集合的偶然。
5. **子集不能复现 external MLP 的优势**（−0.0001…−0.0018 vs 全量 −0.012）。
   两种解释：子集 test 集小 ~45% → 噪声大；或该优势部分依赖 6,828 图的谱面构成。
   现有证据不足以区分，**不要用子集结果推翻全量结论**（全量有 5 个种子、40 个配对）。
6. **方法排序**：published 臂里 P2 是 external MLP ≈ external 线性 > CM3P MLP > CM3P 线性；
   P4 亦然。即换更好的内容表没有改变"线性 vs MLP"的排序，只是把整体水平抬高。

---

## 4b. 本实验**不能**说明"音频信息是冗余的"

这是上一版摘要里的一个过强推论，在此收回。四条原因：

1. **"音频"大体可查，但仍不是受控变量**。官方仓库自带生成脚本
   `external/cm3p/extract_beatmap_embeddings.py`：`include_audio=True`、
   `include_beatmap=True`、处理器来自 checkpoint（`configs/cm3p_hf/`：
   hitsounds/kiai/mania_sv = true、snapping = false、timing = true、max_length 2000、
   16 s/16 s），bf16，逐窗归一化后取均值再归一化。也就是说 published 约等于
   **"我们的 legacy crop + 音频"**。但仍有三个口子：(i) 244K parquet 是否由这个脚本、
   这个权重 revision 生成，仓库内无法证明；(ii) 脚本用 `revision="main"`，权重会漂；
   (iii) 音频预处理（静音、采样）细节不可见。所以只能算"近似 audio ablation"，
   不能算严格因果。
2. **观测方向与"冗余"相反**。如果音频信息冗余，含音频的表示不该系统性更好；
   但 published 在每个方法、每个种子上都更好（−0.011…−0.029）。这更像
   "官方表示带来了我们 audio-off 表缺失的信息"。
3. **"增益收窄"不能识别冗余**。ΔNLL 随基线下移而变小是学习曲线常态；要证明冗余
   需要联合模型：`[C_pub, C_leg, S]` 相对 `[C_pub, S]`、`[C_leg, S]` 是否还有额外增益。
   本实验没有跑。
4. **子集 ceiling 已失真**：captured 1.35–1.71（模型 NLL 低于 cross-fit ceiling），
   P4 ceiling std 0.019。任何"还剩多少空间 / 是否饱和"的推理在子集上都不成立。

**旁证：audio-off 表示接近塌缩**。随机 2,000 张图的跨图余弦：published **0.570**、
legacy（audio-off）**0.898**、canon **0.963**。audio-off 表里几乎每张图都像同一个向量，
差异只靠很少的 token；canon 再把 hitsound/kiai/SV 删掉后更塌。这同时解释了：
(a) published 为什么全面更好；(b) 为什么 legacy/canon 上 external 特征的增量更大
（内容信号太弱，只能靠外部分数补）；(c) 为什么"音频冗余"读不通——音频正是让表示
可分的那部分信息。

**能稳妥写的句子**（本文件已按此修订）：
* published 在子集上全面好于本地 audio-off 表（配对 Δ，40/40）；
* 这个优势在 CM3P-only 模型上比在 external 栈上大；
* canon 的代价在子集上复现；
* 子集复现不了全量的 "external MLP ≫ external 线性"，因此子集只作方向复验。

**要回答"音频是否冗余"，至少需要**：
1. 用同一版代码在 4,042 图上做 **audio-on 重编码**（需 .osz/音频），把 audio 变成受控变量；
2. **联合模型** `[C_pub, C_leg, S]` vs 各自单表，直接检验互补/冗余；
3. 子集上的 **S-only 基线**（`--model ext_blend`），量化每张内容表的边际价值，
   而不是比较跨模型的 Δ。

---

## 5. 局限

* **官方配置大体可查，但 provenance 未证明**：生成脚本与 checkpoint parser 配置都在
  仓库里（audio on、hitsounds/kiai/SV on、snapping off、2000、16 s/16 s），但
  244K parquet 是否由这个脚本、这个权重 revision 生成无法证明；权重用 `revision="main"`
  会漂。因此 published − legacy 只能算"近似 audio ablation"，不能算严格因果。
* **子集选择偏差**：官方只发布了 11,877 图，与我们重合的 4,042 图偏简单、偏冷门
  （§1.2）；结论只在"这批可比的图"上成立，不能外推到全量 6,828 图。
* **ceiling 噪声**：P4 只有 931 cluster、4×4 test 9–10k 行，ceiling std 0.019，
  captured > 1；因此只用配对 ΔNLL，不用 captured。
* 只有 5 个划分种子、模型 seed 固定 0；std 是划分方差。
* 66 / 4,042 图 `checksum_ok=False`（后续 .osu 修订）保留在主分析里。
* 1k top-1000 切片、`n_plays ≥ 2`、ACC 截断等既有口径限制不变。
* P2 与 P4 的 NLL 不可互比（test 集、ceiling、pair 集不同）。

---

## 6. 复现

~~~bash
# 0) 与官方表重合的 step1 子集
python - <<'PY'
import pandas as pd, pyarrow.parquet as pq
s1 = pq.read_table("data/processed/step1_1k_pairbest.parquet").to_pandas()
pub = pq.read_table("data/processed/chart_content_cm3p.parquet").to_pandas()
s1[s1.beatmap_id.isin(set(pub.beatmap_id))].to_parquet(
    "data/processed/step1_1k_pairbest_pub.parquet", index=False)
PY

# 1) 在子集上重划 5 个种子
python scripts/step1_pairbest_resplit.py --in data/processed/step1_1k_pairbest_pub.parquet --out-prefix data/processed/step1_1k_pairbest_p2pub --seeds 1,2,3,4,5
python scripts/step1_p4_split.py --in data/processed/step1_1k_pairbest_pub.parquet --out-prefix data/processed/step1_1k_pairbest_p4pub --seeds 1,2,3,4,5

# 2) 三臂（published / legacy / canon）
for arm in published legacy canon; do
  case $arm in
    published) C=data/processed/chart_content_cm3p.parquet ;;
    legacy)    C=data/processed/chart_content_cm3p_local.parquet ;;
    canon)     C=data/processed/chart_content_cm3p_local_canon.parquet ;;
  esac
  python scripts/run_p2_p4_multiseed.py --protocols p2,p4 --seeds 1,2,3,4,5 --threads 6 --split-p2 data/processed/step1_1k_pairbest_p2pub --split-p4 data/processed/step1_1k_pairbest_p4pub --content-path "$C" --out-dir "results/p2p4_pub/$arm"
done

# 3) 配对对比
python scripts/compare_multiseed_runs.py --baseline results/p2p4_pub/legacy --variant results/p2p4_pub/published --out results/p2p4_pub/published_vs_legacy.md
python scripts/compare_multiseed_runs.py --baseline results/p2p4_pub/legacy --variant results/p2p4_pub/canon     --out results/p2p4_pub/canon_vs_legacy.md
python scripts/compare_multiseed_runs.py --baseline results/p2p4_pub/canon --variant results/p2p4_pub/published --out results/p2p4_pub/published_vs_canon.md
~~~

总耗时 1 h 29 min（三臂串行，6 线程，本地 CPU）。

---

## 7. 产物

| 文件 | 内容 |
|---|---|
| `data/processed/step1_1k_pairbest_pub.parquet` | 与官方表重合的 step1 行（60,039） |
| `data/processed/cm3p_pub_subset.json` | 子集覆盖报告 |
| `data/processed/step1_1k_pairbest_p2pub_s<seed>.parquet` | 子集 P2 划分 |
| `data/processed/step1_1k_pairbest_p4pub_s<seed>.parquet` | 子集 P4 划分 |
| `results/p2p4_pub/<arm>/summary*.csv` | 三臂逐 seed / mean-std 汇总 |
| `results/p2p4_pub/<arm>/<p>/s<seed>/rank_eval.json` | 官方评测（7 个信号） |
| `results/p2p4_pub/published_vs_legacy.md` / `.csv` | 主对比（§3.2） |
| `results/p2p4_pub/canon_vs_legacy.md` / `.csv` | 子集复验（§3.3） |
| `results/p2p4_pub/published_vs_canon.md` / `.csv` | 参考（§3.4） |

---

## 8. 后续可选

1. **把"音频"这一项分离**：本地对 4,042 图做 audio-on 重编码（需要 .osz/音频，本地
   `data/osu_cache` 只有 `.osu`），再按同一子集协议跑一遍，就能把 official − local
   拆成"音频 + 配置"两项。
2. **子集上检验 external MLP 的塌缩**：增加划分种子（10–20 个）或对配对 Δ 做
   cluster bootstrap，看 −0.0018 是否只是噪声。
3. **收紧 ceiling**：子集上改用更多 cross-fit 折或 player-disjoint 评价，避免 captured > 1。
