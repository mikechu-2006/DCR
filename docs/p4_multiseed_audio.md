# 实验：完整 audio-on CM3P 内容表在 P2 / P4 上的复跑（2026-11-18）

> 状态：**已执行**（2026-11-19）。代码/数据/结果均已落盘；见 §7–§8。
>
> 目标：把 HPC 新算的 2,782 张 audio-on 向量与官方 published 的 4,037 张合并成
> **6,819 张的完整 audio-on 表**，在与 [p4_multiseed.md](p4_multiseed.md) 相同的
> P2 / P4 协议下，与 audio-off legacy 表做同划分、同 ceiling 的配对对照。

---

## 0. 摘要

**结论**：完整 audio-on 表（published + local audio，6,819 图）在 P2 和 P4 上让
4 个 embedding 方法**全部变好，且 5/5 个划分种子同向**：

- **P2**：配对 ΔNLL = **−0.0083 … −0.0208**；CM3P-only 增益最大（线性 −0.0127、MLP −0.0208），
  叠 external 后收窄但仍显著（external 线性 −0.0100、external MLP −0.0083）。
- **P4**：配对 ΔNLL = **−0.0076 … −0.0218**；结构相同（CM3P 线性 −0.0123、MLP −0.0218、
  external 线性 −0.0108、external MLP −0.0076）。
- **方法排序不变**：CM3P + external MLP 仍是两个口径最好（P2 **0.55847**、P4 **0.56731**）；
  external MLP 的 P4 只比 P2 差 ~0.009 nats，与旧实验的 ~0.007 同量级（P2/P4 绝对 NLL 仍不可互比）。
- **配对前提成立**：audio/legacy 每 (protocol, seed) 的 ceiling 逐位相同；fixed Sunny/Daniel/star
  两臂逐位相同（最大 |Δ| = 0）。
- **不是数据变更造成的**：新 legacy 对照相对旧 `p4_multiseed` 只移动 **−0.0055 … +0.0018**
  （P2 全部 ≤ +0.0018），小于 audio 的配对效应；详见 §5。

**限制**：本表是 published 4,037 + local audio 2,782 的混合，全量增益不能单独归因给
HPC 新算的 2,782 张；要拆开需要在 local 子集上单独做 audio-on vs audio-off（见 §6）。

---

## 1. 数据边界：4 张缺口与白名单

### 1.1 无法编码的 4 张图

| beatmap_id | set | 原因 |
|---:|---:|---|
| 259067 | 92190 | `AudioFilename: virtual`，0.82 s 占位音频，无 16 s 窗口 |
| 374464 | 152065 | 同上，0.30 s |
| 486889 / 487516 | 206487 | .osz 里只有 .osu，没有音频文件 |

这些图在 audio-on 口径下没有可用向量；官方 audio-on 表同样出不了向量。按“在下游按缺失处理、
不做插补”的既定规则，从白名单移除。

### 1.2 白名单与重建

- [`manifests/whitelist_1k.csv`](../manifests/whitelist_1k.csv)：21,949 → **21,945** 行。
- [`manifests/whitelist_1k_removed.csv`](../manifests/whitelist_1k_removed.csv)：记录 4 个
  id、set、原因和日期，避免重新生成白名单时静默回填。
- 没有覆盖 canonical 的 `step0_1k` / `step1_1k_pairbest`；本次用版本化的 `_nogap` 产物，
  旧结果与旧文档仍可用旧白名单复现。

严格按“删图后重跑 step1”得到：

| 量 | 旧表 | 新表（本实验） |
|---|---:|---:|
| step0 play 行 | 1,399,615 | 1,398,611（−1,004） |
| pairbest 行 | 109,706 | **109,569**（−137） |
| 谱面 | 6,828 | **6,819**（−9） |
| base player / cluster | 942 | **941**（−1） |
| player segment | 1,039 | 1,039 |

减 9 张而不是 4 张的原因：玩家 `11566799` 在 2023-03-17 的 `487516` 被删后，
2022-04-11 → 2024-02-02 变成 662 天，按 365 天规则必须新开一个 epoch；另外 5 张
published 图（373650 / 476044 / 746952 / 916207 / 1383720）的两次 play 被拆到两个
epoch、各自变成单次对而被剔除。这不是额外清洗，而是删图后重跑 step1 的真实结果。

产物：[`data/processed/step1_1k_pairbest_nogap.parquet`](../data/processed/step1_1k_pairbest_nogap.parquet)
（md5 `276534f512d5dcc2f2d925942cd50a8a`）。

---

## 2. 内容表

### 2.1 组成

| 来源 | source | 行数 | 说明 |
|---|---:|---:|---|
| published（HF precomputed） | 0 | **4,037** | audio-on；bf16 GPU；官方 processor |
| local audio（HPC 新算） | 2 | **2,782** | audio-on；fp32 CPU；crop=legacy，max_length=2000，pad=longest |
| **合计** | | **6,819** | 覆盖新 pairbest 的 6,819 张，100% |

两表都是 audio-on；与 audio-off legacy 表（source=1）**不能混用**。published 的 4,042 张
里，5 张被上面的 epoch 副作用剔除，所以本表是 4,037。

产物：[`data/processed/chart_content_cm3p_audio_on.parquet`](../data/processed/chart_content_cm3p_audio_on.parquet)
（md5 `7180f2f4b40c857613d534396c09b4e4`）。sidecar meta 记录来源计数、配置、缺口和几何检查。

### 2.2 几何检查

| 子集 | 跨图 cos mean | 跨图 cos median |
|---|---:|---:|
| published（4,037） | 0.547 | 0.546 |
| local audio（2,782） | 0.544 | 0.547 |
| audio-off legacy（对照） | 0.893 | 0.899 |

两个 audio-on 子集的几何几乎一致，支持合并。published vs local 的 100 张探针
（[`cm3p_official_probe.md`](cm3p_official_probe.md)）平均 cos 0.99745、95/98 ≥ 0.99。

### 2.3 两个来源的谱面构成不同（解释时要注意）

| 子集 | 图数 | pairbest 行数 | msd_overall mean | sunny_star mean | count_total mean |
|---|---:|---:|---:|---:|---:|
| published | 4,037 | 60,026 | 20.13 | 4.94 | 2,173 |
| local audio | 2,782 | 49,543 | 23.58 | 5.84 | 2,892 |

local audio 子集整体更难、更热门（todo 表按 pair 行数优先计算）。配对检验是**同图内**的
audio-on − audio-off，所以这个选择效应不会直接制造 Δ；但它意味着“增益主要来自哪一批图”
不能从全量结果里直接读出，需要子集归因（见 §6）。

---

## 3. 协议

- **P2**：pair-random 70/30，5 折 chart-disjoint，`--chart-seed 0`；5 个划分种子 1–5。
- **P4**：玩家 cluster 6:4 × 谱面 6:4 分层；训练 6×6、测试 4×4；5 个划分种子 1–5。
- **方法**：CM3P 线性（λ=1e-3）、CM3P MLP h64 L2（98 epoch）、CM3P+external 线性、
  CM3P+external MLP（输入 `[C, msd_overall, sunny_star]`）、Sunny / Daniel / star。
- **两臂**：audio（本表）vs legacy（audio-off，`chart_content_cm3p_local.parquet`）；
  同划分、同 fixed signals、同 ceiling、模型 seed 0。
- 评测：`step2_rank_eval.py --ceiling-fit all --common-maps --strict-pairs`。

新划分（`step1_1k_pairbest_p2_nogap_s*` / `_p4_nogap_s*`）：

| protocol | seed | train rows | test rows | charts |
|---|---:|---:|---:|---:|
| P2 | 1–5 | 76,642–76,741 | 32,828–32,927 | 6,819 |
| P4 | 1–5 | 39,330–41,307 | 16,274–17,600 | 6,819（4,090 train / 2,729 test） |

---

## 4. 结果

### 4.1 audio 臂（5 个划分种子）

**P2（新 chart；ceiling NLL = 0.56104 ± 0.00394）**

| 方法 | test NLL | Pearson(d,b) | Spearman(d,b) |
|---|---|---|---|
| **CM3P + external MLP** | **0.55847 ± 0.00214** | **0.6551 ± 0.0082** | **0.6260 ± 0.0086** |
| CM3P + external 线性 | 0.56861 ± 0.00121 | 0.6293 ± 0.0116 | 0.6022 ± 0.0076 |
| CM3P MLP h64 L2 | 0.59665 ± 0.00366 | 0.6091 ± 0.0074 | 0.5673 ± 0.0089 |
| CM3P 线性 (λ1e-3) | 0.60529 ± 0.00311 | 0.5881 ± 0.0062 | 0.5489 ± 0.0065 |
| Sunny | 0.62651 ± 0.00231 | 0.3452 ± 0.0206 | 0.3810 ± 0.0116 |
| Daniel | 0.62647 ± 0.00239 | 0.3458 ± 0.0191 | 0.3814 ± 0.0114 |
| star | 0.65996 ± 0.00184 | 0.3103 ± 0.0197 | 0.3370 ± 0.0134 |

**P4（新玩家 + 新 chart；ceiling NLL = 0.56618 ± 0.01099）**

| 方法 | test NLL | Pearson(d,b) | Spearman(d,b) |
|---|---|---|---|
| **CM3P + external MLP** | **0.56731 ± 0.00990** | **0.6057 ± 0.0233** | **0.5757 ± 0.0144** |
| CM3P + external 线性 | 0.57395 ± 0.00911 | 0.5867 ± 0.0276 | 0.5669 ± 0.0226 |
| CM3P MLP h64 L2 | 0.59809 ± 0.00741 | 0.5691 ± 0.0292 | 0.5335 ± 0.0185 |
| CM3P 线性 (λ1e-3) | 0.60728 ± 0.01029 | 0.5475 ± 0.0303 | 0.5152 ± 0.0227 |
| Sunny | 0.62934 ± 0.00805 | 0.3274 ± 0.0380 | 0.3544 ± 0.0267 |
| Daniel | 0.62967 ± 0.00583 | 0.3260 ± 0.0419 | 0.3554 ± 0.0281 |
| star | 0.65949 ± 0.00604 | 0.2915 ± 0.0411 | 0.3129 ± 0.0277 |

### 4.2 与旧 `p4_multiseed` 的描述性对照（非配对）

旧实验 6,828 图、旧划分；本实验 6,819 图、新划分。Δ = audio − 旧结果，负值更好。
这里只作量级/方向参照，**不是**逐 seed 配对检验；主判据是 §4.3 的同划分 legacy 对照。

| 方法 | P2 Δ | P2 seeds 变好 | P4 Δ | P4 seeds 变好 |
|---|---:|---:|---:|---:|
| CM3P + external 线性 | −0.00924 | 5/5 | −0.01084 | 4/5 |
| CM3P + external MLP | −0.00756 | 4/5 | −0.00612 | 3/5 |
| CM3P 线性 | −0.01089 | 5/5 | −0.01683 | 5/5 |
| CM3P MLP h64 L2 | −0.01998 | 5/5 | −0.02731 | 5/5 |

fixed 三个信号在旧/新运行间只差 −0.0022…−0.0007，符合“不同 test pair 集/ceiling”的预期；
它们不依赖内容表，只作一致性检查。

### 4.3 配对主检验：audio vs 同划分 legacy

同 split seed、同 ceiling、同 fixed signals；Δ = audio − legacy，**负值 = audio 更好**。

**P2（新 chart）**

| 方法 | legacy NLL | audio NLL | ΔNLL | seeds 变好 | ΔPearson | ΔSpearman |
|---|---|---|---|---|---|---|
| **CM3P + external 线性** | 0.57859 ± 0.00185 | **0.56861 ± 0.00121** | **−0.00998 ± 0.00119** | **5/5** | +0.0528 | +0.0561 |
| **CM3P + external MLP** | 0.56677 ± 0.00117 | **0.55847 ± 0.00214** | **−0.00830 ± 0.00172** | **5/5** | +0.0295 | +0.0424 |
| **CM3P 线性 (λ1e-3)** | 0.61796 ± 0.00286 | **0.60529 ± 0.00311** | **−0.01266 ± 0.00098** | **5/5** | +0.0483 | +0.0412 |
| **CM3P MLP h64 L2** | 0.61741 ± 0.00389 | **0.59665 ± 0.00366** | **−0.02076 ± 0.00201** | **5/5** | +0.0628 | +0.0500 |

**P4（新玩家 + 新 chart）**

| 方法 | legacy NLL | audio NLL | ΔNLL | seeds 变好 | ΔPearson | ΔSpearman |
|---|---|---|---|---|---|---|
| **CM3P + external 线性** | 0.58478 ± 0.00771 | **0.57395 ± 0.00911** | **−0.01083 ± 0.00242** | **5/5** | +0.0483 | +0.0534 |
| **CM3P + external MLP** | 0.57488 ± 0.00867 | **0.56731 ± 0.00990** | **−0.00757 ± 0.00250** | **5/5** | +0.0275 | +0.0391 |
| **CM3P 线性 (λ1e-3)** | 0.61962 ± 0.00959 | **0.60728 ± 0.01029** | **−0.01234 ± 0.00230** | **5/5** | +0.0444 | +0.0427 |
| **CM3P MLP h64 L2** | 0.61988 ± 0.00713 | **0.59809 ± 0.00741** | **−0.02180 ± 0.00449** | **5/5** | +0.0615 | +0.0537 |

fixed Sunny / Daniel / star 在两臂逐位相同（最大 |Δ| = 0.00e+00），确认同一批 split 文件和
同一份 fixed 信号表。ceiling 也逐 (protocol, seed) 相同：P2 0.56104 ± 0.00394、
P4 0.56618 ± 0.01099。

逐种子 ΔNLL（audio − legacy，负值更好）：

| 方法 | P2 s1 | s2 | s3 | s4 | s5 | P4 s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|---|---|---|---|
| CM3P + external 线性 | −0.0109 | −0.0097 | −0.0084 | −0.0113 | −0.0096 | −0.0071 | −0.0105 | −0.0133 | −0.0127 | −0.0106 |
| CM3P + external MLP | −0.0103 | −0.0091 | −0.0087 | −0.0075 | −0.0058 | −0.0071 | −0.0044 | −0.0109 | −0.0090 | −0.0063 |
| CM3P 线性 | −0.0129 | −0.0141 | −0.0117 | −0.0128 | −0.0118 | −0.0112 | −0.0123 | −0.0143 | −0.0091 | −0.0147 |
| CM3P MLP h64 L2 | −0.0239 | −0.0211 | −0.0191 | −0.0207 | −0.0190 | −0.0216 | −0.0184 | −0.0294 | −0.0185 | −0.0210 |

### 4.4 训练侧证据

同一批模型/种子的 train NLL（P2 是 5 折拼出的 train 信号在全部 train 行上的 NLL，P4 是 6×6 单次拟合；两臂同协议，可比）：

| 方法 | P2 Δtrain | P2 seeds | P4 Δtrain | P4 seeds |
|---|---:|---:|---:|---:|
| CM3P + external 线性 | −0.01071 | 5/5 | −0.00392 | 5/5 |
| CM3P + external MLP | −0.00808 | 5/5 | −0.00019 | 3/5 |
| CM3P 线性 | −0.01173 | 5/5 | −0.00178 | 5/5 |
| CM3P MLP h64 L2 | −0.01985 | 5/5 | −0.00821 | 5/5 |

P2 的 train Δ 与 test Δ 几乎同幅（例如 CM3P MLP −0.0199 vs −0.0208），说明不是 test 端的偶然；
P4 的 external MLP train Δ ≈ 0 而 test Δ = −0.0076，更像 6×6 训练集只有 ~40k 行、模型已接近
容量上限，而不是内容表变差。

---

## 5. 数据变更检查：新 legacy 对照 vs 旧的 p4_multiseed

同一 audio-off 内容表（`chart_content_cm3p_local.parquet`），只把图集从 6,828 换成
6,819、并重新抽 5 个划分。Δ = legacy_new − old，**非配对**；它量化“少 9 张图 + 少 1 个玩家
+ 新划分”的移动，用来判断 §4.3 的 audio 效应是否可能由数据变更解释。

ceiling：P2 old 0.56126 ± 0.00348 → new 0.56104 ± 0.00394；P4 old 0.56540 ± 0.00859 →
new 0.56618 ± 0.01099。

| 方法 | P2 legacy−old | P2 audio−old | P4 legacy−old | P4 audio−old |
|---|---:|---:|---:|---:|
| CM3P + external 线性 | +0.00073 | −0.00924 | −0.00001 | −0.01084 |
| CM3P + external MLP | +0.00075 | −0.00756 | +0.00145 | −0.00612 |
| CM3P 线性 (λ1e-3) | +0.00178 | −0.01089 | −0.00449 | −0.01683 |
| CM3P MLP h64 L2 | +0.00077 | −0.01998 | −0.00552 | −0.02731 |

fixed：P2 Sunny −0.00217 / Daniel −0.00122 / star −0.00157；P4 −0.00177 / −0.00135 / −0.00072。
这些只来自 test pair 集/ceiling 变化，与内容表无关。

**读法**：

- 数据变更的最大移动是 P4 CM3P MLP 的 **−0.00552**；P2 全部 ≤ **+0.00178**。
- audio 的配对效应是 **−0.0076 … −0.0218**，比数据变更大 1.4–4 倍；方向也一致。
- `audio−old` = `audio−legacy`（§4.3）+ `legacy−old`（本表），三列自洽。
- 所以“audio 表更好”不是少 9 张图/新划分造成的；旧 `p4_multiseed` 仍是有效的外部参照。

---

## 6. 讨论与局限

1. **混合来源**：本表 = published + local audio-on，不是单一来源。因此全量增益不能全部
   归因给“HPC 新算的 2,782 张”；published 在 4,042 子集上已经比 legacy 好
   （[p4_multiseed_pub.md](p4_multiseed_pub.md)）。要单独回答新算向量的贡献，需要在
   2,782 张 local 子集上单独跑 audio-on vs audio-off。
2. **来源与难度/热度相关**：§2.3 显示 local 子集更难、更热门。配对 Δ 仍然有效，但
   “哪批图贡献了增益”需要子集分析。
3. **模型 seed 固定 0**：5 个种子只改变 train/test 划分；std 是划分方差，不是训练方差。
4. **图集与 p4_multiseed 不同**：本实验 6,819 图，旧实验 6,828 图；test 集、ceiling、
   pair 集都不同。绝对 NLL 不可跨图集比较，主判据是同一划分内的配对 Δ。
5. **1k top-1000 切片**、`n_plays ≥ 2`、ACC 截断等既有口径限制不变。
6. **P2 与 P4 不可互比**：test 集、ceiling、pair 数不同。

---

## 7. 复现

~~~bash
# 1) 白名单 + 过滤后的 step0 / step1（本次已生成，保留作审计）
#    manifests/whitelist_1k.csv 已删 4 行；step0_1k_nogap.csv 已过滤 1,004 条 play
python scripts/step1_pairbest.py \
    --tag 1k --in data/processed/step0_1k_nogap.csv \
    --out data/processed/step1_1k_pairbest_nogap.csv --no-csv \
    --train-frac 0.70 --split-method pair-random --split-seed 20260928 \
    --min-plays-per-pair 2 --max-best-acc 0.998 --acc-floor 0.90 \
    --gap-days 365 --min-plays-per-player 20

# 2) P2 / P4 5 个划分种子
python scripts/step1_pairbest_resplit.py \
    --in data/processed/step1_1k_pairbest_nogap.parquet \
    --out-prefix data/processed/step1_1k_pairbest_p2_nogap --seeds 1,2,3,4,5
python scripts/step1_p4_split.py \
    --in data/processed/step1_1k_pairbest_nogap.parquet \
    --out-prefix data/processed/step1_1k_pairbest_p4_nogap --seeds 1,2,3,4,5

# 3) 两臂
python scripts/run_p2_p4_multiseed.py --protocols p2,p4 --seeds 1,2,3,4,5 --threads 6 \
    --split-p2 data/processed/step1_1k_pairbest_p2_nogap \
    --split-p4 data/processed/step1_1k_pairbest_p4_nogap \
    --content-path data/processed/chart_content_cm3p_audio_on.parquet \
    --out-dir results/p4_multiseed_audio/audio
python scripts/run_p2_p4_multiseed.py --protocols p2,p4 --seeds 1,2,3,4,5 --threads 6 \
    --split-p2 data/processed/step1_1k_pairbest_p2_nogap \
    --split-p4 data/processed/step1_1k_pairbest_p4_nogap \
    --content-path data/processed/chart_content_cm3p_local.parquet \
    --out-dir results/p4_multiseed_audio/legacy

# 4) 配对主检验 + 数据变更检查
python scripts/compare_multiseed_runs.py \
    --baseline results/p4_multiseed_audio/legacy \
    --variant results/p4_multiseed_audio/audio \
    --out results/p4_multiseed_audio/audio_vs_legacy.md
python scripts/compare_multiseed_unpaired.py \
    --baseline results/p4_multiseed --variant results/p4_multiseed_audio/legacy \
    --out results/p4_multiseed_audio/legacy_vs_p4_multiseed.md

# 5) 三臂汇总（也可直接 bash scripts/watch_p4_multiseed_audio.sh 等两臂完成自动执行）
python scripts/summarize_p4_multiseed_audio.py \
    --out-prefix results/p4_multiseed_audio/combined_summary
~~~

---

## 8. 产物

| 文件 | 内容 |
|---|---|
| `manifests/whitelist_1k.csv` | 21,945 行（删 4 张 audio 缺口） |
| `manifests/whitelist_1k_removed.csv` | 4 张缺口的 id / set / 原因 |
| `data/processed/step0_1k_nogap.csv` | 过滤后的 step0 play 表（1,398,611 行） |
| `data/processed/step1_1k_pairbest_nogap.parquet` | 6,819 图 / 109,569 行 |
| `data/processed/chart_content_cm3p_audio_on.parquet` | 6,819 × 512 audio-on 内容表 |
| `results/p4_multiseed_audio/audio/` | audio 臂的 summary / rank_eval / signals |
| `results/p4_multiseed_audio/legacy/` | legacy 对照臂 |
| `results/p4_multiseed_audio/audio_vs_legacy.md` / `.csv` | 配对 ΔNLL / ΔPearson / ΔSpearman、逐 seed Δ |
| `results/p4_multiseed_audio/legacy_vs_p4_multiseed.md` / `.csv` | 数据变更检查（非配对） |
| `results/p4_multiseed_audio/audio_vs_p4_multiseed.md` / `.csv` | 旧实验参照（非配对） |
| `results/p4_multiseed_audio/combined_summary.md` / `.csv` | old / legacy / audio 三臂汇总 + 配对 Δ |
| `results/p4_multiseed_audio/analysis.log` / `ANALYSIS_DONE` | watcher 日志与完成标记 |
| `scripts/compare_multiseed_unpaired.py` | 不同划分下的非配对均值对照 |
| `scripts/summarize_p4_multiseed_audio.py` | 三臂汇总与配对 Δ |
| `scripts/watch_p4_multiseed_audio.sh` | 两臂完成后的后台分析 watcher |
| `legacy/docs/plans/p4_multiseed_audio_plan.md` | 本次手动执行计划（已归档） |
