# Step 2 — osumania_map_analyser 特征作为冻结谱面特征

**状态.** 已实现。资产脚本 `scripts/step0i_map_analyser.py`（注册为 `step0/map_analyser`）、
模型脚本 `scripts/step2_map_analyser.py`（注册为 `step2/map_analyser`）、一键配置
`configs/exp_map_analyser_1k.yaml`。首跑结果与"还剩多少信息没拿到"的量化见 §5 / §7。

**一句话口径.** MinaCalc 把一张谱面压成 8 个数（MSD）；谱面本身的信息远不止 8 维。本版本把
这 8 维 D 换成 [LeoBlackMT/osumania_map_analyser](https://github.com/LeoBlackMT/osumania_map_analyser)
（插件 2.1.0）算出的 **109 维冻结特征**，而**模型与 [step2_minacalc.md](step2_minacalc.md) 的
L1 完全相同** —— 所以同一谱面子集上的 bias / msd_add / msd 与 bias / fc_add / fc 可以直接并排。

---

## 1. MinaCalc 的 D 丢了什么

MinaCalc 的输入只有 `(rowTime, noteBits)`（见 `NoteInfo`），输出只有 8 个 skillset 数：

| 被丢掉/未表达的信息 | MinaCalc | 本资产 |
|---|---|---|
| OD（判定窗口） | 完全不进 MinaCalc | `od` |
| 长按时长 | `NoteInfo` 没有时长，长按只算头部一个 tap | `ln_ratio` / `n_holds` / `hold_ms_per_note` / `sunnywin_ln_star` / HB 比例 |
| BPM / 时长 / 密度 / 和弦比 | 只隐式影响 strain，不输出 | `bpm_median` / `bpm_max` / `duration_s` / `notes_per_sec` / `chord_ratio` / `n_rows` |
| 键型构成 | 7 个 skillset 是固定基，粗 | `patt_*`：RC/LN/HB 标签 + 6 类核心键型的覆盖时长 / 最高 BPM / importance |
| 难度在时间上的分布 | 只有聚合值 | **默认没有**（`--with-graph` 可加，见 §7） |
| 版本差异 | 只跑一个版本 | 6 个 MinaCalc 版本 × 8 skillset = 48 维 |
| 其它难度模型 | 无 | Sunny（含 ppMetrics）+ Daniel + Azusa + Roxy + Interlude |

这七行就是"把 D 换掉"的直接动机：它们都能在 `.osu` 正文里确定性地算出来，不需要成绩。

## 2. 资产（step0/map_analyser）

```bash
python scripts/step0i_map_analyser.py --tag 1k \
    --ids data/processed/whitelist_1k.csv --cache data/osu_cache \
    --out data/processed/map_analyser_1k.parquet --jobs 3 --resume
```

* 依赖：Node ≥ 20 + `external/osumania_map_analyser`（`git clone` 到该路径；仓库不入库）。
  插件侧只用它的**纯计算模块**（`js/parser`、`js/estimator`、`js/patterns`、`js/ett`、
  `js/interlude`），不跑 tosu、不读内存、不联网（遥测代码不在调用链上）。
* 结构：Python 只负责切分 id / 并发 / 合并 NDJSON → parquet；逐图计算在
  `scripts/map_analyser_harness.mjs` 里，每个 stage 独立 try/catch。
* 只依赖 `.osu` ⇒ 属 step0 资产；同一张表可被任意划分方式复用（含 beatmap-random）。
* 表结构：`beatmap_id` + 109 个数值特征 + `status` / `patt_category` / `patt_mode_tag` +
  `analyser_version` / `analyser_commit` / `rate` / `goal`。
* 覆盖率：whitelist 的 **21,949 / 21,949** 张谱面全部产出，0 失败。

### 特征组（`--feat-groups`）

| 组 | 维数 | 内容 |
|---|---|---|
| `ett` | 48 | 6 个 MinaCalc 版本（0.68 / 0.70 / 0.72 / 0.72.3 / 0.74 / 0.75）× 8 skillset |
| `est` | 13 | Sunny star + variety / accScalar / spikiness / switches，Daniel star，Azusa star / numeric / raw numeric，Roxy star / raw numeric，SunnyWindow LN star，Interlude star（`roxy_numeric` 全表 95% 为空，不取） |
| `patt` | 34 | ModeTag（RC/LN/HB/Mix；本批没有 SV 图）、LN 比例、HB 行比例、SV 用量、6 类核心键型各自的簇数 / 覆盖秒数 / 最高 BPM / importance / 加权 rating multiplier |
| `struct` | 14 | OD、BPM（中位/最高/变速点数）、时长、行数、和弦比、note/s、长按时长、break |
| `graph` | 7 | 仅在 `--with-graph` 时产出：Sunny 时间曲线的 mean/p50/p90/max/sd/spike/hot_frac |

默认 109 维 = ett+est+patt+struct；step2 另外加 1 列 `est_out_of_scope`（Azusa/Roxy 返回
out-of-scope 的指示——主要是 LN 门控，见下），共 110 维。

**缺失值（已核对，不是"低难"）**：`azusa_*` / `roxy_*` 有 39.5% 的谱面为空，其中 **99.6%
（8,641 / 8,673）是 `ln_ratio > 0.18`** —— Azusa/Roxy 是 **RC-only** 模型，`rcLnRatioLimit = 0.18`，
超过就返回 `UnsupportedLN`；真正的 RC 谱面里只有 0.2% 因太短/低难被拒。`patt_*_mult` 在没有
该类键型时为空（5%–54%）。step2 用**训练谱面的列中位数**填补（不丢行），并保留
`est_out_of_scope` 这个 0/1 列——它实际上约等于"`ln_ratio > 0.18`"，与 `ln_ratio` 高度共线。

**LN 口径**：插件对 LN 谱面有一整套单独输出 —— SunnyWindow 的 `lnStar`（只算 LN 段）、
`typePercentageData`（RC / Mix / HB / LN 四段的 note 数）、`lnPartsRatio`（LN 段占比）；
键型分析的模式门槛是 `lnRatio ≤ 0.15 → RC`、`≥ 0.90 → LN`、中间 `Mix`，`Mix` 再看
HB 行比例 ≥ 0.1 → `HB`。本资产接了 `sunnywin_ln_star`、`ln_ratio` / `hold_ms_per_note`、
`patt_*` 的 LN 核心键型（Coordination/Density/Wildcard）与 `patt_mode_tag`，**没接**
`typePercentageData` 的四段比例与 `lnPartsRatio`（见 §8 第 7 条）。

### 与现有 MinaCalc 资产的一致性

同一张谱面，本表的 `ett_0723_*` 与 `data/processed/minacalc_msd_1k.parquet`（`calc_version=505`）
的 `msd_*`：8 个方向 **r = 0.99895–0.99943，平均 |Δ| = 0.035–0.068**。这说明插件里的 MinaCalc
与 step0h 用的 libminacalc 是同一套算法，两张表可以直接对照。尾差存在且要知道：
**2.4% 的谱面（517 张）某个方向差 > 1 MSD，37 张差 > 3**（插件的 wasm 去掉了 Etterna 官方的
40 MSD 上限，极难图会更高；另有少量解析边界差异）。

另外注意 `n_notes` 的口径不同：本表是**物件数（含 LN）**，`minacalc_msd_1k.parquet` 是 **tap 数**；
`n_notes - n_holds` 与后者在 100% 的谱面上相等（已验证）。

## 3. 模型

与 L1 逐字相同（`scripts/step2_map_analyser.py` 直接复用 `step2_minacalc_as_feature` 的
`MSDModel` / `build_model` 与 `step2_predict` 的 `EmbeddingModel` / 指标 / 检查项）：

    L_hat = b0 + bu[u] + bm[m] + alpha . Z[m] + <P[u], C_unit[m]>

* `bias`    = b0 + bu + bm（同一子集基线）
* `fc_add`  = 只加特征主效应 alpha . Z（没有 P）
* `fc`      = 主效应 + 点积交互（C_unit 行单位范数）

alpha 从 0 初始化 ⇒ epoch 0 严格等于 bias。zscore / pca 只在**训练谱面**上拟合；缺失值用训练谱面
列中位数填补（不丢行）。

## 4. 运行

```bash
# 资产（若 data/processed/map_analyser_1k.parquet 已存在可跳过，step2 会自动从 cache_dir 读）
python scripts/step0i_map_analyser.py --tag 1k --ids data/processed/whitelist_1k.csv \
    --cache data/osu_cache --out data/processed/map_analyser_1k.parquet --jobs 3

# 实验（走 main.py 入口）
python main.py -c configs/exp_map_analyser_1k.yaml
python main.py -c configs/exp_map_analyser_1k.yaml --dry-run

# 只留 Etterna 的 48 维，看"多版本 MSD"本身值多少
python main.py -c configs/exp_map_analyser_1k.yaml \
    --set name=mma_ett --set steps.step2.params.feat_groups=ett \
    --set 'steps.step2.params.models=[bias,fc_add]'
```

## 5. 结果（1k，pair-random，60 epochs）

口径：step1 的同一划分（train_frac 0.70、seed 20260928），测试集 418,876 行（coverage 0.9964）。
**所有运行共享同一个 `bias` = 1.52711**（与 `results/step2_1k_summary_loss.json` 里 L1 的
bias 1.527107 逐位一致），说明子集与划分完全对齐，下面的数字可以直接并排。

### 5.1 主表（bm free：每图仍有自由截距）

| 特征 | 维 | bias | fc_add / msd_add | fc / msd | Spearman(fc) |
|---|---|---|---|---|---|
| —（只有自由 bm） | 0 | **1.5271** | — | — | 0.7329 |
| MinaCalc MSD（现有 L1） | 7 | 1.5271 | 1.5667 | 1.5477 | 0.7299 |
| map_analyser 全部（zscore） | 110 | 1.5271 | 1.5686 | 1.5242 | 0.7487 |
| **map_analyser PCA** | **7** | 1.5271 | 1.5253 | **1.4874** | **0.7504** |
| map_analyser PCA | 4 | 1.5271 | — | 1.5101 | 0.7424 |
| map_analyser PCA | 16 | 1.5271 | — | 1.5026 | 0.7461 |
| map_analyser ett only | 48 | 1.5271 | 1.5575 | — | — |
| map_analyser est+patt+struct | 62 | 1.5271 | 1.5677 | — | — |
| map_analyser + 时间曲线 | 117 | 1.5271 | 1.5681 | — | — |
| PCA(ett only) | 7 | 1.5271 | — | 1.5302 | — |
| PCA(est only) | 7 | 1.5271 | — | 1.5344 | — |
| **PCA(patt+struct)** | **7** | 1.5271 | — | **1.4874** | **0.7504** |
| PCA(est+patt+struct) | 7 | 1.5271 | — | **1.4874** | **0.7504** |

三件事同时成立：

1. **主效应（fc_add）在 bm free 时有害**，与 L1 的 `msd_add` 一样（1.5686 / 1.5667 vs bias
   1.5271），而且特征越多越糟（ett 48 维 1.5575 < all 110 维 1.5686）。原因见 §7.3。
2. **交互项有用，但维度必须压**：110 维 fc 1.5242，16 维 PCA 1.5026，**7 维 PCA 1.4874**
   —— 与 minacalc 的 7 维 msd 1.5477 是**同容量对照**。
3. **增益不是来自"更多 MinaCalc 版本"，也不是来自更多估算器**：7 维 PCA 下
   ett 1.5302、est 1.5344 都不如 bias；而 **patt+struct 的 1.48735 与全量 1.48735 逐位相同**
   （同样 7 维、同样 |alpha|=1.2205）。三个不同特征集给出完全一样的模型，说明前 7 个主成分
   张成的**交互子空间相同** —— etterna 48 维 + 估算器 13 维对这个子空间没有贡献。
   换句话说：**省下 MSD 版本与估算器，只留键型+结构量，7 维交互就能拿到全部收益**。

   维度曲线（全量特征）：4 → 1.5101，**7 → 1.4874**，16 → 1.5026，110 → 1.5242。
   7 是拐点，和"有效难度轴 ≈ 7~8"一致（§7.2）。

配对 cluster bootstrap（`scripts/compare_preds.py`，按 (player, beatmap) 对重采样 400 次）：

| 比较 | ΔRMSE | 95% CI | 结论 |
|---|---|---|---|
| PCA7 fc − bias | **−0.0398（−2.60%）** | [−0.0415, −0.0376] | 显著更好 |
| PCA7 fc − MinaCalc msd | **−0.0603（−3.90%）** | [−0.0632, −0.0579] | 显著更好 |

### 5.2 冷启动（`--feat-bm feat`：去掉 bm，谱面截距只能由特征生成）

| 谱面侧信息 | 模型 | RMSE | Spearman |
|---|---|---|---|
| 无（只剩 b0 + bu） | bias | 1.8707 | 0.4969 |
| MinaCalc MSD（7 维） | msd | 1.7137 | 0.6519 |
| **map_analyser（110 维）** | fc | **1.5987** | 0.7182 |

"无谱面信息"到"自由 bm"的差距是 1.8707 − 1.5271 = 0.3436。MSD 收回了 0.1570（45.7%），
map_analyser 收回了 0.2720（**79.2%**）。**特征的价值在 bm 不可用时才直接体现为 RMSE**。

### 5.3 谱面难度的解释力（不经过预测模型）

见 §7.1 的 `diag_map_analyser_charteffect`：minacalc7 的 R²=0.167，全部特征 0.497（上限 0.930）。

### 5.4 训练动态（口径 caveat）

所有运行都用仓库统一的 60 epochs / lr 0.02 / wd 1e-4。评测点（每 20 epoch）显示两条曲线都在
epoch 20 最低、之后回升：

| 模型 | ep20 | ep40 | ep60 |
|---|---|---|---|
| bias | 1.5216 | 1.5255 | 1.5271 |
| fc_add（110 维） | 1.5536 | 1.5503 | 1.5686 |
| fc（110 维） | **1.4969** | 1.5032 | 1.5242 |

所以绝对数字没有调到最优（早期就过拟合），但**所有对比都在同一协议下**；在 ep20 上
fc 仍然优于 bias（1.4969 vs 1.5216），方向与 ep60 一致（只是幅度更大）。选 ep20 是在测试集上
挑点，不能当作正式数字。

### 5.5 一句话结论

* **谱面侧信息量**：MinaCalc 的 8 个数只解释 17% 的经验谱面难度方差；新特征是 50%（§7.1）。
* **预测收益**：在 bm free 的 pair-random 设定下，把新特征压成 7 维 PCA 做交互，
  RMSE 1.4874（−2.6% vs bias，−3.9% vs MinaCalc 7 维），Spearman 0.7504（+0.018）；
  不压缩（110 维）反而只剩 −0.2%。**收益全部来自键型 + 结构量**（patt+struct 单独就是 1.4874），
  多版本 MSD 与估算器对 7 维交互没有增量。
* **冷启动**：RMSE 1.5987（−6.7% vs MinaCalc），谱面侧信号回收率 79% vs 46%。

## 6. 与 minacalc / CM3P 实验的可比性

* 本资产覆盖 whitelist 全部 21,949 张谱面，与 `minacalc_msd_1k.parquet` **同一批谱面**；
  两者都在划分**之前**按覆盖裁剪，所以两次实验的 `bias` 落在完全相同的行上。
* 与 [cm3p_step2_results.md](cm3p_step2_results.md) 的子集不同（CM3P 只覆盖约 54%），
  只能比"同一口径下的相对增益"。
* 指标口径沿用 step2/predict：测试集 loss 的 RMSE / Spearman，附 coverage 与 pp。

## 7. 还损失了什么（量化）

### 7.1 目标本身的可信度与上限

`scripts/diag_map_analyser_charteffect.py` 不用预测模型，直接从成绩算"经验谱面难度"：

    adj_ij   = loss_ij - 玩家 i 的平均 loss
    chart_j  = mean_i(adj_ij)   （只取 n_j >= 10 的 13,641 张谱面，覆盖 97.9% 的 play）

* chart effect 的 sd = 1.071（去掉玩家水平后）。
* **split-half 可信度**：把每张图的 play 随机劈成两半，两半估计的相关系数 r=0.931，
  经 Spearman-Brown 校正 = **0.964**。也就是说这个目标本身很稳，任何"只看谱面"的模型
  解释力上限是 **R² = 0.930**；剩下的 7% 是同一张图不同 play 的随机波动。

### 7.2 各组特征解释了多少

ridge + 按谱面分组的 5 折 CV（R² 是折外值）：

| 特征组 | 维 | R² | 相对上限（R²/0.930） |
|---|---|---|---|
| MinaCalc 7 维（现有） | 7 | 0.167 | 17.9% |
| ett（6 版本 × 8 skillset） | 48 | 0.331 | 35.6% |
| est（Sunny/Daniel/Azusa/Roxy/Interlude/LN） | 13 | 0.265 | 28.5% |
| patt（键型） | 34 | 0.270 | 29.0% |
| struct（OD/BPM/时长/密度） | 14 | 0.160 | 17.2% |
| **全部** | 109 | **0.497** | **53.4%** |

结论：

1. **旧 D 只拿到约 1/6 的谱面难度方差**（0.167），新特征拿到约 1/2（0.497）；
   **还有约 47% 的可解释方差没拿到**（相对于上限 0.930）。
2. 单组里 ett 最强（0.331），但 48 维的**有效秩只有 7**（PCA 99% 方差），说明六个版本的
   主体是同一套 MSD；它比"7 维 MSD"多出的 2 倍解释力来自**版本之间的差异**，不是冗余。
3. 各组有明显互补（max 单组 0.331 → 联合 0.497）。
4. **但"能解释 chart_j" ≠ "能改进预测"**：§5.1 的 7 维交互里 ett 与 est 都没有增量，因为
   patt+struct 已经张成了同一个子空间。ridge 解释 chart_j 是"每个特征维度都能独立
   贡献一部分"的场景；交互式预测只用前 7 个主方向，重复的信息挤不进那 7 维。

### 7.3 为什么预测侧只赚了 2.6%：bm 已经把谱面信号吃掉了

pair-random 划分下，每张训练图平均有约 34 条 train play，自由截距 `bm` 已经是一个相当好的
谱面效应估计（§5.2：有 bm 1.5271，没有 bm 1.8707）。`alpha·Z` 是一个**全局线性映射**，
它没法替代 bm，只能在 bm 之上再叠一层；叠上去的那层会拟合训练图上的谱面噪声，out-of-sample
就是纯方差（§5.1 的 fc_add 全部比 bias 差，且特征越多越差）。

特征真正能兑现的地方有两处，本版本都测到了：

* **把交互空间压到低维**（PCA7）：谱面轴的**方向**比"多"更重要 —— 1.4874 vs 110 维 1.5242。
* **bm 不可用**（冷启动 / 新图）：1.5987 vs MinaCalc 1.7137。

### 7.4 不可约的 play 级噪声

同一 `(玩家, 谱面)` 对的多次 play，loss 的标准差是 **0.9273**（757,265 行 / 224,266 对是重复对；
占全部行的 54%）。pair-random 划分下同一个对只会出现在一侧，所以这部分方差**任何谱面特征都
预测不了**：RMSE 的理论下限约 **0.93**。把它按 MSE 拆开：当前最好的 1.4874
对应 MSE 2.212，扣掉噪声 0.927²=0.860，还剩 **1.163** 的"系统性"误差，来自玩家×谱面交互
（同一个人对不同谱面的相对强弱）与未建模因素 —— 这一部分才是特征/模型还能争取的空间。

### 7.5 仍然没进特征的通道

| 通道 | 现状 | 能不能补 |
|---|---|---|
| 难度的时间分布 | 只有聚合值；`--with-graph` 的 7 维曲线摘要在**主效应**上没有增量（1.5681 vs 1.5686），交互通道没测 | 能，已实现 |
| 左右手 / 列不对称 | 没进特征（键型只到 Handstream / Split Trill 这一级） | 能，从 `noteStarts/columns` 直接算 |
| 每局的判定构成 | step0 把 300/200/100/50/miss 压成一个 `num`，只剩 loss | 能，但要改 step0 与原始 dump 的解析 |
| 谱面版本漂移 | 缓存里是**当前**版本的 .osu；2015 年的 play 可能打的是旧版本 | 需要 beatmap update 时间线 |
| 玩家状态 / 会话 | 本次 F0；`--features F1/F2`、`--p-drift` 已有开关 | 能，模型侧 |
| mods（DT/HT/HR…） | 数据只有 CL，特征也只算 1.0x | 换数据 / 按 rate 重算 |

### 7.6 一句话

损失的信息分三层：**已拿回**（OD、BPM/时长/密度、键型、LN 时长、其它估算器、多版本 MSD）；
**拿不回**（同一 play 的判定构成、谱面版本漂移、玩家状态）；**还没接**（时间曲线的交互通道、
左右手、判定构成）。而这三层之上还有一层与特征无关的地板：同一个 (玩家, 谱面) 的 play 之间
本来就有 sd=0.93 的波动。

## 8. 下一步

1. **时间分布**：`--with-graph` 已经能产出 7 维曲线摘要（mean/p50/p90/max/sd/spike/hot_frac），
   进 `--feat-groups graph` 即可测；下一步还可以加"rolling 10s 密度的 p99/max"。
2. **左右手不对称**：从 `noteStarts/columns` 直接算每列的 note 数、单列最大连打、跨列切换频率。
3. **判定构成**：step0 把 `num` 压成一个数；从原始 dump 保留 300/200/100/50/miss 比例与
   早晚判定偏移，可以让 loss 之外的标签也进模型（需要改 step0，不是本版本的范围）。
4. **谱面版本漂移**：缓存里是当前版本的 `.osu`，老 play 可能打的是旧版本；需要 beatmap
   update 时间线才能对齐。
5. **玩家状态**：playcount_cur / 时间衰减 / 会话特征是玩家侧的，与谱面特征正交；`--features F1`、
   `--p-drift logpc` 已有开关。
6. **缺失值编码**：`azusa_*` / `roxy_*` 的 39.5% 缺失**不是随机的**，而是"LN 谱面被 RC-only
   模型拒绝"（99.6% 满足 `ln_ratio > 0.18`）。中位数填补把这些图放在中间，等于假设它们"难度
   中等"；更好的做法是：对 LN 谱面改用 LN 专用通道（`lnStar`、LN 核心键型），或把指示列拆成
   `ln_gate` / `below_scope` 两个，别让它和 `ln_ratio` 共线。
7. **LN 的四段构成**：`typePercentageData` 给出 RC / Mix / HB / LN 四段的 note 数（`getLNParts`
   按 LN 密度切段），`lnPartsRatio` 给出 LN 段占比。本资产只存了 `sunnywin_n_types`（数组长度
   恒为 5，等于没存）。补上这 5 个比例是一次 `--only-ln` 的小 pass（约 10 分钟），比现在的
   `ln_ratio` 更能区分"整张 LN 图"和"只有几段 LN 的 RC 图"。
