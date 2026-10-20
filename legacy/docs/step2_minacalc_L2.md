# Step 2 — MinaCalc 的 gap / IRT 模型（L2）

**状态.** 已实现：`scripts/step2_minacalc_L2.py`，在 `dcr/steps.py` 里注册为
**`step2/minacalc_L2`**；`configs/exp_minacalc_L2_1k.yaml` 可一键跑（`python main.py`）。
上游资产是 [step0h_minacalc_msd.py](../scripts/step0h_minacalc_msd.py) 产出的
`minacalc_msd_{tag}.parquet`。

**一句话口径.** L1 把 MSD 当**加性/双线性特征**（`alpha·D + <P, C>`）；L2 把"谱面难度"与
"玩家技能"放进**同一把尺子**做差，再用与 `loss = log(1-ACC)` 精确对齐的 logit 链接：

    x(u,m) = a + bu[u] + bm[m] + sum_k w_k * (theta[u,k] - D[m,k])
    ACC    = sigmoid(x)
    loss   = log(1 - ACC) = -softplus(x)

`D` = 冻结的 MSD（训练谱面上逐技能 z-score）；`theta` = 可学的玩家技能向量（K = 技能数）；
`w_k = softplus(w_raw_k) > 0`，保证"技能越高 loss 越小、谱面越难 loss 越大"。

---

## 1. 为什么是 gap，不是点积

`b0 + bu[u] + bm[m] + <P_u, D_m>` 是**加性**的：某个方向的难度涨 1 点，对所有玩家的影响
都被同一个 `<P_u,·>` 线性项吸收。但实际语义是"**难度与能力比较**"——同一个 +1 MSD，
对高手几乎无感、对新手直接掉 ACC。gap 把这件事写进公式：`w_k (theta_k - D_k)`。

链接函数也不是随手挑的：标签是 `loss = log(1-ACC)`，若令 `ACC = sigmoid(x)`，则
`loss = -softplus(x)` 是**恒等式**，不需要再拍一个 "MSD → ACC" 的映射。MSE 仍在 loss 空间
（与 step2/predict 同一指标），所以两边数字可比。

## 2. 玩家侧向量 theta 怎么处理

| 项 | 做法 | 理由 |
|---|---|---|
| 来源 | **R1：可学**，`theta[u, k]` 每个实体每个技能一个自由度 | 不反解 SSR（R2），避免用测试局自身的历史把答案漏进去 |
| 初始化 | `--theta-init chartmean`：该玩家**训练局**所打谱面的平均 D | 直接落在"谱面难度"这把尺子里，收敛快；没有训练局的实体为 0 |
| 平移规范 | `gauge_fix()`：每维中心化，位移量折进 `b0` | `theta += c` 只改变 `b0`，是纯规范自由度；Adam 是逐参数归一化的，不钉住会漂 |
| 尺度规范 | `--theta-prior 1e-2`：每维 std 拉向 1 | `theta → a·theta, w → w/a` 是平坦方向，预测不变但数值不可解释 |
| 不吃 wd | `theta` 在 `model.no_decay` 里 | wd 把 theta 拉向 0、gauge 又把它中心化，两股力只会把尺度压小 |

**读数**：训练后 `theta` 的单位是"1 = 1 个谱面难度 sd"。`summary["gap"][model]["theta_std_per_skill"]`
应接近 1；`w_k` 是各技能方向的"区分度"——它小，说明这一维对 loss 几乎没有解释力。

## 3. 三个模型（`--models`）

| 名称 | 公式 | 用途 |
|---|---|---|
| `bias` | `b0 + bu + bm` | 同一谱面子集上的 step2/predict 基线，可直接比 |
| `gap` | 上面的完整式子（共享 `w_k`） | L2 本体：theta 与 D 同一把尺子 |
| `gapfree` | `a + bu + bm + sum alpha_k*theta_k - sum beta_k*D_k` | 消融：两边各一套权重，**没有**"同一把尺子"的约束 |

三者的差别只在交互项，`b0/bu/bm`、F1/F2、`p_drift`、`drop_bias` 完全共享。

## 4. 开关

| 参数 | 取值 | 含义 |
|---|---|---|
| `--link` | `logit`（默认）/ `identity` | logit 是 L2 本体；identity 退化成直接线性预测 loss（L1 式消融） |
| `--gap-mode` | `linear` / `hinge` / `both` | hinge 只惩罚 `relu(D - theta)`（"打不动"才扣分，对应 Chisel 的单边扣分模型） |
| `--gap-shared` | `on`（默认）/ `off` | 只作用于 `gap`；`gapfree` 固定 off |
| `--gap-bm` | `free`（默认）/ `msd` | `msd` 去掉 `bm[m]`：谱面截距完全由 `-sum w_k*D_k` 生成，每图自由参数 = 0 |
| `--theta-init` | `chartmean`（默认）/ `zero` | 见 §2 |
| `--theta-prior` | 默认 `1e-2` | 0 = 关（theta 尺度不可解释） |
| `--w-init` | 默认 `1.0` | `softplus` 之前的初始化；调小可以让模型从"接近 bias"起步 |
| `--msd-reduce` | `zscore`（默认）/ `raw` | `pca` 不允许：theta 会失去"每个技能一项"的语义 |
| `--msd-skills` | 默认 7 个方向 | `overall` 是它们自己的聚合，默认排除（共线） |

## 5. 产物

* `results/step2_{tag}_{model}_{features}_{gap 标签}_{target}.npz`：
  `theta` (n_ent, K)、`w`/`v`（/ `alpha`,`beta`）、`d_std` (n_item, K)、`b0/bu/bm`、`msd_cols`。
* `results/step2_{tag}_summary_{target}.json` 的 `runs[variant]` 里新增 `gap` 段：
  每模型的 `weights`、`theta_mean_per_skill`、`theta_std_per_skill`；
  `msd` 段记录资产路径、`calc_version`、覆盖率裁掉了多少行。
* `--save-preds` 时另存 `..._preds_minacalc_L2_{slug}_{target}.npz`（含 `pred_<model>`），
  可直接喂给 `legacy/scripts/compare_preds.py`。

## 6. 与 L1 / step2/predict 的可比性

* 划分（step1 的 `split` 列）、词表、损失、指标、检查项全部沿用 step2_predict；
* MSD 覆盖率过滤发生在**划分之前**（与 L1 同位置），所以 `bias` 就是同一子集上的基线；
* 与 `step2/minacalc_as_feature`（L1）的差别只有谱面项的用法，可以直接并排放进同一张表。

## 7. 首跑结果（1k，pair-random，60 epochs，lr 0.02，batch 8192）

| 配置 | test loss RMSE | 备注 |
|---|---|---|
| `bias` | **1.5271** | 与 `results/step2_1k_summary_loss.json` 里 L1 的 bias **完全一致** ⇒ 口径一致、可并排比较 |
| `gap`（bm free，logit） | 1.5810 | w 收敛到 `[0.29, 0.02, 0.05, 0.07, 0.52, 0.16, 0.06]` |
| `gapfree`（bm free，logit） | 1.5837 | 与 gap 同量级 |
| `gap`（identity 链接） | 1.5274 | **w → 0**：优化器主动把 gap 关掉，退化成 bias |
| `gap`（`--gap-bm msd`，logit） | 1.7649 | 谱面截距完全由 MSD 生成，明显差于自由 `bm` |
| `gap`（`--gap-bm msd` + hinge） | 1.7859 | 同上 |

对照 L1 已跑过的同子集数字：`msd` 1.5477、`msd_add` 1.5667、`msd --msd-bm msd` 1.7137；
CM3P `mf_dot`（dim16）为 1.4548。

**结论（只对 pair-random + 1k）**：MSD 没有跑赢"自由玩家/谱面截距"的基线。identity 链接下
优化器把 w 压到 0，是最直接的证据。原因很朴素：`bm` 有 19,640 个自由参数，MSD 只有 7 维，
且这 7 维还背着 LN 缺失、OD 缺失、Etterna 键型与 mania 键型不完全对齐的系统误差。
L2 的价值必须在 `bm` **不可用**的设定下才可能体现（整图留出 / 冷启动谱面），见 §8.6。

## 8. 已知坑

1. **goal=0.93 是 Etterna 的 score 语义**，不是 mania ACC。所以 D 只在训练谱面上标准化、
   只当相对量用；绝对刻度交给 `b0/bu/bm`（`--gap-bm free`）或由 `-sum w_k*D_k` 生成（`msd`）。
2. **LN/长按**：MinaCalc 的 NoteInfo 没有时长维度。建议用资产表里的 `hold_ratio` 做 rice 子集
   的对照实验（尚未接入特征）。
3. **OD 不在 MSD 里**，而标签是 ACC 的函数：下一步应从 `beatmap_meta` join `diff_overall`
   做谱面主效应，否则 gap 会吸收一部分"判定窗口"的方差。
4. **`--gap-bm free` 时模型对谱面仍有每图自由截距**，gap 只是叠加项；想看"完全由 MSD 生成
   谱面截距"的版本用 `--gap-bm msd`（每图自由参数 = 0，对冷启动谱面更诚实）。
5. **每个技能维的 w 单独可读**，但 7 个方向高度相关：w 小不代表该方向不重要，可能只是被
   另一维先解释掉了。要做"哪个方向重要"的结论，建议逐个方向单独跑一次。
6. **整图留出（beatmap-random）目前评测不了**：词表只从训练行建，测试谱面若没在训练集出现过，
   `map_ids` 返回 -1、这些行会被 `te_ok` 过滤掉（coverage → 0）。要让 MSD 在冷启动上发挥作用，
   需要给 `--gap-bm msd` 加一个 content-only 回退：未登录谱面用 MSD 表的 D、`bm` 记为一个共享的
   `bm_unseen`。这是 L3 之前最值得做的一步。

## 9. 下一步（未实现）

* **R2**：用 `step0h --goal` 网格产出 `D_m(g)`，反解每局隐含技能做零训练基线（只用该局之前的历史）。
* **L3**：缓存逐 interval 难度，直接用 MinaCalc 自己的得分模型前向预测，再校准到 mania ACC。
* **OD / hold_ratio**：作为谱面主效应加进来，和 `--gap-bm msd` 是同一类"谱面侧协变量"。
