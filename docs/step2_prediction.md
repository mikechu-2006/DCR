# Step 2 — 预测任务（play → loss）

**目标.** 在 step1 清洗后的 play 表上训练**预测模型**：给定一条游玩，预测它的 `loss`（成绩）。
本步只做**预测**，产出玩家/谱面的潜向量；**定数（官方 star）回归是 step3**，不在本步范围内。

**状态.** 方案 + 已实现并运行（`scripts/step2_predict.py`，**PyTorch 实现**；旧的手写 numpy
实现退役到 `scripts/legacy/step2_predict_numpy.py`，只作对照）。1k 与 10k 都已跑完主表，1k 另有三组消融（§7）。

**一句话口径.** 观测单位是 **play**（一条游玩一行），标签是 `loss`，模型是
**`(玩家, 谱面) → loss` 的低秩潜向量回归**；切分是**固定 seed 下随机 30% 的
`(player_id, beatmap_id)` 有序对**作测试集；**训练读 step1，测试读 step0**；
主指标是测试集上的 **play 级 RMSE**。谱面向量 `C/D` 是交给 step3 的接口。

---

## 1. 这一步在哪里

| 步骤 | 名称 | 输入 | 输出 | 本文档 |
|---|---|---|---|---|
| 0 | 预处理 | 原始 dump | `step0_{tag}.csv`（play 表，5 列）+ `beatmap_meta_{tag}.csv` | [step0_preprocessing.md](step0_preprocessing.md) |
| 1 | 数据清洗 | step0 play 表 | `step1_{tag}.csv`（同 5 列，行更少） | [step1_cleaning.md](step1_cleaning.md) |
| **2** | **预测任务** | **step1（训练）+ step0（测试）** | **`step2_{tag}_{model}_{features}_{target}.npz` + `step2_{tag}_summary_{target}.json`** | **yes** |
| 3 | 定数回归 | step2 的谱面向量 + `beatmap_meta` | 官方 star（定数）的回归模型与误差 | no（另立文档） |

**边界一句话：step2 的标签是 `loss`（可观测的行为），step3 的标签是 `star`（谱面属性）。**
step2 不知道也不使用 star；step3 不训练 play 级模型，只把 step2 学到的 `C/D/bm` 当特征去回归 star。
这与旧管线一致（旧 `train_stage1*.py` = 本步；旧 `stage2_*.py` = step3）。

**step2 不产任何中间表。** 它读 step1 与 step0，产出一组潜向量 `.npz` 与一份报告 `.json`。

### 与旧实现（stage-1）的关系

本步的**方法**沿用旧实现，但**总体不同、必须重跑**。旧 `train_stage1*.py` 读的是
`plays_v2_*` / `charts_v2_*_v3.parquet`，那是**已被 step0 取代的总体**（NM+DT+HT 三档、
`chart_id = beatmap_id*4 + rate`、含 `NF`、cell 中位 ACC 门槛 + playcount/n_scores 门槛）。

| 项 | 旧（stage-1） | 本版（step2） |
|---|---|---|
| 曲目 | `chart_id = beatmap_id*4 + rate`，三档 | **`beatmap_id` 就是曲目**，只有 1.0×（无 `rate` 列） |
| 观测 | 1,875,436 局（1k，含 DT/HT） | 674,436 训练 play（1k，全 1.0×） |
| 清洗 | ACC 中位 + attempts + n_scores 三条门槛 | **行级两条**：逐条 ACC > 0.90、每玩家最早 30% 删除 |
| 标签范围 | `loss ≤ 0`，max ≈ −0.27 | `loss ∈ [−12.4663, **−2.3026**]`（step1 的痕迹） |
| 单位 | cell（中位标签）+ play | **play（唯一口径，无聚合）** |
| 切分 | 10% cell 留出 | **30% 有序对留出**；**测试集取自 step0（未清洗）** |
| 元数据 | 已 join 进 cell 表 | step2 **完全不用**（star 属 step3） |

→ **旧 README §7 的 mirt 数字（1k 0.9332/0.7900、10k 1.0605/0.6043）不可直接搬用**，只可作为量级参考。

---

## 2. 任务定义

### 2.1 标签：`loss`，不是 `ACC`

沿用 step0 §4.1 的定义：`loss = log(1 − ACC)`，**递减**于 ACC（越小 = 打得越好）。

用 loss 而不是 ACC 的理由：

1. **与上游一致**：step0/step1 的列就是 `loss`，ACC 门槛也是在 loss 空间判的，全流程一个尺度。
2. **误差可加**：loss 是 `1−ACC` 的对数，所以 loss 空间的残差 ≈ `1−ACC` 的**乘性**误差；
   报告时换算成 pp 很直接：`|ΔACC| = |exp(loss_pred) − exp(loss_true)| · 100`。
3. **步长均匀**：ACC 在高分段（0.98→0.99）的分辨率远低于 loss 空间，直接用 ACC 做 MSE 会让高分段样本几乎不起作用。

代价（必须知道）：loss 单位自带 `1/(1−ACC)` 放大因子，所以**高 ACC 的样本在 loss 空间被放大**，
同一模型在 ACC 单位下的"学习率排序"会翻转（见 [learning_rate_heterogeneity.md](learning_rate_heterogeneity.md)）。
报告时**同时给 loss 与 pp 两套数**（§7 就是这么报的）。

### 2.2 观测单位：**play，唯一口径**

> **一行 = play 表的一条记录。** 训练、评价、指标全部在 play 上。

本版**不做任何聚合**：

- **不建 cell 表**（`(player, beatmap)` 的中间表不存在）；
- **不取成绩中位 / 均值**，不做 cell 级标签；
- 不做基于 cell 的回归预测。

后果（正面）：

- 标签就是观测到的**那一次**游玩，没有"用中位数代表一段时间水平"的近似；
- 顺序与时间信息天然保留，不需要靠聚合列带出来；
- step1 的两条行级规则与 step2 的单位**一致**（都是行），链路没有"行 → cell → 行"的来回。

后果（负面，见 §9）：

- 单次游玩的抖动（loss 单位 σ ≈ 0.65）**全部**留在标签里，不能靠 cell 中位降噪；
- 因此**不能再算 split-half 标签噪声，也就没有 `excess` 这个扣噪指标**（§6）。
  取而代之的是"**固定测试集**"——测试集对所有配置完全相同，RMSE 直接可比。

**顺序特征用 `playcount_cur`，不另造 `seq`。** F1（§2.4）直接取 step1 已有的
`playcount_cur` 作 `ln(playcount_cur)`，**不做重排、不重编号**。
注意它在 step1 删行后有空洞（[step1_cleaning.md](step1_cleaning.md) §2.4：1k 13.10% / 10k 13.75% 的
pair 有空洞），所以 `ln(playcount_cur)` 是"**该 cell 的第几条记录**"而不是"第几把" —— 这是刻意保留的原义。

### 2.3 泛化设定：**留出 30% 的 `(player_id, beatmap_id)` 有序对**（唯一设定）

```
pairs      = 有序对全集（默认取自 step0）
test_pairs = 固定 seed 下随机 30%
train      = step1 中 pair ∉ test_pairs 的全部 play      （step1，清洗后）
test       = step0 中 pair ∈  test_pairs 的全部 play      （step0，未清洗）
```

- **切分单位是有序对**，不是 cell、不是谱面、不是 play。一个 pair 只会落在一边。
- **谱面与玩家都不留出**：留出的只是"某个玩家打某张图"这一对。所以每个测试谱面在训练里
  仍有别的玩家的 play，模型**能给测试对打分**（不像"留出谱面"那样是冷启动）。§7 实测覆盖率 **99.6%**。
- **测试集取自 step0，这是刻意的。** step1 的两条规则（ACC > 0.90、每玩家最早 30%）是
  **训练侧的处理手段**；用它们去筛测试集就是"以处理结果为条件"（[cleaning_plan.md](cleaning_plan.md) §3.2）。
  所以测试标签用**未清洗**的 step0：它包含低 ACC 的乱打与入门期游玩，是**目标总体**的样子。
  模型在清洗过的数据上训练，却在没清洗过的数据上被评价 —— 这正是要测的东西。
- **固定 seed**（`--split-seed`，默认 `20260928`），单次划分。测试集对所有配置完全相同，
  RMSE **跨配置直接可比**，不需要 `excess` 之类的相对量。
- 有序对池子由 `--pair-universe` 决定，默认 `step0`（见 §12.1 的讨论与 §7 的对照）。

### 2.4 特征三档

| 档 | 特征 | 默认 | 依据 |
|---|---|---|---|
| **F0 静态** | 只用 `(player_id, beatmap_id)` 的 id → 潜向量 `P/C/D` + 偏置 | **✔** | 旧 stage-1 基线 |
| **F1 + 顺序** | 加 `ln(playcount_cur)`（§2.2，原值，不重排） | 可选 | cell 内 ΔACC ≈ **0.0134·ln(n)**，60–100 次饱和；`beta_loss` 均值 **−0.377**（[play_order_analysis.md](play_order_analysis.md)、[learning_rate_heterogeneity.md](learning_rate_heterogeneity.md)） |
| **F2 + 时间** | 玩家侧换成 `(player_id, 自然年)` 实体 + 年内线性漂移 | **✘（非默认，保留开关）** | 练习占 80–90%、日历漂移 10–20%（cell 内 γ≈0.146 pp/年）；冷启动首把 ≈1.0–1.1 pp/年（[practice_vs_time.md](practice_vs_time.md)） |

**F2 默认关闭**，`--features F2 --entity player-year` 显式开启（§4.2）。

**明确不做：**

- **不做遗忘核**。`forgetting_curve.md` 的结论是 **1h–2y 区间没有遗忘**；"旧练习影响小"是**凹性**
  （边际 ∝ 1/j），不是衰减。若需要"最近性"，只加 `Δt_prev < 10min` 的**同会话指示**（连打收益低 0.2 pp）。
- **不把原始 `timestamp` 当特征**。绝对日期是典型的泄漏风险特征（step0 §13.5）；F2 只从它派生日历年与 `tau`。
- **不 join `osu_user_beatmap_playcount`（attempts）**。step0 的 `playcount_cur` 已是从记录算的下界，
  step1 也没引入 attempts；引入它会重新定义总体（旧版剔除②），留作后续消融而非默认。

---

## 3. 数据准备：直接用 step0 / step1，不产中间表

step2 **直接读 `step1_{tag}`（训练）与 `step0_{tag}`（测试）**（`.parquet` 优先），不写任何中间表。
内存里只做三件事：

| 步骤 | 说明 |
|---|---|
| 有序对键 | `key = player_id · (max(beatmap_id)+1) + beatmap_id`（int64，单射），`np.unique` 后按固定 seed 抽 30% |
| 词表 | **只从训练行**建 entity / item 索引；测试行映射不上的记为"未覆盖" |
| F1/F2 特征 | `ln(playcount_cur)`；F2 另算 `year` 与 `tau = (dayofyear−1)/365` |

**刻意不做的事：**

| 不做 | 为什么 |
|---|---|
| cell 表 / cell 聚合 / 中位 / 均值 / max / min | 观测单位是 play（§2.2） |
| split-half 标签噪声、`SE`、`excess`、训练权重 | 都建立在 cell 聚合上，随 cell 表一起取消（§6） |
| 重排 `seq` / 重编号 `playcount_cur` | 用 step1 原值（§2.2） |
| 收敛档判定（D0–D3） | 旧 `convergence_filter.py` 读 `chart_id` 且建在旧总体上，**不可直接用** |
| 再做一遍 ACC 门槛 / `attempts ≤ 2` | step1 已做行级筛选；playcount 表按 step0 决定不 join |
| 谱面元数据 / 元数据特征 | 属 step3；step2 完全不用（pair 级切分不需要冷谱面外推） |

---

## 4. 模型族

统一写成：

```
L̂ = b0 + b_u + b_m + 交互项
```

| 名称 | 交互项 | 参数 | 说明 |
|---|---|---|---|
| `bias` | — | `b0, b_u, b_m` | 只有偏置，基线 |
| `mf_dot` | `<P_u, C_m>` | `P, C` | 经典矩阵分解 |
| **`mirt`** | `Σ_i (P_ui − C_mi)·D_mi` | `P, C, D` | 本项目的签名模型（README §7）；`C` 是"水平向量"、`D` 是"区分度" |

优化用 Adam（与旧实现一致），损失 MSE；梯度由 autograd 给出、优化器是 `torch.optim.Adam`
（`scripts/step2_predict.py`）。`--device auto` 默认选 CUDA（没有则 CPU），`--threads` 可钉线程数。

**实现注记（torch 引擎）.**
参数是普通 `nn.Parameter`；损失 `((pred − y)²).mean()`，反向走 autograd，优化器
`torch.optim.Adam(lr, weight_decay=wd)`（与旧手写 Adam 同一公式：`p -= lr·m̂/(√v̂+eps)`，wd 加在梯度上）。
**只把模型真正用到的参数交给 Adam**（`bias` 不碰 `P/C/D`），嵌入梯度就是稠密张量（`index_add`），
不需要稀疏技巧或手写 backward。

**与旧 numpy 实现的数值关系.** 初始化与 batch 顺序仍取 `numpy.random.default_rng(seed)` 的**同一串随机数**
（P/C/D 的抽取顺序也一致），所以两者从**逐位相同的初始参数**出发。实测（同一 batch、同一初值）：
前向最大差 `2e-9`、梯度 `5e-7`、5 步 Adam 后参数 `6e-8` —— 只有浮点求和次序的差别，
文档 §7 的数字因此可以直接沿用（见 §7 的 engine 注记）。

### 4.1 不可辨识性（step3 必须知道）

`mirt` 里 `(P − C)·D` 只有 `C·D` 与 `D` 可识别，**`C` 本身有 `(dim−1)` 维规范自由度**。
旧实测：`C` 单独回归 star 几乎没用（R² 0.017–0.122），必须 **`C + D` 一起用**（README §8）。
→ step2 的产物**必须同时输出 `C` 和 `D`**，只给 `C` 会让 step3 无从下手。

### 4.2 实体扩展（F2）：**非默认，保留开关**

默认实体是 `player_id`。`--features F2 --entity player-year` 时，玩家侧换成
`(player_id, 自然年)` 实体，年内线性漂移：

```
P_eff(u, y, t) = P[u, y] + (tau − 0.5)·V[u, y]        tau = 年内第几天 / 365
```

（`train_stage1_yearly.py` 的同一形式）。**默认关闭**（`V ≡ 0`），但代码路径与 CLI 开关保留。
§7 给了它打开后的实测（1k）：RMSE 明显下降，但**覆盖率同时从 99.6% 掉到 83.8%**
（测试年份的 `(player, year)` 实体常常在训练里没有），两者不能直接比。

### 4.3 超参默认值

| 超参 | 默认 | 说明 |
|---|---|---|
| `dim` | 16 | 1k 实测 **8 与 16 等价**（§7e），建议 1k 用 8；10k 未扫 |
| `wd`（weight decay） | 1e-4 | 旧版 wd=1e-6 过拟合、1e-3 欠拟合，1e-4 最优（README §6） |
| `lr` / `optim` | 0.02 / Adam | 沿用 |
| `epochs` | 60（固定，不早停） | 保持"单一测试集"的干净性 |
| 初始化 | `std=0.05`，偏置置 0 | 沿用 |
| 损失 | MSE | — |

### 4.4 两个可选机制（2026-10-04 新增，默认关）

**(a) 玩家侧漂移 `--p-drift logpc`** —— 把 F1 的"全局标量斜率"升级为"每玩家、每维的斜率"：

```
P_eff[u] = P[u] + Pc[u] · ln(playcount_cur)      Pc ∈ R^(n_entity × dim)，初始化为 0
```

* `Pc` 与 `P` 同形，是**漂移速率向量**（每维一个学习速度）；`ln(playcount_cur)` 仍取 step1 原值（§2.2）。
* 初始化 **0**：从"无漂移"出发，漂移必须自己挣出来（与 F2 的 `V` 同一约定）。
* 与 F1 的标量 `w1` **正交、可叠加**（`--features F1 --p-drift logpc` 同时有全局项与每玩家项）；
  与 F2 的 `V` 是两种不同的漂移轴（`V` 沿自然年 `tau`，`Pc` 沿记录序号 `ln(playcount_cur)`）。
* 可辨识性：`P` 与 `Pc` 靠**同一玩家跨谱面的 `ln(playcount_cur)` 变化**分开；单条记录的 pair 只提供
  `P + Pc·x1` 这一个组合（1k 占 71.5%），所以 `Pc` 是玩家级、不是 pair 级参数。

**(b) D 的约束 `--d-constraint {nonneg,simplex}`** —— 用重参数化**精确**满足，不做投影、不裁剪：

| 取值 | 变换 | 含义 |
|---|---|---|
| `none`（默认） | `D = Z` | 自由，与原实现一致 |
| `nonneg` | `D = softplus(Z)` | `D > 0`，仍可整体缩放 |
| `simplex` | `D = softmax(Z, dim)` | `D ≥ 0` **且** `Σ_i D_i = 1`（每谱面一个单纯形权重） |

* `Z` 才是参数（原始 logits 存进 npz 的 `D_raw`）；`wd` 加在 `Z` 上 → 对 `simplex` 是**把 D 拉向均匀
  分布** `1/dim`（不是拉向 0），对 `nonneg` 是拉向一个小的正常数。
* **语义代价（必须知道）**：`Σ_i D_mi (P_ui − C_mi) = Σ_i D_mi P_ui − Σ_i D_mi C_mi`，后一项只依赖谱面
  → 永远被 `bm[m]` 吸收（无约束时也如此，见 §4.1）。因此
  1. `simplex` 把 D 变成"该谱面在 dim 个玩家因子上的**凸组合权重**"；原来 D 可取任意线性组合，
     现在只能取凸组合 —— 假设空间**严格变小**（`P` 仍可自由缩放，所以是形状限制而非幅度限制）；
  2. `C` 在两种口径下都退化成"谱面常数"、与 `bm` 共线；step3 真正能用的谱面向量是 **D**（单纯形）
     与 `bm`；
  3. 好处是规范性 + 可解释性：非负、和为 1、无符号翻转，可以直接看"这张图吃哪几维"。
* **实测注意（2026-10-04，1k）**：给 D 加约束时**不要沿用给自由参数调好的 `wd=1e-4`**。Adam 的
  weight decay 是"每步固定幅度的拉回"，加在 softmax logits 上就是把 `Z` 拉回 0 —— 也就是**均匀 D**，
  而均匀点恰好是 softmax 雅可比最小（`~1/dim`）的对称点，模型会停在"几乎没有交互项"的解上
  （实测归一化熵 = 1.000、RMSE 掉回 bias 基线 1.5239）。两个修法：`--d-logit-wd 0`（**推荐**，
  logits 不进 wd、P/C 仍按 `--wd`）或把全局 `wd` 降到 `1e-5`。另外 `simplex` 的 logits 默认按
  `std=0.5` 初始化（`--d-init-std` 可改），避免所有谱面从同一个均匀 D 出发。修好后 D 会高度分化
  （1k：`mean max_i D_i = 0.83`、91.6% 的谱面有一维 > 0.5），代价只有 +1.0%（§7）。
* `nonneg`（只加 `D ≥ 0`、不加和为 1）**不推荐**：`softplus` 的尺度自由，`wd` 小则发散、大则塌缩，
  1k 上扫 `0 / 1e-5 / 3e-5 / 1e-4` 没有一档接近自由 D（最好 1.5231）。
* **2026-10-04 决定：不采用**（用户拍板，改走 §4.5 的 `mirt_exp`）。`--d-constraint` 开关保留、
  默认 `none`，上面的实测留档备查。

### 4.5 第三种交互：`mirt_exp`（2026-10-04 新增）

用户拍板的算式 —— 把每维的"漏失"先在 **(1−ACC) 空间**加权求和，再取 log 回 log loss：

```
S_i(u,m) = P_ui − C_mi
tmp(u,m) = Σ_i D_mi · e^(−S_i(u,m)) = Σ_i D_mi · e^(C_mi − P_ui)
ℓ̂(u,m)  = b0 + [b_u + b_m (+ w1·x1)] + ln tmp(u,m)
```

**它等价于什么.** 令 `E_mi := D_mi·e^(C_mi) > 0`，则 `ℓ̂ = b0 + ln ⟨E_m, e^(−P_u)⟩`：一个**正内积再取
log**。写成 log-sum-exp 就是 `ln Σ_i e^(ln D_mi − S_i)`；各项差距大时 ≈ `max_i(ln D_mi − S_i)`，
是**"瓶颈 / 最弱一环"型**的归纳偏置，而 `mirt` 的 `⟨P−C, D⟩` 是**加权和型**（允许正负抵消）。
两者不是同一假设空间的改写：`mirt_exp` 是**更紧、但也更高方差**的一族（§7 实测）。

**可辨识性（step3 必须知道）**

1. `C` 与 `D` **只通过乘积 `E = D ⊙ e^C` 出现**（`C_mi → C_mi+a`、`D_mi → D_mi·e^(−a)` 不改变模型）
   → step3 能用的谱面侧向量只有 **E**。实测 1k 上 `C` 几乎不动（`‖C‖₂ ≈ 0.055` = 初始化尺度，
   而 `‖D‖₂ ≈ 0.30`），即 `E ≈ D`。
2. `D` 的**整体尺度被 `b0` 吸收**（`D → λD` ⟺ `b0 → b0 − ln λ`；实测 `b0 ≈ −5.09`，`mirt` 是 −5.41）
   → **不需要 `ΣD=1` 的归一化**，这正是取消单纯形约束的数学原因。
3. `b_u` 与 `P` 的"每玩家整体平移"、`b_m` 与 `E` 的"每谱面整体缩放"各是一组重参数化 ——
   留不留都不改变表达力，只改变优化路径与正则化口径（本步保留，与旧模型一致）。

**必须遵守的三条实现约定**

* `ln tmp` 要求 `tmp > 0` → `D` 必须为正：`mirt_exp` **强制** `D = softplus(Z)`（`--d-constraint none`
  自动提升为 `nonneg`，`simplex` 也兼容）。自由 `D` 会让 `tmp` 变负、`ln` 直接 NaN。
* D 的 logits **默认不进 weight decay**（`--d-logit-wd 0`）：wd 把 `Z` 拉向 0 `⇒ D → 0 ⇒ ln tmp → −∞`，
  而尺度本来就简并给 `b0`。实测 `--d-logit-wd 1e-4` 把 F1 上的 RMSE 从 1.4614 拉到 1.5020。
* 实现用 `logsumexp(log D − S)`，与 `ln Σ D e^(−S)` 代数等价，避免 `exp` 溢出。

---

## 5. 训练协议

### 5.1 切分：30% 有序对（唯一设定，§2.3）

```
ukey      = np.unique(pair_key)                       # 有序对全集（--pair-universe，默认 step0）
is_test   = default_rng(20260928).random(len(ukey)) < 0.30
train     = step1 rows with pair ∉ test_pairs
test      = step0 rows with pair ∈  test_pairs
test_clean= step1 rows with pair ∈  test_pairs        # 次要口径，见 §6.2
```

- 一个 pair 只在一边；同一 pair 的所有 play（step0 与 step1 的）都归同一边。
- 玩家与谱面**都不留出**（§2.3）。
- 单次固定划分；测试集固定 → RMSE 跨配置直接可比。
- **不再有双 eval 集**（初版的 E_full / E_clean 取消）：只有这一个测试集。
  训练侧处理（若将来引入收敛过滤等）**只作用于训练集**（`cleaning_plan.md` §3.2 的原则不变）。

### 5.2 词表与覆盖率

- 词表**只从训练行**建：训练里出现过的 entity 与 item 才有向量。
- **覆盖率** = 测试 play 中"该玩家与该谱面都在训练词表里"的占比。
  1k 实测 **99.63%**（未覆盖的 0.37% 来自只在测试对里出现的 id）。**覆盖率 < 60% 一律标注不可用**。
- 可选门槛（`--min-plays-per-chart` / `--min-charts-per-player`，默认 1/1 = 关）：
  只影响"哪些 id 有向量"，不影响总体定义。

### 5.3 优化

| 项 | 默认 | 说明 |
|---|---|---|
| epoch | **固定 60，不早停** | 不用测试集做任何选择 |
| batch | 8192 | 沿用 |
| 随机性 | 单次划分；关键对照跑多 seed 报 mean±sd | 测试集固定，seed 只影响初始化/打乱 |

**没有训练权重**（§2.2：cell 表取消 → `w_cell` 不存在）。

---

## 6. 评价

### 6.1 指标（全部在 play 上）

| 指标 | 定义 | 说明 |
|---|---|---|
| **RMSE（loss）** | 测试集上的预测误差 | **主指标** |
| MAE（loss） | 同上，绝对误差 | 抗离群，辅助 |
| pp 误差 | `\|exp(loss_pred) − exp(loss_true)\| · 100` | 人话版本；报 p50/p90、>1pp / >5pp 占比 |
| Spearman | 预测值与标签的秩相关 | 只看排序质量时用 |
| **覆盖率** | 测试 play 中能打分的占比 | 必须报；< 60% 标注不可用 |
| paired CI | 同一测试 play 上 bootstrap 重采样（配置间配对差值） | 判定"是否真的更好" |

**明确不用：**

- **不用 `SE` / `excess`** —— 它们需要"同一 (player, beatmap) 的多次成绩"估 split-half 噪声，
  本版没有 cell 表、也不做成绩聚合（§2.2）。
- **不用 R²** —— 同一测试集下 RMSE 已经够用，且 R² 对"预测均值"的基线过于宽容。

### 6.2 两个测试口径

| 口径 | 数据 | 回答 |
|---|---|---|
| **`step0`（主）** | 测试对的 **step0** 行（未清洗） | 模型在**目标总体**（含乱打与入门期）上有多准 |
| `step1`（次） | **同一批 pair** 的 **step1** 行（清洗后） | 换到干净标签上有多准 —— 与主口径的差就是清洗带来的标签变化 |

两个口径**用同一批 pair**，所以可以并排看；但它们不是同一个总体，**不能互相替代**。

### 6.3 分层报告（建议，待做）

在测试集上再按下面几刀拆开（临时 `groupby`，不建表）：该 pair 在训练侧的 play 数（1 / 2 / 3–5 / 6+）、
该玩家的 play 数、`playcount_cur` 原始值分箱。§7 暂未做。

### 6.4 对照与验收（建议）

| 编号 | 配置 | 说明 |
|---|---|---|
| A0 | `bias` | 下限基线 |
| A1 | `mf_dot` | 低秩对照 |
| A2 | **`mirt`** | 主候选 |
| B1 | `mirt` + F1 | 顺序特征是否值钱 |
| B2 | `mirt` + F2 | 时间特征是否值钱（**非默认，仅消融**） |
| C1 | `mirt` + `--pair-universe step1` | 换有序对池子（§12.1） |

**验收标准（建议）：**

1. 主指标（step0 测试集上的 RMSE）：`mirt`（或 `mf_dot`）显著低于 `bias`；
2. 覆盖率 ≥ 60%；
3. `B1` / `B2` 相对 `A2` 的 RMSE 有可复现的下降（**且必须同时报覆盖率**，因为 F2 会掉覆盖）；
4. 结论与数字写回本文档 §7 与 README。

---

## 7. 实测

跑法：`python scripts/step2_predict.py --tag 1k --models bias,mf_dot,mirt --epochs 60`；
10k 额外加 `--batch 32768`。**torch 实现**：`--device auto`（本机 `torch 2.13.0+cpu`，6 线程 CPU）。

> 本节全部数字已于 2026-10-04 用 torch 引擎**原样复跑**：与旧 numpy 实现在同一划分、同一初值下
> **逐条一致**（全部指标最大差 `5e-6`，即 JSON 六位小数的舍入），因此下表数字不变，只更新了运行时间。

### 1k — 划分与规模（默认 `--pair-universe step0`）

| 量 | 值 |
|---|---|
| 有序对全集（step0） | **866,616** |
| 测试对 | **259,500（29.94%）** |
| 训练 play（step1） | **674,436** |
| 测试 play（step0） | **420,401** |
| 测试对的 step1 行（次要口径） | 289,981 |
| 训练词表 | 987 entity / 19,636 item |
| 测试行覆盖率 | **99.63%**（418,856 / 420,401） |
| 运行时间 | 3 个模型合计 **139 s**（torch / 6 线程 CPU；旧 numpy 版 237 s） |

### 1k — 模型对比

| 模型 | step0 RMSE | step0 MAE | Spearman | pp p50 | pp p90 | \| step1 RMSE | step1 Spearman |
|---|---|---|---|---|---|---|---|
| `bias` | 1.5236 | 1.1765 | 0.7330 | 0.2717 | 3.271 | 1.1988 | 0.8435 |
| `mf_dot` | **1.4548** | 1.0597 | 0.7632 | 0.2426 | 2.904 | 1.0126 | 0.8856 |
| **`mirt`** | 1.4582 | **1.0585** | **0.7633** | 0.2429 | 2.865 | **1.0087** | **0.8861** |

（n = 418,856 / 288,833；覆盖率均 99.63% / 99.60%）

- 交互项相对纯偏置的增益很明显：step0 **1.5236 → 1.4582**（−4.3%），Spearman 0.733 → 0.763。
- `mf_dot` 与 `mirt` 几乎打平（step0 差 0.2%，step1 差 0.4%）；`mirt` 在 pp 分位数与 step1 上略好。
- **pp p50 ≈ 0.24 pp**：一半的测试游玩，预测 ACC 与真实 ACC 的差在 0.24 个百分点以内。
- 主口径（step0）RMSE 比次要口径（step1）高约 45%（1.458 vs 1.009）：**未清洗的标签本身更难预测**
  —— 它含低 ACC 的乱打与入门期游玩，这些正是 step1 删掉的部分。

### 1k — 消融

| 配置 | 测试对 | 覆盖率 | step0 RMSE | Spearman | 备注 |
|---|---|---|---|---|---|
| `mirt` F0（默认） | 259,500 | 0.9963 | 1.4582 | 0.7633 | 主口径 |
| `mirt` **F1**（+`ln(playcount_cur)`） | 259,500 | 0.9963 | **1.4290** | 0.7724 | −2.0%；顺序特征有一点用 |
| `mirt` **F2**（player-year + 年内漂移） | 259,500 | **0.8376** | **1.0886** | 0.8764 | ⚠️ 覆盖率掉 16 pp，**与 F0 不可直接比** |
| `mirt` F0，`--pair-universe step1` | 173,194 | 0.9962 | 1.1106 | 0.8624 | ⚠️ 换池子后测试集变"易"，**不可与上表比** |

三点必须连在一起读：

1. **F1 的增益小而真实**（1.4582 → 1.4290，−2.0%），但 `playcount_cur` 有空洞（§2.2），
   所以它衡量的是"记录序号"而非"第几把"。
2. **F2 的 RMSE 大幅下降（−25%）几乎全是"覆盖率换来的"**：它把 16% 的测试行判为不可打分
   （那些 `(player, year)` 实体没在训练里出现），剩下的是更容易的那部分。
   要判断 F2 是否真有用，必须在**同一子集**上比（§12.3）。
3. **`--pair-universe step1` 的测试集明显更容易**（1.1106 vs 1.4582）：池子换成 step1 的 577,999 对后，
   被抽中的都是"熬过清洗"的密集对，它们的 step0 行也偏向高质量。这正说明
   **"用 step1 选测试集"会把清洗的代价藏起来** —— 所以默认用 step0（§12.1）。

### 1k — 两个新机制（2026-10-04，§4.4；60 epoch，其余同主表）

**(a) Pc 漂移 `--p-drift logpc`**

| 配置 | step0 RMSE | Δ vs 基线 | Spearman | pp p50 | step1 RMSE | step1 Spearman |
|---|---|---|---|---|---|---|
| `mirt` F0（基线） | 1.4582 | — | 0.7633 | 0.243 | 1.0087 | 0.8861 |
| `mirt` F1（全局 `w1·ln pc`） | 1.4290 | −2.0% | 0.7724 | 0.234 | 1.0016 | 0.8884 |
| `mirt` + Pc 漂移 | 1.4363 | −1.5% | 0.7706 | 0.236 | 1.0001 | 0.8887 |
| **`mirt` F1 + Pc 漂移** | **1.4225** | **−2.4%** | **0.7747** | 0.234 | 1.0014 | 0.8884 |

漂移值钱，但**大部分增益来自"存在一条 `ln(playcount_cur)` 斜率"这件事本身**（F1 的全局标量），
每玩家/每维的异质性只再多 0.45%（1.4290 → 1.4225）；两者叠加是目前 1k 最好的配置。
尺度参考：1k 上 `‖Pc‖₂` 均值 0.44、`‖P‖₂` 均值 1.45；测试集上漂移项 sd ≈ 0.22、交互项 sd ≈ 0.54。

**(b) D 约束 `--d-constraint`**

| 配置（60 epoch） | step0 RMSE | Δ vs 自由 D | Spearman | step1 RMSE |
|---|---|---|---|---|
| 自由 D（基线，`wd=1e-4`） | 1.4582 | — | 0.7633 | 1.0087 |
| `simplex`，`wd=1e-4` 直接加在 logits 上 | 1.5239 | **+4.5%** | 0.7331 | 1.1946 |
| `simplex`，全局 `wd=1e-5` | 1.4741 | +1.1% | — | — |
| **`simplex`，`--d-logit-wd 0`（推荐）** | **1.4723** | **+1.0%** | 0.7537 | 1.0410 |
| `nonneg`（只 `D>0`），最好一档 `wd=3e-5` | 1.5231 | +4.4% | — | — |
| 自由 D + Pc 漂移（对照） | 1.4363 | — | 0.7706 | 1.0001 |
| **`simplex` + Pc 漂移（推荐）** | **1.4385** | **+0.15%** | 0.7651 | 1.0177 |

读法：

1. **约束本身很便宜，但必须换正则化口径**：`simplex` 单独用 +1.0%，叠上 Pc 漂移后只 +0.15%
   （1.4385 vs 1.4363）—— 基本白送。**朴素地把 `wd=1e-4` 也施加到 softmax logits 上才是 +4.5% 的
   来源**（把 D 拉回均匀的对称点），不是 `ΣD=1` 本身（§4.4b）。
2. **约束真的起作用**：最终 `simplex` 模型的 D 高度分化 —— `mean max_i D_i = 0.83`、归一化熵 0.17、
   **91.6% 的谱面有一维权重 > 0.5**；自由 D 则是 49.9% 的分量为负、归一化熵 0.81。
   也就是说 `simplex` 把 D 变成了"这张图主要考哪一维"的近乎 one-hot 权重。
3. **`nonneg` 不推荐**：尺度自由 → `wd` 小则爆、大则塌，四档都不行；要加约束就用 `simplex`。
4. 约束并没有把模型"变弱"：交互项 sd 自由 D 0.535 / `simplex` 0.651，只是换了参数化。
5. 产物：`simplex` 的 npz 里 `D` 是**约束生效后**的矩阵（每行非负、和=1），`D_raw` 是 logits；
   `--p-drift logpc` 额外多一个 `Pc`（§8.1）。校验项 10 每次都会验约束（实测 `max|ΣD−1| ≈ 2e-7`）。

**(c) 第三种交互 `mirt_exp`（§4.5）+ Pc 漂移四格消融**

| 模型 | F0 | F0 + Pc 漂移 | Δ | F1 | F1 + Pc 漂移 | Δ |
|---|---|---|---|---|---|---|
| `mirt`（`⟨P−C, D⟩`） | **1.4582** | **1.4363** | −1.50% | **1.4290** | **1.4225** | −0.45% |
| `mirt_exp`（`ln Σ D·e^(−S)`） | 1.5012 | 1.4739 | −1.82% | 1.4695 | 1.4614 | −0.55% |

（1k，60 epoch，同一划分；格内是 step0 RMSE。`mirt_exp` 200 epoch 复跑：F0 1.5028、F1+Pc 1.4674，
`mirt` 同轮 1.4575 / 1.4202 —— **两边都已收敛，差距不是训练不够**。）

读法：

1. **Pc 漂移对两种交互都值钱，幅度相近**（−0.45% ~ −1.8%）：F0 上收益最大（完全没有
   `ln(playcount_cur)` 信息），已经带 F1 标量时只再多 0.45–0.55%。
2. **`mirt_exp` 四个格子都差 ~2.7%**。原因**不是深尾**：按标签切开，尾部（ACC<0.90，占 1.62%）
   3.30 vs 3.21、主体 1.455 vs 1.413，两边都差 ~3%；而且它**训练 RMSE 更低**
   （F0 0.9067 vs 0.9220；F1+Pc 0.9533 vs 0.9710）→ 同 dim 下**有效容量更大、更方差化**。
3. 在**清洗后的 step1 口径**上两者几乎打平（F0 1.0092 vs 1.0087；F1+Pc 1.0020 vs 1.0014）：
   差异几乎全部来自未清洗的低质量行 —— 那正是主口径要测的东西。
4. 结论：**主口径仍用 `mirt`**；`mirt_exp` 作为"瓶颈型/ACC 空间相加"的对照保留在代码里
   （`--models mirt_exp`），它的 `E = D ⊙ e^C` 也可以给 step3 用（但只有 E 可辨识）。

**(d) 主效应消融 `--drop-bias {u,m,both}`**（1k，60 epoch，F0 除注明外）

| 配置 | step0 RMSE | vs 完整 |
|---|---|---|
| `mirt`（b0 + b_u + b_m） | 1.4582 | — |
| `mirt --drop-bias m`（只去 b_m） | **1.4577** | −0.03% |
| `mirt --drop-bias u`（只去 b_u） | 1.5065 | +3.3% |
| `mirt --drop-bias both` | 1.5070 | +3.3% |
| `mirt_exp`（b0 + b_u + b_m） | 1.5012 | — |
| `mirt_exp --drop-bias both` | 1.5166 | +1.0% |
| `mirt_exp` F1+Pc（有主效应） | 1.4614 | — |
| `mirt_exp` F1+Pc `--drop-bias both` | **1.4560** | −0.4% |
| `bias --drop-bias both`（纯常数 b0） | 2.1699 | — |

1. **`b_m` 在 `mirt` 里完全可去**（1.4577 vs 1.4582）：`⟨P−C, D⟩` 展开后的 `−Σ_i C_mi D_mi` 本来就是
   每谱面常数，与 `b_m` 共线 —— 两者只是同一自由度的不同参数化。
2. **`b_u` 去不得**（+3.3%）：交互项对同一玩家在不同图上会变，表达不了"该玩家整体平移多少"。
3. **`mirt_exp` 里两个主效应几乎都能去**（F0 +1.0%，F1+Pc 反而 −0.4%，是它目前最好的成绩）：
   与 §4.5 的重参数化一致（`b_u` ≡ P 每玩家平移、`b_m` ≡ E 每谱面缩放），去掉只改变优化路径与 wd 口径。
4. 主效应与交互项**可部分互相替代**：只有交互的 `mirt --drop-bias both`（1.5070）仍好于只有主效应的
   `bias`（1.5236）；两者齐全最好（1.4582）。纯常数（2.1699）说明主效应这一层单独贡献最大。

**(e) 维度消融 `--dim {2,3,4,8,16,32}`**（1k，60 epoch，step0 RMSE）

| dim | `mirt` F0 | `mf_dot` F0 | `mirt_exp` F0 | `mirt` F1+Pc |
|---|---|---|---|---|
| 2 | 1.4782 | 1.4777 | 1.5328 | 1.4408 |
| 3 | 1.4689 | 1.4674 | 1.5198 | 1.4325 |
| 4 | 1.4652 | 1.4628 | 1.5001 | 1.4297 |
| 8 | **1.4582** | 1.4556 | **1.4981** | **1.4216** |
| 16（默认） | 1.4582 | 1.4548 | 1.5012 | 1.4225 |
| 32 | 1.4585 | **1.4544** | 1.5067 | — |

`mirt` F0 的训练/测试随维度：

| dim | 2 | 3 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 训练 RMSE | 0.9907 | 0.9662 | 0.9545 | 0.9324 | 0.9218 |
| 测试 RMSE | 1.4792 | 1.4696 | 1.4661 | 1.4591 | 1.4588 |
| 测试−训练 | 0.4885 | 0.5034 | 0.5116 | 0.5267 | 0.5370 |
| 潜向量参数 | 80.5 k | 120.8 k | 161.0 k | 322.1 k | 644.1 k |

读法：

1. **dim 8 就够了**：`mirt` F0 在 8 与 16 上**逐位相同**（1.45824 vs 1.45823）；最好的配置
   `mirt` F1+Pc 在 dim 8 反而最好（**1.4216** vs 16 的 1.4225，−0.07%）；dim 32 不再有增益。
2. **训练误差一直在降（0.99 → 0.92），测试误差在 dim 8 就停了** —— 典型的"参数多了只记住训练集"，
   差距（测试−训练）随维度单调拉大。
3. **低维是真欠拟合**：dim 2 比 dim 16 差 +1.4%（F0）/ +1.3%（F1+Pc），dim 3–4 已接近饱和。
4. 顺带：dim ≤ 4 时 `mf_dot` 略好于 `mirt`（D 向量在这个数据量下挣不回自己的参数），dim ≥ 8 打平。
5. **结论：1k 建议 `--dim 8`**（同精度、参数量减半、给 step3 的 C/D 也更稳）；10k 没扫，数据量 ×9.5，
   16 可能仍有意义（待定）。
6. 非默认维度会写进 variant 与文件名（`..._dim8...`），不会覆盖 dim 16 的规范产物。

### 10k

划分（同 `--pair-universe step0`，`--batch 32768`，60 epoch，3 模型合计 **624 s**
（torch / 6 线程 CPU，峰值内存 3.0 GB；旧 numpy 版 1142 s）。
**10k 只跑了主表**，未做 F1 / F2 / 池子消融。

| 量 | 值 |
|---|---|
| 有序对全集（step0） | **7,784,519** |
| 测试对 | **2,334,078（29.98%）** |
| 训练 play（step1） | **6,384,278** |
| 测试 play（step0） | **4,007,074** |
| 测试对的 step1 行 | 2,735,046 |
| 训练词表 | 9,776 entity / 21,870 item |
| 测试行覆盖率 | **100.00%**（4,006,975 / 4,007,074） |

| 模型 | step0 RMSE | step0 MAE | Spearman | pp p50 | pp p90 | pp >1pp | \| step1 RMSE | step1 Spearman |
|---|---|---|---|---|---|---|---|---|
| `bias` | 1.3727 | 1.0717 | 0.6874 | 0.6151 | 4.106 | 40.2% | 1.1511 | 0.8129 |
| `mf_dot` | **1.3327** | 1.0311 | 0.7079 | 0.5901 | 4.038 | 39.4% | 1.0761 | 0.8381 |
| **`mirt`** | 1.3331 | **1.0209** | **0.7096** | 0.5839 | 3.942 | 38.9% | **1.0444** | **0.8456** |

- 交互项增益比 1k 小：step0 **1.3727 → 1.3331**（−2.9%），Spearman 0.687 → 0.710。
- **pp 误差明显比 1k 差**（p50 0.58 vs 0.24 pp）：10k 的玩家池是 top-10000（水平跨度比 top-1000 宽得多），
  预测**绝对 ACC** 更难；但 Spearman 0.71 说明**排序**仍可用。
- 与 1k 一样，`mf_dot` 与 `mirt` 打平（差 0.03%）。

### 与 1k 的对比要点

| | 1k | 10k |
|---|---|---|
| 训练 play | 674,436 | 6,384,278 |
| 测试 play | 420,401 | 4,007,074 |
| 覆盖率 | 99.63% | 100.00% |
| `bias` → `mirt` RMSE | 1.5236 → 1.4582（−4.3%） | 1.3727 → 1.3331（−2.9%） |
| pp p50 | 0.243 | 0.584 |
| 主/次口径 RMSE 比 | 1.4582 / 1.0087 = **1.45×** | 1.3331 / 1.0444 = **1.28×** |

- 数据量上去后**交互项的边际增益变小**（−4.3% → −2.9%），但绝对误差也更低（1.458 → 1.333）——
  和旧 README §7 观察到的"规模上去后 1k 的优势缩水"是同一现象。
- **pp 误差变差而 Spearman 变好**：10k 池子更宽，绝对值难猜但相对顺序更清楚。

---

## 8. 产物与接口契约（给 step3）

### 8.1 潜向量文件

`data/processed/step2_{tag}_{model}_{features}{_dimN}{_pd}{_dnonneg|_dsimplex}{_py}{_ustep1}_{target}.npz`
（`_dimN` 仅当 `--dim ≠ 16`；同上，`wd`/`epochs`/`seed` 等超参只记在 summary 的 `args` 里）
（`_pd` 仅当 `--p-drift logpc`；`_d*` 仅当 `--d-constraint` 非默认；`_py` 仅当 `--entity player-year`；
`_ustep1` 仅当 `--pair-universe step1`）：

```
entity_ids    int64  (n_entities,)      # player_id；F2 时是 player_id*100+(year-2000)
beatmap_ids   int32  (n_charts,)        # = chart 键（无 rate）
P             f32    (n_entities, dim)
Pc            f32    (n_entities, dim)  # 仅 --p-drift logpc：漂移速率
V             f32    (n_entities, dim)  # F2 关闭时全 0
C             f32    (n_charts, dim)
D             f32    (n_charts, dim)    # **约束生效后**的 D（simplex 时每行非负、和=1）
D_raw         f32    (n_charts, dim)    # 仅 --d-constraint 非默认：softmax/softplus 的 logits
b0            f32    (1,)
bu            f32    (n_entities,)
bm            f32    (n_charts,)
```

### 8.2 报告

`data/processed/step2_{tag}_summary_{target}.json`，按 `runs[variant]` **累积**多次运行，每条含
`args` / `mechanism`（p_drift、d_constraint、生效的 d_init_std 与 d_logit_wd）/ `engine`（torch 版本、
设备、线程、确定性开关）/ `split`（池子、对数、测试对数、seed）/ `sizes` / `results`（每个模型 × 两个口径）
/ `checks`（§10 的逐条结果）/ `seconds`。
本文档 §7 的数字全部可从该 JSON 复现。

### 8.3 契约（step3 依赖）

1. **必须输出 `C` 与 `D`**（不可辨识性，§4.1），以及 `bm` 与 `b0`；
2. **必须输出 `beatmap_ids`**，与 `beatmap_meta_{tag}.csv` 的 `beatmap_id` 可直接 join；
3. **必须记录训练词表**（哪些 beatmap/player 有向量），供 step3 判断覆盖率；
4. **必须记录统一划分**（测试对集合 / seed），使 step3 能用**同一批谱面**做对照；
5. `dim`、`wd`、`seed`、`model`、`features`、`entity` 必须写进 summary。

---

## 9. 已知局限与风险

1. **单次游玩的抖动全在标签里。** 没有 cell 中位降噪（σ ≈ 0.65），RMSE 的下限被抬高；
   play 级的 RMSE **不能**与旧 cell 级数字直接比。
2. **测试集是未清洗的 step0**，所以主口径 RMSE 天然比清洗标签高（§7 实测 1.458 vs 1.009）。
   这是刻意的（§2.3），但比较模型时必须所有人都在同一口径上。
3. **`playcount_cur` 有空洞且是下界**（step1 §2.4、§5.2）：只数已记录的成功游玩，覆盖率 ≈28.6%。
   F1 用它原值，所以 `ln(playcount_cur)` 是"记录序号"不是"第几把"。
4. **step1 砍掉了每玩家最早的 30%** → 学习曲线**前段缺失**；训练侧的 F1 起点不是第 1 把。
5. **71.51%（1k）/ 69.59%（10k）的 `(player, beatmap)` 对只有一条记录**：F1 在这些对上无信息；
   而且它们的标签是单次游玩，噪声最大。
6. **`bias` / `mf_dot` / `mirt` 都用同一批超参**（dim16 / wd1e-4 / 60 epoch），没有逐模型调参；
   差异里可能混着调参不足。
7. **覆盖率会随配置变化**（F2 掉到 83.8%），**不同覆盖率的 RMSE 不可直接比**（§7 已标注）。
8. **只有 1.0×**：无 DT/HT，与旧 README 的 rate 相关结论（§6）无关，其数字不可复用。
9. **`loss` 的 max = −2.3026 是硬截断**（step1 删掉了 ACC ≤ 0.90）：训练侧看不到深尾，
   而测试侧（step0）**有**深尾 → 这是主口径误差的一大来源。

---

## 10. 校验清单（脚本每次运行都跑）

下面 9 条在 `scripts/step2_predict.py` 的 `Checklist` 里逐条实现，**每次运行全部执行**：
任一条 FAIL 直接抛错、**不写任何 npz / summary**。第 8 条用一个合成的小训练路径自检
（同 seed 跑两遍，参数`torch.equal` 逐位相同）；每条的结果同时写进 summary 的 `checks`。

| # | 断言 | 期望 |
|---|---|---|
| 1 | `pair_never_crosses_split`：任一有序对的 play 只出现在一侧 | 恒成立 |
| 2 | `train_and_test_disjoint`：训练行的 pair 与测试行的 pair 交集为空 | 恒成立 |
| 3 | `test_pair_frac`：测试对数 / 池子对数 ≈ 0.30 | ±1% |
| 4 | `loss_in_range`：测试（step0）`loss ∈ [−12.4663, −0.2669]` | 恒成立 |
| 5 | `coverage_reported`：每个配置都报覆盖率与 n_scored | 恒成立 |
| 6 | `vocab_from_train_only`：词表 id 集合 == 训练行里出现的 id 集合 | 恒成立 |
| 7 | `no_test_leak`：测试行从不进梯度；没有用测试集做任何选择 | 恒成立 |
| 8 | `determinism`：同 seed 重跑得到逐位相同的指标 | 恒成立 |
| 9 | `npz_matches_summary`：npz 的 id 数与 summary 的 `n_entities` / `n_items` 一致（含 `Pc`/`D_raw` 形状） | 恒成立 |
| 10 | `d_constraint_holds`：`--d-constraint` 非默认时，导出的 `D` 真的满足约束 | 仅约束开启时跑；`simplex` 实测 `max|Σ_i D_i − 1| ≈ 2e-7` |

---

## 11. 配置与复现

| 参数 | 默认 | 含义 |
|---|---|---|
| `--tag` | `1k` | 决定读 `step0_{tag}` / `step1_{tag}`、写 `step2_{tag}_*` |
| `--target` | `loss` | 标签列 |
| `--models` | `bias,mf_dot,mirt` | 要训的模型（逗号分隔）；可选 `mirt_exp`（§4.5） |
| `--dim` | 16 | 潜向量维度；1k 建议 8（§7e） |
| `--epochs` / `--lr` / `--wd` / `--batch` | 60 / 0.02 / 1e-4 / 8192 | 优化 |
| `--features` | `F0` | `F0 / F1 / F2`（§2.4） |
| `--entity` | `player` | `player / player-year`（F2） |
| `--pair-universe` | `step0` | 有序对池子取自哪张表（§12.1） |
| `--test-pair-frac` | 0.30 | 测试对比例 |
| `--split-seed` | 20260928 | 划分 seed |
| `--seed` | 0 | 训练 seed（初始化 / 打乱） |
| `--eval-every` | 20 | 每多少个 epoch 打印一次测试 RMSE（仅监控） |
| `--p-drift` | `none` | `none / logpc`：`P_eff = P + Pc·ln(playcount_cur)`（§4.4a） |
| `--d-constraint` | `none` | `none / nonneg / simplex`：D 的 softplus / softmax 重参数化（§4.4b） |
| `--drop-bias` | `none` | `none / u / m / both`：去掉 `b_u`、`b_m` 主效应（§7d） |
| `--d-logit-wd` | 约束开启时 0，否则同 `--wd` | 原始 D logits 的 weight decay（§4.4b 的实测注意） |
| `--d-init-std` | 0.05（`simplex` 时 0.5） | 原始 D logits 的初始化标准差 |
| `--device` | `auto` | `auto / cpu / cuda`；`auto` = 有 CUDA 用 CUDA，否则 CPU |
| `--threads` | 0 | `torch.set_num_threads()`；0 = 不动默认值（跨机器复现需相同线程数） |

```bash
# 1k 主表
python scripts/step2_predict.py --tag 1k --models bias,mf_dot,mirt --epochs 60
# 顺序特征消融
python scripts/step2_predict.py --tag 1k --models mirt --features F1
# 时间特征消融（非默认）
python scripts/step2_predict.py --tag 1k --models mirt --features F2 --entity player-year
# 换有序对池子
python scripts/step2_predict.py --tag 1k --models mirt --pair-universe step1
# 10k
python scripts/step2_predict.py --tag 10k --models bias,mf_dot,mirt --epochs 60 --batch 32768
```

---

## 12. 未决问题（需要拍板）

1. **有序对池子取 step0 还是 step1？**（`--pair-universe`，默认 **step0**）
   - **step0**：测试对是"全部有序对"的随机 30%，**不被清洗筛过** → 符合"测试集不由处理手段筛"的原则。
     代价：测试对里有一批在 step1 里没有记录的对（1k 的池子 866,616 对，step1 只有 577,999）。
   - **step1**：训练与测试是"同一批对"的互补划分，没有浪费；但**测试对本身是被清洗选出来的**
     （§7 实测：RMSE 1.11 vs 1.46，明显更容易）。
   - 本文档默认 step0，理由与 §7 的对照都写在上面。
2. **F2 到底有没有用？** 现在它 RMSE 更低但覆盖率掉 16 pp，**必须换成同一子集再比**
   （例如只在"F2 也能打分的行"上重算 F0 的 RMSE）。建议下一步就做这个对照。
3. **要不要加 paired bootstrap CI？** §6.1 列了，但脚本还没实现；主表里 `mf_dot` 与 `mirt` 只差 0.2%，
   没有 CI 不好下结论。
4. **`dim` 8 vs 16、`wd` 1e-4 vs 3e-4** 没扫；1k 只有 674k 训练行 / 19,636 谱面，16 维可能偏多。
5. **要不要跑多 seed？** 现在所有数字都是单次划分 + 单 seed。
6. **`mlp` 非线性对照要不要补？** 它回答"低秩假设够不够"，但需要额外的实现。
7. **测试集分层报告**（§6.3）没做：按 pair 的 play 数、玩家 play 数、`playcount_cur` 分箱。
8. **step0 的深尾怎么处理？** 主口径测试集含 ACC 很低的乱打（loss 可到 −12），
   这些行拉高了 RMSE；要不要单独报一个"ACC ≥ 0.90 子集"的 RMSE 作为附加指标（不改主口径）。
