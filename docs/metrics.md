# 指标：难度排序的成对 NLL

本文件定义 **rank-NLL**（成对比较的对数损失）及其归一化指标。它是"系统给的难度序能不能预测玩家实际会赢哪一张图"的损失，也是 step2 DL 的可选训练目标。实现见 [`dcr/rank_metrics.py`](../dcr/rank_metrics.py)，评测入口见 [`scripts/step2_rank_eval.py`](../scripts/step2_rank_eval.py)，结果见 [docs/rank_eval.md](rank_eval.md)。

> 与现有指标的关系：RMSE / Spearman 评价的是 **loss 的数值预测**；rank-NLL 评价的是 **难度序**。两者回答不同问题，不能互相替代。现有 global Spearman 主要由玩家能力项贡献（loss 方差中只有 22.7% 是 between-player），rank-NLL 则把玩家能力消掉，只留下"同一玩家对两张图的相对表现"。

---

## 1. 记号

| 符号 | 含义 |
|---|---|
| $p$ | 玩家段（`player_id`，step1 的 365 天空窗拆分后的实体） |
| $m$ | 谱面（`beatmap_id`） |
| $S_p$ | 玩家 $p$ 玩过的谱面集合，$l_p = \lvert S_p \rvert$ |
| $s_{pm}$ | 游玩表现，越高越好：$s_{pm} = -\mathrm{loss}_{pm} = -\log(1-\mathrm{ACC})$ |
| $d_m$ | 系统难度，越高越难 |
| $y_{pij}$ | $1[s_{pi} > s_{pj}]$，即玩家 $p$ 在 $i$ 上表现更好 |
| $n_{ij}$ | 在 $i,j$ 都被玩过的玩家中，$i$ 胜过 $j$ 的次数 |
| $n_{ji}$ | 反向次数；$n_{ij}+n_{ji}$ 是这一对的总比较数 |
| $N$ | $\sum_{i<j}(n_{ij}+n_{ji})$，总比较数 |

$n_{ij}$ 是 Bradley–Terry 的**充分统计量**。把 $n_{ij}$ 先变成经验概率 $\hat p_{ij}=n_{ij}/(n_{ij}+n_{ji})$ 再算 NLL 是两步估计：$\hat p$ 本身带测量误差，且 1k 里 57.9% 的谱面对只出现一次、76.8% 是全票 0/1，这个误差不能忽略。正确写法是直接把计数放进似然（§3）。

### 1.1 pairbest 的数据结构（为什么可以简化）

本指标的实验输入固定为 step1/pairbest：每个 `(player_id, beatmap_id)` 恰好一行。由此：

* **固定信号不需要 train/test**：`msd_overall`、star、analyser 都是 content-only，和 `split` 无关；
  runner 默认 `--split all`，直接在所有行上算。只有用模型学出来的 `bm` 当难度时，才需要 `--split test`。
* **没有重复观测**：不存在"同一对多次 play"的聚合；一行就是一个观测。重复 pair 会被 runner
  显式拒绝（`step2_rank_eval.py` 的 duplicate 检查），不会静默重复计数。
* **`n_plays` 不是权重**：它是清洗时的 best-of-k 选择变量，不是重复次数；指标不使用它，
  但它仍是选择偏差（§10 第 1 条）。
* **pair 是派生量**：每个玩家的 $C(l_p,2)$ 个 pair 不是独立 Bernoulli；点估计用全部 pair，
  区间必须按 `player_id_base` 做 cluster bootstrap，不能把 pair 当独立样本。
* **ties**：一行一个 ACC；精确 tie 直接 0.5/0.5，不需要 Davidson tie 模型。
* **player epoch**：`player_id` 是段，`player_id_base` 是 cluster；within-player 比较用段，
  ceiling/bootstrap 用 base。
* **ceiling 仍然要 cross-fit**：信号固定不代表 ceiling 固定；ceiling 是从同一批 pair 估的，
  样本内会被高估（1k 全量 0.4878 vs cross-fitted 0.5629）。固定信号可以用 `--folds 1` 快速看
  in-sample ceiling，但它比 cross-fitted 更低（0.4878 vs 0.5629），`captured` 反而更低（更保守），**不要和 cross-fitted 的数字混比**。

---

## 2. 观测模型

对每个玩家、每个无序对 $\{i,j\}\subseteq S_p$，假设

~~~text
q_ij = P(i 胜 j) = sigmoid( g(d_j) - g(d_i) )
~~~

其中 $g$ 是**单调 link**。$d_j > d_i$（$j$ 更难）时 $g(d_j)-g(d_i)>0$，所以 $q_{ij}>0.5$：$i$ 更可能赢。三种实现：

| link | $g(d)$ | 性质 | 用途 |
|---|---|---|---|
| `linear` | $\beta\,z(d)$，$\beta\ge 0$ | 只对仿射变换不变；$\beta$ 是 signal strength | 系统声称有**可比尺度**时（如 MSD） |
| `rank` | $\beta\,\mathrm{rank}(d)/M$ | 单调，scale-free | 系统只声称一个**序**时 |
| `spline` | $\sum_k \beta_k \phi_k(z(d))$，8 个 hat 函数 | 最灵活，但不保证单调 | 诊断"非线性尺度吃掉了多少 NLL" |

$z(d)$ 是去均值/除标准差后的 $d$。`linear` 和 `rank` 的 $\beta$ 用一维凸问题的导数二分求 MLE（不是网格，也不是有界 Brent；后者在接近可分时会停在上界）。$\beta=0$ 表示"这个方向没有信号"，此时 NLL 等于 $\log 2$。

> **符号约定**：$d$ 必须满足"越高越难"。如果信号是"越高越简单"，先取负；`beta>=0` 的约束会把方向反过来时自动退化为 $\beta=0$（NLL=$\log2$），而不是偷偷拟合一个负斜率。

---

## 3. 损失

~~~text
L(d) = - sum_{i<j} [ n_ij * log q_ij + n_ji * log(1 - q_ij) ]
N    = sum_{i<j} (n_ij + n_ji)
nll  = L(d) / N            # nats per comparison
~~~

$\log 2 = 0.693147$ 是"所有对都 50/50"的基线。$L(d)/N < \log 2$ 才说明有信号。

**ties**：默认把 $s_{pi}=s_{pj}$ 拆成 0.5/0.5（等价于对 50/50 目标算 log-loss）；`--tie drop` 可改为直接丢弃。ACC 由离散 hit 数算出，精确 tie 会有，必须显式处理，否则全部算给"$j$ 胜"会引入偏差。

**为什么不用经验概率**：

~~~text
错误：  p_hat_ij = n_ij / (n_ij + n_ji)
        L = - sum [ p_hat_ij log q_ij + p_hat_ji log q_ji ]
正确：  L = - sum [ n_ij log q_ij + n_ji log(1 - q_ij) ]
~~~

两者的差别是 plug-in 偏差：前者把 $\hat p$ 的估计噪声当成真值，噪声大小随覆盖度变化；后者是 exact likelihood，$\hat p$ 只是它的充分统计量。1k 的谱面对计数中位数是 1（$q_{50}=1$），这个偏差不是小数。

---

## 4. 归一化：ceiling 与 captured

原始 NLL 在信号弱时被压缩在 $\log 2$ 附近（MSD 的 1k 值只比 $\log2$ 低 0.053 nats），所以**不能只报 NLL**。需要两个参照：

### 4.1 无约束 BT（ceiling）

让每张谱面有一个自由的 $b_m$：

~~~text
q_ij = sigmoid( b_j - b_i )
b    = argmin_b  - sum [ n_ij log q_ij + n_ji log(1 - q_ij) ]
~~~

这是**任何全局难度序**能达到的最优成对似然（在 BT 模型内），也就是 ceiling。用 L-BFGS + 解析梯度拟合（[Hunter 2004](https://doi.org/10.1214/aos/1079120141) 的 MM 也可）。

### 4.2 cross-fitting

**必须 cross-fit**：在样本内，$b$ 有 6,828 个自由参数，会吸收谱面级的噪声/选择效应，ceiling 被高估。做法：

1. 按 `player_id_base` 聚成 cluster（同一玩家的多个 epoch 不能分到两边）；
2. 把 cluster 随机分到 $K$ 折（默认 $K=2$）；
3. 对每折，用其余折的玩家拟合 $b$，在该折玩家上算 NLL；
4. 只保留"两张图都在训练折出现过"的 held-out pair（否则 $b$ 对未见过的图是任意的）；
5. ceiling $=\sum$ held-out NLL $/ \sum$ held-out 比较数。

**拟合行与评测行必须分开**：第 3 步的"其余折的玩家"应该用**全部行**（train+test）拟合 $b$，
NLL 只在当前 `--split` 的 held-out 行上算。若 `--split test` 时连 $b$ 也只用 test 行拟合，
test 只有 33k 行、4,565 张图、约 1.14M 次比较（全量 110k 行、6,828 张图、12.5M 次），
6,828 个自由参数会严重欠拟合：1k 上实测 test-only ceiling = **0.6399**（coverage 0.833），
比 MSD 的信号 NLL 0.6385 还高，
`captured` 会算出 **1.03**（>1，物理上不可能）。用全部行拟合、test 行评测，同一个信号是
ceiling **0.5628**、`captured` **0.419**（coverage 0.9396；`--strict-pairs` 同 pair 集后是 **0.440**）。
runner 的 `--ceiling-fit all`（默认）就是这个口径；
`--ceiling-fit split` 只用于复现旧数字。

1k pairbest 上：样本内 ceiling = 0.4878，cross-fitted = 0.5629（本仓库 runner 的 2 折；
换折分配约 0.57，高估 13–17%）；说明这个修正不是可选的。

固定信号可以用 `--folds 1` 跳过 cross-fit（全部玩家拟合并评估），但那是 **in-sample ceiling**：
1k 全量会从 0.5629 降到 0.4878，`captured` 相应变小（MSD：40.7% → 25.8%，更保守）。它适合快速比较同一批固定信号，
不适合和 cross-fitted 表混用。

### 4.3 两个归一化量

~~~text
bits_saved = 1 - (L(d)/N) / log 2
captured   = (log 2 - L(d)/N) / (log 2 - L*/N)
~~~

* `bits_saved`：相对于"全 50/50"去掉了多少熵，$\in(-\infty,1]$。
* `captured`：相对于 ceiling 去掉了多少**可实现的**熵，理想值 1，坏信号可以 <0。它才是主指标。

> `captured` 的分母是 cross-fitted ceiling，而 ceiling 本身是一个 noisy 的估计（1k 上 split-half 的 $b$ 相关只有 0.582，Spearman–Brown 校正到全数据约 0.736），所以 `captured` 是**下界**：ceiling 越 noisy，所有系统的 captured 都被压低。系统之间的排序仍然有效；绝对值要谨慎解释。

> **pair 集对齐**：captured 的分母是 ceiling 覆盖到的 pair，分子默认是该 split 的全部 pair；两者在
> 1k 上相差约 6%（ceiling_coverage 0.94）；MSD 在 test split 上 captured 0.419（全 pair）vs 0.440
> （严格同集），差 2 个点。正式对比建议加 `--strict-pairs`，让分子只在同一批
> 覆盖 pair 上算；输出里的 `nll_all` / `captured_all` 仍保留全 pair 口径，`n_strict` 应等于
> `ceiling_n`（不等就是 ceiling 与 signal 的 pair 集没对齐）。

---

## 5. 辅助指标：逐玩家 Spearman

~~~text
rho_p = Spearman( d|S_p, -s_p|S_p )        # 只取 l_p >= min_l（默认 10）
~~~

$d$ 越高越难、$-s$ 越高越差，所以正相关 = 系统序和玩家自己的序一致。聚合三种，**必须同时报**：

| 聚合 | 公式 | 估计的总体 |
|---|---|---|
| l-weighted | $\sum l_p\rho_p / \sum l_p$ | 随机 (玩家, 谱面) 对 |
| macro | $\frac1P\sum \rho_p$ | 随机玩家 |
| Fisher-z | $\tanh\!\big(\frac{\sum (l_p-3)\operatorname{atanh}\rho_p}{\sum (l_p-3)}\big)$ | 近似正态的平均相关 |

$l$ 加权在 null 下近似逆方差（$\operatorname{Var}(\rho_p)\approx 1/(l_p-1)$），但在异质性下正确权重是 $1/(\tau^2+1/(l_p-3))$，不是 $l$。所以 l 加权不是"更对"，只是"另一个总体"；macro 是每个玩家等权。Spearman 本身**不是 proper scoring rule**，只作为描述统计和稳健性检查，不单独作为主指标。

**归一化**：

~~~text
norm_rho = (rho_system - rho_random) / (rho_ceiling - rho_random)
~~~

其中 `rho_random` 是用 map-level 随机向量、同样 groups/min_l 算出的 l-weighted 均值（1k 上约 $-0.007\pm0.015$，不是精确的 0）；`rho_ceiling` 是 cross-fitted $b$ 的逐玩家 Spearman（1k 上约 0.61，min_l=10；l≥3 时 0.596）。归一化后 0 = 随机，1 = ceiling。

---

## 6. 权重与估计量（最容易踩的坑）

同一份数据上，三个指标隐含三个不同的加权：

| 指标 | 每个玩家的权重 | 直觉 |
|---|---|---|
| per-instance NLL（$\sum$ 所有 pair） | $C(l_p,2)\approx l_p^2/2$ | 随机**谱面对**，重度玩家主导 |
| l-weighted Spearman | $l_p$ | 随机 **(玩家, 谱面) 对** |
| macro NLL / macro Spearman | $1$ | 随机**玩家** |

所以"逐玩家 Spearman 按 l 加权"和"per-instance NLL"**不是同一个总体**。要公平比较两个指标，必须固定权重：

* 要 macro：`nll_macro = mean_p( nll_p )`，$\rho$ 用 macro；
* 要 l：`nll_lw = sum_p l_p nll_p / sum_p l_p`，$\rho$ 用 l-weighted；
* 要 pair：per-instance NLL，$\rho$ 用 $l^2$ 加权。

1k 上三者的差别不大（MSD：NLL micro 0.6402 / macro 0.6449；$\rho$ lw 0.451 / macro 0.444 / $l^2$ 0.460），但这是经验巧合，不是保证。**在论文里写清楚你报的是哪个。**

---

## 7. 训练目标

### 7.1 从评价到训练

如果模型输出的是 per-(玩家, 谱面) 的预测 loss $\hat L_{pm}$（越低越好），把"难度"取成 $-\hat L$，成对损失就是

~~~text
L_rank = - sum_p sum_{i<j in B_p}
           [ y_pij * log sigmoid( beta * (Lhat_pj - Lhat_pi) )
           + (1-y_pij) * log(1 - sigmoid(...)) ]
~~~

$B_p$ 是 batch 内玩家 $p$ 的谱面集合。这就是 RankNet / Bradley–Terry 的 pairwise log-loss；梯度是标准的 pairwise 梯度，可直接反传。$\beta$ 可以固定（例如 1；此时损失对 $\hat L$ 的**尺度**敏感），也可以学习（此时只约束序，尺度自由）。

### 7.2 与 MSE 的关系

* 只训 `L_rank`：模型只学"同一玩家内两张图的相对顺序"，**不校准 loss 的水平**；RMSE/pp 会变差。
* 只训 MSE：模型校准 loss，但排序不一定最优（MSE 被极端 ACC 主导）。
* 推荐：$\mathcal{L} = \mathrm{MSE} + \lambda\,\mathcal{L}_{rank}$，$\lambda$ 从 0.1–1 扫；或者两阶段（先 MSE，再 rank 微调）。

### 7.3 batch 与采样

当前 `step2/predict` 的 batch 是随机行；一个 batch 里同一玩家的行数很少，pair 更少。要用 rank loss，必须：

* 按玩家分组的 batch（每个 batch 取若干玩家 × 每个玩家的全部/采样谱面），或
* 显式的 pair sampler（每个 batch 采 $B$ 个 player 和每个 player 的 $k$ 张图，构造 $C(k,2)$ 对）。

否则 $\mathcal{L}_{rank}$ 的方差很大，且大部分 batch 没有有效 pair。

### 7.4 注意事项

* **不可辨识的部分**：rank loss 对"每个玩家加一个常数"和"对 $\hat L$ 做单调变换"不变；如果模型还要输出绝对 loss，必须保留 MSE 或事后校准。
* **leakage**：训练用 train pair，评测用 test pair；ceiling 也必须在 train 上拟合、test 上评估。
* **ties**：训练时 $s_{pi}=s_{pj}$ 的 pair 应跳过或给 0.5 目标，不能当负例。指标用 0.5；
  `--rank-ties half` 让训练损失与指标一致（默认 `skip` 保持旧实验不变）。
* **数值**：$\beta$ 太大时 sigmoid 饱和，log 要 clip；建议 $\beta$ 初始化 1、上限 1e2。

### 7.5 启用命令

~~~bash
# 主模型：MSE + 0.1 * rank loss
python scripts/step2_predict.py --tag 1k --models bias,mf_dot \
    --rank-loss add --rank-weight 0.1 --rank-beta 1.0 ...

# 只用 rank loss（输出是难度分数，不是校准后的 loss；RMSE/pp 会变差）
python scripts/step2_predict.py --tag 1k --models bias,mf_dot \
    --rank-loss only --rank-beta 1.0 ...

# CM3P MLP 同样支持
python scripts/step2_cm3p_mlp.py ... --rank-loss add --rank-weight 0.1
~~~

默认 `--rank-loss none`，所以旧实验逐位不变。启用后建议用 `scripts/step2_rank_eval.py`
在 test pair 上评估 rank-NLL（而不是只看 RMSE）。CM3P 内容→rank-NLL 的完整实验
（M0–M2、P2、new-player val）见 [docs/cm3p_rank.md](cm3p_rank.md)。

---

## 8. 报告模板

每次 rank eval 至少报：

| 字段 | 含义 |
|---|---|
| `n_rows` / `n_players` / `n_maps` / `n_instances` | 覆盖 |
| `nll_per_instance` | 原始损失（`--strict-pairs` 时是覆盖 pair 上的） |
| `n_instances_used` / `strict_pairs` | 参与 captured 的 pair 数 / 是否同集 |
| `captured_all` / `nll_all` | 全 pair 口径，作对照 |
| `bits_saved` | 相对 $\log2$ |
| `ceiling_nll` / `ceiling_coverage` | cross-fitted ceiling 及其覆盖 |
| `captured` | **主指标** |
| `link.kind` / `link.beta` | link 与 signal strength |
| `per_player.lw` / `macro` / `fisher_l3` | 辅助指标 |
| `norm_rho_lw` / `norm_rho_macro` | 归一化后的辅助指标 |
| `rho_bhat` / `tau_bhat` | 与 BT 尺度的秩相关（Spearman / Kendall） |
| `pearson_bhat` | 与 BT 尺度 `b` 的线性（Pearson）相关；同 b、同 map 集，对 `d` 的单调非线性缩放敏感 |
| `bootstrap.*` | 按 `player_id_base` cluster bootstrap 的 95% CI |
| `coverage_rows` / `coverage_maps` | 信号覆盖 |

**不要**只报 raw NLL 或 raw Spearman：前者在 $\log2$ 附近被压缩，后者没有绝对零点。

---

## 9. 自检

runner 在写 JSON 前应检查：

1. $\sum(n_{ij}+n_{ji}) = N$，且 $n_{ij}+n_{ji}$ 与 pair 表一致；
2. 随机 map 向量的 NLL $\approx \log2$（$\pm$ 0.002）；
3. ceiling NLL $\le$ 任何信号 NLL（**在相同 pair 集上**；容差 1e-6；否则拟合没收敛）；
4. `captured` 对随机信号 $\approx 0$，对 ceiling 自身 $\approx 1$；
5. `beta` $\ge 0$；`beta=0` 时 NLL 必须等于 $\log2$；
6. 所有 $d$ 有限；覆盖率和 tie 数已报；
7. cross-fit 的 `ceiling_coverage` 不能太低（1k 上约 0.94；显著低于 0.95 要在报告里说明）。
8. `captured <= 1`（容差 1e-6）。若 >1，先查 ceiling 是否只用了 `--split` 行拟合；
   应加 `--ceiling-fit all`，让 $b$ 用全部行、只在 split 行上评测。

---

## 10. 局限

* **加性假设**：BT 假设所有玩家共享一个 item 序；玩家×谱面的交互（专精、键型偏好）进入残差。若交互很大，全局序只是 consensus，[docs/rank_eval.md](rank_eval.md) 的 split-half 与 per-player 分布是诊断。
* **选择偏差**：$S_p$ 是玩家自己选的（MNAR），pairbest 又按 ACC 截断。1k 的结论只在"top-1000、$ACC\in(0.90,0.998]$、$n_{plays}\ge2$"这个切片上成立；不要外推到全难度段。
* **ACC 难度 ≠ pattern 难度**：ACC 受 OD 影响；MSD/star/analyser 测的是 pattern。两者是不同 construct（1k 上 disattenuated 相关只有约 0.26）。
* **ceiling 是模型不是真值**：它受 BT 假设、link、cross-fit 折数影响；报 `ceiling_coverage` 和 split-half 可信度。
* **稀疏**：57.9% 的谱面对只出现一次；虽然似然用了计数（不是 $\hat p$），但谱面级 $b$ 的方差仍然大，ceiling 的 split-half 相关只有 0.58。

---

## 11. 命令

~~~bash
# 单个信号
python scripts/step2_rank_eval.py \
    --in data/processed/step1_1k_pairbest.parquet \
    --signal-table data/processed/rank_signals_1k.parquet \
    --signals msd_overall --min-l 10 --folds 2 --bootstrap 400 \
    --ceiling-fit all --out results/rank_eval_1k/msd.json

# 多个信号（ceiling 按 map 集缓存，共享）
python scripts/step2_rank_eval.py \
    --in data/processed/step1_1k_pairbest.parquet \
    --signal-table data/processed/rank_signals_1k.parquet \
    --signals msd_overall,ett_075_overall,sunny_star,daniel_star,sunnywin_ln_star \
    --min-l 10 --folds 2 --bootstrap 400 --ceiling-fit all --common-maps \
    --out results/rank_eval_1k/all_signals.json \
    --per-player-out results/rank_eval_1k/all_signals_pp.csv

# 只用 test 行（模型 bm 的口径）。--ceiling-fit all 是默认值：b 用全部行拟合，
# 只在 test 行评测；写成 --ceiling-fit split 会复现旧的欠拟合 ceiling（captured 可能 >1）。
python scripts/step2_rank_eval.py ... --split test --ceiling-fit all --strict-pairs
~~~

ceiling 缓存在 `data/processed/rank_eval_cache/ceiling_<hash>.npz`；key 是 held-out map 集 + fold seed + min_l + split，重跑同一 map 集不会重拟合。
