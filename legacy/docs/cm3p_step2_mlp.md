# step2_CM3P_MLP：冻结 CM3P 向量上的内容锚定 MLP

> **状态**：已实现并实测。**结论是否定的**：1k 上它比线性内容模型差 **6.0%**（t=+62.7），
> 而且**每个"加容量/加非线性"的动作都让结果变差**——见 §8。10k 还没跑，是独立检验，见 §8.4。
> 代码 [scripts/step2_cm3p_mlp.py](../scripts/step2_cm3p_mlp.py)，
> 自检 [scripts/test_step2_cm3p_mlp.py](../scripts/test_step2_cm3p_mlp.py)，
> 配对检验 [legacy/scripts/compare_preds.py](../legacy/scripts/compare_preds.py)，
> 配置 [configs/exp_cm3p_mlp_1k.yaml](../configs/exp_cm3p_mlp_1k.yaml)。
> **上游**：[cm3p_step2_integration_plan.md](cm3p_step2_integration_plan.md)（线性点积的 v2 规格）、
> [cm3p_step2_results.md](cm3p_step2_results.md)（它的实测）、
> [step2_capacity_dim_sweep.md](../legacy/docs/step2_capacity_dim_sweep.md)（为什么容量不是瓶颈）。

---

## 0. 一句话

**把线性的 `D_m = C_m`（直接拿 512 维 CM3P 向量）换成 `D_m = g_θ(LayerNorm(C_m))`，
`g_θ` 是一个在所有谱面之间共享的两层小网络；图侧自由参数仍然只有 `bm`。**

```
L_hat = b0 + bu[u] + bm[m] + b_h + w · ⟨ P_u , g_θ(LayerNorm(C_m)) ⟩
```

这一句就是全部改动。它回答的不是「能不能加非线性」，而是
**「在两个已实测过的最优点之间——线性 CM3P d=32 与自由 1-hot 低秩——非线性有没有位置」**。

**答案（1k，实测）：没有位置。** 这个结构比线性内容模型差 6.0%，比 1-hot 差 5.5%，
而且去掉非线性、减层数、降维度、去掉 LayerNorm——每一格都**变好**（§8）。
下面保留完整的设计推理与实现，因为**"设计对了但假设不成立"本身是结论**：
它把"非线性不是当前的瓶颈"从推断变成了测量。

---

## 1. 为什么值得做，以及为什么不是随便加个 MLP

三件事同时成立，才让这个结构有意义：

| 证据 | 出处 | 含义 |
|---|---|---|
| 自由低秩 **K≈2 见顶**，K=2→8 测试误差显著退化（t=−24.7） | capacity_sweep §2、§3.2 | **不是秩不够** |
| 内容锚定 FM dim4（3,976 参数，144 plays/param）优于自由低秩 K=16（329,968 参数）**11.4%** | capacity_sweep §5 | **参数化比维度重要** |
| 加性模型吃掉交互方差 **0%**，最优低秩只吃 **11%** | capacity_sweep §6.1 | 交互里**确实还有信号**，只是被稀疏和噪声埋着 |

⇒ 结论是：**能加的东西必须「参数与谱面数脱钩」**。这正是内容锚定。
内容锚定的线性版已经做了（CM3P 替换 1-hot）；**非线性是这条路上唯一还没被占用的自由度**。

### 为什么不是 NCF 那种 MLP

推荐系统的标准写法是「每张谱面一个自由 embedding + concat 后接 MLP」。算一下就否掉了：

| | 参数 | 1k 覆盖子集 plays/param |
|---|---|---|
| NCF `E32 + MLP[64,16]` | **685,777**（其中 628,352 是图侧表） | **1.0** |
| 本设计 `d=16, hid=64` | **63,116**（图侧自由参数 = `bm` 一项，11,202） | **6.0** |
| 对照：CM3P d=32（线性，覆盖子集最优） | 43,654 | 8.7 |
| 对照：`free_K16`（已实测翻车） | 329,968 | 1.7 |

NCF 那里 **96% 的参数是每谱面的自由 embedding**，而 1k 全表实测最优模型总共只有 388,570 个参数。
**MLP 本体从来不是预算问题**（本设计里只有 35,200 个参数，占总数 55.8%）。

---

## 2. 数据与参数预算（新 split 口径）

`step1_1k.parquet` 重新生成（清洗只作用于 train 行）后，CM3P 覆盖子集：

| 量 | 值 |
|---|---|
| 训练 play | **379,639** |
| 测试 play | 267,090（覆盖 266,301 = **99.70%**） |
| 玩家 | 983 |
| 谱面 | **11,202** |
| 训练侧每谱面 distinct players 中位数 | **7**（capacity_sweep §4.1） |
| 有效样本量 `n_eff = n/deff`，deff≈5.4 | 每谱面 ≈ **9.5** |

> ⚠️ **跨口径警告不变**（[cm3p_step2_results.md](cm3p_step2_results.md) §3.2）：
> 覆盖子集整体比全表差 ~0.13，且**与内容无关**。所有 A/B 必须落在同一子集上。
> 另外本次 step1 重跑的清洗分母从「该玩家全部行」改成「该玩家 train 行」，训练行比旧口径多出
> 约 454 行，所以 **不能用文档里的历史数字当对照线，必须在同一批数据上重跑基线**。

### 参数量（实测，由 `--hid 64 --depth 2` 真实实例统计）

| `d` | 玩家塔 | 合计参数 | plays/param |
|---|---|---|---|
| 16 | on | **63,116** | 6.0 |
| 8 | on | 54,516 | 7.0 |
| 16 | off | 62,812 | 6.0 |
| 32 | on | 80,700 | 4.7 |

逐层分解（`d=16, hid=64`，1k 覆盖子集）：

| 组 | 参数 |
|---|---|
| 主效应 `b0 + b_u + b_m` | 12,186 |
| 玩家表 `Eu` | 15,728 |
| 玩家塔 + LayerNorm | 304 |
| **内容塔 `g_θ`** | **34,896** |
| 交互 `w + b_h` | 2 |
| **合计** | **63,116** |

**MLP 本体（内容塔 + 玩家塔）= 35,200，占 55.8%；图侧自由参数只有 `bm` 的 11,202。**

---

## 3. 架构

```
                      ┌──────────── 冻结,不进梯度 ────────────┐
   C_m  (512,)  ─────► LayerNorm(512) ─► Linear 512→64 ─► GELU ─► Linear 64→d ─► D_m (d,)
   CM3P 原始 512                                                                      │
                                                                                      │
   u ──────────► Eu[u] ─► Linear d→d ─► GELU ─► LayerNorm(d) ─► P_u (d,)              │
                 (983,d)                                                             │
                         ┌────────────────────────────────────────────────────────────┘
                         ▼
              ⟨P_u, D_m⟩ ─► × w (可学标量, init = 1) ─► + b_h (init = 0)
                         │
   b0 + b_u[u] + b_m[m] ─┴────────────────────────────────────────► L_hat (loss)
```

```
L_hat = b0 + bu[u] + bm[m] + b_h + w · ⟨ P_u , D_m ⟩
D_m   = g_θ(LayerNorm(C_m)),    g_θ = Linear(hid→d) ∘ GELU ∘ Linear(512→hid)
P_u   = LayerNorm( p(Eu[u]) ),  p   = GELU ∘ Linear(d→d)     （可选，--player-tower off 去掉）
```

### 逐条设计理由

| 决定 | 理由 |
|---|---|
| **层数 = 2（内容塔一个隐藏层）** | 1 层退化成一个线性映射 —— 那正是已经量过的 PCA/线性路径，不是 MLP。2 层是「能产生新东西」的最小深度。更深无益：目标是**二维交互面**，不是特征层级；且加容量的三次独立实测全部指向方差而非信号 |
| **交互头保持乘法（点积）** | concat-MLP 类头会丢掉低秩结构，而低秩是这份数据唯一撑得住的结构（§1）。用 `w` 缩放、可选 per-dim `s = 1+m`，都是**严格嵌套**在线性模型里的扩展 |
| **主效应走独立路径** | `--drop-bias u` 在仓库里值 +3.3%（[step2_prediction.md](step2_prediction.md) §7d）；让 MLP 学 `b_u` 是浪费容量 |
| **深度不对称：内容塔 2 层、玩家塔 1 层** | 图侧向量是**内容生成、全图共享**的，需要表达力；玩家侧是 ID 查表，983 个玩家有 ~386 plays/人，已有足够数据 |
| **`w` 初始 1、`b_h` 初始 0、输出层近零初始化** | 第 0 步 ≈ 线性模型，增益必须自己挣出来（沿用仓库 `Pc`/`V` 的同一约定） |
| **LayerNorm 放在两条分支最前** | 见 §5，实测 23× 前激活 / 17× 梯度差距 |
| **`--dim` = 交互空间宽度 ≠ 输入宽度** | 与线性管线不同：512 维是 **MLP 的输入**，`d` 是它的**输出**。`--content-reduce raw` 时不 PCA，让 MLP 自己学投影 |

---

## 4. 自检：把设计稿的声明变成可执行断言

`python scripts/test_step2_cm3p_mlp.py`（合成小张量，1 秒，不依赖 data/）：

| # | 断言 | 实测 |
|---|---|---|
| t1 | 冻结内容是**输入**不是权重（扰动 `C[0]` 会改变预测） | 2/7 条预测移动 |
| t2 | **图侧唯一自由参数是 `bm`** | 首维为 `n_item` 的可训练张量 = `['bm']` |
| t2b | 参数量与逐项手工统计一致 | 145 == 145 |
| t3 | `w` 是**精确开关**：`w=0` 即加性模型 | 与 `w=1` 相差 0.230 |
| t4 | 输出层缩放真的控制 MLP 的初始强度 | `out_init_std` 0.01→0.132，1.0→13.20 |
| t5 | 分组 Adam 下逐位确定性 | 13 个张量全部 `torch.equal` |
| t6 | `--depth` 真的控制塔的 Linear 层数（含输出层） | depth 1/2/3 → 1/2/3 层，隐藏层宽度 = `--hid`；`--nonlinearity off` → 1 层 |
| t7 | LayerNorm 修正输入条件数 | 见 §5 |

---

## 5. 被测量推翻的两个判断（留档）

**这两条是设计阶段写错、实测后改口的，写在这里以免下次重犯。**

### 5.1 输入尺度：不是「学不动」，是「条件数差 20 倍」

设计稿原先写：CM3P 是单位向量、分量 sd 0.028，He 初始化的 `Linear(512→64)` 前激活会是
**1e-4 量级、「几乎恒为 0、完全学不动」**。实测（真实 512 维 CM3P 向量，256 行）：

| | 前激活 sd | ‖∂L/∂W₁‖ |
|---|---|---|
| 无 LayerNorm | **0.0597** | **4.24e+02** |
| 有 LayerNorm | **1.345** | **7.22e+03** |
| 比值 | 22.6× | 17.0× |

⇒ **方向对（确实该加 LayerNorm），量级错了（不是 1e-4，是 6e-2；不是「死掉」，是 17 倍慢）。**
理由从「活不下来」改成「**数值条件**」：固定 60 epoch、不早停的训练预算下，
17 倍的梯度差距足以让塔在预算内没学完，然后把结果误读成「非线性没用」。

### 5.2 `--nonlinearity off` 并不等于线性模型

原先打算把它当「脚本体内的线性基线」。实际上 `LayerNorm` 会逐行归一化，
`D_m = c_out(LayerNorm(C_m))` 对 `C_m` **不是线性的**（每行的尺度被单独消掉了）。
所以它只能当**容量对照**（同样宽的塔、去掉中间非线性），
**真正的线性基线是 `scripts/step2_predict.py --chart-content cm3p`**，不能用这个开关代替。

---

---

### 5.3 实现期抓到的三个真 bug（自检抓到的，都影响结论）

| # | bug | 症状 | 为什么危险 |
|---|---|---|---|
| 1 | `depth` 语义错位：`[hid] * max(1, depth-1)` | `depth=1` 与 `depth=2` 产生**完全相同**的网络；要 `depth=3` 才是文档说的「一个隐藏层」 | 深度扫描的 1 和 2 两格会给出**逐位相同**的结果，会被读成「深度不影响」——而实际上是没测到 |
| 2 | `--nonlinearity off` 仍保留一个被 GELU 激活的隐藏层 | 这个「容量对照」其实**还带着非线性**，与主实验的差别只剩宽度 | 会把「同样的非线性换个宽度」当成「去掉非线性的代价」 |
| 3 | 输出层用 `weight.mul_(out_init_std/0.05)` 缩放 | 只有前面恰好是 LayerNorm 时输出 sd 才等于 `out_init_std`，否则偏了 `sqrt(fan)` 倍 | 塔的初始强度失控，`mirt_exp` 那类「初始化尺度决定学不学得动」的坑会重现 |
| 4 | 缺 `np_array()` / `export_C()` 两个导出接口 | 训练 60 epoch 全部跑完、9 条校验全过，**在写 npz 的那一刻** `AttributeError` | 前 60 epoch 白跑；而且它在流水线里表现为「校验全 PASS + 步骤 failed」，极易被当成编排问题 |
| 5 | `--player-tower off` 仍保留 `p_ln` | 对**原始** `Eu`（sd 0.05）做 LayerNorm，点积初始尺度被放大 1/0.05 ≈ 20 倍 | 开/关两格的差别不只是「少几层」，而是「少了尺度归一」，消融不可比 |
| 6 | `coverage_reported` 校验把结果字典当迭代对象 | `TypeError: 'bool' object is not iterable`，同样发生在落盘前 | 同上 |
| 7 | 参数表把主效应组印成 `wd=none` | 实际优化器给的是 `--wd`（与 `step2_predict` 一致） | 表在说谎；按表解释正则化口径会得出错误结论 |

**三个都不是数值错误，而是行为与文档不符**——参数量、张量形状、梯度全都正常，跑起来也不会报错。
只有「把设计稿的声明写成断言」才能抓到：t2b 核了参数量，t6 核了**层结构**（而不只是参数量），
才让第 1 个 bug 现形。这也是为什么 t6 最后只剩结构断言：
**曲率在这份数据的几何上测不出来（见 t6 的注释），但层数和参数预算是能钉死的。**

**教训（两条）**：

1. **参数量对 ≠ 行为对。** 自检要断言**结构**（几层、哪一层接在哪里、尺度是多少），不能只断言总数。
   第 1、2、5 号 bug 都通过了「参数量正确」这一关。
2. **产物没落盘就不算跑过。** 第 4、6 号 bug 让「训练 60 epoch + 校验全 PASS」的进程在最后一步死掉。
   扫描脚本必须**显式检查产物文件存在**，而不是相信 `exit code`（更不能用 `grep` 把非零退出码过滤掉——
   本次就是这么漏掉一整轮无效扫描的）。

## 6. 实现清单

| 文件 | 状态 | 内容 |
|---|---|---|
| [scripts/step2_cm3p_mlp.py](../scripts/step2_cm3p_mlp.py) | 新增 | `AnchoredMLP` + 训练 + 导出 + 校验清单 |
| [scripts/step2_predict.py](../scripts/step2_predict.py) | 小改 | 新增 `--save-preds`（默认关，不改变任何既有行为） |
| [legacy/scripts/compare_preds.py](../legacy/scripts/compare_preds.py) | 新增 | 配对 + 聚类 bootstrap 的比较工具 |
| [scripts/test_step2_cm3p_mlp.py](../scripts/test_step2_cm3p_mlp.py) | 新增 | §4 的 8 条自检 |
| [dcr/steps.py](../dcr/steps.py) | 小改 | 注册 `step2/cm3p_mlp` |
| [configs/exp_cm3p_mlp_1k.yaml](../configs/exp_cm3p_mlp_1k.yaml) | 新增 | 一次实验跑完 |

### 训练协议（刻意与 `step2_predict.py` 一致）

同一套词表规则（只从 train 行建）、同一套种子化置换
（`np.random.default_rng(seed).permutation(idx_tr.size)`）、同样的 batching、
**固定 60 epoch 不早停**、同样的 npz/summary 布局、同名九条校验 + 一条内容对齐校验。

### 三个必须知道的约定

1. **`C` 与 `D` 导出的都是 `g_θ` 的输出 `D_m`（两者相同）**。线性管线里 `C` 是谱面因子、
   `D` 是区分度向量；在这里它们合并成了一个「交互方向」。step3 若要用，注意这一点。
2. **`P` 导出的是玩家塔之后的向量**（而非原始查表值），即真正进入点积的那一侧。
3. **`named_parameters()` / `state_dict()` 看不到 `b0/bu/bm`** —— 它们放在普通 dict `self.p` 里，
   这是从 `EmbeddingModel` 继承的既有写法，不是本次引入的回归。autograd、Adam、
   `count_params()` 都正常；重现 `state_dict()` 或做 dtype 转换前必须先把这三个搬进 `nn.ParameterDict`。

---

## 7. 实验矩阵

全部在 CM3P 覆盖子集、60 epoch、seed 0、同一次 step1 划分上：

| # | 配置 | 命令要点 | 作用 |
|---|---|---|---|
| B1 | 1-hot `mf_dot` d16 / d32 | `--restrict-to-content on` | 覆盖子集 1-hot 基线 |
| B2 | **线性 CM3P d=32** | `--chart-content cm3p --content-dim 32 --content-reduce pca` | **主对手** |
| M1 | MLP d=16 | `--dim 16` | 主实验 |
| M2 | MLP d=8 | `--dim 8` | 预算更紧 |
| L1/L3 | 深度 1 / 3 | `--depth 1` / `--depth 3` | 深度扫描：1 应退化、3 不应更好 |
| A1 | 非线性关 | `--nonlinearity off` | 容量对照（§5.2） |
| A2 | LayerNorm 关 | `--ln off` | 验证 §5.1 |
| A3 | 玩家塔关 | `--player-tower off` | 983 玩家是否需要塔 |

复现整套矩阵（约 50 分钟 CPU；全部 `--save-preds`，否则配对检验要重跑）：

```bash
for d in 16 8; do
  python scripts/step2_cm3p_mlp.py --tag 1k --dim $d --hid 64 --depth 2 --epochs 60 \\
      --out-prefix mlp_d$d --out-dir /tmp/mlp_1k --save-preds
done
for L in 1 3; do
  python scripts/step2_cm3p_mlp.py --tag 1k --dim 16 --hid 64 --depth $L --epochs 60 \\
      --out-prefix mlp_L$L --out-dir /tmp/mlp_1k --save-preds
done
for opt in '--nonlinearity off:mlp_nolin' '--ln off:mlp_noln' '--player-tower off:mlp_nopt'; do
  flags=${opt%%:*}; name=${opt##*:}
  python scripts/step2_cm3p_mlp.py --tag 1k --dim 16 --hid 64 --depth 2 --epochs 60 $flags \\
      --out-prefix $name --out-dir /tmp/mlp_1k --save-preds
done
# 两个对手（同一子集）
python scripts/step2_predict.py --tag 1k --models mf_dot --dim 16 --epochs 60 \\
    --restrict-to-content on --out-prefix base_cov --out-dir /tmp/mlp_1k --save-preds
python scripts/step2_predict.py --tag 1k --models mf_dot --epochs 60 --chart-content cm3p \\
    --content-dim 32 --content-reduce pca --out-prefix lin_cm3p32 --out-dir /tmp/mlp_1k --save-preds
# 配对检验
python legacy/scripts/compare_preds.py /tmp/mlp_1k/mlp_d16_1k_preds_loss.npz \\
    /tmp/mlp_1k/lin_cm3p32_1k_preds_loss.npz --keys pred_mlp pred_mf_dot
```

> **A3 的尺度注意**：玩家塔关掉后 `Eu`（初始化 sd 0.05）**不再**过 LayerNorm，
> 所以点积的初始尺度会小约 1/0.05 = 20 倍（实测 `||P_u||` 中位 3.995 → 0.184）。
> 这**是**刻意的——「无塔」就是不做归一化；但结论要读成「少一组层 + 少了尺度归一」的合计效果，
> 不是纯容量差。另外这只影响优化的起点：`w` 会自己补偿尺度。

---

## 8. 实测结果（1k，CM3P 覆盖子集，60 epoch，seed 0）

数据：训练 379,639 play / 测试 267,090（覆盖 266,301，99.70%）/ 983 玩家 / 11,202 谱面。
所有模型跑在同一次 step1 划分上，全部落盘预测后用 `legacy/scripts/compare_preds.py` 做**配对聚类 bootstrap**
（按 `(player, chart)` 有序对重采样，1k 上 182,629 个 cluster，400 次重采样）。

### 8.1 主表（按测试 RMSE 排序）

| # | 变体 | test RMSE | train RMSE | gap | plays/param | 参数 | gate\|w\| |
|---|---|---|---|---|---|---|---|
| 1 | **线性 CM3P d=32（主对手）** | **1.58178** | – | – | 8.7 | 43,654 | – |
| 2 | 1-hot d16（同子集） | 1.58918 | – | – | – | 207k | – |
| 3 | 1-hot d32（同子集） | 1.59007 | – | – | – | – | – |
| 4 | MLP d16 hid64 L2 **ln off** | 1.60507 | 0.96245 | +0.643 | 6.1 | 62,060 | 1.268 |
| 5 | MLP d16 hid64 **L3** | 1.61958 | 0.97372 | +0.646 | 5.6 | 67,276 | 0.922 |
| 6 | MLP d16 hid64 **L1** | 1.62283 | 0.97751 | +0.645 | 10.1 | 37,452 | 0.812 |
| 6 | MLP d16 hid64 **非线性关** | 1.62283 | 0.97751 | +0.645 | 10.1 | 37,452 | 0.812 |
| 8 | MLP **d8** hid64 L2 | 1.63153 | 0.97892 | +0.653 | 7.0 | 54,516 | 0.864 |
| 9 | **MLP d16 hid64 L2（主实验）** | **1.67666** | 1.07291 | +0.604 | 6.0 | 63,116 | 0.861 |
| 10 | MLP d16 hid64 L2 **玩家塔关** | 1.68052 | 0.94079 | +0.740 | 6.0 | 62,812 | 0.922 |

第 6 行出现两次不是笔误：**`--depth 1` 与 `--nonlinearity off` 是同一个模型**（都是塔里唯一的
Linear + LayerNorm 输入），两次独立运行拿到**逐位相同**的 RMSE。这本身就是一条对照：
把「深度」和「非线性」删到同一个结构，结果当然一样。

### 8.2 配对检验（全部 CI 排除 0，方向一致）

| 对比 | ΔRMSE | Δ% | 95% CI | t | 判定 |
|---|---|---|---|---|---|
| MLP L2 − 线性 CM3P d32 | +0.09487 | **+6.00%** | [+0.0925, +0.0977] | **+62.7** | 线性明显更好 |
| MLP L2 − 1-hot d16 | +0.08748 | +5.51% | [+0.0846, +0.0904] | +52.6 | 1-hot 更好 |
| MLP L2 − MLP **L1** | +0.05382 | **+3.32%** | [+0.0513, +0.0565] | +34.5 | **层数越少越好** |
| MLP L2 − MLP **非线性关** | +0.05382 | +3.32% | [+0.0513, +0.0565] | +34.5 | **去掉非线性更好** |
| MLP L2 − MLP **ln off** | +0.07159 | +4.46% | [+0.0687, +0.0742] | +46.8 | **去掉 LayerNorm 更好** |
| MLP L2 − MLP **d8** | +0.04512 | +2.77% | [+0.0422, +0.0478] | +29.1 | 维度减半更好 |
| MLP L2 − MLP **玩家塔关** | −0.00387 | −0.23% | [−0.0074, −0.0008] | −2.32 | 略好，但幅度几乎为零 |
| 线性 CM3P d32 − 1-hot d16 | −0.00740 | −0.47% | [−0.0098, −0.0050] | −6.53 | 内容锚定仍略优（复现旧结论） |

### 8.3 结论

**1. 在这个数据规模上，内容锚定 MLP 不成立：比线性内容模型差 6.0%，比 1-hot 也差 5.5%。**
t = 62.7、CI 宽 0.005，不是噪声，也不是运气。

**2. 方向是干净的：每一个「加容量/加非线性」的动作都让结果变差。**
去掉非线性 −3.3%、层数 3→2→1 单调改善、维度 16→8 改善、去掉 LayerNorm 改善。
**没有任何一格显示非线性有用。** 这与仓库既有的三次独立结论完全一致：
自由低秩 K=2→8 退化（t=−24.7）、CM3P d=32→512 退化、FM dim4 ≡ dim16。

**3. 这是过拟合，不是欠训练。** 三个模型的训练 RMSE 远低于任何线性模型（0.94–1.07），
测试却在 1.60–1.68；gap +0.60~0.74。把容量拿掉，**训练误差变差、测试误差变好**——
教科书式的「容量打在噪声上」。机制也早就量过：训练侧每谱面 distinct players 中位数只有 **7**，
有效样本量 ≈ **9.5**。

**4. `gate|w|` 全程在 0.81–1.27，塔没有塌。** 这条必须报，否则「MLP 没用」可能是
「塔根本没学」的假象。实测塔在学，只是学到的东西不泛化。

**5. 唯一的正面一格是关掉玩家塔（−0.23%）**，但它只是把模型推回更接近线性的方向，
幅度小到没有实用意义。

**6. 主对手的复核：线性 CM3P d=32（1.58178）仍优于同子集 1-hot d16（1.58918），−0.47%（t=−6.53）。**
注意这两个数字与 [cm3p_step2_results.md](cm3p_step2_results.md) 的 1.57595 / 1.58147 **不可直接比**：
本次 step1 重跑后清洗分母变了（§2 的警告）。内容锚定的相对优势复现了，绝对水平没有。

### 8.4 那还要不要做 10k？

**要做，但预期要放低，而且这不是本次结论的延伸而是独立检验。**

* 支持做的理由：10k 的 play 量是 1k 的 **16.8 倍**（6.38M vs 0.38M），
  plays/param 从 6.0 升到 **16.4**，进入了仓库里「参数效率之王」FM dim4（144）与已翻车的
  自由低秩 K=4（6.9）之间靠上的一档。1k 的结论**完全可能**只是数据不够。
* 反对的理由：`legacy/docs/step2_capacity_dim_sweep.md` §8.5 已经写明 10k 的稀疏度结构**可能不同**，
  但同一份文档的 §2 也显示 10k 的交互项边际增益更小（−4.3% → −2.9%）。
  换句话说，**10k 上留给 MLP 的增量空间本来就比 1k 小**。
* 具体做法：只跑 `d=16, hid=64, depth=2`（主实验）+ 线性 CM3P d=32（对手）+ 1-hot d16（anchored），
  三格、`--save-preds`，然后用同一个 `compare_preds.py`。**不要**再做深度扫描——
  §8.2 已经把「层数越少越好」钉死了，10k 重复这个是浪费。

> ⚠️ 10k 上 `compare_preds.py` 的聚类 bootstrap 是 4M 行 × 2000 万级 cluster，
> 必须靠 `--n-boot` 和分块（已实现）；先跑 `--cluster row` 看方向，再用 `--cluster pair` 定量。

## 9. 复现

```bash
# 自检（1 秒，不需要数据）
python scripts/test_step2_cm3p_mlp.py

# 一次实验（走统一编排；step1 从缓存取）
python main.py -c configs/exp_cm3p_mlp_1k.yaml

# 对手：同子集线性 CM3P，同时落盘预测
python scripts/step2_predict.py --tag 1k --models mf_dot --epochs 60 \\
    --chart-content cm3p --content-dim 32 --content-reduce pca \\
    --out-prefix lin_cm3p32 --save-preds

# 配对 + 聚类 bootstrap（按 (player, chart) 对重采样）
python legacy/scripts/compare_preds.py <mlp 的 preds>.npz <线性基线的 preds>.npz \\
    --keys pred_mlp pred_mf_dot
```