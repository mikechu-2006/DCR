# 实验：难度序的 rank-NLL（MinaCalc / osumania_map_analyser）

本文件记录两个外部难度系统在 **1k / step1 pairbest** 上的 rank-NLL 结果，以及它们和项目里已有
step2 实验的关系。指标定义见 [docs/metrics.md](metrics.md)；实现见
[`dcr/rank_metrics.py`](../dcr/rank_metrics.py) 与 [`scripts/step2_rank_eval.py`](../scripts/step2_rank_eval.py)；
原始 JSON 在 `results/rank_eval_1k/`。

**一句话结论**

* **MinaCalc `msd_overall`**：NLL **0.64017**，cross-fitted ceiling **0.56288**，**captured 40.7%**，
  per-player lw Spearman **0.451**；单看 skillset，`msd_jackspeed` 反而更强（captured 37.1%，
  lw 0.410），`msd_stream` 最弱（16.4%）。
* **osummania_map_analyser** 的 `sunny_star` / `daniel_star` 最强：captured **47.2% / 47.3%**，
  lw **0.511 / 0.505**；`ett_068_overall` 44.6%，`interlude_star` 41.0%，`ett_075_overall` 37.5%。
* **LN 专项 `sunnywin_ln_star` 没有在 LN 上赢**：全量上被 34% 的 0 稀释为 0 信号；在 Mix/LN 子集上
  captured **30.8%**（lw 0.479），低于通用 `sunny_star` 的 **39.7%**（lw 0.542）。
* **两个指标大体同向，但不完全同序**：RC 子集上 Roxy/Azusa 的逐玩家 lw 更高（0.524/0.528），
  但 MSD 的 captured 更高（0.4094 vs 0.4066/0.3891）。这正是 metrics.md §10 说的
  "Spearman 每对等权、NLL 更看重高置信 pair"。

---

## 0. 口径与数据

| 项 | 值 |
|---|---|
| 输入 | `data/processed/step1_1k_pairbest.parquet` |
| 观测单位 | 一行一个 (玩家段, 谱面) 对 |
| 行 / 玩家段 / 谱面 | 109,706 / 1,039 / 6,828 |
| 比较数 $N$ | 12,476,783（每个玩家的 $C(l_p,2)$ 求和） |
| 唯一谱面对 | 3,758,711；**57.9% 只出现一次，76.8% 全票 0/1** |
| 玩家 cluster | `player_id_base`（同一玩家的多个 epoch 不拆到两边） |
| link | `linear`：$g(d)=\beta z(d)$，$\beta\ge0$ 由一维凸问题的导数二分求 MLE |
| ceiling | 无约束 BT，按 cluster 做 2 折 cross-fit；held-out 只保留"两张图都在训练折出现过"的 pair |
| ceiling 值 | **0.56288**（覆盖 93.6% 的 held-out pair） |
| 逐玩家 Spearman | $l_p\ge10$，l-weighted / macro / Fisher-z 三种都报 |
| 置信区间 | 按 `player_id_base` cluster bootstrap 300 次 |

> ceiling 的**拟合行**是全部行（`--ceiling-fit all`，runner 默认），**评测行**是 `--split` 指定的行。
> 本表的 full-coverage 信号都是 `--split all`，拟合行 = 评测行 = 全量，所以旧数字不变。
> `--split test` 的旧口径只用 test 行拟合 $b$，会严重欠拟合：1k 实测 test-only ceiling 0.6399
> （coverage 0.833），比 MSD 的 0.6385 还高，captured 会 >1；同一信号用全行拟合是 0.5628 / 0.419
> （coverage 0.9396；`--strict-pairs` 同 pair 集后 captured 0.440）。
> 子集表（§4.3/§4.4）若重跑，`--ceiling-fit all` 会用全部 6,828 张图拟合 $b$，ceiling 可能略低；
> 必须连同 baseline 一起重跑，不能把新 captured 和旧表直接比。
> 需要分子/分母严格同 pair 集时加 `--strict-pairs`（1k test 上 coverage 0.9396）；它要求
> ceiling 缓存里有 seen-fold mask，旧缓存需 `--no-cache` 重算。

### 0.1 两张表不是一回事（这是最容易误读的地方）

| 项 | **step1/pairbest**（本文件 §3/§4 用） | step1/clean（legacy，§2 的旧实验） |
|---|---|---|
| 文件 | `data/processed/step1_1k_pairbest.parquet` | `data/processed/step1_1k.parquet` |
| 观测单位 | (玩家段, 谱面) 对 | play 行 |
| 行 / 玩家 / 谱面 | **109,706 / 1,039 / 6,828**（cluster 942） | 1,399,615 / 990 / 20,655 |
| `n_plays` | ≥2（每对只留最高成绩） | 未折叠 |
| loss 范围 | [-6.214, -2.303]（ACC 0.900–0.998） | [-12.466, -0.267] |
| split | train 76,567 / test 33,139 | train 671,105 / test 415,710 / dropped 312,800 |
| 1k 基线 RMSE | **bias 0.7443 / mf_dot 0.7015** | bias ≈1.52 / mf_dot 1.4548 |

> **如果看到 1.5 左右的 RMSE，那一定是 legacy clean（§2 的旧实验），不是本文件的 rank 实验。**
> 本文件的 §3/§4 全部在 pairbest 上；`results/rank_eval_1k/msd.json` 的 `meta.in` 就是
> `data/processed/step1_1k_pairbest.parquet`，`n_rows_total=109,706`、`n_maps_total=6,828`
> 可以逐项核对。

> **不要混口径**：项目里已有的 `step2/minacalc`、`step2/minacalc_L2`、`step2/map_analyser`
> 是 **legacy clean** 的 play 级回归实验（test 418,876 行），它们的 RMSE 不能和这里的 NLL 并排；
> §2 只做定性对照。
>
> **pairbest 的结构**：每个 `(player_id, beatmap_id)` 一行；对固定内容信号（MSD/star/analyser）
> 不需要 train/test，runner 用 `--split all`。`n_plays` 是 best-of-k 的选择变量，不是重复观测；
> 每个玩家的 $C(l_p,2)$ 个 pair 是派生量，CI 按 player cluster。ceiling 仍然 cross-fit；
> `--folds 1` 只是快速看 in-sample ceiling（ceiling 本身偏乐观，captured 反而偏低）。

---

## 1. 指标回顾（细节见 metrics.md）

对每个玩家、每个无序对 $\{i,j\}\subseteq S_p$：

~~~text
y_pij = 1[ s_pi > s_pj ]                  s = -loss = -log(1-ACC)
q_ij  = sigmoid( beta * (z(d_j) - z(d_i)) )
L(d)  = - sum_{i<j} [ n_ij log q_ij + n_ji log(1-q_ij) ]
N     = sum_{i<j} (n_ij + n_ji)
~~~

~~~text
bits_saved = 1 - (L(d)/N) / log 2
captured   = (log 2 - L(d)/N) / (log 2 - L*/N)      L* = cross-fitted ceiling
~~~

$\log2=0.693147$；$L^*/N=0.56288$；所以 captured 的分母是 $0.13027$。原始 NLL 只差 0.03–0.06 nats，
captured 把它放大到 0–47%，这是这张表主要看 captured 的原因。

---

## 2. 附：已有的 legacy clean MinaCalc 回归实验（**不是** pairbest）

> **先看这一句**：§2 的 RMSE 1.5271 / 1.5477 是 **legacy clean**（1,399,615 行、play 级、
> loss [-12.47, -0.27]）的数字，**不是 pairbest**。pairbest（109,706 行）的同一模型是
> bias **0.7443** / mf_dot **0.7015**（`experiments/20261020_193449_pairbest_1k`）。
> 本文件的 rank 实验（§3/§4）全部在 pairbest 上；§2 只是把已有实验的细节归档，不能拿它的
> RMSE 和 rank-NLL 并排。

这一节是"已经做完的 MinaCalc 实验"的细节，**不是** rank-NLL。来源：
[legacy/docs/step2_minacalc.md](../legacy/docs/step2_minacalc.md)、
[legacy/docs/step2_minacalc_L2.md](../legacy/docs/step2_minacalc_L2.md)、
[legacy/docs/step2_map_analyser.md](../legacy/docs/step2_map_analyser.md)。

### 2.1 资产

`data/processed/minacalc_msd_1k.parquet`（21,949 张白名单谱面）：

* 列：`beatmap_id, msd_overall, msd_stream, msd_jumpstream, msd_handstream, msd_stamina,
  msd_jackspeed, msd_chordjack, msd_technical, calc_version, keycount, n_notes, n_holds,
  hold_ratio, rate, goal`；
* 后端：本地 libminacalc（`calc_version=505`），goal=0.93，6 skills 的 required_skill；
* 只依赖 `.osu` 正文，不依赖成绩；
* **已知坑**：goal=0.93 是 Etterna 的 score 语义，不是 mania ACC；NoteInfo 没有长按时长（长按只算
  头部一个 tap）；OD 不在 MSD 里。

### 2.2 L1：MSD 作为冻结特征

模型 `L_hat = b0 + bu + bm + alpha·D_std + <P,C_unit>`；`bias` 是同一子集基线。
1k、legacy clean、pair-random、60 epochs：

| 配置 | test RMSE | Spearman | 说明 |
|---|---|---|---|
| `bias` | **1.5271** | 0.7329 | 基线 |
| `msd_add`（7 维主效应） | 1.5667 | — | 主效应在 bm free 时有害 |
| `msd`（主效应 + 交互） | **1.5477** | 0.7299 | 最好的 MSD 配置 |
| `msd --msd-bm msd`（冷启动） | 1.7137 | 0.6519 | 谱面截距完全由 MSD 生成 |
| 冷启动的 `bias` | 1.8707 | 0.4969 | 无谱面信息 |

### 2.3 L2：gap / IRT 形式

模型 `x = a + sum_k w_k (theta_k - D_k)`，`loss = -softplus(x)`（logit 链接）：

| 配置 | test RMSE | 备注 |
|---|---|---|
| `bias` | **1.5271** | 与 L1 的 bias 逐位一致 |
| `gap`（bm free, logit） | 1.5810 | w 收敛到 [0.29,0.02,0.05,0.07,0.52,0.16,0.06] |
| `gapfree`（bm free, logit） | 1.5837 | 同量级 |
| `gap`（identity 链接） | 1.5274 | **w→0**，优化器主动把 gap 关掉，退化成 bias |
| `gap --gap-bm msd`（冷启动） | 1.7649 | 明显差于自由 bm |

### 2.4 这些实验说明了什么、没说明什么

* 说明：在 **bm free** 的 pair-random 设定下，MSD 作为特征没有稳定增益（1.5477 vs 1.5271），
  因为每张图的自由截距 `bm` 已经吸收了谱面效应；MSD 的价值在 **bm 不可用**（冷启动）时才体现为 RMSE。
* 没说明：这些 RMSE 不回答"MSD 的难度序对不对"。RMSE 可以靠 `bm` 和 `bu` 拿到，序是另一回事。
  这就是下面 rank-NLL 实验的动机。

---

## 3. MinaCalc 的 rank-NLL 实验（1k pairbest）

### 3.1 信号

`msd_overall`（6 skills 的 aggregate，goal=0.93；越高越难）。
还在同一张表上跑了 3 个 skillset 作对照：`msd_stream` / `msd_jackspeed` / `msd_technical`。

### 3.2 主结果

| 信号 | NLL | ceiling | captured | lw $\rho$ | macro $\rho$ | Fisher $\rho$ | $\beta$ |
|---|---|---|---|---|---|---|---|
| `msd_overall` | **0.64017** | 0.56288 | **0.4067** | **0.4513** | 0.4498 | 0.4692 | 0.729 |
| `msd_jackspeed` | 0.64480 | 0.56288 | 0.3711 | 0.4095 | 0.4118 | 0.4227 | 0.565 |
| `msd_technical` | 0.66408 | 0.56288 | 0.2232 | 0.3204 | 0.3477 | 0.3329 | 0.475 |
| `msd_stream` | 0.67180 | 0.56288 | 0.1639 | 0.2468 | 0.2600 | 0.2543 | 0.337 |

`msd_overall` 的 95% cluster-bootstrap CI：captured **[0.356, 0.451]**，NLL **[0.6338, 0.6477]**，
lw $\rho$ **[0.437, 0.465]**。bits_saved $=1-0.64017/\log2=0.0764$。

**读法**：

* 40.7% 的"可实现成对 log-loss 降幅"被 MSD 的序拿到；换成 entropy 口径是 7.64%。
* 三个 skillset 里 `msd_jackspeed` 最强，`msd_stream` 最弱。说明在这批 top-1000 玩家上，
  "jackspeed 难度"比"stream 难度"更能区分同一玩家对两张图的相对表现；`msd_overall` 是 aggregate，
  被弱方向拉低。
* `msd_overall` 与 analyser 的 `ett_0723_overall` 数值几乎一样（NLL 0.64017 vs 0.64088，lw 0.451 vs 0.449），
  与两张表在谱面级 $r=0.999$ 的已知一致性吻合——这是一个 sanity check。
* 与行为隐含的 BT scale 的相关只有 $\rho(d,b)=0.237$；用 split-half 可信度 0.736 去衰减校正后约 0.276。
  MSD 测的是 **pattern 难度**，行为 $b$ 测的是 **ACC 难度**；两者不是同一个 construct（见 §7）。

---

## 4. osumania_map_analyser 的 rank-NLL 实验

### 4.1 资产与版本

`data/processed/map_analyser_1k.parquet`：

* 插件 **2.1.0**，commit `448ed7c789af3d715acfb8370e41244cc3c00865`；
* 21,949 张白名单谱面 **全部产出，0 失败**；
* 135 列 = `beatmap_id` + 109 数值特征 + `status` / `patt_category` / `patt_mode_tag` +
  `analyser_version` / `analyser_commit` / `rate` / `goal`；
* 特征组：`ett` 48（6 个 MinaCalc 版本 × 8 skillset）、`est` 13（Sunny / Daniel / Azusa /
  Roxy / Interlude / SunnyWindow LN）、`patt` 34（键型）、`struct` 14（OD/BPM/时长/密度）；
* 集成方式与特征分组见 [legacy/docs/step2_map_analyser.md](../legacy/docs/step2_map_analyser.md)；
  上游是 [LeoBlackMT/osumania_map_analyser](https://github.com/LeoBlackMT/osumania_map_analyser)，
  本仓库的 clone 不入库（当前 `external/osummania_map_analyser` 已不在；缓存表就是资产）。

**LN 口径**（来自插件）：`sunnywin_ln_star` 只算 LN 段；`patt_mode_tag` 按
`lnRatio ≤ 0.15 → RC`、`≥ 0.90 → LN`、中间 `Mix`（`Mix` 再看 HB 行比例）；
本批 21,949 张里 RC 11,816 / HB 5,267 / Mix 4,734 / LN 128。

### 4.2 全量（6,828 张 step1 谱面，12.48M 比较，ceiling 0.56288）

| 信号 | 来源 | NLL | captured | lw $\rho$ | macro $\rho$ | $\rho(d,b)$ |
|---|---|---|---|---|---|---|
| `daniel_star` | Daniel | **0.63156** | **0.4728** | 0.5053 | 0.5066 | 0.2509 |
| `sunny_star` | Sunny | 0.63165 | 0.4721 | **0.5114** | **0.5104** | 0.2459 |
| `ett_068_overall` | MinaCalc 0.68 | 0.63506 | 0.4459 | 0.4610 | 0.4593 | 0.2363 |
| `interlude_star` | Interlude | 0.63975 | 0.4099 | 0.4586 | 0.4543 | 0.2403 |
| `msd_overall` | 本地 MinaCalc 505 | 0.64017 | 0.4067 | 0.4513 | 0.4498 | 0.2374 |
| `ett_0723_overall` | MinaCalc 0.72.3 | 0.64088 | 0.4012 | 0.4486 | 0.4469 | 0.2380 |
| `ett_075_overall` | MinaCalc 0.75 | 0.64434 | 0.3746 | 0.4393 | 0.4367 | 0.2375 |
| `msd_jackspeed` | 本地 MinaCalc | 0.64480 | 0.3711 | 0.4095 | 0.4118 | 0.2586 |
| `star` | osu! star | 0.66085 | 0.2479 | 0.3296 | 0.3310 | 0.2231 |
| `notes_per_sec` | analyser struct | 0.66102 | 0.2466 | 0.3059 | 0.3000 | 0.1753 |
| `msd_technical` | 本地 MinaCalc | 0.66408 | 0.2232 | 0.3204 | 0.3477 | 0.2062 |
| `msd_stream` | 本地 MinaCalc | 0.67180 | 0.1639 | 0.2468 | 0.2600 | 0.1693 |
| `n_notes` / `count_total` | analyser / meta | 0.68666 | 0.0498 | 0.1051 | 0.1132 | 0.1237 |
| `od` | analyser struct | 0.69315 | 0.0000 | 0.0188 | 0.0118 | 0.0021 |
| `ln_ratio` | analyser patt | 0.69315 | 0.0000 | −0.0878 | −0.0723 | −0.1156 |
| `sunnywin_ln_star` | SunnyWindow LN | 0.69315 | 0.0000 | −0.0211 | 0.0058 | −0.0956 |

> 全部 full-coverage 信号共享同一 pair 集和 ceiling（0.56288），所以可以直接比。
> `sunnywin_ln_star` 在**全量**上 34% 的谱面是 0（无 LN），这些 0 把序稀释成了随机；
> 它的有意义结果在 §4.3 的 LN 子集。

**读法**：

* analyser 的两个估算器（Sunny/Daniel）最强：`sunny_star` / `daniel_star` captured 47.2% / 47.3%，
  比 `msd_overall`（40.7%）高约 6.5 个百分点，比 `ett_068`（44.6%）高约 2.6 个百分点。
* `ett_068` > `ett_0723` > `ett_075`，即**旧版 MinaCalc 的序在这批数据上略好**；
  三者高度相关，差异不大（0.4459 / 0.4012 / 0.3746），不要过度解读版本效应。
* `od` 是 0 信号：OD 不在 MSD 里（§2.1），但在这个 within-player 序里也没有可测的正向作用；
  `notes_per_sec` 有 24.7%，说明密度本身是一个不弱的难度代理。
* `ln_ratio` 的 lw 是负的：LN 比例高的图对这批玩家整体**更容易**（或 LN 比例与慢速/低密度共线），
  所以它只能当内容控制，不能当难度。
* `star` 只有 24.8%，是 osu! 的 star rating；它和 MSD/analyser 的差距说明"官方 star"不是
  一个很强的 within-player 难度序。

### 4.3 LN 子集：sunnywin_ln_star 到底有没有用？

两个 LN 子集：

* **Mix/LN**：`patt_mode_tag in {Mix,LN}`，4,862 张白名单图；step1 里实际被玩到的 1,395 张，
  32,125 行，1,179,698 个比较；ceiling **0.51151**。
* **ln_ratio ≥ 0.15**：10,136 张白名单图；step1 里 2,590 张，41,480 行；ceiling **0.55047**。

| 信号 | Mix/LN captured | Mix/LN lw | ln≥0.15 captured | ln≥0.15 lw |
|---|---|---|---|---|
| `sunny_star` | **0.3973** | **0.5415** | **0.4684** | **0.5121** |
| `daniel_star` | 0.3628 | 0.5224 | 0.4260 | 0.4946 |
| `interlude_star` | 0.3375 | 0.5073 | 0.3967 | 0.4759 |
| `ett_075_overall` | 0.3291 | 0.5005 | 0.3897 | 0.4697 |
| `msd_overall` | 0.3273 | 0.4957 | 0.3880 | 0.4652 |
| `sunnywin_ln_star` | 0.3078 | 0.4792 | 0.3742 | 0.4533 |
| `ln_ratio` | 0.0219 | 0.1238 | 0.0281 | 0.1120 |

**结论**：LN 专项 **没有**在 LN 图上赢过通用 `sunny_star`；它和 `msd_overall`/`ett_075` 同档，
略低于 `daniel_star`/`interlude_star`。可能的原因：

1. Mix 图里 RC 段占了不少权重，LN-only rating 忽略了 RC 部分；
2. `sunnywin_ln_star` 只算 LN 段，对"LN 段很难但整体不难"的图会高估；
3. LN 子集的 ceiling 更低（0.511/0.550 vs 0.563），captured 本身跨子集不可比，只能在子集内比。

> 注意：128 张纯 LN 图在 step1 里被玩到的更少，单独评测的 pair 数不够；上表是 Mix/LN 和
> `ln_ratio≥0.15` 的折中。要严格回答"纯 LN 上谁更强"，需要更大样本或把 RC/LN 分开建模。

### 4.4 RC 子集：Azusa / Roxy 的 native coverage

`ln_ratio ≤ 0.18` 且 `azusa_star`/`roxy_star` 非空：4,501 张 step1 谱面，70,906 行；
ceiling **0.56851**。

| 信号 | captured | lw $\rho$ | macro $\rho$ | 备注 |
|---|---|---|---|---|
| `sunny_star` | **0.5034** | 0.5219 | 0.5193 | 全量最强 |
| `daniel_star` | 0.4840 | 0.5196 | 0.5141 | |
| `msd_overall` | 0.4094 | 0.4450 | 0.4331 | |
| `roxy_star` | 0.4066 | **0.5243** | **0.5199** | lw 高于 MSD，captured 略低 |
| `azusa_star` | 0.3891 | **0.5283** | **0.5251** | 同上 |
| `ett_075_overall` | 0.3561 | 0.4230 | 0.4105 | |
| `star` | 0.3385 | 0.3955 | 0.3900 | |

**这是两个指标分歧最明显的地方**：Roxy/Azusa 的逐玩家 lw 比 MSD 高约 0.08，但 captured 低约
0.003–0.020。解释：Roxy/Azusa 在"平均每一对"上排得更准（Spearman 每对等权），但 MSD 在
高置信/极端 pair 上更好（NLL 更看重这些）。如果系统的用途是"提示最难的图"，MSD 的 captured
更相关；如果是"平均排序一致性"，Roxy/Azusa 更好。**两个都报，并说明用途。**

---

## 5. 横向比较：谁更好？

在同一 pair 集、同一 ceiling（0.56288）下，full-coverage 的排序是：

~~~text
daniel_star    0.4728
sunny_star     0.4721
ett_068        0.4459
interlude      0.4099
msd_overall    0.4067
ett_0723       0.4012
ett_075        0.3746
msd_jackspeed  0.3711
star           0.2479
notes_per_sec  0.2466
msd_technical  0.2232
msd_stream     0.1639
n_notes        0.0498
od / ln_ratio / sunnywin_ln_star  0
~~~

**不要过度排名**：

* 前 3 的 captured CI 有重叠（daniel [0.424,0.516]、sunny [0.416,0.524]、ett_068 [0.406,0.479]）；
  lw $\rho$ 的 CI 把 daniel/sunny（约 [0.488,0.527]）与 ett_068（[0.444,0.474]）分开，
  但 daniel 和 sunny 之间分不开。要正式比较两个系统，用 **paired $\Delta$NLL**（同一 cluster
  bootstrap 下逐簇求差），不要用两个独立 CI 是否重叠来判断。
* 0.63–0.66 的 NLL 只差 0.03 nats，captured 把它放大到 0–47%；captured 的绝对值依赖 ceiling，
  跨数据集/子集不可比。
* 所有外部信号的 $\rho(d,b)$ 都只有 0.22–0.26（衰减校正后约 0.26–0.28）。这不是"谁抄谁"的问题，
  而是 **ACC 难度和 pattern 难度是两个 construct**：MSD/star/analyser 测的是谱面 pattern，
  行为 $b$ 测的是 top-1000 玩家在 ACC 上的相对表现（受 OD、长按、准度稳定性影响）。

---

## 6. 与已有 step2 实验的关系

| 实验 | 口径 | 指标 | 结论 |
|---|---|---|---|
| `step2/minacalc`（L1） | legacy clean，play 级 | RMSE | MSD 特征在 bm free 时无稳定增益（1.5477 vs bias 1.5271）；冷启动 1.7137 vs 1.8707 |
| `step2/minacalc_L2` | legacy clean，play 级 | RMSE | gap/IRT 形式没有超过 bias（1.5810 vs 1.5271）；identity 链接退化成 bias |
| `step2/map_analyser` | legacy clean，play 级 | RMSE | 7-PC 交互 1.4874（−2.6% vs bias）；冷启动 1.5987（回收 79%） |
| **本文件的 rank-NLL** | **pairbest，pair 级** | **NLL / captured** | **MSD 40.7%；Sunny/Daniel 47%；LN 专项 30.8%（不赢通用）** |

> 上表前三行的 1.4–1.9 是 **legacy clean** 的 RMSE；pairbest 的对应基线是
> bias **0.7443** / mf_dot **0.7015**（`experiments/20261020_193449_pairbest_1k`）。
> 不要跨表比较。

它们回答不同问题：

* RMSE 问"能不能预测 loss 的数值"；rank-NLL 问"能不能预测同一玩家对两张图的相对表现"。
* RMSE 可以靠 `bm`（每图自由截距）拿到；rank-NLL 评价的是 **序**，`bm` 就是 ceiling 的一部分。
* 所以"MSD 特征没有改进 RMSE"和"MSD 的序 captured 40.7%"**不矛盾**：前者是 bm free 的回归，
  后者是 bm 被替换成 MSD 序后的 pairwise 似然。

---

## 7. 注意事项（按重要性）

1. **选择偏差**：$ACC\in(0.90,0.998]$，$n_{plays}\ge2$，top-1000。最难（≤0.90）和最容易（>0.998）
   的图被删掉，而难度序在两端最有区分度。本文件的 40–47% 只在这个切片上成立。
2. **ACC 难度 ≠ pattern 难度**：$\rho(d,b)\approx0.24$ 是构造差异，不是"外部系统都错了"。
   要评估 pattern 难度，需要另一套行为标签（例如 pass/fail、或按 OD 分层）。
3. **ceiling 是 noisy 的**：split-half 的 $b$ 相关 0.582（Spearman–Brown 校正 0.736）；
   cross-fit 覆盖 93.6% 的 held-out pair。captured 是**下界**，系统间排序仍有效。
4. **pair-random 划分**：外部信号是 content-only，不依赖 step1 的 split，所以没有训练泄漏；
   但 ceiling 的 $b$ 是在所有玩家上 cross-fit 的。若要声称"难度泛化到新图"，用 beatmap-disjoint
   重跑（信号不变，ceiling 会变）。
5. **单图覆盖**：`ett_*`/`sunny_*`/`daniel_*`/`interlude_*` 全覆盖 21,949；`azusa_*`/`roxy_*` 只有
   RC 的 60.5%（39.5% 是 `UnsupportedLN`），所以 §4.4 是它们的 native 子集，不能和 §4.2 直接比。
6. **LN 专项**：`sunnywin_ln_star` 全量的 0 不是"简单"，是"没有 LN 段"；必须用 LN 子集。
   Mix/LN 的 1,395 张 step1 谱面里，纯 LN 只有 128 张白名单图的一小部分，样本不足以单独下结论。
7. **ties**：ACC 由离散 hit 数算出；runner 默认把 tie 拆成 0.5/0.5，并报 `n_ties`。
8. **不要只报 raw NLL**：0.63–0.66 的差异在 $\log2$ 附近被压缩；必须同时报 `captured` 和
   `ceiling_nll`。

---

## 8. 复现

~~~bash
# 1) 信号表（已有 data/processed/rank_signals_1k*.parquet 可跳过）
python - <<'PY'
import pandas as pd
m = pd.read_parquet("data/processed/map_analyser_1k.parquet")
msd = pd.read_parquet("data/processed/minacalc_msd_1k.parquet")
meta = pd.read_csv("data/processed/beatmap_meta_1k.csv")
cols = ["beatmap_id","sunnywin_ln_star","sunny_star","daniel_star","azusa_star","roxy_star",
        "interlude_star","ett_075_overall","ett_0723_overall","ett_068_overall","ln_ratio",
        "patt_ln_percent","od","notes_per_sec","n_notes","patt_mode_tag"]
sig = (m[cols]
       .merge(msd[["beatmap_id","msd_overall","msd_stream","msd_jackspeed","msd_technical"]],
              on="beatmap_id", how="left")
       .merge(meta[["beatmap_id","star","count_total"]], on="beatmap_id", how="left"))
sig.to_parquet("data/processed/rank_signals_1k.parquet", index=False)
PY

# 2) MinaCalc（固定内容信号：--split all，不需要 train/test）
#    --folds 1 是 in-sample ceiling（ceiling 偏低、captured 也偏低），快速对比时才用；正式数字用 --folds 2。
python scripts/step2_rank_eval.py \
    --in data/processed/step1_1k_pairbest.parquet \
    --signal-table data/processed/rank_signals_1k.parquet \
    --signals msd_overall --split all --min-l 10 --folds 2 --bootstrap 300 \
    --out results/rank_eval_1k/msd.json

# 3) analyser 全量（ceiling 复用 cache）
python scripts/step2_rank_eval.py \
    --in data/processed/step1_1k_pairbest.parquet \
    --signal-table data/processed/rank_signals_1k.parquet \
    --signals star,ett_075_overall,ett_0723_overall,ett_068_overall,sunny_star,daniel_star,interlude_star,msd_overall,msd_jackspeed,msd_technical,msd_stream,notes_per_sec,n_notes,od,ln_ratio,sunnywin_ln_star \
    --split all --ceiling-fit all --common-maps --min-l 10 --folds 2 --bootstrap 300 \
    --out results/rank_eval_1k/full.json

# 4) LN / RC 子集（信号表先按 patt_mode_tag / ln_ratio 过滤，见 §4.3/§4.4）
python scripts/step2_rank_eval.py ... --signal-table data/processed/rank_signals_1k_mix.parquet ...
python scripts/step2_rank_eval.py ... --signal-table data/processed/rank_signals_1k_ln.parquet ...
python scripts/step2_rank_eval.py ... --signal-table data/processed/rank_signals_1k_rc.parquet ...
~~~

产物：

| 文件 | 内容 |
|---|---|
| `results/rank_eval_1k/msd.json` | MinaCalc `msd_overall` |
| `results/rank_eval_1k/full.json` | 全部 full-coverage 信号 |
| `results/rank_eval_1k/ln_mix.json` | Mix/LN 子集 |
| `results/rank_eval_1k/ln15.json` | `ln_ratio≥0.15` 子集 |
| `results/rank_eval_1k/rc.json` | RC 子集（Azusa/Roxy native） |
| `results/rank_eval_1k/summary.csv` | 上述 JSON 的合并表（group / signal / captured / lw / CIs） |
| `results/rank_eval_1k/full_pp.csv` / `msd_pp.csv` | 逐玩家 rho（full / MinaCalc） |
| `data/processed/rank_eval_cache/ceiling_*.npz` | cross-fitted ceiling 缓存（按 map 集 + seed 键） |
