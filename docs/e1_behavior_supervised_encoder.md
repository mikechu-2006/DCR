# 问题 1：让同一玩家的胜负直接训练谱面编码器

> 2026-11-24。状态：**E1 seed 1 报告**，不是 5-seed 最终结论。
> 本文回答最初的问题 1，并记录训练目标、三臂设计与单 seed 结果。
> 方法与全量执行计划见 [behavior_supervision_plan.md](plans/2026-11-24_behavior_supervision_plan.md)。
>
> **2026-11-24 E1_fix 修订**：旧 §5/§6 的 NLL 与 CI 来自旧的 train-mode/dropout
> 导出，且导出的是最后权重而不是 checkpoint。F1–F3 修复后的确定性复评见
> [e1_fix_legacy_results.md](e1_fix_legacy_results.md)；本文 §5/§6 已按
> deterministic legacy checkpoint 重写。F4 窗口对照已完成（见 §6.2.1）；
> clean seed 1 三臂已完成（固定 1M/last），本文 §5/§6 以 clean 为主表；
> 独立摘要见 [e1_fix_clean_results.md](e1_fix_clean_results.md)。

## 0. 一句话回答

**是：解冻 data2vec 的上层并 finetune。** 但监督信号不是“每张图的 ACC 标签”，
而是**同一玩家两张图的相对胜负**；训练目标是

\[
\boxed{
\min_{\theta,\phi}\;
-\frac{1}{|\mathcal{P}|}
\sum_{(u,i,j)\in\mathcal{P}}
\left[
y_{uij}\log \sigma(d_j-d_i)
+(1-y_{uij})\log(1-\sigma(d_j-d_i))
\right]
}
\]

其中 \(d_m=h_\phi(\mathrm{Norm}(\mathrm{pool}(E_\theta(x_m))))\)，
\(y_{uij}=1[\mathrm{ACC}_{ui}>\mathrm{ACC}_{uj}]\)。
这是 within-player 的 Bradley–Terry / RankNet 负对数似然；玩家能力在
同一人的两张图比较里被消掉，所以**没有玩家 embedding**，也不是 ACC 回归。

seed 1 结果（H4 test，1M pair views）：

| arm | clean test NLL | captured | legacy NLL |
|---|---:|---:|---:|
| unfrozen top-4 | **0.571056** | **0.6795** | 0.571274 |
| random init | 0.574241 | 0.6618 | 0.576013 |
| frozen same-head | 0.595329 | 0.5444 | 0.592361 |
| D2V linear content（参照） | 0.575467 | 0.6550 | — |

结论：

* **H1a（行为监督改善表征）**：clean unfrozen 比同头冻结低 **0.02427**，
  两个 CI 排除 0；seed 1 支持，5/5 待补。
* **H1b（遮蔽预训练有用）**：clean unfrozen 比 random 低 **0.003185**，两个 CI
  都排除 0，但远未到预注册的 \(-0.005\)；seed 1 按判据 inconclusive。
* **相对最强冻结基线**：unfrozen 与 content-table D2V linear（0.575467）的
  Δ 是 \(-0.00441\)；player CI 排除 0，chart CI 跨 0。random 与 content 的
  Δ 是 \(-0.00123\)，两个 CI 都含 0。

---

## 1. 原问题

旗舰原话的核心是：

> 让同一玩家的胜负直接训练谱面编码器；输入沿用已有的四列 onset/hold 序列；
> 两张谱共享同一个编码器，各自输出难度 \(d\)，用
> \(P(\text{玩家在 A 的 ACC 更高})=\sigma(d_B-d_A)\) 训练；保留完整谱面监督，
> 让梯度经过窗口聚合传回读音符的网络；再比较冻结 data2vec、解冻上层、
> 同架构随机初始化三组。

这里先把两个容易混淆的词钉死：

| 说法 | 含义 |
|---|---|
| “胜负” | 不是游戏胜负，而是**同一个玩家段**在两张图上的 ACC 谁更高 |
| “难度 \(d\)” | 一张图的**全局**难度标量；不是“这个玩家觉得多难” |
| “finetune” | 用行为 pair 的梯度更新编码器的可训练参数，不只是训练读出 |
| “冻结” | 编码器权重固定，只训练 \(d\) 头 |

---

## 2. 训练目标

### 2.1 标签

对玩家段 \(u\) 在图 \(m\) 上的最佳 play，记

\[
a_{um}=1-\exp(\mathrm{loss}_{um})
\]

即 ACC（\(\mathrm{loss}=\log(1-\mathrm{ACC})\)）。对同一玩家的两张图 \(i,j\)：

\[
y_{uij}=
\begin{cases}
1 & a_{ui}>a_{uj}\\
0.5 & a_{ui}=a_{uj}\\
0 & a_{ui}<a_{uj}
\end{cases}
\]

ties 用 0.5，与评测的 `tie="half"` 一致。

### 2.2 模型输出

\[
d_m=h_\phi\!\left(
\mathrm{Norm}\!\left(
\frac{\sum_w v_w\cdot \frac{1}{v_w}\sum_t e_{m,w,t}}{\sum_w v_w}
\right)
\right)
\]

其中 \(e_{m,w,t}\) 是编码器在第 \(w\) 个 16 s 窗口、第 \(t\) 帧的输出，
\(v_w\) 是该窗口有效帧数；\(\mathrm{Norm}\) 是 L2 归一化；\(h_\phi\) 是
LayerNorm → Linear(512,256) → GELU → Dropout → Linear(256,1)。
\(d_m\) 越高越难。

训练时每张图随机采 \(K=4\) 个 16 s 窗口；评测时用全部不重叠窗口。
梯度从 pair loss 回到 \(d\)，再回到窗口聚合，再回到每个窗口的编码器输出。

### 2.3 损失

\[
q_{ij}=\sigma(d_j-d_i)
\]

\[
\ell_{uij}
=
-\left[
y_{uij}\log q_{ij}
+(1-y_{uij})\log(1-q_{ij})
\right]
\]

训练目标：

\[
\mathcal{L}
=
\frac{1}{|\mathcal{P}|}
\sum_{(u,i,j)\in\mathcal{P}}\ell_{uij}
\]

\(\mathcal{P}\) 是**同一玩家段内**的两两图对。全量 full 的 within-player
pair 约 2.55 亿；当前 run 采样 **1,000,000 pair views**，是对全量目标的
随机估计。

### 2.4 一个数字例子

玩家 u：A 的 ACC 0.97，B 的 ACC 0.94，所以 A 赢（\(y=1\)）。

* 模型 \(d_A=1.2,\ d_B=1.5\)：\(q=\sigma(0.3)=0.574\)，
  \(\ell=-\log 0.574=0.555\)；
* 模型反过来 \(d_A=2.0,\ d_B=1.0\)：\(q=\sigma(-1.0)=0.269\)，
  \(\ell=1.31\)。

梯度会把 \(d_B\) 往上推、\(d_A\) 往下推；编码器要在 B 的音符里找
“让它更难”的特征，在 A 里找“让它更简单”的特征。

### 2.5 为什么玩家项消失了

若玩家表现为

\[
s_{um}=\theta_u-d_m+\epsilon_{um}
\]

则同一玩家的两张图相减：

\[
s_{ui}-s_{uj}
=(d_j-d_i)+(\epsilon_{ui}-\epsilon_{uj})
\]

玩家能力 \(\theta_u\) 被消掉。因此模型不需要玩家 embedding；它学的是
**全局难度序**；玩家之间的分歧不会被建模，只会被 BT 似然平均掉
（这是第 3 条要处理的）。

### 2.6 它不是什么

* 不是对 ACC 做 MSE 回归：绝对 ACC 混入了玩家水平；
* 不是 per-player 条件预测：没有玩家输入；
* 不是逐事件判定比例（那是第 2 条）；
* 不是玩家分歧/参考玩家签名（那是第 3 条）。

### 2.7 正则与训练噪声

目标之外还有 AdamW weight decay（encoder 0.01、head 0.05）、dropout 0.1、
梯度裁剪 1.0、chart-disjoint val 上的 rank-NLL 早停，以及非有限 loss/grad
的 batch 跳过（见 §5.3）。这些是训练稳定性和模型选择，不是目标函数本身。

---

## 3. 模型与三臂

### 3.1 编码器

D2V audio-style：

* 输入：4 列 onset + 4 列 hold，100 Hz × 8 通道；
* conv：8→256→256→512，stride 2，输出 50 Hz；
* Transformer：12 层 / 512d / 8 heads / FFN 2048；
* 16 s 窗口 = 800 个 50 Hz token；
* 图向量：窗口内有效帧均值 → 跨窗口按有效帧数加权 → L2；
* d 头：512→256→1。

### 3.2 三臂

| 项 | 冻结 | 解冻 top-4 | 随机初始化 |
|---|---|---|---|
| 初始化 | D2V checkpoint | 同一 checkpoint | 随机 |
| conv | 冻结 | 冻结 | 训练 |
| blocks 0–7 | 冻结 | 冻结 | 训练 |
| blocks 8–11 + final norm | 冻结 | **训练** | 训练 |
| d 头 | 训练 | 训练 | 训练 |
| 可训练参数 | 0 + head | **12,610,560** | **38,824,960** |
| 输入/窗口/聚合/损失/pair | 同 | 同 | 同 |

“解冻 top-4”= 最后 4 个 Transformer block + final LayerNorm；conv 与底部 8 层冻结。

### 3.3 梯度路径

~~~text
chart A ──► windows ──► E_theta ──► pool ──► Norm ──► h_phi ──► d_A
chart B ──► windows ──► E_theta ──► pool ──► Norm ──► h_phi ──► d_B
                                      │
                       y = 1[ACC_A > ACC_B]
                       loss = -[y log σ(d_B-d_A) + (1-y) log(1-σ(d_B-d_A))]
                                      │
                     gradient → h_phi → pool → every window → E_theta
~~~

同一个编码器处理 A、B 两次（Siamese），不是把两张图一起喂进去。

---

## 4. 协议

| 项 | 值 |
|---|---|
| 划分 | H4：固定 1,365 张 D2V-unseen chart 作 test charts；玩家按 `player_id_base` 60/40 |
| 训练 | train players × 非 holdout charts；content 过滤后 845,952 行 / 16,701 图 |
| val | train players × 10% chart-disjoint；89,895 行 / 1,870 图 |
| test | test players × holdout charts；136,670 行 / 1,362 图 |
| 主指标 | strict-pair rank-NLL，`--ceiling-fit all` |
| test pairs | 3,517,702 |
| ceiling | 0.513480 |
| 训练预算 | 1,000,000 pair views |
| batch | 4 players × 6 charts（去重后 ≤24 张图；≤96 窗口） |
| 窗口 | 训练 K=4 个随机 16 s；评测全部窗口 |
| LR | encoder \(5\times10^{-5}\)，head \(5\times10^{-4}\) |
| 精度 | bf16（A800） |
| 早停 | val rank-NLL，patience 5，eval every 2000 steps |
| seed | 0（模型）；H4 split seed 1 |

注意：冻结臂在 val 早停时停在 928,250 pair views / 18,000 steps；
unfrozen 与 random 都跑满约 1M views。因此冻结臂的训练预算略少。

---

## 5. 结果

### 5.1 主表（H4 test，seed 1）

| arm | 训练预算 | clean NLL | captured | legacy NLL |
|---|---|---:|---:|---:|
| **unfrozen top-4** | 1,000,021 views | **0.571056** | **0.6795** | 0.571274 |
| random init | 1,000,021 views | 0.574241 | 0.6618 | 0.576013 |
| frozen same-head | 1,000,021 views | 0.595329 | 0.5444 | 0.592361 |

clean run 固定 1M pair views、不早停、导出 `last.pt`；三臂 `bad_batches=0`，
说明 F2 的稳定 BT loss 生效。legacy checkpoint 的复评见
[e1_fix_legacy_results.md](e1_fix_legacy_results.md)。

### 5.2 固定参照

| 方法 | test NLL |
|---|---:|
| ceiling（自由 per-chart b） | 0.51348 |
| D2V linear（content table，全窗口） | 0.57547 |
| D2V MLP（content table，全窗口） | 0.57699 |
| CM3P + external MLP | 0.58034 |
| Sunny | 0.60390 |
| MinaCalc | 0.62449 |
| star | 0.63810 |

### 5.3 配对 ΔNLL

同一 test 对、同一 ceiling；负值 = A 更好。

| 对比 | Δ | player 95% CI | chart 95% CI |
|---|---:|---|---|
| unfrozen − frozen same-head | **−0.024273** | [−0.025809, −0.022783] | [−0.030429, −0.018437] |
| unfrozen − random | **−0.003185** | [−0.003964, −0.002430] | [−0.006023, −0.000311] |
| unfrozen − D2V linear content | **−0.004411** | [−0.006152, −0.002649] | [−0.010671, +0.001906] |
| random − D2V linear content | **−0.001226** | [−0.003014, +0.000601] | [−0.007388, +0.005138] |

（clean seed 1；bootstrap 2000；strict pairs；player 按 `player_id_base` 聚类并
保留重复次数；chart 用 `mult[a]*mult[b]` 权重。）

### 5.4 bad batches

clean run 中非有限 loss/grad 的 batch：

* unfrozen：**0**；
* random：**0**；
* frozen：**0**。

F2 把 loss 改成 FP32 `BCEWithLogits` 后，1M pair views × 3 臂没有任何坏
batch。旧报告的 110/90/8 是旧 BT loss 时代的计数，不能归因于 D2V 前端。

---

## 6. 解读

### 6.1 H1a：行为监督改善表征

clean seed 1（固定 1M/last）中，unfrozen 比同头冻结低 **0.02427 nats**，
player CI [−0.02581, −0.02278]，chart CI [−0.03043, −0.01844]，两个 CI 都
排除 0。在这个**同头、同 pair、同 K=4 训练窗口、同 1M pair views** 的控制里，
行为梯度让表征更适配行为任务。按预注册判据（Δ ≤ −0.003，5/5，CI 排除 0），
seed 1 满足效应量与 CI，但只有 1 个 split seed，不能写 5/5。

F4 已排除“冻结臂只看了 K=4 窗口”的解释；见 §6.2.1。

### 6.2 unfrozen 不再与最强冻结基线打平

clean seed 1 中，unfrozen 与 content-table D2V linear（0.57547）的 Δ 是
**−0.00441**，player CI [−0.00615, −0.00265] 排除 0，chart CI
[−0.01067, +0.00191] 跨 0。因此：

* 点估计上，unfrozen **超过了**“全窗口冻结内容 + 线性读出”这个强基线；
* 但 chart 侧仍不稳健，不能写“全面超过”；
* 旧报告的 −0.00019/CI 跨 0 是 train-mode/dropout 与“导出最后权重而非 checkpoint”
  共同造成的，不是真实打平。

同头冻结与 content-table 冻结的差距（0.59533 vs 0.57547）仍部分来自
聚合/读出：同头冻结用 K=4 随机窗口 + MLP 头，content-table 用全窗口均值 +
线性头。

### 6.2.1 F4：窗口/聚合不是 H1a 的解释

F4 用同一份冻结 D2V 逐窗口特征、同一个 ChartHead、同一个 pair sampler 和
1M pair views，只改窗口策略：

| 条件 | 训练窗口 | 评测窗口 | test NLL |
|---|---|---|---:|
| c0 | K4 | cap24 | 0.59620 |
| c1 | all | all | 0.59757 |
| c2 | K4 | all | **0.59444** |
| c3 | all | cap24 | 0.60039 |

* c1 − c0 = **+0.00137**：全窗口训练 + 全窗口评测反而略差；
  player CI [0.00065, 0.00204] 排除 0，chart CI 跨 0。
* c2 − c0 = −0.00176：只放开评测窗口略好，但远小于 0.021。
* c3 − c0 = +0.00419：只放开训练窗口明显更差。
* c1 比 unfrozen 仍高 **+0.02629**，两个 CI 排除 0。

因此 **H1a 不能归因于“冻结臂只看了 K=4 个窗口”**；全窗口训练没有追平
unfrozen。F4 是 head-only、c0 也不是原 frozen run 的 bitwise 复现（预算/RNG
不同），但 c0/c1 的相对比较足以排除窗口解释。

### 6.3 H1b：遮蔽预训练只提供很小的边际

clean seed 1 中，unfrozen 比 random 低 **0.003185**：

* 方向对；
* player CI [−0.00396, −0.00243] 排除 0；
* chart CI [−0.00602, −0.00031] 也排除 0；
* 但点估计远未达到预注册的 −0.005。

因此 seed 1 **不支持**“遮蔽预训练提供可迁移先验”的强结论；更准确的说法是
“有很小的正向边际，按预注册判据 inconclusive”。clean random 的 NLL 是
0.57424，甚至略低于 content-table 的 0.57547；旧报告“random 明显更差”
不成立。legacy 复评的 −0.00474 也没有跨过 −0.005。

### 6.4 random 与 content-table 基本打平

clean random 比 D2V linear content 低 **−0.00123**，player CI
[−0.00301, +0.00060]、chart CI [−0.00739, +0.00514] 都包含 0。点估计略好，
但按 CI 仍是打平。所以：

* 冻结的 D2V 内容向量仍是一个很强的 chart-only 特征；
* 从零做行为 finetune 与它打平，既没有明显超过，也不像旧报告那样明显更差；
* behavior finetune 相对 content-table 的优势目前只在 unfrozen 臂上，
  且 chart 侧不稳健。

### 6.5 适用边界

* 只有 seed 1；不能做 5/5 或跨 seed 稳定性声明；
* 主表是 clean seed 1：固定 1M pair views、不早停、导出 `last.pt`；
  legacy checkpoint 复评作为敏感性分析，见 [e1_fix_legacy_results.md](e1_fix_legacy_results.md)；
* clean run 的 bad batches 为 0；旧报告的 110/90/8 属于旧 BT loss 时代；
* content-table baseline 是不同头/不同聚合，不是纯“同头冻结”；
* H4 test 高星偏置（star 均值 4.71），不能外推到低星；
* `max_windows=24` 覆盖 99.36% 帧，长图仍有 0.64% 帧未覆盖；F4 报全窗口。

---

## 7. 对第 2 条（E2）的含义

E2 的核心问题仍是“判定比例是否比 rank-NLL-only 提供额外信息”，但当前
代码结构决定了一个坑：**不要**把 `--freeze all` 当作 E2 主实验。

* `head`（难度 d）和 `event_head`（六档 logits）是彼此独立的；
* `judge_loss` 的梯度只经过 `event_head` 和 encoder；
* 如果 encoder 完全冻结，梯度不会进入 `head`，A1 与 A0 的 d 在结构上相同。

因此 E2 若启动，主实验应使用：

1. `--freeze topk --unfreeze-blocks 4`（现有 E2 Slurm 已是这个），或
2. 冻结 encoder + 在两个头之前加一个**共享可训练 adapter**，让 judge loss
   能进入共享参数。

`frozen all` 只适合检查“冻结帧特征能否预测判定比例”，不能检验 H2。
A0/A1 仍必须共享同一 backbone，且本报告不启动 E2。

---

## 8. 复现

### 8.1 H4

~~~bash
python scripts/step1_holdout_split.py \
  --in data/processed/step1_full_pairbest.parquet \
  --holdout-ids data2vec_main/splits/holdout/eval_ids.txt \
  --out-prefix data/processed/step1_full_holdout \
  --seeds 1,2,3,4,5 --player-frac 0.6 --bins 5 --val-chart-frac 0.1
~~~

### 8.2 unfrozen

~~~bash
python scripts/step2_chart_encoder.py \
  --task rank --init d2v \
  --ckpt data2vec_main/runs/main_holdout/checkpoints/last.pt \
  --freeze topk --unfreeze-blocks 4 \
  --split-table data/processed/step1_full_holdout_s1.parquet \
  --osu-dir data/osu_cache \
  --window-sec 16 --frame-hz 100 --windows-per-chart 4 \
  --players-per-batch 4 --charts-per-player 6 \
  --pair-views 1000000 --eval-every 2000 --patience 5 \
  --lr-encoder 5e-5 --lr-head 5e-4 --bf16 --device cuda --seed 0 \
  --out-dir runs/behavior/e1/unfrozen_k4/s1
~~~

### 8.3 另外两臂

* random：`--init random --freeze none`；
* frozen：`--init d2v --ckpt ... --freeze all`。

### 8.4 确定性重打分与配对

~~~bash
# 1) 旧 checkpoint 确定性重打分
sbatch scripts/slurm/behavior_e1_fix_legacy_score.slurm

# 2) strict eval + 配对（bootstrap 2000）
python scripts/e1_fix_eval.py \
  --runs runs/behavior/e1_fix/legacy/s1 \
  --split-table data/processed/step1_full_holdout_s1.parquet \
  --baseline-dir results/behavior/baselines/h4_s1 \
  --out-dir runs/behavior/e1_fix/legacy/s1/eval \
  --cache-dir data/processed/rank_eval_cache \
  --bootstrap 2000 --seed 0

# 3) clean seed 1 主表
python scripts/e1_fix_eval.py \
  --runs runs/behavior/e1_fix/clean/s1 \
  --split-table data/processed/step1_full_holdout_s1.parquet \
  --baseline-dir results/behavior/baselines/h4_s1 \
  --out-dir runs/behavior/e1_fix/clean/s1/eval \
  --cache-dir data/processed/rank_eval_cache \
  --bootstrap 2000 --seed 0
~~~

### 8.5 产物

~~~text
runs/behavior/e1_fix/clean/s1/<arm>/        d_test.parquet, last.pt, metrics.json
runs/behavior/e1_fix/clean/s1/eval/         eval_*.json, paired_*.json, summary.md
runs/behavior/e1_fix/legacy/s1/<arm>/score/ d_test.parquet, score_metrics.json
runs/behavior/e1_fix/legacy/s1/eval/        eval_*.json, paired_*.json, summary.md
runs/behavior/e1/<arm>/s1/checkpoint.pt     legacy checkpoint（只读）
~~~

---

## 9. 下一步

1. **F4 已完成**：全窗口冻结头没有追平 unfrozen，窗口/聚合不是 H1a 的解释
   （c1−c0=+0.00137，c1−unfrozen=+0.02629）。
2. **clean seed 1 已完成**：三臂固定 1M/last，`bad_batches=0`；H1a −0.02427，
   H1b −0.003185。
3. **补 seeds 2–5**：H1a 需要 5/5；H1b 当前 −0.0032 远未到 −0.005，需要更多
   split seed 判断这个边际是否稳定。
4. **bad batches 已清零**：F2 的稳定 BT loss 在 clean 1M × 3 臂上 0 个坏 batch。
5. **再做 E2**：若启动，主实验用 top-4 或共享 adapter，不用 `--freeze all`；
   本报告不启动 E2。









