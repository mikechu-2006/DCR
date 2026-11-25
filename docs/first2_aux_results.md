# Experiment 1 结果：first2 辅助训练

> **状态：已执行（2026-11-24）。结论：没有可用的稳定提升；不进入 10k/full。**
>
> 问题：用“每对前两条记录的最好成绩”作辅助标签，让同一个谱面难度分同时拟合
> 历史最佳序与 first2 序，能否改善正式测试的历史最佳 rank-NLL？
>
> 数据：1k（6,828 图）与 randommerge（20,388 图），P4 双重冷启动，5 个 split seed。
> 模型：linear_extra，legacy CM3P 512d + msd_overall,sunny_star（1k）/
> msd_overall,sunny_star,log_n（randommerge）；λ_ridge=1e-3；模型 seed 0。
> 评价：同一 test pair 上的 historical-best rank-NLL；paired ΔNLL 按
> player_id_base cluster bootstrap 400 次。

---

## 0. 一句话结论

**first2 作为“早期练习信号”没有可迁移信息。** 1k P4 上 A1 有一个很小的
−0.00120 nats 增益（5/5 为负），但它来自 n_plays=2 的 pair——这些 pair 的
first2 标签等于历史最佳，辅助项只是把主损失再算一遍、给这些 pair 额外加权。
只加 n_plays>2 的 pair（A2）增益消失；打乱 first2 标签（A6）反而变差。
randommerge P4 5 seeds 的 A1 均值只有 **+0.00002**，没有复现 1k 的小增益。

因此：**不继续 10k/full，不把 first2 标签用于生产。** 如果对“给 n_plays=2 的
pair 加权”本身感兴趣，应该直接做样本加权实验，而不是 first2 辅助损失。

---

## 1. 实验臂

| 臂 | 辅助标签 | 辅助损失范围 | 说明 |
|---|---|---|---|
| A0 | 无 | — | baseline |
| A1 | first2 | 全部 train pair | 主实验 |
| A2 | first2 | 仅 n_plays > 2 | 只加“有信息”的 pair |
| A3 | first2 | first2 ACC ∈ (0.90, 0.998] | 目标支撑匹配 |
| A6 | first2 打乱 | 全部 | 控制：破坏 first2 的配对 |

λ_aux 在训练侧 20% player clusters 的 historical-best 验证 NLL 上选；
grid = 0, 0.05, 0.1, 0.2, 0.5, 1.0。A0 固定 λ=0。

---

## 2. 1k P4：5 seeds

| 臂 | mean test NLL | ΔNLL vs A0（mean ± std） | seed 为负 | 95% CI 排除 0 | Pearson | Spearman | 逐玩家 lw | 逐玩家 macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.58479 | — | — | — | 0.5374 | 0.5094 | 0.5340 | 0.5251 |
| **A1** | **0.58351** | **−0.00120 ± 0.00043** | **5/5** | **5/5** | 0.5383 | 0.5087 | 0.5366 | 0.5283 |
| A2 | 0.58465 | −0.00003 ± 0.00065 | 2/5 | 1/5 | 0.5362 | 0.5074 | 0.5350 | 0.5263 |
| A3 | 0.58352 | −0.00115 ± 0.00050 | 5/5 | 5/5 | 0.5363 | 0.5076 | 0.5369 | 0.5286 |
| A6 | 0.58563 | +0.00062 ± 0.00047 | 1/5 | 1/5（负向） | 0.5377 | 0.5088 | 0.5303 | 0.5203 |

读法：

* A1/A3 的 NLL 小且一致地变好，但幅度只有 −0.0012，低于计划里 0.002 的“成功”阈值。
* 相关性没有实质提升：A1 的 Pearson +0.0009、Spearman −0.0007、逐玩家 lw +0.0026；
  这是“NLL 略好、逐玩家序略好、谱面级 Pearson/Spearman 持平/略差”的混合。
* **A2 基本为零**：只加 n_plays>2 的 pair 后，增益消失。
* **A6 变差**：打乱 first2 后，均值 +0.00062，4/5 为正。
* λ 选择：A1 = [1, 0.2, 1, 1, 1]，A3 = [1, 0.2, 1, 1, 1]；多数选在 grid 上界 1.0。

---

## 3. randommerge P4：5 seeds

| 臂 | mean test NLL | ΔNLL vs A0（mean ± std） | seed 为负 | 95% CI 排除 0 | Pearson | Spearman | 逐玩家 lw | 逐玩家 macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.58202 | — | — | — | 0.8831 | 0.8761 | 0.5611 | 0.5517 |
| A1 | 0.58204 | **+0.00002 ± 0.00017** | 1/5 | 1/5 | 0.8832 | 0.8759 | 0.5608 | 0.5510 |
| A2 | 0.58194 | −0.00007 ± 0.00012 | 2/5 | 1/5 | 0.8831 | 0.8758 | 0.5602 | 0.5499 |
| A3 | 0.58201 | −0.00001 ± 0.00033 | 1/5 | 1/5 | 0.8832 | 0.8760 | 0.5619 | 0.5526 |

读法：

* A1 均值 **+0.00002**，基本是零；5 个 seed 里 4 个为正，s3 的 −0.00021 是唯一负值。
* A2/A3 也都在 ±0.0001 内；相关性持平。
* λ 选择：A1 = [1, 0.2, 0.2, 0, 0]，A2 = [0.2, 0.05, 0.2, 0.2, 0]，
  A3 = [1, 0.5, 1, 0, 0]；s4/s5 多数选 0，模型与 A0 相同。
* **1k 的 −0.0012 没有在 randommerge 上复现。**

---

## 4. 机制：增益来自 n_plays=2 的重复标签

三个观察拼在一起：

1. **A2 消失**：只对 n_plays>2 加辅助损失，1k 与 randommerge 的增益都消失
   （1k −0.00003；randommerge −0.00007）。
2. **A6 变差**：把 first2 标签在 pair 间打乱后，1k 的 A1 从 −0.00120 变成 +0.00062。
   打乱破坏的是 n_plays=2 时“first2 = 历史最佳”的等式；如果 first2 本身有信息，
   打乱应该只是去掉一部分信号，而不是把方向反过来。
3. **A3 ≈ A1**：只去掉 first2 ACC ≤0.90 的 0.65% 极端标签，结果几乎不变；
   说明不是极端早期成绩在起作用。

因此 A1/A3 的小增益来自：n_plays=2 的 pair 上，辅助似然与主似然完全相同，
L = L_hist + λ L_first2 退化为 (1+λ) L_hist，相当于给这些 pair 额外加权。
这是**标签重复/样本加权效应**，不是“前两次尝试提供了谱面难度信息”。

1k 的 n_plays=2 占 53.6%，randommerge 占 49.7%；1k 的加权效应更大，
randommerge 上 λ 又常在验证里选 0，所以没有复现。这个差异本身也说明效应不稳定。

---

## 5. 结论与建议

1. **主假设不成立**：没有证据表明 first2 辅助训练能稳定改善历史最佳 rank-NLL。
2. **不进入 10k/full**：1k 的 −0.0012 低于 0.002 成功阈值，且机制是重复标签；
   randommerge 5 seeds 均值 +0.00002。铺 10k/full 的 160 个 job 不值得。
3. **不用于生产**：first2 标签不进入模型；正式评价口径不变。
4. **如果对加权效应感兴趣**：直接对 n_plays=2 的 pair 做样本权重（或分层损失），
   这是一个更便宜、更可解释的实验；不是 first2 标签的功劳。
5. **A2/A6 的结论比 A1 的 −0.0012 更重要**：它们说明“早期信号”本身无效。

---

## 6. 局限

* 只跑了 **P4**，没有跑 P2；只跑了 **1k + randommerge**，没有跑 10k/full。
  但机制是标签层面的（n_plays=2 重复），不是划分层面的。
* λ_aux grid 上限 1.0；1k 的 A1/A3 多数选在上界，说明验证侧还在继续改善。
  更大的 λ 可能放大“重复标签加权”的效应，但不会让 first2 的早期信号变得有用。
* L-BFGS-B 多次报 TOTAL NO. OF ITERATIONS REACHED LIMIT；A0 与现有
  randommerge od_l1 baseline 一致（0.58202 vs 0.58202），说明主链路没有跑偏。
* randommerge 的 captured >1（ceiling 欠拟合），本报告不使用 captured；
  只看 NLL 与 paired ΔNLL。
* A4（random-2）没有跑；A2 + A6 已经足以定位机制。若以后要区分“任何随机两条”
  与“前两条”，再补 A4。
* 5 个 split seed 不是独立样本；std 是划分方差，不是训练方差。

---

## 7. 复现

关键脚本：

* scripts/step1_first2_labels.py：从 step0 + step1 生成 first2 标签；
* scripts/step2_cm3p_rank_aux.py：共享 d 的双似然 linear_extra；
* scripts/slurm/first2_1k_p4_{train,eval}.slurm：1k P4 5 seeds；
* scripts/slurm/first2_rm5_{train,eval}.slurm：randommerge P4 5 seeds；
* scripts/slurm/first2_a6_{train,eval}.slurm：1k A6 打乱控制。

示例（1k P4 s1 A1）：

~~~bash
python scripts/step2_cm3p_rank_aux.py \
  --in data/processed/step1_1k_pairbest_p4_s1.parquet \
  --first2 data/processed/step1_1k_pairbest_first2.parquet \
  --content-path data/processed/chart_content_cm3p_local.parquet \
  --signal-table data/processed/rank_signals_1k.parquet \
  --extra-signals msd_overall,sunny_star \
  --aux-scope all --aux-weight-grid 0,0.05,0.1,0.2,0.5,1.0 --val-frac 0.2 \
  --name first2_a1 \
  --out-signals results/first2_aux/1k_p4/s1/a1.parquet \
  --out-json results/first2_aux/1k_p4/s1/a1.json
~~~

评测：

~~~bash
python scripts/paired_delta_nll.py \
  --in data/processed/step1_1k_pairbest_p4_s1.parquet \
  --signal-table results/first2_aux/1k_p4/s1/signals.parquet \
  --signals first2_a0,first2_a1,first2_a2,first2_a3 \
  --pairs first2_a1:first2_a0,first2_a2:first2_a0,first2_a3:first2_a0 \
  --split test --bootstrap 400 --out results/first2_aux/1k_p4/s1/paired_delta.json
~~~

---

## 8. 产物

| 路径 | 内容 |
|---|---|
| results/first2_aux/summary.csv | 1k/randommerge × 5 seeds × 臂的 NLL/Δ/CI/相关性/λ |
| results/first2_aux/1k_p4/s{1..5}/ | 信号、rank_eval、paired_delta、A6 |
| results/first2_aux/rm_p4/s{1..5}/ | randommerge 信号、rank_eval、paired_delta |
| results/first2_aux/pilot/ | randommerge P4 s1 的 pilot（与 5-seed s1 一致） |
| docs/first2_aux_results.md | 本报告 |
