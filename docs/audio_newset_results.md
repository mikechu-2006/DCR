# Experiment 2 结果：整 beatmapset 留出后的 audio 迁移

> **状态：已执行（2026-11-24）。结论：audio 的增益能迁移到“新谱面集”；不是主要来自同 set 泄漏。**
>
> 数据：1k 的 6,819 图 / 109,569 对；audio-on = published 4,037 + local audio 2,782；
> audio-off = legacy。P4 双重冷启动，5 个 split seed；模型 seed 0。
> 评价：historical-best rank-NLL 的 paired ΔNLL；player_id_base cluster bootstrap。

---

## 0. 一句话结论

2a（标准 P4 上的新 set 条件子集）和 2b（整 beatmapset 留出的 P4）都显示：
**audio-on 相对 audio-off 的增益在新 set 上没有消失。** 2b 的 external 线性
ΔNLL = −0.01091（5/5），external MLP = −0.00581（5/5）；跨 set 严格子集
Δ = −0.01138 / −0.00631，仍然 5/5。external 线性几乎不掉，MLP 缩小约 20–25%，
但仍稳定为负。

因此：**现有 audio 增益不是“同曲多难度”泄漏的假象；audio 编码了可跨歌曲迁移的
难度线索。** 但由于 randommerge/full 的音频补不出来，这个结论只覆盖 1k 的
6,819 图，不能外推到 randommerge/full。

---

## 1. 2a：标准 P4 上的新 set 条件子集

对每个 test player，只保留“两张测试图的 beatmapset_id 都未出现在 6×6 训练图”
的比较。覆盖率按 seed 为 18.2% / 18.0% / 17.1% / 10.8% / 15.4%（mean 15.9%）；
按图算，44–50% 的 test 图属于“set 未在训练侧出现”。

| 方法 | 全 test ΔNLL | 新 set 子集 ΔNLL | 新 set seeds 为负 | 新 set CI<0 |
|---|---:|---:|---:|---:|
| CM3P + external 线性 | −0.01153 | **−0.01753 ± 0.00746** | 5/5 | 5/5 |
| CM3P + external MLP | −0.00770 | **−0.01128 ± 0.00610** | 5/5 | 5/5 |
| CM3P 线性 | −0.01188 | −0.01154 ± 0.00873 | 4/5 | 4/5 |
| CM3P MLP | −0.02091 | −0.01804 ± 0.01015 | 5/5 | 5/5 |

读法：

* 强方法（external 线性/MLP）在新 set 子集上的增益**没有缩小，反而更大**；
  线性/MLP 基本持平。没有“新 set 上增益消失”的模式。
* new_cross（两张图来自不同 set）与 new_set 几乎一样；new_same 只占 0.017%
  的比较，噪声大，不单独解读。
* 这是标准 P4 上的条件子集，不是重新划分；它的作用是“在已有模型上先看方向”，
  真正的干净检验是 2b。

---

## 2. 2b：整 beatmapset 留出的 P4

### 2.1 划分

* 3,806 个 set；玩家侧沿用 P4 的 60/40 活跃度分层；
* set 侧按 set popularity（distinct player clusters）分 5 层，层内按图数贪心 60/40；
* 5 个 seed 的分配图数都是 **4,091 / 2,728（60.0% / 40.0%）**；
* 实际有 train/test 行的图：3,350–3,536 / 2,018–2,176；train 行 35,886–38,606，
  test 行 18,022–19,791；group_overlap = 0；
* 所有 test set 都未在 6×6 训练侧出现。

### 2.2 主表（官方 strict-pair 评测）

| 方法 | legacy NLL | audio NLL | ΔNLL（audio−legacy） | seeds 变好 | ΔPearson | ΔSpearman |
|---|---:|---:|---:|---:|---:|---:|
| **CM3P + external 线性** | 0.59376 ± 0.00564 | 0.58285 ± 0.00787 | **−0.01091 ± 0.00366** | **5/5** | +0.0583 | +0.0566 |
| **CM3P + external MLP** | 0.58592 ± 0.00649 | 0.58012 ± 0.00730 | **−0.00581 ± 0.00425** | **5/5** | +0.0337 | +0.0437 |
| CM3P 线性 | 0.62912 ± 0.00468 | 0.62110 ± 0.00732 | −0.00801 ± 0.00309 | 5/5 | +0.0480 | +0.0329 |
| CM3P MLP | 0.63476 ± 0.00662 | 0.62112 ± 0.00582 | −0.01364 ± 0.00261 | 5/5 | +0.0603 | +0.0582 |

固定信号 Sunny / Daniel / star 两臂逐位相同（最大绝对差 = 0），确认同一 split、
同一 fixed 表。ceiling NLL = 0.55947 ± 0.01429；captured 0.75–0.85，没有标准
randommerge 那种 >1 的欠拟合。

逐 seed Δ（external）：

* 线性：−0.0116 / −0.0134 / −0.0075 / −0.0152 / −0.0068；
* MLP：−0.0035 / −0.0133 / −0.0048 / −0.0044 / −0.0029。

### 2.3 跨 set 严格子集

set-P4 里所有 test set 都未在训练侧出现；new_cross 再要求两张测试图来自
不同 set，占 99.6% 的比较：

| 方法 | ΔNLL（all） | ΔNLL（cross-set） | cross-set seeds 为负 | cross-set CI<0 |
|---|---:|---:|---:|---:|
| CM3P + external 线性 | −0.01144 | **−0.01138** | 5/5 | 5/5 |
| CM3P + external MLP | −0.00633 | **−0.00631** | 5/5 | 5/5 |
| CM3P 线性 | −0.00779 | −0.00771 | 5/5 | 5/5 |
| CM3P MLP | −0.01341 | −0.01333 | 5/5 | 5/5 |

（这一小节用 all-pair 的 paired 脚本；与 2.2 的官方 strict-pair 数只差
~0.0005，结论不变。）

### 2.4 与标准 P4 的对比

| 方法 | 标准 P4 Δ | set-P4 Δ | 保留比例 |
|---|---:|---:|---:|
| CM3P + external 线性 | −0.01153 | −0.01144 | ~99% |
| CM3P + external MLP | −0.00770 | −0.00633 | ~82% |

* external 线性几乎不变；external MLP 缩小约 18–25%，但仍 5/5 为负、超过
  −0.004 的判据。
* 绝对 NLL 不可跨 split 比较；只比 Δ。
* 2a 的 new-set 条件子集比 2b 的 set-P4 更大（−0.0175 vs −0.0114），
  这是子集构成差异，不是“set-P4 更差”；以 2b 为准。

---

## 3. 判据

计划判据：Δ_setP4 ≤ −0.004、≥4/5 seed 为负、≥4/5 CI 排除 0、跨 set ≤ −0.003。

* external 线性：−0.01091（5/5，CI<0），跨 set −0.01138（5/5，CI<0）→ **通过**；
* external MLP：−0.00581（5/5，CI<0），跨 set −0.00631（5/5，CI<0）→ **通过**；
* 四个方法全部 5/5 为负；没有方法反转；
* 相关性同向：ΔPearson +0.0337…+0.0603，ΔSpearman +0.0329…+0.0582。

**结论：audio 增益能迁移到整 set 留出的新谱面集；没有证据表明它主要是同 set
泄漏。**

---

## 4. 边界与限制

1. **只覆盖 1k 的 6,819 图**：randommerge/full 的音频补不出来（网络/磁盘限制），
   所以不能把这里的“通过”外推到 randommerge/full。Experiment 2 的结论只到 1k。
2. **编码器预训练暴露**：2b 排除的是“监督训练侧的同 set 图”，不能排除 CM3P 的
   audio 编码器在预训练语料里见过这些歌。结论应写成“超出监督同 set 的迁移”。
3. **只有 1k top-1000 切片**：ACC 截断、n_plays≥2、P4 协议不变；不外推到全玩家。
4. **set-P4 的谱面构成变了**：分配图 60/40，实际有行图约 63/37；test pair 集与
   标准 P4 不同，绝对 NLL 不可比，只看 Δ。
5. **2a 是条件子集**：模型仍在标准 P4 上训练，只是评测时筛 pair；它不能替代 2b。
6. **captured 不使用**：set-P4 的 captured 0.75–0.85，但不同 split 的 captured
   不可直接比；只看 NLL 与 paired Δ。
7. **音频来源混合**：published 4,037 + local 2,782；local 子集更难/更热门。
   本轮不拆分来源；通过的是“混合表”的结论。

---

## 5. 对项目的影响

* **1k 上 audio-on 是真实增益**：在新 set 上 external 线性 −0.011，MLP −0.006；
  如果 1k 是目标图集，audio-on 值得保留。
* **不能据此给 randommerge/full 补音频**：数据条件不允许；这两个图集仍用
  audio-off legacy。
* **不需要再做“同曲泄漏”专项**：2b 已经回答了；下一步如果继续 audio，应该
  转向“哪类歌/哪类段落贡献了增益”，而不是再证明一次迁移。
* **Experiment 3（D2V 残差）不受影响**：D2V 是谱面 piano roll，不是真实音频；
  两条线独立。

---

## 6. 产物

| 路径 | 内容 |
|---|---|
| results/audio_newset/subset_s{1..5}.json | 2a 逐 seed、逐 subset、逐方法的 Δ/CI/覆盖 |
| results/audio_newset/subset_summary.csv | 2a 逐 seed 明细 |
| results/audio_newset/subset_means.csv | 2a 5-seed 均值 |
| data/processed/step1_1k_pairbest_p4sets_s{1..5}.parquet | 整 set 留出 P4 划分 |
| results/audio_newset/p4sets/audio_vs_legacy.md / .csv | 2b 官方 strict-pair 主表 |
| results/audio_newset/p4sets/{audio,legacy}/p4/s*/ | 两臂信号与 rank_eval |
| results/audio_newset/p4sets/subset/subset_means.csv | 2b all / new_cross 的 Δ/CI |
| scripts/eval_newset_audio.py | 2a/2b 子集评测 |
| scripts/step1_p4_split_sets.py | 整 set P4 划分 |
| docs/audio_newset_results.md | 本报告 |
