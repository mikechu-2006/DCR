# E1_fix F4：K4 vs 全窗口冻结对照

> 2026-11-24。状态：**head-only 对照结果**。不是 clean seed-1，也不是 5-seed。
> 目的：判断 frozen same-head 的弱势有多少来自 K=4 随机窗口/聚合，而不是
> 行为梯度本身。

## 0. 一句话

**窗口/聚合不是 H1a 的主要解释。** 全窗口训练没有让冻结头追平 unfrozen；
相反，`all/all` 比 `K4/cap24` 高 **+0.00137**，`all/cap24` 比 `K4/cap24`
高 **+0.00419**。冻结头仍比 unfrozen 差约 **0.026**。

## 1. 设计

同一份冻结 D2V 逐窗口特征（20,018 图，缓存一次），同一个 ChartHead、
同一个 pair sampler、同样的 1M pair views；只改训练/评测窗口策略：

| 编号 | 训练窗口 | 评测窗口 | 作用 |
|---|---|---|---|
| c0 | K=4 随机 | 前 24 窗 | 当前 frozen same-head 口径的重实现 |
| c1 | 全部 | 全部 | 最强冻结读出（同头、全窗口） |
| c2 | K=4 随机 | 全部 | 分离评测截断 |
| c3 | 全部 | 前 24 窗 | 分离训练窗口覆盖 |

## 2. 结果（H4 test，strict pairs）

| 条件 | test NLL | captured |
|---|---:|---:|
| c0 K4 / cap24 | 0.59620 | 0.5396 |
| c1 all / all | 0.59757 | 0.5320 |
| c2 K4 / all | **0.59444** | **0.5494** |
| c3 all / cap24 | 0.60039 | 0.5163 |
| unfrozen top-4（参照） | **0.571274** | **0.6783** |
| frozen same-head（legacy checkpoint） | 0.592361 | 0.5610 |
| frozen same-head（clean 1M/last） | 0.595329 | 0.5444 |
| D2V linear content（参照） | 0.575467 | 0.6550 |

## 3. 分解

| 对比 | Δ NLL | player CI | chart CI |
|---|---:|---|---|
| c1 − c0 | **+0.001370** | [+0.000646, +0.002035] | [−0.003792, +0.006271] |
| c2 − c0 | **−0.001760** | 待补 | 待补 |
| c3 − c0 | **+0.004190** | 待补 | 待补 |
| c1 − unfrozen | **+0.026296** | [+0.024557, +0.028115] | [+0.020122, +0.033058] |
| c0 − unfrozen | **+0.024924** | [+0.023195, +0.026627] | [+0.019009, +0.031027] |

表中 CI 为 bootstrap 200；c2/c3 的 CI 只作描述，主判定看 c0/c1。

按计划 §6.4 的规则：`|Δ_window| = |c0−c1| = 0.00137 ≤ 0.003`，窗口策略
不是主要解释；H1a 不能归因于“冻结臂只看了 4 个窗口”。

## 4. 解释

1. **训练时看全部窗口没有帮助**：c3 比 c0 差 0.00419，c1 比 c0 差 0.00137。
   在 1M pair views 的预算下，冻结特征 + 全窗口更像“更多噪声/过拟合”，
   不是更强的读出。
2. **评测放开到全窗口有很小的正收益**：c2 比 c0 好 0.00176；说明 cap24 的
   0.64% 帧缺失有一点影响，但远小于 0.02 的 H1a 差距。
3. **冻结头仍远落后于 unfrozen**：c1 比 unfrozen 高 0.0263，c0 高 0.0249。
   所以 H1a 不是“窗口采样/聚合”造成的。
4. c0=0.59620 接近旧 frozen 报告的 0.59570，但比 deterministic frozen
   checkpoint（0.59236）差 0.0038；这是因为本对照是 head-only 1M views，
   原 frozen checkpoint 是 step 17000 / 876k views，预算和采样不同。
   c0/c1 的相对比较仍有效，c0 不应替代 frozen 主结果。

## 5. 不能说的话

* 不能把 F4 当作 encoder 训练对照：encoder 完全冻结，只训练头。
* 不能把 c0 当作原 frozen same-head 的 bitwise 复现：采样 RNG、checkpoint
  和预算都不同。
* 不能因为 c1 仍差 0.026 就断言“行为监督一定改善表征”：F4 只排除了窗口/
  聚合这一种解释；clean seed 1 仍未完成。
* 只有 H4 s1；不能做 5/5。

## 6. 产物与复现

~~~text
runs/behavior/e1_fix/control/s1/
  features.npz                 # 20,018 图逐窗口冻结特征
  c0_d_test.parquet ... c3_d_test.parquet
  c0_head.pt ... c3_head.pt
  metrics.json
  eval/c0.json ... c3.json

runs/behavior/e1_fix/control/s1/eval/  # c0/c1 vs unfrozen 的 paired Δ（200 bootstrap）
~~~

命令：

~~~bash
# CPU 即可：特征已缓存，只训练头
python scripts/e1_frozen_window_control.py \
  --split-table data/processed/step1_full_holdout_s1.parquet \
  --osu-dir data/osu_cache \
  --ckpt data2vec_main/runs/main_holdout/checkpoints/last.pt \
  --feature-cache runs/behavior/e1_fix/control/s1/features.npz \
  --pair-views 1000000 --seed 0 --device cpu \
  --conditions c0,c1,c2,c3 \
  --out-dir runs/behavior/e1_fix/control/s1
~~~
