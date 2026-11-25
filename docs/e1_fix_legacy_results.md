# E1_fix：旧 seed-1 的确定性重打分

> 2026-11-24。状态：**legacy checkpoint 复评**，不是 clean seed-1，也不是 5-seed 结论。
> 本文只回答“旧报告里的数字有多少是评测/导出造成的”，并给出 H4 s1 strict-pair
> 重算结果。clean seed-1 和 F4 窗口对照仍未完成。

## 0. 一句话

旧报告的 **NLL 排序和 H1a 方向**在确定性重打分后仍成立，而且更干净；但两个
重要表述被推翻：

1. unfrozen 不再是与 content-table **打平**，而是 **−0.00419**（player CI 排除 0，
   chart CI 跨 0）；
2. random 不再明显差于 content-table，而是 **+0.00055**（两个 CI 都含 0）。

H1b 仍差一点到预注册的 −0.005：**−0.00474**，但两个 95% CI 首次都排除 0；
按原判据仍是 **inconclusive**。

## 1. 为什么旧数字会偏

旧 `encode_charts` 只把 encoder 设为 eval，难度头 dropout 仍开启；函数返回时
还把 encoder 切回 train。最终导出又从内存中的 `best_state`（活引用）恢复，
而 `save_every` 会用同名 `checkpoint.pt` 覆盖 best。结果是：

* 同一 checkpoint 的分数不稳定；
* 旧 `d_test` 来自**最后权重**，而 checkpoint 文件停在较早的 best 保存点；
* 旧报告把 `best_val` 写在 metrics 里，但导出的不一定是 best。

E1_fix 的 F1/F2/F3 修了：

* eval 时 encoder + head 都 eval，退出恢复原状态；
* `best.pt` / `last.pt` 分离，CPU clone 快照；
* BT loss 改 FP32 BCEWithLogits；
* strict-pair / `player_id_base` / 保留 bootstrap multiplicity 的配对统计。

## 2. 复评对象

| 项 | 值 |
|---|---|
| 划分 | H4 player-split seed 1 |
| test | 136,670 行 / 1,362 图 / 3,517,702 strict pairs |
| checkpoint | `runs/behavior/e1/<arm>/s1/checkpoint.pt` |
| 性质 | legacy / last-saved checkpoint；不是旧报告内存中的最后权重 |
| 评测 | `max_windows=24`，全窗口覆盖 99.36% 帧；repeat eval 两次，max abs diff = 0 |
| bootstrap | 2000 |
| ceiling | 0.513480（与旧报告相同） |

## 3. 确定性主表

| arm | checkpoint step | test NLL | captured | 旧 d_test NLL |
|---|---:|---:|---:|---:|
| **unfrozen top-4** | 19,000 | **0.571274** | **0.6783** | 0.575264 |
| random init | 19,000 | **0.576013** | 0.6520 | 0.579521 |
| frozen same-head | 17,000 | **0.592361** | 0.5610 | 0.595696 |
| D2V linear content | — | 0.575467 | 0.6550 | 0.575467 |

确定性重打分把三臂都改善了约 0.003–0.004，但排序不变：
unfrozen < random < frozen。

## 4. 配对 Δ（strict pairs，bootstrap 2000）

负值 = A 更好；CI 是 2.5/97.5 百分位。

| 对比 | Δ | player CI | chart CI |
|---|---:|---|---|
| unfrozen − frozen same-head | **−0.021088** | [−0.022725, −0.019473] | [−0.026251, −0.015789] |
| unfrozen − random | **−0.004739** | [−0.005561, −0.003937] | [−0.008331, −0.001266] |
| unfrozen − D2V linear | **−0.004193** | [−0.005368, −0.002995] | [−0.009627, +0.001189] |
| random − D2V linear | **+0.000547** | [−0.000837, +0.001873] | [−0.005137, +0.006265] |

## 5. 与旧报告相比，哪些话要改

| 旧报告 | E1_fix |
|---|---|
| unfrozen − content = −0.00019，CI 跨 0，“打平” | **−0.00419**；player CI 排除 0，chart CI 跨 0；“点估计更好，但 chart 侧仍不稳健” |
| random − content = +0.00406，player CI 排除 0 | **+0.00055**；两个 CI 都含 0；“与 content 打平” |
| H1b chart CI 含 0 | **两个 CI 都排除 0**，但 Δ=−0.00474 仍高于 −0.005 门槛；按判据 inconclusive |
| H1a −0.02043 | **−0.02109**；方向和显著性不变，略强 |
| H1a “同预算” | 仍不成立：legacy checkpoint 是 step 19000/17000，不是旧报告的最终权重 |

## 6. 不能说的话

* 不能把这当作 **clean seed 1**：它是 legacy checkpoint 的确定性复评。
* 不能把 H1b 写成“支持”：点估计 −0.00474 没有达到预注册的 −0.005。
* 不能把 unfrozen 写成“全面超过 content-table”：chart CI 仍含 0。
* 不能做 5/5 或跨 seed 声明：只有 H4 s1。
* 不能把旧 `d_test` 和这次 `d_test` 混在同一张主表里；必须标 legacy/deterministic。

## 7. 复现

~~~bash
# 1) 确定性重打分（三臂）
sbatch scripts/slurm/behavior_e1_fix_legacy_score.slurm

# 2) strict eval + 配对
python scripts/e1_fix_eval.py \
  --runs runs/behavior/e1_fix/legacy/s1 \
  --split-table data/processed/step1_full_holdout_s1.parquet \
  --baseline-dir results/behavior/baselines/h4_s1 \
  --out-dir runs/behavior/e1_fix/legacy/s1/eval \
  --cache-dir data/processed/rank_eval_cache \
  --bootstrap 2000 --seed 0
~~~

产物：

~~~text
runs/behavior/e1_fix/legacy/s1/<arm>/score/
  d_test.parquet
  d_val.parquet
  score_metrics.json
  window_coverage.parquet

runs/behavior/e1_fix/legacy/s1/eval/
  eval_<arm>.json
  paired_*.json
  summary.json / summary.md
~~~

## 8. 下一步

1. 等 F4 的 K4 vs 全窗口 2×2：判断同头冻结的弱势有多少来自窗口/聚合。
2. 跑 clean seed 1（固定 1M、last）确认 legacy checkpoint 的结论不是
   “best checkpoint 选择”造成的。
3. 只有 clean seed 1 通过 Gate 才补 seeds 2–5。
