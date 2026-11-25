# E1_fix clean seed 1：固定 1M/last 结果

> 2026-11-24。状态：**clean seed 1 结果**，不是 5-seed 最终结论。
> 主表固定 1M pair views、不早停、导出 `last.pt`；评测使用 F1/F2/F3 修复后的
> 确定性 code（eval 时 encoder+head 都 eval，BT loss 为 FP32 BCEWithLogits）。

## 0. 一句话

clean seed 1 中：

* **H1a 成立且更强**：unfrozen − frozen = **−0.02427**，两个 CI 排除 0。
* **H1b 仍不成立**：unfrozen − random = **−0.003185**，两个 CI 排除 0，但
  远未到预注册的 −0.005；按判据 inconclusive。
* unfrozen 比 content-table 低 **−0.00441**；random 与 content-table
  **−0.00123**，CI 含 0，即打平。
* F4 已排除“K=4 窗口/聚合”作为 H1a 的解释。

## 1. 主表（H4 test，1,362 图，3,517,702 strict pairs）

| arm | clean NLL | captured | legacy NLL |
|---|---:|---:|---:|
| **unfrozen top-4** | **0.571056** | **0.6795** | 0.571274 |
| random init | **0.574241** | 0.6618 | 0.576013 |
| frozen same-head | **0.595329** | 0.5444 | 0.592361 |
| D2V linear content（参照） | 0.575467 | 0.6550 | — |

三臂均：

* 1,000,021 pair views；
* `bad_batches = 0`；
* `last.pt` 导出，`max_windows=24`，repeat eval 两次一致。

## 2. 配对 Δ（bootstrap 2000）

| 对比 | Δ | player CI | chart CI |
|---|---:|---|---|
| unfrozen − frozen same-head | **−0.024273** | [−0.025809, −0.022783] | [−0.030429, −0.018437] |
| unfrozen − random | **−0.003185** | [−0.003964, −0.002430] | [−0.006023, −0.000311] |
| unfrozen − D2V linear | **−0.004411** | [−0.006152, −0.002649] | [−0.010671, +0.001906] |
| random − D2V linear | **−0.001226** | [−0.003014, +0.000601] | [−0.007388, +0.005138] |

## 3. 与 legacy checkpoint 复评的差别

| 项 | legacy | clean |
|---|---:|---:|
| H1a | −0.02109 | **−0.02427** |
| H1b | −0.00474 | **−0.00319** |
| unfrozen − content | −0.00419 | **−0.00441** |
| random − content | +0.00055 | **−0.00123** |
| frozen NLL | 0.59236 | 0.59533 |

clean 的 frozen 臂比 legacy 差 0.003，原因不是 code，而是协议：clean 固定
1M/last，legacy 复评用的是较早的 best checkpoint（step 17000）。H1a 在两种
口径下都成立；H1b 在 clean 口径下更小，说明“预训练边际”不稳健。

## 4. F4 的作用

同头、冻结 D2V、1M pair views 的 2×2：

| 条件 | 训练窗口 | 评测窗口 | NLL |
|---|---|---|---:|
| c0 | K4 | cap24 | 0.59620 |
| c1 | all | all | 0.59757 |
| c2 | K4 | all | **0.59444** |
| c3 | all | cap24 | 0.60039 |

c1−c0=+0.00137，c1−unfrozen=+0.02629。全窗口训练没有追平 unfrozen；
窗口/聚合不是 H1a 的解释。细节见 [e1_fix_f4_results.md](e1_fix_f4_results.md)。

## 5. 适用边界

* 只有 H4 s1；不能做 5/5 或跨 seed 稳定性声明。
* 主表是 clean fixed 1M/last；best-checkpoint 结果作为 legacy/敏感性分析。
* content-table baseline 是不同头/聚合，不是同头冻结。
* H4 test 高星偏置（star 均值 4.71）。
* F4 是 head-only 对照，不是 encoder 训练。
* E2 未启动。

## 6. 产物

~~~text
runs/behavior/e1_fix/clean/s1/
  unfrozen_k4/{last.pt,d_test.parquet,metrics.json}
  random/{last.pt,d_test.parquet,metrics.json}
  frozen_samehead/{last.pt,d_test.parquet,metrics.json}
  eval/{eval_*.json,paired_*.json,summary.md}
~~~

复现：

~~~bash
sbatch scripts/slurm/behavior_e1_fix_train.slurm
python scripts/e1_fix_eval.py \
  --runs runs/behavior/e1_fix/clean/s1 \
  --split-table data/processed/step1_full_holdout_s1.parquet \
  --baseline-dir results/behavior/baselines/h4_s1 \
  --out-dir runs/behavior/e1_fix/clean/s1/eval \
  --cache-dir data/processed/rank_eval_cache \
  --bootstrap 2000 --seed 0
~~~
