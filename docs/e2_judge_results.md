# E2 结果：六档判定比例弱监督（H4 seed 1，固定预算）

## 结论

**H2 不成立。** 在固定 300k pair views、关闭早停、同一 H4 test 集与同一
strict-pair 协议下：

- rank-only: 0.586204
- + 六档比例 (sqrt_inv): 0.585076，Δ = **−0.001128**
- + 打乱比例 (placebo): 0.583627，Δ = **−0.002577**
- + miss-only: 0.586127，Δ = −0.000077
- + 六档比例 (unweighted): 0.585515，Δ = −0.000688

主臂相对 rank 的改善只有 −0.00113，远低于预注册的 −0.003；更关键的是
**placebo 比主臂更好**（低 0.00145）。因此不能把这点差异解释为“判定分布
提供了 chart 相关信号”。

首轮 pilot 的 judge−rank = −0.010774 是**预算伪影**：rank/shuffled/single
在 7k steps 早停，judge 跑到 11.3k steps。固定预算后差异消失。这是本轮最
重要的负结果。

## 主表

| arm | test NLL | captured | Δ vs rank | 说明 |
|---|---:|---:|---:|---|
| rank | 0.586204 | 0.5952 | — | 同 enc/同预算 |
| judge | 0.585076 | 0.6015 | −0.001128 | 主臂 |
| judge_shuffled | **0.583627** | **0.6096** | **−0.002577** | placebo 最好 |
| single_grade_miss | 0.586127 | 0.5957 | −0.000077 | 单档位 |
| judge_none | 0.585515 | 0.5991 | −0.000688 | 无 class weight |

所有臂：300,019 pair views、bad_batches = 0、导出 last.pt。

## 配对 ΔNLL

已取回 judge−rank 的配对 bootstrap：

| 对比 | Δ | player CI | chart CI |
|---|---:|---|---|
| judge − rank | −0.001128 | [−0.001606, −0.000645] | [−0.003891, +0.001413] |

其余比较的完整 paired bootstrap 在 HPC 登录节点断线时未取回；但 H2 的门槛是
Δ ≤ −0.003，**点估计 −0.001128 已经不过门**，且 placebo 点估计更低。
因此结论不依赖缺失的 CI。

## 事件级操纵检查

| arm | event NLL | baseline NLL | Δ | MAX ρ | miss ρ |
|---|---:|---:|---:|---:|---:|
| rank | 0.822122 | 0.822122 | 0.000000 | 0.616 | 0.317 |
| judge (sqrt_inv) | 0.950418 | 0.822122 | **+0.128295** | 0.702 | 0.465 |
| judge_shuffled | 0.947787 | 0.822122 | +0.125665 | 0.702 | 0.419 |
| single_grade_miss | 1.540341 | 0.822122 | +0.718219 | 0.264 | 0.469 |
| judge_none | 0.817331 | 0.822122 | −0.004791 | 0.725 | 0.455 |

- 预注册的**未加权 event NLL** 对 sqrt_inv 主臂失败（+0.128）；原因是
  sqrt_inv 权重把大量概率从 MAX 移向稀有档位：预测 MAX 均值 0.477，
  观测 0.647。
- judge_none 的 event NLL 方向正确（−0.0048）但幅度远低于
  5%/0.02 的门槛；MAX/miss Spearman 明显高于 baseline。
- 即使忽略 event NLL 的校准问题，rank-NLL 本身也没有超过 placebo，
  所以不能声称“判定头学到了可迁移的 chart 信号”。

## 解释

1. **早停是 pilot 的主要混杂**：固定预算后，主臂相对 rank 只剩 −0.0011，
   与 H4 between-seed 噪声同量级。
2. **placebo 最好**：说明多任务/正则化可能带来一点收益，但不是“六档判定
   分布的形状”带来的；按预注册，A2 与 A1 同幅度改善时应改写为
   “多任务/容量有效”，而不是 H2。
3. **class weight 有校准代价**：sqrt_inv 让稀有档位主导，未加权 event NLL
   变差；judge_none 校准更好，但 rank 收益仍接近 0。
4. **单档位 miss 没有收益**：A3 ≈ A0，进一步说明不是“miss 率”单独驱动。
5. **不能据此说“判定分布完全无用”**：本轮只测 seed 1、H4、固定 λ=1.0、
   24 窗口；但按当前主判据，没有可声称的 rank 改善。

## 限制

- 只有 H4 seed 1；seeds 2–5 未跑。
- H4 test 高星偏置；P4-full 未复核。
- λ 只测 1.0；没有 λ 网格或 focal/温度校准。
- 事件级 QA 的阈值是预注册的；本轮 sqrt_inv 主臂未过，不能事后改口径。
- 逐音定位仍不可识别；本文不发布难点热图。

## 复现

~~~bash
/opt/slurm/bin/sbatch --array=0-4 scripts/slurm/behavior_e2_fixed.slurm
/opt/slurm/bin/sbatch --dependency=afterok:<array_job> \
  scripts/slurm/behavior_e2_fixed_eval.slurm
~~~

产物：

- runs/behavior/e2_fixed/<arm>/s1/metrics.json
- runs/behavior/e2_fixed/<arm>/s1/d_test.parquet
- runs/behavior/e2_fixed/<arm>/s1/events_qa.json
- runs/behavior/e2_fixed/eval/eval_*.json
- runs/behavior/e2_fixed/eval/summary.md（若已生成）
