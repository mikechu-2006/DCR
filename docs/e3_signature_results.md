# E3 结果：参考玩家签名蒸馏（H4 seed 1）

## 结论

**H3 不成立，而且方向相反。** 在同一个 H4 test 集、同一训练预算、同一
strict-pair 评测下：

- 只蒸馏**平均教师难度**（B1, mean）显著优于 rank-only：Δ = **−0.013217**。
- 蒸馏**整组 64 个参考玩家签名**（B2, full）虽然也优于 rank-only
  （Δ = −0.006119），但比 mean **差 +0.007098**，两个 CI 都排除 0。
- 打乱签名（B3, shuffled）比 rank-only 还差 +0.003751，说明多任务/容量
  本身没有带来增益。

因此这条线目前的可靠结论是：**平均教师难度是有用的辅助信号；保留玩家之间
的分歧（K=64 签名）在全局 rank-NLL 上没有帮助，反而有害。** 不能据此声称
“保留了可迁移的玩家分歧”。

## 主表

| arm | test NLL | captured | n_strict |
|---|---:|---:|---:|
| rank (B0) | 0.587878 | 0.5859 | 3,517,702 |
| **mean (B1)** | **0.574661** | **0.6595** | 3,517,702 |
| full (B2) | 0.581759 | 0.6200 | 3,517,702 |
| shuffled (B3) | 0.591629 | 0.5650 | 3,517,702 |

## 配对 ΔNLL（bootstrap 2000）

| 对比 | Δ | player CI | chart CI |
|---|---:|---|---|
| full − mean | **+0.007098** | [+0.005879, +0.008287] | [+0.003082, +0.011674] |
| full − rank | −0.006119 | [−0.006737, −0.005525] | [−0.008557, −0.003800] |
| mean − rank | **−0.013217** | [−0.014509, −0.011856] | [−0.018256, −0.008626] |
| shuffled − rank | +0.003751 | [+0.002718, +0.004813] | [+0.000208, +0.007235] |

## 教师 QA

| 项 | 值 |
|---|---:|
| teacher held-out rank-NLL (bias) | 0.616778 |
| teacher held-out rank-NLL (mf_dot) | 0.596716 |
| Δ (mf_dot − bias) | **−0.020061** |
| 参考玩家数 K | 64 |
| 签名维度 | 64 |
| 签名 std 均值 | 0.065928 |
| 与 teacher_mean 的最大 |corr| | 0.541 |

教师 held-out 是 **vocab-preserving 的 play 留出**，不是 cold-player 检验：
mf_dot 的 vocab 只从 train 建，cold player 没有 P[u]，按玩家留出会得到 0 个
test 行。这个限制在 [E2/E3 计划](plans/2026-11-24_e2_e3_plan.md) §5 问题 10
有记录。

## 解释

1. **mean 臂的增益是真实的**：它不是多任务容量，因为 shuffled 臂在同头、
   同 λ、同预算下比 rank 更差；mean 的 −0.0132 两个 CI 都排除 0。
2. **full 臂的 64 维目标太强/太噪**：每个图有 64 个辅助目标，λ=1.0 时
   MSE 可能压过 rank 信号；val NLL 也显示 full 比 mean 差（0.60155 vs
   0.59997），不是 test 偶然。
3. **不能推断“玩家分歧完全无用”**：本轮只测了 K=64、λ=1.0、seed 1；
   K=16 或更小 λ 可能不同。但主假设 B2 < B1 未成立，不能把 full 的
   rank 改善写成“分歧有用”。
4. **教师本身有信号**：held-out NLL 比 bias 低 0.0201；full 失败更像是
   学生侧的目标/容量问题，而不是教师完全无效。

## 限制

- 只有 H4 seed 1；seeds 2–5 未跑。
- H4 test 高星偏置；P4-full 未复核。
- K=64、λ=1.0 是单点，未做 K=16/128 或 λ 网格。
- 教师 QA 不是 cold-player；签名可能含有教师 in-sample 结构。
- 结论限定为“全局 rank-NLL”，不涉及逐音定位或可命名技能维度。

## 复现

~~~bash
# 教师 + 签名
/opt/slurm/bin/sbatch scripts/slurm/behavior_e3_teacher.slurm
# 四臂
/opt/slurm/bin/sbatch --array=0-3 --dependency=afterok:<teacher_job> \
  scripts/slurm/behavior_e3_train.slurm
# 评测
/opt/slurm/bin/sbatch --dependency=afterok:<train_job> \
  scripts/slurm/behavior_e3_eval.slurm
~~~

产物：

- results/behavior/e3/teacher_s1/signature.parquet
- results/behavior/e3/teacher_s1/signature.json
- runs/behavior/e3/<arm>/s1/metrics.json
- runs/behavior/e3/eval/summary.md
- runs/behavior/e3/eval/summary.json
