# Step 2 — MinaCalc MSD 作为冻结谱面特征（L1）

**状态.** 已实现（scripts/step2_minacalc_as_feature.py + 资产脚本 scripts/step0h_minacalc_msd.py）。
冒烟测试用**合成** MSD 表跑通全流程（只验证流程，数值无意义）。

**一句话口径.** 把 MinaCalc 的 MSD 向量当作**冻结的谱面特征**接进 step2 现有的低秩模型：
谱面侧不再自由学，玩家侧仍然自由学；划分、损失、指标、检查项与
[step2_prediction.md](step2_prediction.md) 完全一致，所以 --models bias 就是**同一谱面子集**上的基线。

---

## 1. 模型

    L_hat = b0 + bu[u] + bm[m] + alpha . D_std[m] + <P[u], C_unit[m]>
             \_____ 主效应 _____/   \_______ 交互 _______/

| 项 | 含义 | 开关 |
|---|---|---|
| D_std[m] | MSD 向量（标准化后，保留幅度） | 由 --msd-reduce 决定 |
| alpha | 每个技能方向一个系数，**从 0 初始化** ⇒ epoch 0 严格等于 step2/predict 的同名模型 | --msd-main |
| bm[m] | 每图自由截距 | --msd-bm msd 时去掉，截距完全由 MSD 生成（每图自由参数 = 0） |
| C_unit[m] | 行单位范数的 MSD（与 CM3P 的约定一致） | 交互项 <P[u], C[m]> |
| P[u] | 玩家潜向量，维度 = MSD 维数 | 一直可学（R1：不反解 SSR，避免泄漏） |

三个模型名（--models）：

* bias    —— b0 + bu + bm（基线；MSD 开关对它无效）
* msd_add —— 只加主效应（没有 P、没有交互）
* msd     —— 主效应 + 点积交互（--msd-main 决定要不要主效应）

## 2. 资产（step0/minacalc_msd）

    python scripts/step0h_minacalc_msd.py --tag 1k \
        --ids manifests/whitelist_1k.csv --cache data/osu_cache \
        --lib build/libminacalc.so --out data/processed/minacalc_msd_1k.parquet --jobs 8

* 只依赖 .osu（不依赖成绩）⇒ 属 step0 资产：跑一次进 cache_dir，step2 只读表。
* 后端：--lib（ctypes 直连 [kangalio/minacalc-standalone](https://github.com/kangalio/minacalc-standalone)
  的 C API）或 --calc-bin（子进程协议，见脚本 docstring）。
* 输出列：beatmap_id, msd_overall..msd_technical, calc_version, keycount, n_notes, n_holds,
  hold_ratio, rate, goal。**calc_version 是必需的**：263 与 5xx 不可比，step2 默认拒绝没有版本列
  的表（--allow-unversioned-msd 可放行）。

## 3. 运行

    python main.py -c configs/exp_minacalc_1k.yaml            # step1 → step2/minacalc_as_feature
    python main.py -c configs/exp_minacalc_1k.yaml --dry-run

关键参数：--msd-path / --msd-skills / --msd-reduce {raw,zscore,pca} / --msd-main /
--msd-bm {free,msd} / --dim（zscore 与 raw 时 --dim 必须等于维数；pca 时是主成分数）。
其余参数与 step2/predict 同名同义。

## 4. 与 CM3P 实验的可比性

* 两者的谱面子集**不同**（CM3P 覆盖约 54%，MSD 覆盖全部 4K 谱面），所以只能比"同一口径下的
  相对增益"，不要直接搬 [cm3p_step2_results.md](cm3p_step2_results.md) 的数字。
* 想严格对齐：用本脚本的 bias 作基线（它跑在被裁到有 MSD 的谱面上），
  或在 step2/predict 里用 --chart-content none --restrict-to-content on 复现同一子集。
* 指标口径沿用 step2/predict：测试集 loss 的 RMSE（未清洗标签），附 ACC/pp。

## 5. 已知坑

1. **MSD 的 goal=0.93 是 Etterna 的 score 语义**，不是 mania ACC。所以只当**相对**特征用：
   标准化只在训练谱面上拟合，绝对刻度交给 b0/bu/bm。
2. **LN/长按**：MinaCalc 的 NoteInfo 没有时长维度，长按只算头部那一个 tap。表里存了 hold_ratio，
   建议在 rice 子集上单独报一次指标。
3. **OD 不在 MSD 里**，而标签是 ACC 的函数 —— OD 应作为单独特征加进来（尚未接入）。
4. msd_overall 是 7 个方向的聚合（max(aggregate_skill, max)），默认不参与特征以免共线。
5. 版本必须锁：表里的 calc_version 会写进 summary 的 msd 段。

## 6. 下一步

* **L2**：把 bu/bm + 点积换成 gap 形式 x = a + sum_k w_k (theta_k - D_k) 与
  loss = -softplus(x)（logit/IRT 链接，与 loss = log(1-ACC) 精确对齐）。
  **已实现**：`scripts/step2_minacalc_L2.py`（`step2/minacalc_L2`），
  说明与首跑结果见 [step2_minacalc_L2.md](step2_minacalc_L2.md)。
* **R2**：用 --goal 网格产出 D_m(g)，反解每局隐含技能做零训练基线（只用该局之前的历史）。
* **L3**：缓存逐 interval 难度，直接用 MinaCalc 自己的得分模型前向预测，再校准到 mania ACC。
* **换特征**：把 8 维 D 换成 osumania_map_analyser 的 109 维冻结特征（多版本 MSD + 估算器 +
  键型 + 结构量）。同一模型、同一子集下 PCA-7 由 1.5477 降到 1.4874，冷启动由 1.7137 降到
  1.5987。说明与剩余信息损失见 [step2_map_analyser.md](../legacy/docs/step2_map_analyser.md)。
