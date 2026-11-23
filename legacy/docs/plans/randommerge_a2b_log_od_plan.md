# A2b + log + OD 在 linear_extra 上的 P2/P4 验证（计划）

> 2026-11-20。目标：在 **A2b 窗口特征**和 **log 特征**的基础上，加入
> osu!mania 的 **OD（Overall Difficulty，准度判定区间）**作为谱面元数据，
> 用 `linear_extra` 在 **P2 / P4** 两个口径下同时检查 **NLL** 和
> **Pearson / Spearman（d vs b）**。
>
> 背景结果（已完成，见 [randommerge_pool_results.md](randommerge_pool_results.md)）：
> - canon A2b：P2 0.59322、P4 0.59680；A0L：0.59504 / 0.59611。
> - A2b − A0L：P2 **−0.00182（5/5）**，P4 **+0.00069（0/5，更差）**。
> - A0L − A0：P2 −0.00076、P4 −0.00079（5/5）；A1（逐维 std）更差。
> - 主线 legacy `linear_extra`：P2 0.58326、P4 0.58271。
> - **OD 的先验很低**：1k rank eval 里 `od` 是 0 信号
>   （[rank_eval.md](rank_eval.md) §2.1；full coverage，但 within-player 序上
>   captured/lw 都为 0）。本实验不是"再证明 OD 单独有效"，而是看
>   **OD 在 A2b/log 之上有没有增量、以及 P4 冷启动下会不会不同**。

---

## 0. 先澄清两个定义

**（1）A2b 已经含 log。**
当前 A2b = A0 + `mean/sd/q75/q95/max(z)` + `log mean_norm` + `log n_win`
= **519**。所以如果直接写 "A2b + log + OD"，log 会重复计算。本计划把 log
拆成两个显式列，便于消融：

- `log_n = log(n_win)`：窗口数，**crop-independent**，来自
  `.cache/cm3p/index.parquet`；P2/P4/legacy 都能用。
- `log_mn = log(mean_norm)`：窗口一致性，**canon-specific**；只在 canon
  内容表里有；legacy 内容表没有对应值（除非重编码 legacy 窗口）。

**主臂**是 `A2b + OD`（519 + 1 = 520），其中 log 就是 A2b 自带的两个 log；
**不额外再加**。如果要把 log 作为独立可调项，用 §3 的 C4/C5 消融。

**（2）OD 的来源。**
`data/processed/rank_signals_1k.parquet` 已有 `od` 列：

- 0–10 的谱面设置；合并集 20,388 图 **100% 覆盖**；
- 合并集 mean 7.544，sd 0.752；离散且偏斜（8.0 占 5,782/21,949）；
- OD 不在 MSD 里，crop-independent；作为 `linear_extra` 的 extra signal
  时在训练图上 z-score，和 `msd_overall` / `sunny_star` 一样处理。

**（3）legacy A2b 需要先重编码窗口。**
现有 legacy 只有**图级均值** `data/processed/chart_content_cm3p_local.parquet`
（512），没有窗口库；`.cache/cm3p/win` 和 `.cache/cm3p/win_canon_columns_on`
都是 **canon**。A2b 需要每张图的 `W_t`，所以 **legacy A2b 必须用
`CROP=legacy WINDOWS=on` 重新编码** 21,949 图（合并集 20,388）的
~201,766 个窗口（GPU，A800；见 §6）。OD 和 `log_n` 不需要重编码；
legacy 的 `log_mn` 随这次重编码一起得到。重编码后 legacy A2b 的流程和
canon A2b 完全一样：legacy A0 特征 → per-seed legacy A0 head → A2b →
`linear_extra` → P2/P4 eval。

---

## 1. 目标与假设

- **H1（OD 增量）**：在 A2b 之上加 OD，P2/P4 的 NLL 至少 −0.0005 且
  ≥4/5 seed 同向；Pearson/Spearman 不同向变差。
- **H2（log 增量）**：在 legacy 内容表上加 `log_n`（免费、无 GPU）是否
  也能给出 A0L 在 canon 上的 −0.0008；A2b 里的 logs 是否值得保留。
- **H3（A2b 的稳健性）**：控制 OD 后，A2b 的 P2 增益是否还在；
  P4 已知为零，OD 是否能救回来（先验：不太可能）。
- **不期待**：OD 单独成为强信号；它很可能和 `star`/`msd_overall` 共线，
  被 `linear_extra` 现有的 extra signals 吸收。这个实验的价值是
  **用同一套 P2/P4 + 相关性指标把 OD 的增量钉死**，而不是再赌一次。

---

## 2. 数据与特征

- 数据：randommerge 20,388 图；P2/P4 各 5 个 split seed；模型 seed 0；
  `step1_randommerge_c_{p2,p4}_s{1..5}.parquet`。
- 内容表：
  - **canon A2b**：`data/processed/chart_pool_features_randommerge/{p2,p4}/s{seed}/a2b.parquet`
    （上一轮已建好，519）。
  - **canon A0L**：`data/processed/chart_pool_features_randommerge/shared/a0l.parquet`
    （已有，514）。
  - **canon A2b0**（仅消融用）：A2b 去掉两个 log，512+5 = **517**；
    需要在 `dcr/cm3p_pool.py` 新增 `a2b0` variant 并按 seed 构建。
  - **legacy A0**：`data/processed/chart_content_cm3p_local.parquet`（512，
    主线内容表；无需重编码）。
- 额外信号表：`data/processed/rank_signals_randommerge_od.parquet`，
  列 `beatmap_id, msd_overall, sunny_star, daniel_star, star, od, log_n`。
  - `od` 来自 `rank_signals_1k.parquet`；
  - `log_n = log(n_win)` 从 `.cache/cm3p/index.parquet` merge；
  - 20,388 图全覆盖；`log_mn` 不放进 extra（A2b 自带；legacy 无）。
- 模型：`step2_cm3p_rank.py --model linear_extra --lam 1e-3`；
  P2 加 `--chart-only --chart-folds 5 --chart-seed 0`；P4 单次 fit；
  `--extra-signals` 按 arm 给；`--signal-table` 用新表。

---

## 3. 实验臂（每臂 × P2/P4 × 5 seeds）

| arm | 内容 | extra signals（在 msd_overall,sunny_star 之外） | 宽度 | 目的 |
|---|---|---|---:|---|
| C0 | canon A0L | — | 514 | canon 元数据对照 |
| C1 | canon A0L | od | 515 | OD 单独（在 A0L 上） |
| C2 | canon A2b | — | 519 | 上一轮 A2b（对照） |
| C3 | canon A2b | od | 520 | **主臂：A2b + log + OD** |
| C4 | canon A2b0 | od | 518 | 可选：去掉 logs，看 log 贡献 |
| C5 | canon A2b0 | od, log_n | 519 | 可选：只保留 log_n |
| L0 | legacy A0 | — | 512 | 主线对照 |
| L1 | legacy A0 | log_n | 513 | log n 在 legacy 上的 free gain |
| L2 | legacy A0 | od | 513 | OD 在 legacy 上 |
| L3 | legacy A0 | log_n, od | 514 | **legacy 主候选（免费）** |
| L4 | legacy A2b | od | 520 | 可选；**需要 legacy 窗口重编码**（§6），A2b 自带 legacy log |

说明：
- `linear_extra` 的 X = 内容 `Cs` + extra；内容表与 extra 都按训练图
  z-score；OD/log_n 是单列，和 `msd_overall` 同一处理。
- C4/C5 需要新增 `a2b0` variant；如果不想动特征代码，可以只跑
  C0–C3 + L0–L3（8 臂），log 的贡献用 A0L−A0 和 L1−L0 间接回答。
- L1/L2/L3 不需要 GPU、不需要新内容表；这是最便宜、也最接近生产的一组。
- 不建议在 `linear_extra` 里显式加 `OD × A2b` 交互；本计划先看主效应，
  交互是后续（需要手工乘出 519 列，成本高、可解释性差）。

---

## 4. 评测与指标

每个 (protocol, seed)：

1. 训练 §3 的所有 arm，写 `signals/od_{arm}.parquet`，列名 `od_{arm}`；
2. 把各 arm + `sunny_star,daniel_star,star` 合成一张 signal table；
3. `step2_rank_eval.py`：
   `--split test --ceiling-fit all --common-maps --strict-pairs --folds 2
   --bootstrap 0 --signals od_c0,...,od_l3,sunny_star,daniel_star,star`
4. 从 `rank_eval.json` 取 `nll_per_instance`、`pearson_bhat`、`rho_bhat`
   （`tau_bhat` 可附）；同一 seed、同一 `common-maps/strict-pairs` 下比较。

**指标口径**：
- **主指标**：`nll_per_instance`（strict NLL）。
- **次指标**：`pearson_bhat`（d vs b）、`rho_bhat`（Spearman）；
  `tau_bhat` 可选。**不引用 `captured`**：这个合并集上 captured 1.1+，
  ceiling 不是上界（见 [randommerge_p2p4_results.md](randommerge_p2p4_results.md) §5）。
- NLL 的配对 CI 用 `paired_delta_nll.py`（bootstrap 400）；Pearson/Spearman
  没有现成 CI，报告逐 seed Δ、mean ± sd、≥4/5 符号；不做 5-seed 上的
  强显著性宣称。

**预先指定的对比（避免 10 个 arm 的多重比较）**：

| 对比 | 回答的问题 |
|---|---|
| C3 − C2 | OD 在 A2b 上的增量（主） |
| C1 − C0 | OD 在 A0L 上的增量 |
| L2 − L0 | OD 在 legacy 上的增量 |
| L3 − L1 | 控制 log_n 后 OD 在 legacy 上的增量 |
| L1 − L0 | log_n 在 legacy 上的增量（免费收益） |
| C3 − C4 | A2b 的两个 log 在 OD 之上的贡献（可选） |
| C3 − C1 | 控制 OD 后 A2b 相对 A0L 的增益 |
| C2 − C0 | 上一轮 A2b 相对 A0L 的增益（对照） |

---

## 5. 判定门（建议）

- **OD 有用**：C3−C2 ≤ **−0.0005** 且 ≥4/5 seed 同向，且 Pearson/Spearman
  不出现明显反向（|ΔPearson| ≤ 0.002，或至少 4/5 seed 不显著变差）。
  L3−L1 若同向，说明 OD 在 legacy 上也有增量，可进生产。
- **OD 无用**：C3−C2 在 0 附近或为正，或 Pearson/Spearman 变差；
  保持 legacy `linear_extra` 不加 OD（和 1k rank eval 的 0 信号一致）。
- **log_n 免费收益**：L1−L0 ≤ −0.0005 且 ≥4/5；则把 `log_n` 加到 legacy
  内容表（513 维），这是不需要 GPU 的改动。
- **A2b 稳健性**：C3−C1 ≤ −0.001 且 ≥4/5 说明 pooling 在 OD 控制下仍有
  P2 增益；P4 若仍 ≥0，不要为了 P4 加 A2b。
- 所有结论以 **NLL 为主**；相关性只作为"序是否变好/变差"的辅助，
  两者冲突时以 NLL 和预注册的对比为准。

---

## 6. 成本与 HPC

- 特征：A2b/A0L 已存在；A2b0（C4/C5）需要 10 个 seed 的构建（~10 min/seed，
  可 array）；legacy A0 直接用现有表。
- 信号表：一次性 merge，秒级。
- 训练：8 臂（C0–C3, L0–L3）× 10 (protocol, seed) = **80 个任务**；
  P2 每任务 ~22 min、P4 ~7 min；8 线程，`--array=0-79%12`，wall ~2.5–3 h。
  加 C4/C5 则 100 个任务，多 ~30–60 min。
- 评测：10 个 (protocol, seed) 各一次，每次 8–10 个信号，~16 min/seed；
  array 10 任务，~20 min。
- 汇总：NLL/Pearson/Spearman 的 per-seed 表和 paired 表，~分钟级。
- 不需要 GPU、不需要 MLP、不需要 legacy 窗口重编码。

**legacy A2b 的 GPU 重编码（可选，不在 80 任务内）**

如果要把 A2b 也做到 legacy（L4），先补 legacy 窗口库。现有 legacy 只有图级
均值，没有 `W_t`；`.cache/cm3p/win*` 都是 canon。用现成的
`cm3p_embed.slurm`（`CROP`/`WINDOWS` 可覆盖）：

```bash
CROP=legacy WINDOWS=on \
WINDOW_DIR=.cache/cm3p/win_legacy \
WINDOW_INDEX=.cache/cm3p/index_legacy.parquet \
EMBED_OUT=data/processed/chart_content_cm3p_local_legacy_recheck.parquet \
SKIP_PROBE=1 \
sbatch -p i64m1tga800u --gres=gpu:1 --time=04:00:00 scripts/slurm/cm3p_embed.slurm
```

- 21,949 图（合并集 20,388）约 201,766 窗口；A800 全量 45–90 min + 队列；
  `.cache/cm3p/win_legacy` ~0.2 GB float16；索引 `index_legacy.parquet`。
- 一次重编码可复用：以后 legacy 的 A1/A2/A2b/A2c/A3/A4 都能用这份窗口库。
- 重编码后：用 legacy `a0` 特征（或现有 legacy 表）按 seed 拟合 legacy
  A0 head，再用 `cm3p_pool.py --index index_legacy --win-dir win_legacy
  --head-mode fold/full` 建 legacy A2b；L4 = legacy A2b + OD。
- 成本：GPU 1 次 + 10 个 seed 的 A0 head/A2b 特征（~1–2 h，可 array）
  + L4 的 10 个训练任务（P2 22 min / P4 7 min，可与其他 arm 一起）。
- 预期：canon A2b 的 P2 增益只有 0.0018、P4 为零；legacy A2b 大概率类似。
  如果只想看生产能不能再降一点，先跑免费的 L0–L3；有信号再决定是否付
  GPU；如果把 legacy 窗口特征当长期资产，这次重编码本身值得做。

**建议的 HPC 任务链**：
`od_signals`（新表）→ `od_train`（80）→ `od_eval`（10）→ `od_summary`（1）。
smoke 用 P2 seed 1 的 C2/C3/L0/L3 四个 arm 先跑，确认 extra signal 和
P2/P4 的 `step2_cm3p_rank` 参数无误，再放 80 任务。

---

## 7. 风险

| 风险 | 处理 |
|---|---|
| OD 先验为 0（1k rank eval） | 明确这是"增量/交互"测试；C3−C2 为主；失败就 drop，成本低 |
| OD 与 msd_overall/star 共线 | 报告 OD 与这些信号的 Pearson；若 |r| > 0.8，解释为"被现有 extra 吸收"，不夸大 |
| OD 是谱面设置不是 content | 单独标为 metadata arm；不要和"content-only"结论混在一起 |
| A2b 已含 log，重复计算 | C3 用 A2b（含 log）+ OD；C4/C5 才拆 log；文档和信号命名写清楚 |
| P2 head 的 train-feature 泄漏 | 沿用上一轮的 per-seed 5 折 head；P2 结果加同样的 caveat；P4 head 是 train-only |
| 10 个 arm 的多重比较 | 只报 §4 的预注册对比；其余作描述性 |
| captured 异常 | 不引用 captured；只看 NLL/Pearson/Spearman |
| 相关性没有 CI | 逐 seed Δ + 符号 + mean±sd；不做强显著性宣称 |

---

## 8. 时间线

1. 建 `rank_signals_randommerge_od.parquet`；如做 C4/C5，先加 `a2b0`
   并跑 10 个 seed 的特征（array）。
2. P2 seed 1 的 C2/C3/L0/L3 smoke（~1 h），核对：
   - extra signals 在训练图上 z-score、OD/log_n 没有 NaN；
   - P2 的 chart folds 与 head 对齐；
   - `rank_eval.json` 有 NLL/Pearson/Spearman。
3. 提交 80（或 100）任务 train + 10 eval + summary。
4. 出表：每个 arm × P2/P4 的 NLL、Pearson、Spearman；预注册对比的
   Δ、CI、逐 seed 符号；决定是否把 `log_n`（以及 OD）加进 legacy 生产。

---

## 9. 需要确认的点

1. **"log" 具体指什么？** 当前 A2b 已含 `log n_win` 和 `log mean_norm`。
   如果主臂就是 `A2b + OD`，log 已经在里面，不用再加；如果要把 log 作为
   独立 feature，请用 C4/C5 的消融口径。
2. **legacy A2b 要不要做？** 能做，但需要先重编码 legacy 窗口（§0(3)/§6）：
   现有 legacy 只有图级均值，没有 `W_t`；`.cache/cm3p/win*` 都是 canon。
   A800 全量 ~45–90 min + 队列；重编码后 legacy A2b（L4）和 canon A2b 一样跑。
   **建议**：先跑免费的 L0–L3（A0+log_n/OD）；如果 log_n/OD 有信号，或要把
   legacy 窗口特征当长期资产，再做重编码；否则不值得为 P2 的 ~0.0018 付 GPU。
3. **OD 作为 extra signal 还是内容列？** 本计划用 extra（和 msd_overall/
   sunny_star 一致）；如果希望它进入内容 z-score，可以改，但线性模型下
   两者只差 λ 的缩放，不值得开第二个变量。
4. **C4/C5 的 `a2b0` 消融要不要做？** 如果不做，log 的贡献只能用
   A0L−A0 和 L1−L0 间接回答；做了更干净，但多 10 个特征任务。

---

## 10. 执行记录（2026-11-20）

- **决定**：先跑 **L0–L3**（legacy A0 + `log_n` / `od`），全部在 HPC CPU
  `i64m512u`；L4（legacy A2b）暂缓，需要 GPU legacy 窗口重编码（§6）。
- **新信号表**：`data/processed/rank_signals_randommerge_od.parquet`
  （21,949 行；合并集 20,388 全覆盖；无 NaN；`od` vs
  `msd_overall/sunny_star/daniel_star/star` 的 Pearson 0.45–0.51，
  不是完全共线；`log_n = log n_win`，来自 `.cache/cm3p/index.parquet`）。
- **驱动**：`scripts/run_od_legacy.py`；冒烟 `scripts/smoke_od_legacy.py`
  （本地合成数据通过：4 arms、P2/P4、NLL/Pearson/Spearman、paired 都在）。
- **slurm**：`randommerge_od_legacy_train.slurm`（40 任务，`%12`）、
  `randommerge_od_legacy_eval.slurm`（10）、
  `randommerge_od_legacy_summary.slurm`（1）；
  提交 `randommerge_od_legacy_submit.sh`。
- **HPC job（2026-11-20 13:20）**：train **14031717**（0-39%12）、
  eval **14031718**（0-9）、summarize **14031719**；全部 CPU `i64m512u`。
- **结果（2026-11-20 14:07，全部 COMPLETED）**：
  - L0 精确复现主线：P2 0.58326、P4 0.58271。
  - **L1 `log_n`**：P2 −0.00079（5/5）、P4 −0.00069（5/5）；
    Pearson/Spearman +0.0005–0.0007（5/5）。
  - **L2 `od`**：P2 −0.00016、P4 −0.00008；控制 `log_n` 后
    L3−L1 只有 P2 −0.00013（5/5）、P4 −0.00006（3/5）→ 近噪声。
  - **L3** 绝对最好（P2 0.58234、P4 0.58196），但只比 L1 多
    0.00013/0.00006；**OD 不加**。
  - **判定**：`log_n` 通过（免费、免 GPU）；OD 不通过（和 1k rank eval 的
    0 信号一致）；L4/A2b 暂缓；C 臂未跑。
  - **生产建议**：legacy `linear_extra --extra-signals msd_overall,sunny_star,log_n`，
    内容表不变。完整表格见 [randommerge_od_legacy_results.md](randommerge_od_legacy_results.md)。
