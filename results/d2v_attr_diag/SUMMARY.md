# D2V 属性通道预实验诊断：C1 / C3 / C4

> 2026-11-24，subagent "diag"。全部本地 CPU，未训练、未改其他源码。
> 写作用域：scripts/build_meta_content.py、scripts/content_mapper_probe.py、results/d2v_attr_diag/**（另按要求生成 data/processed/chart_content_meta_summary.parquet 及其 .meta.json）。
> 结构化汇总见同目录 summary.json；复现命令见 §6。

## 0. 结论速览

1. **C1 metadata-only rank 很弱**：12 维图级属性摘要的 P2-all 5 折 linear probe test NLL = **0.66018**，
   captured = **0.2528**（evaluation pairs 1,144,617）。对照 D2V-8ch 512d 内容表 **0.58632 / captured 0.8193**、
   CM3P legacy P2 linear **0.61618 ± 0.00293**（引用，不重算）：差 **+0.0739** / **+0.0440** nats。
   图级属性不是内容表的替代品；属性通道只能以"附加在内容表上"的 paired ΔNLL 衡量。
2. **C3 mapper 探针**：D2V-8ch 内容上 top-1 = **0.2308**（majority 0.0478，chance 0.02，macro-F1 0.2025，
   n_test=2,049 charts / 50 mappers）→ 冻结的 512d 内容保留了强作者风格签名。metadata-only 对照 top-1 = 0.1459，
   说明 D2V 相对 12 维手写属性多编码了约 8.5 pt 的 mapper 可识别信息。
3. **C4 属性可解码性**（ridge，512d -> 属性，test R2）：log2_bpm_p50 **0.383** > hold_frac_gt1s **0.326**
   > hs_any_ext_frac **0.295** >> log1p_n_sv_changes **0.107** > kiai_frac **0.078** > vol_resolved_p50 **0.054**。
   SV / kiai / volume 在 8ch 内容里几乎不可线性解码——正是结构化属性通道要补的容量缺口；
   BPM / hold / hitsound 已有相当信息，新通道在这几项上的边际收益预计很小。
4. **实现正确性交叉检查**：同一 set-disjoint split 下，12-d metadata 内容表对 5 个直接特征目标的 C4 R2 = 1.0000
   （这是 tautology，仅证明切分/标准化/R2 实现无错误、无泄漏），hs_any_ext_frac = 0.9858（三组分 hitsound frac 可合成）。

## 1. 内容表（build_meta_content.py 的产物）

- 输出：data/processed/chart_content_meta_summary.parquet，**21,949 行 x 6 列**，
  契约与 D2V 内容表一致：beatmap_id(int64) / embedding(12 维 float32 np.array) / n_windows(=1) /
  checksum_ok(=True) / source("meta_summary") / d2v_rev("n/a")；beatmap_id 集合与 chart_content_d2v_holdout.parquet 完全一致。
- 12 特征（顺序即契约）：log2_bpm_p50, log1p_n_red, log1p_n_sv_changes, kiai_frac, hold_frac_gt1s,
  log_hold_ms_p50, hs_whistle_frac, hs_finish_frac, hs_clap_frac, vol_resolved_p50_100,
  log1p_notes_per_sec, log1p_span_s。
- 缺失：只有 hold_ms_p50 缺失（1,806 张无 hold 图，8.2%），用 step1 训练图（6,143 张）中位数 **162.0 ms**
  （log 值 5.0876）填补。原值覆盖 91.8%。
- bpm_p50 裁剪到 [1, 1000]：原表有 3 张 <=0 BPM（0 BPM 红线）和 13 张 >1000 BPM（最高 27,688）——
  不裁剪 log2 会非有限/极端离群。其余特征原始覆盖 100%。

## 2. C1：metadata-only P2 rank probe

命令与产物：
- python3 scripts/step2_cm3p_rank.py --in data/processed/step1_1k_pairbest.parquet --content-path data/processed/chart_content_meta_summary.parquet --model linear --chart-only --chart-folds 5 --chart-seed 0 --threads 8 --out-signals results/d2v_attr_diag/meta_summary_p2_signals.parquet --out-json results/d2v_attr_diag/meta_summary_p2_rank.json
- 27.0s；6,828 charts，test 766,369 pairs / 1,144,617 comparisons。

结果（P2-all 口径，ceiling NLL 0.5627606，与 8ch 基线同一 ceiling）：

| 指标 | C1 metadata-12 | D2V-8ch 512d（引用） | CM3P legacy（引用） |
|---|---:|---:|---:|
| test NLL / comparison | **0.66018** | 0.58632 | 0.61618 ± 0.00293 |
| captured | **0.2528** | 0.8193 | —（未引用 captured）|
| captured per-player LW / macro / median | 0.3145 / 0.3023 / 0.3139 | 0.5024 / 0.4724 / 0.5241 | — |
| Pearson / Spearman / Kendall (bhat) | 0.2924 / 0.3798 / 0.2711 | 0.5829 / 0.5314 / 0.3808 | — |
| ΔNLL vs D2V-8ch | **+0.07386** | — | — |
| ΔNLL vs CM3P legacy | **+0.04400** | — | — |

来源：D2V 数字 results/d2v_main/eval/p2_all_d2v.json + p2_all_d2v_rank.json（gate 同一 --in/--
chart-folds 5/split test/ceiling-fit all）；CM3P legacy P2 linear 0.61618 ± 0.00293 来自
docs/cm3p_rank.md:493（5-seed P2 runner），按任务要求只引用，不在本次重算。

λ 敏感性（同命令唯一改动 --lam）：1e-3 -> 0.65985；1e-2（任务命令默认，因 --chart-only 跳过 P1 调参）
-> 0.66018；1e-1 -> 0.66391。结论对 λ 稳健（仍比 8ch 差 +0.0735 以上）。
注：0.58632 基线的 best_lam=1e-3（其 P1 验证折选出）；本次主结果按任务给定命令用默认 1e-2，
两者差异 <3e-4，不影响比较结论。

读取提示：12 个图级属性是"弱先验"；C1 低不等于属性无用——CM3P canon 消融显示这三类属性在 token 空间
值 ~0.02 nats（eval_protocol.md §1.3），路径依赖内容表+属性融合，而不是 metadata-only rank。

## 3. C3：mapper-ID probe（D2V-8ch）

- 切分：beatmapset 整组 70/30（numpy default_rng(0)）；train 15,445 charts / 4,800 sets，
  test 6,504 charts / 2,057 sets（21,949 charts / 6,857 sets）。
- 特征在 train charts 上 z-score；top-50 mappers by train chart count（每类 56-219 张 train 图）；
  multinomial LogisticRegression(max_iter=200, lbfgs, C=1.0)（sklearn 1.7.2）。
- 结果：test top-1 **0.2308**（2,049 张 test 图），majority baseline **0.0478**，chance 0.02，
  lift over majority **+0.1830**，macro-F1 **0.2025**，weighted-F1 0.2313。
- 对照（同一 split，12-d metadata summary）：top-1 0.1459，macro-F1 0.1108。
  => D2V 的内容向量比手写属性多约 8.5 pt 的 mapper 风格可识别性。

## 4. C4：属性可解码性（D2V-8ch 512d，ridge）

同一 set-disjoint split；特征在 train z-score；每目标 RidgeCV(alpha 由 train charts GCV 选，6 个候选
1e-1..1e4)；test R2：

| 目标 | test R2 | train R2 | Pearson(test) | alpha | mean-predictor R2 |
|---|---:|---:|---:|---:|---:|
| log1p(n_sv_changes) | **0.1066** | 0.2178 | 0.3416 | 10 | -0.0000 |
| kiai_frac | **0.0784** | 0.1928 | 0.3047 | 10 | -0.0034 |
| log2(bpm_p50) | **0.3827** | 0.4133 | 0.6198 | 10 | -0.0002 |
| hold_frac_gt1s | **0.3256** | 0.3764 | 0.5773 | 10 | -0.0002 |
| vol_resolved_p50 | **0.0542** | 0.1759 | 0.2616 | 10 | -0.0001 |
| hs_any_ext_frac | **0.2954** | 0.3549 | 0.5461 | 10 | -0.0007 |

(train R2 明显高于 test R2 说明 512 维仍有可观的 mapper/曲目记忆；这是表征诊断不是泛化结果。)

## 5. 坑 / 注意事项

1. **bpm_p50 退化值**：3 张 <=0、13 张 >1000（max 27,688）。log2 前必须 clip（本次 [1,1000]），
   否则非有限；这些值本身是解析/恶意 timing 的边界，不是真实 BPM。
2. **hold_ms_p50 缺失 1,806 张（无 hold 图）**：按任务要求用训练集均值填补（162 ms），
   这会把"无 hold"图与"中位 hold 长度"图混为一谈；如需精确应加 has_hold 指示位。
3. **--chart-only 跳过 P1 调参**：best_lam 回落默认 1e-2（基线 0.58632 用的是验证折选出的 1e-3）。
   影响 <3e-4，但复现/报告时应显式写 --lam。
4. **C3 的 LBFGS 未收敛**：max_iter=200 到限（n_iter_=200），top-1 是下界，macro-F1 同样保守；
   要精确值需加大 max_iter（任务规定 200，故保留并记录）。
5. **C3 只在 top-50 mapper 子集上评测**（5,131 train / 2,049 test 图），绝对准确率不能与其它 K 或
   全 mapper 池的数字直接比；majority baseline 也随之变高。
6. **set-disjoint 但非 mapper-disjoint**：同一 mapper 的其他 set 可在 train 出现，C3 度量的是
   "风格签名可读"，不是"对未见 mapper 泛化"。
7. **C4 的 12-d 对照是 tautology**（5 个目标直接是摘要特征），R2=1.0 仅作 pipeline 无泄漏的 sanity check；
   不要把它当作 metadata 的强度证据。hs_any_ext_frac=0.9858 是因为三个 hitsound 分量 frac 可合成它。
8. **C1 的绝对值不可跨数据集比**：只与同一 --in/split/ceiling 的数字比；本表已固定 P2-all 的 ceiling 0.5627606。
9. **内容表覆盖 21,949 图**（含 D2V 预训练池图）；C1/C3/C4 都是表征/内容诊断，不是难度泛化结论；
   也未涉及 beatmapset/mapper 泄漏之外的时间漂移问题。

## 6. 文件与复现

产物（results/d2v_attr_diag/）：
- meta_summary_p2_signals.parquet, meta_summary_p2_rank.json（C1 主结果）
- meta_summary_p2_signals_lam0.001.parquet / ..._lam0.1.parquet + 对应 JSON（λ 敏感性）
- meta_summary_p2_rank_eval.json（captured / Pearson / Spearman / per-player）
- mapper_probe_d2v8.json（C3）、attr_decodability_d2v8.json（C4）
- mapper_probe_meta12.json、attr_decodability_meta12.json（对照/sanity）
- logs/*.log、summary.json、SUMMARY.md（本文件）

复现：
    python3 scripts/build_meta_content.py
    python3 scripts/step2_cm3p_rank.py --in data/processed/step1_1k_pairbest.parquet --content-path data/processed/chart_content_meta_summary.parquet --model linear --chart-only --chart-folds 5 --chart-seed 0 --threads 8 --out-signals results/d2v_attr_diag/meta_summary_p2_signals.parquet --out-json results/d2v_attr_diag/meta_summary_p2_rank.json
    python3 scripts/step2_rank_eval.py --in data/processed/step1_1k_pairbest.parquet --signal-table results/d2v_attr_diag/meta_summary_p2_signals.parquet --signals cm3p_rank --split test --ceiling-fit all --folds 2 --bootstrap 300 --out results/d2v_attr_diag/meta_summary_p2_rank_eval.json
    python3 scripts/content_mapper_probe.py --content-path data/processed/chart_content_d2v_holdout.parquet --out-dir results/d2v_attr_diag --mode both
    python3 scripts/content_mapper_probe.py --content-path data/processed/chart_content_meta_summary.parquet --out-dir results/d2v_attr_diag --mode both --tag meta12 --mapper-out results/d2v_attr_diag/mapper_probe_meta12.json --attr-out results/d2v_attr_diag/attr_decodability_meta12.json

环境：python 3.13.9 / numpy 2.3.5 / pandas 2.3.3 / scipy 1.16.3 / sklearn 1.7.2，8 核本地 CPU。
