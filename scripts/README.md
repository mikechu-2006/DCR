# scripts/ —— 脚本索引

实现层：一个 step 一个程序，都能脱离 `main.py` 单独运行。统一入口是
[`main.py`](../main.py)，流程见 [docs/procedure.md](../docs/procedure.md)。
退役脚本在 [legacy/scripts/](../legacy/scripts/)。

## 管线 step

| 文件 | step 版本 | 说明 |
|---|---|---|
| `step0_preprocess.py` | `step0/preprocess` | dump → play 表（5 列）+ 谱面元数据（17 列） |
| `step0c_chart_content.py` | `step0/chart_content` | beatmap_id → CM3P 512 维内容向量；`--crop canon`（T1）+ `--windows on`（T2 窗口导出/分桶）；见 docs/cm3p_crop_windows.md |
| `cm3p_token_budget.py` | — | 离线 T0/T1 审计：token/note、截断、hold split、**core stream 裁剪前后完全一致**；`--verify-cm3p` 在集群上对真实 tokenizer |
| `step0d_fetch_osu.py` | `step0/fetch_osu` | 按 id 清单抓 `.osu` **正文** |
| `step0g_whitelist.py` | `step0/whitelist` | 白名单表 + 缓存状态（`cacheable`） |
| `step0e_validate_mix.py` | `step0/validate_mix` | 校验不同来源的 CM3P 向量能否混用 |
| `step0_union_datasets.py` | `step0/union` | 数据集并集：多个 step0 play 表 → 合并 play 表 + 合并 meta（play 级去重、重排 playcount）；见 docs/full_union_report.md |
| `step1_pairbest.py` | `step1/pairbest` | **默认 A**：一对一行（规则见 docs/data.md） |
| `step1_clean.py` | `step1/clean` | legacy play 级：划分后清洗 train 侧 |
| `step1_pairbest_resplit.py` | — | 重新生成 pair-random 70/30 划分（P2 多种子；split 是 step1 最后一步，等价于重跑 step1 换 split-seed） |
| `step1_p4_split.py` | — | P4 双重冷启动划分：玩家/谱面各分层 60/40，6×6 train / 4×4 test；见 docs/p4_multiseed.md |
| `step1_content_subset.py` | — | step1 表按内容表过滤（CM3P 训练子集；randommerge / full 的 `*_content` 表） |
| `step2_predict.py` | `step2/predict` | bias / mf_dot（+ CM3P 内容向量；mirt 已退役） |
| `step2_cm3p_mlp.py` | `step2/cm3p_mlp` | CM3P 内容锚定 MLP |
| `test_step2_cm3p_mlp.py` | — | 上述 MLP 的 7 条自检（合成张量，1 秒，不需要 data/） |
| `step2_rank_eval.py` | — | 难度序的成对 NLL（BT ceiling / captured / 逐玩家 Spearman / 与 BT 尺度 `b` 的 Pearson、Spearman）；ceiling 默认用全部行拟合、只在 `--split` 行评测；见 docs/metrics.md |
| `test_rank_metrics.py` | — | rank-NLL / BT / ceiling 拟合-评测拆分 / strict pairs / torch pairwise loss 的 24 条合成自检（不需要 data/） |
| `test_step2_rank_eval.py` | — | rank runner 的端到端合成测试：`--ceiling-fit all|split`、`--common-maps`、`--strict-pairs`（不需要 data/） |
| `step2_cm3p_rank.py` | — | CM3P 内容→per-chart 难度，按 rank-NLL counts 训练（M0/M1/M2；P1/P2/content ceiling）；见 docs/cm3p_rank.md |
| `run_cm3p_rank_experiments.py` | — | 上述模型的实验驱动（P1/P2/published/合并评测） |
| `run_p2_p4_multiseed.py` | — | P2/P4 5 个**划分种子** × 7 方法的实验驱动与 mean/std 汇总；见 docs/p4_multiseed.md |
| `cm3p_pool.py` | — | T3 窗口池化 A0–A6：特征（a0–a5）或 A6 chart-disjoint TCN 信号 |
| `run_cm3p_pool_experiments.py` | — | T3 驱动：A0–A6 → P2 5 折 → 官方 NLL + paired ΔNLL + Go/no-go |
| `stratified_delta_nll.py` | — | 计划 §9：在 `count_total` 分位层内比较 legacy vs canon 的 paired ΔNLL |
| `test_cm3p_windows.py` | — | T0/T1/T2 合成自检（裁剪审计、分桶、导出、column patch、离线会计） |
| `test_cm3p_pool.py` | — | T3 合成自检（A0–A5 宽度/A0 等价、A6 小样本、驱动 merge/summary） |
| `paired_delta_nll.py` | — | 同一 cluster/pair 的配对 ΔNLL + bootstrap CI |
| `player_disjoint_val.py` | — | 20% held-out player cluster 的 val NLL 表（含 fixed baseline） |

公共路径参数（不传时保持旧行为）：

* `step0_preprocess.py`：`--dump-dir --beatmap-src --whitelist --out --meta-out --work-dir`
* `step0c_chart_content.py`：`--tag --meta-dir --ids-file --in --out`；
  T1/T2：`--crop canon|legacy --windows on|off --mania-columns auto|on|off --window-dir --window-index`
* `step0d_fetch_osu.py`：`--ids-file --out-dir --state`
* `step0g_whitelist.py`：`--tag --meta --cache --out --manifest-out --dropped-out --json`
* `step0e_validate_mix.py`：`--computed --probe-npz --json`
* `step0_union_datasets.py`：`--play PATH=TAG --meta PATH=TAG --out --meta-out --report`
* `step1_*.py`：`--in --out` + 清洗/划分参数
* `step2_*.py`：`--train-path --proc-dir --out-dir` + 模型参数（见各自 `--help`）

## 分析

| 文件 | 说明 |
|---|---|
| `minacalc_whitelist_stats.py` | 白名单谱面的 MinaCalc 全曲/分段难度分布；产物在 `results/minacalc/whitelist_stats/` |
| `inspect_score.py` | 单张谱面的 MinaCalc 7 维难度 / 所需技能诊断（`--diag --json`）；示例见 `results/minacalc/inspect_score_example.txt` |
| `fit_player_vectors.py` | 玩家技能向量拟合（MiniCalc 6 维 LM vs CM3P+MLP 对照）；产物在 `results/player_vector_fit/` |
| `index_experiments.py` | 读 `experiments/*/manifest.json` 生成 `experiments/INDEX.md`（本机索引，实验目录不入库） |
| `union_report.py` | 并集数据集报告：来源重叠与复活、来源 step1 覆盖、标签、每图支持度、内容与 star；产物 docs/full_union_report.md |

```bash
python scripts/minacalc_whitelist_stats.py --jobs 8   # 分片并行 + 合并
python scripts/minacalc_whitelist_stats.py --merge    # 只重算汇总/报告
python scripts/inspect_score.py 4305472 -p 20 19 18 17 18 19 --diag
python scripts/fit_player_vectors.py --dataset 10k --players 100 --seed 20261023
python scripts/index_experiments.py                   # 写 experiments/INDEX.md
```

## 共享模块

`parse_dump.py` 解析 data.ppy.sh 的 SQL dump，由 `step0_preprocess.py` 导入；
不是独立 step。

## 运维

| 文件 | 说明 |
|---|---|
| `remote_cm3p_run.sh` | 远程机器补齐 CM3P 向量的一键流程 |
| `slurm/cm3p_fetch.slurm` | 集群上抓 `.osu` 正文 |
| `slurm/cm3p_embed.slurm` | 集群上跑 CM3P 推理 |
| `../copy_folder.sh` | 本机 → HPC 整包同步（`experiments/`、`legacy/` 不同步） |

## 已归档

MinaCalc 相关脚本（`step0h_minacalc_msd.py`、`step2_minacalc_*.py`、
`step2_fit_skills.py` 等）以及分析/诊断/旧管线脚本都在
[legacy/scripts/](../legacy/scripts/)；MinaCalc 包本身见
[minacalc/](../minacalc/README.md)。
