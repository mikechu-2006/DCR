# P4 audio-on 实验：手动执行计划（2026-11-19）

> **执行完成（2026-11-19 01:44）。** 结果见 docs/p4_multiseed_audio.md §4.3/§5。
>
> **Goal 已移除（无活动 goal）。** 本文件是唯一执行清单；长任务用后台 watcher，
> 不在对话里反复 job_output(wait) / sleep 轮询。

## 0. 目标

把 HPC 新算的 2,782 张 audio-on 向量与 published 的 4,037 张合并成 6,819 张完整表，
在 P2/P4 协议下与 audio-off legacy 做同划分、同 ceiling 的配对对照；写文档并汇报。

## 1. 已完成

- [x] 4 张无法编码的图从 manifests/whitelist_1k.csv 移除：21,949 -> **21,945**。
  - 缺口：259067 / 374464 / 486889 / 487516。
  - 记录：manifests/whitelist_1k_removed.csv。
- [x] 严格重建（A 口径）：
  - data/processed/step0_1k_nogap.csv：1,398,611 行（删 1,004 条 play）。
  - data/processed/step1_1k_pairbest_nogap.parquet：**109,569 行 / 6,819 图 / 941 base players / 1,039 segments**。
  - 额外丢 5 张 published 图 + 1 个玩家（365 天 epoch 副作用），已在文档说明。
- [x] 完整 audio-on 内容表：
  - data/processed/chart_content_cm3p_audio_on.parquet：**6,819 = 4,037 published + 2,782 local audio**。
  - 512 维、单位范数、100% 覆盖；md5 7180f2f4b40c857613d534396c09b4e4。
  - 几何：published 0.547 / local 0.544 / audio-off 0.893。
- [x] 新划分 seeds 1-5：
  - data/processed/step1_1k_pairbest_p2_nogap_s{1..5}.parquet
  - data/processed/step1_1k_pairbest_p4_nogap_s{1..5}.parquet
- [x] **audio 臂完成**：results/p4_multiseed_audio/audio/summary.csv / summary.md。
  - P2 ceiling 0.56104 +/- 0.00394；P4 ceiling 0.56618 +/- 0.01099。
  - P2 external MLP 0.55847、P4 external MLP 0.56731，均为该臂最好。
- [x] 文档骨架：docs/p4_multiseed_audio.md（§4.1/§4.2 已填，§0/§4.3/§5 待填）。
- [x] 新增比较/汇总脚本：
  - scripts/compare_multiseed_unpaired.py
  - scripts/summarize_p4_multiseed_audio.py
  - scripts/watch_p4_multiseed_audio.sh
- [x] 索引：results/README.md、README.md、docs/p4_multiseed.md §10、docs/cm3p_official_probe.md 已加指针。

## 2. 执行完成

- 主作业 bash-9543a3615de745c4：已于 2026-11-19 01:43:58 完成（exit 0）。
- watcher bash-6212aa4741d34736：已于 01:44:28 完成并写 ANALYSIS_DONE。
  - 日志 results/p4_multiseed_audio/analysis.log（无 Traceback）
  - 产物：
    - results/p4_multiseed_audio/audio_vs_legacy.md（配对主检验）
    - results/p4_multiseed_audio/legacy_vs_p4_multiseed.md（数据变更检查）
    - results/p4_multiseed_audio/audio_vs_p4_multiseed.md（描述性参照）
    - results/p4_multiseed_audio/combined_summary.md/.csv（old / legacy / audio 三臂汇总）

## 3. 收尾步骤（已完成）

- [x] 等 watcher 完成；检查 ANALYSIS_DONE 和 analysis.log。
- [x] 读 combined_summary.md、audio_vs_legacy.md，确认 5 seed、ceiling 逐位相同、4 方法 5/5。
- [x] 填 docs/p4_multiseed_audio.md §0 / §4.3 / §5（另加 §4.4 训练侧证据）。
- [x] 更新 README.md 结果速览、results/README.md 索引。
- [x] 验证：audio/legacy 各 70 行、paired 8 行、combined 14 行、fixed 信号逐位相同。
- [x] 汇报：配对主结论、P2/P4 差异、混合来源/归因限制。

## 4. 主判据

- **主检验**：audio - legacy，同 split seed、同 ceiling、同 fixed signals。负值 = audio 更好。
- **数据变更检查**：legacy_new - p4_multiseed，图集/划分不同，非配对，只作量级参照。
- **不要**把 audio_new vs p4_multiseed 当配对检验；两者 split 不同。
- 绝对 NLL 不可跨 6,819 / 6,828 图集比较。
- 来源混合：published + local audio；全量增益不能单独归因给新算的 2,782 张。

## 5. 若 watcher 失败

1. 看 results/p4_multiseed_audio/analysis.log 尾部与 run_all.log 尾部；
2. 若只是某个比较脚本失败，手动重跑对应命令；
3. 若主作业失败，先修原因再决定是否重跑 legacy 臂；
4. 不要恢复 goal；按本文件手动继续。

## 6. 最终结果

- P2：4 个 embedding 方法配对 ΔNLL **−0.0083 … −0.0208**，全部 5/5；
  CM3P + external MLP = **0.55847**。
- P4：4 个 embedding 方法配对 ΔNLL **−0.0076 … −0.0218**，全部 5/5；
  CM3P + external MLP = **0.56731**。
- 数据变更（legacy_new − old）：最大 **−0.00552**（P4 CM3P MLP）；P2 全部 ≤ +0.00178。
- fixed Sunny / Daniel / star 两臂逐位相同；ceiling 逐 (protocol, seed) 相同。
