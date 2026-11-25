# D2V 结构化属性输入：训练数据审计（自动生成）

由 data2vec_main/scripts/audit_attributes.py 生成；汇总 results/d2v_attr_audit/summary.json，逐图表在 data/processed/d2v_attr_audit_per_chart.parquet（不入库）。

* 谱面 **21949**，音符 33,016,586，hold 5,939,098，谱面时长合计 913.9 h。
* 解析失败/空图：0。

## 1. 覆盖率（按谱面）

| 属性 | 覆盖率 |
|---|---:|
| 任意 kiai | 76.1% |
| 任意有效 SV 变化 | 22.7% |
| 任意非普通 hitsound | 45.4% |
| 任意逐音符 sample set 覆写 | 7.4% |
| 任意逐音符 volume 覆写 | 42.3% |
| 解析后 sample set 取值 >1 | 26.6% |
| 解析后 volume 取值 >1 | 58.6% |
| timing point volume 有过变化 | 22.3% |
| 有红线 BPM | 100.0% |

## 2. 取值分布

| 字段 | p05 | p50 | p95 | mean |
|---|---:|---:|---:|---:|
| n_notes | 270 | 1098 | 4060 | 1504 |
| span_s | 60.6 | 127.1 | 293.9 | 149.9 |
| notes_per_sec | 2.972 | 8.887 | 17.98 | 9.33 |
| n_red | 1 | 1 | 20.6 | 12.81 |
| n_inherited | 0 | 5 | 92 | 55.48 |
| n_bpm_values | 1 | 1 | 12 | 5.136 |
| bpm_p50 | 103 | 174.5 | 250 | 178.8 |
| n_sv_changes | 1 | 1 | 81 | 52.96 |
| sv_changes_per_10s | 0.03948 | 0.09117 | 5.546 | 2.97 |
| frac_time_sv_off1 | 0 | 0 | 0.0933 | 0.01898 |
| n_kiai | 0 | 4 | 12 | 4.486 |
| kiai_frac | 0 | 0.4494 | 0.8904 | 0.4408 |
| hold_ms_p50 | 88 | 234 | 714 | 310.7 |
| hold_ms_p95 | 257 | 634.8 | 2386 | 883.5 |
| hold_frac_gt1s | 0 | 0.01493 | 0.3 | 0.06444 |
| hs_normal_frac | 0.244 | 1 | 1 | 0.7899 |
| hs_whistle_frac | 0 | 0 | 0.4461 | 0.1176 |
| hs_finish_frac | 0 | 0 | 0.1204 | 0.02697 |
| hs_clap_frac | 0 | 0 | 0.3119 | 0.07699 |
| hs_any_ext_frac | 0 | 0 | 0.756 | 0.2101 |
| hs_n_combos | 1 | 1 | 7 | 2.844 |
| ss_auto_frac | 0.9625 | 1 | 1 | 0.9769 |
| ss_drum_frac | 0 | 0 | 0 | 0.005086 |
| vol_inherit_frac | 0.3188 | 1 | 1 | 0.8866 |
| vol_resolved_p50 | 15 | 50 | 80 | 48.79 |
| vol_resolved_distinct | 1 | 2 | 8 | 3.004 |
| ss_resolved_n_distinct | 1 | 1 | 2 | 1.296 |
| tp_vol_p50 | 15 | 50 | 80 | 47.5 |
| tp_vol_distinct | 1 | 1 | 5 | 1.65 |
| est_release_frac | 0 | 0.06167 | 0.4064 | 0.1124 |
| est_hs_ext_frac | 0 | 0 | 0.3469 | 0.104 |
| est_sv_change_frac | 0.0002781 | 0.000641 | 0.03955 | 0.01661 |

## 3. 解析器缺口 / 边界

* charts_with_kiai_effects_not_1: **145**
* notes_in_those_charts: **271309**
* charts_with_tp_volume0: **375**
* note_frac_volume_raw0: **0.919941480321436**

## 4. beatmapset / mapper 分组（泄漏证据）

* 有 meta 的谱面 21949，beatmapset 6857，mapper 1399。
* 多难度 set 内谱面占比 **90.6%**（每 set 中位 3 张，max 35）。
* D2V holdout：eval 图 1365，set 1146；其中 **78.6% 的 set 也出现在预训练池**，对应 **81.7% 的 eval 图**。
* 同一 set 内不同难度对（38774 对）共享：BPM 网格 81.4%、SV 变化数 87.3%、kiai 数 97.1%、完整 timing 签名 85.3%。

## 5. 读法

* has_sv_change 用有效组数 > 1 判定；初始 SV 组恒存在，单 SV 谱面不算变化。
* volume / sample set 的 *_frac 是原始值（0 = inherit）；vol_resolved_* / ss_resolved_* 是解析后的值。
* hitsound 原始值 0 与 1 都表示普通命中；whistle/finish/clap 是 bit 1/2/3。
* n_effects_noncanonical = kiai 为真但 effects != 1（例如 9 = kiai + omit barline）；现有 dcr/cm3p_offline.py 的字符串相等判断会漏掉这些。
