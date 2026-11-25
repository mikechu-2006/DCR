# D2V 结构化属性输入：实现与消融计划（2026-11-24）

> 状态：**计划文档**。本机已完成只读数据审计、通道原型与划分可行性验证；
> 没有执行任何 GPU 训练，没有改动 D2V 训练代码与既有权重/内容表。
>
> 目标读者：后续实现者（HPC 执行）与评审。所有“已核实”的数字都给了可复现命令；
> “未验证”的设计选择都标了 gate。
>
> 相关：根 README；data2vec_main/README.md；docs/p4_multiseed_canon.md；
> docs/e1_behavior_supervised_encoder.md；docs/d2v_residual_results.md；
> docs/plans/2026-11-24_behavior_supervision_plan.md。
>
> 配套材料（本次生成）：
> * 审计脚本 <code>data2vec_main/scripts/audit_attributes.py</code>，汇总
>   <code>results/d2v_attr_audit/summary.json</code> / <code>summary.md</code>（逐图表在
>   <code>data/processed/d2v_attr_audit_per_chart.parquet</code>，不入库）。
> * 通道原型 <code>docs/plans/2026-11-24_d2v_attr_evidence/proto_attr_roll.py</code>（Benchmark 与数值 sanity）。
> * 代码触点清单 <code>docs/plans/2026-11-24_d2v_attr_evidence/code_touchpoints.md</code>（file:line 级）。
> * 划分可行性验证（本地，输出在 /tmp）：mapset-disjoint split + P2/P4 生成命令见 §5.3、附录 D。

---

## 0. 执行摘要

### 0.1 一句话

不动 backbone（12L/512d）、不改 512 维内容向量契约、不动既有 8ch 基线产物的前提下，
在现有连续 piano roll 上按 **release / tempo-SV-kiai / hitsound-volume 三组** 追加结构化属性通道，
先用“现有 holdout 的 chart-clean 续训”做**快速归因**，再用**新的 mapset-disjoint 划分**做
**决定性检验**；主指标仍是未见谱面上的成对 rank-NLL。成功判据是**相对 8ch（同训练预算）的配对增益**，
不是打败 CM3P legacy。

### 0.2 六个判断（带已核实数字）

1. **canon 39/40 不能外推。** canon 对照唯一变量是内容表（同 checkpoint 配置、同 P2/P4 协议），
   删掉的是 CM3P checkpoint 已经学过的输入信号；D2V 是另一种表示和预训练目标，没有离散 tokenizer。
   本计划把它当“这些信息可能对难度序有用”的**存在性线索**，不当 D2V 的预期效应。出处：
   docs/p4_multiseed_canon.md（§0/§2，ΔNLL +0.0047…+0.0208，40 个 seed×方法组合 39 个为正）。

2. **属性信息在语料里真实存在且量级不小。** 全量审计 21,949 张图 / 33,016,586 音符 / 5,939,098 个 hold /
   913.9 小时：kiai 覆盖 **76.1%** 的图、平均占谱面时长 **44.1%**；非普通 hitsound **45.4%** 的图；
   逐音符 volume 覆写 **42.3%** 的图；有效 SV 变化只出现在 **22.7%** 的图里，但一旦出现就很密
   （4,982 张图合计 **116 万**个变化组，这些图 p50=14、p95=1,211/图）；100% 的图有红线 BPM，
   全库 **281,258** 条红线（5,821 张图 >1 条）、**259,309** 处 BPM 变化（n_red p95=20.6）。数值见 §2。

3. **最大威胁不是“加不进去”，而是“学到同曲/同 mapper 的风格”。** 90.6% 的图处在多难度 beatmapset 内
   （6,857 sets，p50=3 张，max=35）；同一 set 内不同难度共享：kiai 数 **97.1%**、有效 SV 变化数 **87.3%**、
   非普通 hitsound 标记 **99.4%**、完整 timing 签名 **85.3%**（38,774 个 set 内难度对）。
   现 D2V 的 1,365 张“干净”eval 图里，**81.7%** 与预训练池共用 beatmapset。
   → **必须新增 mapset-disjoint 划分**（本地已验证可行，§5.3），否则 SV/kiai 组的任何增益都不可解释。

4. **训练侧改动边界清楚。** 通道数只进入 3 个硬点（charts.py 的 N_CHANNELS/HOLD_OFFSET、
   model.py 的 Conv1d(in_channels)、config 的 in_channels）；EMAState 只覆盖 Transformer（不含 conv），
   **扩容不需动 EMA**；conv1 权重 [256,8,5] 零初始化扩列后，step 0 与旧模型**逐位等价**（函数保持），
   新列从第 1 步开始获得非零梯度。主要 plumbing 是 6 条 state_dict 载入路径、optimizer moments 重建、
   以及 train.py 的 total_steps/LR schedule 语义（详见 §4 与触点清单 §10）。

5. **评价功效是瓶颈，而且已被本地验证可以解决。** 现 clean 口径 P4-hold 只有 1,365 图、
   每 seed 约 3.2 万次比较（strict-pairs），P4-hold 的 MDE 约 **0.012**（不能否定小效应）；P4 主口径的 MDE 约 **0.005**。
   本地已生成并验证 mapset-disjoint 划分：**1,716 个 set / 5,236 张 pairbest eval 图（25.0%）/
   16,487 张预训练图**，eval/pretrain（仅 pairbest 图）star 均值 3.385/3.393 对齐；
   probe 侧用现成 set-P4 脚本做 chart 整组 60/40，每 seed **75k–80k 测试行 / 1.03M–1.19M 次比较**
   （本地已验证 group_overlap=0）；10 seeds 下可检测约 **0.003**。

6. **两个反向证据必须写进前提。** E1（docs/e1_behavior_supervised_encoder.md §6.3）显示遮蔽预训练
   只提供约 0.0047 的边际（预注册判据 inconclusive）；D2V 残差报告（docs/d2v_residual_results.md §5）
   明确“不建议直接投入完整四列编码器”。因此本计划定位为**输入属性归因实验**：
   即使全零结果也有信息（说明这些属性不能通过该自监督目标进入表示），并为下一步（行为监督/辅助目标）
   提供判据，而不是承诺“新 D2V 一定更强”。

### 0.3 不做什么（硬边界）

* 不做离散 tokenizer、不做事件序列模型（Phase 2，条件见 §3.1）。
* 不改 dim=512 / 12 层 / 8 头 / window_sec=16 / frame_hz=100 / 窗口 1600 帧 / 池化方式
  （8ch 模型的输出契约与所有 probe 的可比性靠这些）。
* **不动** <code>configs/audio_style_base.yaml</code>、<code>data2vec_main/runs/main_holdout/</code>、
  <code>splits/holdout/</code>、<code>data/processed/chart_content_d2v_holdout.parquet</code>、
  <code>deploy/models/d2v_linear_extra.json</code>——旧 ckpt 的 resume/score/E1/E2/benchmark/deploy 复现依赖它们
  （触点清单 §13）。
* 不修 <code>dcr/cm3p_offline.py</code> 的 CM3P 语义（它的 kiai 字符串相等是“对上游 token 流的转录模型”，
  改了会影响已有 CM3P 内容表的可复现性）；新前端**另写**属性解析，并把差异记录下来（§2.3）。
* 本阶段不重跑 E1/E2、VSRG benchmark、deploy 表（作为可选后续，见 §6.3）。

### 0.4 里程碑与门槛

| 里程碑 | 内容 | 产物 | 门槛 |
|---|---|---|---|
| M0 | 前端/模型/train/extract 改动 + 单测 + smoke（本地 CPU） | 新 config、新测试、smoke OK | 全部测试过；8ch 路径 byte-兼容 |
| M1 | 数据审计 + 通道原型 + mapset 划分可行性 | 本文 §2/§3/§5.3 | **已完成** |
| M2 | Protocol S：4 个续训 run（A0-cont/A1/A2/A3）+ P4（主）/P4-hold（旁证）探针 | 4 张内容表 + summary | G1：A0-cont 不退化；新通道权重被使用；P4 上至少一组 mean ΔNLL ≤ **−0.005** 且 5/5 同向（仅筛选，不作结论） |
| M3 | Protocol D：mapset-disjoint 划分 + 从零训练（A0′ 与赢家；预算允许则 4 臂） | 新划分 + 新内容表 + set-P4 10 seeds | G2：set-P4 上 mean ΔNLL ≤ −0.005（5 seeds, 5/5）且 10 seeds 复核 ≤ −0.003（≥8/10）；shuffle/metadata/mapper 对照通过 |
| M4 | 报告：组间归因、负对照、泄漏结论、下一步判据 | docs 更新 + results | 预注册判据逐条给结论（支持/不确定/失败） |

### 0.5 算力总表（A800 基准，实测 9.114 step/s、47m02s/25,720 step）

| 阶段 | GPU | 说明 |
|---|---|---|
| Protocol S 续训 5 epoch × 4 臂 | 约 48 min | steps/epoch=1,286（20,584 图），6,430 step/臂 ≈ 12 min |
| Protocol S 抽取（6,828 张 pairbest 图） | 约 32 min（4 臂） | 218,525 窗=27m19s → 约 68k 窗/臂 ≈ 8 min |
| Protocol D 从零 20 epoch（默认 2 臂；4 臂约 2.5 h） | 约 1.3 h | 16,487 图 → 1,031 step/epoch，20,620 step/臂 ≈ 38 min |
| Protocol D 抽取（eval 池 5,462 张） | 约 12–24 min（2–4 臂） | 约 55k 窗/臂 ≈ 6 min |
| Probe（CPU） | 约 4–7 h | P4/P4-hold/set-P4 × seeds × linear/linear_extra；可拆 CPU job |
| 合计（默认路径） | **约 2.6 h GPU + 5–8 h CPU** | 不含 HPC 排队；D 做满 4 臂则约 3.8 h GPU；队列历史中位等待约 3.5 h |

Protocol S/D 分开提交（Gate 决定是否花 D 的 GPU）。

### 0.6 执行状态（2026-11-25 更新）

1-seed pilot 已执行完毕（Protocol S 4 臂 + Protocol D 4 臂；**shuffle 负对照因 HPC 账号提交额度被阻塞，未完成**），
完整结果与踩坑见 [d2v_attr_results.md](../d2v_attr_results.md)。要点：

* **D 协议（mapset-disjoint，从零 20 epoch）**：a20d（+tempo/beat/bar/SV/kiai）
  在 5,204 张未见歌曲 eval 图上 **ΔNLL = −0.0246（linear）/ −0.0292（+ext）**（单 seed，
  约为 5-seed MDE 的 5–6 倍）；a27d 次之（−0.0093 / −0.0189）；a12d 基本为零。
* **S 协议（8ch ckpt 续训 5 epoch）**：a20 方向一致但幅度小（P2 +ext −0.0055、
  P4 −0.0017、P4-hold −0.0014），单 seed 噪声内；a27 在 P4-hold 变差。
* **组内拆分（2026-11-25 补跑）**：a20d − SV/kiai（a17d，17ch）显示 SV+kiai 承担几乎全部增益
  （相对 a20d +0.0233/+0.0238），tempo/beat/bar 单独近零；a17d 的 masked loss 反而更低
  → 自监督 loss 不能作为难度有用性的代理。单 seed，无 CI。
* **C3/C4**：属性通道确实被编码（SV/kiai/BPM 可解码性 0.07–0.16 → 0.67–0.88），
  但 mapper 指纹同步上升 2.7–5pt（meta 臂最高），与“风格元数据”先验一致。
* **仍缺**：5 split seeds 的 CI、shuffle 负对照结果（HPC 账号触发 AssocGrpSubmitJobsLimit：队列空、连 trivial job 也无法提交，且 job 被外部 scancel，需要账号或管理员处理；重试链 scripts/hpc_attr_shuffle_retry.sh 待额度恢复）、mapper-disjoint 复核、组内拆分。


---

## 1. 背景与如何读既有证据

### 1.1 三个既有结果

**（a）legacy/canon（CM3P）。** 同 checkpoint 配置、同 P2/P4、同 5 个划分种子，唯一变量是内容表：
canon 裁掉 hitsound/volume、kiai、mania SV 后，P2 ΔNLL +0.0047…+0.0197、P4 +0.0064…+0.0208，
40 个 seed×方法组合 39 个变差，训练 NLL 也同向变差（不是过拟合）。这说明**这些 token 对 CM3P 的难度序
有用且与 msd/Sunny 互补**。但它检验的是“删掉已学输入”的敏感度，不能推出“D2V 加回会变好”。

**（b）D2V 现状（8ch，from-scratch，holdout 20,584 图预训练）。**

| 口径 | D2V 线性 | D2V + external 线性 | 最强 CM3P 基线 |
|---|---:|---:|---:|
| P2 全 6,828 图 | 0.5863 | 0.5810 | legacy MLP+ext 0.56602 |
| P4 全 5 seeds | 0.59301 ± 0.00417 | 0.58561 ± 0.00459 | legacy MLP+ext 0.57343 ± 0.00433 |
| P4-hold（1,365 图，编码器未见） | 0.60471 ± 0.00994 | 0.60280 ± 0.01215 | （未在同一口径下对照 legacy MLP+ext） |
| VSRG Dan MAE | — | D2V-L 0.4527 | RM-L 0.4216；Sunny 0.3367 |

读法：D2V 8ch 只稳定赢 canon+linear，**没有赢最强基线**。所以本计划的对照基数是 8ch D2V 自己
（同预算、同划分、配对 Δ），CM3P 数字只作外部参照。

**（c）两个反向证据。** E1：unfrozen（遮蔽预训练初始化）比 random 低 0.004739，未达预注册 −0.005，
判 inconclusive；D2V 残差：在强基线上有 P4 干净口径的残差增益（MLP 基线 5/5 CI<0），但
“不足以支持完整四列编码器”，下一篇应做小规模列感知探针。这两条决定了本计划必须自带**停止规则**，
不能把“加属性”当成默认正确方向。

### 1.2 三个要回答的问题（预注册）

| 编号 | 问题 | 主臂对照 |
|---|---|---|
| Q1 | hold release 脉冲是否在 8ch 之外提供新信息（长押结束判定）？ | A1 − A0-cont |
| Q2 | BPM/beat phase/有效 SV/SV 变化点/kiai 是否改变可学结构，而不只是风格？ | A2 − A1 |
| Q3 | hitsound/sample set/volume 是难度信号还是 mapper/风格？ | A3 − A2 |

Q3 的预期偏向负结果（诊断臂）：若在 mapset-clean 划分上无增益，就关闭“legacy 音效元数据进内容编码器”
这个方向，不再追加事件序列版本。

---

## 2. 数据审计（已完成，可复现）

脚本：<code>data2vec_main/scripts/audit_attributes.py</code>；全量 21,949 图单进程 254.5 s。
汇总：<code>results/d2v_attr_audit/summary.md</code>；逐图表 parquet 不入库。

### 2.1 语料与覆盖率

| 项 | 值 |
|---|---|
| 图 / 音符 / hold / 时长 | 21,949 / 33,016,586 / 5,939,098 / 913.9 h |
| 解析失败 / 非 4K mania | 0 / 0（mode=3、CircleSize=4 全部） |
| 任意 kiai | **76.1%** 的图；平均 kiai 时长占比 44.1%（p50 0.449，p95 0.890） |
| 任意有效 SV 变化 | **22.7%** 的图（4,982 张）；全库 1,162,528 个有效变化组；有变化的图 p50=14、p95=1,211/图 |
| 非普通 hitsound | **45.4%** 的图；音符级 whistle 11.8% / finish 2.7% / clap 7.7% |
| 逐音符 sample set 覆写 | 7.4% 的图；解析继承后 sample set 取值 >1 的图 26.6% |
| 逐音符 volume 覆写 | 42.3% 的图；解析后 volume 取值 >1 的图 58.6% |
| timing point volume 有变化 | 22.3% 的图 |
| 红线 BPM | 100% 的图；n_red p50=1、p95=20.6；n_bpm_values p95=12 |

### 2.2 取值分布（设计归一化用）

| 字段 | p05 | p50 | p95 | 备注 |
|---|---:|---:|---:|---|
| BPM | 103 | 174.5 | 250 | log2 归一化可覆盖约 30–480 |
| notes/s | 3.0 | 8.9 | 18.0 | 与难度相关（只作背景） |
| 谱面时长 (s) | 60.6 | 127.1 | 293.9 | 16 s 窗口约 10 个/图 |
| hold 时长 (ms) | 88 | 234 | 635 | p95 max 2,386；hold 占音符 18% |
| hold 时长 >1 s 的比例 | 0 | 0.015 | 0.30 | 长押是 LN 图的主要结构 |
| n_sv_changes | 1 | 1 | 81 | 高度长尾；首次组恒存在 |
| 时间占比 |sv−1|>0.05 | 0 | 0 | 0.093 | 平均 1.9% 时间偏离 1 |
| kiai 数/图 | 0 | 4 | 12 | 与 set 共享 |
| 解析后 volume | p50 50、p10 15、p90 80 | | | 0–100 |
| 解析后 sample set 取值数 | 1 | 1 | 2 | mean 1.30 |
| 估计帧占比：release 脉冲 | 0 | 0.062 | 0.406 | 平均 0.112 |
| 估计帧占比：非普通 hitsound 脉冲 | 0 | 0 | 0.347 | 平均 0.104 |
| 估计帧占比：SV 变化脉冲 | 0.0003 | 0.0006 | 0.040 | 平均 0.017 |

**设计含义**：kiai、BPM、beat phase、SV 是“慢变量/稠密”通道；release、hitsound、volume 是“稀疏脉冲”通道；
两类的注入方式与归一化必须分开处理（§3.2–3.4）。SV 只在 22.7% 的图里非平凡，因此 A2 的增益如果出现，
应该主要来自有 SV 变化的子集（§5.4 分层诊断）。

### 2.3 解析语义与三处缺口（必须先修/绕开）

1. **共享 parser 不暴露这些字段。** <code>dcr/cm3p_offline.py</code> 的 HitObject 只有 time/kind/x/end，
   TimingPoint 只有 offset/mpb/meter/uninherited/kiai；hitsound、sample set、volume（timing point 与逐音符）
   都在原始行里但被丢弃。逐音符字段位置：<code>hitSound=field4</code>；
   <code>hold 的 field5 = endTime:normalSet:additionSet:index:volume:filename</code>，
   <code>circle 的 field5 = hitSample</code>；timing point 的 <code>sampleSet=field3, volume=field5, effects=field7</code>。
2. **kiai 的 effects 位掩码 bug（量化）。** 现有转录用 <code>f[7] in ("1","1.0","true","True")</code>，
   而 .osu 的 effects 是位掩码（bit0=kiai，bit3=omit first barline）。审计发现 **145 张图 / 271,309 个音符**
   的 kiai 发生在 <code>effects != 1</code> 的 timing point 上（如 effects=9）；抽样统计 effects 取值
   {0:179,828, 1:34,324, 8:11,091, 9:2,369}，即 **1.04% 的 TP 行**。新前端必须用 <code>effects & 1</code>，
   且不能改共享 parser（会改 CM3P token 流）。
3. **“0 = 继承”两条规则 + 边界值。** 逐音符 volume=0 表示继承 TP volume（**92.0% 的音符**是 0）；
   sample set=0 表示 auto/继承（**97.7% 的音符**是 0）；另有 375 张图存在 TP volume=0。
   上游 CM3P 的解析给出了既有约定：<code>_get_hitsounds</code> 用 <code>t+5 ms</code> 查询 TP、
   volume 取“音符值≠0 否则 TP 值”、hitsound 取 <code>&amp; 14</code>（只留 whistle/finish/clap）。
   新前端默认对齐这些约定（便于和 legacy 语义对照），唯一有意偏离：上游把 TP sampleSet=0 当 soft(2)，
   新前端保留 0 作为 auto，并用独立 <code>ss_auto</code> 通道表示（§3.3）。
4. 3 张图 hold end ≤ start（<code>bad_hold_end</code>）；release 通道需 clamp。

### 2.4 泄漏证据（本计划最重要的数字）

| 证据 | 值 |
|---|---|
| 多难度 beatmapset 内的图 | 19,876 / 21,949 = **90.6%**（p50=3、p95=6、max=35 张/set） |
| 全部 set 内难度对（38,774）共享 | BPM 网格 **81.4%**、有效 SV 变化数 **87.3%**、kiai 数 **97.1%**、hitsound 标记 **99.4%**、完整 timing 签名 **85.3%** |
| 现 D2V “干净” eval（1,365 图 / 1,146 set）中 set 也出现在预训练池 | 901 / 1,146 = **78.6%** |
| 对应 eval 图占全部 eval 图 | 1,115 / 1,365 = **81.7%** |
| eval / pretrain 的 mapper 数 | 519 / 1,372（同 mapper 的其它谱面仍在预训练池） |

**结论**：现协议是 chart-disjoint，不是 mapset-disjoint。对 8ch（只编码音符结构）影响有限，
但对 **timing/SV/kiai/meta 组是致命的**：模型可以在预训练里从同 set 的另一难度学到该 set 的 timing 风格，
P2/P4 的“新图”并不新。Protocol S 的结果只能标注为 “chart-clean, mapset-leaky”；
决定性结论必须用 §5.3 的 mapset-disjoint 划分。

### 2.5 对设计的 5 条硬约束

1. 新前端**另写**属性解析（不改 <code>dcr/cm3p_offline.py</code> 语义），用 <code>effects &amp; 1</code>，
   对 volume/sample set 做继承解析，复刻上游 <code>t+5ms</code> 约定并在测试里锁死。
2. 属性通道**不得改变 n_frames**（仍由 note end 派生）→ index cache、窗口网格、218,525 窗口基线全不变
   （触点清单契约 C4）。
3. base 8 通道的顺序与量纲不变（0–3 onset，4–7 hold），新通道只能追加在 8.. 之后（零初始化扩容等价的
   前提，契约 C3）。
4. 所有属性归一化用**全局固定变换**，不做 per-chart z-score（否则把“绝对 BPM/SV/时长”这些可能有用、
   且同 set 共享的信息洗掉，也不符合预训练时 on-the-fly 生成的条件）。
5. 每个属性字段必须能回答“解析后值域是多少、窗口边缘怎么处理、异常的 0/None 怎么处置”，
   写进测试；不允许“先加进去再看”。

---

## 3. 输入设计：连续 piano roll + 结构化属性通道

### 3.1 为什么第一轮不做 tokenizer（以及它和你提的 typed embedding 的关系）

你提的设计是：类别字段用 embedding、连续字段用归一化数值投影、时间同时保留毫秒差与相对拍位。
在**帧栅格**上，这个设计有一个几乎等价的、改动最小实现：

| typed embedding 里的成分 | 帧栅格通道里的实现 | 说明 |
|---|---|---|
| 类别字段（tap/hold、hitsound 类型、sample set） | 每个类别一条**独立脉冲通道**（one-hot 在时间轴上） | conv1 对每个通道有独立权重，等价于“每类一个可学习投影”；不需要稀疏词表 |
| 连续字段（volume、SV、BPM、hold 时长） | 一条**归一化数值通道**（脉冲幅度或分段常值） | 线性投影 + 非线性由 conv/GELU 提供 |
| 绝对时间 | 帧索引（10 ms 分辨率） | 窗口内绝对时间天然保留；你提的“毫秒差”在帧栅格里就是帧差 |
| 相对拍位 | sin/cos 拍相位通道 | 显式、连续、无 wraparound 跳变 |
| 和弦列集合 | 4 列 onset/hold 通道（已是集合语义） | 不依赖任意列顺序 |
| 字段间关系 | conv 在时间轴上混合同一帧的多通道 | 第一轮不做显式特征交叉（留给事件序列版） |

**为什么第一轮不换事件序列 tokenizer**：(a) 现有 8ch ckpt、extract 契约（512 维、窗口池化）、
所有 probe 都是帧栅格；(b) 事件序列会同时改变 tokenizer + 预训练目标 + 池化，**不能做单变量归因**；
(c) 本计划要回答的是“哪些属性组有信息”，帧栅格通道足够，且与现有 D2V 的可比性最好。
事件序列版（每个事件：时间、列/列集合、note 类型、hold 时长、属性；同一时刻和弦按集合；
时间同时带 ms 差与 beat 位置）作为 **Phase 2**，触发条件见 §6.2。

### 3.2 通道规范（A0–A3 四臂）

**A0 = 现有 8ch**：onset_0..3（sigma=8 ms 高斯，clip 0..4）+ hold_0..3（0/1）。

| 组 | 通道 | 条数 | 编码 | 值域（归一化后） | 依据 |
|---|---|---:|---|---|---|
| **R release** | release_0..3 | 4 | hold end 帧的 per-column 高斯脉冲（同 kernel） | [0, 1]（clip） | hold 5.94M、p95 时长 635 ms；现有 hold 通道只给“持续”，release 给“结束事件” |
| **T tempo** | bpm | 1 | 分段常值 log2(BPM/120)/2 | [−1, 1]（clip） | n_red p95=20.6；全库 281,258 条红线 / 259,309 处 BPM 变化；显式 tempo 是音乐序列模型的常规输入 |
| **T tempo** | beat_sin, beat_cos | 2 | 当前红线内 beats=(t−off)/mpb 的 sin/cos(2π·beats) | [−1, 1] | 相对拍位；sin/cos 避免锯齿相位跳变 |
| **T tempo** | bar_sin, bar_cos | 2 | 当前红线内 (beats/meter) mod 1 的 sin/cos（小节相位；REMI/Octuple 的 bar 层级） | [−1, 1] | n_red p95=20.6，meter 来自红线；相位层级有音乐序列模型证据 |
| **T tempo** | sv | 1 | CM3P 归一化有效 SV 的 log2/2 | [−1.5, 1.5]（clip） | 与 legacy SVG 语义一致（sv_groups normalized=True） |
| **T tempo** | sv_change | 1 | 有效 SV 组起点的单帧脉冲 | {0,1} | 变化点本身是编谱事件 |
| **K kiai** | kiai | 1 | 分段常值 0/1 | {0,1} | 76.1% 图、44% 时长；可能反映局部强度/编谱惯例 |
| **H meta** | hs_whistle, hs_finish, hs_clap | 3 | 音符 onset 帧脉冲（hitsound &14） | {0,1} | 45.4% 图有非普通 hitsound；与 legacy 语义一致 |
| **H meta** | ss_soft, ss_drum, ss_auto | 3 | onset 帧脉冲（继承解析后；auto=原始 0） | {0,1} | 26.6% 图解析后 set 取值 >1 |
| **H meta** | volume | 1 | onset 帧脉冲，幅度=继承解析后 volume/100 | [0, 1] | 42.3% 图有逐音符覆写；58.6% 图解析后 volume 取值 >1 |

**臂**：A0=8；A1=8+4=**12**；A2=12+8=**20**（tempo 1+beat 2+bar 2+SV 2+kiai 1）；A3=20+7=**27**。
可选变体（不进主网格）：per-column meta（+3×4 通道）只在 A3 出现信号且归因需要时做；
hold duration（+4，A1d）只在 release 有信号后做消融。

### 3.3 精确定义（实现按此写，测试按此断言）

**记号**：frame_hz=100，window=1600 帧；note 时间 t 转帧 f=round(t·100/1000)；
hold 的 end 是 exclusive（现有行为），release 脉冲打在 f_end=round(end·100/1000)，高斯核 sigma=8 ms、半径 3 帧。

1. **base 8**：完全沿用现有 <code>build_window</code> 代码（不改语义）。
2. **release_r（r=0..3）**：对每个 hold，把高斯核（幅度 1）加到 release_r 的 f_end 邻域；
   多条 hold 结束同帧时相加，最后 <code>np.clip(..., 0, 1)</code>。窗口外扩展：事件在
   [start−radius, start+window+radius) 内都要参与（与现有 onset 的高斯写入一致），避免窗口边缘漏脉冲。
3. **bpm**：取生效红线（offset ≤ t 的最后一条；t 早于首红线时 clamp 到首红线，与 CM3P 一致）；
   <code>ch = clip(log2(60000/mpb / 120) / 2, −1, 1)</code>。
4. **beat_sin/cos 与 bar_sin/cos**：同一红线段内 <code>beats=(t−off)/mpb</code>，
   beat 相位 = <code>sin/cos(2π·beats)</code>；小节相位 = <code>sin/cos(2π·(beats/meter))</code>
   （meter 取生效红线的 meter，最小 1）。红线切换时相位按新网格重置（osu 语义）；
   首红线前 clamp 到首红线（相位为负但连续可导）。
5. **sv**：用 CM3P 定义的有效归一化滚动速度（<code>sv_groups(..., normalized=True)</code>）：
   红线→1.0；非红线→<code>clip(−100/mpb, 0.01, 10)</code>，再乘 <code>median_mpb / 当前红线 mpb</code>；
   <code>ch = clip(log2(sv)/2, −1.5, 1.5)</code>。帧到组的映射用 searchsorted（分段常值）。
6. **sv_change**：每个有效 SV 组的起点帧脉冲 1.0（组定义与 sv_groups 完全一致，含“值未变化则合并”规则）。
7. **kiai**：timing point 的 <code>effects &amp; 1</code> 组成切换点序列，按“最后一条 offset ≤ t 的 TP 的 kiai 状态”
   广播到帧。首 TP 前为 0。
8. **hitsound**：<code>hs &amp; 14</code>；whistle=bit1、finish=bit2、clap=bit3（bit0 normal 不设通道，
   因为普通命中已由 onset 通道表达）。
9. **sample set**：逐音符 raw s；<code>s=0</code> 时继承 HP（hitsound point）的 sample set；
   HP 也 0 时保持 0（auto），并同时置 <code>ss_auto=1</code>。解析后 1=normal、2=soft、3=drum；
   通道只设 soft(2)/drum(3)/auto，normal 用全零表示。
   **与上游差异**：上游 <code>_get_hitsounds</code> 把 TP sampleSet=0 当 soft(2)；我们保留 auto 语义，
   并在报告里注明（若需要严格 legacy 对照可加开关 <code>meta.legacy_soft_default: true</code>）。
10. **volume**：逐音符 raw v；<code>v=0</code> 时取 HP 的 volume（HP 查询时刻用 <code>t+5 ms</code>，复刻上游）；
    <code>clip(v, 0, 100) / 100</code>，在 onset 帧写单帧幅度。
    （单帧而不是高斯：逐音符属性是事件属性，不需要扩散；和弦同帧多音时取最大值。）
11. **HP 查询**：生效 TP = offset ≤ t+5 ms 的最后一条 timing point（**所有** TP，包括非红线），
    与上游 <code>hitsound_point_at</code> 一致。
12. **异常**：hold end ≤ start（3 张图）clamp 到 start+1 帧；无红线图（审计为 0 张）bpm/beat 全零，断言记录；
    非 mania/非 4K 直接跳过（ChartError 已有）。

### 3.4 归一化与量纲原则

* 只做全局固定变换（log2、除以 100、sin/cos、clip）；**不做** per-chart 标准化、不做 per-window 归一化。
* 慢变量（bpm/sv/kiai/beat）保持分段常值，不要人为平滑（卷积会处理）。
* 脉冲通道幅度 ≤1，与 base hold 通道同量级；onset 通道的 clip 0..4 只作用于 base 前 4 通道
  （现有 <code>roll[:4]</code> 是隐式契约，触点清单 C3）。
* 稀疏组（release/hs/volume）不做过采样/不加重（避免改变自监督目标的样本分布）；稀疏性诊断进报告。
* 不把“同 set 共享”的字段做去均值（那会人为削弱泄漏，也会让 Protocol D 的检验失去意义）。

### 3.5 性能与内存（实测 + 预估）

本地只读实测（触点清单 §9，200 图）：

| 项 | 值 |
|---|---|
| parse_chart | median 9.16 ms、mean 11.89 ms（README 约 12 ms/图） |
| build_window(1600 帧) | median 0.36 ms、mean 0.44 ms；平均 8.96 窗/图 |
| +release+2 impulse 原型（20ch 循环实现） | 0.399 → 0.549 ms/窗（**+37%**） |
| per-frame 属性向量化（searchsorted + take） | 25.7 µs/属性/1600 帧；共享一次 search ≈ 30–40 µs/窗 |
| conv1 FLOPs（8→27ch） | +38.9 MMAC/样本，约占整模型前向 39 GMAC 的 **+0.1%** |
| 输入张量（batch16） | 0.8 MB → 2.8 MB/step（27ch），H2D 影响 <1% |

本计划的通道原型（未优化，python 循环逐 event，且重复读文件）实测：base 0.70 ms/窗、
属性 1.46 ms/窗（release+meta+timing），parse 13.51 ms/窗（双读）。
生产实现要求：**单次 parse 返回全部原始属性** + release/meta 用按时间排序数组 + bisect 取窗口内事件
+ timing 用向量化 searchsorted；目标属性开销 ≤0.4 ms/窗。训练样本总成本仍由 parse 主导（约 +1–3%），
抽取数据准备约 +10–15%；这两个都在 DataLoader 预取预算内（基线 6 workers 已验证 9.1 step/s）。

内存：LRU 缓存的事件对象（NoteEvent）加 3 个 int 后每图约 90 KB → 512 项约 46 MB/worker，可接受；
**禁止**把 100 Hz 稠密属性数组放进 LRU（10 attr × 13.5k 帧 × 4B ≈ 0.54 MB/图 → 512 项 ≈ 276 MB/worker）。

### 3.6 spec 版本与通道数

| 臂 | 通道 | 组 |
|---|---:|---|
| A0 / A0-cont | 8 | base |
| A1 | 12 | base + release |
| A2 | 20 | base + release + tempo(含 bar 相位)/SV/kiai |
| A3 | 27 | base + release + tempo(含 bar 相位)/SV/kiai + meta |

spec 用 config 中的有序组名 + 每组的开关字典表示（附录 A），派生出通道名列表和
<code>model.in_channels</code>；spec 的 hash 写入 ckpt 的 cfg、内容表 <code>source</code> tag
（如 <code>d2v_a20_v1</code>）与 index cache 的 meta（§4.6）。

---

## 4. 训练与实现

### 4.1 ChannelSpec、配置与单一真源

新增 <code>data2vec_main/d2v_main/attributes.py</code>（前端属性解析与通道构建）与
<code>configs/audio_style_a12.yaml</code> / <code>audio_style_a20.yaml</code> /
<code>audio_style_a27.yaml</code>（**不动** audio_style_base.yaml）。

~~~yaml
# configs/audio_style_a20.yaml（示意；完整见附录 A）
data:
  osu_dir: data/osu_cache
  window_sec: 16.0
  frame_hz: 100
  gaussian_sigma_ms: 8.0
  min_valid_sec: 1.0
  channels:                        # 有序 spec：唯一真源
    version: attr_v1               # spec hash 的种子，写进 ckpt.source 与 index cache meta
    groups:
      base: {}                     # 恒 8 通道（0-3 onset, 4-7 hold）
      release: {kernel: gaussian}  # +4
      tempo: {bpm: true, beat_phase: sin_cos, bar_phase: sin_cos}   # +5
      sv: {level: true, change_pulse: true, normalized_by_median_mpb: true}  # +2
      kiai: {}                     # +1
model:
  # in_channels 不手填：由 data.channels 派生，启动时 assert 两者一致
train: {...}
~~~

* 派生函数 <code>channel_spec(cfg) -> ChannelSpec</code>：返回有序通道名、分组到通道的切片、
  以及 spec 的短 hash（如 <code>a20-3f9c</code>）。
* <code>model.in_channels</code> 从 spec 派生；train/extract/E1 入口 assert
  <code>x.shape[1] == model.conv.conv1.in_channels == spec.n_channels</code>
  （现在完全没有这个 assert，触点清单 §10.1）。
* spec hash 写入：ckpt 的 <code>cfg</code>、内容表 <code>source</code> tag（如
  <code>d2v_a20_v1</code>）、index cache 的 <code>_meta</code>（见 4.6）。

### 4.2 前端改动（charts.py / data.py / 新模块）

行号引用触点清单 §2–§3；这里只给实现要点、签名与不变量。

1. **<code>attributes.py</code>（新）**：
   <code>parse_attributes(path) -> ChartAttrs</code>，单次读文件、一次返回：
   * 逐音符：<code>(t, col, end, hitsound, sample_set_raw, volume_raw, sample_index)</code>；
   * timing：<code>red_offsets, red_mpb, tp_offsets, tp_volume, tp_sample_set, kiai_switches</code>；
   * 有效 SV 组：<code>sv_offsets, sv_values</code>（用与 <code>dcr.cm3p_offline.sv_groups</code>
     相同的归一化规则实现，但只算一次、不存 token）；
   * 断言：时间有序、mpb>0、hold end>=start（clamp）。
   <code>build_attribute_window(attrs, start, window_frames, frame_hz, spec) -> (C_attr, W) float32</code>：
   release/meta 用按时间排序 + bisect 取窗口内事件；tempo/sv/kiai 用帧时间数组 + searchsorted
   向量化填充（lit-review 建议的“慢变量 = level + change pulse”实现）。
2. **<code>charts.py</code>**：<code>NoteEvent</code> 加 <code>hitsound/sample_set/volume</code>
   三个 int（<code>slots=True</code>）；<code>parse_chart</code> 返回值扩展但不改
   <code>n_frames</code> 定义（契约 C4）；<code>build_window</code> 增加可选
   <code>attrs</code>/<code>spec</code> 参数，先按现有逻辑构造 base 8 通道，再追加属性块；
   onset 的 <code>np.clip(roll[:4], 0, 4)</code> 保持只作用于前 4 通道（契约 C3）。
   <code>build_roll</code>（只有 smoke 用）同步参数化。
3. **<code>data.py</code>**：<code>ChartWindowDataset</code> 从 <code>cfg["data"]["channels"]</code>
   构造 spec 并缓存 <code>ChartAttrs</code>（LRU 512 图；**不缓存稠密 per-frame 数组**）；
   <code>__getitem__</code> 调 <code>build_window(..., spec)</code>；启动时 assert 通道数。
   collator 不改（通道在 x 里）。
4. **窗口边缘**：release 脉冲与 onset 一样需要 <code>[start−radius, start+window+radius)</code>
   的事件参与，否则窗口边缘的 hold 结束会系统性丢脉冲；测试里用“把一个 hold 的 end 放在窗口
   0/1599 帧附近”断言。
5. **性能目标**：属性构建 ≤0.4 ms/窗（原型 1.46 ms 是未优化的 python 逐 event + 双读文件）；
   训练样本总成本 +1–3%，抽取数据准备 +10–15%（触点清单 §9 的预算）。

### 4.3 模型（conv1）与 EMA/优化器

* 唯一形状变化：<code>conv.conv1.weight</code> [256, 8, 5] → [256, C, 5]；bias 不变。
  参数量增加 1,280×(C−8)：C=12 → +5,120；C=20 → +15,360；C=27 → +24,320（39.1M 的 0.04–0.06%）。
* **EMA 不动**：<code>EMAState.master</code> 只覆盖 student Transformer（smoke ckpt 实测 26 个张量、无 conv），
  conv 是 student/teacher 共享 module（model.py:100–107），扩通道后 teacher 自动看到同一输入。
* **优化器需要重建**：AdamW 参数顺序里 <code>conv.conv1.weight</code> 是 state 索引 1
  （索引 0 是根级 <code>mask_emb</code>）；扩容时把 <code>exp_avg/exp_avg_sq</code> 扩成新形状、
  前 8 列拷贝旧值、新列为 0；其它参数 state 不动。
* 不引入 per-group 1×1 conv / FiLM（第一轮保持纯 concat：零初始化扩容才是严格函数保持；
  融合方式的消融见 §6.2 可选）。

### 4.4 checkpoint 扩容与训练 schedule

* 新增 <code>d2v_main/checkpoint.py</code>：<code>expand_model_state(sd, new_in)</code> 与
  <code>expand_train_ckpt(ckpt, new_in)</code>（模型 + optimizer；EMA 跳过），
  以及 <code>infer_in_channels(ckpt)</code>（从 <code>conv.conv1.weight.shape[1]</code> 或
  <code>ckpt["cfg"]</code> 推断，供 extract / E1 / score-only 用）。
* <code>train.py</code> 新增 <code>--init-from CKPT</code> 与 <code>--expand-channels</code>：
  只加载 model state（可扩容），不加载 optimizer/step，fresh schedule；
  与现有 <code>--resume</code>（严格同构、接着原 step）分开。
* **Protocol S（快速筛选）的续训方案**：从既有 8ch <code>runs/main_holdout/checkpoints/last.pt</code> 续训
  **5 epoch**（6,430 step，约 12 min/臂），warmup 300 step，peak LR **5e-5**（0.5× base），
  cosine，clip 1.0，bf16，batch 16，workers 6；EMA 从加载后的 teacher 重新起（tau 同 base）；
  mask p/length 与 base 完全一致。理由：短续训 + 小学习率是 continued-pretraining 的稳妥做法
  （lit-review §3，DAPT/ULMFiT/ControlNet 证据）；5 epoch 是“2–5 epoch”区间的上界，给新通道学习时间。
* **A0-cont 控制臂**：同配置、同 5 epoch、8ch、<code>--init-from</code> base（不扩容）。
  它同时是“额外训练”的对照，也是“容量对照”：
  在零初始化 + 续训下，如果新通道输入全零，其权重梯度恒为 0，函数上与 A0-cont **恒等**；
  所以 A0-cont 已经覆盖了“多 1.2–2.4 万参数”的容量解释，不需要单独的零通道臂。
* **总步数语义（坑）**：train.py 的 cosine 用 <code>total_steps = steps_per_epoch × epochs</code>，
  续训时必须让 <code>--epochs</code> 表示本次新增 epoch，而不是“累计 20+5”；否则
  <code>start_step ≥ total_steps</code>，LR 立刻落到最小值。实现上给 train.py 加
  <code>--total-steps</code> 覆盖或把 <code>--init-from</code> 与 <code>--resume</code> 的
  schedule 语义分开，并在日志里打印 <code>steps_per_epoch/total_steps/warmup</code>。
* **Protocol D（决定性检验）的从零方案**：从零 20 epoch，其余超参与 base 完全一致（47min/臂的基线配方），
  以新划分的 pretrain_ids 训练。

### 4.5 训练监控与“新通道是否被使用”的健康检查

续训最危险的失败模式是 **零初始化通道一直不长**（ReZero 效应）。训练中记录并在第 1 个 epoch
做 gate：

| 指标 | 期望 | 不满足时 |
|---|---|---|
| 各属性组 conv1 新列权重 <code>mean(abs(w))</code> | 第 1 epoch 后 > 1e-3，且随 step 增长 | 启用 <code>--conv1-new-lr-mult 5</code>（只给新列参数组更高 LR）或“冻结主干只训新列 2k step”的 warm start |
| 各属性组梯度范数占 conv1 的比例 | >0 且非单调消失 | 同上 |
| masked loss：A0-cont vs A1/A2/A3 | 组间差异小（输入消融不应造成 loss 大幅变化） | 大幅变差（>0.05）说明新通道干扰收敛 → 降 LR/减通道 |
| A0-cont last vs 原 base：P4-hold NLL | 不能明显退化（> +0.003 视为遗忘） | 降 LR、减 epoch；或 Protocol D 直接从零训练 |

中间 checkpoint（step 2000/4000）只做**诊断**（loss、参数量、探针），
**主结论一律用 last checkpoint**（避免在同一个 holdout 上选 checkpoint）。

### 4.6 index cache、provenance 与不可覆盖产物

* <code>n_frames</code> 定义不变 → 现有 <code>chart_index_100.json</code> 仍有效，**不重建**；
  但 <code>index_cache._meta</code> 增加 <code>channel_spec_hash</code>（只为防止未来有人把
  timing 尾部并入 n_frames 时静默复用旧缓存）。
* 内容表契约不变：列 = <code>[beatmap_id, embedding, n_windows, checksum_ok, source, d2v_rev]</code>，
  embedding 512 维 L2≈1；<code>source</code> 必须用新 tag（如 <code>d2v_a20_v1</code>）。
  新表写新路径，**不得覆盖** <code>chart_content_d2v_holdout.parquet</code>（README/benchmark/deploy 依赖）。
* 本次改动的 8ch 兼容性要求：base yaml + 旧 ckpt 的 train resume / extract / E1 score-only /
  E1 resume / deploy parity 全部仍可复现（触点清单 §13.13）。

### 4.7 关键不确定性：自监督目标是否会“使用”新字段

这是本计划最重要的科学风险，必须写在前提里：data2vec 的损失只在 masked 帧上回归 teacher 表示，
student 的梯度经共享 conv 进入 conv1。**如果某个属性组对“预测被遮蔽的谱面内容”没有帮助，
模型可以不学它**（新列保持接近零），于是冻结内容向量里没有这些信息，probe 也看不到增益。
反过来说，本实验实际检验的是“**能被内容预测目标吸收的属性**是否携带难度信息”；这个条件比
“属性是否携带难度信息”更严格。

处理方式（不改变主臂、只加诊断与备选）：

1. 先跑**属性可解码性诊断 D2**（CPU）：用内容向量线性回归/分类 chart 级属性统计
   （kiai_frac、n_sv、BPM、hold 时长、volume 分布等）。若 A1–A3 的向量对这些属性的可解码性并不比
   A0 高，说明通道没被学进去 → 主臂的零结果不能解释为“属性无用”，而应进入备选方案。
2. **备选 A（辅助属性预测头，只做一个小实验）**：在 masked 帧上加辅助损失，预测该帧的
   kiai/SV/volume/hitsound（仅当该帧有音符）；与主臂同样 5 epoch。作用是强制编码，代价是目标函数改变，
   所以只作为 gate 失败后的 follow-up，不进主网格。
3. **备选 B（行为监督路线）**：把新通道接到 E1 的行为监督编码器（同一 <code>build_window</code>，
   梯度直接来自成对难度）。这是更可能让“难度有用但内容预测无用”的属性发挥作用的路线；
   属于 Phase 3，本计划只留接口与预算（§6.3）。

### 4.8 测试清单（M0 的 Definition of Done）

新增 <code>data2vec_main/scripts/test_attribute_channels.py</code>（纯合成 + 3 张真图，CPU）：

1. **通道数/顺序**：A0=8、A1=12、A2=20、A3=27；base 8 与现有 <code>build_window</code>
   在 <code>spec=base</code> 时逐位相同（<code>np.array_equal</code>）。
2. **零初始化等价**：用 smoke 8ch ckpt 扩容到 12/20/27 后，两种输入（新通道任意非零）下
   模型输出在 step 0 **逐位相同**；第 1 步后新列梯度非零。optimizer state 扩容后
   <code>step</code> 保留、旧列拷贝正确。
3. **release**：hold 在窗口边界（end=start−1、start、start+1、start+1599、start+1600）
   各有一条脉冲出现在正确帧；per-column 不串列。
4. **BPM/beat**：合成 2 条红线（120→240 BPM），断言 bpm 与 sin/cos 分段正确、红线处相位重置；
   首红线前 clamp。sine 在整拍处=0、cos=1（容差 1e-6）。
5. **SV**：合成 inherited timing point，与 <code>dcr.cm3p_offline.sv_groups(normalized=True)</code>
   逐组一致；change pulse 数量/位置正确；clip 值域断言。
6. **kiai**：<code>effects=9</code> 的 TP 必须产生 kiai 切换（回归测试 §2.3 的 bug）；
   effects=8 不产生。
7. **volume/sample set 继承**：note volume 0 → TP volume；TP volume 0 → 0（并记录）；
   sample set 0 → TP set；TP set 0 → auto 通道=1；t+5 ms 边界（TP 在 t+3 ms 时取新 TP）。
8. **n_frames 不变量**：同一图在 A0/A3 下 <code>n_frames</code> 与窗口网格完全一致；
   index cache 复用不重建。
9. **值域/有限性**：所有通道 finite；按组断言 min/max 在 §3.2 表范围。
10. **smoke 扩展**：<code>smoke_test.py</code> 增加 a12 全链路（train 3 step + extract），
    断言 parquet 6 列不变、norm=1；base8 smoke 保持通过。

---


---

## 5. 评价协议（两级：快速筛选 S + 决定性 D）

### 5.1 现有口径的边界与可检测效应（来自本仓库实测 sd）

| 口径 | 图数 | 编码器是否见过 | 每 seed 比较数 | sd(paired Δ) | MDE（5 seeds, 95%） | 用途 |
|---|---:|---|---:|---:|---:|---|
| P2-all | 6,828 | 是（transductive） | 1.14M | ~0.006（单 split 半宽） | ~0.006 | 辅助，不作迁移声明 |
| **P4（6×6/4×4）** | 6,828 | 是（transductive） | 2.1M | 0.0040 | **0.005** | **Protocol S 主判据**（筛选） |
| P4-hold | 1,365 | 否（chart-level） | 31.9k | 0.0095–0.010 | **0.012** | 只作干净旁证；**不能否定 <0.012 的效应** |
| set-P4（6,828 池，已有实现） | 6,828 | 是（transductive） | 18–20k 行 | 0.0037 | 0.0045 | probe 侧 set-disjoint 稳健性 |
| **D-protocol set-P4（本计划，25% set 留出）** | **5,236 eval 池** | **否（chart + mapset 都不相交）** | **75k–80k 行 / 1.03M–1.19M 比较** | 待实测（估计 0.004） | 估计 0.005（5 seeds）/ **0.003（10 seeds）** | **Protocol D 主判据** |
| randommerge | 20,388 | 是（大面积 transductive） | 2.15–2.89M | 0.0035–0.0009 | 0.0043（P4）/ **0.0011（P2）** | 高功效但非干净；D2V 数字本地缺（HPC job 14076959 pending） |
| VSRG Dan | 1,382 | — | — | — | 0.02–0.03 MAE | 外部粗门，不作小效应来源 |

规则：**同一 split、同一 pair 集、同一 ceiling、同一次 run 内做配对 Δ**；绝对 NLL 跨口径不可比
（eval-proto §5、compare_multiseed_runs.py 会断言 ceiling 逐 seed 相同）。

### 5.2 Protocol S：现有 holdout 上的快速筛选（chart-clean, mapset-leaky）

* 目的：在花 Protocol D 的 GPU 前，用现有 8ch checkpoint 做**机制归因**与“通道是否被目标吸收”的检查；
  同时验证用户提出的零初始化续训方案是否可用。
* 训练：A0-cont / A1 / A2 / A3 四个续训 run（§4.4，5 epoch，约 12 min/臂），同 seed、同数据顺序。
* 抽取：**6,828 张 pairbest 图**（P4 主口径需要；P4-hold 的 1,365 张是其子集），约 8 min/臂：
  <code>python -m d2v_main.extract --config configs/audio_style_a20.yaml --ckpt ... \
  --ids-file data/processed/charts_1k.txt --out ../data/processed/chart_content_d2v_a20-1k.parquet --tag d2v_a20_v1</code>
  （图表清单可用 <code>step1_1k_pairbest.parquet</code> 的唯一 beatmap_id 落地成 txt。）
* 探针（与历史 D2V 同一 runner）：
  * **P4（主，筛选）**：<code>run_p2_p4_multiseed.py --protocols p4 --seeds 1..5 --models linear,linear_extra --content-path ... --out-dir results/d2v_attr/a20/p4</code>；
    成本约 12.5 min/臂（linear+linear_extra 5 seeds，eval-proto 实测）。
  * P4-hold（旁证）：<code>--split-p4 data/processed/step1_1k_pairbest_p4_d2vhold</code>，约 4.5 min/臂；
    **只支持 |Δ| ≥ 0.012 的结论**。
  * 与 A0-cont 逐 seed 配对（<code>compare_multiseed_runs.py</code>，断言 ceiling 相同）。
* 依赖：HPC 上 <code>data2vec_main/runs/main_holdout/checkpoints/last.pt</code>（737 MB，仅 HPC）。
  若该 checkpoint 丢失：跳过 Protocol S（Protocol D 不依赖它），或先重跑 8ch 基线 20 epoch（47 min）。
* 判据（筛选，G1）：
  * **G1a 通道健康**：第 1 epoch 后各组新列 mean|w| > 1e-3（否则先修正 LR/warm start，见 §4.5）；
  * **G1b 不遗忘**：A0-cont 的 P4 NLL 不比原 base 差 > 0.003；
  * **G1c 信号**：任一 add-one 组在 P4 上 mean Δ ≤ **−0.005** 且 **5/5 seeds 为负**（MDE=0.005）。
    只作“值得进 D”的筛选，**不作最终结论**；P4 是 transductive，不能写“新谱面泛化”。

### 5.3 Protocol D：mapset-disjoint 划分 + 从零训练（决定性检验）

#### 5.3.1 划分设计（已本地生成并验证，seed 0）

* 以 <code>beatmap_meta_full.csv</code> 的 <code>beatmapset_id</code> 为单位（100% 覆盖 21,949 张缓存图），
  随机置乱后按图数累积到 **25%**，得到 **1,716 个 held-out sets**。
* eval 池：held-out sets 在 <code>data/osu_cache</code> 内的 **5,462 张**图
  （其中 **5,236 张**在 full pairbest 里有 pair，构成评测池；过滤后 468,702 行 / 17,751 玩家）。
* 预训练池：其余 **16,487 张**缓存图（相对旧 holdout 的 20,584 少 20%）。
* 质量检查（实测）：eval / pretrain（仅 pairbest 图）star 均值 **3.385 / 3.393**；
  pretrain 全池 star 3.304（含 1,004 张非 pairbest 缓存图，本身就偏易，与现有 D2V 预训练池同源）。
* **为什么是 25% 而不是 15%**（本地扫过的取舍）：

| held-out set 比例 | eval 图 | 预训练图 | set-P4 测试图 | 测试比较/seed（近似） |
|---:|---:|---:|---:|---:|
| 15% | 3,139 | 18,670 | ~1,150 | 0.24M–0.37M |
| 20% | 4,187 | 17,577 | ~1,490 | 0.41M |
| **25%** | **5,236** | **16,487** | **~1,930** | **1.03M–1.19M** |
| 30% | 6,281 | 15,420 | ~2,220 | 0.95M |

  25% 行是实际 set-P4 split（seeds 1–3）跑出来的数，其余比例是同一近似口径的扫描；
  25% 在“预训练数据仍够（16.5k 图）”与“测试比较数 ~1.1M/seed”之间最平衡；
  预训练池缩 20% 对**所有臂同等**，配对 Δ 不受影响；绝对 NLL 会略差，不跨协议比较。

#### 5.3.2 Probe 侧也要 set-disjoint（已有脚本，直接复用）

* 用现成 <code>scripts/step1_p4_split_sets.py</code>（Experiment 2b，chart 侧按 beatmapset 整组 60/40，
  player 侧活动度分层 60/40；已跑通并被 runner 消费）：
  ~~~bash
  python3 scripts/step1_p4_split_sets.py \
    --in data/processed/step1_mclean25_pairbest.parquet \
    --meta data/processed/beatmap_meta_full.csv \
    --out-prefix data/processed/step1_mclean25_p4sets \
    --seeds 1,2,3,4,5,6,7,8,9,10
  ~~~
  本地已验证（seeds 1–3）：chart groups 943/773，实际有行图 ~3,016–3,022 / 1,930–1,939，
  test rows 75,116–80,489，test players 6,237–6,299，**group_overlap = 0**。
* 注意：Experiment 2b 用的是 <code>step1_1k_pairbest_nogap.parquet</code> + meta_1k；
  本计划用 full pairbest + meta_full 重新生成，**绝对 NLL 与 Experiment 2b 不可比，只比 Δ**。
* 注意：checked-in 的 P4-hold split 有 provenance 缺口（来自 nogap 子集、无生成脚本、
  重跑不能 bit 复现）——所以 Protocol S 的 P4-hold 直接复用现有文件，不重新生成（eval-proto §6.5）。

#### 5.3.3 训练与评测

* D2V 从零训练 4 臂（A0′/A1′/A2′/A3′，20 epoch，pretrain_ids=16,487，约 38 min/臂）
  或预算紧时只做 **A0′ + Protocol S 的赢家**（约 1.3 h）。
* 抽取 eval 池 5,462 张（约 5–6 min/臂）→ 新内容表 <code>..._mclean.parquet</code>。
* 探针：<code>run_p2_p4_multiseed.py --protocols p4 --seeds 1..10 --models linear,linear_extra --content-path ... --split-p4 data/processed/step1_mclean25_p4sets --out-dir results/d2v_attr_d/...</code>。
* **CM3P legacy 参照**：同一套 set-P4 split 上跑 <code>chart_content_cm3p_local.parquet</code>
  （linear/linear_extra/MLP/mlp_extra），纯 CPU，用来回答“clean 口径下 D2V+属性 vs CM3P legacy”。
* 可选 mapper-disjoint 复核（只做 A0″ + 赢家，2 个 run ≈1.3 h）：按 mapper 整组留出，
  已实测可行（236 mappers / 3,361 缓存图 / 3,190 pairbest 图 / 16,851 玩家 / star 3.253 vs 3.313）；
  用于排除“同 mapper 其它谱面泄漏风格”（Mapperatorinator 分类器证据，lit-review §5）。

#### 5.3.4 判据（声明级，G2）

* **G2a（主）**：赢家 vs A0′ 的 set-P4 配对 ΔNLL（10 seeds）：mean ≤ **−0.003**、≥ **8/10** seeds 为负、
  per-seed 配对 bootstrap CI 多数排除 0；同时 P4-hold（chart-clean）方向一致（不要求幅度）。
* **G2b（强）**：5 seeds 筛选时 mean ≤ −0.005 且 5/5；10 seeds 复核 ≤ −0.003 且 ≥8/10。
* **G2c（防泄漏）**：若 Protocol S 有信号但 Protocol D 塌回 0（或增益 < 一半）→ 判为
  song/mapset 风格捷径；不进入下游，不写“属性携带难度信息”，只写诊断。
* **G2d（防伪迹）**：组内 shuffle 控制（C2）的 Δ 必须比真通道小一半以上；
  metadata-only（C1）不能显著优于 A0′；mapper-ID probe（C3）不应大幅上升。

### 5.4 指标、统计与报告

* 主指标：<code>step2_rank_eval.py</code> 的成对 BT NLL（P4/set-P4：<code>--split test --ceiling-fit all --common-maps --strict-pairs</code>）；
  <code>captured</code> 仅参考，不跨 ceiling 比较。
* 报告：**per-seed ΔNLL、mean±sd、符号计数、per-seed 配对 CI（cluster bootstrap，player/chart 两套）**、
  ceiling、test pair 数、以及**该口径实测 MDE = 1.24×sd**。5/5 符号检验 p=0.031，不足以“强确认”，
  所以 D 至少 10 seeds。
* 辅助：Pearson(d,b)/Spearman(d,b)、按 LN/SV 变化/kiai 覆盖/难度分带分层、CM3P/D2V 参照。
* 不把 seed-level std 当独立样本做 t 检验（5/10 个 split 共享同一批图与玩家）。
* 报告里必须写“变化来自哪一组、在该口径下是否超过 MDE”；低于 MDE 的一律写“不可检测”。

### 5.5 负对照与诊断（先做，纯 CPU/便宜）

| 编号 | 内容 | 何时做 | 输出 |
|---|---|---|---|
| C0 | A0-cont / A0′ | 每个协议内 | 额外训练/容量对照（零初始化下等价性见 §4.4） |
| C1 | **metadata-only 基线**：每图 12–15 个属性汇总特征（BPM、n_red、n_sv、kiai_frac、hold_frac、hs 比例、volume 统计）走同一 rank probe | **Stage 0，现在就能做**（特征来自审计 parquet） | 若单独就能解释难度序 → 通道增益是捷径 |
| C2 | **组内 shuffle**：每组通道沿时间轴随机 roll（保留组内结构与统计，破坏与音符对齐），同臂同预算 | Protocol D 的赢家确定后 | 增益是否来自对齐信息 |
| C3 | **mapper-ID probe**：512d 内容向量 → mapper_id 线性分类，比较 A0 与属性臂 | Stage 0 先测 8ch 基线；每臂出来后复测 | 属性是否让向量更像 mapper 指纹 |
| C4 | **属性可解码性**：512d → chart 级属性统计的线性回归 R²（A0 vs 属性臂） | 每臂抽取后 | 编码器是否真的学进属性（§4.7 的关键诊断） |
| C5 | 分层 | 每臂 | LN / SV 变化 / kiai 覆盖 / 难度分带上的 Δ |
| C6 | 可选：attribute-only 内容表（去掉 base8，只留属性通道，从零训练） | 若 C1/C3 提示捷径 | 属性通道单独能到多好 |

C1 与 C3 的 8ch 基线不需要任何新训练，**现在就能跑**，用来在烧 GPU 前判断风险。

### 5.6 不要引用/不要说的话

* 不引用跨口径绝对 NLL；不用 transductive 的 P2/P4 声称“新谱面泛化”。
* 不用 P4-hold（MDE 0.012）否定 <0.012 的效应；不用 P2-hold（MDE 0.015）单独下结论。
* 不把 <code>captured</code> 用于跨 ceiling 比较；randommerge 的 captured 不可引用。
* 不把 hitsound/volume/SV/kiai 的增益直接写成“难度信号”，必须先过 C1/C3 与 Protocol D。
* 不用“5/5 同向”单独作为强确认（p=0.031）。

---

## 6. 实验矩阵、预算与执行顺序

### 6.1 矩阵

| 编号 | 内容 | 初始化 | epoch | 评测 | 目的 | GPU | 抽取 | probe |
|---|---|---|---:|---|---|---:|---|---|
| **Z0** | C1 metadata-only + C3 mapper probe + C4 8ch 基线 | — | — | 现有 1k 表 | CPU 预检，先做 | — | — | <30 min |
| **S-A0** | 8ch 续训（A0-cont） | base ckpt | 5 | P4 + P4-hold | 续训/容量对照 | 12 min | 6,828 图 ≈8 min | ~15 min |
| **S-A1** | 12ch（+release） | base + zero | 5 | 同上 | Q1 | 12 min | 同上 | ~15 min |
| **S-A2** | 20ch（+tempo/beat/SV/kiai） | base + zero | 5 | 同上 | Q2 | 12 min | 同上 | ~15 min |
| **S-A3** | 27ch（+hitsound/volume） | base + zero | 5 | 同上 | Q3 | 12 min | 同上 | ~15 min |
| **D-A0′** | 8ch | from scratch | 20 | set-P4 10 seeds | D 基线 | 38 min | 5,462 图 ≈6 min | ~1–2 h |
| **D-Aw′** | 12/20/27（S 的赢家） | from scratch | 20 | 同上 | 主声明 | 38 min | 同上 | ~1–2 h |
| **D-A1′..A3′** | 其余臂（可选） | from scratch | 20 | 同上 | 完整归因 | 38 min/臂 | 同上 | 同上 |
| **D-CM3P-L** | CM3P legacy 512d 表 | 冻结 | — | 同一 set-P4 | clean 参照 | — | — | ~30 min |

合计（默认路径，D 只做 2 臂）：**GPU ≈ 48 min + 1.3 h ≈ 2.1 h**；抽取 ≈ 32 min + 12 min；
CPU ≈ 4–7 h。若 D 做满 4 臂：GPU ≈ 3.5 h。

### 6.2 可选/后续（条件触发）

| 项 | 触发 | 成本 |
|---|---|---|
| S-C2 组内 shuffle（A2r/A3r） | S 出现 ≥−0.005 信号 | 12 min/臂 |
| A1d release+duration（8 通道） | Q1 有信号、需要拆 release vs duration | 12 min |
| T 组内拆分：tempo/beat vs SV vs kiai | Q2 有信号、需要归因 | 3×12 min |
| measure/bar phase 开关 | Q2 有信号 | 12 min |
| per-group 1×1 conv 融合 | 通道 concat 有信号、想看容量 | 改模型 + 2 run |
| 辅助属性预测头（§4.7 备选 A） | 主臂 null 但 C4 显示通道未被吸收 | 1 run + 新损失 |
| mapper-disjoint 复核 | Protocol D 通过 | 2×38 min |
| randommerge 高功效 screen | S 在边界（−0.003…−0.005）且需要更高功效 | 全表抽取 + 数小时 CPU |
| VSRG benchmark / deploy / E1 新臂 | 对外发布或 Phase 3 | 见 eval-proto §2.7 |

### 6.3 执行顺序

1. **Stage 0（本地 CPU，先做）**：M0 代码 + 单测 + smoke；Z0 的 C1/C3/C4 预检（用审计 parquet +
   现有 8ch 内容表），把“属性会不会只是捷径”的风险先量化。
2. **Protocol S（HPC）**：4 个续训 job（可并发）+ 抽取 → P4/P4-hold 探针 → G1 结论。
3. **Protocol D（HPC，G1 后）**：mapset-disjoint 划分（已设计验证）+ 从零训练 + 抽取 +
   set-P4 10 seeds + CM3P-L 参照 → G2 结论。
4. 可选 mapper-disjoint / 下游。

队列：本机无 GPU；HPC 用 <code>emergency_gpu</code> / <code>i64m1tga800ue</code>（历史中位等待约 3.5 h）。
S 的 job 短（12 min 计算），可投 debug 队列；D 的 38 min/run 需正常 GPU 队列。

---

## 7. 代码改动清单（文件级）

> 行号来自 <code>docs/plans/2026-11-24_d2v_attr_evidence/code_touchpoints.md</code>（2026-11-24 工作树）。
> 规模：S ≤ 50 行、M 50–200 行、L > 200 行（含测试）。

| # | 文件 | 改动 | 规模 | 测试 |
|---|---|---|---|---|
| 1 | <code>data2vec_main/d2v_main/attributes.py</code>（新） | 单次 parse 的 <code>ChartAttrs</code>（逐音符属性 + timing 段 + 有效 SV 组）；<code>ChannelSpec</code>（组→切片/名字/hash）；<code>build_attribute_window</code>（release/meta 事件循环 + timing 向量化）；<code>effects &amp; 1</code>、volume/sample 继承、t+5 ms | L | test_attribute_channels 全部 |
| 2 | <code>data2vec_main/d2v_main/config.py</code> | DEFAULTS 加 <code>data.channels</code>；<code>channel_spec(cfg)</code> 派生 in_channels；spec hash | S | 配置单测 + 非法组合报错 |
| 3 | <code>data2vec_main/configs/audio_style_a12/a20/a27.yaml</code>（新） | 三个新配置；**base.yaml 不动** | S | smoke 跑 a12 |
| 4 | <code>data2vec_main/d2v_main/charts.py</code> | NoteEvent 加 3 字段（slots）；parse_chart 返回 attrs；build_window/build_roll 接 spec；<code>roll[:4]</code> clip 限定 base；改 slice_window docstring | M | base8 逐位一致；n_frames 不变 |
| 5 | <code>data2vec_main/d2v_main/data.py</code> | spec 传入 + ChartAttrs 缓存 + assert 通道数；不缓存稠密 per-frame | M | dataset shape/通道名 |
| 6 | <code>data2vec_main/d2v_main/model.py</code> | ConvFeatureEncoder 去掉 in_channels 默认值；启动 assert spec 一致；EMA 不动 | S | 8ch/20ch forward |
| 7 | <code>data2vec_main/d2v_main/checkpoint.py</code>（新） | <code>expand_model_state</code>/<code>expand_train_ckpt</code>/<code>infer_in_channels</code>；optimizer moment 扩容 | M | 零初始化等价性 + state 形状 |
| 8 | <code>data2vec_main/d2v_main/train.py</code> | <code>--init-from</code>（只载 model、可扩容、fresh schedule）；<code>--total-steps</code> 或修正 epochs 语义；<code>--conv1-new-lr-mult</code>；通道健康日志（每组权重范数/梯度范数） | M | 续训 3 step、loss finite、LR 曲线打印 |
| 9 | <code>data2vec_main/d2v_main/extract.py</code> | 优先从 ckpt cfg 推断 in_channels/spec；assert；新 tag；输出路径防护（拒绝覆盖旧表） | S | 8ch 旧表仍能由旧 ckpt 复现 |
| 10 | <code>data2vec_main/d2v_main/index_cache.py</code> | meta 加 <code>channel_spec_hash</code>（n_frames 不变，cache 仍复用） | S | cache 复用/失效 |
| 11 | <code>data2vec_main/d2v_main/splits.py</code> | 新 protocol <code>mapset_holdout</code>：group=beatmapset_id、frac、seed、meta_full；写 pretrain_ids（16,487）/eval_ids（5,462）/summary | M | 本地已验证 25% 划分统计 |
| 12 | <code>scripts/filter_pairbest_to_ids.py</code>（新） | 过滤 full pairbest 到 eval ids，输出 step1_mclean25_pairbest.parquet | S | 行数/图数断言 |
| 13 | <code>scripts/step1_p4_split_sets.py</code>（复用，已有） | 不改代码；用 meta_full 重新生成 10 seeds | — | group_overlap=0 |
| 14 | <code>scripts/step2_cm3p_rank.py</code>（可选） | <code>--chart-group-table</code>：P2 的 5 折 group-aware（不用则 P2 只作辅助且标注 probe-leaky） | S | 折内无 group 泄漏 |
| 15 | <code>scripts/step2_chart_encoder.py</code> | <code>load_encoder</code> 按 ckpt 推断/扩容 in_channels；score-only 兼容旧 ckpt；ckpt 加 in_channels/spec 字段 | M | 旧 8ch E1 ckpt score-only 仍可跑；新 ckpt 可 init |
| 16 | <code>scripts/e1_frozen_window_control.py</code> | 特征缓存 meta 加 spec hash（防静默复用旧特征） | S | meta 不匹配→重建 |
| 17 | <code>data2vec_main/scripts/test_attribute_channels.py</code>（新） | §4.8 全部断言（合成 + 3 真图） | L | — |
| 18 | <code>data2vec_main/scripts/smoke_test.py</code> | 参数化 8ch 断言；加 a12 全链路 smoke | M | SMOKE OK |
| 19 | <code>data2vec_main/scripts/audit_attributes.py</code>（已写） | 数据审计（已完成，见 §2） | L | 255 s 全量跑通 |
| 20 | <code>docs/</code> | M4 结果文档 + data2vec_main/README §5 更新（8ch 描述、输入限制 #11） | M | — |
| 21 | <code>scripts/build_meta_content.py</code>（新） | 从审计 per-chart 表构造 12–15 维 chart 摘要特征 → 内容表（C1 metadata-only 基线） | S | 行数/图数/有限性 |
| 22 | <code>scripts/content_mapper_probe.py</code>（新） | 内容向量 → mapper_id 线性分类（C3 mapper-ID probe），按图切分 train/test | S | 8ch 基线与随机基线 |

**明确不改**：<code>dcr/cm3p_offline.py</code>（CM3P 语义与可复现性）、
<code>deploy/*</code>（本阶段不发布）、<code>scripts/benchmark_*</code>（不在本阶段重跑）、
<code>splits/holdout/</code> 与既有内容表。

---

## 8. 风险与回退

| # | 风险 | 概率 | 影响 | 缓解/回退 |
|---|---|---|---|---|
| R1 | 增益来自 song/mapset 风格，不是难度 | 中–高 | 结论错误 | Protocol S vs D 分离；C1/C3；D 塌回 0 即判泄漏（G2c），不写结论 |
| R2 | 零初始化新通道“死掉”（不学） | 中 | 假阴性 | G1a 通道健康；conv1 新列高 LR / 冻结主干 warm start；D 从零训练彻底绕开 |
| R3 | 续训干扰/遗忘（A0-cont 退化） | 中 | S 口径无效 | G1b；降 LR/epoch；D 从零不受影响 |
| R4 | SSL 目标不吸收与内容预测无关的属性（§4.7） | 中–高 | 假阴性 | C4 属性可解码性；辅助头（备选 A）；行为监督（Phase 3） |
| R5 | 功效不足（效应 < MDE） | 高 | 无法结论 | 预注册阈值 + 报告实测 MDE；D 用 10 seeds；可选 randommerge 高功效 screen |
| R6 | 8ch base ckpt 只存在 HPC / 丢失 | 中 | S 无法做 | 先确认 HPC 文件；丢失则重跑 8ch 20 epoch（47 min）或跳过 S（D 不依赖） |
| R7 | 改坏旧 8ch 路径（6 条 load + 契约） | 中 | 复现断裂 | base.yaml 不动；checkpoint.infer_in_channels；M0 测试覆盖旧 ckpt；改动只加新文件 |
| R8 | 属性解析语义错（effects/volume/set/t+5ms） | 中 | 通道值错 | §4.8 合成测试 + 抽样 20 张图与上游规则逐音符对照；记录与上游的有意偏离 |
| R9 | 新内容表漏图 → ceiling/map 集变化 | 中 | 不可比 | 抽取断言 100% 覆盖 eval ids；用 --common-maps；coverage 检查 |
| R10 | runner 静默复用旧结果 / 重跑漂移 | 中 | 数字错 | 每臂新 out-dir；--force-eval；同一次 run 内配对；MLP 漂移 ~3e-4 写进报告 |
| R11 | HPC 排队/单卡限制 | 高 | 周期长 | S/D 拆短 job；4 臂可并发；优先 debug 队列跑 bench |
| R12 | 划分不可复现/被后续改动 | 低–中 | 口径漂移 | 划分 ids 与 summary 落盘（小文件入库），固定 seed 0 |
| R13 | eval/pretrain star 失衡（3.385 vs 全池 3.304） | 低 | 解释偏差 | 报告 star 分布；必要时按 star 分层重抽 sets 并重跑（D 之前必须定稿） |
| R14 | 新 ckpt 改变 E1/E2/benchmark 的既有结论 | 中 | 误引用 | 旧结果继续用旧 ckpt；新臂只在本计划口径内报告；E1/E2 重跑是独立 Phase 3 |
| R15 | 数据版权/使用条款 | 低 | — | 沿用现有数据（data/raw 快照）；不抓 API best（README 已说明） |

---

## 9. 过程中遇到的问题（本次调查发现）

> 每条都给证据与处理状态；“未解决”= 需要实现/实验阶段决定。

### 9.1 数据与解析

**P1. 共享 parser 不暴露属性字段。** <code>dcr/cm3p_offline.py</code> 的 HitObject/TimingPoint
丢弃 hitsound、sample set、volume、effects；所有字段都在原始行里（位置见 §2.3）。
影响：不能直接复用现有 parse 路径，必须另写属性解析（或扩展 dataclass）。
处理：新增 <code>data2vec_main/d2v_main/attributes.py</code>，不动共享 parser（§3.3/§7）。

**P2. kiai 的 effects 位掩码 bug（量化）。** 现有转录用字符串相等解析 effects；
全量审计发现 **145 张图 / 271,309 个音符**的 kiai 落在 <code>effects != 1</code> 的 TP 上
（effects 取值含 9 = kiai + omit barline；抽样 1.04% 的 TP 行）。上游 CM3P 用 bitmask（正确），
所以这是本地转录的缺陷。影响：若新前端照抄，会在这些图上丢 kiai；若改共享 parser，会改 CM3P token 流。
处理：新前端用 <code>effects &amp; 1</code>；共享 parser 不动；在测试里加 effects=9 回归用例（§4.8 第 6 条）。

**P3. “0 = 继承”两条规则 + 上游语义。** 逐音符 volume=0 占 **92.0%**（继承 TP volume）、
sample set=0 占 **97.7%**（auto）；375 张图存在 TP volume=0；上游 <code>_get_hitsounds</code>
用 <code>t+5 ms</code> 查询 TP、hitsound 取 <code>&amp; 14</code>、TP sampleSet=0 当 soft(2)。
影响：不解析继承就会把“默认值”当特征；与上游偏离会导致与 legacy 语义不可对照。
处理：复刻 t+5ms 与 &amp;14；volume 继承；sample set 保留 auto 语义并记录与上游的偏离（§3.3 第 9/11 条）。

**P4. 3 张图 hold end ≤ start（bad_hold_end）**、0 张非 4K/非 mania。
处理：release 通道 clamp；parse 断言。

### 9.2 评价与泄漏

**P5. 现协议是 chart-disjoint，不是 mapset-disjoint；泄漏已量化。**
90.6% 的图在多难度 set 内；同 set 内不同难度共享 kiai 数 **97.1%**、SV 变化数 **87.3%**、
完整 timing 签名 **85.3%**、hitsound 标记 **99.4%**；现 D2V “干净” eval 的 1,365 张里
**81.7%** 与预训练池共用 set。影响：SV/kiai/meta 组即使有增益也无法区分“迁移的难度规律”与“认出了歌/作者”。
处理：Protocol D（mapset-disjoint 划分 + 从零训练）；Protocol S 只作筛选并明确标注 leaky。

**P6. 现有 clean 口径功效不足。** P4-hold 只有 1,365 图、MDE ≈ 0.012；P2-hold 单 split MDE ≈ 0.015。
影响：不能用它否定 <0.01 的效应，也不能做单通道消融的判据。
处理：Protocol S 用 P4（MDE 0.005）筛选；Protocol D 用 25% set 留出（1.0–1.2M 比较/seed）+ 10 seeds。

**P7. set-P4 已实现（可复用），但 P4-hold 的 split 有 provenance 缺口。**
<code>scripts/step1_p4_split_sets.py</code> 已实现且 Experiment 2b 跑过；但 split parquet 本地缺（HPC-only）、
Experiment 2b 用的是 nogap 表 + meta_1k。P4-hold 的 checked-in split 来自 nogap 子集、无生成脚本、
用当前代码重跑不能 bit 复现。处理：D 的 probe split 用 set-P4 在 full pairbest + meta_full 上重新生成
（已本地验证 group_overlap=0）；P4-hold 直接复用现有文件、不重新生成（§5.3.2/§5.2）。

**P8. 5/5 符号检验 p=0.031，不足以强确认。** 处理：D 用 10 seeds；预注册阈值 + 报告实测 MDE。

### 9.3 训练工程

**P9. 通道数扩展的 6 条 checkpoint 载入路径 + optimizer moments + LR schedule 都有坑。**
（触点清单 §10：train resume / extract / E1 init / E1 resume / E1 score-only / e1_frozen_window_control；
AdamW 里 conv1.weight 是 state 索引 1；<code>total_steps=steps_per_epoch×epochs</code> 在续训时必须
表示“本次 epoch”，否则 LR 直接落到最小；没有任何 in_channels assert）。
处理：<code>checkpoint.py</code> + <code>--init-from</code>/<code>--total-steps</code> + 入口 assert（§4.4/§7）。

**P10. EMA 不受影响（好消息）。** EMAState 只覆盖 Transformer，conv 是 student/teacher 共享 module；
扩容 conv1 不需要动 EMA。处理：照旧；测试里断言 EMA master 不含 conv。

**P11. LRU 缓存内存与解析成本。** 现有 LRU 512 图约 90 KB/图；不能把 100 Hz 稠密属性数组缓存
（0.5 MB/图 → 数 GB/worker）。parse 约 12 ms/图，主导训练样本成本；属性构建必须向量化到 ≤0.4 ms/窗。
处理：缓存 ChartAttrs（稀疏段+事件），per-frame 现算；实现后跑 200 图 benchmark 验收（§3.5）。

**P12. 自监督目标可能不“使用”新字段（§4.7）。** 这是方法论层面的风险：span-mask 回归只奖励
“有助于预测被遮蔽内容”的属性；kiai/SV/volume 与内容预测可能弱相关。处理：先做 C4 属性可解码性；
备选辅助头 / 行为监督（Phase 3），并在报告里区分“属性无用”与“该目标未吸收”。

### 9.4 环境与资源

**P13. 本机无 GPU、无 HPC 队列直连。** 基线 ckpt <code>runs/main_holdout/checkpoints/last.pt</code>（737 MB）
只在 HPC；本机只有 smoke ckpt。处理：Protocol S 依赖 HPC ckpt；Protocol D 从零训练不依赖；
计划按 HPC 提交（§6.3）。若本地要复现 S，需先 rsync ckpt（未纳入仓库）。

**P14. 本沙箱禁用了 multiprocessing 信号量。** 审计脚本的 ProcessPool 版本在本地报
<code>PermissionError: [Errno 13]</code>，只能单进程跑（21,949 图 254.5 s）。不影响仓库脚本在 HPC 上运行，
但本地验证多进程路径不可行。处理：审计脚本默认 workers=1 也可完成（已跑通）。

### 9.5 方法论张力（必须写在前提）

**P15. canon 39/40 不可外推。** 见 §1.1(a)。处理：只当先验量级参考（属性信息量级约 0.005–0.02 nats），
不当预期效应。

**P16. E1 与 D2V 残差报告与本方向存在张力。** E1 显示遮蔽预训练边际小（0.0047，未达预注册）；
D2V 残差报告明确“不建议直接投入完整四列编码器”。处理：本计划定位为**归因 + 停止规则**；
若 D 失败，结论是“这些属性不能通过该自监督目标进入表示”，并为行为监督路线省下成本。

**P17. 既有 runner 的坑（会直接影响结论正确性）。** 静默复用已有结果（已存在 rank_eval.json 就跳过）、
两次 P2 口径不可混、ceiling 不可跨口径比、MLP 重跑漂移 ~3e-4、randommerge captured 不可引用、
VSRG 需在 RUNS 里注册新 key。处理：每臂新 out-dir + --force-eval；同 run 内配对；报告写明口径（§5.4/§6）。

**P18. mapper 身份可从谱面片段识别（12.5% top-1 / 3,731 mappers）。**
影响：只做 mapset-disjoint 仍可能保留 mapper 其它谱面的风格泄漏。
处理：C3 mapper-ID probe 每臂都报；D 通过后可选 mapper-disjoint 复核（§5.3.3）。

---

## 10. 边界与“不能说的话”

1. 在 Protocol D 通过之前，**不能**写“这些属性携带难度信息”；只能写“在 chart-clean/mapset-leaky
   口径下有/无增益”或“该自监督目标能/不能吸收这些属性”。
2. 不能把 Protocol S 的 P4（transductive）增益写成“新谱面泛化”；P4-hold 的 MDE≈0.012，
   低于它的差异一律“不可检测”，不是“null”。
3. 不引用生产计算器（MinaCalc/osu SR/Daniel/Sunny）的结论来支持属性有效性——它们都只用音符/时间
   （lit-review §2.1），本实验才是本仓库的第一手消融。
4. 跨口径绝对 NLL、captured、VSRG MAE 不可互比；randommerge 的 captured 不可引用。
5. 本计划不重跑 E1/E2/VSRG/deploy；新 ckpt 对行为线/产品线的影响是 Phase 3，任何相关数字不得从本计划推出。
6. 不能从 checksum_ok=True 推断内容表与 .osu 版本一致（D2V extract 未重算 md5）；
   新表以 <code>source</code> tag + <code>d2v_rev</code> 作 provenance，必要时真算 md5。
7. 若最终只得到“属性无增益”，**不能**推出“osu 难度与 SV/kiai/hitsound 无关”；
   只能推出“在本仓库的连续 piano roll + 该自监督目标 + 该评测口径下，没有可检测的难度序增益”。

---

## 附录 A：config 全例（audio_style_a20.yaml）

~~~yaml
seed: 0

data:
  osu_dir: data/osu_cache
  ids_file: null
  window_sec: 16.0
  frame_hz: 100
  gaussian_sigma_ms: 8.0
  min_valid_sec: 1.0
  channels:
    version: attr_v1
    groups:
      base: {}                                    # 8：onset x4 + hold x4（不可变）
      release: {kernel: gaussian, clip: 1.0}      # +4
      tempo:
        bpm: {ref: 120, log2: true, clip: 1.0}    # +1
        beat_phase: sin_cos                        # +2
        bar_phase: sin_cos                         # +2（meter 来自红线）
      sv:
        level: {log2: true, clip: 1.5, normalized_by_median_mpb: true}  # +1
        change_pulse: true                         # +1
      kiai: {}                                     # +1
      # meta（A3 才开）：
      # meta:
      #   hitsound: {bits: [whistle, finish, clap]}        # +3
      #   sample_set: {soft: true, drum: true, auto: true} # +3
      #   volume: {inherit_from_tp: true, time_offset_ms: 5}  # +1

model:
  # in_channels 由 data.channels 派生（A0=8, A1=12, A2=20, A3=27），此处不手填
  conv_hidden: 256
  dim: 512
  layers: 12
  heads: 8
  ff_mult: 4
  dropout: 0.1
  top_k: 6
  mask_prob: 0.065
  mask_length: 10
  loss_beta: 1.0
  max_frames: 4096

train:
  batch_size: 16
  lr: 5.0e-5            # Protocol S：0.5x base；Protocol D 用 base 的 1.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.05    # 300/6,430 step
  epochs: 5             # Protocol S；D 用 20
  grad_clip: 1.0
  ema_start: 0.999
  ema_end: 0.9999
  ema_anneal_steps: 10000
  ema_fp32: true
  log_every: 50
  save_every: 1000
  num_workers: 6
  amp: true
  device: auto
~~~

---

## 附录 B：命令序列（草案）

### B1. 本地（已完成，可复现）

~~~bash
cd /home/mikechu/Documents/Projects/DCR

# 1) 数据审计（全量 21,949 图，单进程约 255 s）
cd data2vec_main
python scripts/audit_attributes.py --workers 1
#    -> results/d2v_attr_audit/summary.json / summary.md
#        data/processed/d2v_attr_audit_per_chart.parquet（不入库）

# 2) 通道原型 benchmark（200 图）
cd .. && python docs/plans/2026-11-24_d2v_attr_evidence/proto_attr_roll.py --n 200

# 3) 现有 smoke（M0 之后应包含 a12 版本）
cd data2vec_main && python scripts/smoke_test.py && cd ..
~~~

### B2. Stage 0：C1 metadata-only 与 C3 mapper probe（CPU，先做）

~~~bash
# C1 特征表：从审计 per-chart 表构造 12-15 维 chart 特征 -> 内容表（任意 D 维，step2_cm3p_rank 支持）
python scripts/build_meta_content.py \
  --audit data/processed/d2v_attr_audit_per_chart.parquet \
  --out data/processed/chart_content_meta_summary.parquet

# 与 8ch D2V / CM3P legacy 在同一 1k 口径上比较（linear 就够）
python3 scripts/step2_cm3p_rank.py \
  --in data/processed/step1_1k_pairbest.parquet \
  --content-path data/processed/chart_content_meta_summary.parquet \
  --model linear --chart-only --chart-folds 5 --chart-seed 0 --threads 8 \
  --out-signals results/d2v_attr_diag/meta_summary_signals.parquet \
  --out-json    results/d2v_attr_diag/meta_summary_rank.json

# C3 mapper-ID probe（8ch 基线）：512d -> mapper_id 线性分类，训练/测试按图切分
python scripts/content_mapper_probe.py \
  --content data/processed/chart_content_d2v_holdout.parquet \
  --meta data/processed/beatmap_meta_full.csv \
  --out results/d2v_attr_diag/mapper_probe_d2v8.json
~~~

（B2 的两个脚本是 M0 新增的小工具；见 §7。）

### B3. Protocol S（HPC）：续训 + 抽取 + P4/P4-hold

~~~bash
# 环境
export PYTHONNOUSERSITE=1
PY=$HOME/.conda/envs/DCR/bin/python
cd data2vec_main

# 0) 确认基线 8ch ckpt
ls -lh runs/main_holdout/checkpoints/last.pt       # 737 MB

# 1) 四条续训（12 min/臂）；ARM=base 就是 A0-cont（8ch 控制臂），a12/a20/a27 为属性臂
#    对 ARM in base a12 a20 a27 各执行一次：
$PY -m d2v_main.train --config configs/audio_style_$ARM.yaml \
  --init-from runs/main_holdout/checkpoints/last.pt --expand-channels \
  --ids-file splits/holdout/pretrain_ids.txt \
  --out-dir runs/attr/$ARM --device cuda \
  --epochs 5 --batch-size 16 --num-workers 6 --log-every 50

# 2) 抽取 6,828 张 pairbest 图（P4 需要；P4-hold 子集在内）
python3 - <<'PY'
import pandas as pd
ids = pd.read_parquet('../data/processed/step1_1k_pairbest.parquet', columns=['beatmap_id'])['beatmap_id'].unique()
open('../data/processed/charts_1k.txt','w').write('\n'.join(map(str, sorted(ids)))+'\n')
PY
#    对每个 ARM：
$PY -m d2v_main.extract --config configs/audio_style_$ARM.yaml \
  --ckpt runs/attr/$ARM/checkpoints/last.pt \
  --ids-file ../data/processed/charts_1k.txt \
  --out ../data/processed/chart_content_d2v_$ARM-1k.parquet \
  --device cuda --batch-size 32 --num-workers 6 --tag d2v_$ARM

# 3) P4（主）与 P4-hold（旁证）
cd ..
for ARM in base a12 a20 a27; do
  python3 scripts/run_p2_p4_multiseed.py --protocols p4 --seeds 1,2,3,4,5 \
    --models linear,linear_extra \
    --content-path data/processed/chart_content_d2v_$ARM-1k.parquet \
    --out-dir results/d2v_attr/$ARM/p4 --threads 8
  python3 scripts/run_p2_p4_multiseed.py --protocols p4 --seeds 1,2,3,4,5 \
    --models linear,linear_extra \
    --content-path data/processed/chart_content_d2v_$ARM-1k.parquet \
    --split-p4 data/processed/step1_1k_pairbest_p4_d2vhold \
    --out-dir results/d2v_attr/$ARM/p4_hold --threads 8
done

# 4) 与 A0-cont 逐 seed 配对
python3 scripts/compare_multiseed_runs.py \
  --baseline results/d2v_attr/base/p4 --variant results/d2v_attr/a20/p4 \
  --out results/d2v_attr/a20/p4/vs_base.md
~~~

### B4. Protocol D（HPC）：mapset-disjoint 划分 + 从零训练 + set-P4

~~~bash
cd data2vec_main
# 1) 划分（新 protocol；写 pretrain_ids 16,487 / eval_ids 5,462）
$PY -m d2v_main.splits --protocol mapset_holdout \
  --holdout-frac 0.25 --seed 0 \
  --meta ../data/processed/beatmap_meta_full.csv \
  --out-dir splits/mapset_holdout

cd ..
# 2) 过滤 probe 表 + set-grouped P4 split（已有脚本）
python3 scripts/filter_pairbest_to_ids.py \
  --in data/processed/step1_full_pairbest.parquet \
  --ids data2vec_main/splits/mapset_holdout/eval_ids.txt \
  --out data/processed/step1_mclean25_pairbest.parquet
python3 scripts/step1_p4_split_sets.py \
  --in data/processed/step1_mclean25_pairbest.parquet \
  --meta data/processed/beatmap_meta_full.csv \
  --out-prefix data/processed/step1_mclean25_p4sets \
  --seeds 1,2,3,4,5,6,7,8,9,10

# 3) 从零训练（A0' + 赢家；38 min/臂；ARM=base 为 A0'，a20 仅为赢家示例，换成实际胜出组）
cd data2vec_main
for ARM in base a20; do
  $PY -m d2v_main.train --config configs/audio_style_$ARM.yaml \
    --ids-file splits/mapset_holdout/pretrain_ids.txt \
    --out-dir runs/attr_d/$ARM --device cuda --epochs 20 \
    --batch-size 16 --num-workers 6 --log-every 50
done

# 4) 抽取 eval 池 5,462 张（约 6 min/臂）
for ARM in base a20; do
  $PY -m d2v_main.extract --config configs/audio_style_$ARM.yaml \
    --ckpt runs/attr_d/$ARM/checkpoints/last.pt \
    --ids-file splits/mapset_holdout/eval_ids.txt \
    --out ../data/processed/chart_content_d2v_$ARM-mclean.parquet \
    --device cuda --batch-size 32 --num-workers 6 --tag d2v_$ARM-mclean
done

# 5) set-P4 10 seeds + CM3P legacy 参照
cd ..
for ARM in base a20; do
  python3 scripts/run_p2_p4_multiseed.py --protocols p4 --seeds 1,2,3,4,5,6,7,8,9,10 \
    --models linear,linear_extra \
    --content-path data/processed/chart_content_d2v_$ARM-mclean.parquet \
    --split-p4 data/processed/step1_mclean25_p4sets \
    --out-dir results/d2v_attr_d/$ARM/setp4 --threads 8
done
python3 scripts/run_p2_p4_multiseed.py --protocols p4 --seeds 1,2,3,4,5 \
  --models linear,linear_extra,mlp,mlp_extra \
  --content-path data/processed/chart_content_cm3p_local.parquet \
  --split-p4 data/processed/step1_mclean25_p4sets \
  --out-dir results/d2v_attr_d/cm3p_legacy/setp4 --threads 8
~~~

### B5. 可选：mapper-disjoint 复核

~~~bash
# 用同一套代码把 group 换成 mapper_id（新 script 或 --group-col mapper_id）
python3 scripts/make_group_holdout_split.py --group-col mapper_id --frac 0.15 --seed 0 \
  --meta data/processed/beatmap_meta_full.csv \
  --out-dir data2vec_main/splits/mapper_holdout
# 之后重复 B4 的 filter/train/extract/set-P4（2 个臂）
~~~

---

## 附录 C：产物清单与入库规则

| 产物 | 路径 | 入库? |
|---|---|:--:|
| 审计汇总 | <code>results/d2v_attr_audit/summary.json</code>、<code>summary.md</code> | ✓ |
| 审计逐图表 | <code>data/processed/d2v_attr_audit_per_chart.parquet</code> | ✗ |
| 审计脚本 | <code>data2vec_main/scripts/audit_attributes.py</code> | ✓ |
| 通道原型 | <code>docs/plans/2026-11-24_d2v_attr_evidence/proto_attr_roll.py</code>（可提升为 scripts/） | ✓（文档附件） |
| 代码触点清单 / 评测协议 / 文献综述 | <code>docs/plans/2026-11-24_d2v_attr_evidence/*.md</code> | ✓（文档附件） |
| 本计划 | <code>docs/plans/2026-11-24_d2v_structured_inputs_plan.md</code> | ✓ |
| 新 config | <code>data2vec_main/configs/audio_style_a*.yaml</code> | ✓ |
| 新测试 | <code>data2vec_main/scripts/test_attribute_channels.py</code> | ✓ |
| 新划分 ids | <code>data2vec_main/splits/mapset_holdout/*.txt,summary.json</code> | ✗（splits/ 被 .gitignore；可复现物是 seed + 生成命令） |
| probe split 表 | <code>data/processed/step1_mclean25_*.parquet</code> | ✗（可由命令重建） |
| 新 content 表 | <code>data/processed/chart_content_d2v_*_v1_*.parquet</code> | ✗ |
| ckpt / runs | <code>data2vec_main/runs/attr*/</code> | ✗（.gitignore） |
| probe 结果 | <code>results/d2v_attr*/</code>（JSON/CSV/MD） | ✓ |
| M4 报告 | <code>docs/d2v_attr_results.md</code> | ✓ |

---

## 附录 D：本次已完成/可复现的证据索引

| 证据 | 位置 | 状态 |
|---|---|---|
| 全量属性审计（21,949 图 / 33.0M 音符 / 913.9 h） | <code>results/d2v_attr_audit/summary.{json,md}</code> | 已跑（254.5 s，单进程） |
| 泄漏统计（set 内共享率、eval/pretrain 重叠） | 同上 <code>group_stats</code> | 已算 |
| 通道原型（数值范围、构建成本） | <code>docs/plans/2026-11-24_d2v_attr_evidence/proto_attr_roll.py</code> | 已跑（200 图 / 400 窗） |
| 代码触点（6 条 load 路径、EMA/optimizer、契约） | <code>docs/plans/2026-11-24_d2v_attr_evidence/code_touchpoints.md</code> | 完成 |
| 评测协议与基线、MDE、坑 | <code>docs/plans/2026-11-24_d2v_attr_evidence/eval_protocol.md</code> | 完成 |
| 文献与设计依据 | <code>docs/plans/2026-11-24_d2v_attr_evidence/literature.md</code> | 完成 |
| mapset-disjoint 25% 划分 + set-P4（seeds 1–3） | /tmp/mclean25（临时，不入库；<code>data2vec_main/splits/</code> 被 .gitignore） | 已验证 group_overlap=0、1.03–1.19M 比较/seed |
| D2V 8ch 基线数字与 provenance | <code>docs/plans/2026-11-24_d2v_attr_evidence/eval_protocol.md</code> §1 | 完成 |

**HPC-only（本地不可复现）**：<code>data2vec_main/runs/main_holdout/checkpoints/last.pt</code>、
<code>data2vec_main/cache/chart_index_100.json</code>、若干 split 表（set-P4/H4）、
randommerge D2V 探针（job 14076959 pending）。

---

## 附录 E：Definition of Done（M4）

1. 代码：§4.8 全部测试通过；base8 的 train/resume/extract/E1 score-only 仍可复现（旧 ckpt）。
2. Protocol S：4 臂 ×（P4 5 seeds + P4-hold 5 seeds）+ C1/C3/C4 诊断；G1 逐条给结论。
3. Protocol D：新划分落盘 + 从零训练（≥2 臂）+ set-P4 10 seeds + CM3P-L 参照；G2 逐条给结论。
4. 报告：per-seed Δ、mean±sd、符号、CI、MDE、分层、负对照；明确区分“可检测/不可检测”。
5. 文档：<code>docs/d2v_attr_results.md</code>；data2vec_main/README 的 8ch 描述与输入限制更新；
   本计划回填实际数字与偏差。
6. 回滚检查：旧内容表 / base yaml / splits/holdout / deploy JSON 全部未被覆盖。
