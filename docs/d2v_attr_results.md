# D2V 结构化属性输入：1-seed pilot 结果（2026-11-25）

> 状态：**已执行**（1 个划分种子，split seed = 1）。计划见
> [2026-11-24_d2v_structured_inputs_plan.md](plans/2026-11-24_d2v_structured_inputs_plan.md)；
> 数据审计见 [../results/d2v_attr_audit/summary.md](../results/d2v_attr_audit/summary.md)。
>
> 口径：单 split seed，无 CI；计划里 5-seed MDE 约 0.005（P4）/ 约 0.012（P4-hold），
> 单 seed 只会更宽。下文的 Delta 一律是相对**同一次运行、同一划分**的 8ch 控制臂，
> 不跨口径比较绝对 NLL。
>
> HPC：i64m1tga800ue（A800）+ i64m512u（probe）；全部 job 2026-11-24 23:01 至 2026-11-25 13:11。
> 复现入口：scripts/hpc_attr_run3.sh（训练/抽取/probe 链）、
> scripts/hpc_attr_dprobe2.sh（D 阶段 probe 重跑）、
> scripts/hpc_attr_shuffle.sh（shuffle 负对照）、
> scripts/analyze_attr_pilot.py（汇总表）。

## 0. 一句话结论

在**全新的 mapset-disjoint 划分**上从零训练的 D2V 中，给 100 Hz piano roll 追加
**tempo / beat+bar 相位 / 有效 SV / SV 变化点 / kiai** 这组结构化通道（20 通道）后，
未见歌曲上的成对难度序明显变好：**DeltaNLL = -0.0246（线性）/ -0.0292（+external）**，
约为 5-seed MDE 的 5 到 6 倍；同一 probe/split 上它也比 CM3P legacy 最强参照
（MLP+ext 0.58696）低 **0.0237**。相比之下：**release 单组基本无效应**；
**hitsound/sample set/volume 元数据组不改善难度序**（a27d 反而比 a20d 差 0.010 到 0.015），
但会抬高 mapper 指纹（C3 +4 到 5pt），与“风格元数据”判断一致。**shuffle 负对照暂缺**（HPC 账号提交额度被另一
会话的 array job 占用；已提交的 3 个 job 于 15:41 被外部 scancel，重试链见 §3.3），
因此当前还不能排除“通道数/统计伪迹”这一解释。

**2026-11-25 补跑的组内拆分**（a20d − SV/kiai → a17d，17 通道，同 D 协议同 split）显示：
删掉 SV+kiai 后 linear 回到 8ch 水平（Δ vs a0d −0.001，噪声内）、+ext 只剩 −0.005，
相对 a20d 变差 **+0.0233 / +0.0238**。即 **a20d 的难度序增益几乎全部来自 SV+kiai**，
tempo/beat/bar 单独几乎没有可检测效应；而 a17d 的 masked loss 反而最低
（0.0263 < a20d 0.0352 < a0d 0.0588），说明自监督 loss 不能作为难度有用性的代理。
详见 §3.4。

**SV vs kiai 拆分（a19d = a20d − kiai，19 通道，2026-11-26 补跑）**：三点分解
a17d（−SV−kiai）→ a19d（+SV）→ a20d（+kiai）显示 **SV 贡献更大**：
加 SV 使 NLL 降 **−0.0154（linear）/ −0.0169（+ext）**，再加 kiai 再降
**−0.0079 / −0.0069**；SV 单独就把 a19d 拉到接近 a20d 的位置（相对 a0d −0.0167/−0.0222）。
C3 mapper 指纹上 a19d（0.1191）不高于 8ch 基线 a0d（0.1340），而 a20d 最高（0.1532）；
C4 上 a19d 的 SV-change R2 达 0.743 而 kiai R2 只有 0.060。即：
**SV 的增益更像读谱/视觉难度；kiai 增益较小且同步抬高 mapper 指纹（更像风格/段落代理）**。
详见 §3.5。

在“从已有 8ch checkpoint 续训 5 epoch”的快速筛选协议（S）上，同一 a20 组的效应小得多
（P2 +ext -0.0055、P4 -0.0017、P4-hold -0.0014），方向一致但均在单 seed 噪声内。
**结论以 D 协议为准**：S 的 P4/P4-hold 仍受 mapset 泄漏与续训预算限制。

## 1. 实现（M0）

新增/修改文件（全部带测试）：

| 文件 | 内容 |
|---|---|
| data2vec_main/d2v_main/attributes.py | .osu 原始字段单次解析（hitsound / sample set / volume / effects 位掩码 / hold end / timing points）；19 条属性通道构建；ChannelSpec（组到切片、spec 版本、shuffle 负对照开关）；SV 组直接复用 dcr.cm3p_offline.sv_groups 保证 legacy 语义 |
| data2vec_main/d2v_main/checkpoint.py | conv1 零初始化扩容（8 到 C，step 0 函数保持）、AdamW moments 扩容、infer_in_channels |
| data2vec_main/d2v_main/train.py | --init-from / --expand-channels / --total-steps / --conv1-new-lr-mult；通道健康日志（每组 conv1 新列 mean abs w） |
| data2vec_main/d2v_main/data.py | base 8ch 路径不变，属性块 concat；AttributeSpec 单一真源 |
| data2vec_main/d2v_main/extract.py | 从 ckpt cfg 恢复 channels/in_channels；禁止覆盖基线内容表 |
| data2vec_main/configs/audio_style_a12/a20/a27/a20_shuf.yaml | A1=12 / A2=20 / A3=27 通道，shuffle 负对照 |
| data2vec_main/scripts/test_attribute_channels.py | 11 组断言（详见 §5） |

通道组：base 8（不动）+ release 4 + tempo 5（bpm、beat sin/cos、bar sin/cos）
+ sv 2（level、change pulse）+ kiai 1 + meta 7（hs whistle/finish/clap、
ss soft/drum/auto、volume）。A1=12、A2=20、A3=27。

## 2. 训练设置与结果（loss / 通道健康）

| 协议 | 臂 | 初始化 | epoch | 步数/耗时 | 末 masked loss | conv1 新列 mean abs w |
|---|---|---|---:|---|---:|---|
| S | a0cont | 8ch ckpt | 5 | 6,430 / 11m12s | 0.0573 | — |
| S | a12 | 8ch + zero | 5 | 约 6,430 | 0.0569 | release 0.0085 |
| S | a20 | 8ch + zero | 5 | 约 6,430 | **0.0447** | release .0088 / tempo .0052 / sv .0011 / kiai .0040 |
| S | a27 | 8ch + zero | 5 | 约 6,430 | 0.0445 | + meta .0017 |
| D | a0d | scratch | 20 | 20,600 / 35m51s | 0.0588 | — |
| D | a12d | scratch | 20 | 约 20,600 | 0.0587 | release .0729 |
| D | a20d | scratch | 20 | 约 20,600 | **0.0352** | release .0612 / tempo .0471 / sv .0568 / kiai .0471 |
| D | a27d | scratch | 20 | 约 20,600 | **0.0283** | + meta .0475 |

- 两个协议里属性臂的 masked loss 都大幅下降，且新列权重明显非零，通道被自监督目标吸收，
  不是“死通道”。
- S 的 5 epoch 给新通道的学习时间有限（新列 abs w 约 1e-3 到 1e-2），D 的 20 epoch 充分（约 0.05）。
- D 从零训练在 16,487 张 mapset-disjoint 预训练图上（比旧 holdout 的 20,584 少 20%），
  所以**绝对 NLL 不能与历史 D2V 数字比**，只看同协议 Delta。

## 3. 主结果

Delta 相对同 run 的 8ch 控制臂；括号内为 linear / +external 两种 probe。单 seed，仅看方向。

### 3.1 Protocol S（续训 5 epoch；P4/P2 为 transductive，P4-hold 为 chart-clean 但 mapset-leaky）

| 协议 | 臂 | NLL linear | Delta linear | NLL +ext | Delta +ext |
|---|---|---:|---:|---:|---:|
| P2（4,582 图） | a0cont | 0.58191 | — | 0.57651 | — |
| | a12 | 0.58121 | -0.00070 | 0.57572 | -0.00080 |
| | **a20** | 0.57779 | **-0.00412** | 0.57103 | **-0.00548** |
| | a27 | 0.57967 | -0.00224 | 0.57241 | -0.00410 |
| P4（6,828 图） | a0cont | 0.58942 | — | 0.58130 | — |
| | a12 | 0.58779 | -0.00162 | 0.57971 | -0.00159 |
| | **a20** | 0.58839 | -0.00102 | 0.57957 | **-0.00173** |
| | a27 | 0.58900 | -0.00041 | 0.58028 | -0.00102 |
| P4-hold（1,365 图） | a0cont | 0.59450 | — | 0.59441 | — |
| | a12 | 0.59623 | +0.00173 | 0.59557 | +0.00116 |
| | a20 | 0.59395 | -0.00056 | 0.59306 | -0.00135 |
| | a27 | 0.60048 | +0.00597 | 0.59837 | +0.00396 |

读法：a20 在三个口径方向一致（+ext 全部为负），但幅度远低于 5-seed MDE；a27 在
chart-clean 的 P4-hold 上明显变差；a12 只有 release 一项，方向不稳。

### 3.2 Protocol D（从零 20 epoch；**mapset-disjoint**，5,204 eval 图 / 1.16M 比较每 seed）

| 臂 | NLL linear | Delta linear | NLL +ext | Delta +ext | captured(linear) |
|---|---:|---:|---:|---:|---:|
| a0d（8ch） | 0.60023 | — | 0.59242 | — | 0.551 |
| a12d（+release） | 0.60074 | +0.00051 | 0.59090 | -0.00152 | 0.548 |
| **a20d（+tempo/SV/kiai）** | **0.57558** | **-0.02464** | **0.56327** | **-0.02915** | **0.697** |
| a27d（+meta） | 0.59092 | -0.00931 | 0.57352 | -0.01890 | 0.606 |
| CM3P legacy（同 split 参照） | 0.61804 | +0.01781 | 0.58889 | -0.00353 | 0.445 |
| CM3P legacy MLP+ext（参照） | — | — | 0.58696 | -0.00546 | — |

- split：cached-filtered set-P4（5,204 图，group_overlap=0，train 162,524 行 / test 79,551 行 /
  1,925 test 图 / 6,162 玩家 / 1,159,146 次比较），eval 池与预训练 pool 的 mapset 完全不相交。
- **a20d 的增益 -0.025（linear）/ -0.029（+ext）是本轮唯一明显超过噪声的效应**；a27d 也显著
  好于 a0d，但比 a20d 差 +0.010 / +0.015；a12d 基本为零。
- CM3P legacy 参照行只作同 probe/split 的相对参照：CM3P 预训练语料未知且与 D2V 的
  mapset-disjoint pool 不同，**不能**据此声称“D2V 全面超过 CM3P”，但可以说
  “在同一评测流程下，a20d 比 CM3P legacy 内容表强约 0.02 nats”。
- CM3P-legacy 行的 linear 比 +ext 差是因为其 external 信号（msd/sunny）对 CM3P 的增益更大，
  这是既有现象（根 README 第 6 节）。

### 3.3 Shuffle 负对照（未完成：HPC 提交额度阻塞）

负对照设计：与 a20d 完全相同（20 通道、从零 20 epoch、16,487 张 mapset-disjoint 预训练图、
同抽取/同 set-P4 probe），唯一变化是属性块按 chart 做**确定性 roll**（保留统计、破坏与音符
对齐；本地单测 <code>test_shuffle_control</code> 已验证确定性与对齐破坏）。它回答
“a20d 的增益来自对齐信息，还是通道数/统计伪迹”。

执行状态（2026-11-25）：

* 13:12 提交 train/extract/probe 三连（14118322/14118323/14118325），train 在 A800 队列
  PENDING（Priority）约 2.5 h；
* 15:41:36 **三个 job 被外部以 CANCELLED by 0 取消**（本地 watcher 未执行任何 scancel）；
* 之后重新提交全部失败：<code>sbatch: error: AssocGrpSubmitJobsLimit</code>。到 17:07
  队列已空（14116579 已 COMPLETED），但**连一个 1 分钟、1 核的 trivial job 也无法提交**，
  说明是账号/组级提交额度或 QOS 限制而不是队列占用，需要用户或 HPC 管理员处理；
* <code>scripts/hpc_attr_shuffle_retry.sh</code> 会自动重试（每 5 分钟，最多 6 h），
  额度恢复时自动串 extract/probe、拉回结果并写 STATUS5c；本轮结束时仍被阻塞。

**因此：a20d 的 −0.025/−0.029 目前不能排除“多 12 个通道带来的容量/统计效应”**；这是
本轮结论最大的未闭合证据，见 §6。

### 3.4 组内拆分：a20d − SV/kiai（a17d，17 通道）

**实验**：从 a20d 的 20 通道中删掉 SV（level 1 + change pulse 1）与 kiai（1）共 3 通道，
得到 a17d = base 8 + release 4 + tempo 5（bpm、beat sin/cos、bar sin/cos）= **17 通道**。
D 协议同池同 split 从零 20 epoch（新 HPC 账号；job 14125021 训练 / 14125022 抽取 /
14125023 probe），set-P4 seed 1；为严格配对，a20d 在新账号/新 runner 上重跑
（a20d_reprobe，结果与旧账号逐位一致：0.575584 / 0.563269）。

| 臂 | 通道 | masked loss | NLL linear | Delta vs a0d | NLL +ext | Delta vs a0d | Delta vs a20d (lin/ext) |
|---|---:|---:|---:|---:|---:|---:|---:|
| a0d（8ch） | 8 | 0.0588 | 0.60023 | — | 0.59242 | — | — |
| **a17d（− SV/kiai）** | **17** | **0.0263** | 0.59892 | -0.00131 | 0.58704 | -0.00538 | **+0.02334 / +0.02377** |
| a20d（+ SV/kiai） | 20 | 0.0352 | 0.57558 | -0.02464 | 0.56327 | -0.02915 | — |

C3/C4（同 5,462 图 eval 池、同 set-disjoint probe split）：

| 臂 | C3 mapper top-1 | C4 kiai R2 | C4 SV-change R2 | C4 BPM R2 |
|---|---:|---:|---:|---:|
| a0d | 0.1340 | 0.047 | 0.065 | 0.294 |
| a17d | 0.1447 | 0.033 | 0.247 | 0.843 |
| a20d | 0.1532 | 0.675 | 0.730 | 0.876 |

**读法**：

1. **SV+kiai 承担了 a20d 几乎全部难度序增益**。删掉后相对 a0d 的 linear 差回到 −0.001
   （噪声内）、+ext 只剩 −0.005；相对 a20d 变差 **+0.0233（linear）/ +0.0238（+ext）**，
   约为 5-seed MDE（0.005）的 4.7 倍。tempo/beat/bar 这一组单独几乎检测不到难度增益。
2. **自监督 loss 与“难度有用性”脱钩**：a17d 的 masked loss 最低（0.0263）但 rank 最差；
   SV/kiai 给 masked-content 预测增加的是“难预测但与难度相关”的信号，低 loss 本身
   不能作为通道有效的证据。
3. **风格泄漏视角**（呼应 §4/§6）：增益恰好集中在风险最高的两组（SV gimmick、kiai =
   song section）。C4 显示删掉后 kiai 可解码性从 0.675 掉到 0.033、rank 增益同时消失；
   而 C3 mapper 指纹只小幅下降（0.1532 → 0.1447，仍高于 a0d 的 0.1340）。
   因此**单 seed + set-disjoint 下仍无法区分“SV/kiai 携带读谱/视觉难度”与“mapper/歌曲风格”**；
   要拆分需要 mapper-disjoint 或 shuffle/随机化对照（本轮未跑）。
4. **局限**：单 seed、无 CI；a17d 与 a20d 是不同训练 run（同 config/seed/数据，
   a20d_reprobe 证明 probe 可复现）；差值 0.023 远大于预期 run-to-run 噪声，但无正式 CI。

### 3.5 SV vs kiai 拆分：a19d（a20d − kiai，19 通道）

**实验**：a19d = a20d 删掉 kiai 通道（1 条），保留 SV（level + change），即
base 8 + release 4 + tempo 5 + sv 2 = **19 通道**。D 协议同池同 split 从零 20 epoch
（HPC job 14128340 训练 / 14128341 抽取，masked loss 0.0309，w_sv 0.0569）。
HPC CPU probe 队列卡在 Priority，**probe 改在本地 CPU** 用同一 cached set-P4 split、
同 runner 重跑三臂（a17d/a19d/a20d）；a17d/a20d 与 HPC 结果差 <= 1.5e-5（BLAS 噪声）。

三点分解（同 runner、同 split、seed 1）：

| 臂 | 通道 | masked loss | NLL linear | NLL +ext | 相对 a0d (lin/ext) | 阶梯增量 (lin/ext) |
|---|---:|---:|---:|---:|---:|---:|
| a0d（8ch） | 8 | 0.0588 | 0.60023 | 0.59242 | — | — |
| a17d（−SV−kiai） | 17 | 0.0263 | 0.59891 | 0.58704 | -0.00132 / -0.00538 | — |
| **a19d（+SV，−kiai）** | **19** | **0.0309** | **0.58350** | **0.57018** | **-0.01673 / -0.02223** | **-0.01541 / -0.01686（加 SV）** |
| a20d（+SV+kiai） | 20 | 0.0352 | 0.57559 | 0.56327 | -0.02464 / -0.02915 | -0.00791 / -0.00691（加 kiai） |

C3/C4（同 5,462 图 eval 池、同 set-disjoint probe split；C3 test n=470，二项 SE 约 1.5pt）：

| 臂 | C3 mapper top-1 | C4 kiai R2 | C4 SV-change R2 | C4 BPM R2 |
|---|---:|---:|---:|---:|
| a0d | 0.1340 | 0.047 | 0.065 | 0.294 |
| a17d | 0.1447 | 0.033 | 0.247 | 0.843 |
| **a19d** | **0.1191** | **0.060** | **0.743** | 0.844 |
| a20d | 0.1532 | 0.675 | 0.730 | 0.876 |

**读法**：

1. **SV 是更大的贡献者**：约 2/3 的 SV+kiai 增益来自 SV（-0.015/-0.017），kiai 约 1/3
   （-0.008/-0.007）。SV 单独就把 a19d 拉到相对 a0d -0.017/-0.022，接近 a20d 的 -0.025/-0.029。
   这与“SV 影响手感/读谱难度”的机制假设一致。
2. **风格指纹不同**：a19d（SV-only）的 C3 mapper top-1 = 0.1191，不高于 8ch 基线
   （0.1340）；a20d 最高（0.1532）。C4 上 a19d 把 SV-change 编码到 R2=0.743，但 kiai R2
   只有 0.060（a20d 0.675）。即 **kiai 是同时抬高 mapper 指纹的那个成分**，SV 不是。
3. **解释（谨慎）**：SV 的增益更像“视觉滚动/读谱结构”的真实难度信息；kiai 增益较小，
   且与 song-section/mapper 风格纠缠（kiai = 副歌/段落）。但 C3 test 集只有 470 图、
   单 seed，C3 的 1-3pt 差异接近 1-2σ；不能据此单独定罪或洗白。
4. **局限**：单 seed、无 CI；a17/a19/a20 是三次独立训练（同 config family/seed/数据）；
   probe 本地 vs HPC 的复算差 <=1.5e-5。sv/kiai 的交互无法从三点分解中排除（例如
   SV 的作用依赖 tempo 通道存在）。

## 4. C3/C4 诊断（泄漏与可解码性）

C3 = 512d 内容向量到 mapper_id 线性分类（top-50 mapper，按 beatmapset 切分 train/test）；
C4 = 512d 到 chart 级属性的 Ridge 回归 test R2。

| stage | 臂 | C3 top-1 | majority | C4 R2: BPM | hold | SV | kiai | volume | hitsound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S | a0cont | 0.2283 | 0.0198 | 0.166 | 0.142 | 0.159 | 0.074 | 0.018 | 0.249 |
| S | a12 | 0.2420 | 0.0198 | 0.162 | 0.139 | 0.157 | 0.067 | 0.022 | 0.241 |
| S | a20 | 0.2466 | 0.0198 | **0.838** | 0.126 | **0.381** | **0.738** | 0.047 | 0.280 |
| S | a27 | 0.2557 | 0.0198 | 0.836 | 0.125 | 0.381 | 0.736 | 0.179 | 0.323 |
| D | a0d | 0.1340 | 0.0404 | 0.294 | 0.346 | 0.065 | 0.047 | 0.075 | 0.270 |
| D | a12d | 0.1532 | 0.0404 | 0.296 | 0.415 | 0.067 | 0.050 | 0.074 | 0.262 |
| D | a20d | 0.1532 | 0.0404 | **0.876** | 0.341 | **0.730** | **0.675** | 0.049 | 0.287 |
| D | a27d | 0.1851 | 0.0404 | 0.841 | 0.100 | 0.793 | 0.693 | **0.932** | **0.965** |

- **属性确实被编码**：a20/a27 的 SV/kiai/BPM 可解码性相对 8ch 大幅上升（例如 kiai 0.07 到 0.74）。
- **mapper 指纹同步上升**：S +2.7pt、D +4 到 5pt（a27d 最高）；说明 timing/meta 通道让表征更“认识作者”。
- 但 D 的 rank-NLL 显示：**meta 组（a27d）的额外风格信息没有转化为难度序**（比 a20d 差），
  而 tempo/SV/kiai 组（a20d）收益最大。C3/C4 与 rank 结果方向一致。
- C1（metadata-only 基线，计划 §5.5）：仅 12 维图级属性汇总走同一 probe，P2 NLL **0.66018**
  （captured 0.253），远差于 8ch D2V（0.58632）与 CM3P legacy（0.61618）
  → 图级属性不是捷径，“把属性对齐到事件/拍位”才可能有用。

## 5. 解析器与数据问题（本轮实际踩坑）

1. **畸形 hit object**：语料中存在 <code>...,5,4,90568:0:0:0:0</code>（type=5 但 field5 是
   结束时间式值）与 <code>hitSound=99999</code>，第一版解析直接
   <code>OverflowError: Python integer 90568 out of bounds for int8</code>，导致 6 个属性臂
   在第一晚全部失败（8ch 控制臂因不走属性解析而成功）。修复：<code>_clip_int</code> 对
   hitsound / sample set / volume 做范围校验，hold end 做合理性 clamp；回归测试覆盖。
2. **畸形 timing point**：修好 hit object 后仍失败，真正越界值是
   <code>meter=99999</code>（int16）与 <code>volume=99999</code>。修复：
   meter 限 1 到 32、sampleSet 0 到 3、volume 0 到 100、丢弃非有限或超长 offset 的 TP；
   回归测试覆盖。
   两轮共浪费约 2 个训练周期（每次 6 臂，约 40 分钟 GPU），但都在 HPC 完成、未污染基线。
3. **eval 池与内容表覆盖不一致**：mapset-disjoint eval 池有 5,236 张 pairbest 图，其中
   **32 张在 osu_cache 里没有 .osu**（无法抽取内容），导致 D probe 的
   <code>assembled 5204 charts, expected 5236</code> 断言失败。修复：生成 cached-filtered
   split（5,204 图，行数 468,646），group_overlap=0，再跑 D probe；报告中注明口径。
4. **本地编排器被机器重启或休眠打断**：HPC 上的 job 不受影响，但本地监控在 09:02 出现
   DNS 失败、09:41 后停止。恢复后用 detached watcher（ssh 抖动重试、空队列需连续两次确认、
   DependencyNeverSatisfied 视为终止）补齐了全部结果。
5. **HPC 队列与账号并发**：debug 分区账号并发上限使 canary 长时间 PENDING；A800 分区
   有按优先级的等待；index cache 在登录节点构建约 17 分钟（I/O 受限），必须在训练前完成。
6. **split 口径注意**：P4-hold 的 checked-in split 来自 nogap 子集、无生成脚本、不能 bit 复现
   （计划 §5.3.2 已说明），本次直接复用；D 的 set-P4 split 在 full pairbest + meta_full 上
   重新生成，**绝对 NLL 不能与 Experiment 2b（nogap + meta_1k）比，只比 Delta**。

7. **账号提交额度（AssocGrpSubmitJobsLimit）**：shuffle 对照重提交时被账号级额度挡住；
   队列已空、连 trivial job 也被拒，说明不是队列占用，而是账号/组级 QOS 或提交限制，
   需要用户或管理员处理。同一时段的 3 个 shuffle job 还被外部 scancel（CANCELLED by 0）。
   这是后续 5-seed 复跑的运维前提：应确认账号额度或改用有额度的账号/分区。

8. **新 HPC 账号（aiaa3111_l01_066）是空 home**：旧账号 home 不可读，只能从共享
   Python 3.11 建 venv、pip 安装 torch 2.5.1(CUDA) 等，并 rsync 1.2 GB osu_cache +
   小表；因此 a17d 在另一账号训练/抽取/probe。为消除环境差异，a20d 在同一账号重跑
   （与旧账号逐位一致 0.575584 / 0.563269），配对有效。

9. **HPC CPU probe 队列拥堵**：a19d 的 CPU probe（14128342）在 i64m512u 排队
   Priority 超过 1 h，改为**本地 CPU** 完成（同一 split/runner；a17d/a20d 复算差
   <=1.5e-5）。后续轻量 probe 优先本地或 debug 队列（30 min 限制内）。

## 6. 能说什么 / 不能说什么

可以说的：

* 在本仓库的连续 piano roll + data2vec 自监督目标下，**追加 tempo/beat/bar/SV/kiai 通道
  能让从零训练的编码器在 mapset-disjoint 未见歌曲上明显改善难度序**（1 seed 的
  -0.025 / -0.029，约为 5-seed MDE 的 5 到 6 倍；补 5 seeds 可望确认）。
* **release 单组不是主要来源**；**hitsound/volume/sample set 元数据不改善难度序**，
  且显著抬高 mapper 指纹（+4 到 5pt），与“风格元数据”的先验一致。
* 通道确实被编码（C4 可解码性、新列权重、masked loss 均支持），不是“死通道”。

不能说的（本轮限制）：

* 单 split seed，无 CI；不能给出 p 值或“显著”的强结论；不能排除 seed-1 划分的偶然性。
* 不能声称 D2V 全面超过 CM3P：CM3P 参照行只作同 probe/split 的相对参照，且其预训练语料
  与 D2V 的 mapset-disjoint pool 不同；也不满足“D2V 打赢 legacy MLP+ext”的完整口径。
* 不能声称“SV/kiai 就是难度信号”：C3 显示它们同时增加 mapper 指纹；虽然 D 的
  mapset-disjoint 设计排除了“预训练见过同 set”这条路径，但 probe 的 train/test 仍共享
  mapper 群体（C3 的 top-50 mapper 非 mapper-disjoint）。要下机制结论需要 mapper-disjoint
  复核 + shuffle 结果 + 5 seeds。
* a20d 与 a27d 的差异（meta 组变差）是单 seed 下的方向，不足以单独写“meta 有害”的强结论。
* **a20d 的增益尚未通过 shuffle 负对照**：不能排除通道数/容量或统计伪迹；在补上
  shuffle 与 5 seeds 之前，只能写“方向性支持”，不能写“已证明是节奏/滚动信息的作用”。
* 不重跑 E1/E2、VSRG benchmark、deploy；pilot 的新 ckpt 未进入行为或产品线。

## 7. 下一步（建议）

1. **补 5 split seeds**（S/D 各 5 seeds，probe 复用现成 runner）：把 a20d 的 -0.028 变成
   mean±sd + CI，确认是否过 G2（不高于 -0.003 且 5/5）。
2. **补 shuffle 负对照结果**（正在跑）与**组内拆分**（tempo/beat vs SV vs kiai 分别加）
   来定位收益来源。
3. **mapper-disjoint 复核**（A0' + a20d 各 2 个 run，约 1.3h GPU）：排除“同 mapper 风格”。
4. 若 5-seed 确认，再考虑把属性通道接入 E1 行为监督（Phase 3）与 VSRG benchmark。
5. 工程：把 attributes.py 的解析缓存与性能 profiling 做掉（当前 on-the-fly parse 在 D 训练里
   约 30 到 40% 数据时间）；把 shuffle 开关写进正式 spec 文档。

## 8. 产物索引

| 产物 | 路径 |
|---|---|
| 1-seed 汇总表 | results/d2v_attr_run/pilot_table.md |
| S 各臂 summary | results/d2v_attr/{a0cont,a12,a20,a27}/{p4,p4_hold,p2}/summary.csv |
| D 各臂 summary + CM3P 参照 | results/d2v_attr_d_cached/{a0d,a12d,a20d,a27d,cm3p_legacy}/setp4/summary.csv |
| 内容表（本机） | data/processed/chart_content_d2v_a*-{1k,mclean}.parquet |
| C3/C4 诊断 | results/d2v_attr_diag/{mapper_probe_*,attr_decodability_*}.json、summary_arms.json |
| 训练 ckpt | HPC ~/DCR/data2vec_main/runs/{attr,attr_d}/（737MB/个，未拉回） |
| 训练日志 | results/d2v_attr_run/logs/、HPC slurm_attr_* |
| 编排/复现脚本 | scripts/hpc_attr_run3.sh、scripts/hpc_attr_dprobe2.sh、scripts/hpc_attr_shuffle.sh、scripts/analyze_attr_pilot.py、scripts/run_attr_diag_all.sh |
| job id | results/d2v_attr_run/jobs*.txt、sacct*.txt |
