# CM3P 输入裁剪、窗口导出与池化（v5 实现口径）

> 对应计划：[CM3P 输入裁剪与窗口导出：实现计划（v5）](<# CM3P 输入裁剪与窗口导出：实现计划（v4）.md>)
> （文件名沿用 v4；内容已按 v5 修订）。
>
> **v5 口径**：T1 固定启用，T3 是唯一实验变量；A0 是 T3 内部对照，**不要求复现旧表的 NLL**。
> 旧 0.61484 只作历史参考。**本地仓库/数据/产物是权威；HPC 只提供 GPU 算力**，
> 上面的副本可随时重传或丢弃。
>
> 代码：scripts/step0c_chart_content.py（T0/T1/T2 入口）、dcr/cm3p_windows.py（核心）、
> dcr/cm3p_offline.py（离线 token 会计）、scripts/cm3p_token_budget.py（T0/T1 审计 CLI）、
> dcr/cm3p_pool.py + scripts/cm3p_pool.py（T3）、scripts/run_cm3p_pool_experiments.py（T3 驱动）。
> 测试：scripts/test_cm3p_windows.py。

---

## 0.0 三个词，先对齐

- **真实 HF checkpoint**：Hugging Face 上 `OliBomby/CM3P` 发布的**权重 + processor config**
  （`preprocessor_config.json`、tokenizer 配置）。`CM3PProcessor.from_pretrained` 读的是它，
  不是本仓库的 `configs/train/*.yaml`。它也**不是上传到 HPC 的仓库副本**；HPC 上的
  `external/cm3p`、脚本、数据都只是本地同步过去的临时拷贝。
- **OOD（out-of-distribution）输入**：预训练时没见过的 token 类型、token 组合或序列长度。
  例如 `[MANIA_COLUMN_*]` 若从未被发射，对应 embedding 行就没收到过梯度；把列号打开就是把
  未训练的行喂进 attention。OOD 不等于“必然报错”，而是“输出不可信，必须用探针验证”。
  T1 裁剪也是输入分布变化，但它是**删 token**、且是已接受的固定口径；column 是**加模型没见过的 token**，
  两者的风险不同。
- **HPC / 本地**：HPC 只负责跑 T2 的 GPU；**代码、数据、旧表、最终产物都以本地为准**。
  T2 产物必须回传本地；T3 在本机 CPU 做。

---

## 0. 最重要的发现：上游从未发出 [MANIA_COLUMN_*]

计划的 §1.1 写「每个 mania tap = 6 token，含 [MANIA_COLUMN]」。**运行中的代码不是这样**：

    parsing_cm3p.py:501        group.column = self._get_mania_column(...)
    parsing_cm3p.py:57         mania_column: int = None      # Group 的字段
    tokenization_cm3p.py:191   if group.mania_column is not None:   # 永远是 None

解析器把列号写到一个不存在的字段 column 上，tokenizer 读的是从来没被赋值的
mania_column。自 64957b0（2025-08-17，「tokenize all beatmap features」）起就是这样，
git log -S 显示任何提交都没有修过。因此：

* 在上游配置（`add_snapping=True`）下，每个 tap 实际是 **5 token**
  （CIRCLE/TIME_SHIFT/SNAPPING/HITSOUND/VOLUME），hold 是 **8**；
* **真实 HF checkpoint**（`add_snapping=False`）下，tap 是 **4 token**，hold 是 **6**；
  v4 写的 6/10 把从未发射的 column 也算进去了；
* **冻结的 CM3P checkpoint 从未见过任何 [MANIA_COLUMN_*] token**。内容向量对列号是盲的，
  而列号是 mania 最核心的信息之一（jack/trill/chord 的左右手结构都在里面）。

还有第二处：_tokenize_mania_column 把值 clip 到 1..18，而 _get_mania_column 返回
0..columns-1。就算只把属性名改对，4K 的列 0 和列 1 会撞成同一个 token。

处理方式（不静默、不破坏旧结果）：

| 模式 | 行为 |
|---|---|
| --mania-columns auto（默认） | 不 patch；第一次 tokenize 时检测并打印醒目警告；index 里 columns_emitted=false |
| --mania-columns on | 安装 opt-in 运行时 patch：解析器写 1-based 的 mania_column，tokenizer 正常发出列 token；同时打印 OOD 警告 |
| --mania-columns off | 明确接受 column-blind，不再警告 |

patch 在 dcr/cm3p_windows.py:install_mania_column_patch，只包住 _add_group，
并且同时修正 0/1-based 的偏移。**列号可以进 token 流**：词表已有 `[MANIA_COLUMN_1..18]`，
tokenizer 也有发射分支，patch 后就会正常出现。

**那 x 坐标呢？** 也能，但不是等价物。真实 parser 的 `add_positions=True`，tokenizer 会发
`[POS_X_*]`/`[POS_Y_*]`；可是 `_add_group` 只在 mode 0/2 填 `group.x/y`，mania 分支不填。
要 patch `group.x = int(pos[0]); group.y = 0` 才会发；代价是**每 note +2 token**
（`[POS_X_*]` + 常量 `[POS_Y_0]`）。canon 的 2.76 token/note 会变成约 4.76，几乎回到
legacy 的 4.90，**T1 的 token 收益基本被抵消**。语义上 `[POS_X_*]` 是 std/catch 坐标词表，
checkpoint 在 mania 上下文里没见过，仍是 OOD；只是 embedding 行本身训练过，和未训练的
`[MANIA_COLUMN_*]` 行不同。它只能作为 side experiment，不能进主 T1/T3。

**默认不打开，主实验固定 off**：checkpoint 没见过这些 token，直接喂进去是分布外输入。
如果以后想开，必须先做 256 图 off/on 探针（见 §6），确认 embedding 没有崩坏、
P2 上的配对 ΔNLL 没有变差，再把 column **作为固定的 T1 变体**另开一轮；
不能把它和 T3 变体混在一起比较。这个探针是 side experiment，不阻塞本次 T3。

离线审计（scripts/cm3p_token_budget.py）会同时报 off 和 on 两个流：
400 张图上 columns 覆盖 0/705,943 个 note event（0.0%），与上面的代码事实一致。

### 0.1 第二个 T0 发现：真实 HF 配置 ≠ 计划假设

2026-11-04 初读、2026-11-05 用 Hugging Face 的同一份 config 复核
`CM3PProcessor.from_pretrained("OliBomby/CM3P")` 的 processor config。
HPC 只是当时装了 transformers 的机器；这些值是 **checkpoint 自身的属性**，不是 HPC 文件的属性。
HPC 上的仓库副本可以丢，配置事实不会变。本地快照已放在
[configs/cm3p_hf/](../configs/cm3p_hf/README.md)：
`processor_config.json`、`beatmap_parser/preprocessor_config.json`、
`beatmap_tokenizer/tokenizer_config.json`、`config.json`。
注意根目录文件名是 `processor_config.json`，不是 `preprocessor_config.json`；
后者在 `beatmap_parser/` 子目录下。T0 把运行时 snapshot 写进 index/meta，并与这份本地快照核对；
当前实现已硬断言 window/stride/max_time，`max_length` 记录并使用。

| 项 | 计划 / 上游 configs/train/default.yaml | 真实 checkpoint 配置 |
|---|---|---|
| add_snapping | True | **False**（输入里没有 [SNAPPING_*]） |
| add_cls_token | False | **True**（每窗 3 个特殊 token：CLS+BOS+EOS） |
| max_length | 4000 | **2000** |
| window / stride | 16 s / 16 s | 16 s / 16 s（一致） |
| tokenizer.max_time | 16000 | 16000（一致） |

后果：

* 每个 mania tap 实际是 **4 token**（CIRCLE/TIME_SHIFT/HITSOUND/VOLUME），hold 是 6；
  计划写的 6/10 同时高估了 snapping 和 column。
* 计划的 §2 预算表（4000、有 snapping）不成立；旧表
  chart_content_cm3p_local.parquet 也是用**真实配置**（2000、无 snapping、CLS）算的。
  T2 必须用真实配置，才能和 **checkpoint 的输入口径**一致。**旧表不是 T2/A0 的 parity 目标**：
  它只作历史参考；A0 是同一份 canon 窗口库上的 T3 对照。
  T0 现在记录 max_length=2000 并**使用它**做截断和预算（计划的 risk 表：预算重算，方案不变）；
  只有 window/stride/max_time 这类真正的异常才需要 --allow-config-mismatch。
* 裁剪**不能**顺手把 snapping/CLS 打开：checkpoint 没训练过这些 token，那是分布外输入，
  不是「保留核心信息」。CropSpec 只改 hitsounds/kiai/mania_sv 三个决定项，其余从
  checkpoint 继承（实际 snapping=False、CLS=True）；T1 的 core audit 比较的是同一套
  基线 token 流。
* 离线 token 会计已补上 add_snapping/add_cls 两个开关；与真实 tokenizer 在
  20 张图（10 张逐窗口对照）上**完全相等**，见 §5。

---

## 1. T0：先读真实配置，再谈预算

真实 window / max_length 来自 HF preprocessor config，不是本仓库的默认值。
step0c 在任何解析之前做快照和断言（dcr/cm3p_windows.py:read_processor_snapshot /
assert_processor_config）：

* window_length_sec、window_stride_sec、min_window_length_sec、max_length；
* padding、truncation（必须是 longest_first，否则 T2 的截断口径不同）；
* beatmap tokenizer 的 max_time、time_step、add_cls_token；
* parser 的七个 flag（裁剪后、从 checkpoint 继承的实际值）。

断言：

1. window == 16 s；
2. stride == window（不重叠，否则同一 note 会进两个窗口）；
3. tokenizer.max_time >= window*1000。这是核心信息检查：**如果 max_time 只有
   8000 ms 而窗口是 16 s，窗口后半段的 [TIME_SHIFT] 全被 clip 成一个 token，
   时间信息就丢了**。真实配置 max_time=16000，正好覆盖 16 s 窗口。
4. max_length **记录真实值并使用它**（真实 2000）。如果与计划的 4000 不同，打印 NOTE、
   用真实值截断，并按计划的 risk 表重算预算；这不是错误，因为 checkpoint 训练时就是
   2000，强扭成 4000 会改变输入分布。

   `max_length` 是**单个 16 s 窗口的 token 数**，不是按键数，也不是整张图的上限。
   canon 400 图实测窗口 p95=801、max=1,095；同一批窗口里最多 **447 个 note 起点**
   （含 hold end 为 487 个 note group）。按 2.76 token/note 算，2,000 token 约能放 700 个
   note，而实测最多 447 个，所以有约 1.8× 的 token 余量；撞上限 0/4,251，legacy 才有 0.42%。4000 技术上能跑
   （模型 `max_position_embeddings=8192`），但它是新的序列长度分布，而且只影响真正超过
   2000 的窗口（典型窗口仍按实际长度动态 padding，不会统一 pad 到 4000）；它会把固定 T1/T2
   变成第二个变量。只有全量审计显示 canon 确有可观窗口 >2000 时才另开一轮“固定 4000”的变体。

window/stride/max_time 不匹配时默认 SystemExit；--allow-config-mismatch 才继续。
max_length 不是硬断言：实际值写进 index.meta.json / cfg_hash。

输出示例（HPC 实测）：

    [T0] beatmap kwargs: window=16s stride=16s max_length=2000 truncation=longest_first
         tokenizer.max_time=16000ms time_step=10ms add_cls=True
    [T1] crop=canon: hitsounds=False kiai=False mania_sv=False timing=True
         timing_points=True snapping=False

---

## 2. T1：裁剪，以及「核心信息没有丢」的证明

**T1 是固定的预处理，不是实验变量**：本次所有 T3 实验都用同一套 canon。
CropSpec（dcr/cm3p_windows.py）的决策项是前三个；后三个从 checkpoint 继承，不改、不强行打开：

| 开关 | canon | 去掉的 token | 保留的信息 |
|---|---|---|---|
| add_hitsounds | False | 每 note 的 [HITSOUND_*]+[VOLUME_*]（2 个） | — |
| add_kiai | False | [KIAI_ON/OFF]（每次切换 2–3 个） | — |
| add_mania_sv | False | [SCROLL_SPEED_*]（每次变化 3–4 个） | — |
| add_timing | 继承（真实 True） | — | BEAT/MEASURE/TIMING_POINT 网格（切分音、变速） |
| add_timing_points | 继承（真实 True） | — | 红线的 TIMING_POINT（不是 MEASURE） |
| add_snapping | 继承（真实 **False**） | — | 真实 checkpoint 的输入本来就没有 [SNAPPING_*] |

后三行不是 T1 的开关：它们从 checkpoint 继承（真实是 timing/timing_points=True、
snapping=False），既不改也不强行打开——把 checkpoint 没训练过的 snapping 打开是分布外
输入，不是「保留内容」。canon 的语义只是「删掉 hitsound/volume、kiai、mania SV 这三类载荷」。
**T1 的验收是“只删了这三类载荷”，不是“P2 NLL 变好”；T1 不参与 T3 的归因。**
step0c --core-audit N 会对前 N 张图做两次真实 tokenize（交给 tokenizer 的 padding/
truncation 都关掉，拿到完整 token 流；模型用的 ids 才另行按 max_length 截断），
然后逐窗口比较：

    strip_removed(tokens_legacy) == tokens_canon

strip_removed 只去掉 [HITSOUND_*]、[VOLUME_*]、[KIAI_ON/OFF]、[SCROLL_SPEED_*]。
这个比较比计数更严格：它同时检查 note 是否还在、时间/顺序是否变了、窗口归属是否错了。
另外还断言 note-event token 数、column token 数、hold split 数完全一致。
任何不一致 → SystemExit，拒绝导出。

**2026-11-06 修复**：如果 kiai/SV 组正好是窗口里的最后一组，旧版 `strip_removed` 会一直
处于“丢弃该组”的状态，把后面的 `[EOS]` 也吞掉；canon 侧没有这个组，所以 `[EOS]` 还在，
逐窗比较就会出现“legacy 少一个 EOS”的假阳性。现在 `[CLS]`/`[BOS]`/`[EOS]` 会强制结束
丢弃状态并保留；`scripts/test_cm3p_windows.py` 加了 `kiai before EOS keeps EOS` 回归测试。
首次 full run 13593775 就是在这个假阳性上中止的（没有导出任何窗口）。

离线版在 dcr/cm3p_offline.py:audit_chart（同样比较 group 的 (kind,time,snapping,column,
scroll) 元组）。scripts/cm3p_token_budget.py 对全量 21,949 张做预算，并对其中的前 128 张
做 core audit（见 §5 与 index.meta.json）：

* core stream identical = True；
* removed tokens = hitsound/volume + kiai + SV，计数精确；
* 没有 note、没有 timing group 被裁掉（截断情况单独记账，见 §3.3）。

**注意**：这条保证只管「裁剪」；如果窗口超过 max_length，tokenizer 的 LONGEST_FIRST
截断会丢掉窗口尾部的 note（见 §3.3 的 truncation 记账）。canon 把 token 降了 **43.7%**
（400 张样本）；在真实的 max_length=2000 下，canon 窗口撞上限的个数是 **0**
（legacy 是 18/4251 = 0.42%）。计划按 4000 写的 0.31% 不适用：真实上限是 2000，
而旧表本来也是按 2000 截的。

---

## 3. T2：窗口导出 + 分桶

### 3.1 运行方式

--source osu-dir 默认 --windows on（旧的 BeatmapFilesDataset 路径要显式 --windows off）。
canon 的默认输出是 data/processed/chart_content_cm3p_local_canon.parquet，旧表不动。
如果主表已经完整、只是窗口没导全，同一条命令会走「只补窗口」分支（index 里 ready 的图
直接跳过），不需要 --refresh。

    python scripts/step0c_chart_content.py --source osu-dir \
        --ids-file manifests/whitelist_1k.csv --osu-dir data/osu_cache --audio off \
        --crop canon --windows on \
        --out data/processed/chart_content_cm3p_local_canon.parquet

产物：

    .cache/cm3p/win/<beatmap_id>.npy    float16 (n_win, 512)，时间序
    .cache/cm3p/index.parquet           见 3.2
    .cache/cm3p/index.meta.json         配置快照、export 统计、core audit 结果
    <out>.meta.json                     主表的 cfg_hash/crop，merge guard 用

### 3.2 index 的列

beatmap_id, n_win, mean_norm, tok_len[], tok_len_full[], content_md5, cfg_hash,
n_holds_split, n_holds_boundary_2ms, n_holds, n_notes, n_note_tokens,
n_column_tokens, columns_emitted, n_truncated_windows, n_tokens_dropped,
n_core_tokens_dropped, song_length_s, window_starts_ms[]

其中 n_holds_split 是跨 16 s 边界的 hold 数（start//W != end//W），
n_holds_boundary_2ms 是 end 距边界 <=2 ms 的 hold 数。计划只要求前者；
后者是零成本的补充诊断。已知局限不修：跨边界的 hold 会把 [HOLD_NOTE_END] 甩进下一个窗口。

### 3.3 截断记账（核心信息不静默丢失）

tokenize_chart 用 padding=do_not_pad + truncation=do_not_truncate 拿到**完整**窗口，
自己按**真实的** max_length（2000）截断（与 LONGEST_FIRST 一样从尾部截，保留
[CLS]/[BOS]）。每个窗口记：

* tok_len_full（真实长度）、tok_len（模型看到的长度）；
* n_truncated_windows、n_tokens_dropped；
* n_core_tokens_dropped = 被截掉的 [CIRCLE]/[HOLD_NOTE]/[HOLD_NOTE_END] 个数。

--on-truncate warn（默认）打印一次并继续，但**每张受影响的图都在 index 里标出来**；
error 直接失败；allow 不打印。T3 的 --exclude-truncated 可以只在无损图上做比较。
所以「没有核心信息丢失」在实现层面是：要么为零，要么逐图可查、可排除。
要绝对零损失，用 --on-truncate error 让导出在这些图上失败，然后把超长窗口单独切块
（本实现没有做切块；计划只要求记录截断，不要求修；真实 legacy 是 0.42%，canon 是 0）。

### 3.4 BucketSampler 与 padding

dcr/cm3p_windows.py:BucketSampler：

* 桶 key = ceil(len/64)*64，封顶 max_length；
* 每批只从一个桶取，批内随机，批间顺序随机；
* 同 chart 的窗口按 (chart_seq, win_idx) 排序后再 stack，时间序不破；
* 缓冲上限（--window-buffer，默认 4096 窗）防止 21,949 张图一次性进内存；
* 动态 padding：每批 pad 到批内最大长度，不是 2000。

打印 pad_efficiency = sum(len)/(B*max_len) 与 charts/s。400 张图离线估算
（真实 max_length=2000）：长度排序分批的 pad_efficiency 在 batch=8/16/32/64 时约
0.998/0.995/0.990/0.980；如果每窗都 pad 到 2000，只有 **0.213**。这就是分桶的依据。

### 3.5 cfg_hash、续跑与 merge guard

* cfg_hash = sha1(crop flags + 完整 snapshot + model rev + dtype + attn + audio +
  window dtype/max_length + CM3P commit + torch/transformers 版本)。
  分桶参数（--window-batch/--window-buffer/--window-seed）不进 hash：窗口之间没有
  交叉 attention，换批量只影响浮点核的舍入，缓存可以继续用。
* index 里每行的 cfg_hash 必须一致；不一致默认拒绝继续，
  --refresh-windows 才删除 .npy 和 index 重建。
* 续跑：index 里已有、content_md5 相同、.npy 存在的图直接跳过（从 .npy 重算图级均值）。
  index 每 200 张图原子写一次。
* 主表 merge guard：<out>.meta.json 记录 crop/cfg；canon 行不能并进 legacy 表，反之亦然。
  --refresh / --allow-mixed-crop 才放行。旧表没有 sidecar 时按 legacy 处理（--crop legacy 可合并）。

### 3.6 自检：窗口存储真的等于图级向量

每张图 finalize 时写 .npy；对前 N 张（--verify-windows，默认 64，0 = 全部）把 .npy
读回来，算 mean→normalize，与**同一次运行内存里的** float32 均值比 cosine，要求 >= 1 - 1e-6。
fp16 的典型 cosine 偏差在 1e-7 量级；失败会直接报错并建议 --window-dtype float32。
这个自检只证明“窗口存储 → 图级向量”没写错，**不是 canon 与旧表的 parity 检查**。

如果以后要额外做 canon vs 旧表的描述性对比，可以单独用 --crop legacy 跑 N 张图，再与
data/processed/chart_content_cm3p_local.parquet 的同 id 向量比 cosine。那是 side experiment，
不进入 T2 验收，也不要求 A0 复现旧表。

---

## 4. T3：池化 A0–A6

### 4.1 特征与宽度

| 变体 | 定义 | 宽度 |
|---|---|---|
| A0 | mean(W_t) 再归一化（**T3 内部对照**，同一 canon 窗口库） | 512 |
| A1 | 逐维 mean + std（风险项） | 1024 |
| A2 | A0 + mean/sd/q90/q99/max(z_t) + log mean_norm + log n_win | 519 |
| A3 | A2 + 最长高压连续段 + 最长休息段 | 521 |
| A4 | softmax(beta*z_t) 加权 mean（beta=5；--a4-mode lse 可换逐维 LSE） | 512 |
| A5 | minacalc.aggregate_rows(W_t) | 512 |
| A6 | 2 层 dilated TCN 过窗口序列，按 rank 目标训练 | 512（signal） |

**T1 固定，所有 T3 变体共享同一份 canon 窗口库。** A0 是 T3 的均值对照，
不是旧 legacy 表的复现；**不要求 A0 等于 0.61484**，也不要求它复现旧表的 NLL。
旧数字只作历史参考。主判定是 A1–A6 各自 vs A0 的 paired ΔNLL。

A5 的 minacalc.aggregate_rows 按列聚合：把 n_win 行窗口聚成 512 维（计划里简写
aggregate_rows(z_t) 的 z_t 是标量、那样只有 1 维，与表里 A5 的宽度 512 不符；这里按
窗口×512 矩阵实现）。

z_t = w_fold · ((W_t - mu)/sd)。w_fold 来自 A0 的**chart-disjoint 5 折线性头**，
mu/sd 是 A0 训练图上的标准化。图在 fold k 用 w_k；不在 step1 里的图（无标签）
用 w_full。这样 held-out 图的投影没有用过该图的标签，A2/A3/A4 的 P2 口径是干净的。

注：计划写 z_t = w·W_t；实现先把 W_t 用 A0 的 train-chart mu/sd 标准化，因为 w 是在
那个空间里拟合的。A4 的 softmax 前再把 z 在 chart 内标准化，避免 w 的尺度让 beta 饱和。

### 4.2 生成特征

先跑 A0 并 dump heads：

    python scripts/cm3p_pool.py --variants a0 --out-dir data/processed/chart_pool_features_1k
    python scripts/step2_cm3p_rank.py --model linear --chart-only --chart-folds 5 \
        --chart-seed 0 --val-frac 0 --lam 1e-3 --name pool_a0 \
        --content-path data/processed/chart_pool_features_1k/a0.parquet \
        --content-col features --dump-heads results/cm3p_pool_1k/a0_heads.npz \
        --out-signals results/cm3p_pool_1k/pool_a0.parquet \
        --out-json results/cm3p_pool_1k/pool_a0.json

再生成其余特征并拟合：

    python scripts/cm3p_pool.py --variants a1,a2,a3,a4,a5 \
        --heads results/cm3p_pool_1k/a0_heads.npz \
        --out-dir data/processed/chart_pool_features_1k

单变体且不给 --out-dir 时写 data/processed/chart_pool_features_1k.parquet（把 A2 作为主特征时
正好用这个文件名）；多变体用 --out-dir 写 <variant>.parquet。

### 4.3 一键驱动

    python scripts/run_cm3p_pool_experiments.py --variants a0,a1,a2,a3,a4,a5

它做：A0 特征+heads → A1–A5 特征 → 同样的 P2（5 折、seed 0、lambda 1e-3）→ 合并
signals_p2.parquet 的历史 baseline（只作参考）→ step2_rank_eval.py（--split test --ceiling-fit all
--common-maps --strict-pairs）→ paired_delta_nll.py（每个变体 vs A0）→
results/cm3p_pool_1k/{summary.json,SUMMARY.md}。

A6 单独：

    python scripts/cm3p_pool.py --variants a6 --heads results/cm3p_pool_1k/a0_heads.npz \
        --step1 data/processed/step1_1k_pairbest.parquet \
        --out-signals results/cm3p_pool_1k/pool_a6.parquet

A6 用两遍前向/反向（第一遍算 d 与 dL/dd，第二遍分块回传），内存受限的机器也不会 OOM；
它是信号，不是固定的 features 列，所以在驱动里直接并进 signal 表。

### 4.4 Go/no-go（T3 内部，配对）

主判据：**A1–A6 各自 vs A0** 的 P2 strict NLL paired ΔNLL；同一 player cluster、
同一 pair 集、同一 ceiling；cluster bootstrap 的 95% CI 不含 0 才算有增益。
**旧 0.61484 不是门槛**。A0 本身是多少（0.63、0.59…）是 T1/T2 固定口径的结果，
不影响“哪个 T3 变体比 A0 好”的配对比较。绝对 NLL 仍上报，但只作描述。
--common-maps 只保证所有变体在同一个 map 集上比较。

### 4.5 混淆检查（可选，仅 canon vs legacy 描述性对比）

T1 固定后，主实验 A1–A6 vs A0 都在 canon 内部，不需要这一层。
只有当你额外想把 canon（pool_a0）与旧 legacy 表做描述性对比时，才用 count_total 分十层看
「删 hitsound 是否只是掉了流行度代理」：

    python scripts/stratified_delta_nll.py \
        --signal-table results/cm3p_pool_1k/signals_pool.parquet \
        --signals pool_a0,cm3p_linear_cd_l1e3 \
        --pairs pool_a0:cm3p_linear_cd_l1e3 \
        --strata-table data/processed/rank_signals_1k.parquet \
        --stratum-col count_total --strata 10 --split test \
        --out results/cm3p_pool_1k/stratified_pool_a0_vs_legacy.json

run_cm3p_pool_experiments.py --stratify 会在 ΔNLL > 0 时自动跑它。
（rank_signals_1k.parquet 的 count_total 就是 beatmap_meta 的那一列。）

---

## 5. 离线预算与核心信息审计（真实配置，400 张样本）

真实配置 = HF preprocessor config：add_snapping=False、add_cls_token=True、max_length=2000。
离线模型已加这两个开关；在一台装了 DCR 环境的机器上用真实 CM3PProcessor 逐窗口对照，
**10/10 张完全相等**：

    [verify] 10/10 charts match the real tokenizer exactly

命令（本机 400 张抽样；全量 21,949 在集群 CPU 作业里跑）：

    python scripts/cm3p_token_budget.py --osu-dir data/osu_cache --limit 400 \
        --workers 1 --audit 20 --max-length 2000 --add-snapping off --add-cls on

| 量 | legacy（真实 checkpoint 的默认） | canon | canon + columns |
|---|---|---|---|
| tokens / chart (mean) | 8,041 | 4,524 | 6,289 |
| tokens / note（pooled total/total） | 4.90 | 2.76 | 3.83 |
| tokens / window (mean) | 757 | 426 | — |
| tokens / window p50 / p95 | 709 / 1,509 | 409 / 801 | — |
| 撞 max_length=2000 的窗口 | 18 / 4,251 (0.42%) | 0 / 4,251 | 0 |
| pad_efficiency（排序分批，B=16） | — | 0.995 | — |
| 每窗 pad 到 2000 的 pad_efficiency | — | 0.213 | — |

* 裁剪节省 **43.7%** token；attention 的 n^2 约 **3.16x**。
* hold 跨 16 s 边界：978 / 49,801（1.96%）；end 距边界 <=2 ms：14。
* column token：0 / 705,943 note events → **ABSENT**。
* core audit：20 张图 core stream identical、0 problems、0 core tokens dropped。

与计划 §2 的差异：计划写 10,591 tokens/chart、7.47 tokens/note，那是「6 token/tap +
有 snapping + 4000」的算法；真实是「4 token/tap + 无 snapping + 2000」，所以绝对预算
天然低约 1/3，裁剪收益也更大（43.7% vs 27%，n^2 3.16x vs 1.8x）。
**绝对预算以这张表 / results/cm3p_rank_1k/token_budget_all.json 为准**，计划 §2 的数字
只能当历史假设。全量 21,949 由集群 CPU 作业 scripts/slurm/cm3p_budget.slurm 生成
（--audit 128 --strict）；本文件先给 400 张抽样。

> 真实 tokenizer 对照（DCR env；HPC 只是当时唯一装了 transformers 的机器）：
>
>     python scripts/cm3p_token_budget.py --osu-dir data/osu_cache --limit 20 \
>         --verify-cm3p 10 --max-length 2000 --add-snapping off --add-cls on
>
> 逐窗口比较离线模型与 CM3PProcessor 的真实 input_ids 长度；10/10 相等才把上面的表
> 当作预算。

---

## 6. 执行清单（HPC 只跑 GPU；本地是权威）

**执行位置已定**：本地仓库/数据/最终产物是权威；HPC 只负责 T2 的 GPU。
HPC 上的代码、数据、legacy 表副本都可以随时重传或丢弃，不作为唯一状态。
T2 结束必须把内容表、`.cache/cm3p/win/*.npy`、`index.parquet`、两份 `.meta.json`
全部回传本地；T3 在本机 CPU 做。

**T0/T1 的真实校验**：本机没有 transformers/slider/checkpoint，真实 tokenizer 的
逐窗对照需要一台装了 DCR 环境的机器（可以是 HPC，也可以是以后本地装好的环境）。
HPC 登录节点上跑过：

    python scripts/cm3p_token_budget.py --osu-dir data/osu_cache --limit 20 \
        --verify-cm3p 10 --max-length 2000 --add-snapping off --add-cls on

结果：[verify] 10/10 charts match the real tokenizer exactly。真实配置见 §0.1。
另外在 CPU 上用**真实 CM3PModel** 跑过 2 张图的端到端 T2（core audit 通过 → 窗口 .npy →
index → 表 → sidecar），所以非 GPU 路径也验证过；GPU 只是把 21,949 张跑快。
离线审计已被真实 tokenizer 校准到逐窗口相等，但**它不替代真实 checkpoint 的 T0 断言**。

**CPU 单图耗时（2026-11-05，HPC 登录节点，DCR env，CPU-only）**：

| 样本 | 窗口数 | `[win]` 导出 | 总 wall |
|---|---:|---:|---:|
| 1 张（422 notes） | 7 | 25 s | 68.7 s（含模型加载） |
| 4 张随机 | 42（均 10.5/图） | 360 s | 380.1 s（均 95 s/图） |

稳态约 **60–95 s/图**（约 **4–9 s/窗口**，取决于窗口 token 数和 CPU 负载）；首图还要加
约 20–40 s 的 checkpoint 加载。按 90 s/图估算，21,949 张纯 CPU 约 **23 天**，
所以全量必须用 GPU；A800 上 45–90 min 才是可接受量级。CPU 只适合 smoke test 和离线审计。

**legacy 表**：`data/processed/chart_content_cm3p_local.parquet` 以本地为准
（21,949 行、45,741,279 B、source=local-noaudio）。HPC 上那份只是副本，可丢弃。
T3 在本机做，所以 paired ΔNLL 可以在本机直接跑。**不要**把新 canon 表并进它——
merge guard 会拒绝，除非显式 `--allow-mixed-crop`。旧表只作历史参考，
**不是 A0 的复现目标**。

集群上的第一次 canonical run（DCR env）：

    PY=/hpc2hdd/home/aiaa3111_l01_008/.conda/envs/DCR/bin/python \
    CROP=canon WINDOWS=on \
    EMBED_OUT=data/processed/chart_content_cm3p_local_canon.parquet \
    WINDOW_DIR=.cache/cm3p/win WINDOW_INDEX=.cache/cm3p/index.parquet \
    MANIA_COLUMNS=off \
    sbatch -p i64m1tga800ue --gres=gpu:1 -n 8 --mem=64G --time=24:00:00 \
        scripts/slurm/cm3p_embed.slurm

（真实 max_length=2000，所以 T0 不再要求 --allow-config-mismatch；snapping/CLS 从
checkpoint 继承，不要手动打开。--on-truncate 默认 warn，受影响的图逐图写进 index。）

（slurm/remote 脚本的默认仍是 CROP=legacy WINDOWS=off，保持旧流程可复现；
探针阶段固定 --crop legacy --windows off，因为 mix gate 测的是 audio 域偏移，
不是 T1 的裁剪。）

第一次 canon run 固定 `MANIA_COLUMNS=off`：**A0 是 T3 内部对照，不要求它复现旧图级表，
也不要求它等于 0.61484。** T1 固定后，A1–A6 vs A0 的配对比较才是主判据。

**2026-11-09 下游实测（canon vs legacy 内容表）**：把 `13613023` 的 canon 表接到
P2/P4 的 5 个划分种子上（同协议、同划分文件、只换 content 表；fixed 信号不重训），
4 个 embedding 方法**一致变差**：P2 ΔNLL +0.0055…+0.0197、P4 +0.0047…+0.0208
nats，Pearson/Spearman 同向降 0.05–0.08，P2 的 train NLL 也变差 +0.007…+0.017。
方法排序不变（external MLP 仍最好）。即 canon 删掉的 hitsound/volume、kiai、SV
承载了与 msd/Sunny **互补**的难度信息；T1 的口径收益（token −43.7%、可缓存）
不等于下游收益。详见 [p4_multiseed_canon.md](p4_multiseed_canon.md)。

### 可选探针：checkpoint 到底见没见过 column？

公开仓库的代码从未发出列 token，但**已发布 checkpoint 的训练代码不一定等于公开仓库**。
用 256 张已发布图、保持其他裁剪标志为 legacy（隔离 column 这一个变量），各跑一次：

    # off（当前上游行为）
    python scripts/step0c_chart_content.py --source osu-dir \
        --ids-file manifests/cm3p_probe_1k.csv --osu-dir data/osu_cache --audio off \
        --crop legacy --windows off --mania-columns off --out /tmp/probe_off.parquet
    # on（opt-in patch）
    python scripts/step0c_chart_content.py --source osu-dir \
        --ids-file manifests/cm3p_probe_1k.csv --osu-dir data/osu_cache --audio off \
        --crop legacy --windows off --mania-columns on  --out /tmp/probe_on.parquet
    # 两者都与已发布向量比（step0e 会打印 cosine / mantel / recall@1）
    python scripts/step0e_validate_mix.py --computed /tmp/probe_off.parquet --json /tmp/off.json
    python scripts/step0e_validate_mix.py --computed /tmp/probe_on.parquet  --json /tmp/on.json

判读：

* on 的 cosine / recall@1 **明显高于** off → 已发布 checkpoint 很可能见过列 token
  （上游训练代码与公开仓库不同）。这时 canon 应该用 --mania-columns on，并重跑 T2/T3。
* 两者接近 → checkpoint 是 column-blind；保持 off，把「C 对列号盲」写进模型说明。

在做出这个判断之前，不要把 columns 当成「已经修好」：它是 OOD 输入，可能变好也可能变坏。

磁盘：windows 约 **240 MB**（21,949 x 10.6 x 512 x 2B，真实 windows/chart=10.63）
+ index 几 MB。

T2 产物要一起拷回本机（表 + .cache/cm3p/win + index + 两张 meta.json），因为 T3 是纯 CPU、
而 legacy 表（chart_content_cm3p_local.parquet）只在本机；paired ΔNLL 要求新旧信号在
同一台机器、同一批 pair 上算。集群只负责 GPU 的 T2。

---

## 7. OD 与谱面常数

按计划 §4：OD/HP/keycount 不进 CM3P 的输入，也不进 head；C 与 d 都是 OD 盲的。
下游要预测 ACC 必须自己接 OD（scripts/step2_cm3p_rank.py --extra-signals od,... 是现成通道）。
这一条写在 [docs/models.md](models.md) §3.5，避免以后把 d 当绝对难度。

---

## 8. 测试

    python scripts/test_cm3p_windows.py     # crop audit, buckets, exporter, column patch, offline accounting
    python scripts/cm3p_token_budget.py --limit 40 --audit 8 --strict

test_cm3p_windows.py 覆盖：T0 断言、BucketSampler 覆盖/桶隔离/时间序/pad、截断记账、
core audit（含故意丢 note 的负例）、exporter 的 npy/index/parity/resume/base_index、
mania column patch 的 1-based 修正、离线 .osu 会计与截断标记。
