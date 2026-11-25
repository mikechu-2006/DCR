# D2V 输入通道扩展：代码触点清单（read-only inventory）

> 任务：shared task-1（code-inv），2026-11-24。仓库 `/home/mikechu/Documents/Projects/DCR` @ `3d25625`。
> 本文只做静态盘点与本地只读测量；**没有修改任何源文件、没有启动训练**。
> 目标：把一个 8 通道（4 onset + 4 hold）@100 Hz 的 D2V 前端扩成可配置通道集
> （release 脉冲 / BPM·beat·SV·kiai / hitsound·set·volume 等，12 / 18 / 27 通道），
> 列出全部代码触点、必须改的东西、会坏的东西、以及“绝不能动”的契约。
>
> 行号均为当前工作树（未提交状态：另有并行任务在写 `data2vec_main/scripts/audit_attributes.py`，
> 该文件不在本文依赖内）。测量方法见 §12。

---

## 0. TL;DR

1. **通道数在“计算图”里只进入 3 个硬点**，模型主体/损失/池化/probe 数学完全不关心通道数：
   - `data2vec_main/d2v_main/charts.py:26-27`（`N_CHANNELS=8`, `HOLD_OFFSET=4`）以及 `build_roll`/`build_window` 里硬编码的 4 列 onset / 4 列 hold 与 `roll[:4]` clip；
   - `config.py:21` + 两个 YAML 的 `model.in_channels: 8`；
   - `model.py:39,41` 的 `Conv1d(in_channels, ...)`（实际值由 `model.py:85` 从 cfg 读）。
2. **真正的工作量与风险在 plumbing**：6 条 `state_dict` 载入路径、~8 处 config/CLI 固定点、
   2 个 parquet 契约、~6 个 deploy/表校验、~20 个脚本固定 `audio_style_base.yaml` / `main_holdout/last.pt`。
3. **EMA 不受影响**（重要发现）：`EMAState` 的 fp32 master 只覆盖 `TransformerStack`（`model.teacher`），
   **不包含 conv 任何参数**，所以扩 conv1 输入通道不需要动 EMA（`model.py:148-152` 实测 master 26 个张量、首项是 `in_proj_weight`）。
4. **优化器需要处理**：`train.py:113-116` 的 AdamW 含 `conv.conv1.weight`；因根级参数 `mask_emb` 排第一，
   conv1.weight 的 optimizer state 索引是 **1**（本地 smoke ckpt 实测），扩张要按索引/形状重建 moments。
5. **checkpoint 拷贝路径必需**：旧 8ch ckpt（`[256,8,5]`）载入新 12/18/27ch 模型，在
   `train.py:123`（strict=True）、`extract.py:62`（默认 strict）、`step2_chart_encoder.py:685,968`（strict=True）
   会直接 RuntimeError；`step2_chart_encoder.py:224`（strict=False）也**不会**静默跳过 —— shape mismatch 仍然抛错。
6. **数据侧还缺原料**：`dcr/cm3p_offline.py` 当前不暴露 hitsound / sample-set / per-note or timing-point volume；
   kiai 用字符串相等解析 effects，会漏掉 `effects==9`（实测 1.04% 的 timing-point 行）；
   SV 只以负 beatLength 形式存在（`TimingPoint.ms_per_beat`），需自行换算；`sv_groups`/`timing_groups` 已有可复用公式。
7. **推荐姿势**：新增独立 `configs/audio_style_attr.yaml`（`data.channels` spec + 派生 `model.in_channels`），
   **不要原地改 `audio_style_base.yaml`**；否则所有 legacy 8ch ckpt 的 score/resume/E1/E2 路径会连锁失效。
8. 实测性能：`build_window` 0.44 ms/1600 帧窗口；加 release+2 个 impulse 通道后 0.55 ms（+37%），
   但整个 sample 时间里 `parse_chart` ~11.9 ms 占主导 → 训练数据管线影响 ~1%；抽取数据准备 ~+10-15%。
   通道数 8→27 只给总前向 FLOPs 增加 ~0.1%（conv1 只占 conv 的 2-5%）。

---

## 1. 盘点范围与方法

- 静态阅读：`data2vec_main/d2v_main/*.py`、`data2vec_main/configs/*.yaml`、`data2vec_main/scripts/*`、
  `scripts/step2_chart_encoder.py`、`scripts/e1_frozen_window_control.py`、`scripts/run_p2_p4_multiseed.py`、
  `scripts/step2_cm3p_rank.py`、`scripts/*d2v*`、`scripts/slurm/*d2v* / *behavior*`、
  `deploy/*`、`dcr/cm3p_offline.py`、`data2vec_main/README.md`、根 `README.md`、`docs/*.md`。
- 检索：`N_CHANNELS|in_channels|HOLD_OFFSET|build_window|build_roll|parse_chart|slice_window|512|embedding`；
  `grep` 于 2026-11-24 工作树。
- 只读测量（本地 CPU，无 GPU）：parse/build 计时、checkpoint/optimizer/EMA 结构、parser effects 位掩码统计（§12）。

---

## 2. 前端：`data2vec_main/d2v_main/charts.py`

| file:line | item | required change | risk |
|---|---|---|---|
| `charts.py:4` | 模块 docstring "dense 8-channel signal (4 onset + 4 hold)" | 改为 spec 描述 | L（文档） |
| `charts.py:26` | `N_CHANNELS = 8` 模块常量 | 删除/改为函数参数；由 `ChannelSpec` 决定输出通道数 | **H**：build_roll/build_window/所有调用方同源 |
| `charts.py:27` | `HOLD_OFFSET = 4` | 保留为 spec 内 base block 的偏移，只在 base 8ch 兼容层使用 | M：旧 ckpt 语义依赖 0..3 onset / 4..7 hold |
| `charts.py:34-38` | `NoteEvent(t, col, end)`，frozen dataclass，无 slots | 加 `hitsound:int`、`sample_set:int`、`volume:int`（或把 per-note 属性拆成并行数组）；建议 `slots=True` | M：每 event dict 96B→~184B；cache 内存约翻倍（§9.2 实测 90KB/图） |
| `charts.py:45-46` | `@lru_cache(maxsize=512)` + `parse_chart(path, frame_hz)`，返回 `(events, n_frames)` | **缓存键保持与通道 spec 无关**（parse 返回全部原始属性，spec 只影响 roll 构建）；若 parse 输出依赖 spec，必须把 spec hash 加进缓存键 | **H**：键不变可避免重建；若 spec 进 parse 则所有 cache 语义变化 |
| `charts.py:46-69` | `parse_chart` 只取 `circle/hold`，丢弃 timing points、hitsound、sample set、volume | 返回/附带 `timing: tuple[TimingPoint,...]`（或压缩数组）与 per-note 属性；保持 `NoteEvent` 兼容旧字段 | M：lru 缓存对象变大；`parse_chart` 被 data.py/index_cache.py/step2_chart_encoder.py/e1_frozen_window_control.py/smoke_test.py 多处直接调用 |
| `charts.py:68` | `n_frames = max(end or t)+1`（仅 note 派生） | **建议保持不变**（见契约 C4）；新属性通道不得延长 n_frames | **H**：改 n_frames 会改窗口网格/索引缓存/抽取窗口数 218,525 基线 |
| `charts.py:72-77` | `gaussian_kernel(sigma_ms, frame_hz)` | release 脉冲复用同一 kernel；per-frame 通道不需要 kernel | L |
| `charts.py:80-106` | `build_roll`：整图 roll | 参数化 `n_channels`/spec；hold/onset 逻辑抽成 base block | M（`build_roll` 仅 smoke_test 用） |
| `charts.py:82` | docstring "(8, n_frames)…Channels 0..3 onset, 4..7 hold" | 改成 spec 顺序表 | L |
| `charts.py:84` | `np.zeros((N_CHANNELS, n_frames))` | `(spec.n_channels, n_frames)` | H |
| `charts.py:91,122` | `if not 0 <= col < 4: continue`（4 列硬编码） | 用 `spec.n_cols`（4K 下仍 4，但参数化） | L |
| `charts.py:99-104` | hold 通道写入 `HOLD_OFFSET+col`，`np.maximum(...,1.0)` | 保留 base block；新增 release 脉冲块（建议 gaussian add，位置=hold end） | M：release 是 per-event 追加，走同一 python 循环（性能 §9.2） |
| `charts.py:105` | `np.clip(roll[:4], 0.0, 4.0)` | 只 clip onset block（spec.onset_slice）；若 clip 全部通道会破坏 per-frame 属性值域 | **H**：当前 `roll[:4]` 是隐式“前 4 通道=onset”契约 |
| `charts.py:109-139` | `build_window`（训练/抽取主路径） | 同上；新增 `attributes`/spec 参数；per-frame 属性用向量化填充，per-note 属性在同一 event 循环里追加 | **H**：所有数据都走这里 |
| `charts.py:112` | docstring | 更新 | L |
| `charts.py:115` | `np.zeros((N_CHANNELS, window_frames))` | spec 通道数 | H |
| `charts.py:120-125` | `for ev in events:`（遍历**整图** event，只对窗口内做写入） | 保留结构；新增属性写入必须加在窗口判定内 | M：每窗口 O(n_events)，§9.3 实测 |
| `charts.py:132-137` | hold 区间 `np.maximum` | 保留 | L |
| `charts.py:138` | `np.clip(roll[:4])` | 同 105 | H |
| `charts.py:142-152` | `slice_window`：docstring 说 (8,W)，实现用 `roll.shape[0]` 其实通道无关；**全仓无调用方（dead code）** | 更新 docstring 或删除 | L |
| `charts.py:155-164` | `window_starts` | 不变 | — |
| `charts.py:167-196` | `ChartStore`（id→.osu path） | 不变 | — |

---

## 3. 数据集/批处理：`data2vec_main/d2v_main/data.py`、`masking.py`

| file:line | item | required change | risk |
|---|---|---|---|
| `data.py:21-30` | `ChartWindowDataset.__init__` 从 `cfg["data"]` 读 frame_hz/window/sigma/min_valid；**不读通道 spec**（通道数来自 charts 常量） | 新增 `self.channel_spec = cfg["data"].get("channels", BASE8)`，传给 `build_window`；同时 assert spec.n_channels == model 期望 | **H**：当前 data 与 model 通道数没有 assert，配错只在第一个 batch forward 时炸 |
| `data.py:35-42` | 用 `index_cache.load` 或 `parse_chart` 拿 `n_frames` | 不变（前提 n_frames 定义不变，契约 C4） | M |
| `data.py:44-56` | 过滤 `n_frames < min_valid`；eval 生成 `self.index` 窗口网格 | 不变（窗口网格不能变，否则 218,525 窗口基线不可比） | **H** |
| `data.py:68-84` | `__getitem__`：train 随机窗口 / eval 预生成；`76-77` `parse_chart`+ `build_window` | 传 spec / 属性数据；若 release 或 hitsound 需要 extra parse 数据，一并从 parse 结果取 | H |
| `data.py:75-76` | eval 分支同一图调用两次 `parse_chart`（LRU 命中，无实质开销） | 可合并，非必须 | L |
| `data.py:87-93` | `collate_windows` 只 stack `x/valid_len/chart_id/start` | 不变（通道维在 x 里） | — |
| `masking.py:13-27` | span mask 只在时间轴 (B,T)，与通道无关 | 不变 | — |

---

## 4. 模型：`data2vec_main/d2v_main/model.py`

| file:line | item | required change | risk |
|---|---|---|---|
| `model.py:3-8` | docstring "piano roll (B, 8, 100 Hz)" | 文档更新 | L |
| `model.py:39` | `ConvFeatureEncoder.__init__(in_channels: int = 8, ...)` 默认值 8 | 去掉默认值（强制显式传），避免 cfg 漏配时静默 8ch | **H**：静默默认是配错来源 |
| `model.py:41` | `nn.Conv1d(in_channels, hidden, k=5)`，权重形状 `[256, C, 5]` | 无需改逻辑；这是唯一 checkpoint 形状变化的参数 | H（checkpoint 兼容，见 §10.7） |
| `model.py:46-50` | forward 3 层 conv + LayerNorm | 不变（通道只影响 conv1 输入维） | — |
| `model.py:78-93` | `Data2VecModel.__init__`；`85` 用 `cfg["model"]["in_channels"]`；`89` teacher=deepcopy(student) | 建议改为由 `cfg["data"]["channels"]` 派生（单一真源），并 assert 两者一致；teacher 只复制 TransformerStack，**不复制 conv**（conv 由 student 共享，见 `_features`） | **H**：两处独立配置是最大静默错配风险 |
| `model.py:100-107` | `_features`：student/teacher 共享 `self.conv` | 不变；扩通道后 teacher 自动看到同样输入 | — |
| `model.py:129-138` | `parameter_count` | 数值随 C 微增（+1280×(C-8) ≈ 24k @27ch） | L |
| `model.py:141-173` | `EMAState`：`master=[p for p in teacher.parameters()]`；update zip(master, teacher, student) | **无需改**（teacher 是 TransformerStack，无 conv 参数）；实测 smoke ckpt master=26 tensors，无 conv | — |
| `model.py:176-181` | cosine LR | 不变 | — |

---

## 5. 配置：`config.py`、`configs/*.yaml`

| file:line | item | required change | risk |
|---|---|---|---|
| `config.py:12-19` | `data` 默认块（osu_dir/window_sec/frame_hz/sigma/min_valid） | 加 `channels: {...}` spec（有序、带名字/版本）；默认保持 base8 | **H**：所有入口读 DEFAULTS 合并 |
| `config.py:21` | `model.in_channels: 8` | 由 spec 派生（或保留但 assert 相等）；**不要再让用户手填两份** | **H** |
| `configs/audio_style_base.yaml:12` | `in_channels: 8`（主实验基线，被所有脚本/E1/E2/benchmark 引用） | **不要原地改**。新增 `configs/audio_style_attr.yaml`（或 attr12/attr18/attr27） | **H**：原地改会让 legacy 8ch ckpt 的 score/resume/E1/E2 全部 shape mismatch，基线不可复现 |
| `configs/audio_style_smoke.yaml:11` | `in_channels: 8` | smoke 增加一个 attr smoke 配置（可选） | M |
| `config.py:62-72` | `load_config`/`_deep_update`，`dump_config` | 若 spec 是 list[dict]，deep_update 对 list 是整体替换（可行）；建议 dump 稳定排序便于 hash | M |

---

## 6. 预训练：`data2vec_main/d2v_main/train.py`

| file:line | item | required change | risk |
|---|---|---|---|
| `train.py:44-61` | CLI：`--config/--osu-dir/--ids-file/--out-dir/--epochs/--resume/...`，**无通道相关参数** | 加 `--in-channels/--channel-spec`（或只靠 config）；`--resume` 加 expand 支持 | M |
| `train.py:110` | `model = Data2VecModel(cfg)` | 由 spec 派生 in_channels（§5） | H |
| `train.py:111` | `ema = EMAState(model.teacher)` | 不变（无 conv） | — |
| `train.py:113-116` | AdamW 参数 = `[p for p in model.parameters() if p.requires_grad]`；含 conv1.weight（实测索引 1：根级 `mask_emb` 索引 0） | resume 时 expansion：对 conv1.weight 索引重建 moments（§8.2） | **H**：`opt.load_state_dict` 对 shape 不符直接报错 |
| `train.py:117-119` | `steps_per_epoch × epochs` → `total_steps`、warmup | 续训时若沿用原 20 epoch 配置，`total_steps`=25,720 而 `start_step`=25,720 → cosine 立刻落到 min LR。需按“原始+新增 epochs”传 `--epochs 40` 或显式给 `--total-steps` | **H**：否则续训学习率曲线错误 |
| `train.py:121-127` | resume：`model.load_state_dict(ckpt["model"])`（strict=True）、`opt.load_state_dict`、`ema.load_state_dict`、`start_step` | 加 expand helper + 清晰报错（ckpt in_channels vs cfg）；ema 可直接 load（无 conv） | **H**：当前 8ch→12ch resume 必然 RuntimeError |
| `train.py:143-196` | 训练循环：`146` `b,_,t_in=x.shape`；mask/teacher/loss 均通道无关 | 不变 | — |
| `train.py:190-195` | 中间/last ckpt 保存 | 不变 | — |
| `train.py:204-207` | `save_ckpt`：`{"model","optimizer","ema","step","cfg"}` | **保留 cfg**（含新 spec）→ 后续 expand/extract 可读；建议加 `param_names` 或 `in_channels` 顶层字段便于自检 | M |
| `train.py:99` | 打印 git rev | 不变 | — |

---

## 7. 抽取 + parquet 契约：`data2vec_main/d2v_main/extract.py`

| file:line | item | required change | risk |
|---|---|---|---|
| `extract.py:6-7` | 模块 docstring 定义 parquet 契约（beatmap_id, embedding 512d L2, n_windows, checksum_ok, source, d2v_rev） | 契约**不应变**（下游 probe/deploy 依赖）；新增 spec 只进 `source` 字符串 | **H** |
| `extract.py:26-40` | CLI：`--config/--ckpt/--out/--osu-dir/--ids-file/--tag` | `--tag` 必填新值（如 `d2v_main_attr12_v1`）；可加 `--channel-spec` 覆盖 | M |
| `extract.py:56-57` | `ChartWindowDataset(store, cfg, train=False, ...)` | 随 §3 传 spec | H |
| `extract.py:60-62` | `model = Data2VecModel(cfg)`；`model.load_state_dict(ckpt["model"])`（默认 strict=True） | **优先从 ckpt["cfg"] 读 in_channels/spec**（train save_ckpt 已存 cfg），CLI config 仅兜底 + assert；否则“训练用 attr config、抽取用 base config”直接 RuntimeError | **H** |
| `extract.py:64-82` | 逐窗口 forward、按 `valid_out` 加权、按图累计 | 不变（通道无关） | — |
| `extract.py:85-99` | 行：`source=args.tag`（97）、`d2v_rev=git_rev`（86,98）、`checksum_ok=True`（96） | 新表用新 tag；`d2v_rev` 在未提交工作树上可能不区分 spec → 必须靠 tag 区分 | M |
| `extract.py:100-103` | 写 parquet（不建目录外的东西） | 新输出路径，**不得覆盖** `data/processed/chart_content_d2v_holdout.parquet` | **H**：所有 baseline probe/benchmark 引用该文件 |
| `data2vec_main/scripts/smoke_test.py:106-117` | 断言 parquet **列名精确等于** 6 列、embedding norm=1、n_windows>0 | 契约测试；扩通道**不要加列**，否则 smoke 与下游一起炸 | **H** |
| `scripts/slurm/benchmark_d2v_extract.slurm:22-28` | benchmark 抽取：`--osu-dir data/benchmark_d2v_osu --ids-file data/processed/benchmark_d2v_ids.txt --out data/processed/benchmark_content_d2v.parquet --tag benchmark_d2v` | 新 arm 并行一套（新 osu staging 可复用，新 out/tag） | M |
| `scripts/finalize_benchmark_d2v.py:42-44` | 断言 `dims == {512}` | 只要 embedding 仍是 512d 无需改；若哪天改 dim 则这里硬失败 | M |
| `scripts/finalize_benchmark_d2v.py:2-8` | 说明 extract 的 beatmap_id 是 staging 数字文件名 | 契约不变 | — |

---

## 8. 索引缓存 / 划分：`index_cache.py`、`splits.py`

| file:line | item | required change | risk |
|---|---|---|---|
| `index_cache.py:23-24` | `default_cache_path(frame_hz)` → `data2vec_main/cache/chart_index_100.json` | 如果 n_frames 定义不变：**零改动**。若变：路径加 spec 或 meta 加 spec hash | **H**（静默 stale） |
| `index_cache.py:27-37` | `_meta`：osu_dir, n_files, max_mtime_ns, frame_hz | 若通道 spec 影响 n_frames，必须把 spec hash 加进 meta；否则旧缓存仍被判有效 | **H** |
| `index_cache.py:40-62` | `build` 逐图 `parse_chart` 取 n_frames；O(21949) | 重建成本：本机 ~4 min，HPC 登录节点 881.6 s（`data2vec_main/README.md:103-107`） | M |
| `index_cache.py:65-79` | `load`：meta 全等才复用，否则 rebuild | 同上 | H |
| `index_cache.py:82-93` | CLI `--osu-dir/--frame-hz/--out/--force` | 不变 | — |
| `splits.py:39-86` | `--protocol all|complement|holdout`，写 `pretrain_ids.txt`/`eval_ids.txt`/`summary.json` | **无通道依赖**；新 arm 必须复用 `splits/holdout`（20,584 / 1,365, seed 0），不要重新生成 | **H**：重生成会破坏 clean holdout 口径与可比性 |
| `data2vec_main/splits/holdout/summary.json` | 已存在（n_pretrain 20584 / n_eval 1365） | 复用 | M |
| `data2vec_main/runs/smoke/*` | 本地唯一 train.py 产物（8ch tiny，conv1 [32,8,5]） | 作为 expand 单测样本 | — |

---

## 9. 性能关键路径（新增 per-frame 属性通道）

### 9.1 实测基线（本地 CPU，200 张图，§12）
- `parse_chart`：median 9.16 ms / mean 11.89 ms / max 89.72 ms（与 README “约 12 ms/图”一致）。
- `build_window(1600 帧)`：median 0.36 / mean 0.44 / max 1.66 ms；平均 8.96 窗口/图。
- 训练每个 sample 的真实成本 ≈ parse + 1 次 build_window；DataLoader 4 workers 时 ~75 samples/s/worker（300/s 总），
  与 README §5 item 5（`data2vec_main/README.md:112-114`）一致 → **parse 主导**。
- 抽取：每图 parse 一次 + ~9 次 build_window；build_window 占比 ~25%。

### 9.2 加通道的成本结构
- **per-note 类**（release 脉冲、hitsound/sample/volume 脉冲）：必须进 `build_window:120` 的 python 事件循环 → 每 event 增加 slice/kernel 操作。
  原型测量（20 通道 + release + 2 个 impulse）：0.399 → 0.549 ms/窗口（**+37%**）。
- **per-frame 类**（BPM/beat phase/measure/SV/kiai/TP volume）：不要 per-event 循环；用 `np.searchsorted(timing_offsets, frame_times)` 向量化。
  实测单属性 1600 帧填充 25.7 µs；且 BPM/SV/kiai/volume 共享同一组 timing offsets → 一次 search + 多次 take，整体 ~30-40 µs/窗口（相对 build_window +7-9%）。
- **beat/measure 相位**：BPM 变化时才需要分段积分；窗口内通常 0-3 条 red line，用 searchsorted + 少量段循环即可，别逐帧循环。
- 期望总开销：训练 sample ~+1%（parse 主导）；抽取数据准备 ~+10-15%。
- **不要**在 LRU 缓存里存 per-frame 稠密属性数组：事件约 970/图、cache 实测 ~90 KB/图（512 项 ≈ 46 MB/worker）；
  若加稠密 100 Hz 属性数组（~13.5k 帧 × 10 attr × 4B ≈ 0.54 MB/图）→ 512 项 ≈ 276 MB/worker，6 workers 爆内存。
  推荐：缓存 `(events, timing_points, sparse note attrs)`，per-frame 数组按窗口现算；或把 `NoteEvent` 改 `slots=True` 并调小 maxsize。
- 计算图成本可忽略：conv1 MAC/sample = C×256×5×1600（C=8→16.4M，C=27→55.3M，+38.9M）；总前向 ≈39 GMAC/sample → **+0.1%**。
- 输入 tensor 内存/带宽：`(B,C,1600)`，C=27 时是 8ch 的 3.4×（batch16：0.8MB→2.8MB/step），H2D 与 pin_memory 影响 <1%。`extract --batch-size 32` 同理可忽略。

### 9.3 index_cache 受影响的判定
- **只有 n_frames 定义变化时**才需要重建/换 key（契约 C4 建议不变）。
- 若只改通道构建（parse 返回更多字段但 n_frames 不变）：缓存文件、窗口网格、`n_windows`、抽取窗口总数（218,525）全部保持，基线可比。
- 若把 kiai/timing 延伸到 last note 之后来定 n_frames：窗口数、图向量、`n_windows` 全部变化 → 需新 cache 文件 + 新 extract + 所有 probe 重跑，且与基线不可比（不建议）。

---

## 10. in_channels=12/18/27 时会坏什么（逐函数/CLI/契约）

> 通道数本身不改变任何数学；坏点全部是 shape/契约/版本固定点。

### 10.1 数据 → 模型
1. `charts.build_window` 若仍输出 8ch、`model.conv1` 期望 C>8 → 首个 batch `RuntimeError: expected input ... to have C channels`。
   必须参数化 build_window 并加启动 assert（`data.py:21-30`）。
2. `Data2VecModel(cfg)` 的 `in_channels` 与 `data.channels` 是两套值（`model.py:85` vs `charts.py:26`）→ 必须单一真源 + assert。
3. `ChartWindowDataset` 无 spec 参数（`data.py:21`）→ 需加。
4. smoke `roll.shape == (8, n_frames)`（`smoke_test.py:49`）与 `roll[:4]/roll[4:]`（:50-53）→ 必须参数化断言。

### 10.2 checkpoint 载入路径（6 条）
| # | 路径 | 行为（8ch ckpt → 12ch model） | 需要的改动 |
|---|---|---|---|
| 1 | `train.py:123` resume | strict=True → RuntimeError shape mismatch | expand helper + 明确报错 |
| 2 | `extract.py:62` | 默认 strict=True → RuntimeError | 从 ckpt cfg 推 spec，或先 expand |
| 3 | `step2_chart_encoder.py:222-226` E1 init（`--init d2v`） | strict=False **仍会**因 size mismatch 抛 RuntimeError | 在 load 前 expand；或按 ckpt conv1 形状推导 in_channels |
| 4 | `step2_chart_encoder.py:968` E1 resume | strict=True → RuntimeError | expand（conv 冻结时 optimizer 无需处理） |
| 5 | `step2_chart_encoder.py:685` E1 score-only | strict=True → RuntimeError | 旧 ckpt 必须用旧 config/8ch model 打分；新 plan 要保留 legacy 路径 |
| 6 | `scripts/e1_frozen_window_control.py:280-286` | strict=True → RuntimeError | 同上；另其特征缓存 `features.npz` 的 meta 只含 window/frame/sigma/min_valid（:269-277），**不含通道 spec** → 换通道后会静默复用旧特征 |

### 10.3 优化器 / EMA
- optimizer（`train.py:113-116`）：`conv1.weight` 形状 `[256,8,5]`→`[256,C,5]`。`opt.load_state_dict(ckpt["optimizer"])` 会因 shape 不符失败。
  AdamW state 实测：`state[0]`=mask_emb（`[64]`，索引 0），`state[1]`=conv1.weight（`[32,8,5]`）。扩容时对 conv1.weight 对应索引把 `exp_avg/exp_avg_sq` 重建为
  `zeros(new_shape)`、前 old_C 列拷贝旧值（或直接零初始化）、`step` 保留；其它参数 state 按索引不变。
- E1 的 optimizer 只在 `--freeze none` 时包含 conv；`topk/all` 时 conv 不训练 → 无 opt 影响（`step2_chart_encoder.py:188-209`）。
- **EMA：无影响**。`EMAState.master` 只存 `model.teacher`（`TransformerStack`）的参数，conv 不参与 EMA（`model.py:89,148-162`；
  smoke ckpt 实测 26 个 master tensors = 2 层 Transformer + norm）。共享 conv 的 student/teacher 自动看到同一新 conv。
- 参数计数/日志（`model.parameter_count`、E1 打印 `trainable=`）会 +1280×(C-8)（可忽略，但报告里会变）。

### 10.4 抽取 / parquet / 下游
- `extract.py` 的 parquet **schema 不能变**（6 列，512d）→ 新 spec 只能进 `source` tag；smoke_test 精确列断言（:112-113）。
- 新内容表必须新文件 + 新 tag；`deploy/models/d2v_linear_extra.json` 的 `content_md5`（:1565）会拒绝新表（guard 在 `deploy/export_dcr_table.py:129-135`）。
- 新内容表 → benchmark 需要：重跑 `benchmark_d2v_extract.slurm`（新 out）、`finalize_benchmark_d2v.py`（dims==512 通过）、
  `benchmark_d2v_train.slurm`（新 `D2V_*_signals` 列名）、`deploy/export_dan_deploy.py` 重出 JSON（`content_md5` 新）、`deploy/test_parity.py`。
- `scripts/benchmark_d2v_compare.py:81,116-119,150,172` 硬编码 `D2V_L_isotonic/D2V_L_d` 列名 → 新 arm 要么复用命名（覆盖）要么参数化。

### 10.5 脚本/CLI 固定点（全部写死 `audio_style_base.yaml` 和/或 `runs/main_holdout/checkpoints/last.pt`）
| file:line | 固定内容 | 影响 |
|---|---|---|
| `data2vec_main/scripts/train_hpc.sbatch:25-28` | `--config configs/audio_style_base.yaml --ids-file splits/holdout/pretrain_ids.txt --out-dir runs/main_holdout` | 新 arm 复制 sbatch（新 config/out-dir） |
| `data2vec_main/scripts/extract_hpc.sbatch:17-20` | base yaml + `runs/main_holdout/...last.pt` + out `chart_content_d2v_holdout.parquet` | 不能覆盖 out |
| `data2vec_main/scripts/bench_hpc.sbatch:24-26` | base yaml，120-step bench | 新 arm 复制 |
| `data2vec_main/scripts/run_p2_probes.sh:15-18` | 4 个 content 表路径（d2v_holdout / cm3p_canon） | 新表加一行 |
| `data2vec_main/scripts/run_p2_extra.sh:16-19` | 同上 + linear_extra | 新表加一行 |
| `data2vec_main/scripts/run_p4_probes.sh:8-14` | `run_p2_p4_multiseed.py --protocols p4 --seeds 1..5 --models linear,linear_extra --content-path chart_content_d2v_holdout.parquet` | 新表加一行 |
| `data2vec_main/scripts/run_p4_hold_probes.sh:8-16` | 同上 + `--split-p4 data/processed/step1_1k_pairbest_p4_d2vhold` | 新表加一行 |
| `data2vec_main/scripts/run_mlp_probes.sh:8-16` | `--models mlp,mlp_extra` | 新表加一行 |
| `data2vec_main/scripts/run_randommerge_d2v.sh:5-12` | randommerge split + d2v_holdout | 新表加一行 |
| `data2vec_main/scripts/hpc_mlp_probes.sbatch:15-25` / `hpc_randommerge_probes.sbatch:15-21` | 同上，HPC 版 | 新表加一行 |
| `scripts/slurm/benchmark_d2v_extract.slurm:22-28` / `benchmark_d2v_train.slurm:19-30` | benchmark D2V 编码/训练 | 新 arm 复制 |
| `scripts/slurm/d2v_baseline_p2.slurm:22-31` / `d2v_baseline_p4.slurm:25-42` | Experiment3 baseline（内容 = cm3p，d2v 表作 d2v_path） | 不直接改 |
| `scripts/slurm/behavior_e1_train.slurm:24-33`, `behavior_e1_fix_train.slurm:23-40`, `behavior_e1_fix_control.slurm:20-30`, `behavior_e1_fix_control_cpu.slurm:17-27`, `behavior_e1_bench.slurm:21-33`, `behavior_e1_pilot_debug.slurm:24-46`, `behavior_e2_train.slurm:37-48` | 全部 `--init d2v --ckpt data2vec_main/runs/main_holdout/checkpoints/last.pt --config data2vec_main/configs/audio_style_base.yaml` | 新 ckpt 意味着新 E1/E2 arm；旧 arm 必须继续用旧 ckpt 可复现 |
| `scripts/slurm/behavior_e1_fix_legacy_score.slurm:25-30` | `--score-only --checkpoint runs/behavior/e1/$ARM/s1/checkpoint.pt --config ...base.yaml` | 旧 8ch E1 ckpt 打分路径；新 config 化后必须保留可复现（用 ckpt 推断 in_channels） |

### 10.6 deploy / 表
| file:line | 契约 | 12/18/27ch 下是否破坏 |
|---|---|---|
| `deploy/dan_predict.py:42-44` | `CONTENT_DIM=512`、`EXTRA_NAMES`、`VECTOR_DIM=514` | **不破坏**（只要 encoder 输出仍 512d、L2）；但 `d2v_to_dan` 只接受旧 JSON 的 mu/sd |
| `deploy/dan_predict.py:47-50,60-65` | 读 `models/d2v_linear_extra.json`，校验 content_dim/extra_names/len(w) | 新内容表必须配新 JSON（旧 JSON 的 content_md5 不匹配） |
| `deploy/models/d2v_linear_extra.json:6` | `content_dim: 512` | 不破坏 |
| `...json:1565-1566` | `content_md5` / `train_git_commit` pin 住旧内容表 | 新表 → 新 JSON；**不要覆盖**该文件（baseline parity） |
| `deploy/export_dcr_table.py:47-49` | `CONTENT_DIM=512`、`VECTOR_DIM=514` | 不破坏 |
| `deploy/export_dcr_table.py:129-135` | `md5(parquet bytes) != model.content_md5` → exit 2 | 新表必须重出 JSON 才配得上 |
| `deploy/export_dcr_table.py:140,171-173,191` | 读 `["beatmap_id","embedding"]`，dim==512，拼 `[C, msd, sunny]` | 不破坏（embedding 仍 512） |
| `deploy/test_parity.py:27-34,42-64` | 用 `data/processed/benchmark_content_d2v.parquet` + `results/benchmark/pred/D2V_L.parquet` | 新 arm 需要新的 pred/content 对，否则 parity 仍测旧 D2V |
| `deploy/table/dcr_content.bin` + `dcr_content.manifest.json`（manifest `content_md5`, `model_id`） | 产品表当前是 **CM3P**（manifest model.content_type=cm3p） | 若新 D2V 要进产品表，需要新 JSON + 重出表；研究阶段无需 |
| `deploy/README.md:43,192-193` | D2V 512-d 契约文档 | 文档更新（仅当换 encoder 输出） |

### 10.7 checkpoint 拷贝/扩容配方（8ch → C ch，零初始化新列）

精确事实（本地 `runs/smoke/checkpoints/last.pt` 实测）：
- model state：`conv.conv1.weight` = `[hidden, old_C, 5]`、`conv.conv1.bias` = `[hidden]`；其余参数形状全部不变。
- `optimizer["param_groups"][0]["params"]` 的顺序 = `[p for p in model.parameters() if p.requires_grad]`；
  索引 0 = `mask_emb`（根级 Parameter 排最前），**索引 1 = `conv.conv1.weight`**（实测 `state[1].exp_avg.shape == [32,8,5]`）。
- `ema.master` **不含 conv**（只有 TransformerStack）→ EMA 无需扩容。
- train ckpt 顶层：`{"model","optimizer","ema","step","cfg"}`；`cfg["model"]["in_channels"]` 记录原值。

推荐实现（新模块如 `d2v_main/checkpoint.py`，供 train/extract/step2/e1_control 共用）：

```python
import torch

def expand_model_state(sd: dict, new_in: int) -> dict:
    """Zero-init new input channels of conv1. Function-preserving at step 0."""
    w = sd["conv.conv1.weight"]                 # [hidden, old_in, k]
    old_in = int(w.shape[1])
    if new_in == old_in:
        return sd
    if new_in < old_in:
        raise ValueError(f"cannot shrink conv1: {old_in} -> {new_in}")
    nw = torch.zeros(w.shape[0], new_in, w.shape[2], dtype=w.dtype)
    nw[:, :old_in] = w
    sd["conv.conv1.weight"] = nw
    return sd

def expand_train_ckpt(ckpt: dict, new_in: int) -> dict:
    sd = ckpt["model"]
    w = sd["conv.conv1.weight"]                 # [hidden, old_in, k]
    old_shape, old_in = tuple(w.shape), int(w.shape[1])
    if new_in == old_in:
        return ckpt
    if new_in < old_in:
        raise ValueError(f"cannot shrink conv1: {old_in} -> {new_in}")
    nw = torch.zeros(w.shape[0], new_in, w.shape[2], dtype=w.dtype)
    nw[:, :old_in] = w
    sd["conv.conv1.weight"] = nw
    opt = ckpt["optimizer"]
    for gi, group in enumerate(opt["param_groups"]):
        for pos, pid in enumerate(group["params"]):
            st = opt["state"].get(pid)
            if st is None or tuple(st["exp_avg"].shape) != old_shape:
                continue                        # 只有 conv1.weight 命中（形状唯一）
            for key in ("exp_avg", "exp_avg_sq"):
                nz = torch.zeros_like(nw)
                nz[:, :old_in] = st[key]
                st[key] = nz
            print(f"[expand] optimizer state pid={pid} group={gi} pos={pos} expanded")
    if "cfg" in ckpt:
        ckpt["cfg"]["model"]["in_channels"] = int(new_in)
    return ckpt
```
（推理路径只要 `expand_model_state`；E1 的 `--freeze topk/all` ckpt 里 optimizer 不含 conv，命中不到任何 state，自动无操作。）

要点：
- **EMA**：`EMAState.load_state_dict` 不需要动；它在 `train.py:111` 用 `model.teacher` 重建，teacher 里的 conv 是与 student **共享的同一个 module**，
  所以新 conv 对 teacher 立即生效（`model.py:100-107`）。
- **E1 ckpt**（`step2_chart_encoder.py:998-1012`）：`"model"` 同样是 state_dict（key 名相同）→ 只扩容 model state；
  `"opt"` 只在 `last.pt` 且 **conv 可训练**（`--freeze none`）时含 conv：`topk/all` 时 conv 不在 optimizer 里，无需处理。
  注意 E1 ckpt 没有 `cfg`，只有 `args.config` 路径 → in_channels 从 `conv1.weight.shape[1]` 推断。
- **extract.py**：只做推理，模型 state 扩容即可（不需要 optimizer/EMA）；更稳的是读 `ckpt["cfg"]` 决定 in_channels，CLI config 仅兜底。
- **必须断言**：扩容后、load 前 `assert model.conv.conv1.in_channels == spec.n_channels == cfg["model"]["in_channels"]`，
  并在 `data.py` 里 assert `x.shape[1]`；否则 8ch/12ch 混搭只能靠第一个 batch 的 RuntimeError 暴露。
- **续训 schedule**：沿用 `train.py:117-119` 的 `total_steps = steps_per_epoch × epochs` 时，`--epochs` 必须写成
  “原 epochs + 新增 epochs”（如 20→24），否则 `start_step ≥ total_steps`，cosine 直接落到 min LR；
  或给 train.py 加显式 `--total-steps`。
- **等价性**：新列权重为 0 ⇒ 第 0 步 student/teacher 前向对任意输入（含非零新通道）都与旧 8ch 模型逐位相同；
  第 1 步起新列获得非零梯度。这也是“只加信息、不改初始函数”的干净 continuation。

---

## 11. E1/E2 behavior-supervised encoder（重点回答）

**结论：`scripts/step2_chart_encoder.py` 会读 config 的 `in_channels`（间接），会消费 checkpoint，但不消费 content parquet。**

| file:line | item | 说明 / required change | risk |
|---|---|---|---|
| `step2_chart_encoder.py:45-48` | import `load_config, Data2VecModel, ChartStore, build_window, parse_chart, window_starts` | 直接复用 d2v 前端 → 通道扩展自动生效（但需要 spec 传递） | H |
| `step2_chart_encoder.py:103-135` | `ChartCache`：自己的 per-process dict 缓存 `(events, n_frames)`，`120` parse_chart，`132` `build_window(events, s, window_frames, frame_hz, sigma_ms)` | **不读 content parquet**，每次从 .osu 重建窗口；必须让 ChartCache 接收 spec 并传给 build_window。dict 无上限 → 事件对象变大后内存随 split 增长 | **H** |
| `step2_chart_encoder.py:188-209` | `set_trainable`：`conv` 参数在 freeze=none 时训练；teacher/head/mask_emb 冻结 | 通道扩展后 conv1 新列参与训练；freeze=topk 时 conv 不训练（resume 无 opt 问题） | M |
| `step2_chart_encoder.py:212-231` | `load_encoder`：`cfg=load_config(args.config)`（213）→ `Data2VecModel(cfg)`（218）→ `ckpt.get("model",ckpt)` → `load_state_dict(strict=False)`（224） | **是**：in_channels 来自 `--config` YAML，无 CLI 覆盖；**strict=False 不能跳过 shape mismatch** → 旧 8ch ckpt 初始化新模型会抛错。需改为：按 ckpt `conv.conv1.weight.shape[1]` 推导/expand，或显式 expand helper | **H** |
| `step2_chart_encoder.py:234-242` | `make_optimizer`：encoder 一组 + head 一组（+judge/sig head 追加） | conv 可训练时 resume 需 expand opt（同 §10.3） | M |
| `step2_chart_encoder.py:470-500` | `encode`：`t_out=(x.size(2)+1)//2`、`forward_student`、valid-frame 加权 | 通道无关 | — |
| `step2_chart_encoder.py:542-581` | `encode_charts`（评估）：`window_starts` + `cache.build` | 通道无关，但窗口网格不能变（契约 C4） | M |
| `step2_chart_encoder.py:623-644` | `_eval_frame`：读 split parquet + `ChartStore`，建 ChartCache | 无 content 依赖 | — |
| `step2_chart_encoder.py:665-694` | `score_main`：`ck_args=ck.get("args")` 恢复 config/hid/...（672-679）→ `load_config(args.config)`（680）→ `Data2VecModel(cfg)`（684）→ `load_state_dict(ck["model"], strict=True)`（685） | **关键坑**：ckpt 里只存 args.config 的**路径**，不存 in_channels；若原地改 base yaml，旧 E1 ckpt 的 score-only 会直接失败。要么 config 分离，要么从 ckpt state 推断 in_channels | **H** |
| `step2_chart_encoder.py:765-831` | CLI：`--config` 默认 `data2vec_main/configs/audio_style_base.yaml`（771）、`--ckpt`（770）、`--score-only/--checkpoint/--run-dir`（819-830）；**无 --in-channels** | 加 `--in-channels/--channel-spec` 或从 ckpt 推 | M |
| `step2_chart_encoder.py:940,954-959` | train driver：`load_encoder`、`ChartCache(store, args.frame_hz, window_frames, sigma, min_valid)` | spec 传入 | H |
| `step2_chart_encoder.py:966-981` | resume：`model.load_state_dict(ck["model"])` strict=True（968）、`opt.load_state_dict(ck["opt"])`（974） | expand（conv 冻结时只 expand model） | **H** |
| `step2_chart_encoder.py:998-1012` | `save_ckpt`：`snapshot_state(model)` 全模型 state（含 teacher conv 共享权重）+ head/opt/args；`format_version=2`；**不存 in_channels** | 新存档建议加 `in_channels`/`channel_spec` 字段；旧档案靠 conv1 形状推 | M |
| `step2_chart_encoder.py:1162-1172` | 训练后 export：`model.load_state_dict(ck["model"])`、head 等 strict | 同上 | M |
| `scripts/e1_frozen_window_control.py:102-135` | 冻结 encoder 提 per-window 特征（`build_features`，`cache.build`→`forward_student`） | 通道无关逻辑，但需要 spec | M |
| `scripts/e1_frozen_window_control.py:267-286` | 特征缓存 `features.npz` 的复用判定 meta 只有 `[window_sec, frame_hz, sigma_ms, min_valid_sec]`（269-270,273）；模型 load strict=True（286） | **风险**：换通道后 meta 仍匹配 → 会静默复用旧 8ch 特征。应把 spec hash 加入 meta | **H** |
| `scripts/slurm/behavior_e1_train.slurm:37-46` | E1 训练：`--config base.yaml --window-sec 16 ... --init d2v --ckpt main_holdout/last.pt --freeze topk --unfreeze-blocks 4` | 新 arm 要新 config/ckpt；旧 arm 保留 | M |
| `scripts/slurm/behavior_e1_fix_train.slurm:31-40` / `behavior_e2_train.slurm:37-48` | E1_fix / E2 | 同上 | M |
| `scripts/slurm/behavior_e1_fix_legacy_score.slurm:25-30` | 旧 ckpt `--score-only` | 必须保持旧 config 可复现（见上） | H |

**E1/E2 与 content parquet 的关系**：`step2_chart_encoder.py` 只读 split parquet（`:624,847`）与 teacher-signature parquet（`:919`），
不读 `chart_content_d2v*.parquet`；E1/E2 的 encoder 权重来自 D2V checkpoint（raw .osu 现算窗口），
所以**扩通道的 E1 arm 与扩通道的 content 表是两条独立下游**：新 ckpt 直接给 E1 用，不需要先抽 content 表。

**测试文件盘点**：全仓唯一覆盖 D2V 管线的是 `data2vec_main/scripts/smoke_test.py`（train/extract/EMA/parquet 契约）；
`scripts/test_*.py`（test_e1_fix / test_rank_metrics / test_cm3p_pool / test_cm3p_windows / test_step2_* / test_rank_paired_delta）
全部是 CM3P/rank 数学或 E1/E1_fix 辅助函数（ChartHead/bt_loss/snapshot/parquet 后处理）测试，grep `d2v|in_channels` 均无命中 → **扩通道不需要改这些测试**，
但需要用 smoke_test 作为新的契约回归入口。

**文档触点（8 通道/hardcode 描述，需随 plan 更新）**：
`data2vec_main/README.md:21-22,112-114,129-130,146`；根 `README.md:94,357-367`；
`docs/benchmark_d2v.md:72,232`；`docs/experiments.md:183`；`docs/e1_behavior_supervised_encoder.md:198`；
`docs/plans/2026-11-24_behavior_supervision_plan.md:26,49,116,383`；`docs/plans/2026-11-23_three_experiments_plan.md:849`。

---

## 12. 测量方法与证据（可复现）

在本地 CPU、只读方式执行：
- `parse_chart`/`build_window` 计时：`ChartStore('data/osu_cache')` 取 `ids[::110][:200]`，逐图 parse 并逐 1600 帧窗口 build。
  结果：parse 9.16/11.89/89.72 ms（median/mean/max）；build 0.36/0.44/1.66 ms；8.96 windows/chart。
- 扩通道原型：同一循环里 8ch 基线 vs 20ch + release gaussian + 2 个 impulse → 0.399 vs 0.549 ms（+37%），1044 个窗口。
- 向量化 per-frame 属性：`np.searchsorted` + take，1600 帧 25.7 µs/属性。
- checkpoint 结构：`torch.load('data2vec_main/runs/smoke/checkpoints/last.pt')` 实测
  `model['conv.conv1.weight'].shape=[32,8,5]`；`optimizer['state'][0]`=mask_emb `[64]`，`state[1]`=conv1.weight `[32,8,5]`；
  `ema.master` 26 个张量、首项 `in_proj_weight [192,64]`、**无 conv**。
- LRU 缓存内存：tracemalloc 50 图 ≈ 4.5 MB → 90 KB/图；`NoteEvent` `sys.getsizeof=48` + `__dict__=96`。
- kiai effects 缺陷：3136 张图抽样，227,612 条 timing point；effects 取值 `{0:179828, 1:34324, 8:11091, 9:2369}`；
  现有解析 `f[7] in ("1","1.0","true","True")` 与真值 `effects & 1` 不一致 **2369 行 = 1.04%**（`dcr/cm3p_offline.py:205-212`）。
- 未运行任何训练；未修改任何源文件。

---

## 13. Must-not-break contracts（绝不能破）

1. **extract parquet schema**：列名恰好 `[beatmap_id, embedding, n_windows, checksum_ok, source, d2v_rev]`
   （`data2vec_main/scripts/smoke_test.py:112-113`）；`embedding` 512-d float32、L2≈1、`n_windows>0`。
2. **embedding dim 512**：deploy `dan_predict.py:42-44`、`export_dcr_table.py:47-49`、`finalize_benchmark_d2v.py:42-44` 硬校验；
   扩输入通道不要改 `model.dim`。
3. **base 通道顺序/量纲不变**：0..3 = onset（sigma=8ms 高斯、clip 0..4），4..7 = hold（0/1）；
   新通道只能追加在 8.. 之后。这是 8ch ckpt 零初始化 expand 后“函数等价”的前提，也是 `roll[:4]` clip 契约。
4. **n_frames = max(hold end, onset) + 1，仅 note 派生**（`charts.py:68`）。不要用 kiai/timing 延长；
   否则窗口网格（218,525 窗口）、`n_windows`、index cache、窗口覆盖率、P2/P4 折全部变化，基线不可比。
5. **窗口/帧率/骨干超参不变**：window_sec=16、frame_hz=100、窗口 1600 帧→800 token、dim=512、layers=12、heads=8、max_frames=4096。
6. **holdout 划分原样复用**：`data2vec_main/splits/holdout/`（20,584 pretrain / 1,365 eval，seed 0）；
   为新 arm 重新生成 splits 会改变 clean-holdout 口径。
7. **图向量池化不变**：窗口内 valid-frame 均值 → 跨窗口按有效帧加权 → L2 normalize（`extract.py:64-82`）；
   这样 probe 的对比只反映表示变化。
8. **不得覆盖基线产物**：`data/processed/chart_content_d2v_holdout.parquet`、`data/processed/benchmark_content_d2v.parquet`、
   `deploy/models/d2v_linear_extra.json`、`data2vec_main/runs/main_holdout/`、`splits/holdout/`、`results/d2v_main/*`。
   新 arm 一律新文件名/新 out-dir/新 tag。
9. **`audio_style_base.yaml` 保持 8ch 不可变**：旧 ckpt 的 score/resume/E1/E2/benchmark 路径靠它；
   新扩展用新 YAML。若必须共享文件，则所有 load 路径改成“从 ckpt 推断 in_channels”。
10. **单一真源**：`data.channels` spec → 派生 `model.in_channels`；在 train/extract/step2/e1_control 入口 assert
    `x.shape[1] == model.conv.conv1.in_channels`（当前完全没有这个 assert）。
11. **内容表与 deploy JSON 的 content_md5 配对**：新表必须重出 JSON（`export_dan_deploy.py`）才能过 `export_dcr_table.py:129-135`。
12. **parquet 的 `source`/`d2v_rev` 是唯一 provenance**：`d2v_rev` 在未提交工作树上可能不变，新 arm 必须用 `--tag` 区分
    （建议 `d2v_main_attr12_v1` 之类，并在 plan/表名里写明 spec）。
13. **E1/E2 旧结果可复现**：旧 ckpt + 旧 config 必须仍能 score（`step2_chart_encoder.py --score-only`）；
    新 plan 不能只保留“新通道模型”。
14. **不修改 `deploy/table/dcr_content.bin` 契约**（header/record 28B）；研究阶段不需要新表进产品。

---

## 14. Open questions

1. **spec 表示与版本化**：`data.channels` 用 list[dict]（name/kind/encoding/range/version）还是紧凑字符串？
   spec hash 存哪里（config.yaml、ckpt cfg、parquet `source`、index cache meta）？旧 ckpt 没有 spec 时如何推断？
2. **12/18/27 的具体构成**：8 base + 4 release = 12；再 + BPM/beat-phase/measure/SV/kiai（6）= 18；
   再 + hitsound/set/volume（9）= 27？release 是 4 列脉冲还是单通道？beat phase 用 sin/cos（2ch）还是 sawtooth（1ch）？
   这个决定直接决定 C，但代码改动量相同。
3. **归一化/值域**：BPM 用绝对 BPM 还是相对 median（`sv_groups` 用 `scroll*median_mpb/mpb` 的温度不变做法）？
   SV 用 log？kiai 二值？hitsound 用 bitmask→one-hot 还是 4 类+normal？volume 0=inherit、sample-set 0=auto 的解析规则要定。
4. **per-note volume 是否有信息量**：osu mania 里 hitobject volume 只影响音效响度（不影响判定），
   对难度序可能为噪声；需要 data audit（并行任务 `data2vec_main/scripts/audit_attributes.py` 正在写，未定稿）确认覆盖率/分布。
5. **release 语义**：hold 在当前 NoteEvent 里 `end` 是 exclusive；release 脉冲打在 `end` 还是 `end-1`？
   与 E1 的 hold tail 判定（`docs/plans/2026-11-24_behavior_supervision_plan.md:744,835-837`）对齐。
6. **per-frame 通道与 conv 感受野**：BPM/SV/kiai 是慢变量，放 100 Hz 再 stride-2 conv 是否浪费？
   是否只给 50 Hz？是否需要在 conv1 之外拼一个 context 分支（更复杂，不在本清单范围）。
7. **训练策略**：从 8ch ckpt 继续（zero-init 新列，函数等价起点）还是从零训练？
   继续时 epochs/LR schedule 怎么设（`train.py:117-119` 的 total_steps 语义决定必须传更大的 `--epochs` 或新增 `--total-steps`）；
   要不要对 conv1 新列做短暂冻结/warmup？8ch 通道会不会遗忘（catastrophic forgetting）？
8. **比较口径**：新 arm 复用同一 holdout split + 同一 extract/probe 流程，与 D2V 8ch baseline 做 paired per-seed Δ；
   是否需要同时训一个“同通道数、但新通道全零”的 control 来分离“更多参数/更多输入”与“信息量”？
9. **index cache**：是否坚持 n_frames 不变（推荐）？若不变则无需重建；若 plan 想包含 kiai/timing 尾部，
   要新建 `chart_index_100_attr.json` 并把 spec hash 加进 `_meta`（`index_cache.py:27-37`）。
10. **E1/E2 是否重跑**：新 ckpt 会带来新的 E1/E2 arm（encoder 变强/变弱都影响 behavior 结论）；
    是否在 plan 里同步排期，还是先只做 content-vector 主实验？
11. **deploy/benchmark 是否重跑**：新 arm 要不要重出 VSRG benchmark（`D2V-L` 列名）与 deploy JSON？
    若只要研究结论，可跳过；若要产品表，需 content_md5 + parity 全链路。
12. **probe 命令 bug**：`data2vec_main/d2v_main/probe.py:59` 给 `run_p2_p4_multiseed.py` 传 `--models cm3p_linear`，
    但 METHODS 只有 `linear/linear_extra/mlp/mlp_extra` → **README §4 第 5 步的 P4 命令按原样会报错**
    （已用 import 验证：`'cm3p_linear' in METHODS == False`）。实际 P4 走 `run_p4_probes.sh`。
    新 plan 是否顺手修（改 `linear`/新增模型名）？
13. **`slice_window` dead code**（`charts.py:142-152`，全仓 0 调用）保留还是删？
14. **本地无法复现 baseline 训练**：`runs/main_holdout/checkpoints/last.pt`（737MB）只在 HPC，
    本地只有 `runs/smoke/last.pt`；新 arm 的 continue-pretrain 需要在 HPC 上做，或先 rsync checkpoint。
    是否把 checkpoint 纳入本地缓存策略？
15. **kiai 解析 bug 的修法**：修 `dcr/cm3p_offline.py:205-212`（`effects & 1`）会影响 CM3P token 流
    （`kiai_groups` 产出的 token 变化）→ 可能改变已有 CM3P 内容表/复现性。
    是“只在新 D2V 前端里独立解析 effects”，还是“修共享 parser 并接受 CM3P 侧影响”？这是 plan 必须决定的。

---

## 15. 附：当前基线复现命令（README / scripts 原文，带路径）

> 环境（HPC）：`export PYTHONNOUSERSITE=1`；`PY=$HOME/.conda/envs/DCR/bin/python`（torch 2.11.0+cu128）。
> 本地 smoke 用 `python3`。以下 cwd 默认 `data2vec_main/`（除非另注）。

### 15.1 本地 CPU smoke（唯一能立即验证全链路的）
```bash
cd data2vec_main && python3 scripts/smoke_test.py
```
产物：`data2vec_main/runs/smoke/`（config.yaml、metrics.jsonl、checkpoints/last.pt、content.parquet）。
断言：piano roll (8, n_frames)、mask padding、EMA、train 3 步、extract parquet 6 列 + norm=1。

### 15.2 HPC 正式流程（README:76-91；sbatch 版见 §10.5）
```bash
cd data2vec_main
# 0) index cache（首次 ~4 min 本地 / 881.6 s HPC 登录节点）
python -m d2v_main.index_cache --osu-dir data/osu_cache --frame-hz 100
#    产物 data2vec_main/cache/chart_index_100.json（本地当前不存在）

# 1) 划分（holdout = 诚实口径；all = transductive 上界）
python -m d2v_main.splits --protocol holdout --out-dir splits/holdout
#    产物 splits/holdout/{pretrain_ids.txt 20584, eval_ids.txt 1365, summary.json}（本地已存在）

# 2) 预训练（README 用 runs/base_holdout；实际基线 sbatch 用 runs/main_holdout）
python -m d2v_main.train --config configs/audio_style_base.yaml \
    --ids-file splits/holdout/pretrain_ids.txt --out-dir runs/base_holdout --device cuda
#    实际基线（train_hpc.sbatch:25-28）：--out-dir runs/main_holdout --epochs 20 --batch-size 16 --num-workers 6 --log-every 50
#    基线 job 14072398（A800，25,720 steps，47m02s，loss 0.524→0.054）→ runs/main_holdout/checkpoints/last.pt（737MB，仅 HPC）

# 3) 抽取全部 21,949 张（extract_hpc.sbatch:17-20）
python -m d2v_main.extract --config configs/audio_style_base.yaml \
    --ckpt runs/main_holdout/checkpoints/last.pt \
    --out ../data/processed/chart_content_d2v_holdout.parquet \
    --device cuda --batch-size 32 --num-workers 6 --tag d2v_main_holdout
#    产物 data/processed/chart_content_d2v_holdout.parquet（本地已有 45,746,268 B，21,949 行；job 14072889）

# 4) P2（chart-disjoint 5 折）
python -m d2v_main.probe --content-path ../data/processed/chart_content_d2v_holdout.parquet \
    --out-dir ../results/d2v_main/p2 --protocol p2
#    等价 scripts（run_p2_probes.sh:15-18）4 个 cell：p2_all_d2v / p2_all_cm3p / p2_hold_d2v / p2_hold_cm3p
#    内部：scripts/step2_cm3p_rank.py --in data/processed/step1_1k_pairbest.parquet --content-path ... \
#          --model linear --chart-folds 5 --threads 8 --out-signals ... --out-json ...
#    hold 口径用的是 --in data/processed/step1_1k_pairbest_d2v_holdout.parquet

# 5) P4（新玩家+新图，5 seeds）——README 原命令有 bug（probe.py:59 传 cm3p_linear，见 Q12）
python -m d2v_main.probe --content-path ../data/processed/chart_content_d2v_holdout.parquet \
    --out-dir ../results/d2v_main/p4 --protocol p4
#    实际脚本（run_p4_probes.sh:8-14）：
python3 scripts/run_p2_p4_multiseed.py --protocols p4 --seeds 1,2,3,4,5 \
    --models linear,linear_extra \
    --content-path data/processed/chart_content_d2v_holdout.parquet \
    --out-dir results/d2v_main/p4_d2v --threads 8
```

### 15.3 其他已跑基线（scripts 原文）
```bash
# P4-holdout 干净口径（run_p4_hold_probes.sh）
python3 scripts/run_p2_p4_multiseed.py --protocols p4 --seeds 1,2,3,4,5 \
  --models linear,linear_extra --content-path data/processed/chart_content_d2v_holdout.parquet \
  --split-p4 data/processed/step1_1k_pairbest_p4_d2vhold \
  --out-dir results/d2v_main/p4_hold_d2v --threads 8

# MLP 基线（run_mlp_probes.sh:8-16；HPC 版 hpc_mlp_probes.sbatch）
python3 scripts/run_p2_p4_multiseed.py --protocols p4 --seeds 1,2,3,4,5 \
  --models mlp,mlp_extra --content-path data/processed/chart_content_d2v_holdout.parquet \
  --out-dir results/d2v_main/p4_d2v_mlp --threads 8

# content+external linear_extra（run_p2_extra.sh:16-19）
python3 scripts/step2_cm3p_rank.py --in data/processed/step1_1k_pairbest.parquet \
  --content-path data/processed/chart_content_d2v_holdout.parquet \
  --model linear_extra --extra-signals msd_overall,sunny_star --chart-folds 5 --threads 8 \
  --out-signals results/d2v_main/p2_all_d2v_ext_signals.parquet \
  --out-json results/d2v_main/p2_all_d2v_ext_rank.json

# randommerge P2/P4（run_randommerge_d2v.sh；HPC 版 hpc_randommerge_probes.sbatch）
python3 scripts/run_p2_p4_multiseed.py --protocols p2,p4 --seeds 1,2,3,4,5 \
  --models linear,linear_extra --content-path data/processed/chart_content_d2v_holdout.parquet \
  --signal-table data/processed/rank_signals_randommerge_od.parquet \
  --split-p2 data/processed/step1_randommerge_c_p2 --split-p4 data/processed/step1_randommerge_c_p4 \
  --out-dir results/d2v_main/randommerge_d2v --threads 6
```

### 15.4 VSRG benchmark 链路（docs/benchmark_d2v.md:79-92,110-120；slurm 见 §10.5）
```bash
# 编码 1,382 benchmark 图（scripts/slurm/benchmark_d2v_extract.slurm）
PYTHONPATH=data2vec_main python -m d2v_main.extract \
  --config data2vec_main/configs/audio_style_base.yaml \
  --ckpt data2vec_main/runs/main_holdout/checkpoints/last.pt \
  --osu-dir data/benchmark_d2v_osu --ids-file data/processed/benchmark_d2v_ids.txt \
  --out data/processed/benchmark_content_d2v.parquet \
  --device cuda --batch-size 32 --num-workers 6 --tag benchmark_d2v
python scripts/finalize_benchmark_d2v.py   # 断言 dim==512，join benchmark_d2v_map.parquet

# linear_extra 训练（scripts/slurm/benchmark_d2v_train.slurm；job 14077896）
python3 scripts/step2_cm3p_rank.py \
  --in data/processed/step1_randommerge_pairbest_content.parquet \
  --content-path data/processed/chart_content_d2v_holdout.parquet \
  --signal-table data/processed/rank_signals_1k.parquet \
  --extra-signals msd_overall,sunny_star --model linear_extra --lam 1e-3 \
  --val-frac 0 --seed 0 --threads 8 --name cm3p_linear_extra --fit-all \
  --save-model results/benchmark/models/D2V_L_full.pt \
  --out-signals results/benchmark/d2v_linear/train/D2V_L_signals.parquet \
  --out-json results/benchmark/d2v_linear/train/D2V_L.json

# deploy JSON / 表（deploy/README.md:170-186,207-211）
python scripts/export_dan_deploy.py --model results/benchmark/models/D2V_L_full.pt \
  --pred results/benchmark/pred/D2V_L.parquet --content-type d2v \
  --out deploy/models/d2v_linear_extra.json --oof results/benchmark/eval_d2v/oof.parquet \
  --oof-col D2V_L_isotonic
python deploy/export_dcr_table.py --content data/processed/chart_content_cm3p_local.parquet \
  --signals data/processed/rank_signals_1k.parquet --whitelist manifests/whitelist_1k.csv
```

### 15.5 E1/E2 behavior arms（scripts/slurm/*，均引用 8ch ckpt + base.yaml）
- `behavior_e1_train.slurm:24-46`：arms = unfrozen_k4 / random / frozen_samehead，`--init d2v --ckpt data2vec_main/runs/main_holdout/checkpoints/last.pt`，`--config data2vec_main/configs/audio_style_base.yaml`，`--split-table data/processed/step1_full_holdout_s$S.parquet`，`--out-dir runs/behavior/e1/$ARM/s$S`。
- `behavior_e1_bench.slurm:21-33`：30-step GPU bench；`behavior_e1_pilot_debug.slurm`：60k-view pilot。
- `behavior_e1_fix_train.slurm:23-40` / `behavior_e1_fix_control.slurm:20-30`（+`_cpu`）：E1_fix clean + F4 冻结窗口 2×2。
- `behavior_e2_train.slurm:28-48`：E2 judge/single_grade arms。
- `behavior_e1_fix_legacy_score.slurm:22-31`：旧 ckpt score-only（`--checkpoint runs/behavior/e1/$ARM/s1/checkpoint.pt`）。

### 15.6 本地产物现状（影响“能否本地复现”）
- 有：`data/osu_cache/`（21,950 个 .osu）、`data/processed/chart_content_d2v_holdout.parquet`、`benchmark_content_d2v.parquet`、
  `step1_*_d2v_holdout*.parquet`、`splits/holdout/`、`data2vec_main/runs/smoke/`、`results/d2v_main/*`。
- 无（仅 HPC）：`data2vec_main/runs/main_holdout/`（737MB last.pt）、`data2vec_main/cache/chart_index_100.json`。
- 结论：本地可跑 smoke 与 probe（有 content 表 + 信号表），不可跑 baseline pretrain/extract 的 checkpoint 复用。

---

## 16. 建议的实施顺序（供 plan 参考，非本任务承诺）

1. 定义 `ChannelSpec`（names/slices/encoding/range/version）+ `cfg["data"]["channels"]`，base8 作为默认 spec；
   `model.in_channels` 由 spec 派生；所有入口 assert。
2. 扩展 `dcr/cm3p_offline.py`（或在新前端私有 parser）解析 hitsound/sample/volume/timing volume，修 `effects & 1` kiai；
   决定是否影响 CM3P 侧（Q15）。
3. `charts.py`：`NoteEvent` 加字段（slots）、`parse_chart` 返回 timing/attrs、`build_window` 接受 spec/attrs；
   新增 per-frame 向量化填充；`slice_window` 清理。
4. `data.py` + smoke_test 参数化；新增断言。
5. `model.py`：去默认值、spec 派生、assert；不改 EMA。
6. `train.py`：`--in-channels/--resume-expand`；expand helper（model + optimizer；EMA 跳过）；total_steps 语义修正。
7. 新 `configs/audio_style_attr.yaml`（**不动 base**）；新 sbatch 变体；ckpt 拷贝脚本（旧 8ch→新 C，zero-init 新列）。
8. 新 tag 的 extract → 新 content 文件 → 复用现有 probe/eval 流程（新 out-dir），paired Δ vs 基线。
9. E1/E2：新 ckpt 走新 arm；旧 ckpt/旧 config 保留可复现（score-only 用 ckpt 推断 in_channels）。
10. deploy/benchmark：可选，需要时全链路重跑（新 JSON/content_md5/parity）。
