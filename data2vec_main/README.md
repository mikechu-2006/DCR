# data2vec 主实验（route B：from-scratch audio-style）

> 2026-11-23。状态：代码完成；CPU smoke + A40 120-step benchmark 通过；
> **完整预训练已完成**（job 14072398，A800，47m02s）；extraction 已完成（21,949 图，512d）；
> P2 与 P4 5 seeds 已完成，但**只覆盖 canon 表 + linear/linear_extra**；legacy 表与 MLP 基线正在补。
> 在这个口径下 D2V 优于 CM3P；对最强基线（legacy + MLP+ext）尚未成立，见 §6.4/§6.5 的修正。
> 继续预训练（route A，CM3P + column patch + data2vec continue）由另一个 session 负责；
> 本目录只做 **from-scratch audio-style data2vec**，即路线 B / 主实验。

## 0. 一句话

把 4K 谱面当成 audio-style 信号：100 Hz × 8 通道 piano roll → 共享 1D conv（stride 2）→
50 Hz 连续特征 → span mask → student Transformer；teacher 是同一 conv + EMA Transformer，
看完整特征。只在 masked 帧上回归 teacher 顶层 K 层的 LayerNorm 平均表示。
没有离散词表，没有 8-bit code。

## 1. 方法

| 项 | 设置 |
|---|---|
| 输入 | 100 Hz（10 ms）× 8 通道：4 列 onset + 4 列 hold；onset 用 sigma=8 ms 高斯核软化 |
| 特征编码器 | Conv1d(8→256, k=5) → GELU → Conv1d(256→256, k=5) → GELU → Conv1d(256→512, k=5, stride=2) → GELU → LayerNorm，输出 50 Hz × 512 |
| 位置编码 | sinusoidal，**在 mask 之后加**（见 §5 问题 1） |
| student | 50 Hz 特征上 span mask：p=0.065、长度 10 帧（≈200 ms，约 49% 帧），mask 帧替换成可学习向量 |
| teacher | 同一 conv（共享）+ EMA Transformer；输入不 mask |
| 目标 | teacher 顶层 K=6 层，每层 LayerNorm 后平均，只取 masked 帧 |
| 预测 | student 最后一层 + Linear(512→512) 投影头 |
| loss | Smooth-L1（beta=1.0），只在 masked 帧 |
| 窗口 | 16 s = 1600 输入帧 → 800 个 50 Hz 特征；训练随机 crop，抽取时 16 s/16 s 不重叠 |
| 图向量 | 对窗口内有效帧做均值，跨窗口按有效帧数加权，最后 L2 normalize |
| EMA | tau 0.999 → 0.9999，前 30k step 线性；**fp32 master**，每步 copy 到 bf16 teacher |
| 优化 | AdamW lr 1e-4，wd 0.01，warmup 5%，cosine，clip 1.0，bf16（GPU） |
| 规模 | 12 层 / 512d / 8 heads / FFN 2048；trainable 39.1M，teacher 37.8M |

## 2. 文件

    data2vec_main/
      README.md
      configs/
        audio_style_base.yaml       # 主实验配置（12L/512d）
        audio_style_smoke.yaml      # CPU smoke（2L/64d）
      d2v_main/
        charts.py                   # .osu -> NoteEvent -> piano roll / window
        index_cache.py              # beatmap_id -> n_frames 的 JSON 缓存
        data.py                     # Dataset + collator（on-the-fly）
        masking.py                  # span mask
        model.py                    # conv encoder + Transformer + EMA + loss 组件
        train.py                    # 预训练入口
        extract.py                  # checkpoint -> chart_content parquet
        probe.py                    # 调现有 rank-NLL runner
        splits.py                   # all / complement / holdout ID 列表
        config.py, utils.py
      scripts/
        smoke_test.py               # 3 张真图 + CPU 全链路

## 3. 快速开始（本地 CPU）

    cd data2vec_main
    python3 scripts/smoke_test.py

输出应包含 <code>SMOKE OK</code>。它会跑 3 张真图、3 步训练、3 张图抽取，并检查
piano roll、mask padding、前向/反向、EMA、checkpoint、parquet 契约。

## 4. 正式运行（HPC 命令；只跑过 120-step benchmark）

HPC 环境（base env 的 cu130 在 A40 driver 12.9 上不可用）：

    export PYTHONNOUSERSITE=1
    PY=$HOME/.conda/envs/DCR/bin/python   # torch 2.11.0+cu128

把下面的 `python` 换成 `$PY`。

    cd data2vec_main

    # 0) 首次构建 index cache（约 4 分钟；之后 <1 秒）
    python -m d2v_main.index_cache --osu-dir data/osu_cache --frame-hz 100

    # 1) 划分：holdout 是诚实口径；all 是 transductive 上界
    python -m d2v_main.splits --protocol holdout --out-dir splits/holdout

    # 2) 预训练（单卡；本目录没有 DDP）
    python -m d2v_main.train --config configs/audio_style_base.yaml         --ids-file splits/holdout/pretrain_ids.txt         --out-dir runs/base_holdout --device cuda

    # 3) 抽取全部 21,949 张（评测图只做前向，不参与训练）
    python -m d2v_main.extract --config configs/audio_style_base.yaml         --ckpt runs/base_holdout/checkpoints/last.pt         --out ../data/processed/chart_content_d2v.parquet --device cuda

    # 4) P2（chart-disjoint 5 折）
    python -m d2v_main.probe         --content-path ../data/processed/chart_content_d2v.parquet         --out-dir ../results/d2v_main/p2 --protocol p2

    # 5) P4（新玩家 + 新图，5 seeds）
    python -m d2v_main.probe         --content-path ../data/processed/chart_content_d2v.parquet         --out-dir ../results/d2v_main/p4 --protocol p4

P4 只是把 <code>run_p2_p4_multiseed.py</code> 包了一层；指标代码没有重写，
数字与 CM3P 基线可比。

## 5. 实现过程中遇到的问题

1. **mask 与位置编码的顺序（已修）**：第一版先加 sinusoidal 位置再 mask，
   masked 帧全变成同一个 <code>mask_emb</code>，位置信息被抹掉；smoke 里
   <code>pred_var</code> 只有 1.4e-7，学生输出几乎常数。改成 **先 mask、后加位置**
   后，初始 <code>pred_var</code> 到 0.36，50 步内 loss 从 0.50 降到 0.16（3 张图过拟合，
   只作 sanity）。data2vec-audio 的 conv_pos 也是在 mask 之后加的，方向一致。
2. **Dataset 初始化扫全部图（已加 index cache）**：<code>ChartWindowDataset</code>
   为拿到 <code>n_frames</code> 要 parse 每一张图；实测约 12 ms/图，21,949 张约 4 分钟/进程。
   现在 <code>index_cache.py</code> 把 beatmap_id -> n_frames 写成 JSON，按
   osu_dir / 文件数 / max mtime / frame_hz 失效；本机首次约 4 分钟，
   HPC 登录节点实测 881.6 s（14.7 分钟，I/O 更慢），之后 load <1 秒。
3. **本机没有 GPU**：smoke 用 2L/64d；完整 12L/512d 的 A40 benchmark
   已跑（§6.1），A800 尚未实测。
4. **单进程 / 无 DDP**：<code>train.py</code> 只支持单卡。12L/512d、batch 16–32、
   800 帧，一张 A800 应该够；但多卡要另加 <code>torchrun</code> + <code>DistributedSampler</code>。
5. **on-the-fly 的代价**：每个样本要 parse .osu + 建窗口（约 12 ms，未 cache 时）；
   4 个 worker 大约 300 样本/s。21k 图/epoch 的数据时间约 1 分钟，可以接受，
   但 CPU 核少时会成为瓶颈；写 index cache 和/或预生成窗口能缓解。
6. **EMA 精度（已加 fp32 master）**：teacher 前向仍用 bf16，但 EMA 在 fp32
   master 上更新，每步 copy 回 teacher；代价是每步约 151 MB 的 copy。smoke 验证了
   teacher 与 master 一致；bf16 下比同 dtype EMA 更稳。
7. **checksum_ok 恒为 True**：抽取时没有重算 .osu md5；和 CM3P 表合并/比较前要补。
8. **P4 probe 没有本地跑**：<code>probe.py</code> 只做了 dry-run；P4 复用
   <code>run_p2_p4_multiseed.py</code>，需要 HPC 上跑 5 seeds。
9. **数据划分的坑**：非 pairbest 的 15,121 张明显更简单（star 2.66 vs 4.73），
   不能单独当预训练池；<code>holdout</code> 用 complement + 80% pairbest，
   eval 是 star 分层的 20%（本次 seed 0：pretrain 20,584 / eval 1,365，
   star 均值 4.732 / 4.711）。只在 holdout 图上评测会让 P4 的 pair coverage 下降，
   报告时要写清楚。
10. **注意力实现**：用 <code>nn.TransformerEncoderLayer</code>（标准 SDPA），
    800 帧没问题，但不是 FlashAttention；后续可换自定义 block / data2vec 2.0 的
    unmasked-only + conv decoder。
11. **输入通道只到 hold 状态**：没有 velocity、hitsound、hold-end 脉冲；4K mania
    的 hold 用持续通道表示，够 v1 用，但和弦/长条的细节可以再做消融。
12. **短图/坏图**：dataset 初始化会跳过 parse 失败或短于 1 s 的图，并打印数量；
    全量跑之前要看这个数是否异常。
13. **HPC base env 的 cu130 在 A40 上不可用**：gpu3-9 的 driver 是 12.9，
    base python 的 `torch 2.13.0+cu130` CUDA init 直接失败（found version 12090）。
    改用 `~/.conda/envs/DCR` 的 `torch 2.11.0+cu128`（CUDA 12.8，兼容）。
14. **DCR env 缺 PyYAML**：加 `PYTHONNOUSERSITE=1` 后 `import yaml` 失败
    （之前是 user site 的 6.0.3 在兜底）；已把 `pyyaml 6.0.4` 装进 env 的
    site-packages，`import yaml/torch/numpy` 全部通过。
15. **`--log-every` 不是 CLI 参数**：第一版 sbatch 报
    `unrecognized arguments: --log-every`；已加到 `train.py`。
16. **debug 队列只有 30 min**：120 step benchmark 可以，完整 20 epoch 不行；
    全量要换 `i64m1tga800ue` / emergency_gpu，wall-clock 还要加排队时间。

## 6. 本地 smoke 与 HPC benchmark

- 3 张真图（193127/193128/193129），piano roll (8, 9735)，onset 2954 / hold 3797 个非零。
- 50 步（2L/64d，3 张图，batch 1）：loss 0.50 -> 0.16；target_var ~0.33；
  pred_var ~0.17；mask ratio ~0.54；约 7 s。
- 完整链路（train 3 步 + extract 3 图）：约 5 s；parquet 6 列，embedding norm = 1.0。
- 参数：conv 1.0M + student 37.8M + head 0.26M = 39.1M trainable；teacher 37.8M；
  总 76.9M。
- 完整 12L/512d：单样本 800 帧 CPU 前向 9.3 s、teacher 1.6 s、反向 1.5 s；
  mask ratio 0.4925（p=.065, length=10），与 data2vec-audio 的 ~49% 一致。
  这些只是形状/数值 sanity，不是吞吐基准。
- 200 张图 parse + build window：2.36 s（约 12 ms/图）。

### 6.1 HPC A40 120-step benchmark（2026-11-23，job 14070160）

| 项 | 值 |
|---|---|
| 节点 / GPU | gpu3-9，A40 50.9 GB |
| 环境 | DCR conda env，torch 2.11.0+cu128，CUDA 12.8 |
| 数据 | 21,949 张图，index cache 直接 load |
| batch / workers | 16 / 6 |
| 120 step | 30 s end-to-end = **4.04 step/s**；稳态约 5.0 step/s |
| 显存 | **5.52 GB**（batch 16） |
| loss | 0.52 -> 0.13（仍在 warmup 内，只作 sanity） |
| mask ratio | 0.47-0.49 |
| target_var / pred_var | 0.48 -> 0.66 / 0.35 -> 0.42 |

**20 epoch 预估**（batch 16, drop_last）：

| 预训练池 | steps/epoch | total steps | @4.04 step/s | @5.0 step/s |
|---|---:|---:|---:|---:|
| all 21,949 | 1,371 | 27,420 | 1.88 h | 1.52 h |
| holdout 20,584 | 1,286 | 25,720 | 1.77 h | 1.43 h |

- 每 epoch 约 4.3-5.7 min；A40 上 **20 epoch 约 1.5-1.9 h**。
- A800 粗估 1.5-2x，约 **0.9-1.3 h compute**；队列等待另算。
- batch 16 只用了 5.5 GB，batch 32/64 可能更快（未测）。
- `debug` 队列上限 30 min，装不下完整 20 epoch；全量要投 `i64m1tga800ue`（历史中位等待约 3.5 h）或 emergency_gpu。

### 6.2 完整预训练（2026-11-23，job 14072398，emergency_gpu）

| 项 | 值 |
|---|---|
| 队列 / 资源 | emergency_gpu，1×A800，8 核 / 64G，6 h 上限 |
| 数据 | holdout pretrain **20,584** 张 |
| 配置 | 12L/512d，20 epoch，batch 16，6 workers，fp32 EMA |
| steps | **25,720** |
| training | **47m02s**，9.114 step/s（A40 benchmark 4.04 的 2.25x） |
| job elapsed | 1:01:59（含启动/数据） |
| 显存 | 5.53 GB |
| loss | 0.524 -> **0.054** |
| target_var / pred_var | 0.488 -> 0.921 / 0.352 -> 0.857 |
| mask ratio | ~0.48 |
| checkpoint | last.pt 737 MB；每 2,000 step 一个 |

- 实际 compute 比 A800 预估（0.9-1.3 h）还快，wall 约 1 h（无排队）。
- 完整 run 目录 9.4 GB（10 个中间 checkpoint + last.pt）。

### 6.3 Extraction（job 14072889，2026-11-23）

| 项 | 值 |
|---|---|
| 队列 | emergency_gpu，1×A800，batch 32，6 workers |
| elapsed | 27m19s（含 380.6 s index cache 重建） |
| 输出 | `data/processed/chart_content_d2v_holdout.parquet`，44 MB |
| 覆盖 | 21,949 / 21,949 图 |
| 窗口 | 218,525（平均约 10 窗/图） |
| 维度 / 范数 | 512d，L2 norm = 1.0（min/max 1±2e-7） |
| source / rev | d2v_main_holdout / c29431f |

### 6.4 P2 frozen linear probe（本地，2026-11-23）

5-fold chart-disjoint，linear，lam=1e-3，8 threads；`captured` 用 `step2_rank_eval.py` 的 cross-fitted ceiling。

| 口径 | D2V NLL | CM3P canon NLL | ΔNLL | D2V captured | CM3P captured |
|---|---:|---:|---:|---:|---:|
| 全 6,828 图 | **0.5863** | 0.6240 | **-0.0377** | **0.8193** | 0.5300 |
| 仅 1,365 holdout 图 | **0.5971** | 0.6370 | **-0.0399** | **0.7289** | 0.4260 |

**口径修正（2026-11-23）**：本表的 CM3P 是 **canon 表 + linear**；0.6240/0.60335 只对应 `results/p4_multiseed_canon` 的 linear 行，不对应 `results/p4_multiseed`（legacy 表：linear 0.61618、linear+ext 0.57786、MLP 0.61664、MLP+ext 0.56602）。所以“D2V 优于 CM3P+ext”只在 canon+linear 下成立；对最强基线（legacy MLP+ext 0.56602）**不成立**。

- ceiling：全量 0.56276；holdout 子集 0.56145。
- holdout 行是干净口径：编码器预训练时没见过这 1,365 张图；全量行里约 80% 图被预训练见过（transductive 成分）。
- 这是**系统级对比**：D2V 与 CM3P 的 tokenizer、列信息、目标函数、预训练语料全都不同；不能当作“data2vec vs MLM”的单变量结论。CM3P canon 的 0.6240 也和 docs 里 legacy 表的 0.6148 不完全同口径，所以只看同一 runner/同一 step1 下的 Δ。

**content + external（MSD+Sunny，linear_extra）**：

| 口径 | D2V+ext NLL | CM3P+ext NLL | ΔNLL | D2V captured | CM3P captured |
|---|---:|---:|---:|---:|---:|
| 全 6,828 图 | **0.5810** | 0.6009 | **-0.0198** | **0.8599** | 0.7077 |
| 仅 1,365 holdout 图 | **0.5939** | 0.6167 | **-0.0228** | **0.7538** | 0.5808 |

- 外部信号对 CM3P 的增益（-0.023）明显大于对 D2V 的增益（-0.005），说明 D2V 内容表示已经覆盖了 MSD/Sunny 的不少信息。
- docs 的 CM3P+external 0.5756 / 0.9013 在本 runner/本表下没有复现（这里是 0.6009 / 0.7077），所以绝对值和 docs 不可直接比；只看同一 runner 下的 Δ。

### 6.5 P4 multiseed（新玩家 + 新图，5 seeds）

| 方法 | P4 NLL | captured | Pearson(d,b) | Spearman(d,b) |
|---|---:|---:|---:|---:|
| D2V 线性 | **0.59301 ± 0.00417** | **0.7859 ± 0.0475** | 0.5373 ± 0.0154 | 0.4840 ± 0.0207 |
| D2V + external 线性 | **0.58561 ± 0.00459** | **0.8437 ± 0.0468** | 0.5411 ± 0.0149 | 0.4901 ± 0.0192 |
| CM3P 线性 | 0.63050 ± 0.00270 | 0.4914 ± 0.0249 | 0.4520 ± 0.0219 | 0.4084 ± 0.0174 |
| CM3P + external 线性 | 0.60335 ± 0.00515 | 0.7037 ± 0.0266 | 0.4892 ± 0.0247 | 0.4464 ± 0.0215 |
| ceiling | 0.56540 ± 0.00859 | — | — | — |

- P4 是**新玩家 + 新 chart**，5 个划分种子、model seed 0；ceiling 与 docs 一致。
**口径修正（2026-11-23）**：上表 CM3P 是 **canon 表 + linear**；0.63050/0.60335 只与 `results/p4_multiseed_canon` 的 linear 行一致，**不是** `results/p4_multiseed` 的 legacy 数字。`results/p4_multiseed` 的 legacy 表更强：linear 0.62411、linear+ext 0.58479、MLP 0.62540、**MLP+ext 0.57343**。

- D2V content-only 比 **canon linear** 好 0.0375；D2V+ext 比 **canon linear_ext** 好 0.0177；paired per-seed 分别 -0.0375±0.0040 / -0.0177±0.0040（5/5）。
- 但换成最强基线：D2V+ext 0.58561 比 canon MLP+ext 0.59425 好 **0.0086**，比 legacy MLP+ext 0.57343 差 **0.0122**；D2V content 0.59301 与 canon MLP+ext 基本打平（-0.0012），比 legacy MLP+ext 差 **0.0196**。之前“D2V 甚至优于 CM3P+ext”是拿 canon linear 当基线，已修正。
- **caveat**：D2V 预训练用了 6,828 张 pairbest 里的 80%（5,462 张），P4 的“新 chart”是相对监督划分，不是相对编码器预训练；所以 P4 仍有 transductive 成分。干净的 P4-holdout（只留 1,365 张编码器没见过的图）正在跑，结果出来后补。

## 7. 还没做

- P2/P4 主口径已完成；P4-holdout 干净口径正在跑。
- data2vec vs MLM 的单变量消融（route A 的 2×2）不在本目录，需另一个 session 的表来合。
- DDP / 多卡；bf16 稳定性；batch 32/64 调优。
- 32 s 窗口 / 层次化 pooling / CLS-level loss 的消融。
- 与 route A（CM3P continue）的表合并比较：两边的 tokenizer 不同，先各自对自己的
  baseline，再在 P4 上比。
