# CM3P 官方向量复现探针（audio-on）

> 2026-11-10。目标：用 checkpoint 自带的 processor 配置在 HPC 上重算官方发布的 512 维
> 向量，在 100 张随机官方图上验证能否以 cos > 0.99 复现。**结论：98 张可编码图中 95 张
> ≥ 0.99，均值 0.99745，中位 0.99809。** 同配置已用于剩余 2,786 张图。

代码：`scripts/cm3p_official_probe.py`（逐图编码 + 对比）、
`scripts/cm3p_probe_fetch_audio.py`（只在登录节点下音频）、
`scripts/slurm/cm3p_probe_official.slurm`、`scripts/slurm/cm3p_probe_lowdiag.slurm`。
产物：`results/cm3p_probe_official/`（sample.csv、probe.jsonl、summary.json、lowdiag/）。

---

## 1. 复现的官方口径

官方表 `data/processed/chart_content_cm3p.parquet` 来自
[OliBomby/CM3P-Embeddings-244K](https://huggingface.co/datasets/OliBomby/CM3P-Embeddings-244K)，
由 <https://huggingface.co/OliBomby/CM3P> 的 `extract_beatmap_embeddings.py` 生成
（`include_audio=True`、`include_metadata=False`）。探针逐项核对了 checkpoint 的
processor 快照，并复现了这条路径：

| 项 | 值（processor snapshot，实测） |
|---|---|
| 权重 | `OliBomby/CM3P` revision `26cc694`；与 `main` 同一 LFS oid（292,456,522 B） |
| 窗口 / 步长 | 16 s / 16 s（不重叠） |
| beatmap max_length | 2000，`truncation=longest_first`（从尾部截，保留 [CLS] 与音频 token） |
| padding | 官方 `MAX_LENGTH`；探针用 `LONGEST`（被 mask 掉，等价且省算力；`--pad-mode max_length` 可回到字面行为） |
| parser | hitsounds / kiai / mania_sv / timing / timing_points / positions = **true**，snapping = **false** |
| tokenizer | `add_cls_token=true`，`max_time=16000`，`time_step=10` |
| 音频 | 16 kHz 单声道；`hop_length=160`、`window_size=400`、`pad_to_multiple_of=256000`、`audio_length_per_tok=8`、`max_source_positions=1600` |
| 音频 token | 每窗前置 `[AUDIO_BOS]` + 200×`[AUDIO]` + `[AUDIO_EOS]`（200 = 256000/160/8） |
| 池化 | 逐窗向量 = normalize(beatmap_projection(hidden[:, 0]))；图向量 = normalize(逐窗均值) |
| 关键细节 | 有音频时序列第 0 位是 `[AUDIO_BOS]`（不是 [CLS]），池化取的就是它 |
| metadata | 不使用（对 beatmap 塔无影响） |

样本规则：`cm3p_ids_official_4042.csv` 中 `official_checksum_ok=True`，按
`beatmapset_id` 去重（一 set 一张，控制 .osz 下载量），seed 20261110，n=100。

## 2. 结果

| 口径 | n | mean | median | min | p05 | ≥0.99 | ≥0.999 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pilot（前 5 张中 4 张可解码） | 4 | 0.99821 | 0.99833 | 0.99725 | 0.99740 | 4/4 | 0 |
| **全量 100 张** | **98** | **0.99745** | **0.99809** | **0.98129** | **0.99534** | **95/98** | 9 |

- 98 张里 90 张 ≥ 0.996，最低 3 张是 0.9813 / 0.9865 / 0.9872。
- 2 张没有进入统计：`361052` 的 mp3 让 libsndfile 报 `Unspecified internal error`
  （librosa 无法解码，ffmpeg 可以）；`1016315` 的音频首字节是 0 填充的 MPEG 流，
  最初被过严的 magic 检查拒掉（已放宽；librosa 经 audioread 可解码）。
- 低分 3 张的共同点：`tok_max=2000` 且 `n_truncated_windows ∈ {2,2,3}`。

## 3. 低分诊断（results/cm3p_probe_official/lowdiag/）

同一 3 张图，四种口径：

| 变体 | padding | max_length | mean | min |
|---|---|---:|---:|---:|
| F | longest | 2000 | 0.98868 | 0.98129 |
| G | max_length | 2000 | 0.98868 | 0.98129 |
| H | max_length | 4000 | 0.98991 | 0.98519 |
| I | longest（ffmpeg 解码） | 2000 | 失败：站点 ffmpeg 与 conda 的 libp11-kit/libffi 冲突 |

结论：**padding 模式完全不影响结果**（F == G，印证 mask 等价）；把截断上限放到 4000
只把 min 从 0.9813 抬到 0.9852，**截断不是低分主因**。剩余差异更像音频本身（mapset
音频在官方生成后被替换/重编码）或官方 MMRs 侧的音频预处理，而不是 processor 配置。
`decode_audio_ffmpeg` 已改为优先用 imageio-ffmpeg 的静态二进制，绕开站点 ffmpeg 的
动态库冲突。

## 4. 执行位置与数据边界

- **推理只在 HPC**。音频在登录节点用 `scripts/cm3p_probe_fetch_audio.py` 从
  osu.direct 下载，**.osz 只在内存里解析**，落盘的只有 `<beatmap_id>.<ext>`
  音频（`data/probe_audio`，100 张 ≈ 0.4 GB）。本地仓库从未保存 .osz / 音频 / 图片。
- 编码作业是离线的：`sbatch scripts/slurm/cm3p_probe_official.slurm`（CPU 队列，
  16 核，98 张约 15 min；debug 队列 28 min 跑完）。
- 音频按图的 `AudioFilename` 从 .osz 中取（每难度音频不同的 keysound 图也正确）。
- `.osu` 用本地 `data/osu_cache`，逐图 md5 与 whitelist checksum 核对。

## 5. 局限

1. **音频来源不是官方那份**：官方 244K 表使用 Mapperator/MMRs 数据集的音频；探针用
   osu.direct 的当前 .osz。.osu 校验和一致，但若 mapset 的音频在 2025 之后被替换，
   该图就无法逐位复现（3 张低分最可能属于此类）。
2. **官方 dtype 是 bf16（GPU）**，探针是 fp32（CPU）。量级上只会造成 ~1e-3 的
   cos 差，但未单独做 bf16 对照。
3. n=100、一 set 一张；不是 4,042 的全量复现。
4. 截断窗口的低分机制仍未定论（H 只解释了一部分）。

## 6. 扩展到剩余 2,786 张

同一配置已封装为 `scripts/cm3p_embed_todo.py`（分片写 jsonl，含 512 维向量；
`--assemble` 汇总成仓库 content 表 schema，source=2 local-audio）。

音频来源改为 **Hugging Face WebDataset** `project-riz/osu-beatmaps`
（213,068 张图、46,386 条原始音频，2007–2025），由
`scripts/cm3p_fetch_audio_hf.py` 流式扫描：只保留命中 beatmap_id 的音频，不落 .osz；
osu.direct 的 per-IP 配额（8 GiB / 18 h）无法覆盖 1,581 个 set 的 .osz，HF 数据集不需要配额。
扫描后缺口的 set 再用 osu.direct 补（`data/todo_audio_fallback`）。

结果表：`data/processed/chart_content_cm3p_local_audio.parquet`（source=2，audio-on），
sidecar `.meta.json` 记录配置与缺口。与 published 表**可以用**（同一 audio-on 口径），
与本地 audio-off 表（source=1）仍按 mix report 的规则不要混用。

### 6.1 最终结果（2026-11-11）

| 项 | 值 |
|---|---:|
| 目标 | 2,786 |
| 产出向量 | **2,782（99.86%）** |
| 缺口 | 4 张（3 个 set），见下 |
| 范数 | 0.9999998 – 1.0000001 |
| n_windows | 1 – 80 |
| checksum_ok | 2,782 / 2,782 True |
| 跨图 cos（随机 300 张） | mean **0.544** / median 0.547 |
| published 表对照（同口径） | mean **0.569** / median 0.574 |
| 与本地 audio-off 表（source=1） | mean cos 0.331（确认是两种表示，不能混用） |

几何检查与 published 的 audio-on 表示几乎一致（0.544 vs 0.569），说明新增的 2,782 张
确实落在同一空间，而不是 audio-off 那种接近塌缩的表示。

**4 张无法编码的图**（mapset 本身没有可用音频，官方 audio-on 口径同样出不了向量）：

| beatmap_id | set | 原因 |
|---:|---:|---|
| 259067 | 92190 | `AudioFilename: virtual`，0.82 s 占位音频 → 无 16 s 窗口 |
| 374464 | 152065 | 同上，0.30 s |
| 486889 / 487516 | 206487 | .osz 里只有 4 个 .osu，**没有音频文件** |

音频来源：HF WebDataset 覆盖 863 张；osu! 官方（本地会话 + 只上传音频）1,124 张；
osu.direct 补抓其余。HPC 的 IP 在连续请求后被官方限流（TLS 超时），所以音频在本地
串行下载、只 rsync 音频到 HPC，本地不保存 .osz（内存解析）。
最终缺口已写入 `.meta.json`；这 4 张在下游按缺失处理，不做插补。

2026-11-18：这 4 张已从白名单移除（21,949 → 21,945）。[`manifests/whitelist_1k_removed.csv`](../manifests/whitelist_1k_removed.csv)
记录 id / set / 原因；随后把 published 的 4,037 张与本表的 2,782 张合并成 6,819 张
完整 audio-on 表，在 P2/P4 上与 audio-off legacy 做同划分对照，见
[p4_multiseed_audio.md](p4_multiseed_audio.md)。
