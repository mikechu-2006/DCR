# DCR 游戏内封装调研：宿主、运行时与模型分发

> 状态：调研（2026-11-23）。本文重点：**osu!mania 4K**；Malody/Etterna/7K 只在盘点与后续里出现。
> 不改变训练与评测口径。实测数字来自本机（Node v22.23.4 / Intel Core Ultra 5 125H / 单线程），
> 其余引用见文末。

> **2026-11-23 第二版（按你的决定）**：优先 **osu!mania 4K**；宿主采用 **MMA**；
> CM3P 隐向量随客户端分发。按 20k 图 × 512 维算，共 **1.024e7 个浮点数**：
> fp32 约 **40.96 MB**，fp16 约 **20.48 MB**。Malody/Etterna/7K 暂列为后续，不再作为第一版目标。

## 0. 结论

1. **不要把整个 Python 研究栈封装进插件。** 可部署模型本身极小：
   RM_L_full.pt 只有 195 KB，导出成 JSON 约 **34 KB**；纯 JS 线性前向 **0.79 µs/图**，
   和 PyTorch 的最大误差 **6.9e-8**。真正重的是 **CM3P 内容编码**：
   CPU 上 **60–95 s/图**，21,949 张图约 **23 天**；编码必须在离线/服务端做，
   但编码后的向量本身可以随客户端分发（见下一条）。
2. **先做“模型 + 特征提供者 + 宿主适配器”三层，而不是先选 UI。**
   dcr-core 用纯 TS/JS（无 DOM、无 Node 依赖），模型权重用 JSON/二进制，
   Sunny/MSD/CM3P 都是可替换的 provider。这样 osu、Malody、Etterna 只是薄适配器。
3. **第一版直接采用 MMA（MIT）**：fork 后在 js/estimator/ 加 DCR，复用 MMA 的
   osu stable/lazer 数据源、解析器、worker 与 UI；只做 osu!mania 4K。
   这比另写 tosu 计数器少一套 UI/缓存/设置，也天然满足 MMA 的“浏览器模式可用”约束。
4. **隐向量随客户端分发**：20,000 × 512 = **1.024e7 个 float**；
   fp32 约 **40.96 MB**，fp16 约 **20.48 MB**。用二进制表 + MD5 索引，在 Worker 里懒加载；
   不要用 JSON 存 1e7 个浮点数。Malody/Etterna/7K 的桥暂不做。
5. **DCR 的 d 是 Bradley–Terry 潜变量，不是 Dan。** 若要显示段位，必须做校准；
   在 VSRG Dan Benchmark 上，DCR 最好的 RM-L 的 isotonic MAE 是 **0.4216**，
   仍不如单信号 sunny_star 的 **0.3367**。如果目标只是“给玩家一个 DCR 读数”，
   那没问题；如果目标是“比 Sunny 更准的段位”，DCR 目前不是答案。
6. **客户端内容表只覆盖已有约 20k 图。** 新图没有 C，只能 external-only 兜底
   （1k val NLL 0.637 vs 完整模型 0.548，约差 0.09 nats），或服务端补编码。
   Malody/7K 暂不做；CM3P 只吃 .osu 文本，异源图要先转换再编码。

**推荐路线：MMA（fork，osu 4K）+ dcrEstimator + 客户端 C 表。**

---

## 1. DCR 里什么重、什么轻

### 1.1 可部署模型

当前仓库里有两个可以直接部署的 checkpoint（fit-all，非协议模型）：

| 文件 | 类型 | 输入 | 大小 | 说明 |
|---|---|---|---:|---|
| results/benchmark/models/RM_L_full.pt | linear_extra | 512-d CM3P + msd_overall + sunny_star | 195 KB | randommerge 全量：453,406 行 / 20,388 图；**推荐做部署基线** |
| results/benchmark/models/M1_L_full.pt | mlp_extra | 同上 | 223 KB | 1k 全量；MLP 残差，部署更复杂；本版不用 |

两者的公式都是

~~~text
d = f([C, S])
C = 512 维 CM3P 内容向量（冻结；训练图逐维标准化）
S = [msd_overall, sunny_star]（训练图标准化）
d = Bradley–Terry 潜变量；只定义到仿射变换，不是 Dan/星数
~~~

RM_L_full.pt 是线性的：d = w'[C, S]，514 个权重。M1_L_full.pt 是
w'[C,S] + MLP([C,S])，514 → 64 → 64 → 1，带 LayerNorm/GELU。

**部署选择建议：先用 RM-L。** 它更小、训练图更多（20,388 vs 6,819）、
在 VSRG benchmark 上比旧 M1-L 好很多（MAE 0.4216 vs 0.7411）。如果之后确认
MLP 在目标人群上确实更好，再换 M1_L，接口不变。

### 1.2 推理实测（本机 spike）

把 RM_L_full.pt 导出为 JSON（mu/sd、extra_mu/sd、w、d_mu/d_sd），
用纯 JS 做同样的标准化与点积，和 PyTorch 对照：

| 模型 | JSON 大小 | 纯 JS 前向 | 与 PyTorch 最大绝对误差 |
|---|---:|---:|---:|
| RM-L（线性） | 33,962 B | **0.79 µs/图**（约 126 万图/s） | **6.9e-8** |
| M1-L（MLP） | 876,843 B | **49 µs/图** | **~2e-8** |

结论：**模型推理不是性能问题**。34 KB 的 JSON 甚至可以内联进插件；不需要 ONNX、
不需要 WASM、不需要 Python。线性前向的核心就是：

~~~js
function score(model, C, S) {
  let d = 0;
  for (let i = 0; i < model.content_dim; i++)
    d += model.w[i] * ((C[i] - model.mu[i]) / model.sd[i]);
  for (let j = 0; j < S.length; j++)
    d += model.w[model.content_dim + j] * ((S[j] - model.extra_mu[j]) / model.extra_sd[j]);
  return (d - model.d_mu_raw) / model.d_sd_raw;
}
~~~

### 1.3 真正重的是 CM3P

CM3P（OliBomby/CM3P）是一个 ModernBERT 双塔，本仓库用的是 audio-off、crop=legacy、
max_length=2000 的冻结向量。CPU 单图：

| 样本 | 窗口数 | 总 wall |
|---|---:|---:|
| 1 张（422 notes） | 7 | 68.7 s（含模型加载） |
| 4 张随机 | 42 | 380.1 s（均 95 s/图） |

稳态 **60–95 s/图**，21,949 张纯 CPU 约 **23 天**；A800 上 45–90 min 才是可接受量级。
来源：docs/cm3p_crop_windows.md §CPU 单图耗时。

因此部署只有三种选择：

1. **客户端分发 + 本地查表**（本版采用）：20k × 512 × fp32 ≈ **40.96 MB**
   （fp16 约 **20.48 MB**）；按 beatmap MD5 查表，运行时 0 编码成本。
   表里同时放 msd_overall_505 / sunny_star 两个标量，已知图可以完全复现训练特征；
   新图走兜底。
2. **服务端编码**：客户端把 .osu 或 MD5 发给有 GPU 的服务，返回 C；
   适合新图，但引入网络与隐私问题。
3. **不用 C，只外部信号**：Sunny + MSD 的 ext_blend；最轻，但 NLL 从 0.548 退到 0.637。

### 1.4 外部信号的版本敏感性

DCR 的训练特征是有“版本”的：

| 特征 | 训练口径 | 部署时必须 |
|---|---|---|
| sunny_star | osumania_map_analyser commit 448ed7c789af3d715acfb8370e41244cc3c00865 | 固定同一版 Sunny；换版要重训或做一致性测试 |
| msd_overall | MinaCalc standalone calc_version 505，goal 0.93，rate 1.0 | 固定同一版；MMA 自带的是 Etterna 0.68–0.75 的 MSD，数值不等价 |
| CM3P C | OliBomby/CM3P@26cc694，audio off，crop legacy | 固定 revision、裁剪与 pooling；换版等于换输入分布 |

**这是封装时最容易踩的坑**：模型没变，但 msd_overall 换了 MinaCalc 版本，
分数就会漂移，而且很难从 UI 上看出来。建议在仓库里放一个
feature-contract.json（provider 名称/版本/哈希）和一个 100 图 parity 测试，
CI 里对 Python 参考实现做 max |Δd| < 1e-6。

### 1.5 客户端隐向量表（本版核心）

按 20,000 张图、512 维算：

| 精度 | 元素数 | 原始字节 | 说明 |
|---|---:|---:|---|
| fp32 | 10,240,000 | **40,960,000 B（约 40.96 MB / 39.06 MiB）** | 默认；精度最稳 |
| fp16 | 10,240,000 | **20,480,000 B（约 20.48 MB / 19.53 MiB）** | 省一半；查表时解码 512 个 half |

再加上每图两个标量（msd_overall_505、sunny_star），fp32 总共约 **41.1 MB**；
两个标量只增加 160 KB，但能让已知图完全复现训练特征，避免运行时再算 MinaCalc/Sunny。
如果按当前内容表的全量 **21,949** 图算，fp32 约 **45.1 MB**、fp16 约 **22.5 MB**。

**不要用 JSON。** 1e7 个浮点数写成 JSON 是几百 MB 文本，浏览器解析会卡住；用二进制：

~~~text
magic "DCRC" | version u16 | dim u16 | count u32 | dtype u8 | reserved
repeat count:
  md5[16] | C[dim] (fp32 或 fp16) | msd_overall_505 f32 | sunny_star f32
~~~

- 按 md5 排序，记录定长；二分查找或预建 Map<hex, offset>，单图查找 <1 ms。
- 文件放 MMA 插件目录，例如 `model/dcr/content_v1.f32`；Worker 里 `fetch` 后
  `arrayBuffer()`，直接 `new Float32Array(buffer)`，不要再复制一份。
- 40 MB 表 + 1 MB 模型，Worker 常驻内存约 40–50 MB；浏览器 HTTP 缓存会保留；
  可选 IndexedDB 缓存，但不要重复存两份。
- 清单 `content_v1.manifest.json` 记录 count/dim/dtype/cm3p_rev/model_id/sha256；
  模型与表必须同版本原子更新，缓存键带 manifest hash。
- 已知图：读 C + 两个标量，**不再调用 Sunny/MinaCalc**，所以没有版本漂移。
- 未知图：ContentProvider 返回 null；用 MMA 当前 Sunny + MSD 近似，或 external-only，
  结果标 `approx` 并在 UI 提示；不要静默当成精确值。
- 分发：40 MB 可以随插件发，也可以首次运行时下载；CM3P 代码是 MIT，但谱面向量再分发
  涉及谱面内容版权，发布前确认。
- fp16 是否够用：导出后抽 100 图比较 fp16 与 fp32 的 d，`max |Δd|` 可接受再用；
  否则默认 fp32。40 MB 对桌面浏览器不算大，优先正确性。

---

## 2. 现有同类项目与宿主

### 2.1 难度/分析插件

| 项目 | 覆盖 | 形态 | 技术栈 | 性能/体积 | 许可 | 对 DCR 的价值 |
|---|---|---|---|---|---|---|
| ManiaMapAnalyser (MMA) | osu 4/6/7K stable/lazer、Etterna、Malody V/编辑器、Malody 4.3.7 | tosu 计数器 + Tauri 桌面壳 | JS + Rust + WASM（MinaCalc） | 分析 <1 s，壳约 10 MB RAM | MIT | **最佳多游戏宿主/参考**；有 estimator 插件点，桥已解决 |
| osu-star-web | osu 4K | 浏览器 + tosu 插件 | JS + ONNX Runtime Web | 首次加载模型约 16 MB | **无 LICENSE 文件** | 代码干净、适合参考 tosu 计数器；**不能直接复用** |
| Daniel | osu 4K | 独立 exe + tosu | C++（XDC1） | 很轻，选图时计算 | MIT | 单算法、实时 overlay 的轻量参考；不适合扩展 |
| DanOverlay | osu 4K/7K、Etterna、Quaver SR | Python + pywebview 桌面 overlay | Python 3.12 + JS | 单文件 71 MB，分析 0.5–1.2 s | MIT | 最容易塞 torch 模型，但**违反低性能/易改** |
| Companella | osu 4K | C# 桌面 companion | C# + ONNX（模型 298 KB） | 桌面应用，非实时 overlay | MIT | ONNX 小模型分发参考；不适合做 overlay |
| Koishi | osu 4K RC 11–17 | Node CLI | 纯 JS + 6.4 MB model.json | 轻 | MIT | “模型 + JS” 单文件分发的直接参考 |
| xdc1 | osu 4K RC/LN | C++ 算法 | C++ | 轻，可编 WASM | MIT | 4K 算法；不是宿主 |
| Proj7k（随 MMA） | osu 7K | JS/WASM estimator | JS | 轻 | 随 MMA（MIT） | 7K 兜底算法；DCR 未验证 7K |
| StarRatingRebirth | osu 4K | Python 算法 | Python | 参考实现 | **无 LICENSE** | Sunny 口径参考；不要直接搬 |

### 2.2 宿主/插件 API

| 宿主 | 支持游戏 | 插件形态 | 接入难度 | 备注 |
|---|---|---|---|---|
| tosu | osu stable/lazer | static/<counter>/ 网页 + WebSocket | 低 | LGPL-3.0；独立网页，不注入游戏；/files/beatmap/file 可取当前 .osu；本版作为备选 |
| MMA | osu/Etterna/Malody V/4.3.7 | tosu 计数器 + 自带 Tauri 壳 + 游戏桥 | 中 | **本版首选**；MIT；多源、4/6/7K、解析器/转换器/worker/UI 都现成；代码体量大 |
| osu!lazer ruleset | osu lazer | C# ruleset DLL | 高 | ruleset 是“玩法模式”，不是 overlay；官方 API 不面向 overlay；不建议 |
| Quaver | Quaver | **仅编辑器 Lua 插件** | 高 | 运行时没有难度/overlay 插件 API；要么改游戏，要么外置 |
| Malody V | Malody V | 编辑器 Lua 插件；运行时需 BepInEx IL2CPP 桥 | 高 | 无官方运行时插件 API；桥是 Windows-only、版本敏感 |
| Etterna | Etterna | Lua 主题脚本 + 文件 IPC | 中 | 无插件 API；主题更新会覆盖；适合复用 MMA 的 mma_bridge.lua |
| StepMania/ITGmania | 4/7K | Lua 主题 | 中 | 同 Etterna 模式；没有统一插件市场 |

**几个重要事实：**

- **Malody 没有“官方运行时难度插件 API”。** 公开可用的只有：
  1. 编辑器 Lua 插件（Editor:WriteFile/ReadFile/ChartInfo），可以把分析请求写到文件，
     由外部壳读取同目录 .mc 分析；MMA 的 mma_editor.lua 就是这条路。
  2. 皮肤 Lua 脚本（UI/网络请求），不是难度计算。
  3. 运行时选曲/判定读取需要 BepInEx IL2CPP 注入（MMA 的 MMAMalodySelection.dll，MIT fork），
     Windows-only，游戏更新可能失效。
- **Quaver 的插件是“编辑器扩展”，不是运行时 overlay。** 官方文档明确写的是 editor plugins；
  它没有给外部难度读数用的运行时插件接口。
- **osu!lazer 的 ruleset 不是 overlay 插件。** 它替换玩法/判定/计分；把难度卡片塞进去
  既不是 API 的设计目标，也会带来分发与账号风险。osu! 的 overlay 生态在 tosu/StreamCompanion 这一侧。
- **Etterna 只有 Lua 主题 + 文件 IPC。** 这不是“插件”，但足够让外部程序知道当前选了什么图；
  MMA 的桥已经把这条路走通。

---

## 3. 五条封装路线

| 路线 | 做法 | 覆盖 | 工作量 | 性能 | 可改性 | 风险 |
|---|---|---|---:|---|---|---|
| **A. MMA + DCR estimator（本版）** | fork MMA；加 js/estimator/dcrEstimator.js；客户端 C 表；复用数据源/worker/UI | **osu 4K** | 1–2 周 | 低（MMA 已 <1 s；表加载一次） | 高（单个 estimator + 表） | 中；维护 fork/上游合并 |
| **B. 独立 tosu 计数器（备选）** | 纯 JS core；tosu 读当前 .osu；本地算 d；网页卡片 | osu 4K | 1–2 周 | 极低 | 高 | 低；但 UI/缓存/设置要自己做 |
| **C. 独立 Tauri/Electron + 自写桥（后续）** | 自己做壳 + Malody/Etterna 桥 | 4/7K、Malody/Etterna | 4–8 周 | 低–中 | 高 | 高；重复造 MMA 的轮子 |
| **D. Python sidecar / PyInstaller** | 本地 HTTP/进程跑 torch；插件调用 | 取决于壳 | 1 周 | **差**（启动/内存/打包） | 中 | 高；杀软、Python 版本、GPU 依赖 |
| **E. lazer ruleset / Quaver 运行时插件** | 改游戏 | 无 | 高 | 好 | 低 | 高；不是 API 目标，分发困难 |

**建议：A 做第一版；B 只在你不想维护 MMA fork 时做；C 是多游戏后续。**

理由：

- A 复用 MMA 已解决的 osu 数据源、解析、worker、缓存、设置、OBS/游戏内 overlay；
  你只加一个 estimator 和一张表。
- B 的代码面最小，但等于重做 MMA 的 UI/缓存/设置；除非你明确要脱离 MMA。
- C 的 Malody/Etterna 桥是 4–8 周的活；本版只做 osu 4K，不做。
- D 与“性能需求低、支持修改”直接冲突：Python 运行时、torch、CM3P 都不该进客户端。
- E 不是封装，是改游戏；不建议。

---

## 4. 推荐架构：dcr-core + providers + adapters

### 4.1 分层

~~~text
┌────────────────────────────── 宿主：MMA（osu 4K）──────────────────────┐
│  MMA 页面 / tosu 计数器 / 桌面壳                                        │
│  js/estimator/dcrEstimator.js → runAnalysisPipeline → UI                │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │  AnalyzeInput
┌──────────────────────────────────▼─────────────────────────────────────┐
│                         dcr-core（MMA 内模块）                         │
│  parse .osu → 校验 md5 → 查 C 表 → 取 S → model.forward → d             │
│  纯函数；无 DOM；跑在 MMA 的 compute.worker.js 里                        │
└───────┬──────────────────┬──────────────────┬────────────────────────┘
        │                  │                  │
  ContentTable         SignalProvider      Calibrator
  - 客户端 40 MB 表    - 已知图：表内 S     - isotonic / linear
  - 远程 /embed（可选） - 新图：MMA Sunny/MSD - 无校准则只出 raw d
  - null（兜底）       - OD（解析）
~~~

### 4.2 接口契约

~~~ts
export type AnalyzeInput = {
  osuText: string;          // 已转换/已读取的 .osu 文本
  md5: string;              // 谱面 MD5（缓存/索引键）
  rate: number;             // 1.0 / 1.5 / 0.75 / lazer 自定义
  mods: string[];           // 只影响需要重算的特征
};

export type AnalyzeResult = {
  d: number;                // BT 潜变量
  calibrated?: number;      // 可选：Dan/星数刻度
  source: "content+external" | "external-only" | "unknown";
  signals: Record<string, number>;
  modelId: string;
  warnings: string[];
};

export type ContentRecord = {
  C: Float32Array;   // 512 维
  msd: number;       // msd_overall_505（训练口径）
  sunny: number;     // sunny_star（训练口径）
};

export interface ContentTable { get(md5: string): ContentRecord | null; }
export interface SignalProvider { name: string; version: string; compute(input: AnalyzeInput): number | null; }
export interface Calibrator { apply(d: number): number; }
~~~

适配器只做三件事：拿到谱面文本/MD5/rate/mods；调用 analyze；渲染/上报。
不要把解析、特征、模型散落在 UI 代码里。

MMA 的 estimator 返回契约需要适配：它要求 `star`、`numericDifficulty`、`estDiff` 三个字段。
DCR 没有 star 语义，建议：

- `numericDifficulty` 放校准后的 DCR/Dan 值；
- `star` 放一个单调映射值，仅用于 MMA 现有 UI 排序/配色，并在卡片上标 **DCR**；
- 或者小改 MMA，允许 `star=null` + `displayLabel="DCR"`，这是更干净但改动更大的做法。

不要静默把 DCR 的 d 当官方 star 展示。

### 4.3 模型文件

推荐两种格式，二选一：

1. **model.json（最简单）**：34 KB（RM-L），字段：
   schema_version, model_type, content_dim, extra_names, mu, sd, extra_mu, extra_sd, d_mu_raw, d_sd_raw, w
   （MLP 再加 norm_weight/bias, mlp[], out_weight/bias）。
2. **model.bin + model.meta.json（更快加载）**：float32 权重平铺；元数据放 JSON。
   对 34 KB 的线性模型没必要；对 MLP 的 877 KB 有意义。

必须带：

- model_id（权重/schema 的哈希）
- train_git_commit、content_md5、extra_names
- feature_contract（Sunny/MSD/CM3P 版本）
- calibration（可选，单独文件）
- model_card（训练数据、指标、限制）

### 4.4 内容索引（客户端）

本版直接按 §1.5 的表实现；默认 fp32，20k 图约 **41 MB**（含 S 两个标量）。

- 文件：`ManiaMapAnalyser by Leo_Black/model/dcr/content_v1.f32` + `content_v1.manifest.json`。
- 加载：dcrEstimator 首次调用时 fetch → ArrayBuffer → Float32Array；只加载一次；失败则 external-only。
- 查询：beatmap.md5（tosu/壳提供，转小写）；没有 MD5 时用 rawText 的 MD5，或拒绝精确路径。
- 已知图：C/S 直接来自表；**不调用 Sunny/MinaCalc**，所以没有版本漂移。
- 未知图：ContentProvider 返回 null → 外部信号兜底，标 approx，UI 提示。
- 更新：manifest 里带 sha256/model_id；表与模型一起换；缓存键带 hash。
- 体积优化（后续）：fp16 约 20.5 MB；64-d PCA 约 2.8 MB（需重训并复测 NLL）；int8 约 11 MB。

构建步骤（一次性）：

1. 从 `data/processed/chart_content_cm3p_local.parquet` 取 `beatmap_id → embedding`；
2. 从 `data/processed/rank_signals_1k.parquet` 取 `msd_overall`、`sunny_star`；
3. 对 `data/osu_cache/{bid}.osu` 算 MD5；
4. 按 MD5 排序，写定长记录 + manifest；
5. 抽 100 图，用 Python `benchmark_infer.py` 对照 d，最大差 <1e-6 才发布。

注意：表里存**原始** `msd/sunny`（以及原始 C），标准化在客户端用模型里的 mu/sd 做；
不要在打包时先标准化，否则换模型就不能复用表。

### 4.5 缓存

缓存键必须包含：

~~~text
md5 + rate + mods + model_id + content_hash + (未知图时的 provider 版本)
~~~

只缓存 AnalyzeResult；切图/切 mod 时取消在途请求；打歌中不计算。
已知图因为没有运行时特征计算，缓存键可以只带 content_hash；未知图才带 Sunny/MSD provider 版本。

---

## 5. 性能与体积预算

| 步骤 | 实测/估算 | 说明 |
|---|---:|---|
| 线性模型前向 | **0.79 µs** | 本机 Node，单线程，514 维 |
| MLP 前向 | **49 µs** | 514→64→64→1 |
| .osu 解析 | 2–10 ms | 取决于谱面长度；MMA 解析器可复用 |
| 客户端表加载 | 0.1–1 s | 41 MB 本地 fetch；只一次，放 Worker |
| 表查找 | 0.1–1 ms | MD5 二分/Map |
| 已知图特征计算 | **0 ms** | C + msd + sunny 都在表里 |
| 未知图 Sunny | **~6 ms** | 50 图 0.32 s（含 Node 启动），约 6 ms/图 |
| 未知图 MSD（WASM） | 20–100 ms | MMA 整体分析 <1 s；版本不一致要标 approx |
| **总计（已知图）** | **<20 ms/图** | 解析 + 查表 + 模型；不含首次 41 MB 加载 |
| **CM3P 编码** | **60–95 s/图** | 不进客户端 |
| 表体积 | **~41 MB** fp32 | fp16 约 20.5 MB；64-d PCA 约 2.8 MB（需重训） |

内存：41 MB 表 + 约 1 MB 模型 + 解析/UI，Worker 常驻约 50–60 MB；
MMA 壳本身约 10 MB；DanOverlay 单文件 71 MB（对比）。

---

## 6. 各游戏怎么接

### 6.1 MMA（首选）：osu!mania 4K

MMA 是 MIT 的完整插件；fork 后只加一个 estimator 和一张表。具体落点（以本地 v3 源码为准）：

| 文件/位置 | 改动 |
|---|---|
| js/estimator/dcrEstimator.js | 新增；runDcrEstimatorFromText(rawText, options)；加载 model/content；返回 MMA 结果契约 |
| js/pipeline/runAnalysisPipeline.js | 分派分支：estimator === "DCR" 时调用 dcrEstimator |
| js/app/worker/compute.worker.js | 旧估算器路径可选加 DCR；pipeline 路径已能覆盖 |
| config.js | options.estimatorAlgorithm 加 "DCR"；defaults 可选 |
| settings.json | Estimator Algorithm 的 options 加 "DCR" |
| js/parser/settingsParser.js | normalizeEstimatorAlgorithmValue 加 dcr 分支；estimatorAlgorithmSet 自动读 config |
| js/app/ui/display.js | DCR 标签/数值展示；不要冒充官方 star |

流程：

1. MMA 已有 rawText + beatmap.md5（tosu/壳/外部源）；传给 estimator。
2. dcrEstimator 首次调用 fetch model/dcr/content_v1.f32，缓存 Promise。
3. 查 MD5 → C + msd + sunny；加载 34 KB model.json → d。
4. 返回 { star: mapped, numericDifficulty: calibrated, estDiff: "DCR ..." }。
5. 结果缓存键 md5 + rate + mods + content_hash + model_id。
6. 打歌中不计算；未知图标 approx；设置里可开关/切换 DCR。

为什么用 MMA 而不是另写 tosu：

- 数据源/解析/worker/缓存/设置/OBS/游戏内 overlay 都是现成的；
- 浏览器模式（无壳）仍能只靠 tosu 工作，符合 MMA 的架构约束；
- 以后要接 Malody/Etterna/7K，只需要同一个 estimator 再拿一个 source，不用重做 UI。

注意：

- MMA 的 runAnalysisPipeline 是异步的，DCR 的表加载可以 await；
- Worker 里不能直接读本地文件，必须 fetch 静态资源；
- 41 MB 表放插件目录，首次加载；普通静态服务即可；
- 如果不想维护 fork，可以先做成 MMA 的分支/PR；MIT 允许。

### 6.2 独立 tosu 计数器（备选）

如果不想维护 MMA fork：

- 静态网页 + metadata.txt；WebSocket v2；GET /files/beatmap/file 取当前 .osu；
- 复用同一个 dcr-core；model + content 表放同目录；
- UI/缓存/设置要自己做；
- 适合只想要一个轻量卡片、不关心 MMA 其他功能。

### 6.3 Malody V（本版不做）

- 编辑器分析：Malody V 编辑器 Lua 插件 + 外部 companion；参考 MMA 的 mma_editor.lua。
- 选曲/游玩内叠加：BepInEx IL2CPP 桥；复用 MMA 的 MMAMalodySelection.dll（Windows-only）。
- 数据问题：CM3P 只吃 .osu，.mc 要先转换；客户端 C 表没有 Malody 谱面，
  只能 external-only 或服务端编码。DCR 是 4K osu 训练，Malody 分布未验证。
- 结论：等 osu 4K 稳定后再做，不要放进第一版。

### 6.4 Etterna

- 复制 MMA 的 bridges/etterna/mma_bridge.lua（MIT）到主题
  BGAnimations/ScreenSelectMusic decorations/，并在 default.lua 里 LoadActor。
- 它每 0.5 s 把 title/artist/song_dir/step_file/rate/msd_* 写到 Save/MmaBridge.txt。
- companion 监听这个文件 → 读 .sm/.ssc → MMA 的 simfile parser/转换器 → .osu → dcr-core。
- 主题更新会覆盖注入行；安装器要能重装。
- msd_* 是 Etterna MinaCalc 的值，不是 DCR 训练用的 MinaCalc 505；**不要直接当 msd_overall**。
  要么用 DCR 的 standalone 505（native/WASM），要么重训一个用 Etterna MSD 的 DCR 变体。

### 6.5 Quaver

- 官方插件是**编辑器 Lua 扩展**，没有运行时 overlay 接口。
- 如果只是要 Quaver 的 SR，DanOverlay 的 tools/qsr 已经用 Quaver.API 做了一个 394 KB 的 CLI
  （需要 .NET 8 runtime），可以作为“外部读数”参考。
- 要做实时 overlay：只能外置（读日志/文件）或 fork 游戏；不建议为 DCR 走这条路。

### 6.6 osu!lazer ruleset

- 不建议。ruleset 是玩法模式（判定/计分/玩法），不是难度卡片宿主；C#/.NET 分发、版本跟进、
  账号风险都不划算。
- osu! 的 overlay 走 tosu / StreamCompanion；这是社区实际路径。

### 6.7 7K（本版不做）

- DCR 的训练数据/评测是 **4K osu!mania**；CM3P 是 column-blind，但 d 的尺度是 4K 玩家 ACC 序。
  没有 7K 验证前，不要把 DCR 的 7K 读数当结论。
- 7K 现成方案：MMA 的 Polaris（7K RC/LN）、Proj7k（7K 专用引擎）、Quaver.API 的 7K SR。
- 如果要做 DCR 7K：先用 7K 数据重训/校准，或至少做“DCR d 与 7K Dan/成绩”的相关性验证；
  预计算索引也要覆盖 7K 谱面（CM3P 编码一次）。

---

## 7. 特征合同、校准与“不要假装它是 Dan”

### 7.1 特征合同

在仓库里放 deploy/feature-contract.json：

~~~json
{
  "cm3p": {"repo": "OliBomby/CM3P", "rev": "26cc694", "audio": false, "crop": "legacy", "pool": "mean"},
  "sunny": {"repo": "LeoBlackMT/osumania_map_analyser", "commit": "448ed7c", "entry": "runSunnyEstimatorFromText"},
  "msd": {"engine": "MinaCalc", "calc_version": 505, "goal": 0.93, "rate": 1.0},
  "od": {"source": "osu_file"},
  "model": {"file": "RM_L_full.pt", "content_md5": "0c2d5da5191da6976601026cd38371d1"}
}
~~~

配套：

- tests/parity_100.json：100 张图的 C/S/d 参考值；
- node tests/parity.mjs：max |Δd| < 1e-6 才通过；
- CI 在升级 Sunny/MinaCalc/CM3P 时强制跑。

### 7.2 校准

d 是 BT 潜变量，只有仿射尺度。要显示 Dan/星数，需要：

1. 一批带 Dan 标签的谱面（例如 VSRG benchmark，或你自己的标注）；
2. 单调校准（isotonic 最稳；线性只作对照）；
3. 把校准 knots 存成 calibration.json，与 model_id 绑定；
4. UI 明确写“DCR 读数”还是“校准后 Dan”。

**现状数字**（VSRG benchmark，1,382 张 4K）：

| 方法 | isotonic MAE |
|---|---:|
| RM-L（DCR 最好） | 0.4216 |
| Sunny 单信号 | **0.3367** |
| M1-L（旧） | 0.7411 |

所以：**如果产品目标是“比 Sunny 更准的段位”，DCR 现在不该做主角。**
如果目标是“把 DCR 的研究分数带给玩家/作者”，那就显示 raw d 或校准后的 DCR 分，
不要标成官方 Dan。

### 7.3 OD 与绝对 ACC

CM3P 的谱面词表没有 OD/HP/AR，d 是 OD 盲的。要预测“我会打多少 ACC”，
必须把 OD、keycount、玩家能力接进去；这不是当前 d 的语义。UI 上不要把 d 解释成绝对难度/ACC。

---

## 8. 实施计划

| 阶段 | 内容 | 产出 | 时间 |
|---|---|---|---:|
| P0 | 导出 RM-L 到 JSON；生成 20k 图 C+S 表；纯 JS parity | model.json + content_v1.f32 + manifest | 1–2 天 |
| P1 | fork MMA；加 dcrEstimator；接入 pipeline/worker/设置/UI | DCR 算法可在 MMA 里选 | 1–2 周 |
| P2 | 校准/标签/缓存/打包；真机测试 | 可发布的 MMA 分支 | 3–5 天 |
| P3 | 可选：独立 tosu 计数器；或 Malody/Etterna/7K | 多宿主/多游戏 | 后续 |
| P4 | fp16/64-d PCA/int8、蒸馏、新图补编码 | 更小/更稳的部署模型 | 研究项 |

P0 的模型部分已在本机验证过（见 §1.2）；P0 的表生成还需要按 §4.4 的步骤跑一次。

**P1 的验收**：

- MMA 里能选 DCR；4K 谱面出数；
- 100 张图与 Python benchmark_infer.py 的 d_m 最大差 <1e-6（已知图）；
- 首次加载 41 MB 表 <1 s（本地 SSD/普通静态服务）；之后单图 <20 ms；
- 未知图走 external-only，UI 标 approx；
- 不引入 Python/torch/CUDA 运行时依赖；
- 打歌中不计算，切图/切 mod 不串结果。

**P2 的验收**：

- 真实 tosu（stable + lazer）各测 4K 10 张图；
- 校准后的显示有明确标签（DCR/Dan，不冒充官方 star）；
- 打包体积增加约 41 MB，可接受或改首次下载；
- OBS 浏览器源与游戏内 overlay 都能显示。

---

## 9. 风险与待决

1. **特征漂移**：已知图用表内 C/S，天然冻结；未知图仍受 MMA Sunny/MSD 版本影响。
   必须 pin 版本 + parity 测试，并在 UI 区分 exact/approx。
2. **MSD 505**：已知图的 msd_overall_505 已随表分发，不需要 MinaCalc；未知图若用 MMA 的
   Etterna MSD，数值不等价，只能标 approx。可选后续：把 505 编 WASM，或训练一个只用
   sunny_star 的 DCR 变体（external blend 与 Sunny 单独只差约 0.0005 NLL）。
3. **客户端表分发**：41 MB 可接受，但要确认谱面向量再分发的许可/归属；
   CM3P 代码 MIT，谱面内容本身有版权。发布前决定“随插件”还是“首次下载”。
4. **MMA 结果契约**：MMA 要求 star 有限；DCR 没有 star。要么给单调映射并明确标 DCR，
   要么小改 MMA 允许 star=null + label。不要静默冒充官方 star。
5. **内存/首载**：41 MB 表 + Worker 常驻；首次 fetch 可能 0.1–1 s；低端机要测。
   不要 JSON，不要重复复制 ArrayBuffer。
6. **原子更新**：模型、表、manifest 必须同版本；缓存键带 content_hash；否则会出现
   “新模型 + 旧表”的静默错分。
7. **Dan 校准**：没有校准就不要叫 Dan；校准数据与模型绑定。
8. **Malody/7K**：本版不做；客户端表没有这些谱面，DCR 4K 训练也未验证 7K。
9. **许可证**：osu-star-web、StarRatingRebirth 没有 LICENSE，不能直接复制代码；
   tosu 是 LGPL-3.0；Quaver 是 MPL-2.0；MMA/Daniel/DanOverlay/Companella/Koishi/xdc1/DCR 是 MIT。
10. **维护成本**：MMA fork 要跟上游合并；tosu 的内存偏移随 osu 更新；Malody BepInEx
    随游戏更新；Etterna 主题更新覆盖注入。本版只承担 MMA fork 的维护面。

---

## 10. 参考

- DCR 仓库：README.md、docs/models.md、docs/cm3p_rank.md、docs/benchmark_vsrg.md、
  docs/benchmark_linear_extra.md、docs/cm3p_crop_windows.md
- tosu: https://github.com/tosuapp/tosu（LGPL-3.0；计数器/WS API）
- ManiaMapAnalyser: https://github.com/LeoBlackMT/osumania_map_analyser（MIT；多游戏/桥）
- osu-star-web: https://github.com/sky2002/osu-star-web（无 LICENSE）
- Daniel: https://github.com/TheBagelOfMan/Daniel（MIT）
- DanOverlay: https://github.com/acarranzao1a-png/Dan-Overlay（MIT）
- Companella: https://github.com/leyna64/companella（MIT）
- Koishi: https://github.com/AquaStoneLZS/Koishi（MIT）
- xdc1: https://github.com/TheBagelOfMan/xdc1（MIT）
- CM3P: https://github.com/OliBomby/CM3P（MIT）
- Quaver editor plugins: https://quavergame.com/docs/editor/plugins
- Etterna: https://github.com/etternagame/etterna（MIT）
- VSRG Dan Benchmark: https://github.com/LeoBlackMT/VSRG-DanEstimation-Benchmark
