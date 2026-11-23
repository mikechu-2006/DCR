# 实验计划 v2：DCR 的 4 个 CM3P 组合上 VSRG Dan Benchmark（2026-11-19）

> **状态：已执行（2026-11-19）。** 结果与限制见 [benchmark_vsrg.md](benchmark_vsrg.md)。
> 本文保留为执行前的计划记录；实际数值、覆盖率与负迁移结论以报告为准。
> 相对 v1 的变化：
> **不启用 audio-on 臂**；模型权重用本地缓存/HPC；Dan 校准改为**按谱面组的 5 折交叉验证 + 保序回归**。
> 本文只定义方案；尚未跑 benchmark、尚未改代码、尚未提交结果。
>
> **Benchmark 快照**：[VSRG-DanEstimation-Benchmark](https://github.com/LeoBlackMT/VSRG-DanEstimation-Benchmark)
> commit `e15f89b834ad6cdb055480039aa583baff802a43`（2026-11-15）。
> `samples/samples.7z` sha256 `8836e24c…`，`samples/data.csv` sha256 `2b43165a…`。
> 本地只读克隆在 `.cache/benchmark/VSRG-DanEstimation-Benchmark/`。

---

## 0. 先回答：我们的方法输出是不是一个实数？

**是，但要把“原始分数”和“Dan 估计”分开：**

1. 模型对每张谱面输出一个标量实数
   `d_m = w'C`（线性）或 `d_m = w'C + MLP(C)`（MLP）。
   它不是段位标签，也不是分类概率；同一张谱面永远得到一个实数。
2. 这个 `d_m` 是 Bradley–Terry 的**潜变量**，单位是任意的 logit；只有“同一模型内的相对大小”有意义，**不能直接当成 Dan**。
3. benchmark 的 `got` 必须与 `expected` 同为 Reform Dan 数值尺度，所以最终交给 benchmark 的是校准后的实数：
   `got = f(d_m)`，例如 `got = 14.23`。
4. 所以最终方法包含两个实数：**原始难度分数 `d_m`**（用于排序/诊断）和 **Dan 估计 `got`**（用于 benchmark 的 MAE/RMSE）。benchmark 只消费后者；只给顺序不够，因为 MAE 需要数值。

**关于“不同段之间距离不一定相等”：** 同意，不能假设 Dan 刻度等距。因此校准不只用线性回归：
- 主校准：**保序回归（isotonic regression）**，只要求 `f` 单调不减，不要求等距；拟合出的 `f` 是分段线性的实数映射。
- 对照：线性校准 `a + b·d`（会强制等距，仅作 baseline/敏感性）。
- 同时报 Spearman/Kendall/pair accuracy，避免把“校准曲线拟合得好”误读成“难度序更好”。

---

## 1. 契约事实

| 项 | 实际内容 |
|---|---|
| 样本 | `samples/data.csv` **1,382 行** + 1,382 个 `.osu`（4K、Mode 3） |
| 构成 | RC 1,246、LN 102、course 34 |
| 输出 | 每个算法一份 `results/{Algorithm}.csv`，8 列；只覆盖 `got/delta/deltaAbs` |
| `got` | Reform Dan 数值难度（1st dan=1.0，alpha=11.0…），必须与 `expected` 同尺度 |
| 错误行 | `Invalid` / `Missing` / `Failed`，不参与 MAE/RMSE |
| 音频 | **不需要**（本计划已移除 audio-on 臂） |
| 标签分布 | `expected` 1.0–18.3，25–75% 分位 13.4–14.2；常数预测 MAE 1.55 |

**不再需要**：下载音频、rate 伸缩、audio-on 编码、音频覆盖率门槛。benchmark 只提供 `.osu` 这件事不再构成阻塞。

---

## 2. 要测的 4 个组合

| ID | 方法 | 臂 | 输入 | 关键超参 |
|---|---|---|---|---|
| **M1-L** | CM3P + external MLP | legacy（audio-off） | `[C, msd_overall, sunny_star]` | h64、depth 2、dropout 0.1、98 epoch、lr 3e-3、lam 1e-2 |
| **M1-C** | CM3P + external MLP | canon（audio-off） | 同上 | 同上 |
| **M2-L** | CM3P MLP h64 L2 | legacy | `C` | 同上，无 external |
| **M2-C** | CM3P MLP h64 L2 | canon | `C` | 同上 |

- 两个臂都是 audio-off，只有 crop 不同；`legacy − canon` 隔离 hitsound/kiai/SV 的影响。
- DCR 训练数据统一用 **6,819 图 / 109,569 行** 的 `step1_1k_pairbest_nogap.parquet` 和新划分，保证 4 个组合只在内容表上不同；不混用旧的 6,828 图划分。
- benchmark 侧两张内容表预期都覆盖 1,382 行；若个别谱面编码失败，标 `Failed` 并记录，不插补。

---

## 3. 模型权重与编码环境

### 3.1 权重

- 本地已有完整 CM3P checkpoint：`.cache/cm3p_model_26cc694/`（`model.safetensors` 279 MB，含 processor/parser 代码）；
- 本地已有 CM3P 运行环境：`.cache/cm3p_venv/`（Python 3.13、torch 2.13 CPU、transformers 4.55、slider 0.8.2）；
- HPC 上也有（`data/raw/cm3p_model_26cc694` 或 `HF_HOME` 缓存）；
- 若需重新下载：用镜像站，例如
  `HF_ENDPOINT=https://hf-mirror.com huggingface-cli download OliBomby/CM3P --revision 26cc694 --local-dir .cache/cm3p_model_26cc694`；
  下载后核对 `model.safetensors` 大小/哈希，避免镜像半包。

**已验证本地 CPU 可编码**（不是推测）：

- 对 DCR 图 193127、legacy crop、no audio：
  `n_windows=7`，耗时 **6.0 s**，与既有表 cosine **0.9999907**；
- 按每图约 10–15 个 16 s 窗口估算，1,382 张约 **4–8 h/臂**；两臂可夜间跑完，HPC 只是加速选项，不是阻塞项。
- 首次运行需要把 `HF_MODULES_CACHE` / `HF_HOME` 指到工作区（默认 `~/.cache/huggingface` 在本机不可写），例如
  `HF_MODULES_CACHE=$PWD/.cache/hf_modules HF_HOME=$PWD/.cache/hf_home`。

### 3.2 编码

新增 `scripts/benchmark_encode.py`（复用 `cm3p_official_probe.py` 的 `load_model/encode_chart`）：

- 输入按 `row_id`，不是 bid；同一 bid 的 rate 变体不会互相覆盖；
- 两个臂：`legacy` / `canon`，`audio=off`，`max_length=2000`，`pad=longest`，fp32；
- 逐行 JSONL 追加，可断点续跑；
- 输出 `data/processed/benchmark_content_{legacy,canon}.parquet`：
  `row_id, beatmap_id, embedding, n_windows, source, crop, audio, cm3p_rev`。

**Gate G2（编码正确性）**：

- 每个臂重编码 32 张 DCR 图，与既有表比 cosine：`mean cos ≥ 0.999`；
- benchmark 侧两臂覆盖率 100%（编码失败的行单独列出）；
- 几何 sanity：legacy/canon 跨图 cos 约 **0.88**（audio-on 才是 0.54；本计划不涉及）。

---

## 4. 重训、保存、推理

### 4.1 为什么必须重训

`results/p4_multiseed_audio/`、`results/p4_multiseed_canon/` 只有 `signals/*.parquet` 和 JSON，**没有模型权重**；`step2_cm3p_rank.py` 只输出信号。因此必须用相同超参做受控重训并保存。

### 4.2 代码改动

给 `scripts/step2_cm3p_rank.py` 新增：

- `--fit-all`：用全部 109,569 行拟合（不依赖 train/test 划分）；
- `--save-model PATH`：保存
  - 内容标准化 `mu_C/sd_C`；
  - external 标准化 `mu_S/sd_S`（仅 M1）；
  - 线性头 `w` / MLP `state_dict`；
  - 训练集上的 `d_mu/d_sd`；
  - `content_path, content_md5, model, arm, git_commit`。

推理时**固定使用训练集的 `d_mu/d_sd`**，不能对 benchmark 重新 z-score；否则 4 个组合的尺度会随样本组成变化。

新增 `scripts/benchmark_infer.py`：加载保存的模型，对 benchmark 内容表 + signals 推理，输出每行 `d_m`。

### 4.3 拟合口径

- **主模型**：每个组合在全量 109,569 行上拟合一次（M1/M2 各自超参，模型 seed 0）；
- **稳健性**：再用 P4 的 5 个划分种子（`step1_1k_pairbest_p4_nogap_s1..5`）各训一次，报告 benchmark MAE 的种子间 mean±std；P2 5 折 chart-disjoint 作为可选加跑；
- **Gate G3**：新增推理代码在 DCR test 上复现既有信号（差异 < 1e-5）；否则先修代码，不跑 benchmark。

---

## 5. Dan 校准：5 折按谱面组交叉验证

### 5.1 为什么是“按谱面组”的 5 折

benchmark 没有玩家，无法按玩家 cluster 做 leave-one-cluster-out。这里的 LOCV 按 **chart group**：

- 有 `bid` 的行：同一 bid 的多个 rate 变体归为同一组；
- 无 `bid` 的行：用去掉 `xN` 后缀的 name 作为组；
- 用 `StratifiedGroupKFold`（按 family 和 `expected` 分箱）分 5 折，保证每折难度/键型大致平衡，且同图不跨折。

这样每个 `got` 都是 **out-of-fold** 的：拟合校准曲线时没见过该谱面的 `expected`。

### 5.2 校准模型

1. **主口径：isotonic regression**
   - 在 4 折上拟合 `expected ~ f(d_m)`，`f` 单调不减；
   - 在留出折上预测 `got`；
   - 不假设 Dan 等距；`f` 是分段线性实数映射。
2. **对照：线性校准** `expected ~ a + b·d_m`（强制等距，仅作 baseline）。
3. **敏感性**：分 family（RC/LN/course）分别拟合 isotonic；但 course 只有 34 行，结果只作描述，不据此下结论。
4. **报告**：
   - 汇总 out-of-fold 的 MAE、RMSE、bias、Spearman、Kendall、pair accuracy；
   - 每折 MAE 的 mean±std；
   - 按 pattern/family/难度段拆分；
   - 按 `chart_group` bootstrap 1,000 次给 95% CI；
   - 4 个组合用同一套折，组合间做 paired ΔMAE。
5. **in-sample 校准**：额外报“全量拟合校准”的 MAE，作为乐观下界/上界对照；主结论以 out-of-fold 为准。

**关于提交 CSV 的确定性**：out-of-fold `got` 依赖折分配，不是可部署的确定性算法。若最终提交 benchmark，需二选一：
- 提交“全量校准”的确定性 CSV，并注明页面 MAE 含 in-sample 校准，主结论看 5 折；
- 或暂不提交，只把 5 折结果写进文档。
这一点在阶段 6 再决定，不在校准阶段偷换口径。

### 5.3 无标签指标

即使不做校准，也报：
- Spearman(`d_m`, `expected`)、Kendall、pair accuracy；
- 这些指标不依赖 Dan 等距假设，用于判断“排序是否有信息”。

---

## 6. 评测与 baseline

**主表**（4 个组合 + baseline，均在同一批 1,382 行上）：

| 项 | 内容 |
|---|---|
| 常数 baseline | 全局 mean / median 的 MAE |
| 单信号 baseline | `sunny_star` 的 5 折 isotonic/线性校准；`msd_overall` 同上 |
| external-only | 只用 `[msd_overall, sunny_star]` 的 MLP/线性（不含 CM3P） |
| 已发布算法 | Sunny、Daniel、XDC1、Koishi 等 CSV 在同一批行上的 MAE |
| 4 个组合 | M1-L、M1-C、M2-L、M2-C |

**拆分**：
- RC / LN / course；
- `expected` 低（<11）、中（11–15）、高（>15）；
- 排除与 DCR whitelist 重叠的 201 张（其中 90 张也在 published 表）做敏感性；
- 全量拟合 vs P4 5 seed 的差异。

**Gate G4**：校准后 MAE 必须优于同子集常数 baseline；否则结论是“该表示在 Dan 尺度上没有可用信号”，不强行提交。

---

## 7. 代码与产物

| 文件 | 改动 | 目的 |
|---|---|---|
| `scripts/step2_cm3p_rank.py` | 新增 `--fit-all`、`--save-model` | 全量拟合 + 保存权重/标准化统计 |
| `scripts/benchmark_manifest.py` | 新增 | 契约 manifest、rate/group 解析、匹配审计 |
| `scripts/benchmark_signals.py` | 新增 | Sunny（Node 无头）+ MinaCalc 505，带 DCR 对账 |
| `scripts/benchmark_encode.py` | 新增 | 按 row_id 编码 legacy/canon，JSONL 断点续跑 |
| `scripts/benchmark_infer.py` | 新增 | 加载保存的模型，对任意内容表推理 |
| `scripts/benchmark_calibrate_eval.py` | 新增 | 分组 5 折、isotonic/线性校准、MAE/RMSE/相关、bootstrap、CSV 契约 |
| `scripts/test_benchmark_contract.py` | 新增 | 8 列格式、错误行、rate 解析、分组折的合成测试 |
| `docs/benchmark_vsrg.md` | 新增 | 实验报告（结果出来后填） |

产物：

| 路径 | 内容 |
|---|---|
| `results/benchmark/pred/{run}.parquet` | 4 个组合的 `row_id, d_m, fold` |
| `results/benchmark/eval/{run}.json` | 校准曲线、OOF 指标、分 pattern 表 |
| `results/benchmark/summary.md` / `.csv` | 4 组合 + baselines + 消融 + CI |
| `results/benchmark/candidates/{Algorithm}.csv` | 可选 8 列候选提交文件 |
| `docs/benchmark_vsrg.md` | 方法、契约、覆盖率、结果、限制 |

提交命名（4 个）：`DCR-CM3P-XMLP-Legacy`、`DCR-CM3P-XMLP-Canon`、`DCR-CM3P-MLP-Legacy`、`DCR-CM3P-MLP-Canon`。
默认不 push、不开 PR；结果出来后由你决定提交 4 个、2 个还是 0 个。

---

## 8. 已核对的关键事实

| 事实 | 证据 |
|---|---|
| benchmark 1,382 行：RC 1,246 / LN 102 / course 34 | `samples/data.csv` |
| 1,146 行有 bid / 1,116 唯一 bid；822 行有 `xN` 后缀 | 同上；本计划不再依赖音频 |
| 201 张与 DCR whitelist 重叠，90 张与 published 表重叠 | `manifests/whitelist_1k.csv` / `chart_content_cm3p.parquet` |
| 本地 CM3P checkpoint 完整（279 MB） | `.cache/cm3p_model_26cc694/model.safetensors` |
| 本地 CPU 编码可复现 | 193127：7 窗 / 6.0 s / cosine 0.9999907 |
| Sunny 无头值与 DCR 表逐位一致 | 193127：2.101017423142987 |
| MinaCalc 505 与 DCR 表逐位一致 | 193127：10.512499 vs 10.5124988556 |
| 常数预测 MAE 1.55 | 全局 median |

---

## 9. 风险与应对

| 风险 | 触发条件 | 应对 |
|---|---|---|
| 校准过拟合 | isotonic 在 OOF 上明显差于线性 | 主表并列 isotonic/线性；不调超参；分 family 只作描述 |
| Dan 不等距 | 线性校准系统性偏差 | isotonic 主口径；报残差/分位数图 |
| CM3P 对 course/LN OOD | 这些 family 的 MAE/相关明显更差 | 分 family 报告；主结论限定 RC；不对外推 |
| 4 个组合差异小于噪声 | paired ΔMAE 的 CI 跨 0 | 结论写“不可区分”，不排名 |
| 编码太慢 | 单臂 > 12 h | 切 HPC（已有权重/venv）；或分片夜间跑 |
| benchmark 更新 | commit 变化 | 固定 `e15f89b`；不自动 pull；报告注明快照 |
| 201 张与 DCR 重叠 | 结果异常好 | 主表 + 排除重叠的敏感性表并列 |

---

## 10. 时间线

| 天 | 工作 | 产出 |
|---|---|---|
| D1 | 冻结、manifest、signals、编码 32 张对账 | G0/G1/G2 |
| D2 | 两臂 1,382 张编码（本地 CPU 夜间 / HPC） | 两张内容表 |
| D3 | 4 组合全量训练 + P4 5 seed 稳健性 | 模型、`d_m` |
| D4 | 5 折校准、评测、报告 | summary、`docs/benchmark_vsrg.md` |

---

## 11. 需要你确认

1. **LOCV 分组**：我按 **chart group**（同 bid 的 rate 变体同组；无 bid 用去后缀 name）做 5 折。benchmark 没有玩家，所以无法按玩家 cluster 做 LOCV；如果你指的是按玩家/按段位分层，请指出。
2. **校准主口径**：我计划 **isotonic 为主、线性为对照**，两者都报。是否同意？
3. **提交范围**：先出 5 折结果和候选 CSV，不 push/PR；提交 4 个、2 个还是 0 个，结果出来再定。

默认按 **1（chart group）/ 2（isotonic 主 + 线性对照）/ 3（先本地）** 执行。
