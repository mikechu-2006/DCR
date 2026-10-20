# Step 2 — 用 MinaCalc 前向模型拟合玩家 6 维技能

> 状态：**设计定稿，待实现**。本文自包含，压缩上下文后按本文执行即可。
> 所有数字均为本机实测（1k 数据 + 本地 vendored MinaCalc 505）。

---

## 0. TL;DR

用 Etterna MinaCalc 的逐区间/逐事件难度导出，构造一个**可微前向模型**
score(chart, player_6d) -> accuracy，然后在 (player, chart) -> loss 数据上
**块坐标下降**：

1. 固定预测器参数 θ，拟合每个玩家的 6 维技能；
2. 固定玩家技能，拟合 θ；
3. 交替 3–5 轮。

目标函数：r = ln(1 − pred) − ln(1 − obs)，RMSE 在该空间最小。
主用 1k 数据集（891 玩家 / ≈67 万 train play / 20,655 谱面，全部 NM 1.0x）。

---

## 1. 目标与约束

- **目标**：得到每个玩家的 6 维技能向量（stream / jumpstream / handstream /
  jackspeed / chordjack / technical）+ 一组可解释的预测器参数，使预测尽可能接近观测。
- **不是**复刻 Etterna 的 MSD：MinaCalc 本身没有"给定玩家 → 预测成绩"的函数，
  它只有"给定目标分反解技能"。本模型是把它的难度导出后自己拼出来的正向模型。
- 约束：
  * C++ 侧参数（pmods / basescalers / finalscaler）**冻结**（改动需重导出全部谱面）；
  * 玩家耐力系数 e **冻结在 1.0**（与全局 prop 存在规范自由度）；
  * 只用 step1 的 train 行拟合，test 行只做评估。

---

## 2. 已完成资产

### 2.1 本地 MinaCalc（C++）

| 项 | 路径 | 说明 |
|---|---|---|
| 源码 | external/minacalc-standalone/ | Etterna MinaCalc standalone 副本，**calc version 505** |
| 共享库 | external/minacalc-standalone/build/libminacalc.so | C API：create_calc / calc_msd / calc_ssr / calc_version |
| 导出器 | external/minacalc-standalone/build/export_itv | 本项目新增，源码 minacalc/cpp/export_itv.cpp |
| 构建脚本 | minacalc/cpp/build.sh | g++ -std=c++17 -O2 -I <standalone> export_itv.cpp MinaCalc.cpp -o build/export_itv |

导出器用法：

    ./external/minacalc-standalone/build/export_itv <map.osu> <out.bin> [rate=1.0] [goal=0.93]

**二进制格式 v2（小端）**：

    u32 magic = 0x5449434D ("MCIT")
    u32 version = 2
    u32 keycount            # 只支持 4
    u32 numitv              # 0.5s 区间数
    f32 rate
    f32 goal                # 仅溯源，diffs 与 goal 无关
    f32 diffs[2][6][numitv] # 手 × {stream,js,hs,jackspeed,cj,tech} × 区间
    f32 pts[2][numitv]      # 每手每区间点数（= 2 × 音符数）
    u32 n_jack[2]
    per hand: i32 itv[n] + f32 y[n]   # jack 事件（逐行），itv = int((t+0.0005)/0.5)

### 2.2 Python 包 minacalc/

| 文件 | 内容 |
|---|---|
| core.py | Player / ChartData / score / aggregate_rows / loss_fraction / stamina_mods / jack_stam / jack_event_loss / jack_interval_loss / required_skill |
| io.py | export_itv / load_itv / load_chart_itv（npz 缓存，键 = 内容 hash + rate + goal + 版本） |
| prep.py | .osu 解析、**近似**难度模型（仅当导出器缺失时兜底）、load_chart(backend="auto") |
| cpp/export_itv.cpp | 见上 |
| __main__.py | 自检：python3 -m minacalc [map.osu] |

核心签名：

    Player(stream, js, hs, jackspeed, cj, tech, endurance=1.0)   # endurance=None 关闭疲劳

    ChartData(name, rate, interval, pts, diffs, source, meta, jack)
    #   pts   (2, n)
    #   diffs (6, 2, n)
    #   jack  [(itv int64[], y float64[]), ...] 长度 2，可为 None

    score(chart, player, *,
          combine="aggregate",          # aggregate | max | sum
          agg_delta=0.25, agg_mult=1.11, agg_beta=5.0, agg_iters=12,
          order=3,                      # 3 = 三阶泰勒；None = 精确 pow
          p_exp=None,                   # 默认 P_EXP（505 口径）
          use_jack=True,                # 第 4 维用事件模型 12·erf
          tech_jack_extra=True,         # sqrt(min(10%·Max, 0.85·jackloss(0.75x)))
          detail=False) -> float | dict

### 2.3 关键实现事实（**不要重查/重做**）

1. **505 与 527 的差异**：CJ 失分幂次 505 是 **1.7**，527 才改成 1.8；505 只支持 4K；
   pmod 集合更少。本项目以 **505** 为准（P_EXP = [1.7]*5 + [2.0]）。
2. **JackSpeed 的区间难度行是死数据**：base_adj_diff[hand][JackSpeed] 恒等于
   NPSBase × 1.01（pmod 列表为空），Chisel 只读事件表 jack_diff。
3. **jack 难度 = 逐行事件**：y = ms_to_scaled_nps(最快同列间隔) × 1.01，损失
   12·erf(0.04(y−x))，无区间封顶，可把总分打成负数。
4. **calc_ssr 的返回值含 grindscaler**（本次测试 0.9746），calc_msd / ssr=false 不含。
   对照时必须除以它，或用 ssr=false 的直出值。
5. **fastpow 不准**：u^p 相对误差最大 9%；工作点 u≈0.95 处**失分**误差 −22%；
   u=0.99 时 fastpow>1（失分为负）。要干净模型用 order=None。
   但现在 p 已经解放成三次式系数（§4.3-B）：三次式是**模型定义**，不是对 pow 的近似，
   所以这一条只作为"为什么不能用 fastpow"的历史依据。
6. **aggregate 在点数尺度会压缩**：6 维同时给 7% 预算时，按区间合并后是
   **12.78%**（max 16.09%，sum 42.6%），因为各轴热点不重叠。这不是 bug。
7. **score 可以 ≤ 0**（aggregate ×1.11 + jack 无封顶）。因为目标是 ln(1−score)，
   1−score > 1 时 log 有定义，**不需要 clamp**。
8. **chart_id = beatmap_id*4 + rate** 只存在于**旧 v2 表**；当前 step0 是 NM-only
   （MOD_WHITELIST = "CL"），没有 rate 列。

### 2.4 已验证的数字（回归测试用）

测试图：/tmp/mc_test/test.osu（4K、150bpm 16 分 + 每 8 音一 jump、120 区间、1350 分）

C++ 直出（ssr=false, goal=0.93）：

    overall 13.888 | stream 4.180 | jumpstream 13.888 | handstream 2.580
    stamina 13.067 | jackspeed 9.220 | chordjack 11.060 | technical 5.860

我们的模型在这些 x 上的单轴 score（combine="sum", order=None）：

| 轴 | 参考 x | 我们的 score |
|---|---|---|
| stream | 4.180 | 0.9180 |
| jumpstream | 13.888 | 0.9372（参考值是 stamina 精修过的，偏高正常） |
| handstream | 2.580 | 0.9319（同上） |
| jackspeed | 9.220 | **0.9150** |
| chordjack | 11.060 | 0.9242 |
| technical | 5.860 | **0.9306** |

全 6 维、玩家取全部参考值：**score = 0.8722**（见 2.3 第 6 条）。

测试图可随时重建（/tmp/mc_test 不是长期目录）：

    python3 - <<'PY'
    lines = ["osu file format v14", "", "[General]", "AudioFilename: a.mp3", "Mode: 3",
             "", "[Difficulty]", "CircleSize:4", "OverallDifficulty:8", "", "[HitObjects]"]
    t = 1000
    for i in range(600):
        c = i % 4
        lines.append(f"{64 + 128*c},192,{t},1,0,0:0:0:0:")
        if i % 8 == 7:
            c2 = (i + 2) % 4
            lines.append(f"{64 + 128*c2},192,{t},1,0,0:0:0:0:")
        t += 100
    import pathlib; pathlib.Path("/tmp/mc_test").mkdir(exist_ok=True)
    pathlib.Path("/tmp/mc_test/test.osu").write_text("\n".join(lines))
    PY

---

## 3. 数据

### 3.1 输入表

data/processed/step0_1k.csv（70 MB）：

    player_id, beatmap_id, timestamp, playcount_cur, loss

- loss = ln(1 − acc)，负值；中位 −5.23（acc 99.46%），1% 分位 −12.47（acc≈1），
  最大 −0.267（acc 23%）。
- **NM only**（DT/HT/HD/FL 已在 step0 被 whitelist 剔除）。

### 3.2 step1 规则（新版，scripts/step1_clean.py）

| 规则 | 内容 | 结果 |
|---|---|---|
| 0 | 玩家在全表记录数 < 200 → 整人剔除（**先于划分**） | 990 → **891** 人 |
| 划分 | pair-random，train_frac=0.70，seed=20260928 | — |
| 1 | train 中 acc <= 0.90 删除（即 loss >= ln 0.1） | 1.389M → 1.367M |
| 2 | train 中每人按时间排序删最早 drop_frac=0.30 | → **956,610** 行 |
| — | train ≈ 70% | **≈ 670,000 行** |

每人 train 中位 ≈560 条（max ≈8000），谱面 20,655 个。

### 3.3 谱面张量

- 每个 beatmap_id 需要一份 ChartData（diffs / pts / jack）；
- 覆盖要求：**20,655 / 20,655**；
- 存储：20,655 × 6 × 2 × ~500 × 4B ≈ **0.5 GB**，可常驻显存。

---

## 4. 模型

### 4.1 前向

    对每个 0.5s 区间 i、每只手 h：
       d[ss,i,h] = diffs[ss,h,i]                    # C++ 导出
       d[ss,i,h] *= stamina_mods(d, x_ss, e=1.0)    # 逐区间疲劳（e 冻结 1）
       t = clip(1 − x_ss/d[ss,i,h], 0, 1)
       loss_ss = pts[h,i] · (c1·t + c2·t² + c3·t³)   # 三次式，c 见 4.3-B
       loss_js = jack_interval_loss(itv, y, x_js)              # 事件模型，可超过 pts
       L_i = aggregate(loss_{0..5,i}, δ=0.25, β=5) × mult      # 每区间合并
    score = 1 − (Σ_i L_i + tech_extra) / MaxPoints
    tech_extra = Σ_hand sqrt(min(0.1·MaxPoints, 0.85·Σ 12·erf(0.04(y − 0.75·x_tech))))

三次式系数只有两组：**tech 独占** c^2（初值 p=2），
**stream / js / hs / jackspeed / cj** 共享 c^17（初值 p=1.7），见 §4.3-B。

### 4.2 目标函数

    r = ln(1 − pred) − ln(1 − obs)          # 都是负数
    L = Σ w·r² + λ Σ_p ‖x_p − μ‖² + λθ ‖θ − θ0‖²

- **不需要 clamp**（见 2.3-7）；
- ln(1−s) 的灵敏度 = 1/(1−s)：98% 附近是 50% 附近的 50 倍，**高分样本主导**。
  若残差尾部过重，可对 |r| > 5 做 Huber，但默认不做。

### 4.3 参数清单（全部）

**A. 玩家侧**（阶段 A，891 × 6）

| 符号 | 位置 | 初值 | 状态 |
|---|---|---|---|
| x_p,k | 6 维技能 | 全局均值 20 | 拟合 |
| e | 耐力系数 | 1.0 | 冻结（v2 才放开，须先验 mean=1） |

**B. 单点失分三次式**（阶段 B-1，共 6 个）

| 符号 | 作用维 | 初值 | 状态 |
|---|---|---|---|
| c^2 = (c1,c2,c3) | **tech 独占** | (2, −1, 0) | 拟合 |
| c^17 = (c1,c2,c3) | stream / js / hs / jackspeed / cj **共享** | (1.7, −0.595, −0.0595) | 拟合 |

初值与 MinaCalc 505 完全一致（MinaCalc.cpp:430-435）：

    1 - u^p,  p = 2.0 (tech) / 1.7 (其余全部，含 stream)

c^2 = (2, −1, 0)：c3 = 2·1·0/6 = 0，退化成**二次式**（精确等于 p=2）。
c^17 = (1.7, −0.595, −0.0595)：p=1.7 的三阶展开。
**初始模型与 505 逐点相同，只在拟合中偏离**——这是"用三次式代替 p"的验收标准。
注意 **jackspeed 的 c 是死数据**（§2.3-2：losses[3] 被事件模型整行覆盖）。

**C. 区间合并 aggregate**（只 δ 进 B-1，其余冻结）

| 符号 | 初值 | 状态 |
|---|---|---|
| δ（agg_delta） | 0.25 | 拟合 |
| mult（agg_mult） | 1.11 | **冻结** |
| β（agg_beta） | 5.0 | 冻结 |
| 迭代次数 / budget 指数 | 12 / 0.1 | 数值 / 冻结 |

**D. 全局缩放**（冻结，1 个）

| 符号 | 初值 | 状态 |
|---|---|---|
| loss_scale | 1.0 | **冻结**（恒等于 1，公式里可省略） |

**E. 区间疲劳 stamina_mods**（阶段 B-2，6 个）

    ceil 1.075234 | mag 243 | fscale 500 | prop 0.69424 | super_ceil 1.09
    起始 floor = mod = 0.95

    # 6 个全部拟合（阶段 B-2，box ±20%），**不冻结**。

**F. jack 疲劳 jack_stam**（**冻结**，7 个）

    ceil 1.05234 | mag 23 | fscale 750 | prop 0.49424 | super_ceil 1.01
    起始 floor 0.95 / mod 1.0

    # 冻结理由：mod ∈ [0.95, 1.01]，整体幅度只有 +1% / −5%（见 jack_stam 说明），
    # 7 个常数在这个幅度下不可辨识；保持 MinaCalc 原值。

**G. jack 失分形式**（阶段 B-1，2 个）

    12（幅度）| 0.04（斜率）          # loss = 12·erf(0.04·(y − x))

    # 两个都拟合，**不冻结**。但注意：低难度区 erf(0.04Δ) ≈ 0.04Δ，
    # 12 与 0.04 只以乘积 0.48 出现 → 两者近似共线，只有 Δ 进饱和区才分开。
    # 处理：给 12 强 ridge（λθ ×5），报告时给乘积；技术额外项共用这两个常数。
    # 实现时把 core.py:jack_event_loss 里的字面量 12.0 / 0.04 提成参数。

**H. tech 额外项**（冻结，3 个）

    0.1·MaxPoints 上限 | 0.85 系数 | 0.75·x_tech

**计数**：玩家侧 891×6 = 5346；
阶段 B-1 = 6(c) + 2(jack 幅度/斜率) + 1(δ) = **9**；
阶段 B-2 = **6**（区间疲劳）。全局可拟合标量共 **15**（jack 疲劳 7 常数仍冻结）。

mult 与 loss_scale 都冻结成常数后，**模型的绝对尺度完全由 c1 承担**——c1 是
对 loss 的线性缩放，等价于原来的全局缩放。所以"全参考玩家 = 0.8722 vs 0.93"
这个偏差不再靠 loss_scale 修，而是拟合中**上移 c1、下移 x** 自己找到位置。
后果：报告必须同时给 c 和 mean(x)，单看 x 不可解释；且 c1 与 x 的尺度退化
（§4.4）比之前更值得监控。

### 4.4 可辨识性

- **全局尺度**：finalscaler 冻结 → 尺度锚定。
- **每维尺度**：basescalers 冻结 → 不会与玩家技能共线。
- **e vs prop**：e 冻结 1.0，prop 承担总体疲劳强度；v2 放开 e 时须先验 mean(e)=1。
- **未激活维**：某玩家从未打过激活某维的谱面时，该维梯度≈0 → **活动集**冻结在先验。
  判据：Σ_plays max(0, d_k/x_k − 1) < ε（建议 ε=0.05）。
- **c1 与 x 的尺度近似退化**：一阶项是 pts·c1·(1 − x/d)，把某维 x 乘 λ、c1 除 λ，
  失分只差 O(t²)。逐维放开 c1 会多出 5 条近平坦方向；本题只给 stream 独立一组，
  其余五维共享 c^R，等于把 5 条方向压成 1 条。监控 c1 的漂移，必要时固定 mean(x_k)。

---

## 5. 算法

**块坐标下降**，两相都作用在同一个可微图上：

    X = Embedding(891, 6).init(均值)
    θ = θ0
    for outer in 1..3..5:
        # Phase A: 只更新 X
        for step in 1..S_a:
            采样/全量 play → forward → r² + λ‖X−μ‖² → backward → 只 step X
        # Phase B: 只更新 θ
        for step in 1..S_b:
            全量/子采样 → forward → r² → backward → 只 step θ
        # 第 1 轮 Phase B 只动 c(6) + jack 形式(2) + agg_delta
        # 第 2 轮起再动疲劳常数
        warm start：X 跨轮保留

**关键架构**：把 X 做成 **embedding 表**，一次前向覆盖全部（或分块的）play，
autograd 一次性给出所有玩家梯度。不要"逐玩家单独求解"——那是 891 次串行前向。

- 分块：5 万 play/块（≈1.2 GB 显存），梯度累加；
- 优化器：Adam（lr_X 1e-2、lr_θ 1e-3），或 LBFGS；
- 边界：X ∈ [0, 60]（clamp 或 sigmoid 重参数化）；
- 收敛判据：train RMSE 变化 < 1e-4，且 X 均值/方差漂移 < 1e-3。

---

## 6. 实现计划

### P0 — 数据与张量（半天）

1. scripts/step2_fit_skills.py --stage data：
   - 读 step0_1k.csv + step1 的 split；
   - 过滤 split == "train"；
   - 每个 beatmap_id → .osu（data/osu_cache/{id}.osu）→ export_itv → .minacalc_cache/*.npz；
   - 输出 data/processed/fit_1k_plays.parquet：
     player_idx, chart_idx, loss, n_itv, chart_path。
   - **验收**：行数 ≈67 万；chart_idx 覆盖 20,655；无缺失。
2. 打包张量：data/processed/fit_1k_charts.pt
   （diffs 定长 padding + n_itv + pts + jack 事件的扁平化 itv/y + offset）。

### P1 — torch 前向（1–2 天）

minacalc/torch_model.py：

    class Forward(torch.nn.Module):
        def __init__(self, charts, theta): ...
        def forward(self, X, player_idx, chart_idx):
            #  gather  -> (N,6,2,n_max)
            #  stamina scan（沿 n 循环，向量化 N）
            #  loss_fraction（泰勒或精确）
            #  jack: index_add_ 到 (N, n_max)
            #  aggregate: 8 轮 Newton，r >= 0，batched
            #  sum -> score (N,)

- 数值对齐验收：在 /tmp/mc_test/test.itv 上，与 minacalc.core.score 逐 play
  差 < 1e-4；且 2.4 节的 6 个单轴值复现（误差 <0.005）。
- 性能验收：67 万 play 一次前向 ≤ 20 s（A800）。

### P2 — 玩家步（1 天）

- X embedding + Adam/LBFGS，λ 从 1e-3 起，扫 [1e-4, 1e-3, 1e-2, 1e-1]；
- 活动集：按 4.4 冻结未激活维；
- 输出 results/fit_1k/X_iter{k}.npy。

### P3 — θ 步（1 天）

- Phase B-1：c（6 个）+ jack 幅度/斜率（2 个）+ agg_delta，共 9 个；
- Phase B-2：疲劳常数（box ±20%，坐标搜索更稳）；
- 记录 θ 轨迹：c1 上移是全局尺度吸收的主要通道，会与 X 均值反向共移，
  两者必须一起报告。

### P4 — 评估（半天）

- test 上 RMSE（log 空间）+ accuracy 空间 RMSE；
- 与 bias / msd / msd_add 在**同一行集合、同一 split** 上对比；
- 冷启动：每人只用 1 条 play 拟合，预测其余；
- 诊断：X vs 谱面 MSD 的相关；活动维数分布；残差 vs 谱面长度。

### P5 — 可选

- 层次先验 x_p ~ N(μ, Σ)；
- 放开 e（先验 mean=1）。

---

## 7. 风险

| 风险 | 缓解 |
|---|---|
| ln(1−s) 放大高分样本 | 检查残差分布；必要时 Huber |
| 聚合在点数尺度过度惩罚（0.8722） | 由 c1 吸收（loss_scale 已冻结为 1） |
| 未激活维 | 活动集冻结 |
| θ 与 X 共移（c1 ↔ X 均值） | 分阶段更新；两者一起报告 |
| 三次式偏离 505 的失分形状 | 初值严格对齐（§4.3-B），拟合后报告 Δc |
| 12 与 0.04 近似共线 | 强 ridge 盯住 12；报告乘积 12·0.04 |
| 导出覆盖率 | P0 强制 100% |
| 训练/评估不可比 | 固定同一行集合 + split，报告写清 |

---

## 8. 环境与命令

    # 编译导出器
    bash minacalc/cpp/build.sh

    # 单图导出
    ./external/minacalc-standalone/build/export_itv map.osu out.bin 1.0 0.93

    # 自检
    python3 -m minacalc                     # 合成谱
    python3 -m minacalc data/osu_cache/123.osu   # 真实谱（走 C++ 后端）

    # 数据规模核对
    python3 -c "import pandas as pd; df=pd.read_csv('data/processed/step0_1k.csv'); print(len(df), df.player_id.nunique(), df.beatmap_id.nunique())"

- Python：anaconda3（3.13），已装 numpy/pandas；需另装 torch（GPU）。
- 集群：单卡 A800；显存需求 <5 GB。
- 导出的 npz 缓存在 .minacalc_cache/（20,655 图 ≈ 0.5 GB），**不要提交**；
  实现时确认已在 .gitignore 里。

---

## 9. 待定决策（实现前确认）

1. **λ 的量级**：数据厚（≈560 play/玩家 vs 6 参数），建议从 1e-3 起扫。
2. ~~**order**~~：**已定**——不用精确 pow，三次式就是模型（§4.3-B）。
   ~~训练用精确 pow（order=None）还是三阶泰勒（order=3）？~~
   建议精确——泰勒在点数尺度有 ≤1.4% 系统偏差，会与 loss_scale 混淆。
3. ~~**agg_mult 是否可辨识**~~：**已定**——mult 冻结 1.11，loss_scale 冻结 1.0，
   绝对尺度交给 c1（§4.3 计数段）。
4. **step1 产物的确切路径**（split 列所在文件），实现时先 ls experiments/* 确认。
5. **是否需要 CPU 降级路径**（无 GPU 时）。

---

## 10. 已否决/已澄清（避免重复讨论）

- ~~用 MSD 做玩家初值~~：可选加速，不是必需；X 从全局均值起步即可。
- ~~rate 三档（NM/DT/HT）~~：当前 step0 是 NM-only，无 rate 维度。
- ~~n_plays < 7 导致欠定~~：清洗后每人 ≥200，中位 560。
- ~~冻结疲劳常数~~：改为进 θ（全局参数，数据足够识别），第 2 轮才动。
- ~~clamp score 到正数~~：目标是 ln(1−score)，不需要。
- ~~把 aggregate 换回二分~~：现在是 softplus + Newton，可微。
- ~~jack 疲劳常数进 θ~~：冻结在 MinaCalc 原值（mod 只在 [0.95, 1.01] 内，7 个常数辨识不了）。

---

## 11. 实现状态（2026-10-18）

### 已完成

| 阶段 | 产物 | 验收 |
|---|---|---|
| P0a | `scripts/step2_export_charts.py` → `.minacalc_cache/` 20,655 张谱，0 失败，112 s | 100% 覆盖 |
| P0b | `minacalc/data.py` → `data/processed/charts_1k.npz`（6.13M 区间 / 27M jack 事件 / 560 MB） | 扁平存储，peak itv 8671 |
| P1 | `minacalc/torch_model.py`（三次式 + stamina + jack + 隐函数 aggregate） | 与 `core.score` 逐播放对比，maxdiff **1.25e-4** |
| P1 梯度 | aggregate 隐函数 backward | 有限差分对比：6 个分量里 5 个 <1%，主分量 5%（FD 截断噪声） |
| P2/P3 | `scripts/step2_fit_skills.py`，仓库根可跑 | 端到端跑通，B1/B2 的 L-BFGS 都令目标下降 |
| 调度 | `scripts/slurm/step2_fit.slurm` | 待集群恢复后提交 |

### 实现中改掉的三个设定

1. **aggregate 求解器：Newton+夹逼 → 纯二分（48 轮）**。
   F(r) = Σw_j(r) − 2^(0.1r) 对 r 严格单调递减，二分在 [0, max(v)+10] 上必然收敛。
   实测原版"Newton+夹逼"里 **70% 的步被判 bad 回退到二分**（400 列 × 30 轮 8420 次），
   等于一直在用二分干活，还多一层判断。精度：12 轮偏差 0.098 / 0.045，30 轮 3.5e-7，
   二分 48 轮区间宽 3.6e-14。**梯度不受影响**——隐函数定理只用收敛点处的 ∂F/∂r。
2. **`stamina_mods_t` 返回倍率**（不是乘好的难度）。第一版返回 `d*mod` 又被调用方乘了一次，
   `u` 被放大 20 倍，分数从 0.99 掉到 0.27。
3. **jack 事件偏移**：两手事件拼在同一个数组里，手 1 的 `jack_off` 必须整体平移 `J0`，
   否则手 1 读到手 0 的事件（区间下标能到 409/49）。

### 已知偏差

- torch float32 与 numpy float64 的分数差 ≤1.5e-4。**不是求解器造成的**：两边都二分到
  3.6e-14，仍差这个量级 → 来自 float32 的 v 本身（主要是 jack 事件那条链）。
  对拟合无影响（观测噪声在 log 空间约 1）。
- 只保留 GPU 路径；本地 CPU 仅用于冒烟（`--device cpu`）。

### 集群阻塞（2026-10-18）

`hpc2login` 上 `/etc/slurm` 与 `/etc/slurm-llnl/slurm.conf` 都不存在，`sbatch/squeue` 直接失败：

    sbatch: error: s_p_parse_file: unable to status file /etc/slurm-llnl/slurm.conf

用 `SLURM_CONF=$HOME/.slurm/slurm.conf`（从 `/tmp/slurm.conf` 复制）能让 `sbatch --version`
返回 `slurm-wlm 19.05.5`，但提交时控制器连不上：

    sbatch: error: Batch job submission failed: Zero Bytes were transmitted or received

代码与数据已同步到 `/hpc2hdd/home/aiaa3111_l01_008/DCR`（`minacalc/`、`scripts/`、
`data/processed/charts_1k.npz`、新 `step1_1k.parquet`）。集群恢复后：

    cd /hpc2hdd/home/aiaa3111_l01_008/DCR
    sbatch scripts/slurm/step2_fit.slurm
