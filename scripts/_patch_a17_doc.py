from pathlib import Path

p = Path('docs/d2v_attr_results.md')
t = p.read_text()

# ---- 1) §0 conclusion: add the within-group split finding after the D bullet
anchor = "* **C3/C4**：属性通道确实被编码（SV/kiai/BPM 可解码性 0.07–0.16 → 0.67–0.88），"
assert anchor in t
add0 = """* **组内拆分（a20d − SV/kiai，17ch，2026-11-25 补跑）**：删掉 SV 与 kiai 后，linear 回到
  8ch 水平（−0.001，噪声内），+ext 只剩 −0.005；相对 a20d 变差 **+0.0233 / +0.0238**。
  即 **a20d 的难度序增益几乎全部来自 SV+kiai**，tempo/beat/bar 单独几乎没有可检测效应。
  同时 a17d 的 masked loss 反而最低（0.0263 < a20d 0.0352 < a0d 0.0588）：自监督 loss
  不能当作“难度有用性”的代理。
"""
t = t.replace(anchor, add0 + anchor, 1)

# ---- 2) insert §3.4 after the shuffle §3.3 block (before "## 4. C3/C4 诊断")
sec4 = "## 4. C3/C4 诊断（泄漏与可解码性）"
assert sec4 in t
new34 = """### 3.4 组内拆分：a20d − SV/kiai（a17d，17 通道）

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

1. **SV+kiai 承担了 a20d 几乎全部难度序增益**。删掉它们后：相对 a0d 的 linear 差回到
   -0.001（噪声内）、+ext 只剩 -0.005；相对 a20d 则变差 **+0.0233（linear）/ +0.0238（+ext）**，
   约为 5-seed MDE（0.005）的 4.7 倍。tempo/beat/bar 这一组单独几乎检测不到难度增益。
2. **自监督 loss 与“难度有用性”脱钩**：a17d 的 masked loss 最低（0.0263），但 rank 最差；
   说明 SV/kiai 给 masked-content 预测增加的是“难预测但与难度相关”的信号——低 loss 本身
   不能作为通道有效的证据。
3. **风格泄漏视角**（呼应 §4 与 §6）：增益恰好集中在风险最高的两组（SV gimmick、kiai =
   song section）。C4 显示删掉后 kiai 可解码性从 0.675 掉到 0.033、rank 增益同时消失；
   而 C3 mapper 指纹只小幅下降（0.1532 → 0.1447，仍高于 a0d 的 0.1340）。
   因此**单 seed + set-disjoint 下仍无法区分“SV/kiai 携带读谱/视觉难度”与“mapper/歌曲风格”**；
   要拆分需要 mapper-disjoint 复核或 shuffle/随机化对照（本轮未跑）。
4. **局限**：单 seed、无 CI；a17d 与 a20d 是不同训练 run（同 config/seed/数据，
   a20d_reprobe 证明 probe 可复现）；差值 0.023 远大于预期的 run-to-run 噪声，但没有正式 CI。

"""
t = t.replace(sec4, new34 + sec4, 1)

# ---- 3) §5: add ops item 8 (new account bootstrap)
anchor5 = "## 6. 能说什么 / 不能说什么"
assert anchor5 in t
item8 = """8. **新 HPC 账号（aiaa3111_l01_066）是空 home**：旧账号 home 不可读，只能从共享
   Python 3.11 建 venv、pip 安装 torch 2.5.1(CUDA) 等，并 rsync 1.2 GB osu_cache +
   小表；因此本轮 a17d 在**另一账号**训练/抽取/probe。为消除环境差异，a20d 在同一账号
   重跑（结果与旧账号逐位一致），配对有效。

"""
t = t.replace(anchor5, item8 + anchor5, 1)

p.write_text(t)
print("results doc updated with §3.4")

# ---- 4) plan §0.6
pp = Path('docs/plans/2026-11-24_d2v_structured_inputs_plan.md')
pt = pp.read_text()
anchorp = "* **C3/C4**：属性通道确实被编码（SV/kiai/BPM 可解码性 0.07–0.16 → 0.67–0.88），"
assert anchorp in pt
addp = """* **组内拆分（2026-11-25 补跑）**：a20d − SV/kiai（a17d，17ch）显示 SV+kiai 承担几乎全部
  增益（相对 a20d +0.0233/+0.0238），tempo/beat/bar 单独近零；且 a17d 的 masked loss 更低
  → 自监督 loss 不能作为难度有用性代理。单 seed，无 CI。
"""
pt = pt.replace(anchorp, addp + anchorp, 1)
pp.write_text(pt)
print("plan updated")
