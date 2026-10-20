# minacalc/

MinaCalc 式打分的**逐区间可微实现**：把一张 4K 谱面切成固定长度的区间，每区间给出
6 个维度的难度，再用「玩家技能 vs 难度」的失分函数算出得分。它是本仓库保留的
MinaCalc 衍生实现；原先调用它的管线脚本、配置与设计文档已归档到 `legacy/`。

## 文件

| 文件 | 作用 |
|---|---|
| `core.py` | 打分公式：`Player` / `ChartData` / `score` / 聚合 / 耐力与 jack 修正 |
| `prep.py` | `.osu` → `ChartData`；npz 缓存；`load_chart(backend="itv"/"approx"/"auto")` |
| `io.py` | C++ 导出器输出格式的读写（`export_itv` / `load_itv`） |
| `data.py` | 成绩表 → 变长扁平谱面张量与训练 batch |
| `torch_model.py` | torch 前向/反向（聚合用隐函数反传），与 `core.score` 数值一致 |
| `player_step.py` | 逐个玩家精确求解 6 维技能（Levenberg-Marquardt） |
| `cpp/export_itv.cpp` | 调 Etterna MinaCalc 导出逐区间 `base_adj_diff` |
| `cpp/build.sh` | 编译导出器（需要 `external/minacalc-standalone`） |
| `__main__.py` | 自检 + 最小示例 |

## 快速检查

~~~bash
python -m minacalc            # 自检：聚合、三次式、耐力、单调性、解析对照
python -m minacalc chart.osu  # 另外读一张谱（需要先编译 C++ 导出器）
~~~

## 在代码里用

~~~python
from minacalc import Player, load_chart, score

ch = load_chart("chart.osu", backend="auto")   # 有 C++ 导出器就用真实 MSD，否则用简化模型
p = Player(stream=20, js=19, hs=18, jackspeed=17, cj=18, tech=19, endurance=None)
print(score(ch, p))                            # 0~1 的得分
~~~

单点失分是三次式 `c1 t + c2 t^2 + c3 t^3`（`t = clip(1 - x/d, 0, 1)`），
`c` 是自由参数；聚合把每区间 6 维失分合并成一个扣分。版本口径（505 / 527）
见 `core.P_EXP` / `core.P_EXP_527`。

## C++ 区间导出

~~~bash
git clone https://github.com/kangalio/minacalc-standalone external/minacalc-standalone
bash minacalc/cpp/build.sh        # 产出 external/minacalc-standalone/build/export_itv
~~~

没有导出器时 `backend="approx"` 可跑，但它只是结构近似，不是真实 MSD。

## 已归档的管线

以下脚本与文档使用本包，但不再属于 `main.py` 的 step：

* `legacy/scripts/step2_export_charts.py`：逐区间导出到 `.minacalc_cache/`
* `legacy/scripts/step2_fit_skills.py` / `step2_player_step.py` / `step2_bench.py`
* `legacy/scripts/step0h_minacalc_msd.py` 及 MSD 特征模型（L1/L2）
* 设计文档：`legacy/docs/step2_fit_skills.md`、`legacy/docs/step2_minacalc.md`、
  `legacy/docs/step2_minacalc_L2.md`
