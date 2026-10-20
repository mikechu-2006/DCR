"""自检 + 最小示例：python -m minacalc"""

import numpy as np

from .core import (C_REST, C_TECH, ChartData, Player, aggregate_rows,
                   loss_fraction, pow_coef, score)
from .core import default_coef


def _uniform_chart(n=120, d=20.0, notes_per_hand=10):
    pts = np.full((2, n), 2.0 * notes_per_hand)
    diffs = np.full((6, 2, n), float(d))
    return ChartData("uniform", 1.0, 0.5, pts, diffs)


def main():
    print("== aggregate_rows (delta=0.25, mult=1) ==")
    for name, v in [("[20] x7", [[20.0]] * 7), ("[20]", [[20.0]]), ("zeros", [[0.0]])]:
        print(f"  {name:10s} -> {aggregate_rows(v, 0.25, 1.0).round(4)}")

    print("\n== 三次式初值 vs 精确 (1-u^p) ==")
    for p in (1.7, 1.8, 2.0):
        u = np.linspace(0, 1, 6)
        exact = 1 - u ** p
        approx = loss_fraction(u, pow_coef(p))
        err = np.abs(approx - exact)
        print(f"  p={p}: max|err|={err.max():.4f}  at u={u[err.argmax()]:.2f}")
    print(f"  C_TECH={np.round(C_TECH, 4)}  C_REST={np.round(C_REST, 4)}")

    print("\n== 均匀谱 (d=20, 120 区间) ==")
    ch = _uniform_chart()
    for x in (17.0, 19.0, 19.16, 20.0, 22.0):
        p = Player(stream=x, js=x, hs=x, jackspeed=x, cj=x, tech=x, endurance=None)
        print(f"  x={x:5.2f}  aggregate={score(ch, p):.4f}  "
              f"max={score(ch, p, combine='max'):.4f}  (无疲劳)")

    print("\n== 耐力系数的影响 (x=19.16, 120 区间) ==")
    p = Player(stream=19.16, js=19.16, hs=19.16, jackspeed=19.16, cj=19.16,
               tech=19.16, endurance=None)
    for e in (None, 0.8, 1.0, 1.2, 1.5):
        p.endurance = e
        print(f"  endurance={str(e):5s}  score={score(ch, p):.4f}")

    print("\n== 单调性 + 合成谱 ==")
    rng = np.random.default_rng(0)
    n = 200
    pts = np.full((2, n), 12.0)
    diffs = rng.uniform(8, 26, size=(6, 2, n))
    ch2 = ChartData("random", 1.0, 0.5, pts, diffs)
    vals = []
    for x in (8, 12, 16, 20, 24, 28):
        p = Player(stream=x, js=x - 1, hs=x - 2, jackspeed=x - 3, cj=x - 2,
                   tech=x - 1, endurance=None)
        vals.append(score(ch2, p))
    print("  scores:", np.round(vals, 4))
    print("  monotone:", all(b > a for a, b in zip(vals, vals[1:])))

    print("\n== 单维解析对照 ==")
    # 均匀谱、单维、sum 合并：x/d = (1 - (1-goal))**(1/p) 的唯一失分点
    p = Player(**{d: 0.0 for d in ("stream", "js", "hs", "jackspeed", "cj", "tech")},
               endurance=None)
    p.stream = 19.164
    one = ChartData("one", 1.0, 0.5, np.full((2, 10), 20.0),
                    np.concatenate([np.full((1, 2, 10), 20.0),
                                    np.zeros((5, 2, 10))]))
    print(f"  单 stream 轴 x=19.164, 10 区间 -> {score(one, p, combine='sum'):.4f} "
          f"(期望 ~0.93)")

    print("\n== 默认系数 vs p=1.7 / p=2.0（同一张谱） ==")
    base = default_coef()
    for tag, c in [("default", base),
                   ("all 1.7", np.stack([pow_coef(1.7)] * 6)),
                   ("all 2.0", np.stack([pow_coef(2.0)] * 6))]:
        print(f"  {tag:8s} -> score={score(ch, Player(endurance=None), coef=c):.4f}")

    import sys
    if len(sys.argv) > 1:
        from .prep import load_chart
        print(f"\n== C++ 导出后端: {sys.argv[1]} ==")
        ch3 = load_chart(sys.argv[1], backend="itv")
        print(f"  intervals={ch3.n}  MaxPoints={ch3.max_points:.0f}  source={ch3.source}")
        print("  每维均值:", np.round(ch3.diffs.mean(axis=(1, 2)), 2))
        for x in (15, 18, 21, 24):
            p3 = Player(stream=x, js=x, hs=x - 2, jackspeed=x - 4, cj=x - 3,
                        tech=x - 1, endurance=None)
            print(f"  x={x}  score={score(ch3, p3):.4f}")


if __name__ == "__main__":
    main()
