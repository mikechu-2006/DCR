"""MinaCalc 式打分（可微版本）。

设计约定
--------
1. Chisel 的 pbm 全部取 1：起始分就是 MaxPoints，分数上界 1.0，不会超过 MAX。
2. 单点失分是**三次式**（在 u = x/d = 1 处展开，t = clip(1 - u, 0, 1)）：

       loss = c1 t + c2 t^2 + c3 t^3,      c = (c1, c2, c3)

   即 MinaCalc 的 1 - u^p 三阶展开，但有两点不同：
     * c 是**自由参数**（不再由单个 p 决定），p 只用来算初值：pow_coef(1.7)/(2.0)；
     * c0 恒为 0（x = d 时失分为 0）。
   分组（505 口径，见 legacy/docs/step2_fit_skills.md §4.3-B）：
     tech 独占 C_TECH = pow_coef(2.0) = (2, -1, 0)   # c3=0，退化成二次式
     其余五维共享 C_REST = pow_coef(1.7) = (1.7, -0.595, -0.0595)
3. 维度合并：第 i 个区间先算 6 维各自的失分，再用 aggregate() 合并成一个数，从总分扣。
4. 玩家 = 6 维技能 + 可选耐力系数 e（None 表示关闭疲劳模型）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

DIMS = ("stream", "js", "hs", "jackspeed", "cj", "tech")

# 失分幂次。注意版本差异：仓库里 vendored 的 minacalc-standalone 是 505，CJ 用 1.7；
# Etterna develop（527）把 CJ 改成 1.8。默认跟随 505（与导出器一致），要切 527：
#   from minacalc.core import P_EXP_527; score(..., p_exp=P_EXP_527)
P_EXP = np.array([1.7, 1.7, 1.7, 1.7, 1.7, 2.0])
P_EXP_527 = np.array([1.7, 1.7, 1.7, 1.7, 1.8, 2.0])

_FOUR_OVER_SQRT_PI = 4.0 / math.sqrt(math.pi)
_LOG2 = math.log(2.0)
_ZMAX = 200.0          # softplus 输入截断（z 再大时权重已饱和，无需求精确）


def erfc(x) -> np.ndarray:
    """math.erfc 的向量化封装（精确到 double）。"""
    x = np.asarray(x, dtype=float)
    flat = x.ravel()
    out = np.fromiter((math.erfc(v) for v in flat), dtype=float, count=flat.size)
    return out.reshape(x.shape)


def erf(x) -> np.ndarray:
    """math.erf 的向量化封装。"""
    x = np.asarray(x, dtype=float)
    flat = x.ravel()
    out = np.fromiter((math.erf(v) for v in flat), dtype=float, count=flat.size)
    return out.reshape(x.shape)


def pow_coef(p) -> np.ndarray:
    """把 MinaCalc 的失分幂次 p 换成三次式系数 (c1, c2, c3)。

    1 - u^p = 1 - (1-t)^p = p t - p(p-1)/2 t^2 + p(p-1)(p-2)/6 t^3 + O(t^4)
    """
    p = float(p)
    return np.array([p, -0.5 * p * (p - 1.0), p * (p - 1.0) * (p - 2.0) / 6.0])


C_TECH = pow_coef(2.0)      # (2, -1, 0)   tech 独占
C_REST = pow_coef(1.7)      # (1.7, -0.595, -0.0595)  其余五维共享


def default_coef(p_exp=None) -> np.ndarray:
    """逐维系数矩阵 (6, 3)。默认 = 505 口径：tech 用 2.0，其余 1.7。"""
    p = P_EXP if p_exp is None else np.asarray(p_exp, dtype=float)
    return np.stack([pow_coef(v) for v in p])


def loss_fraction(u, coef) -> np.ndarray:
    """三次式失分比例 c1 t + c2 t^2 + c3 t^3，t = clip(1 - u, 0, 1)。

    u = x/d（u >= 1 时为 0）；coef 的最后一维是 (c1, c2, c3)，需能广播到 u。
    c1 = 1.7 / c2 = -0.595 / c3 = -0.0595 时等价于旧的 order=3 三阶泰勒。
    """
    u = np.asarray(u, dtype=float)
    t = np.clip(1.0 - u, 0.0, 1.0)
    c = np.asarray(coef, dtype=float)
    q = c[..., 0] * t + c[..., 1] * t * t + c[..., 2] * t * t * t
    return np.clip(q, 0.0, 1.0)


def aggregate_rows(v, delta=0.25, mult=1.11, beta=5.0, iters=48):
    """aggregate_skill 的 softplus 版本，按列向量化（v: (k, n) -> (n,)）。

    w(z) = softplus_beta(z),  z = 2/erfc(delta*(v-r)) - 2
    解 sum_i w_i(r) = 2^(0.1 r)。

    求解方式：**纯二分**。F(r) 对 r 严格单调递减（每个 w 递减，右端 2^(0.1r) 递增），
    所以二分在 [0, max(v)+10] 上必然收敛，且不依赖导数——这正是原来
    "Newton + 夹逼" 想做的事，但不用再维护 bracket 和 bad-step 判断。
    48 轮 => 区间宽度 10/2^48 ≈ 3.6e-14，float32 下也远低于精度。
    """
    v = np.atleast_2d(np.asarray(v, dtype=float))
    k, n = v.shape
    if k == 0 or n == 0:
        return np.zeros(n)
    def f(r):
        x = np.clip(delta * (v - r), -27.0, 27.0)
        e = erfc(x)
        # 2/erfc - 2 == 2*erf/erfc；后者在 x->0 处没有相消误差
        z = np.clip(2.0 * erf(x) / e, -1.0, _ZMAX)
        w = np.logaddexp(0.0, beta * z) / beta
        return w.sum(axis=0) - np.power(2.0, 0.1 * r)

    lo = np.zeros(n)
    hi = v.max(axis=0) + 10.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fv = f(mid)
        lo = np.where(fv > 0.0, mid, lo)
        hi = np.where(fv < 0.0, mid, hi)
    return 0.5 * (lo + hi) * mult


def stamina_mods(d, x, endurance=1.0, *, ceil=1.075234, mag=243.0, fscale=500.0,
                 prop=0.69424, super_ceil=1.09):
    """MinaCalc 的 StamAdjust，把耐力系数放进门槛：ref = prop * x * e。

    endurance=None -> 关闭疲劳模型；e>1 更抗疲劳，e<1 更容易累。
    """
    d = np.asarray(d, dtype=float)
    if endurance is None:
        return np.ones_like(d)
    ref = prop * max(float(x), 0.0) * float(endurance)
    if ref <= 0.0:
        return np.ones_like(d)
    mods = np.empty_like(d)
    floor, mod, prev = 0.95, 0.95, 0.0
    for i in range(d.size):
        cur = d[i]
        mod += (((prev + cur) * 0.5) / ref - 1.0) / mag
        if mod > 0.95:
            floor += (mod - 0.95) / fscale
        mod = min(max(mod, floor), ceil * floor)
        mod = min(mod, super_ceil)
        mods[i] = mod
        prev = cur
    return mods


@dataclass
class Player:
    stream: float = 20.0
    js: float = 20.0
    hs: float = 20.0
    jackspeed: float = 20.0
    cj: float = 20.0
    tech: float = 20.0
    endurance: float | None = 1.0

    def vector(self) -> np.ndarray:
        return np.array([self.stream, self.js, self.hs, self.jackspeed,
                         self.cj, self.tech], dtype=float)


@dataclass
class ChartData:
    name: str
    rate: float
    interval: float
    pts: np.ndarray       # (2, n)   每手每区间点数
    diffs: np.ndarray     # (6, 2, n) 每维每手每区间难度
    source: str = "approx"
    meta: dict = field(default_factory=dict)
    # 每手的 jack 事件：[(itv int数组, y float数组), ...]，None 表示没有（退回幂律近似）
    jack: list | None = None

    @property
    def n(self) -> int:
        return int(self.pts.shape[1])

    @property
    def max_points(self) -> float:
        return float(self.pts.sum())


def jack_stam(y, x, endurance=1.0, *, ceil=1.05234, mag=23.0, fscale=750.0,
              prop=0.49424, super_ceil=1.01):
    """MinaCalc 的 JackStamAdjust（逐事件递推，参数与区间版不同）。"""
    y = np.asarray(y, dtype=float)
    if endurance is None:
        return y
    ref = prop * max(float(x), 0.0) * float(endurance)
    if ref <= 0.0:
        return y
    out = np.empty_like(y)
    floor, mod, prev = 0.95, 1.0, 0.0
    for i in range(y.size):
        cur = y[i]
        mod += (((prev + cur) * 0.5) / ref - 1.0) / mag
        if mod > 0.95:
            floor += (mod - 0.95) / fscale
        mod = min(max(mod, floor), ceil * floor)
        mod = min(mod, super_ceil)
        out[i] = cur * mod
        prev = cur
    return out


def jack_event_loss(y, x, endurance=1.0, *, amp=12.0, slope=0.04):
    """逐事件点损失 amp*erf(slope*(y-x))（y<=x 时为 0）。

    amp=12 / slope=0.04 是 MinaCalc 的 magic_num 与 0.04F；两者在低难度区只以
    乘积出现（erf(slope*d) ~ slope*d），近似共线，拟合时需要 ridge 盯住 amp。
    """
    y = np.asarray(y, dtype=float)
    if endurance is not None:
        y = jack_stam(y, x, endurance)
    out = np.zeros_like(y)
    m = y > x
    if np.any(m):
        out[m] = float(amp) * np.array([math.erf(slope * (v - x)) for v in y[m]])
    return out


def jack_interval_loss(itv, y, n_itv, x, endurance=1.0, *, amp=12.0, slope=0.04):
    """把事件损失按区间汇总成 (n_itv,)，直接参与 losses 向量。"""
    pen = jack_event_loss(y, x, endurance, amp=amp, slope=slope)
    out = np.zeros(int(n_itv))
    np.add.at(out, np.asarray(itv, dtype=np.int64), pen)
    return out


def score(chart: ChartData, player: Player, *, combine="aggregate",
          agg_delta=0.25, agg_mult=1.11, agg_beta=5.0, agg_iters=30,
          coef=None, p_exp=None, jack_amp=12.0, jack_slope=0.04,
          use_jack=True, tech_jack_extra=True, detail=False):
    """给定 6(+1) 维玩家实力算成绩。

    combine="aggregate" 每区间 6 维失分先 aggregate 再扣（默认，按需求）；
            "max" 取最大失分（对照）；"sum" 直接求和。
    返回 gotpoints / MaxPoints；因为每区间合并值可能大于该区间点数，总分可能 < 0。
    """
    x = player.vector()
    e = player.endurance
    c = default_coef(p_exp) if coef is None else np.asarray(coef, dtype=float)
    total = float(chart.pts.sum())
    lost = 0.0
    per_hand = []
    for h in range(chart.diffs.shape[1]):
        d = chart.diffs[:, h, :].astype(float)
        if e is not None:
            for k in range(d.shape[0]):
                d[k] *= stamina_mods(d[k], x[k], e)
        u = np.divide(x[:, None], d, out=np.full_like(d, np.inf), where=d > 0.0)
        losses = loss_fraction(u, c[:, None, :]) * chart.pts[h][None, :]
        if use_jack and chart.jack is not None:   # 第 4 维改用事件模型（amp·erf）
            jitv, jy = chart.jack[h]
            losses[3] = jack_interval_loss(jitv, jy, chart.n, x[3], e,
                                           amp=jack_amp, slope=jack_slope)
        if combine == "aggregate":
            comb = aggregate_rows(losses, agg_delta, agg_mult, agg_beta, agg_iters)
        elif combine == "max":
            comb = losses.max(axis=0)
        elif combine == "sum":
            comb = losses.sum(axis=0)
        else:
            raise ValueError(f"unknown combine: {combine}")
        lost += float(comb.sum())
        per_hand.append(comb)
    tech_extra = 0.0
    if tech_jack_extra and chart.jack is not None:
        cap = 0.1 * total
        for h in range(chart.diffs.shape[1]):
            jitv, jy = chart.jack[h]
            pen = jack_event_loss(jy, 0.75 * x[5], e,
                                  amp=jack_amp, slope=jack_slope)
            tech_extra += math.sqrt(min(cap, 0.85 * float(pen.sum())))
        lost += tech_extra

    got = total - lost
    out = {"points": got, "score": (got / total) if total else 0.0,
           "lost": lost, "per_interval_loss": per_hand,
           "tech_jack_extra": tech_extra,
           "jack_source": "events" if chart.jack is not None else "powerlaw"}
    return out if detail else out["score"]


def required_skill(chart: ChartData, dim: int, goal=0.93, **kw):
    """单维评分时需要的技能（诊断用）：把其它维难度清零后二分。"""
    sub = ChartData(chart.name, chart.rate, chart.interval, chart.pts,
                    chart.diffs.copy(), chart.source, dict(chart.meta))
    for k in range(6):
        if k != dim:
            sub.diffs[k] = 0.0

    def value(xv):
        p = Player(**{d: 0.0 for d in DIMS})
        p.endurance = None
        setattr(p, DIMS[dim], float(xv))
        return score(sub, p, combine="sum", **kw)

    lo, hi = 0.0, 60.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if value(mid) >= goal:
            hi = mid
        else:
            lo = mid
    return hi
