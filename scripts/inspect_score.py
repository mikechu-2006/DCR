#!/usr/bin/env python3
"""人工检查 minacalc.core.score：给定 beatmap id + 玩家向量，打印难度与预计成绩。

用法
----
    # 玩家向量顺序: stream, js, hs, jackspeed, cj, tech[, endurance]
    python scripts/inspect_score.py 1000026 -p 20,19,18,17,18,19,1.0

    # 只给 6 维时 endurance 默认 1.0；省略 -p 时默认全部 20、endurance=1.0
    python scripts/inspect_score.py 1000026

    # 直接用本地 .osu（跳过 beatmap id 查找），并切换 527 口径
    python scripts/inspect_score.py path/to/chart.osu -p 21,21,20,19,20,21 --p-exp 527

    # 机器可读输出
    python scripts/inspect_score.py 1000026 -p 20,19,18,17,18,19 --json

说明
----
* core.py 的谱面难度只有 6 维；玩家向量的第 7 维是耐力系数 endurance（不是谱面难度）。
  因此本脚本的「7 维难度」= 6 个技能维 + overall：
    - 6 个技能维：core.required_skill(chart, dim, goal)，即其余维清零后达到 goal 所需技能；
      该方法使用 combine="sum" 且关闭疲劳；jackspeed 保留 jack 事件，technical 保留 tech 额外项，
      避免 jackspeed 退回 base_adj_diff 的死数据。和默认 aggregate 分数不完全同口径，仅用于定位。
    - overall：六维取同一个技能值，关闭疲劳，按实际的 combine / jack 开关二分到 goal。
* 预计成绩是 score() 返回的 gotpoints / MaxPoints（Etterna 分数比，不是 mania ACC）。
  按设计它可能 < 0 或 > 1，脚本会给出提示而不是截断。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minacalc.core import DIMS, P_EXP, P_EXP_527, Player, required_skill, score  # noqa: E402
from minacalc.prep import load_chart  # noqa: E402

LABELS = {
    "stream": "stream",
    "js": "jumpstream / js",
    "hs": "handstream / hs",
    "jackspeed": "jackspeed",
    "cj": "chordjack / cj",
    "tech": "technical / tech",
}


def find_osu(spec: str, osu_dir: Path) -> Path:
    p = Path(spec).expanduser()
    if p.exists():
        return p.resolve()
    if spec.isdigit():
        cand = osu_dir / f"{spec}.osu"
        if cand.exists():
            return cand.resolve()
        raise SystemExit(
            f"找不到 beatmap {spec}: {cand} 不存在（用 --osu-dir 指定缓存目录）")
    raise SystemExit(f"既不是已有文件也不是 beatmap id: {spec}")


def parse_player(text: list[str] | None) -> tuple[list[float], float | None, bool]:
    """返回 (6 维技能, endurance, 是否用了默认值)。"""
    if text is None:
        return [20.0] * 6, 1.0, True

    raw = [v for item in text for v in item.replace(",", " ").split() if v]
    vals: list[float | None] = []
    for v in raw:
        if v.lower() in ("none", "off", "-"):
            vals.append(None)
        else:
            try:
                vals.append(float(v))
            except ValueError:
                raise SystemExit(f"无法解析玩家向量中的 {v!r}: {' '.join(text)}")

    if len(vals) == 6:
        end: float | None = 1.0
    elif len(vals) == 7:
        end = vals[6]
        if end is None:
            raise SystemExit("endurance 只能写数值；要关闭疲劳请用 --endurance none")
    else:
        raise SystemExit(
            "玩家向量需要 6 个技能值（stream,js,hs,jackspeed,cj,tech）"
            f"或 7 个（再加 endurance），收到 {len(vals)} 个: {' '.join(text)}")

    skills = [float(v) for v in vals[:6]]
    if any(not math.isfinite(v) for v in skills):
        raise SystemExit(f"玩家向量含非有限值: {' '.join(text)}")
    if any(v < 0 for v in skills):
        print("警告: 玩家向量含负技能值；score() 在数学上可算，但通常没有实际意义",
              file=sys.stderr)
    return skills, end, False


def required_overall(chart, goal: float, combine: str, p_exp, *,
                     use_jack: bool, tech_jack_extra: bool) -> float:
    """六维同值玩家达到 goal 所需技能（与预计成绩同一 combine / 开关口径）。"""
    def sc(x: float) -> float:
        p = Player(**{d: x for d in DIMS}, endurance=None)
        return score(chart, p, combine=combine, p_exp=p_exp,
                     use_jack=use_jack, tech_jack_extra=tech_jack_extra)

    hi = max(float(np.max(chart.diffs)), 60.0) + 10.0
    if sc(0.0) >= goal:
        return 0.0
    if sc(hi) < goal:
        return float("nan")
    lo = 0.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if sc(mid) >= goal:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def fmt(x: float, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float) and not math.isfinite(x):
        return "nan"
    return f"{x:.{nd}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="检查 minacalc.core.score：beatmap id + 玩家向量 -> 难度与预计成绩",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法", 1)[-1] if __doc__ else None)
    ap.add_argument("beatmap", help="beatmap id（在 --osu-dir 下找 <id>.osu）或 .osu 路径")
    ap.add_argument("-p", "--player", metavar="VEC", nargs="+",
                    help="stream,js,hs,jackspeed,cj,tech[,endurance]；"
                         "逗号或空格分隔（空格分隔时建议加引号，或直接写多个参数）")
    ap.add_argument("-e", "--endurance", default=None,
                    help="覆盖向量里的 endurance；'none' 关闭疲劳模型")
    ap.add_argument("--goal", type=float, default=0.93,
                    help="计算所需技能时的目标分数比（默认 0.93）")
    ap.add_argument("--combine", choices=("aggregate", "max", "sum"), default="aggregate",
                    help="score() 的六维合并方式（默认 aggregate）")
    ap.add_argument("--p-exp", choices=("505", "527"), default="505",
                    help="失分幂次口径（默认 505，与 C++ 导出器一致）")
    ap.add_argument("--rate", type=float, default=1.0, help="谱面倍率（默认 1.0）")
    ap.add_argument("--backend", choices=("auto", "itv", "approx"), default="auto",
                    help="谱面难度来源（默认 auto：有 C++ 导出器就用 itv）")
    ap.add_argument("--osu-dir", default=str(ROOT / "data" / "osu_cache"),
                    help="beatmap id -> <id>.osu 的目录")
    ap.add_argument("--cache-dir", default=str(ROOT / ".minacalc_cache"),
                    help="逐区间难度缓存目录")
    ap.add_argument("--no-jack", action="store_true",
                    help="关闭 jack 事件模型（退回幂律近似）")
    ap.add_argument("--no-tech-extra", action="store_true",
                    help="关闭 tech 维的额外 jack 扣分")
    ap.add_argument("--diag", action="store_true",
                    help="额外打印 endurance / jack / combine 开关对分数的影响")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args(argv)

    osu_path = find_osu(args.beatmap, Path(args.osu_dir))
    skills, endurance, used_default = parse_player(args.player)
    if args.endurance is not None:
        if args.endurance.lower() in ("none", "off", "-"):
            endurance = None
        else:
            try:
                endurance = float(args.endurance)
            except ValueError:
                raise SystemExit(f"--endurance 需要数值或 none，收到 {args.endurance!r}")

    if not (0.0 < args.goal < 1.0):
        raise SystemExit(f"--goal 必须在 (0, 1) 内，收到 {args.goal}")
    if not (math.isfinite(args.rate) and args.rate > 0):
        raise SystemExit(f"--rate 必须为正数，收到 {args.rate}")

    p_exp = P_EXP_527 if args.p_exp == "527" else P_EXP
    use_jack = not args.no_jack
    tech_jack_extra = not args.no_tech_extra
    exe = ROOT / "external" / "minacalc-standalone" / "build" / "export_itv"

    chart = load_chart(osu_path, rate=args.rate, cache_dir=args.cache_dir,
                       backend=args.backend, itv_exe=exe)
    player = Player(**dict(zip(DIMS, skills)), endurance=endurance)

    detail = score(chart, player, combine=args.combine, p_exp=p_exp,
                   use_jack=use_jack, tech_jack_extra=tech_jack_extra, detail=True)
    sc = float(detail["score"])
    total = float(chart.max_points)
    got = float(detail["points"])
    lost = float(detail["lost"])
    tech_extra = float(detail["tech_jack_extra"])
    hand_lost = [float(np.sum(np.asarray(v))) for v in detail["per_interval_loss"]]

    # 6 个技能维的「所需技能」用 core.required_skill 的 canonical 口径（sum、无疲劳；
    # jackspeed/technical 保留 jack 相关项）。
    req_dims = [float(required_skill(chart, k, goal=args.goal, p_exp=p_exp))
                for k in range(6)]
    req_overall = required_overall(chart, args.goal, args.combine, p_exp,
                                   use_jack=use_jack, tech_jack_extra=tech_jack_extra)
    req_overall_json = req_overall if math.isfinite(req_overall) else None

    mean_dims = [float(np.mean(chart.diffs[k])) for k in range(6)]
    # diffs[3] 是 base_adj_diff 的死行（NPSBase×1.01），不能代表 jackspeed；
    # 有 jack 事件时用事件均值，和 required_skill 的口径一致。
    if chart.jack is not None:
        ys = [np.asarray(y, dtype=float) for _, y in chart.jack if len(y)]
        if ys:
            mean_dims[3] = float(np.mean(np.concatenate(ys)))

    diag = {}
    if args.diag:
        if endurance is not None:
            diag["endurance_none"] = float(score(
                chart, Player(**dict(zip(DIMS, skills)), endurance=None),
                combine=args.combine, p_exp=p_exp, use_jack=use_jack,
                tech_jack_extra=tech_jack_extra))
        if chart.jack is not None:
            diag["no_jack"] = float(score(
                chart, player, combine=args.combine, p_exp=p_exp,
                use_jack=False, tech_jack_extra=False))
        for c in ("max", "sum"):
            if c != args.combine:
                diag[f"combine_{c}"] = float(score(
                    chart, player, combine=c, p_exp=p_exp,
                    use_jack=use_jack, tech_jack_extra=tech_jack_extra))

    out = {
        "beatmap_id": Path(osu_path).stem,
        "osu_path": str(osu_path),
        "source": chart.source,
        "interval": chart.interval,
        "rate": chart.rate,
        "n_intervals": chart.n,
        "max_points": total,
        "jack_events": None if chart.jack is None else [int(len(j[0])) for j in chart.jack],
        "player": {**dict(zip(DIMS, skills)), "endurance": endurance},
        "player_was_default": used_default,
        "goal": args.goal,
        "combine": args.combine,
        "p_exp": args.p_exp,
        "difficulty": {
            "dims": dict(zip(DIMS, req_dims)),
            "overall": req_overall_json,
            "chart_mean": dict(zip(DIMS, mean_dims)),
        },
        "predicted": {
            "score": sc,
            "points": got,
            "lost": lost,
            "hand_lost": hand_lost,
            "tech_jack_extra": tech_extra,
            "jack_source": detail["jack_source"],
        },
        "diagnostics": diag,
    }

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    e_txt = "off（关闭疲劳）" if endurance is None else f"{endurance:.3f}"
    print(f"beatmap : {Path(osu_path).stem}  ({chart.source}, {chart.n} 区间, "
          f"rate={chart.rate:g}, MaxPoints={total:.0f})")
    print(f"player  : stream={skills[0]:.2f} js={skills[1]:.2f} hs={skills[2]:.2f} "
          f"jackspeed={skills[3]:.2f} cj={skills[4]:.2f} tech={skills[5]:.2f} "
          f"endurance={e_txt}" + ("  [默认值]" if used_default else ""))
    print()

    print(f"== 7 维难度（goal={args.goal:.2f}，均无疲劳；技能维=required_skill，"
          f"overall=六维同值）==")
    print(f"{'维度':<20} {'谱面均值':>9} {'所需技能':>9} {'玩家':>8}")
    for k, d in enumerate(DIMS):
        print(f"{LABELS[d]:<20} {mean_dims[k]:>9.2f} {req_dims[k]:>9.2f} {skills[k]:>8.2f}")
    print(f"{'overall':<20} {'-':>9} {fmt(req_overall, 2):>9} {'-':>8}")
    print("  注: 技能维是 required_skill 的 sum/无疲劳口径；jackspeed 含 jack 事件"
          "（谱面均值取 jack 事件均值，diffs[3] 是死行），technical 含 tech 额外项；"
          "overall 用实际 combine。")
    print()

    print("== 预计成绩 ==")
    print(f"  score   {sc:.6f}  ({sc * 100:.3f}%)   [gotpoints / MaxPoints，不是 mania ACC]")
    print(f"  points  {got:.3f} / {total:.3f}")
    print(f"  lost    {lost:.3f}  = 左手 {hand_lost[0]:.3f} + 右手 {hand_lost[1]:.3f}"
          f" + tech_extra {tech_extra:.3f}")
    print(f"  jack    {detail['jack_source']}")
    if total > 0 and (sc < 0.0 or sc > 1.0):
        print(f"  注意: score 落在 [0,1] 之外（{sc:.4f}）；core.py 的设计允许总分 < 0 或 > MaxPoints。")

    if diag:
        print()
        print("== 诊断（开关对分数的影响）==")
        print(f"  当前配置              {sc:.6f}")
        names = {
            "endurance_none": "endurance=None（关疲劳）",
            "no_jack": "use_jack=False（关 jack 事件）",
            "combine_max": "combine=max",
            "combine_sum": "combine=sum",
        }
        for k, v in diag.items():
            print(f"  {names.get(k, k):<24} {v:.6f}  (Δ {v - sc:+.6f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
