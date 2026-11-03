#!/usr/bin/env python3
"""白名单谱面的 MinaCalc 难度分布统计。

口径
----
* 谱面集合：--whitelist（默认 manifests/whitelist_1k.csv，21,949 张）。
* 难度来源：minacalc 的 C++ 逐区间导出（.minacalc_cache/*_itv_*.npz），
  与本地修改版 minacalc.core.score 使用同一套 505 口径。
* 全曲（chart level）：
    - req_<dim>：minacalc.core.required_skill(chart, dim, goal=0.93)，
      即把其它维清零后达到 goal 所需的技能；jackspeed 走 jack 事件，
      technical 含 tech 额外项（本地修改版的行为）。
    - req_overall：六维取同一技能值，按 combine="aggregate"、无疲劳
      二分到 goal 所需技能（与 scripts/inspect_score.py 的 overall 同口径）。
    - mean_<dim>：谱面内所有 0.5 s 区间、双手难度的算术平均；
      jackspeed 的 diffs[3] 是死行，改用该谱所有 jack 事件的 y 均值。
  对每张谱算出上述标量后，跨谱面求 mean/std。
* 分段（segment level）：
    - interval：把每张谱的每个 0.5 s 区间当成一个样本（双手先平均），
      汇总白名单上的全部区间；jackspeed 用落在该区间的 jack 事件 y 均值。
    - position_decile：每张谱按区间序号等分成 10 段，段内对区间取均值，
      再跨谱面汇总；用于看前段/后段难度剖面。
    - difficulty_band：按全曲 req_overall 分档（固定档 + 分位数档），
      档内对全曲各维求 mean/std。
* std 默认是样本标准差（ddof=1）；区间级数量很大，同时给出 n。

用法
----
    # 一次跑完（自动分片，8 进程并行）
    python scripts/minacalc_whitelist_stats.py --jobs 8

    # 或手动分片 + 合并
    for i in 0 1 2 3; do
      python scripts/minacalc_whitelist_stats.py --shard $i --shards 4 &
    done; wait
    python scripts/minacalc_whitelist_stats.py --merge

产物（默认 results/minacalc/whitelist_stats/）：
    chart_level.csv                  每张谱一行的全曲指标
    whole_song_required_skill.csv    全曲 required_skill 的跨谱 mean/std
    whole_song_interval_mean.csv     全曲区间均值的跨谱 mean/std
    segment_interval.csv             全部 0.5 s 区间的 pooled mean/std
    segment_position_decile.csv      位置十分段的 mean/std
    difficulty_band_fixed.csv        固定难度档内的 mean/std
    difficulty_band_quantile.csv     分位数难度档内的 mean/std
    summary.json                     口径、覆盖、样本量
    report.md                        中文摘要表
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minacalc.core import DIMS, Player, score, required_skill  # noqa: E402
from minacalc.io import load_chart_itv  # noqa: E402

# scipy 的向量化 erf/erfc 只用于加速 aggregate 的二分；数值与 core.erfc 等价到
# double 精度。required_skill 的 sum 口径本身不用它们。
try:
    from scipy.special import erf as _sp_erf, erfc as _sp_erfc
    import minacalc.core as _core
    _core.erf = _sp_erf
    _core.erfc = _sp_erfc
except Exception:  # pragma: no cover - scipy 缺失时退回 math.erf
    pass

DIMS_FULL = ("stream", "jumpstream", "handstream", "jackspeed", "chordjack", "technical")
DIMS_KEY = DIMS  # ("stream","js","hs","jackspeed","cj","tech")
REQ_COLS = [f"req_{d}" for d in DIMS_FULL] + ["req_overall"]
MEAN_COLS = [f"mean_{d}" for d in DIMS_FULL]

DEFAULT_WHITELIST = ROOT / "manifests" / "whitelist_1k.csv"
DEFAULT_OSU_DIR = ROOT / "data" / "osu_cache"
DEFAULT_CACHE_DIR = ROOT / ".minacalc_cache"
DEFAULT_OUT = ROOT / "results" / "minacalc" / "whitelist_stats"
CACHE_SUFFIX = "_itv_1.000_0.930_v2.npz"


# --------------------------------------------------------------------------- io
def read_ids(path: Path) -> list[int]:
    df = pd.read_csv(path)
    col = "beatmap_id" if "beatmap_id" in df.columns else df.columns[0]
    return [int(v) for v in df[col].tolist()]


def cache_path(bid: int, osu_dir: Path, cache_dir: Path) -> Path:
    p = osu_dir / f"{bid}.osu"
    key = hashlib.sha1(p.read_bytes()).hexdigest()[:16] + CACHE_SUFFIX
    return cache_dir / key


# --------------------------------------------------------------- chart metrics
def _jack_arrays(ch, n: int):
    """每手每区间的 jack 事件 y 均值，以及全谱事件加权均值。"""
    per_hand = []
    all_y = []
    for itv, y in (ch.jack or []):
        sums = np.zeros(n, dtype=np.float64)
        cnts = np.zeros(n, dtype=np.int64)
        itv = np.asarray(itv, dtype=np.int64)
        y = np.asarray(y, dtype=np.float64)
        ok = (itv >= 0) & (itv < n)
        np.add.at(sums, itv[ok], y[ok])
        np.add.at(cnts, itv[ok], 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
        per_hand.append(m)
        if y[ok].size:
            all_y.append(y[ok])
    return per_hand, (np.concatenate(all_y) if all_y else np.zeros(0))


def _interval_matrix(ch):
    """返回 (6, n) 的区间难度矩阵，双手平均；jackspeed 用事件均值。"""
    n = ch.n
    mat = np.asarray(ch.diffs, dtype=np.float64).mean(axis=1)  # (6,n)
    if ch.jack is not None:
        per_hand, _ = _jack_arrays(ch, n)
        stack = np.vstack(per_hand)
        cnt = np.isfinite(stack).sum(axis=0)
        with np.errstate(invalid="ignore"):
            mat[3] = np.where(cnt > 0, np.nansum(stack, axis=0) / np.maximum(cnt, 1), np.nan)
    return mat


def _required_overall(ch, goal: float, bisect_iters: int, agg_iters: int) -> float:
    """与 scripts/inspect_score.required_overall 同口径，略减迭代次数。"""
    def sc(x: float) -> float:
        p = Player(**{d: x for d in DIMS_KEY}, endurance=None)
        return score(ch, p, combine="aggregate", agg_iters=agg_iters)

    hi = max(float(np.max(ch.diffs)), 60.0) + 10.0
    if sc(0.0) >= goal:
        return 0.0
    if sc(hi) < goal:
        return float("nan")
    lo = 0.0
    for _ in range(bisect_iters):
        mid = 0.5 * (lo + hi)
        if sc(mid) >= goal:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def chart_metrics(ch, goal: float, n_pos: int, *, need_overall: bool,
                  bisect_iters: int, agg_iters: int):
    n = ch.n
    mat = _interval_matrix(ch)
    _, jy = _jack_arrays(ch, n)
    row = {
        "beatmap_id": int(ch.name),
        "n_intervals": int(n),
        "duration_s": float(n * ch.interval),
        "max_points": float(ch.max_points),
        "n_jack_events": int(len(jy)),
    }
    for k, d in enumerate(DIMS_FULL):
        row[f"req_{d}"] = float(required_skill(ch, k, goal=goal))
        row[f"mean_{d}"] = float(jy.mean()) if k == 3 and jy.size else float(mat[k].mean())
    row["req_overall"] = (_required_overall(ch, goal, bisect_iters, agg_iters)
                          if need_overall else float("nan"))

    pos = {}
    if n_pos > 0 and n > 0:
        for s, idx in enumerate(np.array_split(np.arange(n), n_pos), start=1):
            if idx.size == 0:
                continue
            for k, d in enumerate(DIMS_FULL):
                v = mat[k, idx]
                v = v[np.isfinite(v)]
                if v.size:
                    pos[(s, d)] = (float(v.sum()), float((v * v).sum()), int(v.size))
    return row, mat, pos


# ------------------------------------------------------------------- shard run
def run_shard(args, ids: list[int]):
    out = Path(args.out_dir)
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    my_ids = ids[args.shard::args.shards]
    n_pos = int(args.position_segments)
    dims_n = len(DIMS_FULL)
    gl_sum = np.zeros(dims_n); gl_sq = np.zeros(dims_n); gl_n = np.zeros(dims_n, dtype=np.int64)
    gl_min = np.full(dims_n, np.inf); gl_max = np.full(dims_n, -np.inf)
    pos_sum = np.zeros((n_pos, dims_n)); pos_sq = np.zeros((n_pos, dims_n))
    pos_n = np.zeros((n_pos, dims_n), dtype=np.int64)
    rows = []
    t0 = time.time()
    for j, bid in enumerate(my_ids):
        try:
            ch = load_chart_itv(Path(args.osu_dir) / f"{bid}.osu", cache_dir=args.cache_dir)
            row, mat, pos = chart_metrics(
                ch, args.goal, n_pos, need_overall=not args.no_overall,
                bisect_iters=args.bisect_iters, agg_iters=args.agg_iters)
            rows.append(row)
            for k in range(dims_n):
                v = mat[k]
                v = v[np.isfinite(v)]
                if v.size:
                    gl_sum[k] += float(v.sum()); gl_sq[k] += float((v * v).sum())
                    gl_n[k] += int(v.size)
                    gl_min[k] = min(gl_min[k], float(v.min()))
                    gl_max[k] = max(gl_max[k], float(v.max()))
            for (s, d), (sm, sq, c) in pos.items():
                k = DIMS_FULL.index(d)
                pos_sum[s - 1, k] += sm
                pos_sq[s - 1, k] += sq
                pos_n[s - 1, k] += c
        except Exception as exc:  # noqa: BLE001 - 逐张记录，不中断分片
            print(f"[shard {args.shard}] {bid} FAILED: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
        if (j + 1) % 250 == 0:
            print(f"[shard {args.shard}] {j + 1}/{len(my_ids)}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(parts / f"chart_level_shard{args.shard}.csv", index=False)
    np.savez(parts / f"seg_shard{args.shard}.npz",
             gl_sum=gl_sum, gl_sq=gl_sq, gl_n=gl_n, gl_min=gl_min, gl_max=gl_max,
             pos_sum=pos_sum, pos_sq=pos_sq, pos_n=pos_n)
    print(f"[shard {args.shard}] done {len(rows)}/{len(my_ids)}  "
          f"{time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------- merge
def _stats_from_moments(sm, sq, n):
    sm, sq, n = np.asarray(sm, float), np.asarray(sq, float), np.asarray(n, int)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, sm / np.maximum(n, 1), np.nan)
        var = np.where(n > 1, (sq - sm * sm / np.maximum(n, 1)) / (n - 1), np.nan)
    return mean, np.sqrt(np.maximum(var, 0.0))


def _describe(values, digits=6):
    v = pd.Series(values).dropna().to_numpy(dtype=float)
    if v.size == 0:
        return dict(n=0)
    return {
        "n": int(v.size),
        "mean": round(float(v.mean()), digits),
        "std": round(float(v.std(ddof=1)), digits) if v.size > 1 else 0.0,
        "min": round(float(v.min()), digits),
        "p05": round(float(np.percentile(v, 5)), digits),
        "p25": round(float(np.percentile(v, 25)), digits),
        "median": round(float(np.median(v)), digits),
        "p75": round(float(np.percentile(v, 75)), digits),
        "p95": round(float(np.percentile(v, 95)), digits),
        "max": round(float(v.max()), digits),
    }


def merge(args):
    out = Path(args.out_dir)
    parts = out / "parts"
    files = sorted(parts.glob("chart_level_shard*.csv"))
    if not files:
        raise SystemExit(f"没有分片结果：{parts}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates("beatmap_id").sort_values("beatmap_id").reset_index(drop=True)
    df.to_csv(out / "chart_level.csv", index=False)

    ws_req, ws_mean = [], []
    for d in DIMS_FULL:
        ws_req.append({"dimension": d, **_describe(df[f"req_{d}"])})
        ws_mean.append({"dimension": d, **_describe(df[f"mean_{d}"])})
    ws_req.append({"dimension": "overall", **_describe(df["req_overall"])})
    pd.DataFrame(ws_req).to_csv(out / "whole_song_required_skill.csv", index=False)
    pd.DataFrame(ws_mean).to_csv(out / "whole_song_interval_mean.csv", index=False)

    zs = sorted(parts.glob("seg_shard*.npz"))
    gl_sum = gl_sq = None
    gl_n = None
    gl_min = np.full(len(DIMS_FULL), np.inf); gl_max = np.full(len(DIMS_FULL), -np.inf)
    pos_sum = pos_sq = None; pos_n = None
    for f in zs:
        with np.load(f) as z:
            if gl_sum is None:
                gl_sum, gl_sq = z["gl_sum"].copy(), z["gl_sq"].copy()
                gl_n = z["gl_n"].copy()
                pos_sum, pos_sq = z["pos_sum"].copy(), z["pos_sq"].copy()
                pos_n = z["pos_n"].copy()
            else:
                gl_sum += z["gl_sum"]; gl_sq += z["gl_sq"]; gl_n += z["gl_n"]
                pos_sum += z["pos_sum"]; pos_sq += z["pos_sq"]; pos_n += z["pos_n"]
            gl_min = np.minimum(gl_min, z["gl_min"]); gl_max = np.maximum(gl_max, z["gl_max"])
    mean, std = _stats_from_moments(gl_sum, gl_sq, gl_n)
    seg_rows = []
    for k, d in enumerate(DIMS_FULL):
        seg_rows.append({"dimension": d, "n_intervals": int(gl_n[k]),
                         "mean": round(float(mean[k]), 6), "std": round(float(std[k]), 6),
                         "min": round(float(gl_min[k]), 6), "max": round(float(gl_max[k]), 6)})
    pd.DataFrame(seg_rows).to_csv(out / "segment_interval.csv", index=False)

    pmean, pstd = _stats_from_moments(pos_sum, pos_sq, pos_n)
    pos_rows = []
    for s in range(int(args.position_segments)):
        for k, d in enumerate(DIMS_FULL):
            if pos_n[s, k] == 0:
                continue
            pos_rows.append({"segment": s + 1, "dimension": d,
                             "n_intervals": int(pos_n[s, k]),
                             "mean": round(float(pmean[s, k]), 6),
                             "std": round(float(pstd[s, k]), 6)})
    pd.DataFrame(pos_rows).to_csv(out / "segment_position_decile.csv", index=False)

    ov = df["req_overall"].to_numpy(dtype=float)
    finite = np.isfinite(ov)
    fixed_edges = [0, 5, 10, 15, 20, 25, np.inf]
    fixed_labels = ["[0,5)", "[5,10)", "[10,15)", "[15,20)", "[20,25)", "[25,inf)"]
    bins = np.full(len(df), -1, dtype=int)
    for b in range(len(fixed_edges) - 1):
        m = finite & (ov >= fixed_edges[b]) & (ov < fixed_edges[b + 1])
        bins[m] = b
    _write_band_table(df, bins, fixed_labels, out / "difficulty_band_fixed.csv")

    q = np.nanpercentile(ov[finite], np.linspace(0, 100, 11)) if finite.any() else np.array([])
    qbins = np.full(len(df), -1, dtype=int)
    if q.size == 11:
        for b in range(10):
            lo, hi = q[b], q[b + 1]
            m = finite & (ov >= lo) & ((ov < hi) if b < 9 else (ov <= hi))
            qbins[m] = b
    qlabels = [f"Q{b + 1} [{q[b]:.2f},{q[b + 1]:.2f}{']' if b == 9 else ')'}"
               for b in range(10)] if q.size == 11 else []
    _write_band_table(df, qbins, qlabels, out / "difficulty_band_quantile.csv")

    summary = {
        "whitelist": str(Path(args.whitelist).resolve()),
        "n_charts": int(len(df)),
        "n_intervals_total": int(gl_n[0]),
        "n_jack_events_total": int(df["n_jack_events"].sum()),
        "goal": float(args.goal),
        "interval_s": 0.5,
        "std_ddof": 1,
        "position_segments": int(args.position_segments),
        "required_skill": "minacalc.core.required_skill (modified: jack events for jackspeed/technical)",
        "overall": "6 equal skills, combine=aggregate, no endurance, bisection",
        "interval_mean": "mean over 0.5 s intervals, hands averaged; jackspeed=jack event mean",
        "duration_s": _describe(df["duration_s"]),
        "n_intervals": _describe(df["n_intervals"]),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    md = _render_report(df, out, summary)
    (out / "report.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"[merge] wrote {out}")


def _write_band_table(df, bins, labels, path):
    rows = []
    for b, label in enumerate(labels):
        sub = df[bins == b]
        if len(sub) == 0:
            continue
        row = {"band": label, "n_charts": int(len(sub))}
        for d in DIMS_FULL:
            row[f"req_{d}_mean"] = round(float(sub[f"req_{d}"].mean()), 6)
            row[f"req_{d}_std"] = round(float(sub[f"req_{d}"].std(ddof=1)), 6)
            row[f"mean_{d}_mean"] = round(float(sub[f"mean_{d}"].mean()), 6)
            row[f"mean_{d}_std"] = round(float(sub[f"mean_{d}"].std(ddof=1)), 6)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _md_table(df, cols=None, floatfmt="{:.3f}"):
    d = df if cols is None else df[cols]
    lines = ["| " + " | ".join(map(str, d.columns)) + " |",
             "|" + "|".join(["---"] * len(d.columns)) + "|"]
    for _, r in d.iterrows():
        vals = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                vals.append(floatfmt.format(float(v)))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _render_report(df, out, summary):
    req = pd.read_csv(out / "whole_song_required_skill.csv")
    mean = pd.read_csv(out / "whole_song_interval_mean.csv")
    seg = pd.read_csv(out / "segment_interval.csv")
    pos = pd.read_csv(out / "segment_position_decile.csv")
    bands = pd.read_csv(out / "difficulty_band_fixed.csv")
    qbands = pd.read_csv(out / "difficulty_band_quantile.csv")
    lines = [
        "# 白名单 MinaCalc 难度分布",
        "",
        f"- 谱面：{summary['n_charts']:,} 张（{summary['whitelist']}）",
        f"- 区间：{summary['n_intervals_total']:,} 个 0.5 s 区间；"
        f"jack 事件 {summary['n_jack_events_total']:,} 个",
        f"- goal={summary['goal']:.2f}；std 为样本标准差（ddof=1）",
        "- 难度来源：本地修改版 minacalc，C++ base_adj_diff 逐区间导出",
        "",
        "## 全曲（每张谱一个标量，跨谱统计）",
        "",
        "### required_skill（达到 goal=0.93 所需技能）",
        "",
        _md_table(req, ["dimension", "n", "mean", "std", "min", "p05", "median", "p95", "max"]),
        "",
        "### 区间均值（谱内所有 0.5 s 区间平均）",
        "",
        _md_table(mean, ["dimension", "n", "mean", "std", "min", "p05", "median", "p95", "max"]),
        "",
        "## 分段：0.5 s 区间 pooled",
        "",
        _md_table(seg),
        "",
        "注：jackspeed 的样本是 jack 事件所在区间的均值（没有 jack 的区间不参与），"
        "因此 n 小于其它维；其它维的 n 是全部区间数。区间级统计包含没有 note 的空区间"
        "（MinaCalc 保留空区间，难度与点数为 0），未做 pts>0 过滤。",
        "",
        "## 分段：歌曲位置十分段",
        "",
        _md_table(pos.pivot(index="segment", columns="dimension", values="mean").round(3).reset_index()),
        "",
        "上表是各段 mean；完整 n/std 见 segment_position_decile.csv。",
        "",
        "## 按全曲 overall 难度分档（固定档）",
        "",
        _md_table(bands[["band", "n_charts"] + [f"req_{d}_mean" for d in DIMS_FULL]]),
        "",
        "上表是档内 required_skill 的 mean；完整 std 与区间均值见 difficulty_band_fixed.csv。",
        "",
        "## 按全曲 overall 难度分档（十分位）",
        "",
        _md_table(qbands[["band", "n_charts"] + [f"req_{d}_mean" for d in DIMS_FULL]]),
        "",
        "## 口径说明",
        "",
        "* req_*：required_skill，其余维清零、combine=sum、无疲劳；"
        "jackspeed 使用 jack 事件，technical 含 tech 额外项。",
        "* req_overall：六维同值，combine=aggregate、无疲劳，二分到 goal。",
        "* mean_*：谱内 0.5 s 区间、双手平均；jackspeed 用 jack 事件 y 均值。",
        "* segment_interval：每个区间作为一个样本，双手先平均，再跨全部谱面汇总。",
        "* segment_position_decile：每张谱按区间序号等分 10 段，段内平均后跨谱汇总。",
        "* 固定难度档边界：[0,5) [5,10) [10,15) [15,20) [20,25) [25,inf)，"
        "按 req_overall 分档。",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--whitelist", default=str(DEFAULT_WHITELIST))
    ap.add_argument("--osu-dir", default=str(DEFAULT_OSU_DIR))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--goal", type=float, default=0.93)
    ap.add_argument("--jobs", type=int, default=0,
                    help=">0 时用 subprocess 自动分片并行（每片一个进程）")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true", help="只合并已存在的分片")
    ap.add_argument("--position-segments", type=int, default=10)
    ap.add_argument("--bisect-iters", type=int, default=36,
                    help="overall 二分次数（默认 36，区间宽度 < 1e-9）")
    ap.add_argument("--agg-iters", type=int, default=24,
                    help="aggregate 内部二分次数（默认 24）")
    ap.add_argument("--no-overall", action="store_true",
                    help="不计算 req_overall（可省约 80%% 时间）")
    args = ap.parse_args(argv)

    ids = read_ids(Path(args.whitelist))
    if args.merge:
        merge(args)
        return 0

    if args.jobs and args.jobs > 0 and args.shards == 1:
        import subprocess
        procs = []
        for i in range(args.jobs):
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   "--whitelist", args.whitelist, "--osu-dir", args.osu_dir,
                   "--cache-dir", args.cache_dir, "--out-dir", args.out_dir,
                   "--goal", str(args.goal), "--shard", str(i),
                   "--shards", str(args.jobs),
                   "--position-segments", str(args.position_segments),
                   "--bisect-iters", str(args.bisect_iters),
                   "--agg-iters", str(args.agg_iters)]
            if args.no_overall:
                cmd.append("--no-overall")
            procs.append(subprocess.Popen(cmd))
        for p in procs:
            p.wait()
        if any(p.returncode for p in procs):
            raise SystemExit("至少一个分片失败")
        merge(args)
        return 0

    run_shard(args, ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
