"""谱面预处理：.osu -> ChartData（npz 缓存，避免每次重算）。

难度模型是"结构近似"：NPSBase 用 MinaCalc 的原式，6 个维度用简化后的键型比例修正。
如果你有真实 MSD（minacalc-rs / osumania_map_analyser 导出），直接写同格式的 npz 即可，
score() 不关心来源，只看 diffs/pts。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .core import ChartData

MODEL_VERSION = "approx-v1"
INTERVAL = 0.5
FINALSCALER = 3.632 * 1.06          # MinaCalc 的 finalscaler


def parse_osu(path):
    """返回 ([(time_sec, col, is_hold)], keycount)。只取 tap 行（与 MinaCalc 一致）。"""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    keycount = 4
    for line in text.splitlines():
        if line.startswith("CircleSize:"):
            keycount = int(float(line.split(":", 1)[1]))
    notes, in_hit = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("["):
            in_hit = s == "[HitObjects]"
            continue
        if not in_hit or not s or s.startswith("//"):
            continue
        f = s.split(",")
        if len(f) < 5:
            continue
        x, t, typ = int(f[0]), int(f[2]), int(f[3])
        col = min(keycount - 1, max(0, int(x * keycount / 512)))
        notes.append((t / 1000.0, col, bool(typ & 128)))
    notes.sort(key=lambda n: n[0])
    return notes, keycount


def _ms_to_scaled_nps(ms: float) -> float:
    return 1000.0 / ms * FINALSCALER


def build_chart(notes, keycount=4, rate=1.0, name="", interval=INTERVAL) -> ChartData:
    if not notes:
        raise ValueError("empty chart")
    t = np.array([n[0] for n in notes], dtype=float) / rate
    c = np.array([n[1] for n in notes], dtype=int)
    n_itv = int(np.floor(t.max() / interval)) + 1
    itv = np.minimum((t / interval).astype(int), n_itv - 1)
    half = keycount // 2
    hand = (c >= half).astype(int)                 # 0 = 左手, 1 = 右手

    pts = np.zeros((2, n_itv))
    for i in range(n_itv):
        m = itv == i
        for h in (0, 1):
            pts[h, i] = 2.0 * np.count_nonzero(hand[m] == h)

    diffs = np.zeros((6, 2, n_itv))
    for i in range(n_itv):
        m = itv == i
        tt, cc = t[m], c[m]
        if tt.size == 0:
            continue
        order = np.argsort(tt, kind="stable")
        tt, cc = tt[order], cc[order]

        # 全局行统计（agnostic 层）
        rows = {}
        for ti, ci in zip(tt, cc):
            rows.setdefault(round(float(ti), 6), []).append(int(ci))
        sizes = [len(v) for v in rows.values()]
        taps = sum(sizes)
        singles = sum(s for s in sizes if s == 1)
        jumps = sum(s for s in sizes if s == 2)
        hands = sum(s for s in sizes if s == 3)
        chord_taps = sum(s for s in sizes if s > 1)
        # 全局 jack 计数（相邻行共享列）
        row_list = [set(v) for _, v in sorted(rows.items())]
        jacks = sum(len(a & b) for a, b in zip(row_list, row_list[1:]))

        for h in (0, 1):
            hm = (cc >= half).astype(int) == h
            n_h = int(np.count_nonzero(hm))
            if n_h == 0:
                continue
            nps = n_h * FINALSCALER * 1.6

            # 同列 jack 的最小间隔（本区间内）
            last = {}
            min_jack = np.inf
            for ti, ci in zip(tt[hm], cc[hm]):
                ci = int(ci)
                if ci in last:
                    min_jack = min(min_jack, (ti - last[ci]) * 1000.0)
                last[ci] = ti
            d_jack = _ms_to_scaled_nps(min_jack) * 1.01 if np.isfinite(min_jack) else 0.0

            # 本手间隔的变异系数（tech 用）
            gaps = np.diff(tt[hm]) * 1000.0
            cv = float(np.std(gaps) / np.mean(gaps)) if gaps.size >= 2 and np.mean(gaps) > 0 else 0.0

            denom = max(taps - 1, 1)
            f_stream = np.clip(np.sqrt((singles + 1) / denom * 1.41) *
                               np.clip(4 - jacks, 0.5, 1.0), 0.6, 1.0)
            f_js = np.clip(np.sqrt((jumps + 1) / denom * 2.714), 0.6, 1.1)
            f_hs = np.clip(np.sqrt(0.4 + (hands + 1) / denom * 5.571), 0.6, 1.1)
            # CJ 的第二个因子：真正的 chordjack（共享列的连续和弦），否则压到 0.625
            f_cj = np.clip(np.sqrt((chord_taps + 1) / denom * 5.428) *
                           np.clip(jacks - 2, 0.625, 1.0), 0.6, 1.0)
            f_tech = np.clip(1.0 / np.clip(0.5 + cv, 0.5, 1.5), 0.5, 1.0)

            diffs[:, h, i] = [
                nps * 0.91 * f_stream,
                nps * 0.75 * f_js,
                nps * 0.77 * f_hs,
                d_jack,
                nps * 1.06 * f_cj,
                nps * 1.06 * f_tech,
            ]

    meta = {"keycount": int(keycount), "notes": int(len(notes)),
            "source": MODEL_VERSION, "interval": interval}
    return ChartData(name or "chart", float(rate), float(interval), pts, diffs,
                     MODEL_VERSION, meta)


def load_chart(path, rate=1.0, cache_dir=".minacalc_cache",
               backend="auto", goal=0.93, itv_exe=None) -> ChartData:
    """读 .osu -> ChartData。

    backend:
      "itv"    —— 走 C++ 导出（Calc::base_adj_diff，真实 MSD 难度），推荐
      "approx" —— 用本文件的简化难度模型
      "auto"   —— 有 exporter 就用 itv，否则退回 approx
    """
    p = Path(path)
    if backend not in ("itv", "approx", "auto"):
        raise ValueError(f"unknown backend: {backend}")
    if backend in ("itv", "auto"):
        from .io import DEFAULT_EXE, load_chart_itv
        exe = Path(itv_exe) if itv_exe else DEFAULT_EXE
        if exe.exists():
            return load_chart_itv(p, rate=rate, goal=goal, exe=exe,
                                  cache_dir=cache_dir)
        if backend == "itv":
            raise FileNotFoundError(f"exporter not built: {exe}")
    key = hashlib.sha1(p.read_bytes()).hexdigest()[:16] + f"_{rate:.3f}_{MODEL_VERSION}"
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{key}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"].item()))
        return ChartData(p.stem, meta["rate"], meta["interval"], z["pts"], z["diffs"],
                         meta["source"], meta)
    notes, kc = parse_osu(p)
    ch = build_chart(notes, kc, rate, name=p.stem)
    ch.meta["rate"] = float(rate)
    np.savez_compressed(f, pts=ch.pts, diffs=ch.diffs,
                        meta=np.array(json.dumps(ch.meta)))
    return ch


def save_chart(ch: ChartData, path) -> None:
    """把任意来源的 ChartData 落盘（例如真实 MSD 导出）。"""
    np.savez_compressed(path, pts=ch.pts, diffs=ch.diffs,
                        meta=np.array(json.dumps({**ch.meta, "rate": ch.rate,
                                                  "interval": ch.interval,
                                                  "source": ch.source})))


def load_chart_npz(path, name=None) -> ChartData:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"].item()))
    return ChartData(name or Path(path).stem, meta.get("rate", 1.0),
                     meta.get("interval", INTERVAL), z["pts"], z["diffs"],
                     meta.get("source", "external"), meta)
