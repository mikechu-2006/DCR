"""Prototype: structured attribute channels on top of the 8-channel piano roll.

Scratch code for the plan; not wired into the D2V pipeline.  Validates:
  * per-window cost vs the 8-channel builder,
  * value ranges / finiteness of every proposed channel,
  * exact definitions (BPM, beat phase, effective SV, SV change, kiai,
    hold release).

Usage: python docs/plans/_scratch/proto_attr_roll.py [--n 200]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dcr.cm3p_offline import (  # noqa: E402
    HitObject, OsuMania, TimingPoint, _sections, kiai_groups, sv_groups,
)
from data2vec_main.d2v_main.charts import (  # noqa: E402
    N_CHANNELS, build_window, gaussian_kernel, parse_chart, window_starts,
)

FRAME_HZ = 100
SIGMA_MS = 8.0
WINDOW_SEC = 16.0
GROUPS = {
    "base": ["onset_0", "onset_1", "onset_2", "onset_3", "hold_0", "hold_1", "hold_2", "hold_3"],
    "release": ["release_0", "release_1", "release_2", "release_3"],
    "timing": ["bpm", "beat_sin", "beat_cos", "bar_sin", "bar_cos", "sv", "sv_change", "kiai"],
    "meta": ["hs_whistle", "hs_finish", "hs_clap", "ss_soft", "ss_drum", "ss_auto", "volume"],
}


def _int(x, default=0):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def _float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _sections_raw(path: Path):
    sec = _sections(path.read_text(encoding="utf-8", errors="ignore"))
    tps = []
    for line in sec.get("TimingPoints", []):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        f = [x.strip() for x in line.split(",")]
        if len(f) < 2:
            continue
        mpb = _float(f[1])
        uninherited = (f[6] not in ("0", "0.0", "false", "False")) if len(f) >= 7 else mpb > 0
        effects = _int(f[7]) if len(f) > 7 else 0
        tps.append(dict(off=_float(f[0]), mpb=mpb, meter=max(1, _int(f[2])) if len(f) > 2 else 4,
                        sample_set=_int(f[3]) if len(f) > 3 else 0,
                        volume=_int(f[5]) if len(f) > 5 else 100,
                        uninherited=uninherited, effects=effects, kiai=bool(effects & 1)))
    objects = []
    for line in sec.get("HitObjects", []):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        f = [x.strip() for x in line.split(",")]
        if len(f) < 4:
            continue
        t, flags = _int(f[2]), _int(f[3])
        if flags & 128:
            kind = "hold"
        elif flags & 1:
            kind = "circle"
        else:
            continue
        end = None
        hs = _int(f[4]) if len(f) > 4 else 0
        parts = (f[5].split(":") if len(f) > 5 and f[5] else [])
        if kind == "hold":
            end = _int(parts[0]) if parts else t
            parts = parts[1:]
        objects.append(dict(t=t, kind=kind, end=end, hitsound=hs,
                            sample_set=_int(parts[0]) if parts else 0,
                            volume=_int(parts[3]) if len(parts) > 3 else 0))
    objects.sort(key=lambda o: (o["t"], o["kind"]))
    return tps, objects


def build_timing(path: Path) -> dict:
    """All per-frame timing/meta segments needed by the attribute channels."""
    tps, objects = _sections_raw(path)
    canon = [TimingPoint(tp["off"], tp["mpb"], tp["meter"], tp["uninherited"], tp["kiai"]) for tp in tps]
    osu = OsuMania(path=path, mode=3, circle_size=4, timing=canon,
                   objects=[HitObject(o["t"], o["kind"], 0, o["end"]) for o in objects],
                   n_unsupported=0, _red=[tp for tp in canon if tp.uninherited])
    osu._red_offsets = np.asarray([tp.offset_ms for tp in osu._red], dtype=np.float64)
    red = np.asarray([[tp.offset_ms, tp.ms_per_beat, tp.meter] for tp in canon
                      if tp.uninherited and tp.ms_per_beat > 0],
                     dtype=np.float64).reshape(-1, 3)
    sv = sv_groups(osu, normalized=True, add_snapping=False)
    sv_t = np.asarray([float(g.time) for g in sv], dtype=np.float64)
    sv_v = np.asarray([float(g.scroll) for g in sv if g.scroll is not None], dtype=np.float64)
    kiai = kiai_groups(osu, add_snapping=False)
    kiai_t = np.asarray([float(g.time) for g in kiai], dtype=np.float64)
    # kiai state toggles at each switch; determine state from the TP at that time
    tp_off = np.asarray([tp["off"] for tp in tps], dtype=np.float64)
    tp_kiai = np.asarray([bool(tp["kiai"]) for tp in tps], dtype=bool)
    order = np.argsort(tp_off)
    tp_off, tp_kiai = tp_off[order], tp_kiai[order]
    # volume / sample-set segments for resolution (last TP wins)
    tp_vol = np.asarray([tp["volume"] for tp in tps], dtype=np.float64)[order]
    tp_ss = np.asarray([tp["sample_set"] for tp in tps], dtype=np.int64)[order]
    n_frames = 0
    if objects:
        n_frames = int(max((o["end"] if o["end"] is not None else o["t"]) for o in objects) * FRAME_HZ // 1000) + 1
    return dict(red=red, sv_t=sv_t, sv_v=sv_v, kiai_t=kiai_t,
                tp_off=tp_off, tp_kiai=tp_kiai, tp_vol=tp_vol, tp_ss=tp_ss,
                objects=objects, n_frames=n_frames)


def timing_roll(timing: dict, start: int, window: int, frame_hz: int = FRAME_HZ) -> np.ndarray:
    """(8, window) channels: bpm, beat_sin/cos, bar_sin/cos, sv, sv_change, kiai."""
    tms = (np.arange(start, start + window, dtype=np.float64) / frame_hz) * 1000.0
    out = np.zeros((8, window), dtype=np.float32)
    red = timing["red"]
    if len(red):
        idx = np.searchsorted(red[:, 0], tms, side="right") - 1
        idx = np.clip(idx, 0, len(red) - 1)
        mpb = red[idx, 1]
        off = red[idx, 0]
        meter = np.maximum(1.0, red[idx, 2])
        bpm = 60_000.0 / np.maximum(1e-6, mpb)
        out[0] = np.clip(np.log2(bpm / 120.0) / 2.0, -1.0, 1.0)
        beats = (tms - off) / np.maximum(1e-6, mpb)
        phase = 2.0 * math.pi * beats
        out[1] = np.sin(phase)
        out[2] = np.cos(phase)
        bar_phase = 2.0 * math.pi * (beats / meter)
        out[3] = np.sin(bar_phase)
        out[4] = np.cos(bar_phase)
    sv_t, sv_v = timing["sv_t"], timing["sv_v"]
    if len(sv_t):
        idx = np.clip(np.searchsorted(sv_t, tms, side="right") - 1, 0, len(sv_t) - 1)
        val = np.clip(sv_v[idx], 1e-3, None)
        out[5] = np.clip(np.log2(val) / 2.0, -1.5, 1.5)
        # change pulse: frame index of each segment start inside the window
        pos = np.searchsorted(sv_t, start * 1000.0 / frame_hz, side="left")
        while pos < len(sv_t) and sv_t[pos] < (start + window) * 1000.0 / frame_hz:
            f = int(round(sv_t[pos] * frame_hz / 1000.0)) - start
            if 0 <= f < window:
                out[6, f] = 1.0
            pos += 1
    kiai_t = timing["kiai_t"]
    if len(kiai_t):
        # state at frame t: kiai flag of the last timing point <= t (toggles are TPs)
        idx = np.searchsorted(timing["tp_off"], tms, side="right") - 1
        state = np.where(idx >= 0, timing["tp_kiai"][np.clip(idx, 0, None)], False)
        out[7] = state.astype(np.float32)
    return out


def note_rolls(path: Path, timing: dict, start: int, window: int,
               frame_hz: int = FRAME_HZ, sigma_ms: float = SIGMA_MS) -> tuple[np.ndarray, np.ndarray]:
    """(4, window) release channels and (7, window) metadata channels."""
    rel = np.zeros((4, window), dtype=np.float32)
    meta = np.zeros((7, window), dtype=np.float32)
    kernel = gaussian_kernel(sigma_ms, frame_hz)
    radius = len(kernel) // 2
    t0ms = start * 1000.0 / frame_hz
    t1ms = (start + window) * 1000.0 / frame_hz
    tp_off, tp_vol, tp_ss = timing["tp_off"], timing["tp_vol"], timing["tp_ss"]
    for o in timing["objects"]:
        t_ms = o["t"]
        if not (t0ms - 50.0 <= t_ms < t1ms + 50.0):
            continue
        # resolve per-note volume / sample set (0 = inherit from TP)
        j = int(np.searchsorted(tp_off, t_ms, side="right")) - 1
        vol = o["volume"] if o["volume"] > 0 else (tp_vol[j] if 0 <= j < len(tp_vol) else 100)
        ss = o["sample_set"] if o["sample_set"] > 0 else (tp_ss[j] if 0 <= j < len(tp_ss) else 0)
        f = int(round(t_ms * frame_hz / 1000.0)) - start
        if 0 <= f < window:
            hs = o["hitsound"]
            if hs & 2:
                meta[0, f] = 1.0
            if hs & 4:
                meta[1, f] = 1.0
            if hs & 8:
                meta[2, f] = 1.0
            if ss == 2:
                meta[3, f] = 1.0
            elif ss == 3:
                meta[4, f] = 1.0
            if o["sample_set"] == 0:
                meta[5, f] = 1.0
            meta[6, f] = max(0.0, min(1.0, vol / 100.0))
        if o["kind"] == "hold" and o["end"] is not None:
            e = int(round(o["end"] * frame_hz / 1000.0)) - start
            if -radius <= e < window + radius:
                lo = max(0, e - radius)
                hi = min(window, e + radius + 1)
                if hi > lo:
                    klo = lo - (e - radius)
                    rel[0, lo:hi] += kernel[klo:klo + (hi - lo)]
    return rel, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--osu-dir", default=str(_ROOT / "data/osu_cache"))
    args = ap.parse_args(argv)
    paths = sorted(p for p in Path(args.osu_dir).glob("*.osu") if p.stem.isdigit())[: args.n]
    frame_hz, window = FRAME_HZ, int(WINDOW_SEC * FRAME_HZ)
    t_parse = t_base = t_attr = 0.0
    stats = {k: [] for k in list(GROUPS["timing"]) + list(GROUPS["meta"]) + ["release_sum"]}
    n_windows = 0
    for p in paths:
        t0 = time.time()
        try:
            events, n_frames = parse_chart(str(p), frame_hz)
        except Exception:
            continue
        timing = build_timing(p)
        t_parse += time.time() - t0
        starts = window_starts(timing["n_frames"] or n_frames, window)[:2]
        for start in starts:
            t0 = time.time()
            base = build_window(events, start, window, frame_hz, SIGMA_MS)
            t_base += time.time() - t0
            t0 = time.time()
            rel, meta = note_rolls(p, timing, start, window, frame_hz, SIGMA_MS)
            tim = timing_roll(timing, start, window, frame_hz)
            t_attr += time.time() - t0
            n_windows += 1
            assert base.shape == (N_CHANNELS, window), base.shape
            for i, name in enumerate(GROUPS["timing"]):
                v = tim[i]
                assert np.isfinite(v).all(), name
                stats[name].append(v)
            for i, name in enumerate(GROUPS["meta"]):
                v = meta[i]
                assert np.isfinite(v).all(), name
                stats[name].append(v)
            assert np.isfinite(rel).all()
            stats["release_sum"].append(rel.sum())
    print(f"charts={len(paths)} windows={n_windows}")
    print(f"parse: {t_parse/n_windows*1000:.2f} ms/window (includes timing parse)")
    print(f"base build_window: {t_base/n_windows*1000:.3f} ms/window")
    print(f"attr build (release+meta+timing): {t_attr/n_windows*1000:.3f} ms/window")
    for k, vs in stats.items():
        a = np.concatenate([np.asarray(v).ravel() for v in vs]) if vs else np.asarray([0.0])
        print(f"  {k:12s} mean={a.mean():.4f} std={a.std():.4f} "
              f"min={a.min():.3f} max={a.max():.3f} frac_nonzero={np.mean(a != 0):.4f}")
    # sanity: total channels per arm
    for arm, groups in [("A0", ["base"]), ("A1", ["base", "release"]),
                        ("A2", ["base", "release", "timing"]),
                        ("A3", ["base", "release", "timing", "meta"])]:
        n = sum(len(GROUPS[g]) for g in groups)
        print(f"{arm}: {n} channels ({'+'.join(groups)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
