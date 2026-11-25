"""Audit .osu event attributes that a richer data2vec front-end could encode.

Why
---
The route-B data2vec front-end (data2vec_main/d2v_main/charts.py) reads only
note onset + hold state from .osu.  The plan for structured attribute input
(hold release, BPM / beat phase / effective SV / kiai, hitsound / sample set /
volume) has to start with an audit of what is actually present in the corpus:

  * coverage: how many charts carry each attribute class at all,
  * value distributions: how the fields must be normalised / grouped,
  * resolution rules: per-note volume 0 means "inherit from timing point",
    hitSample normalSet 0 means "auto", so raw values are not the values a
    model should see,
  * parser gaps: dcr/cm3p_offline.py does not currently expose hitsound,
    volume, sample set, timing-point volume, or the effects bitmask; it tests
    kiai with a string equality on the effects column, which silently misses
    effects == 9 (kiai + omit barline).

This script is self-contained: it re-reads the raw lines instead of changing
the shared parser, and builds the canonical CM3P timing/SV/kiai groups through
dcr.cm3p_offline so the audit measures exactly what CM3P would see.

Outputs
-------
  results/d2v_attr_audit/summary.json          aggregates + group-leakage stats
  results/d2v_attr_audit/summary.md            human-readable digest
  data/processed/d2v_attr_audit_per_chart.parquet   per-chart table

Usage
-----
  cd data2vec_main
  python scripts/audit_attributes.py                       # full cache
  python scripts/audit_attributes.py --sample 300 --workers 8
  python scripts/audit_attributes.py --ids-file splits/holdout/pretrain_ids.txt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dcr.cm3p_offline import (  # noqa: E402
    HitObject,
    OsuMania,
    TimingPoint,
    _kv,
    _sections,
    kiai_groups,
    sv_groups,
)

REPO = _ROOT
DEFAULT_OSU = REPO / "data" / "osu_cache"
DEFAULT_META = [
    REPO / "data" / "processed" / "beatmap_meta_1k.csv",
    REPO / "data" / "processed" / "beatmap_meta_10k.csv",
    REPO / "data" / "processed" / "beatmap_meta_random.csv",
]
OUT_JSON = REPO / "results" / "d2v_attr_audit" / "summary.json"
OUT_MD = REPO / "results" / "d2v_attr_audit" / "summary.md"
OUT_PARQUET = REPO / "data" / "processed" / "d2v_attr_audit_per_chart.parquet"

# Hitsound bitmask (stable .osu): 1 normal, 2 whistle, 4 finish, 8 clap.
# Many files write 0 instead of 1 for a plain normal hit; treat 0 and 1 alike.
HS_BITS = {"normal": 1, "whistle": 2, "finish": 4, "clap": 8}
# Gaussian onset kernel at 100 Hz, sigma_ms=8 -> radius 3 frames -> 7 nonzero frames
SPIKE_FRAMES = 7


# --------------------------------------------------------------------- parse
def _int(x: str, default: int = 0) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def _float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def parse_raw(path: Path) -> dict:
    """Single-pass raw parse: everything charts.py currently discards + notes."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    sec = _sections(text)

    tps: list[dict] = []
    for line in sec.get("TimingPoints", []):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        f = [x.strip() for x in line.split(",")]
        if len(f) < 2:
            continue
        mpb = _float(f[1])
        # field 7 (0-based 6) is uninherited; keep the CM3P rule as fallback
        uninherited = (f[6] not in ("0", "0.0", "false", "False")) if len(f) >= 7 else mpb > 0
        effects = _int(f[7]) if len(f) > 7 else 0
        tps.append({
            "off": _float(f[0]),
            "mpb": mpb,
            "meter": max(1, _int(f[2])) if len(f) > 2 else 4,
            "sample_set": _int(f[3]) if len(f) > 3 else 0,
            "sample_index": _int(f[4]) if len(f) > 4 else 0,
            "volume": _int(f[5]) if len(f) > 5 else 100,
            "uninherited": uninherited,
            "effects": effects,
            "kiai": bool(effects & 1),
        })

    notes: list[dict] = []
    for line in sec.get("HitObjects", []):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        f = [x.strip() for x in line.split(",")]
        if len(f) < 4:
            continue
        t = _int(f[2])
        flags = _int(f[3])
        if flags & 128:
            kind = "hold"
        elif flags & 1:
            kind = "circle"
        else:
            kind = "other"          # slider/spinner: illegal in mania, count separately
        end = None
        hs = _int(f[4]) if len(f) > 4 else 0
        sample = f[5].split(":") if len(f) > 5 and f[5] else []
        if kind == "hold":
            # field 5 = endTime:hitSample
            end = _int(sample[0]) if sample else t
            sample = sample[1:]
        ss = _int(sample[0]) if len(sample) > 0 else 0
        sset_idx = _int(sample[2]) if len(sample) > 2 else 0
        vol = _int(sample[3]) if len(sample) > 3 else 0
        notes.append({"t": t, "flags": flags, "kind": kind, "end": end,
                      "hitsound": hs, "sample_set": ss, "sample_index": sset_idx,
                      "volume_raw": vol})
    general = sec.get("General", [])
    difficulty = sec.get("Difficulty", [])
    return {"tps": tps, "notes": notes,
            "mode": _int(_kv(general, "Mode") or "0"),
            "circle_size": _int(_kv(difficulty, "CircleSize") or "4"),
            "n_lines_tp": len(sec.get("TimingPoints", [])),
            "n_lines_obj": len(sec.get("HitObjects", []))}


def _bisect_tp(offsets: np.ndarray, t: float) -> int:
    return int(np.searchsorted(offsets, t, side="right")) - 1


def audit_one(path_str: str) -> dict | None:
    path = Path(path_str)
    cid = int(path.stem)
    try:
        raw = parse_raw(path)
    except OSError:
        return None
    notes = [n for n in raw["notes"] if n["kind"] in ("circle", "hold")]
    tps = raw["tps"]
    if not notes:
        return {"beatmap_id": cid, "error": "no_notes", "n_notes": 0}

    # canonical CM3P objects (for median_mpb / sv_groups / kiai_groups)
    canon_tps = [TimingPoint(tp["off"], tp["mpb"], tp["meter"],
                             tp["uninherited"], tp["kiai"]) for tp in tps]
    osu = OsuMania(path=path, mode=3, circle_size=4, timing=canon_tps,
                   objects=[HitObject(n["t"], n["kind"], 0, n["end"]) for n in notes],
                   n_unsupported=0,
                   _red=[tp for tp in canon_tps if tp.uninherited])
    osu._red_offsets = np.asarray([tp.offset_ms for tp in osu._red], dtype=np.float64)

    t_sorted = np.asarray([n["t"] for n in notes], dtype=np.float64)
    ends = np.asarray([n["end"] if n["end"] is not None else n["t"] for n in notes],
                      dtype=np.float64)
    t0 = float(min(t_sorted.min(), min((tp["off"] for tp in tps), default=t_sorted.min())))
    t1 = float(max(ends.max(), max((tp["off"] for tp in tps), default=ends.max())))
    span_s = max(1e-3, (t1 - t0) / 1000.0)
    n_frames_est = max(1, int(round((ends.max() - t_sorted.min()) / 1000.0 * 100)))

    # --- timing / BPM
    red = [tp for tp in tps if tp["uninherited"] and tp["mpb"] > 0]
    bpms = [60_000.0 / tp["mpb"] for tp in red]
    n_bpm_values = len({round(b, 3) for b in bpms})
    bpm_p50 = float(np.median(bpms)) if bpms else float("nan")
    n_red = len(red)
    n_inherited = sum(1 for tp in tps if not tp["uninherited"])
    offs = np.asarray([tp["off"] for tp in tps], dtype=np.float64)
    notes_before_red = int(np.sum(t_sorted < offs[0])) if len(offs) else 0

    # --- effective SV (CM3P definition: normalised by median mpb)
    try:
        sv = sv_groups(osu, normalized=True, add_snapping=False)
        n_sv = len(sv)
        sv_vals = [float(g.scroll) for g in sv if g.scroll is not None]
    except Exception:
        sv, n_sv, sv_vals = [], 0, []
    sv_arr = np.asarray([v for v in sv_vals if math.isfinite(v)], dtype=np.float64)
    if sv_arr.size == 0:
        sv_arr = np.asarray([1.0])
    frac_sv_off1 = 0.0
    if sv:
        for i, g in enumerate(sv):
            a = float(g.time)
            b = float(sv[i + 1].time) if i + 1 < len(sv) else t1
            if b > a and isinstance(g.scroll, float) and abs(g.scroll - 1.0) > 0.05:
                frac_sv_off1 += (min(b, t1) - max(a, t0))
        frac_sv_off1 = max(0.0, min(1.0, frac_sv_off1 / max(1e-3, t1 - t0)))

    # --- kiai
    try:
        kiai = kiai_groups(osu, add_snapping=False)
    except Exception:
        kiai = []
    kiai_sec = 0.0
    for i, g in enumerate(kiai):
        a = float(g.time)
        b = float(kiai[i + 1].time) if i + 1 < len(kiai) else t1
        if b > a:
            kiai_sec += (min(b, t1) - max(a, t0)) / 1000.0
    kiai_frac = max(0.0, min(1.0, kiai_sec / span_s))
    n_kiai = len(kiai)

    # --- holds
    hold_ms = np.asarray([(n["end"] - n["t"]) for n in notes
                          if n["kind"] == "hold" and n["end"] is not None],
                         dtype=np.float64)
    n_holds = len(hold_ms)
    n_taps = len(notes) - n_holds
    hold_p50 = float(np.percentile(hold_ms, 50)) if n_holds else float("nan")
    hold_p95 = float(np.percentile(hold_ms, 95)) if n_holds else float("nan")
    hold_gt1s = float(np.mean(hold_ms > 1000)) if n_holds else 0.0
    bad_hold_end = int(np.sum(hold_ms <= 0)) if n_holds else 0

    # --- hitsound classes (bits; 0 and 1 both = plain normal)
    n = len(notes)
    hs = np.asarray([x["hitsound"] for x in notes], dtype=np.int64)
    hs_normal = (hs == 0) | ((hs & 1) == 1)
    hs_whistle = (hs & 2) == 2
    hs_finish = (hs & 4) == 4
    hs_clap = (hs & 8) == 8
    hs_ext = hs_whistle | hs_finish | hs_clap
    n_hs_combos = len(set(int(v) for v in hs))

    # --- sample set / volume resolution ("0 = inherit")
    resolved_ss = np.empty(n, dtype=np.int64)
    resolved_vol = np.empty(n, dtype=np.int64)
    for i, note in enumerate(notes):
        j = _bisect_tp(offs, float(note["t"]))
        tp = tps[j] if 0 <= j < len(tps) else {"sample_set": 0, "volume": 100}
        resolved_ss[i] = note["sample_set"] if note["sample_set"] > 0 else tp["sample_set"]
        resolved_vol[i] = note["volume_raw"] if note["volume_raw"] > 0 else tp["volume"]
    ss_raw = np.asarray([x["sample_set"] for x in notes], dtype=np.int64)
    vol_raw = np.asarray([x["volume_raw"] for x in notes], dtype=np.int64)
    tp_vol = (np.asarray([tp["volume"] for tp in tps], dtype=np.float64)
              if tps else np.asarray([100.0]))

    # --- parser-gap counters
    n_effects_noncanonical = sum(1 for tp in tps if tp["kiai"] and tp["effects"] != 1)
    n_tp_vol0 = sum(1 for tp in tps if tp["volume"] == 0)

    def frac_spikes(k: int) -> float:
        return min(1.0, (k * SPIKE_FRAMES) / n_frames_est)

    ss_distinct = int(len(set(int(v) for v in resolved_ss)))
    vol_distinct = int(len(set(int(v) for v in resolved_vol)))
    tp_vol_distinct = int(len(set(int(v) for v in tp_vol)))
    return {
        "beatmap_id": cid,
        "mode": raw["mode"], "circle_size": raw["circle_size"],
        "n_notes": n, "n_taps": n_taps, "n_holds": n_holds,
        "span_s": span_s, "notes_per_sec": n / span_s,
        "n_tp": len(tps), "n_red": n_red, "n_inherited": n_inherited,
        "n_bpm_values": n_bpm_values, "bpm_p50": bpm_p50,
        "bpm_min": float(min(bpms)) if bpms else float("nan"),
        "bpm_max": float(max(bpms)) if bpms else float("nan"),
        "n_bpm_changes": max(0, n_red - 1),
        "notes_before_first_red": notes_before_red,
        "has_red": int(n_red > 0),
        "n_sv_changes": n_sv, "sv_changes_per_10s": n_sv / max(1e-3, span_s) * 10.0,
        "sv_min": float(sv_arr.min()), "sv_max": float(sv_arr.max()),
        "frac_time_sv_off1": frac_sv_off1,
        "has_sv_change": int(n_sv > 1),
        "n_kiai": n_kiai, "kiai_sec": kiai_sec, "kiai_frac": kiai_frac,
        "has_kiai": int(n_kiai > 0),
        "hold_ms_p50": hold_p50, "hold_ms_p95": hold_p95,
        "hold_frac_gt1s": hold_gt1s,
        "hs_normal_frac": float(hs_normal.mean()),
        "hs_whistle_frac": float(hs_whistle.mean()),
        "hs_finish_frac": float(hs_finish.mean()),
        "hs_clap_frac": float(hs_clap.mean()),
        "hs_any_ext_frac": float(hs_ext.mean()),
        "has_nonnormal_hitsound": int(hs_ext.any()),
        "hs_n_combos": n_hs_combos,
        "ss_auto_frac": float((ss_raw == 0).mean()),
        "ss_normal_frac": float((ss_raw == 1).mean()),
        "ss_soft_frac": float((ss_raw == 2).mean()),
        "ss_drum_frac": float((ss_raw == 3).mean()),
        "has_pernote_sample_set": int((ss_raw > 0).any()),
        "ss_resolved_auto_frac": float((resolved_ss == 0).mean()),
        "has_resolved_nonauto_ss": int((resolved_ss > 0).any()),
        "has_multi_resolved_ss": int(ss_distinct > 1),
        "ss_resolved_n_distinct": ss_distinct,
        "vol_inherit_frac": float((vol_raw == 0).mean()),
        "has_pernote_volume": int((vol_raw > 0).any()),
        "vol_resolved_p50": float(np.percentile(resolved_vol, 50)),
        "vol_resolved_p10": float(np.percentile(resolved_vol, 10)),
        "vol_resolved_p90": float(np.percentile(resolved_vol, 90)),
        "vol_resolved_distinct": vol_distinct,
        "has_multi_volume": int(vol_distinct > 1),
        "has_tp_volume_change": int(tp_vol_distinct > 1),
        "tp_vol_distinct": tp_vol_distinct,
        "vol_resolved_min": float(resolved_vol.min()),
        "vol_resolved_max": float(resolved_vol.max()),
        "tp_vol_p50": float(np.percentile(tp_vol, 50)),
        "tp_vol_min": float(tp_vol.min()), "tp_vol_max": float(tp_vol.max()),
        "n_effects_noncanonical": n_effects_noncanonical,
        "n_tp_volume0": n_tp_vol0,
        "n_notes_volume_raw0": int(np.sum(vol_raw == 0)),
        "est_release_frac": frac_spikes(n_holds),
        "est_kiai_frac": kiai_frac,
        "est_sv_change_frac": frac_spikes(n_sv),
        "est_hs_ext_frac": frac_spikes(int(hs_ext.sum())),
        "est_volume_frac": frac_spikes(n),
        "n_unsupported_objects": sum(1 for x in raw["notes"] if x["kind"] == "other"),
        "bad_hold_end": bad_hold_end,
    }


# ----------------------------------------------------------------- grouping
def _load_meta():
    import pandas as pd
    frames = []
    for p in DEFAULT_META:
        if p.exists():
            frames.append(pd.read_csv(p, usecols=["beatmap_id", "beatmapset_id", "mapper_id"]))
    if not frames:
        return None
    return pd.concat(frames).drop_duplicates("beatmap_id")


def group_stats(df, meta, eval_ids: list[int], pre_ids: list[int]) -> dict:
    import itertools
    out: dict = {}
    if meta is None:
        return {"available": False}
    m = meta.set_index("beatmap_id")
    have = [int(x) for x in df["beatmap_id"] if int(x) in m.index]
    sub = m.loc[have]
    vc = sub.groupby("beatmapset_id").size()
    out["n_charts_with_meta"] = int(len(sub))
    out["n_beatsets"] = int(vc.size)
    out["n_mappers"] = int(sub["mapper_id"].nunique())
    out["charts_in_multichart_sets"] = int(vc[vc > 1].sum())
    out["frac_charts_in_multichart_sets"] = float(vc[vc > 1].sum() / max(1, len(sub)))
    out["charts_per_set_p50"] = float(vc.median())
    out["charts_per_set_p95"] = float(vc.quantile(0.95))
    out["charts_per_set_max"] = int(vc.max())

    ev = [i for i in eval_ids if i in m.index]
    pr = [i for i in pre_ids if i in m.index]
    ev_sets = set(m.loc[ev, "beatmapset_id"]) if ev else set()
    pr_sets = set(m.loc[pr, "beatmapset_id"]) if pr else set()
    overlap = ev_sets & pr_sets
    out["holdout_eval_charts"] = len(ev)
    out["holdout_eval_beatsets"] = len(ev_sets)
    out["eval_beatsets_in_pretrain"] = len(overlap)
    out["eval_beatsets_in_pretrain_frac"] = len(overlap) / max(1, len(ev_sets))
    if ev:
        ev_rows = m.loc[ev]
        in_pre = ev_rows["beatmapset_id"].isin(overlap)
        out["eval_charts_in_pretrain_beatsets"] = int(in_pre.sum())
        out["eval_charts_in_pretrain_beatsets_frac"] = float(in_pre.mean())
        out["eval_mappers"] = int(ev_rows["mapper_id"].nunique())
    if pr:
        out["pretrain_mappers"] = int(m.loc[pr, "mapper_id"].nunique())
        out["pretrain_beatsets"] = int(m.loc[pr, "beatmapset_id"].nunique())

    # within-beatmapset signature sharing over the audited charts: BPM grid,
    # effective-SV change count and kiai count are the leak-prone fields.
    by_set: dict[int, list[dict]] = {}
    for _, r in df.iterrows():
        cid = int(r["beatmap_id"])
        if cid not in m.index:
            continue
        by_set.setdefault(int(m.loc[cid, "beatmapset_id"]), []).append({
            "n_red": int(r["n_red"]), "n_sv": int(r["n_sv_changes"]),
            "n_kiai": int(r["n_kiai"]), "bpm_p50": round(float(r["bpm_p50"]), 3),
            "has_nonnormal_hitsound": int(r["has_nonnormal_hitsound"]),
        })
    pairs = same_bpm = same_sv = same_kiai = same_full = same_hs = 0
    for ids in by_set.values():
        if len(ids) < 2:
            continue
        for a, b in itertools.combinations(ids, 2):
            pairs += 1
            same_bpm += a["bpm_p50"] == b["bpm_p50"] and a["n_red"] == b["n_red"]
            same_sv += a["n_sv"] == b["n_sv"]
            same_kiai += a["n_kiai"] == b["n_kiai"]
            same_hs += a["has_nonnormal_hitsound"] == b["has_nonnormal_hitsound"]
            same_full += ((a["n_red"], a["n_sv"], a["n_kiai"])
                          == (b["n_red"], b["n_sv"], b["n_kiai"]))
    if pairs:
        out["within_set_pairs"] = pairs
        out["within_set_same_bpm_grid_frac"] = same_bpm / pairs
        out["within_set_same_n_sv_frac"] = same_sv / pairs
        out["within_set_same_n_kiai_frac"] = same_kiai / pairs
        out["within_set_same_hitsound_flag_frac"] = same_hs / pairs
        out["within_set_same_timing_signature_frac"] = same_full / pairs
    return out


# ------------------------------------------------------------------ summary
def _dist(values, bins=None) -> dict:
    a = np.asarray([v for v in values if v == v], dtype=np.float64)
    if a.size == 0:
        return {}
    d = {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
         "p05": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
         "min": float(a.min()), "max": float(a.max())}
    if bins:
        hist, edges = np.histogram(a, bins=bins)
        d["hist"] = {f"{edges[i]:.4g}..{edges[i+1]:.4g}": int(hist[i])
                     for i in range(len(hist)) if hist[i]}
    return d


def summarize(df) -> dict:
    cols = list(df.columns)
    out: dict = {"n_charts": int(len(df))}
    out["n_charts_with_error"] = int(df["error"].notna().sum()) if "error" in cols else 0
    out["coverage"] = {}
    for key in ["has_kiai", "has_sv_change", "has_nonnormal_hitsound",
                "has_pernote_sample_set", "has_pernote_volume",
                "has_multi_resolved_ss", "has_multi_volume",
                "has_tp_volume_change", "has_red"]:
        if key in cols:
            out["coverage"][key] = float(df[key].fillna(0).astype(bool).mean())
    out["distributions"] = {}
    for key in ["n_notes", "span_s", "notes_per_sec", "n_red", "n_inherited", "n_bpm_values",
                "bpm_p50", "n_sv_changes", "sv_changes_per_10s", "frac_time_sv_off1",
                "n_kiai", "kiai_frac", "hold_ms_p50", "hold_ms_p95", "hold_frac_gt1s",
                "hs_normal_frac", "hs_whistle_frac", "hs_finish_frac", "hs_clap_frac",
                "hs_any_ext_frac", "hs_n_combos", "ss_auto_frac", "ss_drum_frac",
                "vol_inherit_frac", "vol_resolved_p50", "vol_resolved_distinct",
                "ss_resolved_n_distinct", "tp_vol_p50", "tp_vol_distinct",
                "est_release_frac", "est_hs_ext_frac", "est_sv_change_frac"]:
        if key in cols:
            out["distributions"][key] = _dist(df[key])
    for key in ["hs_normal_frac", "hs_whistle_frac", "hs_finish_frac", "hs_clap_frac",
                "hs_any_ext_frac", "ss_auto_frac", "vol_inherit_frac"]:
        if key in cols:
            out["distributions"][key]["note_weighted_mean"] = float(
                (df[key] * df["n_notes"]).sum() / max(1, df["n_notes"].sum()))
    out["parser_gaps"] = {}
    if "n_effects_noncanonical" in cols:
        out["parser_gaps"]["charts_with_kiai_effects_not_1"] = int(
            (df["n_effects_noncanonical"] > 0).sum())
        out["parser_gaps"]["notes_in_those_charts"] = int(
            df.loc[df["n_effects_noncanonical"] > 0, "n_notes"].sum())
    if "n_tp_volume0" in cols:
        out["parser_gaps"]["charts_with_tp_volume0"] = int((df["n_tp_volume0"] > 0).sum())
    if "n_notes_volume_raw0" in cols:
        out["parser_gaps"]["note_frac_volume_raw0"] = float(
            df["n_notes_volume_raw0"].sum() / max(1, df["n_notes"].sum()))
    out["totals"] = {"notes": int(df["n_notes"].sum()),
                     "holds": int(df["n_holds"].sum()),
                     "charts": int(len(df)),
                     "note_hours": float(df["span_s"].sum() / 3600.0),
                     "unsupported_objects": int(df["n_unsupported_objects"].sum())
                     if "n_unsupported_objects" in cols else 0,
                     "bad_hold_end": int(df["bad_hold_end"].sum())
                     if "bad_hold_end" in cols else 0}
    if "mode" in cols:
        out["mode_counts"] = {int(k): int(v) for k, v in df["mode"].value_counts().items()}
        out["circle_size_counts"] = {int(k): int(v) for k, v in
                                     df["circle_size"].value_counts().items()}
    return out


def render_md(summary: dict, group: dict) -> str:
    c = summary["coverage"]
    d = summary["distributions"]
    L: list[str] = []
    A = L.append
    A("# D2V 结构化属性输入：训练数据审计（自动生成）")
    A("")
    A("由 data2vec_main/scripts/audit_attributes.py 生成；汇总 "
      "results/d2v_attr_audit/summary.json，逐图表在 "
      "data/processed/d2v_attr_audit_per_chart.parquet（不入库）。")
    A("")
    A("* 谱面 **" + str(summary["n_charts"]) + "**，音符 "
      + format(summary["totals"]["notes"], ",") + "，hold "
      + format(summary["totals"]["holds"], ",") + "，谱面时长合计 "
      + f"{summary['totals']['note_hours']:.1f} h。")
    A("* 解析失败/空图：" + str(summary.get("n_charts_with_error", 0)) + "。")
    A("")
    A("## 1. 覆盖率（按谱面）")
    A("")
    A("| 属性 | 覆盖率 |")
    A("|---|---:|")
    label = {"has_kiai": "任意 kiai", "has_sv_change": "任意有效 SV 变化",
             "has_nonnormal_hitsound": "任意非普通 hitsound",
             "has_pernote_volume": "任意逐音符 volume 覆写",
             "has_pernote_sample_set": "任意逐音符 sample set 覆写",
             "has_multi_resolved_ss": "解析后 sample set 取值 >1",
             "has_multi_volume": "解析后 volume 取值 >1",
             "has_tp_volume_change": "timing point volume 有过变化",
             "has_red": "有红线 BPM"}
    for k, v in c.items():
        A("| " + label.get(k, k) + " | " + f"{100*v:.1f}% |")
    A("")
    A("## 2. 取值分布")
    A("")
    A("| 字段 | p05 | p50 | p95 | mean |")
    A("|---|---:|---:|---:|---:|")
    for k, s in d.items():
        if not s:
            continue
        A(f"| {k} | {s.get('p05', float('nan')):.4g} | {s.get('p50', float('nan')):.4g} "
          f"| {s.get('p95', float('nan')):.4g} | {s.get('mean', float('nan')):.4g} |")
    A("")
    if summary.get("parser_gaps"):
        A("## 3. 解析器缺口 / 边界")
        A("")
        for k, v in summary["parser_gaps"].items():
            A("* " + k + ": **" + str(v) + "**")
        A("")
    if group and group.get("available", True):
        G = group
        A("## 4. beatmapset / mapper 分组（泄漏证据）")
        A("")
        A(f"* 有 meta 的谱面 {G.get('n_charts_with_meta', 0)}，beatmapset "
          f"{G.get('n_beatsets', 0)}，mapper {G.get('n_mappers', 0)}。")
        A(f"* 多难度 set 内谱面占比 **{100*G.get('frac_charts_in_multichart_sets', 0):.1f}%**"
          f"（每 set 中位 {G.get('charts_per_set_p50', 0):.0f} 张，max "
          f"{G.get('charts_per_set_max', 0)}）。")
        A(f"* D2V holdout：eval 图 {G.get('holdout_eval_charts', 0)}，set "
          f"{G.get('holdout_eval_beatsets', 0)}；其中 "
          f"**{100*G.get('eval_beatsets_in_pretrain_frac', 0):.1f}% 的 set 也出现在预训练池**，"
          f"对应 **{100*G.get('eval_charts_in_pretrain_beatsets_frac', 0):.1f}% 的 eval 图**。")
        A(f"* 同一 set 内不同难度对（{G.get('within_set_pairs', 0)} 对）共享：BPM 网格 "
          f"{100*G.get('within_set_same_bpm_grid_frac', 0):.1f}%、SV 变化数 "
          f"{100*G.get('within_set_same_n_sv_frac', 0):.1f}%、kiai 数 "
          f"{100*G.get('within_set_same_n_kiai_frac', 0):.1f}%、完整 timing 签名 "
          f"{100*G.get('within_set_same_timing_signature_frac', 0):.1f}%。")
        A("")
    A("## 5. 读法")
    A("")
    A("* has_sv_change 用有效组数 > 1 判定；初始 SV 组恒存在，单 SV 谱面不算变化。")
    A("* volume / sample set 的 *_frac 是原始值（0 = inherit）；vol_resolved_* / "
      "ss_resolved_* 是解析后的值。")
    A("* hitsound 原始值 0 与 1 都表示普通命中；whistle/finish/clap 是 bit 1/2/3。")
    A("* n_effects_noncanonical = kiai 为真但 effects != 1（例如 9 = kiai + omit barline）；"
      "现有 dcr/cm3p_offline.py 的字符串相等判断会漏掉这些。")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osu-dir", default=str(DEFAULT_OSU))
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out-json", default=str(OUT_JSON))
    ap.add_argument("--out-md", default=str(OUT_MD))
    ap.add_argument("--out-parquet", default=str(OUT_PARQUET))
    ap.add_argument("--eval-ids", default=str(REPO / "data2vec_main/splits/holdout/eval_ids.txt"))
    ap.add_argument("--pretrain-ids", default=str(REPO / "data2vec_main/splits/holdout/pretrain_ids.txt"))
    ap.add_argument("--no-groups", action="store_true")
    args = ap.parse_args(argv)

    import pandas as pd
    osu_dir = Path(args.osu_dir)
    paths = sorted(p for p in osu_dir.glob("*.osu") if p.stem.isdigit())
    if args.ids_file:
        keep = {int(x) for x in Path(args.ids_file).read_text().split() if x.strip()}
        paths = [p for p in paths if int(p.stem) in keep]
    if args.sample:
        step = max(1, len(paths) // args.sample)
        paths = paths[::step][: args.sample]
    if not paths:
        raise SystemExit(f"no .osu under {osu_dir}")
    print(f"[audit] {len(paths)} charts, workers={args.workers}", flush=True)

    t0 = time.time()
    rows: list[dict] = []
    if args.workers <= 1:
        for i, p in enumerate(paths):
            r = audit_one(str(p))
            if r:
                rows.append(r)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(paths)} {time.time()-t0:.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(audit_one, [str(p) for p in paths], chunksize=64)):
                if r:
                    rows.append(r)
                if (i + 1) % 2000 == 0:
                    print(f"  {i+1}/{len(paths)} {time.time()-t0:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    df = df.sort_values("beatmap_id").reset_index(drop=True)
    print(f"[audit] parsed {len(df)} charts in {time.time()-t0:.1f}s", flush=True)

    eval_ids = ([int(x) for x in Path(args.eval_ids).read_text().split()]
                if Path(args.eval_ids).exists() else [])
    pre_ids = ([int(x) for x in Path(args.pretrain_ids).read_text().split()]
               if Path(args.pretrain_ids).exists() else [])
    meta = _load_meta()
    group = {} if args.no_groups else group_stats(df, meta, eval_ids, pre_ids)
    summary = summarize(df)
    summary["group_stats"] = group
    summary["runtime_sec"] = time.time() - t0
    summary["source"] = {"osu_dir": str(osu_dir), "n_paths": len(paths),
                         "sample": args.sample, "ids_file": args.ids_file}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    Path(args.out_md).write_text(render_md(summary, group))
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)
    print(f"[audit] wrote {args.out_json}", flush=True)
    print(f"[audit] wrote {args.out_md}", flush=True)
    print(f"[audit] wrote {args.out_parquet}", flush=True)
    print(json.dumps({k: summary[k] for k in ("n_charts", "coverage")},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
