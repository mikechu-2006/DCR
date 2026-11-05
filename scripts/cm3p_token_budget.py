#!/usr/bin/env python
"""Offline CM3P token budget + crop audit (no transformers, no slider, no GPU).

Answers the T0/T1 questions of the crop plan on the local .osu cache:

    python scripts/cm3p_token_budget.py --osu-dir data/osu_cache --workers 8

    tokens / chart, tokens / note, windows / chart, tokens / window
    (mean / p50 / p95 / max), legacy vs canon vs canon+mania-columns,
    truncation at max_length, holds split by a window boundary, and

    CORE AUDIT: the note/timing group stream must be byte-identical between
    the legacy flags and the three switched-off flags.

The counter is a transcription of external/cm3p (commit 546f2e2).  Run
--verify-cm3p N on a machine that has transformers+slider to compare it with
the real tokenizer on N charts; the comparison is exact when it passes.

Exit code is non-zero with --strict if a core stream changed, a core token was
dropped by truncation, or the real-tokenizer verification failed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcr import cm3p_offline as off  # noqa: E402


# ------------------------------------------------------------------ workers
MAX_SONG_MS = 4 * 3600 * 1000.0          # a 4K mania chart longer than this is corrupt


def _chart_stats(path: str, window_sec: float, min_window_sec: float,
                 max_length: int, audit: bool, add_snapping: bool = True,
                 add_cls_token: bool = True) -> dict:
    try:
        return _chart_stats_impl(path, window_sec, min_window_sec, max_length, audit,
                                 add_snapping, add_cls_token)
    except Exception as exc:                       # one bad .osu must not kill 21,949
        return {"beatmap_id": -1, "path": path, "error": str(exc)}


def _chart_stats_impl(path: str, window_sec: float, min_window_sec: float,
                      max_length: int, audit: bool, add_snapping: bool = True,
                      add_cls_token: bool = True) -> dict:
    osu = off.parse_osu(path)
    length_ms = off.song_length_ms(osu)
    if not np.isfinite(length_ms) or length_ms > MAX_SONG_MS:
        return {"beatmap_id": -1, "path": path,
                "error": f"implausible song length {length_ms / 1000.0:g}s"}
    windows = off.window_bounds(length_ms, window_sec, min_window_sec)
    full = off.chart_groups(osu, hitsounds=True, kiai=True, sv=True, columns=False,
                            add_snapping=add_snapping)
    crop = off.chart_groups(osu, hitsounds=False, kiai=False, sv=False, columns=False,
                            add_snapping=add_snapping)
    lf = off.window_token_lengths(full, windows, add_cls_token=add_cls_token)
    lc = off.window_token_lengths(crop, windows, add_cls_token=add_cls_token)
    cw = off.window_counts(crop, windows, add_cls_token=add_cls_token)
    note_tokens_on = [c["notes"] for c in cw]          # one column token per note event
    lc_on = [a + b for a, b in zip(lc, note_tokens_on)]
    split, near, holds = off.hold_split_stats(osu, window_sec * 1000.0)
    n_note_events = sum(c["notes"] for c in cw)
    n_column_tokens = sum(c["columns"] for c in cw)     # 0 with the upstream bug
    d = {
        "beatmap_id": int(Path(path).stem.split(" ")[0]) if Path(path).stem.split(" ")[0].isdigit() else -1,
        "path": path,
        "n_notes": osu.n_notes,
        "n_note_events": n_note_events,
        "n_unsupported": osu.n_unsupported,
        "n_windows": len(windows),
        "n_columns": n_column_tokens,
        "holds_total": holds,
        "holds_split": split,
        "holds_near": near,
        "tok_full": lf,
        "tok_canon": lc,
        "tok_canon_on": lc_on,
        "n_timing_tokens": sum(c["timing"] for c in cw),
        "n_kiai_tokens": sum(c["kiai"] for c in cw) * 3,
        "n_sv_tokens": sum(c["sv"] for c in cw) * 4,
        "n_hitsound_tokens": sum(c["hitsound"] for c in cw),
        "song_length_s": length_ms / 1000.0,
    }
    if audit:
        a = off.audit_chart(osu, window_sec=window_sec, min_window_sec=min_window_sec,
                            max_length=max_length, columns=False,
                            add_snapping=add_snapping,
                            add_cls_token=add_cls_token)
        d["audit"] = {
            "core_match": a.core_match, "problems": a.problems,
            "n_core": a.n_core, "token_delta": a.token_delta,
            "removed_hitsound": a.removed_hitsound_tokens,
            "removed_kiai": a.removed_kiai_tokens,
            "removed_sv": a.removed_sv_tokens,
            "n_core_dropped": a.n_core_dropped_crop,
        }
    return d


# ---------------------------------------------------------------- reporting
def _dist(values) -> dict:
    return off.pct_summary(values)


def _pad_efficiency(lengths: list[int], batch: int) -> float:
    """Sort windows by length and batch consecutively: sum len / (B * max len)."""
    if not lengths:
        return 0.0
    a = np.sort(np.asarray(lengths, dtype=np.int64))[::-1]
    used = total = 0.0
    for i in range(0, len(a), batch):
        b = a[i:i + batch]
        total += float(b.sum())
        used += float(batch * b.max())
    return total / max(used, 1e-12)


def _collect_dirs(dirs: list[str], ids_file: str | None) -> list[str]:
    want = None
    if ids_file:
        import pandas as pd
        df = pd.read_csv(ids_file, dtype=str)
        cols = {c.strip().lower(): c for c in df.columns}
        key = cols.get("beatmap_id", df.columns[0])
        want = {str(x).strip() for x in df[key].tolist()}
    out = []
    seen = set()
    for d in dirs:
        p = Path(d)
        if p.is_file():
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                out.append(str(p))
            continue
        for q in sorted(p.rglob("*.osu")):
            if want is not None and q.stem.split(" ")[0] not in want:
                continue
            key = str(q.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(str(q))
    return out


def _print_table(title: str, rows: list[tuple[str, dict]]) -> None:
    print(f"\n{title}")
    print(f"  {'quantity':<22s} {'mean':>9s} {'p50':>9s} {'p95':>9s} {'max':>10s}")
    for name, s in rows:
        if not s:
            continue
        print(f"  {name:<22s} {s['mean']:9.2f} {s['p50']:9.2f} {s['p95']:9.2f} {s['max']:10.2f}")


# ------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osu-dir", action="append", default=None,
                    help="directory (or .osu file) to scan; repeatable "
                         "(default data/osu_cache)")
    ap.add_argument("--ids-file", default=None, help="optional beatmap_id manifest")
    ap.add_argument("--limit", type=int, default=0, help="debug: only this many charts")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--window-sec", type=float, default=16.0)
    ap.add_argument("--min-window-sec", type=float, default=1.0)
    ap.add_argument("--max-length", type=int, default=2000,
                    help="the real HF preprocessor config is 2000; 4000 is the "
                         "upstream training yaml, not the published processor")
    ap.add_argument("--add-snapping", default="off", choices=["on", "off"],
                    help="the real HF preprocessor config has add_snapping=False; "
                         "'on' reproduces the upstream training yaml instead")
    ap.add_argument("--add-cls", default="on", choices=["on", "off"],
                    help="the real HF preprocessor config has add_cls_token=True")
    ap.add_argument("--audit", type=int, default=64,
                    help="core-stream audit on the first N charts (0 = off)")
    ap.add_argument("--verify-cm3p", type=int, default=0,
                    help="compare the counter with the real tokenizer on N charts "
                         "(needs transformers + slider)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if the core audit fails or truncation drops core tokens")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dirs = args.osu_dir or ["data/osu_cache"]
    add_snapping = args.add_snapping == "on"
    add_cls = args.add_cls == "on"
    files = _collect_dirs(dirs, args.ids_file)
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit("no .osu files found")
    print(f"scanning {len(files):,} charts with {args.workers} workers "
          f"(window={args.window_sec:g}s, max_length={args.max_length})", flush=True)

    t0 = time.time()
    rows = []
    if args.workers > 1:
        ex = None
        try:
            ex = ProcessPoolExecutor(max_workers=args.workers)
        except (PermissionError, OSError) as exc:   # sandboxes often block /dev/shm
            print(f"  (process pool unavailable: {exc}; falling back to threads)", flush=True)
            from concurrent.futures import ThreadPoolExecutor
            ex = ThreadPoolExecutor(max_workers=args.workers)
        with ex:
            futs = [ex.submit(_chart_stats, p, args.window_sec, args.min_window_sec,
                              args.max_length, False, add_snapping,
                              add_cls) for p in files]
            for i, f in enumerate(futs):
                rows.append(f.result())
                if (i + 1) % 2000 == 0:
                    print(f"  {i + 1:,}/{len(files):,} ({time.time() - t0:.0f}s)", flush=True)
    else:
        for i, p in enumerate(files):
            rows.append(_chart_stats(p, args.window_sec, args.min_window_sec,
                                     args.max_length, False, add_snapping,
                                     add_cls))
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1:,}/{len(files):,} ({time.time() - t0:.0f}s)", flush=True)

    bad_rows = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    if bad_rows:
        print(f"\n  WARNING: {len(bad_rows)} chart(s) could not be accounted "
              f"(skipped):")
        for r in bad_rows[:10]:
            print(f"    {r['path']}: {r['error']}")

    # audit a subset (re-runs the parser; keep it cheap)
    if args.audit:
        audit_rows = [r for r in rows[:args.audit] if "error" not in r]
        for r in audit_rows:
            a = _chart_stats(r["path"], args.window_sec, args.min_window_sec,
                             args.max_length, True, add_snapping,
                             add_cls).get("audit")
            if a is not None:
                r["audit"] = a
        n_problems = sum(1 for r in audit_rows if r.get("audit", {}).get("problems"))
        n_core_drop = sum(r["audit"]["n_core_dropped"] for r in audit_rows)
        n_mismatch = sum(1 for r in audit_rows if not r["audit"]["core_match"])
        print(f"\ncore audit ({len(audit_rows)} charts): "
              f"core stream identical={n_mismatch == 0}, problems={n_problems}, "
              f"core tokens dropped by truncation={n_core_drop}")
        if n_problems:
            for r in audit_rows:
                if r["audit"]["problems"]:
                    print(f"  !! {r['path']}: {r['audit']['problems'][:2]}")
        ok_core = n_mismatch == 0 and n_problems == 0

    notes = np.array([r["n_notes"] for r in rows], dtype=np.float64)
    wins = np.array([r["n_windows"] for r in rows], dtype=np.float64)
    tok_full = np.array([sum(r["tok_full"]) for r in rows], dtype=np.float64)
    tok_canon = np.array([sum(r["tok_canon"]) for r in rows], dtype=np.float64)
    tok_on = np.array([sum(r["tok_canon_on"]) for r in rows], dtype=np.float64)
    all_full = [x for r in rows for x in r["tok_full"]]
    all_canon = [x for r in rows for x in r["tok_canon"]]
    all_on = [x for r in rows for x in r["tok_canon_on"]]
    n_win_full = len(all_full)

    saved = float(tok_full.sum() - tok_canon.sum())
    saved_pct = 100.0 * saved / max(float(tok_full.sum()), 1.0)
    speed = (float(tok_full.sum()) / max(float(tok_canon.sum()), 1.0)) ** 2
    cols = int(sum(r["n_columns"] for r in rows))
    events = int(sum(r["n_note_events"] for r in rows))
    unsup = int(sum(r["n_unsupported"] for r in rows))
    trunc_full = int(sum(x > args.max_length for x in all_full))
    trunc_canon = int(sum(x > args.max_length for x in all_canon))
    tok_drop_canon = int(sum(max(0, x - args.max_length) for x in all_canon))

    print("\n--- token budget (per chart / per note / per window) ------------------")
    print(f"  charts {len(rows):,}   notes {int(notes.sum()):,}   windows {int(wins.sum()):,}"
          f"   unsupported objects {unsup:,}")
    _print_table("LEGACY flags (hitsound+kiai+sv on, columns off = what produced the old table)", [
        ("tokens / chart", _dist(tok_full)),
        ("tokens / note", _dist(tok_full / np.maximum(notes, 1))),
        ("windows / chart", _dist(wins)),
        ("tokens / window", _dist(all_full)),
    ])
    _print_table("CANON flags (hitsound+kiai+sv off, columns off = current upstream)", [
        ("tokens / chart", _dist(tok_canon)),
        ("tokens / note", _dist(tok_canon / np.maximum(notes, 1))),
        ("windows / chart", _dist(wins)),
        ("tokens / window", _dist(all_canon)),
    ])
    _print_table("CANON + mania columns (opt-in patch, 1-based)", [
        ("tokens / chart", _dist(tok_on)),
        ("tokens / note", _dist(tok_on / np.maximum(notes, 1))),
        ("tokens / window", _dist(all_on)),
    ])
    print(f"\n  savings legacy -> canon: {saved:,.0f} tokens ({saved_pct:.1f}%), "
          f"attention cost ratio n^2 = {speed:.2f}x")
    print(f"  tokens / note: legacy {tok_full.sum() / max(notes.sum(), 1):.3f} -> "
          f"canon {tok_canon.sum() / max(notes.sum(), 1):.3f}")
    print(f"  mania column tokens: {cols:,} emitted vs {events:,} note events "
          f"({100.0 * cols / max(events, 1):.1f}% coverage) "
          f"-> {'PRESENT' if cols else 'ABSENT (upstream group.column bug)'}")
    print(f"  truncation at max_length={args.max_length}: legacy {trunc_full}/{n_win_full} "
          f"windows ({100.0 * trunc_full / max(n_win_full, 1):.2f}%), canon "
          f"{trunc_canon}/{len(all_canon)} ({100.0 * trunc_canon / max(len(all_canon), 1):.2f}%), "
          f"tokens dropped {tok_drop_canon:,}")
    holds = sum(r["holds_total"] for r in rows)
    print(f"  holds split by a 16 s boundary: {sum(r['holds_split'] for r in rows):,}/{holds:,} "
          f"({100.0 * sum(r['holds_split'] for r in rows) / max(holds, 1):.2f}%), "
          f"ends within 2 ms of a boundary: {sum(r['holds_near'] for r in rows):,}")
    print("\n  pad_efficiency (length-sorted batching, canon windows):")
    for b in (8, 16, 32, 64):
        print(f"    batch={b:2d}: {_pad_efficiency(all_canon, b):.3f}")
    print(f"  pad_efficiency if every window padded to max_length: "
          f"{sum(all_canon) / (len(all_canon) * args.max_length):.3f}")

    # verification against the real tokenizer
    verify = None
    if args.verify_cm3p:
        verify = _verify(files[:args.verify_cm3p], args.window_sec, args.min_window_sec)
        if verify is None:
            print("\n  [verify] skipped: transformers/slider not importable here")
        else:
            print(f"\n  [verify] {verify['n_ok']}/{verify['n']} charts match the real "
                  f"tokenizer exactly" + ("" if verify["n_ok"] == verify["n"]
                                          else f" (mismatches: {verify['mismatch'][:3]})"))

    summary = {
        "n_charts": len(rows), "n_notes": int(notes.sum()), "n_windows": int(wins.sum()),
        "window_sec": args.window_sec, "max_length": args.max_length,
        "add_snapping": bool(add_snapping), "add_cls_token": bool(add_cls),
        "legacy": {"tokens_total": int(tok_full.sum()),
                   "pooled_tokens_per_note": float(tok_full.sum()
                                                   / max(notes.sum(), 1)),
                   "tokens_per_chart": _dist(tok_full),
                   "tokens_per_note": _dist(tok_full / np.maximum(notes, 1)),
                   "windows_per_chart": _dist(wins),
                   "tokens_per_window": _dist(all_full),
                   "n_truncated": trunc_full},
        "canon": {"tokens_total": int(tok_canon.sum()),
                  "pooled_tokens_per_note": float(tok_canon.sum()
                                                  / max(notes.sum(), 1)),
                  "tokens_per_chart": _dist(tok_canon),
                  "tokens_per_note": _dist(tok_canon / np.maximum(notes, 1)),
                  "tokens_per_window": _dist(all_canon),
                  "n_truncated": trunc_canon,
                  "tokens_dropped_by_truncation": tok_drop_canon},
        "canon_with_columns": {"tokens_total": int(tok_on.sum()),
                               "pooled_tokens_per_note": float(tok_on.sum()
                                                               / max(notes.sum(), 1)),
                               "tokens_per_chart": _dist(tok_on),
                               "tokens_per_note": _dist(tok_on / np.maximum(notes, 1))},
        "savings_tokens": int(saved), "savings_percent": saved_pct,
        "attention_cost_ratio": speed,
        "column_tokens": cols, "note_events": events,
        "holds_total": holds, "holds_split": sum(r["holds_split"] for r in rows),
        "holds_end_within_2ms": sum(r["holds_near"] for r in rows),
        "pad_efficiency": {str(b): _pad_efficiency(all_canon, b) for b in (8, 16, 32, 64)},
        "pad_efficiency_max_length": float(sum(all_canon) / (len(all_canon) * args.max_length)),
        "verify_cm3p": verify,
        "seconds": round(time.time() - t0, 1),
    }
    if args.audit:
        summary["core_audit"] = {
            "n": len(audit_rows), "n_mismatch": n_mismatch,
            "n_problems": n_problems, "n_core_dropped": n_core_drop,
        }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.json}")

    bad = False
    if args.audit and not ok_core:
        bad = True
    if args.strict and (trunc_canon or tok_drop_canon):
        bad = True
    if verify is not None and verify["n_ok"] != verify["n"]:
        bad = True
    if bad:
        print("\nSTRICT: core-information check failed", file=sys.stderr)
        return 1
    return 0


def _verify(files: list[str], window_sec: float, min_window_sec: float):
    """Compare the offline counter with the real CM3P tokenizer (cluster only)."""
    root = Path(os.environ.get("CM3P_ROOT", "external/cm3p")).resolve()
    if not (root / "cm3p").is_dir():
        return None
    sys.path.insert(0, str(root))
    try:
        from cm3p.processing_cm3p import CM3PProcessor
        from cm3p.parsing_cm3p import load_beatmap
    except Exception:
        return None
    proc = CM3PProcessor.from_pretrained("OliBomby/CM3P")
    # match the defaults we count (no torch, no padding)
    n_ok = 0
    mismatch = []
    for p in files:
        b = load_beatmap(p)
        enc = proc(beatmap=b, audio=None, speed=1.0, padding="do_not_pad",
                   truncation="do_not_truncate", return_tensors=None)
        ids = enc["input_ids"]
        osu = off.parse_osu(p)
        bp = proc.beatmap_parser
        groups = off.chart_groups(
            osu, hitsounds=bool(bp.add_hitsounds), kiai=bool(bp.add_kiai),
            sv=bool(bp.add_mania_sv), columns=False,
            add_timing_points=bool(bp.add_timing_points),
            add_snapping=bool(bp.add_snapping))
        windows = off.window_bounds(off.song_length_ms(osu), window_sec, min_window_sec)
        lens = off.window_token_lengths(groups, windows,
                                        add_cls_token=bool(proc.beatmap_tokenizer.add_cls_token))
        got = [len(x) for x in ids]
        if got == lens:
            n_ok += 1
        else:
            mismatch.append({"path": p, "offline": lens, "real": got})
    return {"n": len(files), "n_ok": n_ok, "mismatch": mismatch}


if __name__ == "__main__":
    raise SystemExit(main())
