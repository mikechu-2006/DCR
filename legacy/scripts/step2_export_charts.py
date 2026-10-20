#!/usr/bin/env python3
"""P0a: 把所有 beatmap 的逐区间难度导成 .minacalc_cache/*.npz（可断点续跑）。

    python3 scripts/step2_export_charts.py --table data/processed/step1_1k.parquet

产物：.minacalc_cache/<osu 内容哈希>_itv_1.000_0.930_v2.npz
每个 npz 含 diffs (6,2,n) / pts (2,n) / jack 事件两对手 + meta。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


def _key(osu: Path) -> str:
    return (hashlib.sha1(osu.read_bytes()).hexdigest()[:16]
            + "_itv_1.000_0.930_v2.npz")


def _worker(args):
    osu_dir, bm, cache_dir, exe = args
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from minacalc.io import load_chart_itv
    osu = Path(osu_dir) / f"{bm}.osu"
    npz = Path(cache_dir) / _key(osu)
    if npz.exists():
        return (bm, "skip", 0)
    try:
        ch = load_chart_itv(osu, cache_dir=cache_dir, exe=exe)
        return (bm, "ok", int(ch.n))
    except Exception as exc:                      # noqa: BLE001
        return (bm, f"fail: {type(exc).__name__}: {exc}", 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/processed/step1_1k.parquet")
    ap.add_argument("--osu-dir", default="data/osu_cache")
    ap.add_argument("--cache-dir", default=".minacalc_cache")
    ap.add_argument("--exe", default="external/minacalc-standalone/build/export_itv")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress", type=int, default=500)
    ap.add_argument("--report", default="data/processed/step2_export_report.json")
    args = ap.parse_args()

    ids = sorted(pd.read_parquet(args.table, columns=["beatmap_id"])
                 ["beatmap_id"].unique().tolist())
    if args.limit:
        ids = ids[:args.limit]
    print(f"charts to export: {len(ids)}", flush=True)

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    tasks = [(args.osu_dir, int(b), args.cache_dir, args.exe) for b in ids]
    stats = {"ok": 0, "skip": 0, "fail": 0}
    failed: list[str] = []
    itv_sum = 0
    t0 = time.time()

    # 注意：本机 /dev/shm 受限，multiprocessing 的 SemLock 会 PermissionError，
    # 这里用线程池——主要开销是 export_itv 子进程，线程足够并行。
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for i, (bm, status, n) in enumerate(pool.map(_worker, tasks), 1):
            if status == "ok":
                stats["ok"] += 1
                itv_sum += n
            elif status == "skip":
                stats["skip"] += 1
            else:
                stats["fail"] += 1
                failed.append(f"{bm}: {status}")
            if i % args.progress == 0:
                el = time.time() - t0
                print(f"  [{i}/{len(tasks)}] ok={stats['ok']} skip={stats['skip']} "
                      f"fail={stats['fail']}  {el:.0f}s  "
                      f"({i / el:.1f}/s, eta {(len(tasks) - i) / (i / el):.0f}s)",
                      flush=True)

    elapsed = time.time() - t0
    report = {"n_charts": len(ids), "stats": stats, "failed": failed[:50],
              "mean_intervals": itv_sum / max(stats["ok"], 1),
              "elapsed_s": round(elapsed, 1)}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "failed"}, indent=1))
    if failed:
        print("FAILED:", failed[:10])
    return 0 if stats["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
