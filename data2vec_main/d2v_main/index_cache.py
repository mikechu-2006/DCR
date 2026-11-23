"""Persistent chart index: beatmap_id -> n_frames (and invalid marker).

Scanning all 21,949 .osu files costs ~4 minutes per process.  The index is a
small JSON cache keyed by the osu directory, file count, max mtime and frame
rate; it is rebuilt automatically when any of those change.

Build it once on the login node:
    python -m d2v_main.index_cache --osu-dir data/osu_cache --frame-hz 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .charts import ChartError, ChartStore, parse_chart
from .utils import repo_root


def default_cache_path(frame_hz: int = 100) -> Path:
    return repo_root() / "data2vec_main" / "cache" / f"chart_index_{frame_hz}.json"


def _meta(store: ChartStore, frame_hz: int) -> dict:
    n = len(store.paths)
    max_mtime = 0
    for p in store.paths.values():
        max_mtime = max(max_mtime, os.stat(p).st_mtime_ns)
    return {
        "osu_dir": str(store.osu_dir),
        "n_files": n,
        "max_mtime_ns": max_mtime,
        "frame_hz": int(frame_hz),
    }


def build(store: ChartStore, frame_hz: int = 100, path: str | Path | None = None) -> dict[int, int]:
    path = Path(path) if path else default_cache_path(frame_hz)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_frames: dict[str, int] = {}
    bad = 0
    for i, (cid, p) in enumerate(store.paths.items(), 1):
        try:
            _, nf = parse_chart(str(p), frame_hz)
        except (ChartError, OSError, ValueError):
            nf = -1
            bad += 1
        n_frames[str(cid)] = int(nf)
        if i % 2000 == 0:
            print(f"[index] {i}/{len(store.paths)} ({time.time() - t0:.0f}s)", flush=True)
    payload = {**_meta(store, frame_hz), "built_at": int(time.time()),
               "n_bad": bad, "n_frames": n_frames}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")))
    tmp.replace(path)
    print(f"[index] built {len(n_frames)} charts ({bad} bad) in {time.time() - t0:.1f}s -> {path}",
          flush=True)
    return {int(k): int(v) for k, v in n_frames.items()}


def load(store: ChartStore, frame_hz: int = 100, path: str | Path | None = None,
         force: bool = False) -> dict[int, int]:
    path = Path(path) if path else default_cache_path(frame_hz)
    if path.exists() and not force:
        try:
            payload = json.loads(path.read_text())
            meta = _meta(store, frame_hz)
            if all(payload.get(k) == meta[k] for k in meta):
                data = {int(k): int(v) for k, v in payload["n_frames"].items()}
                print(f"[index] loaded {len(data)} charts from {path}", flush=True)
                return data
            print("[index] stale cache; rebuilding", flush=True)
        except Exception as exc:
            print(f"[index] unreadable cache ({exc}); rebuilding", flush=True)
    return build(store, frame_hz, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osu-dir", default="data/osu_cache")
    ap.add_argument("--frame-hz", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    store = ChartStore(args.osu_dir)
    path = Path(args.out) if args.out else default_cache_path(args.frame_hz)
    load(store, args.frame_hz, path, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
