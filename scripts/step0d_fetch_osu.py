#!/usr/bin/env python
"""Step 0d - download the .osu bodies a content run needs (pipeline A, remote box).

The dumps ship beatmap metadata but no chart files, so the part of the whitelist that the
published CM3P table does not cover has to be fetched.  This script is deliberately dumb and
resumable:

  * one request per beatmap id against the public endpoint
        https://osu.ppy.sh/osu/{beatmap_id}
    (no OAuth, returns the .osu text);
  * the MD5 is compared with the checksum recorded in the 2026-09-01 dump snapshot, so a
    chart the mapper has since updated is DETECTED rather than silently encoded;
  * a file that is already correct is never re-downloaded, so re-running after a network
    failure only fetches what is still missing.

Input manifest: manifests/cm3p_todo_1k.csv  (columns beatmap_id, checksum[, beatmapset_id])

Then:
    python scripts/step0c_chart_content.py --source osu-dir \
        --ids-file manifests/cm3p_todo_1k.csv --osu-dir data/raw/osu_cache \
        --out data/processed/chart_content_cm3p_add.parquet
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://osu.ppy.sh/osu/{bid}"
UA = "DenseConstantRegressor-cm3p-fetch/0.1 (+research; contact: repo owner)"


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: str) -> list:
    import pandas as pd
    df = pd.read_csv(path, dtype=str)
    cols = {c.strip().lower(): c for c in df.columns}
    if "beatmap_id" not in cols:
        df = pd.read_csv(path, dtype=str, header=None, names=["beatmap_id", "checksum"])
        cols = {"beatmap_id": "beatmap_id", "checksum": "checksum"}
    ids = df[cols["beatmap_id"]].astype("int64").tolist()
    ck = (df[cols["checksum"]].astype(str).str.lower().tolist()
          if "checksum" in cols else [""] * len(ids))
    return list(dict.fromkeys(zip(ids, ck)))


def fetch_one(bid: int, want_ck: str, out_dir: Path, timeout: float, retries: int):
    """-> (status, detail).  status in {ok, cached, mismatch, missing, failed}."""
    dst = out_dir / f"{bid}.osu"
    if dst.exists() and want_ck and md5_file(dst) == want_ck:
        return ("cached", "")
    last = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL.format(bid=bid), headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
            if not body.lstrip().startswith(b"osu file format"):
                last = f"not a .osu payload ({len(body)} bytes)"
                raise ValueError(last)
            dst.write_bytes(body)
            got = md5_file(dst)
            if want_ck and got != want_ck:
                # keep the file (it is still the right chart, just a later revision) but say so
                return ("mismatch", f"md5 {got[:8]} != snapshot {want_ck[:8]}")
            return ("ok", "")
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 403, 451):          # deleted / restricted / unavailable
                return ("missing", f"HTTP {exc.code}")
            last = f"HTTP {exc.code}"
        except Exception as exc:                     # noqa: BLE001 - network is a zoo
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(min(2.0 ** attempt, 20.0) + random.random())
    return ("failed", last)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", default="manifests/cm3p_todo_1k.csv")
    ap.add_argument("--out-dir", default="data/raw/osu_cache")
    ap.add_argument("--workers", type=int, default=4, help="be polite: osu! is a donation-funded site")
    ap.add_argument("--delay", type=float, default=0.25, help="seconds between requests per worker")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N ids")
    ap.add_argument("--dry-run", action="store_true", help="list what would be fetched, fetch nothing")
    ap.add_argument("--state", default="data/raw/osu_cache/_fetch_state.json")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = load_manifest(args.ids_file)
    if args.limit:
        work = work[:args.limit]
    todo = [(b, c) for b, c in work
            if not ((out_dir / f"{b}.osu").exists() and c and md5_file(out_dir / f"{b}.osu") == c)]
    print(f"manifest {args.ids_file}: {len(work):,} ids, {len(work) - len(todo):,} already correct, "
          f"{len(todo):,} to fetch  ->  {out_dir}", flush=True)
    if args.dry_run:
        for b, c in todo[:20]:
            print(f"  would fetch {b} (want md5 {c[:8]})")
        print(f"  ... {len(todo):,} total; nothing was fetched (--dry-run)")
        return

    t0 = time.time()
    res = {}
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_delayed, args.delay, fetch_one, b, c, out_dir, args.timeout, args.retries): b
                for b, c in todo}
        for fut in cf.as_completed(futs):
            bid = futs[fut]
            try:
                status, detail = fut.result()
            except Exception as exc:                 # noqa: BLE001
                status, detail = "failed", str(exc)
            res[str(bid)] = {"status": status, "detail": detail}
            done += 1
            if done % 200 == 0 or done == len(todo):
                print(f"  {done:,}/{len(todo):,}  ({time.time() - t0:.0f}s)  "
                      f"ok={sum(1 for v in res.values() if v['status'] in ('ok', 'cached')):,} "
                      f"mismatch={sum(1 for v in res.values() if v['status'] == 'mismatch'):,} "
                      f"missing={sum(1 for v in res.values() if v['status'] == 'missing'):,} "
                      f"failed={sum(1 for v in res.values() if v['status'] == 'failed'):,}", flush=True)

    # merge into any previous state so a resumed run keeps the full picture
    state_path = Path(args.state)
    state = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except Exception:                            # noqa: BLE001
            state = {}
    state.update(res)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=1))

    counts = {}
    for v in state.values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    have = len(list(out_dir.glob("*.osu")))
    print(f"\n{have:,} .osu files in {out_dir}")
    for k in ("ok", "cached", "mismatch", "missing", "failed"):
        if counts.get(k):
            print(f"  {k:9s}: {counts[k]:,}")
    bad = [b for b, v in state.items() if v["status"] in ("missing", "failed")]
    if bad:
        print(f"  NOTE: {len(bad):,} ids have no usable file; they will simply be absent from "
              f"the content table and step2 will drop them (they are reported, not guessed)")
    print(f"done in {time.time() - t0:.0f}s")


def _delayed(delay: float, fn, *a):
    time.sleep(delay * random.random())
    return fn(*a)


if __name__ == "__main__":
    main()
