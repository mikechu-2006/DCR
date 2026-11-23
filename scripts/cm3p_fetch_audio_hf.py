#!/usr/bin/env python
"""Fetch original osu! audio for a set of beatmap ids from the Hugging Face
WebDataset `project-riz/osu-beatmaps` (213,068 beatmaps, original audio).

Why not the mapset mirrors: osu.direct enforces a per-IP quota
(8 GiB / 18 h + 200 downloads / 6 h), which cannot cover ~1,581 mapsets
(~18 GB of .osz).  This dataset streams through the HF CDN and needs no quota.

Layout (from the dataset card):

    original/data-000000.tar ...
    <key>.json   metadata + the full .osu text of every chart sharing the audio
    <key>.mp3    the ORIGINAL audio file (the extension is .mp3 by convention;
                 the real container may be mp3/ogg/wav)

The scanner downloads one shard (1 GiB) with parallel HTTP Range requests,
walks the tar, and saves the audio of every sample whose "beatmaps" list
contains one of the wanted beatmap ids.  Tar members per sample are ordered
json-then-mp3, so the json decides whether the following audio is kept.

    python scripts/cm3p_fetch_audio_hf.py --shard 0 \
        --ids-csv data/processed/cm3p_ids_todo_2786.csv \
        --out-dir data/todo_audio_hf --work-dir /tmp/hf_shard

Outputs:
    <out-dir>/<key>.mp3                    original audio (one per sample)
    <out-dir>/manifest_<shard>.jsonl       key -> matched beatmap ids
    <out-dir>/done_<shard>.json            resume marker (shard fully scanned)

Run it as a Slurm array (74 shards) on compute nodes: they have outbound
network and the same shared filesystem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131 Safari/537.36"
HF = "https://hf-mirror.com/datasets/project-riz/osu-beatmaps/resolve/main/original"


def log(msg: str) -> None:
    print(msg, flush=True)


def download_shard(url: str, dst: Path, chunk_mb: int, workers: int) -> None:
    """Parallel Range download into dst (chunks written at their offsets)."""
    size = 0
    try:
        head = requests.head(url, headers={"User-Agent": UA}, allow_redirects=True,
                             timeout=30)
        size = int(head.headers.get("content-length", 0))
    except Exception as exc:  # noqa: BLE001
        log(f"  HEAD failed ({exc}); probing with a 1-byte range")
    if size <= 0:
        r = requests.get(url, headers={"User-Agent": UA, "Range": "bytes=0-0"},
                         timeout=30)
        cr = r.headers.get("content-range", "")       # "bytes 0-0/123456"
        size = int(cr.rsplit("/", 1)[-1]) if "/" in cr else int(r.headers.get("content-length", 0))
        if size <= 1:
            raise RuntimeError(f"cannot determine shard size: {r.status_code} {r.headers}")
    chunk = chunk_mb << 20
    ranges = [(s, min(s + chunk, size) - 1) for s in range(0, size, chunk)]
    log(f"  download {dst.name}: {size / 2**30:.2f} GiB in {len(ranges)} chunks "
        f"x {chunk_mb} MiB, {workers} workers")
    t0 = time.time()

    def fetch(part: int) -> int:
        s, e = ranges[part]
        for attempt in range(8):
            try:
                r = requests.get(url, headers={"User-Agent": UA,
                                               "Range": f"bytes={s}-{e}"},
                                 timeout=(20, 180))
                if r.status_code == 429:
                    wait = float(r.headers.get("retry-after", 0) or 0) or 5 * (attempt + 1)
                    time.sleep(wait + 0.5 * part)          # jitter: avoid a thundering herd
                    continue
                r.raise_for_status()
                data = r.content
                if len(data) != e - s + 1:
                    raise ValueError(f"short chunk {part}: {len(data)} != {e - s + 1}")
                with open(dst, "r+b") as fh:
                    fh.seek(s)
                    fh.write(data)
                return len(data)
            except Exception as exc:  # noqa: BLE001
                if attempt == 7:
                    raise
                log(f"  chunk {part} retry ({exc})")
                time.sleep(2 * (attempt + 1))
        return 0

    with open(dst, "wb") as fh:
        fh.truncate(size)
    got = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(fetch, i) for i in range(len(ranges))]
        for fut in as_completed(futs):
            got += fut.result()
    if got != size:
        raise RuntimeError(f"incomplete download: {got} != {size}")
    log(f"  downloaded {size / 2**30:.2f} GiB in {time.time() - t0:.1f}s "
        f"({size / max(time.time() - t0, 1e-9) / 1e6:.1f} MB/s)")


def _consume_members(tf, wanted: set[int], out_dir: Path, manifest_path: Path,
                    shard_idx: int) -> dict:
    """Shared tar walk for the disk and the streaming readers."""
    stats = {"shard": shard_idx, "samples": 0, "audio_saved": 0, "beatmaps_matched": 0,
             "bytes_saved": 0, "seen_ids": 0}
    matched_ids: set[int] = set()
    pending: dict | None = None
    t0 = time.time()
    with manifest_path.open("a", encoding="utf-8") as mf:
        for member in tf:
            name = member.name
            if name.endswith(".json"):
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                try:
                    meta = json.loads(fh.read().decode("utf-8", "ignore"))
                except Exception as exc:  # noqa: BLE001
                    log(f"  [warn] bad json {name}: {exc}")
                    pending = None
                    continue
                stats["samples"] += 1
                bms = meta.get("beatmaps") or []
                hit = []
                for b in bms:
                    bid = b.get("beatmap_id")
                    if bid is not None and int(bid) in wanted:
                        hit.append(int(bid))
                stats["seen_ids"] += len(bms)
                pending = {"key": name[:-5], "meta": meta, "hit": hit} if hit else None
            elif name.endswith((".mp3", ".ogg", ".wav")):
                if pending is None:
                    continue
                key = pending["key"]
                fh = tf.extractfile(member)
                if fh is None:
                    pending = None
                    continue
                dst = out_dir / f"{key}{Path(name).suffix or '.mp3'}"
                tmp = out_dir / f".{key}.part"
                with tmp.open("wb") as out:
                    while True:
                        block = fh.read(1 << 20)
                        if not block:
                            break
                        out.write(block)
                os.replace(tmp, dst)
                meta = pending["meta"]
                hit_set = set(pending["hit"])
                rec = {
                    "key": key,
                    "shard": shard_idx,
                    "audio_file": str(dst),
                    "audio_bytes": dst.stat().st_size,
                    "audio_hash": meta.get("audio_hash"),
                    "audio_length": meta.get("audio_length"),
                    "beatmap_ids": pending["hit"],
                    "beatmapset_ids": sorted({
                        int(b["beatmapset_id"]) for b in meta["beatmaps"]
                        if b.get("beatmap_id") is not None
                        and int(b["beatmap_id"]) in hit_set}),
                }
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                mf.flush()
                stats["audio_saved"] += 1
                stats["beatmaps_matched"] += len(pending["hit"])
                stats["bytes_saved"] += rec["audio_bytes"]
                matched_ids.update(pending["hit"])
                pending = None
        tf.close()
    stats["matched_unique"] = len(matched_ids)
    stats["seconds"] = round(time.time() - t0, 1)
    return stats


def scan_stream(url: str, wanted: set[int], out_dir: Path, manifest_path: Path,
                shard_idx: int) -> dict:
    """Stream one tar from HTTP straight into tarfile (no disk, no Range)."""
    last: Exception | None = None
    for attempt in range(5):
        try:
            r = requests.get(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"},
                             stream=True, timeout=(20, 300))
            if r.status_code == 429:
                wait = 15 * (attempt + 1)
                log(f"  [429] stream -> sleep {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            t0 = time.time()
            tf = tarfile.open(fileobj=r.raw, mode="r|")
            stats = _consume_members(tf, wanted, out_dir, manifest_path, shard_idx)
            stats["stream_seconds"] = round(time.time() - t0, 1)
            r.close()
            return stats
        except Exception as exc:  # noqa: BLE001
            last = exc
            log(f"  stream failed ({type(exc).__name__}: {exc}); retry {attempt + 1}/5")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"stream scan failed: {last}")


def parse_shard(shard: Path, wanted: set[int], out_dir: Path, manifest_path: Path,
                shard_idx: int) -> dict:
    stats = {"shard": shard_idx, "samples": 0, "audio_saved": 0, "beatmaps_matched": 0,
             "bytes_saved": 0, "seen_ids": 0}
    matched_ids: set[int] = set()
    pending: dict | None = None
    t0 = time.time()
    with manifest_path.open("a", encoding="utf-8") as mf, \
            tarfile.open(shard, "r:") as tf:
        for member in tf:
            name = member.name
            if name.endswith(".json"):
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                try:
                    meta = json.loads(fh.read().decode("utf-8", "ignore"))
                except Exception as exc:  # noqa: BLE001
                    log(f"  [warn] bad json {name}: {exc}")
                    pending = None
                    continue
                stats["samples"] += 1
                bms = meta.get("beatmaps") or []
                hit = [int(b["beatmap_id"]) for b in bms
                       if b.get("beatmap_id") is not None and int(b["beatmap_id"]) in wanted]
                stats["seen_ids"] += len(bms)
                pending = {"key": name[:-5], "meta": meta, "hit": hit} if hit else None
            elif name.endswith((".mp3", ".ogg", ".wav")):
                if pending is None:
                    continue
                key = pending["key"]
                fh = tf.extractfile(member)
                if fh is None:
                    pending = None
                    continue
                dst = out_dir / f"{key}{Path(name).suffix or '.mp3'}"
                tmp = out_dir / f".{key}.part"
                with tmp.open("wb") as out:
                    while True:
                        block = fh.read(1 << 20)
                        if not block:
                            break
                        out.write(block)
                os.replace(tmp, dst)
                meta = pending["meta"]
                rec = {
                    "key": key,
                    "shard": shard_idx,
                    "audio_file": str(dst),
                    "audio_bytes": dst.stat().st_size,
                    "audio_hash": meta.get("audio_hash"),
                    "audio_length": meta.get("audio_length"),
                    "beatmap_ids": pending["hit"],
                    "beatmapset_ids": sorted({
                        int(b["beatmapset_id"]) for b in meta["beatmaps"]
                        if b.get("beatmap_id") is not None
                        and int(b["beatmap_id"]) in set(pending["hit"])}),
                }
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                mf.flush()
                stats["audio_saved"] += 1
                stats["beatmaps_matched"] += len(pending["hit"])
                stats["bytes_saved"] += rec["audio_bytes"]
                matched_ids.update(pending["hit"])
                pending = None
    stats["matched_unique"] = len(matched_ids)
    stats["seconds"] = round(time.time() - t0, 1)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--link", action="store_true",
                    help="no download: hardlink <beatmap_id>.<ext> from every manifest_*.jsonl")
    ap.add_argument("--num-shards", type=int, default=74)
    ap.add_argument("--ids-csv", default="data/processed/cm3p_ids_todo_2786.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-dir", default="/tmp/hf_shard")
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mode", choices=["stream", "download", "xet"], default="stream",
                    help="stream = tar over HTTP without disk; xet = hf_xet download "
                         "then local parse (fastest here); download = parallel Range chunks")
    ap.add_argument("--keep-shard", action="store_true",
                    help="download mode: keep the tar (debugging)")
    ap.add_argument("--force", action="store_true", help="re-scan a finished shard")
    args = ap.parse_args()

    import pandas as pd

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.link:
        n = 0
        for mf in sorted(out_dir.glob("manifest_*.jsonl")):
            for line in mf.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                src = Path(rec["audio_file"])
                if not src.exists():
                    continue
                for bid in rec.get("beatmap_ids", []):
                    dst = out_dir / f"{bid}{src.suffix}"
                    if dst.exists():
                        continue
                    try:
                        os.link(src, dst)
                    except OSError:
                        import shutil
                        shutil.copy2(src, dst)
                    n += 1
        print(f"linked {n:,} beatmap audio file(s) in {out_dir}")
        return
    if args.shard is None:
        raise SystemExit("--shard is required unless --link is given")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / f"done_{args.shard}.json"
    if done.exists() and not args.force:
        log(f"shard {args.shard}: already done ({done})")
        return

    df = pd.read_csv(args.ids_csv)
    wanted = set(int(x) for x in df["beatmap_id"])
    log(f"shard {args.shard}/{args.num_shards - 1}: {len(wanted):,} wanted beatmap ids")

    url = f"{HF}/data-{args.shard:06d}.tar"
    manifest = out_dir / f"manifest_{args.shard}.jsonl"
    if args.mode == "stream":
        stats = scan_stream(url, wanted, out_dir, manifest, args.shard)
    elif args.mode == "xet":
        # hf_xet downloads the shard with many parallel chunks (~8+ MB/s on the
        # HKUST-GZ link) and then we parse it from local disk.
        from huggingface_hub import hf_hub_download
        local_dir = work_dir / f"shard_{args.shard:06d}"
        local_dir.mkdir(parents=True, exist_ok=True)
        local = hf_hub_download("project-riz/osu-beatmaps",
                                f"original/data-{args.shard:06d}.tar",
                                repo_type="dataset", local_dir=str(local_dir))
        shard = Path(local)
        stats = parse_shard(shard, wanted, out_dir, manifest, args.shard)
        shard.unlink(missing_ok=True)
        import shutil
        shutil.rmtree(local_dir, ignore_errors=True)
    else:
        shard = work_dir / f"data-{args.shard:06d}.tar"
        download_shard(url, shard, args.chunk_mb, args.workers)
        stats = parse_shard(shard, wanted, out_dir, manifest, args.shard)
        if not args.keep_shard:
            shard.unlink(missing_ok=True)
    done.write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")
    log(f"shard {args.shard}: {json.dumps(stats)}")


if __name__ == "__main__":
    main()
