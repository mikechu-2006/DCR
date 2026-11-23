#!/usr/bin/env python
"""Stage probe audio on a node that HAS network (HPC login node).

The official CM3P table (OliBomby/CM3P-Embeddings-244K) was produced with
include_audio=True.  Reproducing it therefore needs the real mapset audio, which
is not in this repo (data/osu_cache holds .osu text only).  This script downloads
the .osz for every beatmapset in a probe sample and extracts ONLY the audio file
that the chart's own .osu names (AudioFilename), writing

    <out-dir>/<beatmap_id>.<ext>

The .osz is parsed from memory and is NEVER written to disk: no skins, no
backgrounds, no videos.  Run it on the login node; the compute node is assumed
offline and only reads the extracted audio.

    python scripts/cm3p_probe_fetch_audio.py
        --sample results/cm3p_probe_official/sample.csv
        --osu-dir data/osu_cache
        --out-dir /hpc2hdd/$USER/DCR/data/probe_audio
        --workers 4

Resumable: a chart whose <beatmap_id>.* already exists is skipped.  A
manifest.jsonl next to the audio records every attempt (ok/error, bytes, seconds,
member name).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

AUDIO_EXTS = {".mp3", ".ogg", ".wav", ".flac", ".m4a"}
def is_audio(payload: bytes) -> bool:
    """Container / MPEG-frame sniff: ID3, Ogg, RIFF/WAVE, FLAC, MPEG audio."""
    if payload.startswith((b"ID3", b"OggS", b"RIFF", b"fLaC")):
        return True
    return len(payload) >= 2 and payload[0] == 0xFF and (payload[1] & 0xE0) == 0xE0
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0 Safari/537.36")

_LOCK = threading.Lock()


def log(msg: str) -> None:
    with _LOCK:
        print(msg, flush=True)


def parse_audio_filename(osu_path: Path) -> str | None:
    """AudioFilename from the .osu header (the value keeps its case/spaces)."""
    text = osu_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"^AudioFilename\s*:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if not m:
        return None
    name = m.group(1).strip()
    return name or None


def _member_lookup(names: list[str], wanted: str | None) -> str | None:
    """Find the zip member for an osu! AudioFilename (case/path tolerant)."""
    if not names:
        return None
    by_lower = {n.lower(): n for n in names}
    by_base = {}
    for n in names:
        by_base.setdefault(n.replace("\\", "/").split("/")[-1].lower(), n)

    if wanted:
        w = wanted.strip().replace("\\", "/")
        for cand in (w, w.lstrip("./")):
            if cand in names:
                return cand
            if cand.lower() in by_lower:
                return by_lower[cand.lower()]
        base = w.split("/")[-1].lower()
        if base in by_base:
            return by_base[base]
    # fallback: first top-level audio file
    audio = [n for n in names if Path(n).suffix.lower() in AUDIO_EXTS]
    audio.sort(key=lambda n: (("/" in n), len(n)))
    return audio[0] if audio else None


def fetch_osz(url: str, session: requests.Session, timeout: tuple[int, int],
              max_bytes: int) -> bytes:
    last: Exception | None = None
    for attempt in range(3):
        try:
            r = session.get(url, stream=True, timeout=timeout, allow_redirects=True)
            if r.status_code == 429:
                wait = 20 * (attempt + 1)
                log(f"  [429] {url} -> sleep {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            buf = bytearray()
            for chunk in r.iter_content(1 << 20):
                buf += chunk
                if len(buf) > max_bytes:
                    raise ValueError(f"osz larger than --max-mb ({max_bytes >> 20} MiB)")
            if not buf:
                raise ValueError("empty response")
            if bytes(buf[:2]) != b"PK":
                raise ValueError(f"not a zip: {bytes(buf[:16])!r}")
            return bytes(buf)
        except Exception as exc:  # noqa: BLE001 - retry any transport error
            last = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"download failed: {last}")


def extract_for_set(sample_rows: list[dict], osu_dir: Path, out_dir: Path,
                    url_template: str, timeout: tuple[int, int], max_bytes: int,
                    delay: float) -> list[dict]:
    """Download one .osz and pull each row's audio out of it (in memory)."""
    setid = int(sample_rows[0]["beatmapset_id"])
    url = url_template.format(setid=setid)
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    t0 = time.time()
    try:
        data = fetch_osz(url, session, timeout, max_bytes)
    except Exception as exc:  # noqa: BLE001
        return [{"beatmap_id": int(r["beatmap_id"]), "beatmapset_id": setid,
                 "ok": False, "error": f"download: {exc}", "url": url}
                for r in sample_rows]
    dl_sec = time.time() - t0
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        names = zf.namelist()
    except Exception as exc:  # noqa: BLE001
        return [{"beatmap_id": int(r["beatmap_id"]), "beatmapset_id": setid,
                 "ok": False, "error": f"zip: {exc}", "url": url}
                for r in sample_rows]

    out: list[dict] = []
    for row in sample_rows:
        bid = int(row["beatmap_id"])
        osu_path = osu_dir / f"{bid}.osu"
        wanted = parse_audio_filename(osu_path) if osu_path.exists() else None
        member = _member_lookup(names, wanted)
        rec = {"beatmap_id": bid, "beatmapset_id": setid,
               "wanted": wanted, "member": member, "osz_bytes": len(data),
               "download_sec": round(dl_sec, 2), "url": url}
        if member is None:
            rec.update(ok=False, error="no audio member found")
            out.append(rec)
            continue
        try:
            payload = zf.read(member)
        except Exception as exc:  # noqa: BLE001
            rec.update(ok=False, error=f"zip read: {exc}")
            out.append(rec)
            continue
        ext = Path(member).suffix.lower() or ".mp3"
        if len(payload) < 4096:
            rec.update(ok=False, error=f"audio too small ({len(payload)} B)")
            out.append(rec)
            continue
        if ext not in AUDIO_EXTS:
            rec.update(ok=False, error=f"member is not an audio file: {member!r}")
            out.append(rec)
            continue
        if not is_audio(payload):
            # Some mapsets ship an MPEG stream with leading zero padding; the
            # member name is still authoritative (it came from AudioFilename).
            log(f"  [warn] {bid}: magic {payload[:8]!r} not recognised; "
                f"trusting the {ext} extension")
        dst = out_dir / f"{bid}{ext}"
        tmp = out_dir / f".{bid}{ext}.part"
        tmp.write_bytes(payload)
        os.replace(tmp, dst)
        rec.update(ok=True, bytes=len(payload), path=str(dst))
        out.append(rec)
    if delay:
        time.sleep(delay)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True,
                    help="CSV with beatmap_id,beatmapset_id[,checksum] (the probe sample)")
    ap.add_argument("--osu-dir", default="data/osu_cache",
                    help="cache of .osu bodies; used only to read AudioFilename")
    ap.add_argument("--out-dir", required=True, help="where <beatmap_id>.<ext> is written")
    ap.add_argument("--url", default="https://osu.direct/api/d/{setid}",
                    help="osz download template; {setid} is substituted")
    ap.add_argument("--workers", type=int, default=4, help="parallel beatmapsets")
    ap.add_argument("--delay", type=float, default=0.0, help="pause between set downloads (s)")
    ap.add_argument("--timeout", type=float, default=180.0, help="per-request timeout (s)")
    ap.add_argument("--max-mb", type=int, default=400, help="abort a .osz bigger than this")
    ap.add_argument("--limit", type=int, default=0, help="only the first N pending charts")
    ap.add_argument("--force", action="store_true", help="re-download even if audio exists")
    args = ap.parse_args()

    import pandas as pd

    osu_dir = Path(args.osu_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"

    df = pd.read_csv(args.sample)
    for col in ("beatmap_id", "beatmapset_id"):
        if col not in df.columns:
            raise SystemExit(f"{args.sample}: missing column {col}")
    df["beatmap_id"] = df["beatmap_id"].astype("int64")
    df["beatmapset_id"] = df["beatmapset_id"].astype("int64")

    have = {int(p.stem) for p in out_dir.iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS and p.stem.isdigit()}
    pending = df if args.force else df[~df.beatmap_id.isin(have)]
    if args.limit:
        pending = pending.head(args.limit)
    if pending.empty:
        print(f"nothing to do: all {len(df)} chart(s) already have audio in {out_dir}")
        return
    groups = [g.to_dict("records") for _, g in pending.groupby("beatmapset_id", sort=False)]
    print(f"{len(df)} chart(s) in the sample, {len(have)} already staged, "
          f"{len(pending)} to fetch from {len(groups)} beatmapset(s)")
    print(f"audio dir: {out_dir}")

    t0 = time.time()
    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool, \
            manifest.open("a", encoding="utf-8") as fh:
        futures = {pool.submit(extract_for_set, g, osu_dir, out_dir,
                              args.url, (20.0, args.timeout), args.max_mb * (1 << 20),
                              args.delay): g for g in groups}
        for fut in as_completed(futures):
            try:
                records = fut.result()
            except Exception as exc:  # noqa: BLE001
                log(f"[error] beatmapset {futures[fut][0]['beatmapset_id']}: {exc}")
                n_err += 1
                continue
            for rec in records:
                with _LOCK:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                bid = rec["beatmap_id"]
                if rec.get("ok"):
                    n_ok += 1
                    log(f"[ok]   {bid} set={rec['beatmapset_id']} "
                        f"{rec['bytes'] / 1e6:.2f} MB ({rec['member']}) "
                        f"osz={rec['osz_bytes'] / 1e6:.1f} MB in {rec['download_sec']:.0f}s")
                else:
                    n_err += 1
                    log(f"[fail] {bid} set={rec['beatmapset_id']}: {rec.get('error')}")

    el = time.time() - t0
    print(f"\nstaged {n_ok} chart(s), {n_err} failure(s) in {el / 60:.1f} min")
    print(f"manifest: {manifest}")
    print("next: run scripts/cm3p_official_probe.py on a compute node "
          "(--audio-dir points here)")


if __name__ == "__main__":
    main()
