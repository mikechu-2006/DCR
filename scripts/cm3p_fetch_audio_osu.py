#!/usr/bin/env python
"""Fetch missing mapset audio from the OFFICIAL osu! download (osu!direct).

Why: osu.direct's per-IP quota (8 GiB / 18 h) cannot cover the ~775 mapsets
still missing for the 2,786-chart run.  The official endpoint is a different
service with the account's own (much larger) allowance:

    web (session cookie):  GET https://osu.ppy.sh/beatmapsets/<setid>/download?noVideo=1
    API (OAuth2 bearer):   GET https://osu.ppy.sh/api/v2/beatmapsets/<setid>/download

Probed 2026-11-11 from the HPC:
    no auth             -> 403 (API) / 302-to-beatmap-page (web)
    invalid bearer      -> 401 (API)   => the API route is OAuth-gated
    invalid cookie      -> 302 to the beatmap page

Credentials are read from a FILE, never from argv or the environment, and are
never printed:

    ~/.osu_token     OAuth2 access token (Bearer, ~24 h lifetime)   [preferred]
    ~/.osu_session   value of the osu_session cookie (full account)  [fallback]

    python scripts/cm3p_fetch_audio_osu.py \
        --sample results/cm3p_todo_official/missing_now.csv \
        --out-dir data/todo_audio_fallback --token-file ~/.osu_token

Only <beatmap_id>.<ext> is written; the .osz is parsed in memory and discarded.
Resumable (existing audio is skipped) and paced (--delay) to stay well inside
the site's anti-abuse limits.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import requests

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0 Safari/537.36")


def log(msg: str) -> None:
    print(msg, flush=True)


def load_helpers():
    path = Path(__file__).with_name("cm3p_probe_fetch_audio.py")
    spec = importlib.util.spec_from_file_location("cm3p_fetch_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_secret(path: str) -> str:
    """Read a bare token, or pull access_token out of a pasted JSON response."""
    import re as _re
    p = Path(path).expanduser()
    if not p.is_file():
        raise SystemExit(f"credential file not found: {p}")
    mode = p.stat().st_mode & 0o777
    if mode & 0o077:
        log(f"WARNING: {p} is readable by others (mode {mode:o}); chmod 600 it")
    raw = p.read_text(encoding="utf-8").strip()
    if "access_token" in raw:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and obj.get("access_token"):
                return str(obj["access_token"]).strip()
        except Exception:  # noqa: BLE001
            m = _re.search(r'"access_token"\s*:\s*"([^"]+)"', raw)
            if m:
                return m.group(1).strip()
    if " " in raw or "\n" in raw:
        log(f"WARNING: {p} contains spaces/newlines; only the bare access token should be stored")
    return raw.strip().strip('"').strip("'")


def download_set(sess: requests.Session, setid: int, auth: dict, api: bool,
                 delay: float, timeout: float) -> bytes:
    base = "https://osu.ppy.sh/api/v2/beatmapsets" if api else "https://osu.ppy.sh/beatmapsets"
    url = f"{base}/{setid}/download" + ("" if api else "?noVideo=1")
    headers = {**auth, "User-Agent": UA}
    if not api:
        # the web download route requires the beatmapset page as Referer
        headers["Referer"] = f"https://osu.ppy.sh/beatmapsets/{setid}"
    last: Exception | None = None
    for attempt in range(4):
        try:
            r = sess.get(url, headers=headers,
                         allow_redirects=False, stream=True, timeout=(20, timeout))
            if r.status_code == 404:
                r.close()
                raise RuntimeError("404 download unavailable for this mapset")
            if r.status_code == 429:
                wait = int(r.headers.get("retry-after", 30)) or 30
                log(f"  [429] set {setid}: sleep {wait}s")
                r.close()
                time.sleep(wait)
                continue
            if r.status_code in (401, 403):
                r.close()
                raise SystemExit(
                    f"auth rejected (HTTP {r.status_code}) -- the token/cookie is invalid, "
                    f"expired, or lacks the 'public' scope")
            if r.status_code in (301, 302, 303, 307, 308):
                loc = r.headers.get("location", "")
                r.close()
                if not loc:
                    raise RuntimeError("redirect without Location")
                # CDN host (bm*.ppy.sh) must NOT receive the session cookie
                r = sess.get(loc, headers={"User-Agent": UA}, stream=True,
                             timeout=(20, timeout))
            r.raise_for_status()
            ctype = r.headers.get("content-type", "")
            buf = bytearray()
            for chunk in r.iter_content(1 << 20):
                buf += chunk
                if len(buf) > 600 << 20:
                    raise ValueError("osz larger than 600 MiB")
            r.close()
            if bytes(buf[:2]) == b"PK":
                return bytes(buf)
            if "json" in ctype:
                try:
                    js = json.loads(bytes(buf).decode("utf-8", "ignore"))
                    for key in ("url", "download_url", "file", "location"):
                        if isinstance(js, dict) and key in js:
                            rr = sess.get(js[key], headers={"User-Agent": UA},
                                          stream=True, timeout=(20, timeout))
                            data = rr.content
                            if data[:2] == b"PK":
                                return data
                except Exception:  # noqa: BLE001
                    pass
            raise ValueError(f"not a zip (HTTP {r.status_code}, {ctype}, {len(buf)} B)")
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            log(f"  set {setid} retry {attempt + 1}/4 ({type(exc).__name__}: {exc})")
            time.sleep(5 * (attempt + 1))
        finally:
            if delay:
                time.sleep(delay)
    raise RuntimeError(f"download failed: {last}")


def extract(osz: bytes, rows: list[dict], osu_dir: Path, out_dir: Path,
            helpers) -> list[dict]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(osz))
        names = zf.namelist()
    except Exception as exc:  # noqa: BLE001
        return [{"beatmap_id": int(r["beatmap_id"]), "ok": False,
                 "error": f"zip: {exc}"} for r in rows]
    out = []
    for row in rows:
        bid = int(row["beatmap_id"])
        osu_path = osu_dir / f"{bid}.osu"
        wanted = helpers.parse_audio_filename(osu_path) if osu_path.exists() else None
        member = helpers._member_lookup(names, wanted)
        rec = {"beatmap_id": bid, "member": member}
        if member is None:
            rec.update(ok=False, error="no audio member")
            out.append(rec)
            continue
        try:
            payload = zf.read(member)
        except Exception as exc:  # noqa: BLE001
            rec.update(ok=False, error=f"zip read: {exc}")
            out.append(rec)
            continue
        ext = Path(member).suffix.lower() or ".mp3"
        if len(payload) < 4096 or not helpers.is_audio(payload):
            rec.update(ok=False, error=f"bad audio ({len(payload)} B)")
            out.append(rec)
            continue
        dst = out_dir / f"{bid}{ext}"
        tmp = out_dir / f".{bid}{ext}.part"
        tmp.write_bytes(payload)
        os.replace(tmp, dst)
        rec.update(ok=True, bytes=len(payload), path=str(dst))
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True, help="CSV with beatmap_id,beatmapset_id")
    ap.add_argument("--osu-dir", default="data/osu_cache")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--token-file", default="~/.osu_token",
                    help="OAuth2 access token (Bearer) -- preferred")
    ap.add_argument("--session-file", default=None,
                    help="osu_session cookie value -- fallback")
    ap.add_argument("--delay", type=float, default=1.2, help="seconds between mapsets")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1, help="parallel mapsets")
    ap.add_argument("--probe", action="store_true",
                    help="only fetch the first set and report the HTTP flow")
    ap.add_argument("--check", action="store_true",
                    help="only verify the credential against /me, /beatmapsets and /download")
    args = ap.parse_args()

    import pandas as pd

    helpers = load_helpers()
    if args.session_file:
        auth = {"Cookie": f"osu_session={read_secret(args.session_file)}"}
        api = False
        log("auth: session cookie (web endpoint)")
    else:
        auth = {"Authorization": f"Bearer {read_secret(args.token_file)}"}
        api = True
        log("auth: OAuth2 bearer (api/v2 endpoint)")

    if args.check:
        sess = requests.Session()
        setid = 81109
        probes = [
            ("api me", "https://osu.ppy.sh/api/v2/me"),
            ("api beatmapset", f"https://osu.ppy.sh/api/v2/beatmapsets/{setid}"),
            ("api download", f"https://osu.ppy.sh/api/v2/beatmapsets/{setid}/download"),
            ("web download", f"https://osu.ppy.sh/beatmapsets/{setid}/download?noVideo=1"),
        ]
        for name, url in probes:
            try:
                r = sess.get(url, headers={**auth, "User-Agent": UA},
                             allow_redirects=False, timeout=20)
                body = (r.text or "")[:120].replace("\n", " ")
                print(f"  {name:16s} HTTP {r.status_code}  "
                      f"location={r.headers.get('location', '')[:60]}  {body if r.status_code >= 400 else ''}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {name:16s} ERROR {type(exc).__name__}: {exc}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest_osu.jsonl"

    df = pd.read_csv(args.sample)
    df["beatmap_id"] = df["beatmap_id"].astype("int64")
    df["beatmapset_id"] = df["beatmapset_id"].astype("int64")
    have = {int(p.stem) for p in out_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".mp3", ".ogg", ".wav", ".flac", ".m4a")
            and p.stem.isdigit()}
    pending = df[~df.beatmap_id.isin(have)]
    if args.limit:
        pending = pending.head(args.limit)
    groups = [g.to_dict("records") for _, g in pending.groupby("beatmapset_id", sort=False)]
    log(f"{len(pending)} chart(s) / {len(groups)} set(s) to fetch")

    if args.probe:
        groups = groups[:1]
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    abort = threading.Event()
    lock = threading.Lock()

    def work(rows):
        if abort.is_set():
            return []
        sess = requests.Session()          # one session per worker
        setid = int(rows[0]["beatmapset_id"])
        try:
            osz = download_set(sess, setid, auth, api, args.delay, args.timeout)
            return extract(osz, rows, Path(args.osu_dir), out_dir, helpers)
        except SystemExit as exc:
            with lock:
                log(f"auth abort: {exc}")
            abort.set()
            return [{"beatmap_id": int(r["beatmap_id"]), "ok": False,
                     "error": str(exc)} for r in rows]
        except Exception as exc:  # noqa: BLE001
            return [{"beatmap_id": int(r["beatmap_id"]), "ok": False,
                     "error": str(exc)} for r in rows]

    n_ok = n_err = 0
    done = 0
    t0 = time.time()
    with manifest.open("a", encoding="utf-8") as mf, \
            ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(work, rows): rows for rows in groups}
        for fut in as_completed(futs):
            rows = futs[fut]
            setid = int(rows[0]["beatmapset_id"])
            try:
                recs = fut.result()
            except Exception as exc:  # noqa: BLE001
                recs = [{"beatmap_id": int(r["beatmap_id"]), "ok": False,
                         "error": str(exc)} for r in rows]
            with lock:
                for rec in recs:
                    rec["beatmapset_id"] = setid
                    mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    mf.flush()
                    if rec.get("ok"):
                        n_ok += 1
                    else:
                        n_err += 1
                        if rec.get("error"):
                            log(f"[fail] {rec['beatmap_id']} set={setid}: {rec['error']}")
                done += 1
                if done % 10 == 0:
                    el = time.time() - t0
                    log(f"[{done}/{len(groups)}] ok={n_ok} err={n_err} "
                        f"{el:.0f}s ({el / max(done, 1):.1f}s/set)")
            if abort.is_set():
                pool.shutdown(wait=False, cancel_futures=True)
                break
    log(f"done: ok={n_ok} err={n_err} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
