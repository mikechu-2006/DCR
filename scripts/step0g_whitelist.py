#!/usr/bin/env python3
"""Step 0g - the whitelist as a table, plus the .osu cache status.

Why this exists
---------------
Step 0 derives the 4K mania whitelist from the dump (osu_beatmaps), but the dump ships no
chart files.  The content pipeline (CM3P) needs the .osu bodies, and the published 244K
table only covers 11,877 of the 21,949 charts -- the other 11,877 were simply never
fetched, which is why a full-corpus re-embed was impossible.  The cache plan is now:

    * scripts/step0d_fetch_osu.py *attempts* every whitelist chart into data/osu_cache
      (only the .osu text body; no audio, no images, no .osz);
    * this script turns the outcome into the whitelist table that step 0 reads.

Two tables, two audiences
-------------------------
--manifest-out (tracked in git, e.g. manifests/whitelist_1k.csv):

    beatmap_id, checksum, beatmapset_id

  the full whitelist, independent of any local cache -- this is what a fresh clone and
  the HPC side fetch from.

--out (runtime, e.g. data/processed/whitelist_1k.csv):

    beatmap_id, checksum, beatmapset_id, cached, valid, md5_ok, cacheable, fetch_status

  cached        a .osu file exists in the cache dir
  valid         it really looks like a .osu body (BOM-tolerant header check)
  md5_ok        md5(file) == the dump snapshot checksum.  False = a later revision, kept
                and flagged -- exactly the checksum_ok rule of step0c; not a reason to
                drop the chart
  cacheable     cached and valid  ->  step 0 keeps the chart
  fetch_status  from step0d's _fetch_state.json when present (ok/cached/mismatch/missing/
                failed/not_attempted), for diagnosis -- missing means HTTP 404/403/451

step0 (scripts/step0_preprocess.py --whitelist <out>) then drops every chart whose
cacheable is false, from BOTH the play table and the metadata table, so the content table
can cover 100% of what step 0 emits instead of 54%.

The id universe
---------------
The full whitelist is read from --meta (the step0 metadata table) *and*, when it exists,
from the tracked manifest manifests/whitelist_{tag}.csv.  The manifest is what keeps this
idempotent: once step0 starts emitting a narrowed beatmap_meta, --meta alone would shrink
the whitelist on every run.  The manifest is the stable full list, anything in --meta that
the manifest does not know about is reported rather than silently dropped, and
--no-reference falls back to --meta as-is.

Usage
-----
    python scripts/step0g_whitelist.py --tag 1k \\
        --cache data/osu_cache \\
        --out data/processed/whitelist_1k.csv \\
        --manifest-out manifests/whitelist_1k.csv

    # no .osu cache on this machine? still emit the tracked manifest:
    python scripts/step0g_whitelist.py --manifest-out manifests/whitelist_1k.csv --skip-cache
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd

PROC = Path("data/processed")
CACHE = Path("data/osu_cache")
MANIFESTS = Path("manifests")
BOM = b"\xef\xbb\xbf"
HEADER = b"osu file format"
PROBE_BYTES = 64

OUT_COLS = ["beatmap_id", "checksum", "beatmapset_id", "cached", "valid", "md5_ok",
            "cacheable", "fetch_status"]


def md5_head(path: Path) -> tuple[str, bool]:
    """(md5 hex, header looks like a .osu body) in one pass over the file."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        head = fh.read(PROBE_BYTES)
        h.update(head)
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    probe = head[len(BOM):] if head.startswith(BOM) else head
    return h.hexdigest(), probe.lstrip().startswith(HEADER)


def scan_cache(cache: Path, jobs: int) -> tuple[dict[int, tuple[str, bool]], int]:
    """beatmap_id -> (md5, valid) for every *.osu in the cache, plus the bad-name count."""
    files = sorted(p for p in cache.glob("*.osu") if p.is_file())
    found: dict[int, tuple[str, bool]] = {}
    bad_names = 0

    def probe(p: Path):
        try:
            bid = int(p.stem)
        except ValueError:
            return None
        try:
            return bid, *md5_head(p)
        except OSError as exc:                       # unreadable file: treat as absent
            print(f"[cache] cannot read {p.name}: {exc}")
            return None

    with cf.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for item in pool.map(probe, files):
            if item is None:
                bad_names += 1
                continue
            bid, digest, ok = item
            if bid in found and found[bid][0] != digest:
                print(f"[cache] duplicate beatmap_id {bid} with different contents; "
                      f"keeping the first file")
                continue
            found[bid] = (digest, ok)
    return found, bad_names


def load_state(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        print(f"[state] {path} unreadable; fetch_status left empty")
        return {}


def atomic_write_csv(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="1k", help="dataset label (default %(default)s)")
    ap.add_argument("--meta", type=Path, default=None,
                    help="step0 beatmap metadata CSV, i.e. the 4K whitelist "
                         "(default data/processed/beatmap_meta_{tag}.csv)")
    ap.add_argument("--reference", type=Path, default=None,
                    help="full-whitelist manifest used as the id universe "
                         "(default manifests/whitelist_{tag}.csv when that file exists). "
                         "It keeps the table stable once step0 starts emitting a "
                         "*filtered* beatmap_meta; --no-reference takes --meta as-is")
    ap.add_argument("--no-reference", action="store_true",
                    help="ignore the tracked manifest and take the id universe from --meta")
    ap.add_argument("--cache", type=Path, default=CACHE,
                    help="the .osu cache directory (default %(default)s)")
    ap.add_argument("--out", type=Path, default=None,
                    help="whitelist + cache status CSV "
                         "(default data/processed/whitelist_{tag}.csv)")
    ap.add_argument("--manifest-out", type=Path, default=None,
                    help="also write the cache-independent id manifest "
                         "(beatmap_id,checksum,beatmapset_id), e.g. manifests/whitelist_1k.csv")
    ap.add_argument("--dropped-out", type=Path, default=None,
                    help="CSV of the non-cacheable charts and why "
                         "(default <out stem>_dropped.csv)")
    ap.add_argument("--json", type=Path, default=None,
                    help="machine-readable report (default <out stem>_report.json)")
    ap.add_argument("--state", type=Path, default=None,
                    help="fetch_state.json from step0d (default <cache>/_fetch_state.json)")
    ap.add_argument("--jobs", type=int, default=8, help="md5 threads (default %(default)s)")
    ap.add_argument("--skip-cache", action="store_true",
                    help="do not touch the cache: only write the tracked manifest")
    return ap.parse_args(argv)


def read_ids(path: Path) -> pd.DataFrame:
    """(beatmap_id, checksum, beatmapset_id) from a step0 metadata table or a manifest."""
    df = pd.read_csv(path)
    if "beatmap_id" not in df.columns:
        raise SystemExit(f"{path}: no beatmap_id column (has {', '.join(map(str, df.columns))})")
    out = pd.DataFrame({
        "beatmap_id": pd.to_numeric(df["beatmap_id"], errors="coerce").astype("int64"),
        "checksum": (df["checksum"].astype(str).str.lower()
                     if "checksum" in df.columns else pd.Series("", index=df.index)),
        "beatmapset_id": (pd.to_numeric(df["beatmapset_id"], errors="coerce").astype("Int64")
                          if "beatmapset_id" in df.columns
                          else pd.Series(pd.NA, index=df.index, dtype="Int64")),
    })
    dupes = int(out.beatmap_id.duplicated().sum())
    if dupes:
        raise SystemExit(f"{path}: {dupes} duplicate beatmap_id; refusing to guess")
    return out.sort_values("beatmap_id", kind="stable").reset_index(drop=True)


def resolve_universe(args, tag: str) -> tuple[pd.DataFrame, Path | None]:
    """The id universe: the tracked full manifest when there is one, else the metadata table.

    step0/preprocess now *narrows* beatmap_meta to the cacheable charts, so the metadata
    table can no longer be assumed to be the full whitelist.  The tracked manifest (written
    by this script once, committed) is the stable version of it; anything in --meta that the
    manifest does not know about is reported rather than silently dropped.
    """
    meta_path = args.meta or (PROC / f"beatmap_meta_{tag}.csv")
    if args.no_reference:
        ref_path = None
    elif args.reference:
        ref_path = args.reference
    else:
        # the 10k dump ships no osu_beatmaps.sql and borrows the 1k table, so the 1k
        # manifest is the same whitelist -- prefer the tag's own file when there is one
        cands = [MANIFESTS / f"whitelist_{tag}.csv"] + (
            [MANIFESTS / "whitelist_1k.csv"] if tag != "1k" else [])
        ref_path = next((p for p in cands if p.exists()), cands[0])
    if not meta_path.exists() and not (ref_path and ref_path.exists()):
        raise SystemExit(f"{meta_path} not found -- run scripts/step0_preprocess.py first "
                         f"(or point --reference at a manifest)")
    meta = read_ids(meta_path) if meta_path.exists() else None
    if meta is not None:
        print(f"[meta] {meta_path}: {len(meta):,} rows")
    ref = read_ids(ref_path) if (ref_path and ref_path.exists()) else None
    if ref is None:
        return meta, None                                    # type: ignore[return-value]
    if meta is None:
        print(f"[ref] {ref_path}: {len(ref):,} rows (--meta missing; the manifest is the "
              f"whitelist universe)")
        return ref, ref_path
    unknown = sorted(set(meta.beatmap_id) - set(ref.beatmap_id))
    if unknown:
        print(f"[ref] WARNING: {len(unknown):,} ids in {meta_path} are not in {ref_path} "
              f"(e.g. {unknown[:5]}) -- a new dump?  re-emit the manifest with "
              f"--manifest-out, or pass --no-reference")
    print(f"[ref] {ref_path}: {len(ref):,} rows -- the whitelist universe")
    return ref, ref_path


def main() -> None:
    args = parse_args()
    t0 = time.time()
    tag = args.tag
    meta_path = args.meta or (PROC / f"beatmap_meta_{tag}.csv")
    out_path = args.out or (PROC / f"whitelist_{tag}.csv")
    dropped_path = args.dropped_out or out_path.with_name(out_path.stem + "_dropped.csv")
    json_path = args.json or out_path.with_name(out_path.stem + "_report.json")

    meta, ref_path = resolve_universe(args, tag)
    print(f"[universe] {len(meta):,} 4K mania charts"
          + (f" from {ref_path}" if ref_path else f" from {meta_path}"))

    # ---------------------------------------------------------------- tracked manifest
    if args.manifest_out:
        manifest = meta[["beatmap_id", "checksum", "beatmapset_id"]]
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(manifest, args.manifest_out)
        print(f"[manifest] wrote {args.manifest_out}  ({len(manifest):,} rows, "
              f"{args.manifest_out.stat().st_size / 1e3:.0f} KB)")

    if args.skip_cache:
        print("[cache] --skip-cache: no cache scan, no status table")
        return

    # ------------------------------------------------------------------ cache scan
    if not args.cache.exists():
        raise SystemExit(f"{args.cache} not found -- run scripts/step0d_fetch_osu.py first")
    found, bad_names = scan_cache(args.cache, args.jobs)
    state = load_state(args.state or (args.cache / "_fetch_state.json"))
    mb = sum(p.stat().st_size for p in args.cache.glob("*.osu") if p.is_file()) / 1e6
    print(f"[cache] {args.cache}: {len(found):,} .osu files ({mb:.0f} MB, "
          f"{bad_names} non-numeric names)")

    ids = meta["beatmap_id"]
    md5 = pd.Series({b: v[0] for b, v in found.items()}, dtype="object")
    valid = pd.Series({b: v[1] for b, v in found.items()}, dtype="object")

    table = meta.copy()
    table["cached"] = ids.isin(md5.index)
    table["valid"] = ids.map(valid).astype("boolean").fillna(False).astype(bool)
    got = ids.map(md5)
    ok = pd.Series(pd.NA, index=table.index, dtype="boolean")
    keep = table["cached"].to_numpy()
    ok.loc[keep] = (got[keep].to_numpy() == table.loc[keep, "checksum"].to_numpy())
    table["md5_ok"] = ok
    table["cacheable"] = table["cached"] & table["valid"]
    table["fetch_status"] = [
        str(state.get(str(b), {}).get("status") or ("not_attempted" if not c else ""))
        for b, c in zip(ids, table["cached"])
]
    table = table[OUT_COLS]

    # ---------------------------------------------------------------- dropped / report
    dropped = table[~table["cacheable"]].copy()
    dropped["reason"] = [
        "invalid_payload" if c and not v else s
        for c, v, s in zip(dropped["cached"], dropped["valid"], dropped["fetch_status"])
]
    atomic_write_csv(dropped[["beatmap_id", "checksum", "beatmapset_id", "reason"]],
                     dropped_path)
    atomic_write_csv(table, out_path)

    extra = sorted(set(found) - set(ids.tolist()))
    by_reason = dropped["reason"].value_counts().to_dict()
    report = {
        "meta": str(meta_path),
        "cache": str(args.cache),
        "whitelist": int(len(table)),
        "cached": int(table["cached"].sum()),
        "valid": int(table["valid"].sum()),
        "md5_ok": int(table["md5_ok"].fillna(False).sum()),
        "md5_mismatch": int((table["md5_ok"] == False).sum()),        # noqa: E712
        "cacheable": int(table["cacheable"].sum()),
        "dropped": int(len(dropped)),
        "dropped_by_reason": {str(k): int(v) for k, v in by_reason.items()},
        "dropped_ids": sorted(int(b) for b in dropped["beatmap_id"]),
        "cache_files_not_in_whitelist": extra,
        "cache_invalid_payload": int((table["cached"] & ~table["valid"]).sum()),
        "status": "ok",
}
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=1))

    # ----------------------------------------------------------------------- print
    n = len(table)
    print(f"\n[whitelist] {out_path}  ({n:,} rows)")
    print(f"  cacheable   {report['cacheable']:,}/{n:,} = {report['cacheable'] / n:.2%}")
    print(f"  cached      {report['cached']:,}   valid {report['valid']:,}   "
          f"md5_ok {report['md5_ok']:,}   md5_mismatch {report['md5_mismatch']:,} "
          f"(later revision, kept)")
    print(f"  dropped     {report['dropped']:,}  ->  {dropped_path}")
    for k, v in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"      {k:14s} {v:,}")
    if extra:
        print(f"  NOTE: {len(extra):,} cached files are not in the whitelist "
              f"(e.g. {extra[:5]})")
    print(f"[report] {json_path}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
