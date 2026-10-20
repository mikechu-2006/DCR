#!/usr/bin/env python
"""Step 0 (distilled): raw osu!mania dump -> play table + beatmap metadata.

This is the compact reference implementation of the step0 population. The
production entry point is scripts/step0_preprocess.py; the rules are identical:

  * 4K mania only: osu_beatmaps.playmode == 3 and diff_size == 4.
  * mod policy: canonicalise MR / SD / PF to "off"; keep a play iff what remains
    is exactly CL. NF is rejected (not canonicalised): it adds plays that have no
    CL analogue, and those plays are the low-ACC tail.
  * classic ACC: MAX and 300 both weigh 300, plus one pseudo-250 judgement so a
    perfect play still has 1 - ACC > 0.
        ACC  = (300*(c300+cMAX) + 200*c200 + 100*c100 + 50*c50 + 250)
               / (300*(total+1))
        loss = log(1 - ACC)
  * playcount_cur = 1-based rank of the recorded play inside its
    (player, beatmap) cell, ordered by (timestamp, score_id).
  * modern scores.sql and legacy osu_scores_mania_high.sql are deduplicated by
    legacy score id (modern is a superset on the 1k dump).

Outputs:
  data/processed/step0_{tag}.csv or .parquet   (player_id, beatmap_id, timestamp,
                                                playcount_cur, loss)
  data/processed/beatmap_meta_{tag}.csv        (17 columns, keyed by beatmap_id)

Examples:
  python preprocess/step0_preprocess.py \
      --dump-dir data/raw/extracted/2026_09_01_performance_mania_top_1000 --tag 1k
  python preprocess/step0_preprocess.py \
      --dump-dir data/raw/extracted_10k/2026_09_01_performance_mania_top_10000 \
      --beatmap-src data/raw/extracted/2026_09_01_performance_mania_top_1000 \
      --tag 10k --out data/processed/step0_10k.parquet
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

MANIA_RULESET = "3"
N_TAIL = 7
FLUSH = 500_000

CLASSIC = (("perfect", 300), ("great", 300), ("good", 200), ("ok", 100), ("meh", 50))
LEGACY_TOTAL = (6, 7, 8, 9, 10, 11)
LEGACY_NUM = ((10, 300), (8, 300), (11, 200), (7, 100), (6, 50))
LEGACY_BITS = ((1, "NF"), (32, "SD"), (16384, "PF"), (1073741824, "MR"))
NEUTRAL = frozenset({"MR", "SD", "PF"})
ALLOW = frozenset({"CL"})

RAW_COLS = ["user_id", "beatmap_id", "ts", "sid", "loss", "legacy_score_id"]
RAW_DTYPES = {"user_id": np.int32, "beatmap_id": np.int32, "sid": np.int64,
              "loss": np.float32, "legacy_score_id": np.int64}
OUT_COLS = ["player_id", "beatmap_id", "timestamp", "playcount_cur", "loss"]

META_FIELDS = (
    ("beatmap_id", 0), ("beatmapset_id", 1), ("mapper_id", 2), ("version", 5),
    ("star", 19), ("bpm", 27), ("max_combo", 20), ("count_total", 8),
    ("diff_overall", 14), ("diff_drain", 12), ("hit_length", 7),
    ("total_length", 6), ("playcount", 21), ("passcount", 22),
    ("approved", 17), ("last_update", 18), ("checksum", 4),
)
META_INT = ("beatmap_id", "beatmapset_id", "mapper_id", "max_combo", "count_total",
            "hit_length", "total_length", "playcount", "passcount", "approved")
META_FLOAT = ("star", "bpm", "diff_overall", "diff_drain")
META_STR = ("version", "last_update", "checksum")
META_N_FIELDS = 29

STATS: collections.Counter = collections.Counter()


def note(key: str, n: int = 1) -> None:
    STATS[key] += n


# ----------------------------------------------------------------- SQL parsing
def iter_insert_lines(path: Path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("INSERT INTO"):
                yield line


def rows_simple(line: str):
    """Rows whose fields have no commas inside quoted strings (legacy table)."""
    body = line[line.index("VALUES") + 6:].strip()
    body = body[1:]
    if body.endswith(");"):
        body = body[:-2]
    elif body.endswith(")"):
        body = body[:-1]
    for row in body.split("),("):
        yield [v[1:-1] if len(v) >= 2 and v[0] == "'" and v[-1] == "'" else v
               for v in row.split(",")]


def parse_careful(line: str):
    """MySQL INSERT VALUES parser that handles commas and escapes in strings."""
    i = line.index("VALUES") + 6
    n = len(line)
    row, field = [], []
    in_str = esc = has = quoted = False
    while i < n:
        c = line[i]
        if in_str:
            if esc:
                field.append(c); esc = False
            elif c == "\\":
                esc = True
            elif c == "'":
                in_str = False; quoted = True
            else:
                field.append(c)
        elif c == "'":
            in_str = True; has = True
        elif c == ",":
            row.append(("".join(field) if quoted else ("NULL" if not has else "".join(field))))
            field = []; has = quoted = False
        elif c == "(":
            row, field = [], []; has = quoted = False
        elif c == ")":
            row.append(("".join(field) if quoted else ("NULL" if not has else "".join(field))))
            yield row
            field = []; has = quoted = False
        elif c == ";":
            return
        elif c in " \t\r\n":
            pass
        else:
            field.append(c); has = True
        i += 1


# --------------------------------------------------------------- beatmaps/meta
def read_beatmaps(dump: Path, mode: str, keys: str) -> tuple[set[int], pd.DataFrame]:
    rows = []
    for line in iter_insert_lines(dump / "osu_beatmaps.sql"):
        for r in parse_careful(line):
            note("beatmap_rows")
            if len(r) < META_N_FIELDS:
                note("beatmap_short")
                continue
            if r[16] != mode or r[13] != keys:
                continue
            rows.append([r[i] for _, i in META_FIELDS])
    ids = {int(v[0]) for v in rows}
    meta = pd.DataFrame(rows, columns=[c for c, _ in META_FIELDS])
    for c in META_INT:
        meta[c] = pd.to_numeric(meta[c], errors="coerce").astype("int64")
    for c in META_FLOAT:
        meta[c] = pd.to_numeric(meta[c], errors="coerce").astype("float32")
    for c in META_STR:
        meta[c] = meta[c].astype(str)
    return ids, meta.sort_values("beatmap_id", kind="stable").reset_index(drop=True)


TRUEISH = ("true", "1", "1.0", "yes", "y", "t")


def read_whitelist(path: Path) -> tuple[set[int], pd.DataFrame]:
    if not path.exists():
        raise SystemExit(f"--whitelist {path} not found")
    wl = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "beatmap_id" not in wl.columns:
        raise SystemExit(f"--whitelist {path}: no beatmap_id column")
    wl = wl.copy()
    wl["beatmap_id"] = pd.to_numeric(wl["beatmap_id"], errors="coerce").astype("int64")
    for col in ("cacheable", "osu_cached", "cached"):
        if col in wl.columns:
            flag = wl[col].astype(str).str.strip().str.lower().isin(TRUEISH)
            print(f"[step0] whitelist flag {col}: {int(flag.sum())} of {len(wl)} kept", flush=True)
            wl = wl[flag]
            break
    if wl.beatmap_id.duplicated().any():
        raise SystemExit(f"--whitelist {path}: duplicate beatmap_id rows")
    return set(wl.beatmap_id.tolist()), wl


def write_meta(meta: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out, index=False)
    print(f"[step0] meta rows={len(meta)} dup={int(meta.beatmap_id.duplicated().sum())}", flush=True)


# --------------------------------------------------------------------- label
def loss_of(acc_num: float, total: float) -> float:
    acc = (acc_num + 250.0) / (300.0 * (total + 1.0))
    return float(np.log1p(-acc))


def decode_legacy_mods(mask: int) -> frozenset | None:
    known = 0
    for bit, _ in LEGACY_BITS:
        known |= bit
    if mask & ~known:
        return None
    toks = {name for bit, name in LEGACY_BITS if mask & bit}
    if "PF" in toks:
        toks.discard("SD")
    return frozenset(toks)


# -------------------------------------------------------------------- sources
def iter_modern(dump: Path, ids: set[int]):
    src = dump / "scores.sql"
    if not src.exists():
        print("[step0] no scores.sql -> modern branch skipped", flush=True)
        return
    for line in iter_insert_lines(src):
        body = line[line.index("VALUES") + 6:].strip()[1:]
        if body.endswith(");"):
            body = body[:-2]
        for row in body.split("),("):
            parts = row.split(",")
            if len(parts) < 12 + N_TAIL:
                continue
            head, tail = parts[:12], parts[-N_TAIL:]
            note("modern_rows")
            bid = int(head[3])
            if bid not in ids:
                continue
            note("modern_4k")
            if head[2].strip("'") != MANIA_RULESET:
                note("modern_wrong_ruleset")
                continue
            raw = ",".join(parts[12:-N_TAIL]).strip()
            if raw[:1] == "'":
                raw = raw[1:-1]
            data = json.loads(raw.replace('\\"', '"'))
            tokens = frozenset(m["acronym"] for m in data.get("mods", []))
            if tokens - NEUTRAL != ALLOW:
                note("modern_mod_dropped")
                continue
            stats = data.get("statistics", {})
            total = float(sum(stats.values()))
            if total <= 0:
                note("modern_empty")
                continue
            num = float(sum(stats.get(k, 0) * w for k, w in CLASSIC))
            legacy_id = tail[1]
            yield (int(head[1]), bid, tail[4].strip("'"), int(head[0]),
                   loss_of(num, total), int(legacy_id) if legacy_id.isdigit() else -1)


def iter_legacy(dump: Path, ids: set[int], drop: set[int]):
    src = dump / "osu_scores_mania_high.sql"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    cache: dict[int, frozenset | None] = {}
    for line in iter_insert_lines(src):
        for r in rows_simple(line):
            note("legacy_rows")
            bid = int(r[1])
            if bid not in ids:
                continue
            note("legacy_4k")
            mask = int(r[13])
            if mask not in cache:
                mods = decode_legacy_mods(mask)
                cache[mask] = None if mods is None else ((frozenset({"CL"}) | mods) - NEUTRAL)
            canon = cache[mask]
            if canon != ALLOW:
                note("legacy_mod_dropped")
                continue
            sid = int(r[0])
            if sid in drop:
                note("legacy_deduped")
                continue
            total = float(sum(int(r[i]) for i in LEGACY_TOTAL))
            if total <= 0:
                note("legacy_empty")
                continue
            num = float(sum(int(r[i]) * w for i, w in LEGACY_NUM))
            yield (int(r[2]), bid, r[14], sid, loss_of(num, total), -1)


# ---------------------------------------------------------------------- build
def write_plays(rows: list, out: Path, writer=None):
    df = pd.DataFrame(rows, columns=RAW_COLS)
    df["ts"] = pd.to_datetime(df["ts"], format="%Y-%m-%d %H:%M:%S")
    for c, t in RAW_DTYPES.items():
        df[c] = df[c].astype(t)
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(out, tbl.schema)
    writer.write_table(tbl)
    return writer


def build_raw(dump: Path, ids: set[int], scratch: Path) -> Path:
    scratch.mkdir(parents=True, exist_ok=True)
    out = scratch / "plays_raw.parquet"
    t0, drop, buf, writer = time.time(), set(), [], None
    for row in iter_modern(dump, ids):
        buf.append(row)
        if row[5] >= 0:
            drop.add(row[5])
        if len(buf) >= FLUSH:
            writer = write_plays(buf, out, writer)
            buf = []
    if buf:
        writer = write_plays(buf, out, writer)
    print(f"[step0] modern kept {STATS['modern_4k'] - STATS['modern_mod_dropped'] - STATS['modern_empty'] - STATS['modern_wrong_ruleset']:,} "
          f"of {STATS['modern_4k']:,} 4K rows  {time.time() - t0:.0f}s", flush=True)

    buf = []
    for row in iter_legacy(dump, ids, drop):
        buf.append(row)
        if len(buf) >= FLUSH:
            writer = write_plays(buf, out, writer)
            buf = []
    if buf:
        writer = write_plays(buf, out, writer)
    if writer is None:
        raise SystemExit("no plays kept")
    writer.close()
    print(f"[step0] legacy kept {STATS['legacy_4k'] - STATS['legacy_mod_dropped'] - STATS['legacy_empty'] - STATS['legacy_deduped']:,} "
          f"of {STATS['legacy_4k']:,} 4K rows  {time.time() - t0:.0f}s", flush=True)
    return out


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True, type=Path)
    ap.add_argument("--beatmap-src", type=Path, default=None,
                    help="directory holding osu_beatmaps.sql (default: --dump-dir; 10k uses the 1k dump)")
    ap.add_argument("--whitelist", type=Path, default=None,
                    help="optional table with beatmap_id and a cacheable flag")
    ap.add_argument("--tag", default="1k")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--meta-out", type=Path, default=None)
    ap.add_argument("--no-meta", action="store_true")
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--count-dtype", default="int32")
    ap.add_argument("--float-dtype", default="float32")
    ap.add_argument("--keep-raw", action="store_true")
    args = ap.parse_args()

    out = args.out or Path(f"data/processed/step0_{args.tag}.csv")
    meta_out = args.meta_out or Path(f"data/processed/beatmap_meta_{args.tag}.csv")
    scratch = args.work_dir or Path(f"data/interim/step0_{args.tag}")
    beatmap_src = args.beatmap_src or args.dump_dir
    t0 = time.time()

    ids, meta = read_beatmaps(beatmap_src, MANIA_RULESET, "4")
    print(f"[step0] 4K beatmaps: {len(ids)} from {beatmap_src}", flush=True)
    if not ids:
        raise SystemExit("no beatmaps found -- wrong --beatmap-src?")

    if args.whitelist:
        keep, wl = read_whitelist(args.whitelist)
        n_before = len(ids)
        ids &= keep
        print(f"[step0] whitelist: {len(ids)} of {n_before} 4K beatmaps", flush=True)
        if not ids:
            raise SystemExit("--whitelist left no beatmaps")
        meta = meta[meta.beatmap_id.isin(ids)].reset_index(drop=True)
        if "checksum" in wl.columns and "checksum" in meta.columns:
            both = meta[["beatmap_id", "checksum"]].merge(
                wl[["beatmap_id", "checksum"]], on="beatmap_id", suffixes=("", "_wl"))
            diff = int((both.checksum != both.checksum_wl.astype(str).str.lower()).sum())
            if diff:
                print(f"[step0] WARNING: {diff} whitelist checksums differ", flush=True)

    if not args.no_meta:
        write_meta(meta, meta_out)

    raw_path = build_raw(args.dump_dir, ids, scratch)
    plays = pd.read_parquet(raw_path)
    n_raw = len(plays)
    plays = plays[plays.loss.notna()]
    print(f"[step0] union plays: {n_raw:,}", flush=True)

    key = ["user_id", "beatmap_id"]
    plays = plays.sort_values(key + ["ts", "sid"], kind="stable").reset_index(drop=True)
    plays["playcount_cur"] = plays.groupby(key, sort=False).cumcount() + 1
    result = (plays[["user_id", "beatmap_id", "ts", "playcount_cur", "loss"]]
              .rename(columns={"user_id": "player_id", "ts": "timestamp"}))
    result["playcount_cur"] = result["playcount_cur"].astype(args.count_dtype)
    result["loss"] = result["loss"].astype(args.float_dtype)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    if out.suffix == ".parquet":
        result.to_parquet(tmp, index=False)
    else:
        result.to_csv(tmp, index=False, date_format="%Y-%m-%d %H:%M:%S")
    tmp.replace(out)
    if not args.keep_raw:
        raw_path.unlink(missing_ok=True)

    dup = int(result.duplicated(["player_id", "beatmap_id", "playcount_cur"]).sum())
    grp = result.groupby(["player_id", "beatmap_id"], sort=False)["playcount_cur"]
    contiguous = bool((grp.max() == grp.size()).all()) and bool((grp.min() == 1).all())
    print(f"[step0] rows={len(result):,} players={result.player_id.nunique():,} "
          f"beatmaps={result.beatmap_id.nunique():,} cells={grp.ngroups:,}", flush=True)
    print(f"[step0] dup={dup} contiguous={contiguous} "
          f"pc>=1={bool((result.playcount_cur >= 1).all())} "
          f"loss<0={bool((result.loss < 0).all())}", flush=True)
    print(f"[step0] loss mean {result.loss.mean():.4f} sd {result.loss.std():.4f} "
          f"min {result.loss.min():.4f} max {result.loss.max():.4f}", flush=True)
    print(f"[step0] wrote {out}  {time.time() - t0:.0f}s", flush=True)
    if dup or not contiguous:
        raise SystemExit("step0 post-condition failure")


if __name__ == "__main__":
    main()
