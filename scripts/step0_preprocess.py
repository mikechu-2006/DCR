#!/usr/bin/env python3
"""Step 0 preprocessing: raw osu!mania dump -> play table + beatmap lookup.

    step0_{tag}.csv / .parquet   player_id, beatmap_id, timestamp, playcount_cur, loss
    beatmap_meta_{tag}.csv       beatmap_id -> star, bpm, count_total, ...

one row per qualifying play and one row per 4K mania beatmap.  See
docs/step0_preprocessing.md for the full spec.

Design notes
------------
* Reads the dump SQL directly; every path and switch is a CLI argument.
* Streams both score tables through a parquet scratch file, so the 10k legacy
  table (16.9 M rows) never has to fit in memory as Python objects.
* `playcount_cur` is the 1-based rank of the play inside its (player, beatmap)
  cell, ordered by (timestamp, score_id).  It counts *recorded successful* plays
  only -- it is a lower bound on the real play count, by design.
* The modern table (`scores.sql`) is a superset of the legacy table
  (`osu_scores_mania_high.sql`) in the 1k dump, so legacy rows are deduplicated
  against it via `legacy_score_id`.  When there is no modern table (10k) the
  dedup set is empty and the step is a no-op.
* Mods are handled in two stages, and the split is the whole point.  The mods
  that do not change the chart (MR/SD/PF -- see --neutral-mods) are first
  *canonicalised away*: forgotten, not merely tolerated.  The whitelist is then
  a membership test on whatever is left (default: exactly CL).  One vocabulary
  serves both sources -- the legacy bitmask is decoded into the same tokens
  before the same two stages -- so the two halves cannot drift.
* Canonicalising instead of enumerating subsets is what keeps the whitelist
  from growing combinatorially, and it is also the honest reading of the data:
  a SD/PF play is not a new kind of observation, it is the same observation the
  dump already contains for players who restart a bad attempt by hand (3.1).
* NF is the exception and is rejected outright, not canonicalised.  It is
  chart-neutral but it is not *observation*-neutral: it rescues a run that would
  otherwise have been abandoned, and an abandoned run leaves no row at all, so
  NF adds rows with no CL analogue.  See REJECTED_CHART_NEUTRAL_MODS below.
* The beatmap metadata table is a pure projection of `osu_beatmaps` over the same
  4K whitelist -- no derived columns, no status-code interpretation, no join to
  `osu_beatmapsets` / `osu_beatmap_difficulty` / `osu_beatmap_failtimes`.
* The 10k dump has no `osu_beatmaps.sql` and no `scores.sql`; `--beatmap-src`
  points the whitelist/metadata pass at the 1k dump, which is the same dump date.

Usage
-----
    # 1k: play table (CSV) + beatmap metadata
    python scripts/step0_preprocess.py \
        --dump-dir data/raw/extracted/2026_09_01_performance_mania_top_1000 --tag 1k

    # 10k: legacy only, whitelist/metadata borrowed from the 1k dump
    python scripts/step0_preprocess.py \
        --dump-dir data/raw/extracted_10k/2026_09_01_performance_mania_top_10000 \
        --beatmap-src data/raw/extracted/2026_09_01_performance_mania_top_1000 \
        --tag 10k --out data/processed/step0_10k.parquet
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_dump import iter_insert_lines, parse_careful, rows_simple  # noqa: E402

MANIA_RULESET = "3"
N_TAIL = 7          # pp, legacy_score_id, legacy_total_score, started_at,
                    # ended_at, unix_updated_at, build_id
FLUSH = 500_000

# classic weight -> key in the modern JSON `statistics` object
CLASSIC = (("perfect", 300), ("great", 300), ("good", 200), ("ok", 100), ("meh", 50))

# legacy column indices (see the row layout comment in iter_legacy)
LEGACY_TOTAL = (6, 7, 8, 9, 10, 11)
LEGACY_NUM = ((10, 300), (8, 300), (11, 200), (7, 100), (6, 50))

# --- mod whitelist ---------------------------------------------------------
# The policy is expressed in two parts, and the split is the whole point:
#
#   * `--neutral-mods` lists the mods that do not change the chart.  They are
#     *canonicalised to "off"*: the row keeps its identity as a play, but the mod
#     is forgotten.  SD/PF do not alter the judgement weights (so they do not
#     alter ACC), and they only automate what a player would otherwise do by
#     hand -- restart on a miss.  A hand-restarted miss leaves no record at all
#     (3.1), so a SD/PF play is not a new kind of observation, it is the same
#     observation the dump already contains for players who restart manually.
#     MR changes nothing at all in a 4K chart.
#   * `--mod-whitelist` is then the set of *canonical* mod sets that survive.
#     With the defaults this is just `CL`, i.e. keep a row iff, after dropping
#     MR/SD/PF, what is left is exactly CL.
#
# Everything else -- DT/NC/HT (different rate => different chart), HD/FL/EZ/HR/
# FI/RD/IN (different read semantics), ScoreV2 -- is not neutral and not
# whitelisted, so it is rejected.  Note that canonicalising rather than
# enumerating is what keeps the whitelist from growing as 2^|neutral|: adding a
# mod to `--neutral-mods` does not require touching `--mod-whitelist`.
MOD_WHITELIST = "CL"
NEUTRAL_MODS = "MR,SD,PF"

# NF is deliberately NOT in NEUTRAL_MODS, and it is the only mod in that
# category.  It does not change the chart, but it changes *which plays get
# recorded*: it lets a run finish that would otherwise have been abandoned, and
# an abandoned run leaves no row at all (3.1).  So an NF play is not a
# relabelling of a CL play -- it is the one case that *adds* rows with no CL
# analogue.  Measured: NF rows have mean ACC 0.83 against 0.98 for the rest, and
# they are the only source of rows below ACC 0.10 -- without NF the file's
# minimum ACC is 0.234 (1k) / 0.127 (10k) instead of 0.0002 / 0.00003.
#
# So NF is *rejected* by the whitelist test rather than canonicalised away.  It
# is listed here only so the intent is greppable and so the startup log records
# it; it is not a switch.  See docs/step0_preprocessing.md 8.13b and 9 D1.
REJECTED_CHART_NEUTRAL_MODS = "NF"

# acronym -> bit in the legacy `enabled_mods` bitmask.  This table is the set of
# bits we know how to decode; a mask carrying any *other* bit is rejected
# outright rather than silently ignored (see decode_legacy_mods).
LEGACY_MOD_BITS = "NF=1,SD=32,PF=16384,MR=1073741824"

CL_SET = frozenset({"CL"})

RAW_COLS = ["user_id", "beatmap_id", "ts", "sid", "loss", "legacy_score_id"]
RAW_DTYPES = {"user_id": np.int32, "beatmap_id": np.int32, "sid": np.int64,
              "loss": np.float32, "legacy_score_id": np.int64}
OUT_COLS = ["player_id", "beatmap_id", "timestamp", "playcount_cur", "loss"]

# --- beatmap metadata projection -------------------------------------------
# output column -> index in an `osu_beatmaps` row.  Row layout (see the CREATE
# TABLE in osu_beatmaps.sql):
#    0 beatmap_id      1 beatmapset_id   2 user_id (mapper)  3 filename
#    4 checksum        5 version          6 total_length      7 hit_length
#    8 countTotal      9 countNormal     10 countSlider      11 countSpinner
#   12 diff_drain     13 diff_size       14 diff_overall     15 diff_approach
#   16 playmode       17 approved        18 last_update      19 difficultyrating
#   20 max_combo      21 playcount       22 passcount        23 youtube_preview
#   24 score_version  25 osu_file_version 26 deleted_at       27 bpm
#   28 lazer_only
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
META_N_FIELDS = 29      # full row; anything shorter cannot be projected safely

STATS: dict[str, int] = {}
# KEPT counts the mod set as the *source* spelled it; CANON counts what is left
# after canonicalisation.  Keeping both is what makes the collapse visible.
KEPT: dict[str, collections.Counter] = {"modern": collections.Counter(),
                                        "legacy": collections.Counter()}
CANON: dict[str, collections.Counter] = {"modern": collections.Counter(),
                                         "legacy": collections.Counter()}


def note(key: str, n: int = 1) -> None:
    STATS[key] = STATS.get(key, 0) + n


def mod_label(tokens) -> str:
    return "+".join(sorted(tokens)) or "(none)"


def print_mod_breakdown() -> None:
    """Which mod sets survived, per source, raw and canonicalised.

    Worth printing every run.  The raw block is the check that the legacy
    bitmask decode and the modern token filter agree about what 'allowed'
    means; the canonical block is the check that the neutral mods really did
    collapse (every row should land on one set once NF/SD/PF/MR are dropped).
    """
    for src in ("modern", "legacy"):
        tot = sum(KEPT[src].values())
        print(f"[step0] {src} kept, as the source spelled it ({tot} rows):",
              flush=True)
        for k, v in KEPT[src].most_common():
            print(f"          {k:14s} {v:9d}  {v / tot * 100:6.3f}%", flush=True)
        print(f"[step0] {src} kept, after canonicalisation:", flush=True)
        for k, v in CANON[src].most_common():
            print(f"          {k:14s} {v:9d}  {v / tot * 100:6.3f}%", flush=True)


def loss_of(acc_num: float, total: float) -> float:
    """ACC = (num + 250) / (300 * (total + 1)); loss = log(1 - ACC).

    The extra pseudo-250 judgement keeps 1 - ACC > 0 for a perfect play.
    """
    acc = (acc_num + 250.0) / (300.0 * (total + 1.0))
    return float(np.log1p(-acc))


def parse_names(spec: str) -> frozenset:
    """'MR,NF,SD,PF' -> {'MR','NF','SD','PF'} -- plain acronym list."""
    return frozenset(t.strip() for t in spec.split(",") if t.strip())


def parse_allow(spec: str) -> list[frozenset]:
    """'CL,CL+MR' -> [{'CL'}, {'CL','MR'}] -- allowed *canonical* mod sets."""
    return [frozenset(t.split("+")) for t in spec.split(",") if t.strip()]


def neutral_mask(bits: tuple[tuple[int, str], ...], neutral: frozenset) -> int:
    """Bitmask of the neutral mods -- the legacy fast path, for reporting."""
    m = 0
    for bit, name in bits:
        if name in neutral:
            m |= bit
    return m


def canonicalize(tokens: frozenset, neutral: frozenset) -> frozenset:
    """Forget the neutral mods: a play's chart identity is what is left.

    This is the *only* place the policy is applied, and both sources go through
    it, so "CL", "CL+NF", "CL+SD+PF" and "CL+MR" all reduce to {"CL"}.
    """
    return tokens - neutral


def parse_legacy_bits(spec: str) -> tuple[tuple[int, str], ...]:
    """'NF=1,SD=32' -> ((1,'NF'), (32,'SD')) -- legacy bitmask decoding."""
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, bit = item.partition("=")
        out.append((int(bit), name))
    return tuple(out)


def decode_legacy_mods(mask: int, bits: tuple[tuple[int, str], ...]) -> frozenset | None:
    """Legacy `enabled_mods` bitmask -> the modern table's mod token set.

    Returns None if the mask carries any bit that `bits` does not name.  That
    case must be a *rejection*: decoding only the known bits would silently
    ignore DT/HD/HT/EZ/..., so `mask=64` (DoubleTime) would decode to an empty
    token set and be mistaken for a plain CL play.

    The one non-obvious rule: osu!stable always sets the SuddenDeath bit
    *together with* the Perfect bit, so `SD|PF` (16416) means Perfect, not
    "SuddenDeath and Perfect".  On the 1k dump the PF bit never occurs without
    the SD bit (16384 alone: 0 rows; 16416: 10,303 rows), and the modern table
    spells the same play `CL+PF`.  Folding SD away when PF is present is what
    makes the two sources agree.

    Under the default policy this fold is *subsumed* by canonicalisation -- SD
    and PF are both neutral, so `{SD,PF}` and `{PF}` both reduce to nothing.  It
    is kept because it is a fact about the bitmask rather than about the policy:
    with `--neutral-mods MR,NF` the fold is the difference between reading 16416
    as `CL+PF` (right) and as `CL+SD+PF` (wrong).

    Legacy rows are classic by construction, so 'CL' is added by the caller.
    """
    known = 0
    for bit, _ in bits:
        known |= bit
    if mask & ~known:
        return None
    toks = {name for bit, name in bits if mask & bit}
    if "PF" in toks:
        toks.discard("SD")
    return frozenset(toks)


# ------------------------------------------------- 4K whitelist + metadata
def read_beatmaps(dump: Path, mode: str, keys: str) -> tuple[set[int], pd.DataFrame]:
    """One pass over osu_beatmaps.sql -> (4K whitelist ids, metadata frame).

    Both halves of step 0 need the same whitelist, so they are produced together
    and can never disagree about which beatmaps exist.
    """
    rows = []
    for line in iter_insert_lines(dump / "osu_beatmaps.sql"):
        for r in parse_careful(line):
            note("beatmap_rows")
            if len(r) < META_N_FIELDS:
                note("beatmap_short")
                continue
            # 13 diff_size (keys) ... 16 playmode
            if r[16] != mode or r[13] != keys:
                continue
            rows.append([r[i] for _, i in META_FIELDS])

    ids = {int(v[0]) for v in rows}
    df = pd.DataFrame(rows, columns=[c for c, _ in META_FIELDS])
    for c in META_INT:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("int64")
    for c in META_FLOAT:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    for c in META_STR:
        df[c] = df[c].astype(str)
    return ids, df.sort_values("beatmap_id", kind="stable").reset_index(drop=True)


def write_beatmap_meta(meta: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out, index=False)
    n_dupe = int(meta.beatmap_id.duplicated().sum())
    print(f"[step0] meta rows={len(meta)} dup(beatmap_id)={n_dupe} "
          f"star {meta.star.min():.3f}..{meta.star.max():.3f} "
          f"median {meta.star.median():.3f}", flush=True)
    print("[step0] meta approved counts: "
          + " ".join(f"{int(k)}:{v}" for k, v in
                     meta.approved.value_counts().sort_index().items()),
          flush=True)
    print(f"[step0] wrote {out}", flush=True)


# ------------------------------------------------------------------- modern
def iter_modern(dump: Path, ids: set[int], allow: list[frozenset],
                neutral: frozenset):
    """scores.sql -> (user_id, beatmap_id, ts, sid, loss, legacy_score_id).

    The `CL` token is the classic-scoring marker, and it is load-bearing: every
    modern 4K row *without* CL has `legacy_score_id = NULL` (100,163 / 100,163
    on the 1k dump), i.e. it is a lazer ScoreV2 score, while CL rows almost all
    have one.  So "keep only CL" is exactly "keep only the plays the legacy
    table would have stored", and modern `mods=[]` is NOT the same play as
    legacy `enabled_mods=0` (that one is spelled `CL` here).
    """
    src = dump / "scores.sql"
    if not src.exists():
        print("[step0] no scores.sql in dump -> modern branch skipped", flush=True)
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
            if head[2].strip("'") != MANIA_RULESET:   # no-op on the 2026_09 dumps
                note("modern_wrong_ruleset")
                continue
            raw = ",".join(parts[12:-N_TAIL]).strip()
            if raw[:1] == "'":
                raw = raw[1:-1]
            data = json.loads(raw.replace('\\"', '"'))
            tokens = frozenset(m["acronym"] for m in data.get("mods", []))
            canon = canonicalize(tokens, neutral)
            if canon not in allow:
                note("modern_mod_dropped")
                continue
            KEPT["modern"][mod_label(tokens)] += 1
            CANON["modern"][mod_label(canon)] += 1
            stats = data.get("statistics", {})
            total = float(sum(stats.values()))
            if total <= 0:
                note("modern_empty")
                continue
            num = float(sum(stats.get(k, 0) * w for k, w in CLASSIC))
            legacy_id = tail[1]
            yield (int(head[1]), bid, tail[4].strip("'"), int(head[0]),
                   loss_of(num, total), int(legacy_id) if legacy_id.isdigit() else -1)


# ------------------------------------------------------------------- legacy
def iter_legacy(dump: Path, ids: set[int], allow: list[frozenset],
                bits: tuple[tuple[int, str], ...], drop: set[int],
                neutral: frozenset):
    """osu_scores_mania_high.sql -> (user_id, beatmap_id, ts, sid, loss).

    Raw row layout:
        0 score_id   1 beatmap_id  2 user_id   3 score     4 maxcombo  5 rank
        6 count50    7 count100    8 count300  9 countmiss 10 countgeki
        11 countkatu 12 perfect    13 enabled_mods          14 date    15 pp

    Legacy rows are classic by construction, so 'CL' is implicit and never
    stored -- `enabled_mods=0` is the stable "no mods" play, which the modern
    table spells `CL`.  The bitmask is decoded and then canonicalised exactly
    like the modern token set, so the two sources meet at one test.
    """
    src = dump / "osu_scores_mania_high.sql"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    # bitmask -> (raw label, canonical set), or None when the mask carries a
    # bit we cannot name.  Caching the finished decision keeps the per-row cost
    # to one dict lookup across the 16.9 M-row 10k table.
    cache: dict[int, tuple[str, frozenset] | None] = {}
    for line in iter_insert_lines(src):
        for r in rows_simple(line):
            note("legacy_rows")
            bid = int(r[1])
            if bid not in ids:
                continue
            note("legacy_4k")
            mask = int(r[13])
            if mask not in cache:
                mods = decode_legacy_mods(mask, bits)
                if mods is None:
                    cache[mask] = None
                else:
                    tokens = CL_SET | mods
                    cache[mask] = (mod_label(tokens),
                                   canonicalize(tokens, neutral))
            hit = cache[mask]
            if hit is None or hit[1] not in allow:
                note("legacy_mod_dropped")
                continue
            sid = int(r[0])
            if sid in drop:                   # already present in the modern table
                note("legacy_deduped")
                continue
            KEPT["legacy"][hit[0]] += 1
            CANON["legacy"][mod_label(hit[1])] += 1
            total = float(sum(int(r[i]) for i in LEGACY_TOTAL))
            if total <= 0:
                note("legacy_empty")
                continue
            num = float(sum(int(r[i]) * w for i, w in LEGACY_NUM))
            yield (int(r[2]), bid, r[14], sid, loss_of(num, total), -1)


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


def build_raw(dump: Path, ids: set[int], allow: list[frozenset],
              bits: tuple[tuple[int, str], ...], neutral: frozenset,
              scratch: Path) -> Path:
    """Stream both sources into one parquet.  Two passes: the modern pass has to
    finish before the legacy pass can be deduplicated against it."""
    scratch.mkdir(parents=True, exist_ok=True)
    out = scratch / "plays_raw.parquet"
    t0 = time.time()

    drop: set[int] = set()
    buf, writer = [], None
    for row in iter_modern(dump, ids, allow, neutral):
        buf.append(row)
        if row[5] >= 0:
            drop.add(row[5])
        if len(buf) >= FLUSH:
            writer = write_plays(buf, out, writer)
            buf = []
    if buf:
        writer = write_plays(buf, out, writer)
    print("[step0] modern: kept %d of %d 4K rows (dedup set %d)  %.0fs"
          % (STATS.get("modern_4k", 0) - STATS.get("modern_mod_dropped", 0)
             - STATS.get("modern_empty", 0) - STATS.get("modern_wrong_ruleset", 0),
             STATS.get("modern_4k", 0), len(drop), time.time() - t0), flush=True)

    buf = []
    for row in iter_legacy(dump, ids, allow, bits, drop, neutral):
        buf.append(row)
        if len(buf) >= FLUSH:
            writer = write_plays(buf, out, writer)
            buf = []
    if buf:
        writer = write_plays(buf, out, writer)
    if writer is not None:
        writer.close()
    print("[step0] legacy: kept %d of %d 4K rows (dropped %d dup)  %.0fs"
          % (STATS.get("legacy_4k", 0) - STATS.get("legacy_mod_dropped", 0)
             - STATS.get("legacy_empty", 0) - STATS.get("legacy_deduped", 0),
             STATS.get("legacy_4k", 0), STATS.get("legacy_deduped", 0),
             time.time() - t0), flush=True)
    print_mod_breakdown()
    return out


# ------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True, type=Path,
                    help="directory holding the score tables")
    ap.add_argument("--beatmap-src", type=Path, default=None,
                    help="directory holding osu_beatmaps.sql (default: --dump-dir). "
                         "The 10k dump has no osu_beatmaps.sql, so it points here "
                         "at the 1k dump -- same dump date, same table.")
    ap.add_argument("--tag", default="1k", help="label for the output filename")
    ap.add_argument("--out", type=Path, default=None,
                    help="play table; .parquet writes parquet, anything else CSV")
    ap.add_argument("--meta-out", type=Path, default=None,
                    help="beatmap metadata CSV (default data/processed/beatmap_meta_{tag}.csv)")
    ap.add_argument("--no-meta", action="store_true",
                    help="skip the beatmap metadata table")
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--mod-whitelist", default=MOD_WHITELIST,
                    help="comma-separated allowed *canonical* mod token sets, "
                         "i.e. what is left after --neutral-mods is dropped "
                         "(default %(default)s)")
    ap.add_argument("--neutral-mods", default=NEUTRAL_MODS,
                    help="comma-separated mods that do not change the chart; "
                         "they are canonicalised to 'off' before the whitelist "
                         "test (default %(default)s)")
    ap.add_argument("--legacy-mod-bits", default=LEGACY_MOD_BITS,
                    help="acronym=bit map for decoding legacy enabled_mods "
                         "(default %(default)s)")
    ap.add_argument("--mode", default=MANIA_RULESET, help="osu_beatmaps.playmode")
    ap.add_argument("--keys", default="4", help="osu_beatmaps.diff_size (4K)")
    ap.add_argument("--count-dtype", default="int32")
    ap.add_argument("--float-dtype", default="float32")
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep the intermediate parquet")
    args = ap.parse_args()

    out = args.out or Path(f"data/processed/step0_{args.tag}.csv")
    meta_out = args.meta_out or Path(f"data/processed/beatmap_meta_{args.tag}.csv")
    scratch = args.work_dir or Path(f"data/interim/step0_{args.tag}")
    allow = parse_allow(args.mod_whitelist)
    bits = parse_legacy_bits(args.legacy_mod_bits)
    neutral = parse_names(args.neutral_mods)
    rejected = parse_names(REJECTED_CHART_NEUTRAL_MODS)
    beatmap_src = args.beatmap_src or args.dump_dir
    t0 = time.time()

    # A neutral mod named in the whitelist can never match: canonicalisation
    # removes it before the test.  Almost always a typo, so fail loudly.
    clash = neutral & {t for s in allow for t in s}
    if clash:
        raise SystemExit(f"--mod-whitelist names neutral mods {sorted(clash)}; "
                         f"canonicalisation drops them before the test, so no "
                         f"row could ever match")
    # The chart-neutral-but-rejected set (NF) must stay rejected: if it leaked
    # into either the neutral list or the whitelist it would silently come back
    # into the file, which is the exact failure mode this constant guards.
    leak = rejected & (neutral | {t for s in allow for t in s})
    if leak:
        raise SystemExit(f"REJECTED_CHART_NEUTRAL_MODS overlaps the policy: "
                         f"{sorted(leak)}. NF must be neither neutral nor "
                         f"whitelisted, or its rows reappear in the play table.")

    nbits = neutral_mask(bits, neutral)
    print("[step0] neutral mods (canonicalised to 'off'): "
          + (" ".join(sorted(neutral)) or "(none)")
          + f"   [legacy mask 0x{nbits:x}]", flush=True)
    print("[step0] mod whitelist (post-canonical): "
          + " ".join("+".join(sorted(t)) or "(none)" for t in allow), flush=True)
    print("[step0] chart-neutral but REJECTED (not canonicalised): "
          + (" ".join(sorted(rejected)) or "(none)"), flush=True)
    print("[step0] legacy bit decode: "
          + " ".join(f"{n}={b}" for b, n in bits), flush=True)

    ids, meta = read_beatmaps(beatmap_src, args.mode, args.keys)
    print(f"[step0] {args.mode} / {args.keys}K beatmaps: {len(ids)} "
          f"(scanned {STATS.get('beatmap_rows', 0)} rows, "
          f"{STATS.get('beatmap_short', 0)} too short, from {beatmap_src})", flush=True)
    if not ids:
        raise SystemExit("no beatmaps found -- wrong --beatmap-src / --mode / --keys?")

    if not args.no_meta:
        write_beatmap_meta(meta, meta_out)

    raw_path = build_raw(args.dump_dir, ids, allow, bits, neutral, scratch)
    plays = pd.read_parquet(raw_path)
    n_raw = len(plays)
    plays = plays[plays.loss.notna()]
    print(f"[step0] union plays: {n_raw}", flush=True)

    key = ["user_id", "beatmap_id"]
    plays = plays.sort_values(key + ["ts", "sid"], kind="stable").reset_index(drop=True)
    plays["playcount_cur"] = (
        plays.groupby(key, sort=False).cumcount() + 1).astype(args.count_dtype)

    out.parent.mkdir(parents=True, exist_ok=True)
    result = (plays[["user_id", "beatmap_id", "ts", "playcount_cur", "loss"]]
              .rename(columns={"user_id": "player_id", "ts": "timestamp"}))
    result["loss"] = result["loss"].astype(args.float_dtype)
    # Write to a temp name and rename, so an interrupted run cannot leave a
    # truncated file at the path downstream code reads.  The 10k write is ~190 MB
    # and takes tens of seconds; being killed mid-write otherwise looks exactly
    # like a successful run until someone tries to open it.
    tmp = out.with_name(out.name + ".part")
    if out.suffix == ".parquet":
        # 10k is ~13 M rows; CSV would be ~700 MB for no benefit
        result.to_parquet(tmp, index=False)
    else:
        result.to_csv(tmp, index=False, date_format="%Y-%m-%d %H:%M:%S")
    tmp.replace(out)

    if not args.keep_raw:
        raw_path.unlink(missing_ok=True)

    # ---- post-conditions (docs/step0_preprocessing.md section 6) -------------
    dup = int(result.duplicated(["player_id", "beatmap_id", "playcount_cur"]).sum())
    grp = result.groupby(["player_id", "beatmap_id"], sort=False)["playcount_cur"]
    contiguous = bool((grp.max() == grp.size()).all()) and bool((grp.min() == 1).all())
    print(f"[step0] rows={len(result)} players={result.player_id.nunique()} "
          f"beatmaps={result.beatmap_id.nunique()} cells={grp.ngroups}")
    print(f"[step0] dup(playcount_cur)={dup} | contiguous 1..n={contiguous} | "
          f"pc>=1={bool((result.playcount_cur >= 1).all())} | "
          f"loss<0={bool((result.loss < 0).all())}")
    print(f"[step0] timestamp {result.timestamp.min()} .. {result.timestamp.max()}")
    print(f"[step0] loss mean {result.loss.mean():.4f} sd {result.loss.std():.4f} | "
          f"playcount_cur mean {result.playcount_cur.mean():.3f} max "
          f"{result.playcount_cur.max()}")

    # ---- beatmap metadata post-conditions (docs/step0_preprocessing.md 6.2) --
    if not args.no_meta:
        dup = int(meta.beatmap_id.duplicated().sum())
        monotone = bool(meta.beatmap_id.is_monotonic_increasing)
        positive = bool((meta.star > 0).all() and (meta.count_total > 0).all()
                        and (meta.bpm > 0).all())
        lengths = bool((meta.hit_length <= meta.total_length).all())
        # the only cross-file invariant: both halves share one whitelist
        missing = int(len(set(result.beatmap_id.unique()) - set(meta.beatmap_id)))
        print(f"[step0] meta dup={dup} | sorted={monotone} | star/count/bpm>0={positive}"
              f" | hit<=total={lengths} | play-beatmaps missing from meta={missing}")

    print(f"[step0] wrote {out}  total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
