#!/usr/bin/env python3
"""One-off probe: does a modern 4K row without a `CL` token correspond to a
legacy row, or is it a lazer-only score?

This decides whether canonicalising to "exactly CL" silently drops real plays.
The modern table spells classic scoring as an explicit `CL` mod, while the
legacy table is classic by construction and never stores it -- so "no mods" is
`mask=0` in legacy but `mods=[]` in modern.  The question is whether those two
spellings are the same play.

`legacy_score_id` is the join key: osu-web fills it when the score is also
present in the legacy table (i.e. submitted by stable).  If the `(none)` modern
rows all have a legacy_score_id, they are stable scores and the legacy pass will
pick them up anyway.  If they are NULL, they are lazer-only and dropping them
loses plays.

    python scripts/probe_modern_cl.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_dump import iter_insert_lines  # noqa: E402
from step0_preprocess import N_TAIL, read_beatmaps  # noqa: E402

DUMP = Path("data/raw/extracted/2026_09_01_performance_mania_top_1000")

ids, _ = read_beatmaps(DUMP, "3", "4")
print(f"4K whitelist: {len(ids)} beatmaps\n")

# token set -> Counter{ 'legacy_score_id' : n, 'NULL' : n }
xtab: dict[frozenset, collections.Counter] = collections.defaultdict(
    collections.Counter)
n = 0
for line in iter_insert_lines(DUMP / "scores.sql"):
    body = line[line.index("VALUES") + 6:].strip()[1:]
    if body.endswith(");"):
        body = body[:-2]
    for row in body.split("),("):
        parts = row.split(",")
        if len(parts) < 12 + N_TAIL:
            continue
        head, tail = parts[:12], parts[-N_TAIL:]
        if int(head[3]) not in ids:
            continue
        n += 1
        raw = ",".join(parts[12:-N_TAIL]).strip()
        if raw[:1] == "'":
            raw = raw[1:-1]
        data = json.loads(raw.replace('\\"', '"'))
        toks = frozenset(m["acronym"] for m in data.get("mods", []))
        has = "legacy_score_id" if tail[1].strip().upper() != "NULL" else "NULL"
        xtab[toks][has] += 1

print(f"modern 4K rows: {n}\n")
print(f"{'mod set':24s} {'has legacy_id':>14s} {'NULL (lazer-only)':>18s}")
print("-" * 60)
for toks, c in sorted(xtab.items(), key=lambda kv: -sum(kv[1].values()))[:24]:
    label = "+".join(sorted(toks)) or "(none)"
    print(f"{label:24s} {c['legacy_score_id']:14d} {c['NULL']:18d}")

cl = sum(sum(c.values()) for t, c in xtab.items() if "CL" in t)
nocl = sum(sum(c.values()) for t, c in xtab.items() if "CL" not in t)
nocl_null = sum(c["NULL"] for t, c in xtab.items() if "CL" not in t)
nocl_leg = sum(c["legacy_score_id"] for t, c in xtab.items() if "CL" not in t)
print("-" * 60)
print(f"rows WITH    CL: {cl:9d}  ({cl / n * 100:6.3f}%)")
print(f"rows WITHOUT CL: {nocl:9d}  ({nocl / n * 100:6.3f}%)"
      f"   of which NULL={nocl_null}, has legacy_id={nocl_leg}")
print(f"\nrows WITHOUT CL and without a legacy counterpart: {nocl_null}")
