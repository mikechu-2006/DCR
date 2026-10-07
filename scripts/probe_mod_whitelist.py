#!/usr/bin/env python3
"""One-off probe: what mod encodings actually occur on 4K plays in the 1k dump?

Needed before settling the step-0 mod policy.  The legacy table stores mods as a
bitmask, so we must know (a) whether Perfect is stored as `PF` alone (16384) or
as `SD|PF` (32|16384), (b) whether NoFail/SuddenDeath co-occur with each other,
and (c) which token sets the modern table actually emits -- in particular whether
`CL` is always present.  It is not (see probe_modern_cl.py).

    python scripts/probe_mod_whitelist.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_dump import iter_insert_lines, rows_simple  # noqa: E402
from step0_preprocess import N_TAIL, read_beatmaps  # noqa: E402

DUMP = Path("data/raw/extracted/2026_09_01_performance_mania_top_1000")

# osu! stable mod bits (only the ones we care about)
BIT_NAMES = {1: "NF", 32: "SD", 16384: "PF", 1 << 30: "MR"}
CANDIDATE_MASK = sum(BIT_NAMES)          # NF | SD | PF | MR

ids, _ = read_beatmaps(DUMP, "3", "4")
print(f"4K whitelist: {len(ids)} beatmaps\n")

# ------------------------------------------------------------------ legacy
cnt: collections.Counter = collections.Counter()
for line in iter_insert_lines(DUMP / "osu_scores_mania_high.sql"):
    for r in rows_simple(line):
        if int(r[1]) in ids:
            cnt[int(r[13])] += 1

total = sum(cnt.values())
print(f"legacy: {total} 4K rows, {len(cnt)} distinct enabled_mods values")
print("  top 30 by frequency:")
for k, v in cnt.most_common(30):
    names = "+".join(n for b, n in sorted(BIT_NAMES.items()) if k & b)
    print(f"    {k:12d}  {v:9d}  {v / total * 100:6.3f}%   {names or '(no mods)'}")

inside = {k: v for k, v in cnt.items() if k & ~CANDIDATE_MASK == 0}
print(f"\n  rows whose bits are a subset of NF|SD|PF|MR: {sum(inside.values())}"
      f"  ({sum(inside.values()) / total * 100:.3f}%)")
print("  of those, the distinct masks are:")
for k, v in sorted(inside.items(), key=lambda kv: -kv[1]):
    names = "+".join(n for b, n in sorted(BIT_NAMES.items()) if k & b)
    print(f"    {k:12d}  {v:9d}   {names or '(no mods)'}")

both = int(cnt.get(32 | 16384, 0))
print(f"\n  SD|PF together (mask 16416): {both}")
print(f"  PF alone (16384): {int(cnt.get(16384, 0))}")
print(f"  SD alone (32):    {int(cnt.get(32, 0))}")

# ------------------------------------------------------------------ modern
cc: collections.Counter = collections.Counter()
n_mod = 0
for line in iter_insert_lines(DUMP / "scores.sql"):
    body = line[line.index("VALUES") + 6:].strip()[1:]
    if body.endswith(");"):
        body = body[:-2]
    for row in body.split("),("):
        parts = row.split(",")
        if len(parts) < 12 + N_TAIL:
            continue
        head = parts[:12]
        if int(head[3]) not in ids:
            continue
        raw = ",".join(parts[12:-N_TAIL]).strip()
        if raw[:1] == "'":
            raw = raw[1:-1]
        data = json.loads(raw.replace('\\"', '"'))
        cc[frozenset(m["acronym"] for m in data.get("mods", []))] += 1
        n_mod += 1

print(f"\nmodern: {n_mod} 4K rows, {len(cc)} distinct mod token sets")
print("  top 30 by frequency:")
for k, v in cc.most_common(30):
    print(f"    {'+'.join(sorted(k)) or '(none)':22s} {v:9d}  {v / n_mod * 100:6.3f}%")
