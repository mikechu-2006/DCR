#!/usr/bin/env python3
"""One-off probe: `loss` distribution per kept mod set, both sources.

Why this exists: under the step-0 policy the neutral mods (MR/NF/SD/PF) are
canonicalised away, so all of them are kept as plain CL plays.  That is a
*deliberate* choice -- but it is not a free one, because the four are not
alike:

  * MR is inert.
  * SD and PF are *fail* mods: they delete the play if you drop a note / hit a
    non-MAX judgement.  A *recorded* SD/PF play is therefore selected to be
    clean, and if PF forces an all-MAX play then every CL+PF row sits at the ACC
    ceiling and `loss` is constant given the note count.  The counter-argument
    (and the reason they are still kept) is that this is exactly what a player
    does by hand -- restart on a miss -- and a hand-restarted attempt leaves no
    record at all, so the selection is present in the CL rows too.
  * NF is the opposite: it lets a play *finish* that would otherwise have been
    abandoned, so CL+NF rows are the low-ACC tail.

This script is the check on how much of that tail the canonicalisation folds
into the CL bucket.  Run it after changing --neutral-mods.

    python scripts/probe_mod_loss.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_dump import iter_insert_lines, rows_simple  # noqa: E402
from step0_preprocess import (CLASSIC, CL_SET, LEGACY_NUM, LEGACY_TOTAL,  # noqa: E402
                              MOD_WHITELIST, N_TAIL, NEUTRAL_MODS,
                              canonicalize, decode_legacy_mods, loss_of,
                              parse_allow, parse_legacy_bits, parse_names,
                              read_beatmaps)

DUMP = Path("data/raw/extracted/2026_09_01_performance_mania_top_1000")
ALLOW = parse_allow(MOD_WHITELIST)
NEUTRAL = parse_names(NEUTRAL_MODS)
BITS = parse_legacy_bits("NF=1,SD=32,PF=16384,MR=1073741824")

ids, _ = read_beatmaps(DUMP, "3", "4")
print(f"4K whitelist: {len(ids)}")
print(f"policy: neutral={sorted(NEUTRAL)} whitelist={sorted(map(sorted, ALLOW))}\n")


def report(name: str, per: dict[str, list]):
    rows = sorted(per.items(), key=lambda kv: -len(kv[1]))
    n = sum(len(v) for _, v in rows)
    print(f"{name} ({n} kept rows, as the source spelled the mods):")
    print(f"  {'mod set':12s} {'rows':>9s} {'share':>7s} {'mean ACC':>9s}"
          f" {'mean loss':>10s} {'min loss':>9s} {'max loss':>9s}"
          f" {'ACC>=0.9999':>12s}")
    for k, v in rows:
        acc = 1.0 - np.exp(np.array(v))
        print(f"  {k:12s} {len(v):9d} {len(v) / n * 100:6.2f}% {acc.mean():9.5f}"
              f" {np.mean(v):10.4f} {min(v):9.4f} {max(v):9.4f}"
              f" {np.mean(acc >= 0.9999) * 100:11.2f}%")
    print()


# ------------------------------------------------------------------ legacy
per: dict[str, list] = collections.defaultdict(list)
clean = collections.Counter()
for line in iter_insert_lines(DUMP / "osu_scores_mania_high.sql"):
    for r in rows_simple(line):
        if int(r[1]) not in ids:
            continue
        mods = decode_legacy_mods(int(r[13]), BITS)
        if mods is None:
            continue
        toks = CL_SET | mods
        if canonicalize(toks, NEUTRAL) not in ALLOW:
            continue
        label = "+".join(sorted(toks))
        total = float(sum(int(r[i]) for i in LEGACY_TOTAL))
        if total <= 0:
            continue
        num = float(sum(int(r[i]) * w for i, w in LEGACY_NUM))
        per[label].append(loss_of(num, total))
        if "PF" in toks:
            # does PF really mean "all MAX"?
            c50, c100, c300, cmiss = (int(r[6]), int(r[7]), int(r[8]), int(r[9]))
            cgeki, ckatu = int(r[10]), int(r[11])
            clean[(cmiss > 0, ckatu > 0, c100 > 0, c50 > 0, c300 > 0)] += 1

report("legacy", per)
print("legacy CL+PF rows, by whether any non-MAX judgement is present")
print("  (cmiss>0, ckatu>0, c100>0, c50>0, c300>0) -> rows")
for k, v in sorted(clean.items(), key=lambda kv: -kv[1])[:8]:
    print(f"  {str(k):40s} {v:8d}")
print()

# ------------------------------------------------------------------ modern
per = collections.defaultdict(list)
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
        toks = frozenset(m["acronym"] for m in data.get("mods", []))
        if canonicalize(toks, NEUTRAL) not in ALLOW:
            continue
        stats = data.get("statistics", {})
        total = float(sum(stats.values()))
        if total <= 0:
            continue
        num = float(sum(stats.get(k, 0) * w for k, w in CLASSIC))
        per["+".join(sorted(toks))].append(loss_of(num, total))

report("modern", per)
