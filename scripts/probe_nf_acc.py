#!/usr/bin/env python3
"""One-off probe: how much of the NF tail is actually high-ACC?

`NF` (NoFail) is the one neutral mod that is *not* a pure relabelling: it lets a
play finish that would otherwise have been abandoned, so it adds rows with no
`CL` analogue.  The open question is how many of those rows are genuinely bad
(which is the point of the mod) versus just ordinary plays where the player
happened to have NF on.

Counts are on the **deduplicated** set, i.e. exactly the NF rows that land in
`step0_{tag}.csv` -- modern CL+NF rows, plus legacy CL+NF rows that the modern
table does not already have.

    python scripts/probe_nf_acc.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_dump import iter_insert_lines, rows_simple  # noqa: E402
import step0_preprocess as s  # noqa: E402

BANDS = [(0.0, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.85),
         (0.85, 0.90), (0.90, 0.95), (0.95, 0.99), (0.99, 0.999), (0.999, 1.01)]
BITS = s.parse_legacy_bits(s.LEGACY_MOD_BITS)
ALLOW = s.parse_allow(s.MOD_WHITELIST)
NEUTRAL = s.parse_names(s.NEUTRAL_MODS)
ACC90 = np.log(0.10)          # loss >= this  <=>  ACC >= 0.90


def has_nf(tokens) -> bool:
    return "NF" in tokens


def hist(losses: list) -> dict:
    acc = 1.0 - np.exp(np.array(losses))
    return {b: int(((acc >= b[0]) & (acc < b[1])).sum()) for b in BANDS}


def report(tag: str, nf: list, other: list):
    print(f"===== {tag} =====")
    n_nf, n_other = len(nf), len(other)
    for name, vals in (("NF on", nf), ("NF off", other)):
        if not vals:
            continue
        a = 1.0 - np.exp(np.array(vals))
        hi = int((a >= 0.90).sum())
        print(f"{name:7s} rows={len(vals):>9d}  mean ACC={a.mean():.5f} "
              f"median={np.median(a):.5f}  ACC>=0.90: {hi:>8d} = {hi/len(vals)*100:6.2f}%"
              f"   (ACC>=0.95: {(a>=0.95).mean()*100:5.2f}%)")
    print()
    print(f"{'ACC band':>14s} {'NF on':>10s} {'share':>8s} | "
          f"{'NF off':>10s} {'share':>8s}")
    hn, ho = hist(nf), hist(other)
    for b in BANDS:
        label = f"[{b[0]:.3f},{b[1]:.3f})"
        print(f"{label:>14s} {hn[b]:>10d} {hn[b]/max(n_nf,1)*100:7.2f}% | "
              f"{ho[b]:>10d} {ho[b]/max(n_other,1)*100:7.2f}%")
    print()
    # cumulative from the low-ACC end.  `loss` DECREASES with ACC, so
    # "loss >= log(1-x)" is the same set as "ACC <= x" -- stated in ACC terms to
    # avoid exactly that sign trap.
    ln, lo = np.array(nf), np.array(other)
    print("cumulative from the low-ACC end (loss is a DECREASING function of ACC):")
    print(f"{'ACC <=':>9s} {'loss >=':>10s} {'NF on':>9s} {'of NF':>8s} | "
          f"{'NF off':>10s} {'of non-NF':>10s} | {'NF share':>9s} {'vs base':>8s}")
    base = n_nf / max(n_nf + n_other, 1) * 100
    for acc_thr in (0.10, 0.40, 0.63, 0.78, 0.90, 0.95, 0.99, 0.999):
        loss_thr = np.log1p(-acc_thr)
        k_n = int((ln >= loss_thr).sum())
        k_o = int((lo >= loss_thr).sum())
        share = k_n / max(k_n + k_o, 1) * 100
        print(f"{acc_thr:>9.3f} {loss_thr:>10.4f} {k_n:>9d} "
              f"{k_n/max(n_nf,1)*100:7.2f}% | {k_o:>10d} "
              f"{k_o/max(n_other,1)*100:9.3f}% | {share:8.2f}% "
              f"{share/base:7.1f}x")
    print()
    print(f"ACC range: NF on [{1-np.exp(ln.max()):.6f}, {1-np.exp(ln.min()):.6f}]  "
          f"NF off [{1-np.exp(lo.max()):.6f}, {1-np.exp(lo.min()):.6f}]")
    print(f"base rate: NF is {base:.3f}% of all kept rows "
          f"({n_nf} of {n_nf + n_other})")
    print()


# ------------------------------------------------------------------ 1k
DUMP1 = Path("data/raw/extracted/2026_09_01_performance_mania_top_1000")
ids, _ = s.read_beatmaps(DUMP1, "3", "4")

nf, other, drop = [], [], set()
for line in iter_insert_lines(DUMP1 / "scores.sql"):
    body = line[line.index("VALUES") + 6:].strip()[1:]
    if body.endswith(");"):
        body = body[:-2]
    for row in body.split("),("):
        parts = row.split(",")
        if len(parts) < 12 + s.N_TAIL:
            continue
        head, tail = parts[:12], parts[-s.N_TAIL:]
        bid = int(head[3])
        if bid not in ids:
            continue
        raw = ",".join(parts[12:-s.N_TAIL]).strip()
        if raw[:1] == "'":
            raw = raw[1:-1]
        data = json.loads(raw.replace('\\"', '"'))
        toks = frozenset(m["acronym"] for m in data.get("mods", []))
        if s.canonicalize(toks, NEUTRAL) not in ALLOW:
            continue
        lid = tail[1]
        if lid.isdigit():
            drop.add(int(lid))
        stats = data.get("statistics", {})
        total = float(sum(stats.values()))
        if total <= 0:
            continue
        num = float(sum(stats.get(k, 0) * w for k, w in s.CLASSIC))
        (nf if has_nf(toks) else other).append(s.loss_of(num, total))

n_mod, n_leg_nf = len(nf) + len(other), 0
for line in iter_insert_lines(DUMP1 / "osu_scores_mania_high.sql"):
    for r in rows_simple(line):
        bid = int(r[1])
        if bid not in ids:
            continue
        mods = s.decode_legacy_mods(int(r[13]), BITS)
        if mods is None:
            continue
        toks = s.CL_SET | mods
        if s.canonicalize(toks, NEUTRAL) not in ALLOW:
            continue
        if int(r[0]) in drop:
            continue
        total = float(sum(int(r[i]) for i in s.LEGACY_TOTAL))
        if total <= 0:
            continue
        num = float(sum(int(r[i]) * w for i, w in s.LEGACY_NUM))
        if has_nf(toks):
            nf.append(s.loss_of(num, total))
            n_leg_nf += 1
        else:
            other.append(s.loss_of(num, total))

report("1k (deduplicated, both sources)", nf, other)
print(f"of the {len(nf)} NF rows, {n_leg_nf} came from the legacy table "
      f"(the rest are modern; both were already counted above)\n")

# ------------------------------------------------------------------ 10k
DUMP10 = Path("data/raw/extracted_10k/2026_09_01_performance_mania_top_10000")
nf10, other10 = [], []
for line in iter_insert_lines(DUMP10 / "osu_scores_mania_high.sql"):
    for r in rows_simple(line):
        bid = int(r[1])
        if bid not in ids:
            continue
        mods = s.decode_legacy_mods(int(r[13]), BITS)
        if mods is None:
            continue
        toks = s.CL_SET | mods
        if s.canonicalize(toks, NEUTRAL) not in ALLOW:
            continue
        total = float(sum(int(r[i]) for i in s.LEGACY_TOTAL))
        if total <= 0:
            continue
        num = float(sum(int(r[i]) * w for i, w in s.LEGACY_NUM))
        (nf10 if has_nf(toks) else other10).append(s.loss_of(num, total))

report("10k (legacy only, no modern table)", nf10, other10)
