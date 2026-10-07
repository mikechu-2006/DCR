"""Build the (player, chart, log-loss) dataset from the parsed osu! dumps.

Chart = (beatmap_id, rate bucket).  Mod policy (per user decision):
  * CL (classic scoring) and MR (mirror) are not gameplay mods -> stripped / ignored
  * DT and NC are the same -> one "DT" bucket, treated as a *new chart*
  * HT gets its own bucket
  * NF is harmless -> kept; HD/FL/EZ/HR/... are excluded from training (they change what is judged)
  * cells whose median accuracy < 0.90 are flagged low_acc ("random hitting") and excluded

Outputs (data/processed/):
  plays_4k.parquet    one row per play (legacy + modern union, deduped)
  charts_4k.parquet   one row per (user, chart)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

INTERIM = Path("data/interim")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

RATE_MASK = 64 | 256 | 512               # DT | HT | NC
LEGACY_BITS = [(1, "NF"), (2, "EZ"), (4, "TD"), (8, "HD"), (16, "HR"), (32, "SD"), (64, "DT"),
               (128, "RX"), (256, "HT"), (512, "NC"), (1024, "FL"), (2048, "AT"), (4096, "SO"),
               (8192, "AP"), (16384, "PF"), (32768, "4K"), (65536, "5K"), (131072, "6K"),
               (262144, "7K"), (524288, "8K"), (1048576, "FI"), (2097152, "RD"), (4194304, "CN"),
               (8388608, "TP"), (16777216, "9K"), (33554432, "CO"), (67108864, "1K"),
               (134217728, "3K"), (268435456, "2K")]
DROP_MODS = {"HD", "FL", "EZ", "HR", "AT", "RX", "AP", "SO"}   # NF is harmless and stays
MIN_ACC = 0.90                                                      # median-accuracy floor per cell


def decode_legacy(bitmask: int) -> str:
    return "+".join(name for bit, name in LEGACY_BITS if bitmask & bit)


def normalise(mods: str) -> str:
    toks = [t for t in mods.split("+") if t and t not in ("CL", "MR")]
    if "NC" in toks and "DT" not in toks:
        toks.append("DT")
    return "+".join(sorted(set(toks)))


def rate_bucket(mods_norm: str) -> int:
    t = set(mods_norm.split("+"))
    if "DT" in t or "NC" in t:
        return 1
    if "HT" in t:
        return 2
    return 0


def main():
    bm = pd.read_csv(INTERIM / "beatmaps.csv")
    bm4 = bm[(bm.playmode == 3) & (bm["keys"] == 4)].copy()
    bm4_ids = set(bm4.beatmap_id.astype("int64"))
    print("4K mania beatmaps:", len(bm4))

    # ---------------- modern source ----------------
    s = pd.read_csv(INTERIM / "scores_detail.csv")
    for c in ["n_perfect", "n_great", "n_good", "n_ok", "n_meh", "n_miss", "max_total"]:
        s[c] = pd.to_numeric(s[c], errors="coerce").fillna(0).astype("int64")
    s = s[s.beatmap_id.isin(bm4_ids)].copy()
    m = pd.DataFrame({
        "score_id": s.id.astype("int64"), "legacy_score_id": pd.to_numeric(s.legacy_score_id, errors="coerce"),
        "user_id": s.user_id, "beatmap_id": s.beatmap_id,
        "c320": s.n_perfect, "c300": s.n_great, "c200": s.n_good, "c100": s.n_ok,
        "c50": s.n_meh, "cmiss": s.n_miss, "max_total": s.max_total,
        "mods_raw": s.mods.fillna(""), "date": s.ended_at, "source": "modern",
        "pp": pd.to_numeric(s.pp, errors="coerce"),
    })

    # ---------------- legacy source ----------------
    h = pd.read_csv(INTERIM / "mania_high.csv")
    for c in ["count50", "count100", "count300", "countmiss", "countgeki", "countkatu", "enabled_mods", "pp"]:
        h[c] = pd.to_numeric(h[c], errors="coerce").fillna(0).astype("int64")
    h = h[h.beatmap_id.isin(bm4_ids)].copy()
    l = pd.DataFrame({
        "score_id": h.score_id.astype("int64"), "legacy_score_id": np.nan,
        "user_id": h.user_id, "beatmap_id": h.beatmap_id,
        "c320": h.countgeki, "c300": h.count300, "c200": h.countkatu, "c100": h.count100,
        "c50": h.count50, "cmiss": h.countmiss, "max_total": 0,
        "mods_raw": [decode_legacy(int(v)) for v in h.enabled_mods],
        "date": h.date, "source": "legacy", "pp": h.pp,
    })
    print("modern 4K rows:", len(m), "legacy 4K rows:", len(l))

    dup_ids = set(m.legacy_score_id.dropna().astype("int64"))
    keep = ~l.score_id.isin(dup_ids)
    print("legacy rows duplicated in modern (dropped):", int((~keep).sum()))
    plays = pd.concat([m, l[keep]], ignore_index=True)

    # ---------------- mods / chart id ----------------
    plays["mods_norm"] = plays.mods_raw.map(normalise)
    plays["rate"] = plays.mods_norm.map(rate_bucket)
    plays["chart_id"] = plays.beatmap_id.astype("int64") * 4 + plays.rate
    plays["dropped_mod"] = plays.mods_norm.str.split("+").apply(lambda t: bool(set(t) & DROP_MODS))
    plays["clean"] = ~plays.dropped_mod
    print("rate buckets:", plays.rate.value_counts().to_dict())
    print("plays with excluded mods (FL/EZ/HR/NF/...):", int(plays.dropped_mod.sum()),
          "-> clean plays:", int(plays.clean.sum()))

    # ---------------- labels ----------------
    j = plays[["c320", "c300", "c200", "c100", "c50", "cmiss"]]
    plays["total"] = j.sum(axis=1)
    plays = plays[plays.total > 0].copy()
    t = plays.total.astype(float)
    plays["acc_flat"] = (300 * (plays.c300 + plays.c320) + 200 * plays.c200 + 100 * plays.c100 + 50 * plays.c50) / (300 * t)
    plays["acc_v2"] = (305 * plays.c320 + 300 * plays.c300 + 200 * plays.c200 + 100 * plays.c100 + 50 * plays.c50) / (305 * t)
    for name, acc, q in (("flat", plays.acc_flat, 300.0), ("v2", plays.acc_v2, 305.0)):
        loss = 1.0 - acc
        plays[f"loss_{name}"] = np.log(np.clip(loss, 25.0 / (q * t), None))
        plays[f"perfect_{name}"] = loss <= 0

    plays = plays.merge(bm4[["beatmap_id", "beatmapset_id", "star", "max_combo", "countTotal",
                             "playcount", "passcount", "approved", "version"]], on="beatmap_id", how="left")
    pc = pd.read_csv(INTERIM / "playcount.csv")
    pc["playcount"] = pd.to_numeric(pc["playcount"], errors="coerce")
    plays = plays.merge(pc.rename(columns={"playcount": "attempts"}), on=["user_id", "beatmap_id"], how="left")
    plays.to_parquet(OUT / "plays_4k.parquet", index=False)
    print("wrote plays_4k.parquet", plays.shape)

    # ---------------- chart-level cells ----------------
    def build_cells(df, tag):
        g = df.groupby(["user_id", "chart_id"], sort=False)
        agg = g.agg(
            beatmap_id=("beatmap_id", "first"), rate=("rate", "first"),
            n_scores=("score_id", "size"),
            acc_flat_med=("acc_flat", "median"), acc_flat_max=("acc_flat", "max"),
            acc_flat_mean=("acc_flat", "mean"), acc_flat_min=("acc_flat", "min"),
            acc_v2_med=("acc_v2", "median"), acc_v2_max=("acc_v2", "max"),
            loss_flat_med=("loss_flat", "median"), loss_flat_max=("loss_flat", "max"),
            loss_v2_med=("loss_v2", "median"), loss_v2_max=("loss_v2", "max"),
            total_med=("total", "median"), star=("star", "first"), max_combo=("max_combo", "first"),
            attempts=("attempts", "max"), map_playcount=("playcount", "first"),
            date_first=("date", "min"), date_last=("date", "max"),
        ).reset_index()
        agg["attempts_per_score"] = agg.attempts / agg.n_scores
        agg.to_parquet(OUT / f"cells_{tag}.parquet", index=False)
        print(f"wrote cells_{tag}.parquet {agg.shape}  charts={agg.chart_id.nunique()}")
        return agg

    clean = plays[plays.clean]
    def add_flags(c):
        c["low_acc"] = c.acc_v2_med < MIN_ACC
        c["primary"] = ~c.low_acc
        return c

    cells = add_flags(build_cells(clean, "chart"))          # all rate buckets, clean mods
    add_flags(build_cells(clean[clean.rate == 0], "chart_nm"))
    add_flags(build_cells(clean[clean.rate == 1], "chart_dt"))
    cells.to_parquet(OUT / "charts_4k.parquet", index=False)
    print("cells with median acc_v2 < %.2f (dropped as 'random hitting'): %d / %d = %.3f"
          % (MIN_ACC, int(cells.low_acc.sum()), len(cells), cells.low_acc.mean()))
    hi = cells[cells.primary]
    print("primary cells:", len(hi), "| users:", hi.user_id.nunique(), "| charts:", hi.chart_id.nunique())

    for tag, df in (("nm", clean[clean.rate == 0]), ("dt", clean[clean.rate == 1]),
                    ("nm+dt", clean[clean.rate != 2])):
        c = df.groupby("beatmap_id").user_id.nunique()
        print(f"[{tag}] plays={len(df)} users={df.user_id.nunique()} maps={df.beatmap_id.nunique()} "
              f"median players/map={c.median():.0f}")


if __name__ == "__main__":
    main()
