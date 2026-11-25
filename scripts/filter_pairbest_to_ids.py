#!/usr/bin/env python
"""Filter the full pairbest table to the Protocol-D eval pool.

Protocol D (d2v_main.splits --protocol mapset_holdout) holds out whole
beatmapsets.  splits.py writes eval_ids.txt as the 5,462 *cached* charts of the
held-out sets.  That list also contains 258 cached charts with no pairbest rows
and cannot contain the 32 pairbest charts of the held-out sets that are missing
from data/osu_cache, so a plain chart-id filter yields 5,204 charts /
468,646 rows instead of the eval pool used by the probes.

Default mode "mapset" uses the held-out mapset list written next to the split
(splits/mapset_holdout/eval_mapsets.txt) and keeps every full-pairbest row whose
chart belongs to one of those sets: 5,236 charts / 468,702 rows / 18,590 player
segments (17,751 base players).  This reproduces the reference
data/processed/step1_mclean25_pairbest.parquet exactly.  Without an
eval_mapsets.txt the mapsets are derived from --ids + --meta, which loses the
17 held-out sets that have no cached chart.

--mode ids keeps only rows whose beatmap_id is literally in --ids (used only
when a pairbest-chart-level id list such as eval_pairbest_ids.txt is passed).

Usage
-----
python scripts/filter_pairbest_to_ids.py \
  --in data/processed/step1_full_pairbest.parquet \
  --ids data2vec_main/splits/mapset_holdout/eval_ids.txt \
  --out data/processed/step1_mclean25_pairbest.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Reference numbers of the default Protocol-D artifact (mode=mapset).
EXPECT_ROWS = 468702
EXPECT_CHARTS = 5236
EXPECT_PLAYER_SEGMENTS = 18590


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path,
                    default=Path("data/processed/step1_full_pairbest.parquet"))
    ap.add_argument("--ids", type=Path,
                    default=Path("data2vec_main/splits/mapset_holdout/eval_ids.txt"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/processed/step1_mclean25_pairbest.parquet"))
    ap.add_argument("--meta", type=Path,
                    default=Path("data/processed/beatmap_meta_full.csv"),
                    help="beatmap meta csv with beatmap_id/beatmapset_id")
    ap.add_argument("--mapsets", type=Path,
                    default=Path("data2vec_main/splits/mapset_holdout/eval_mapsets.txt"),
                    help="held-out beatmapset ids written by d2v_main.splits "
                         "mapset_holdout; when missing, mapsets are derived "
                         "from --ids + --meta")
    ap.add_argument("--mode", choices=["mapset", "ids"], default="mapset",
                    help="mapset: keep all pairbest charts of the mapsets named "
                         "by --ids; ids: strict beatmap_id membership")
    ap.add_argument("--no-check", action="store_true",
                    help="do not assert the reference row/chart counts")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    df = pd.read_parquet(args.inp)
    if "beatmap_id" not in df.columns:
        raise SystemExit(str(args.inp) + ": no beatmap_id column")
    ids = [int(x) for x in args.ids.read_text().split()]
    if not ids:
        raise SystemExit(str(args.ids) + ": empty id file")

    if args.mode == "mapset":
        meta = pd.read_csv(args.meta, usecols=["beatmap_id", "beatmapset_id"])
        ids_set = set(ids)
        if args.mapsets.exists():
            ids_sets = set(int(x) for x in args.mapsets.read_text().split())
            print(f"[filter] mapsets from {args.mapsets}", flush=True)
        else:
            # fallback: the sets named by the (cache-level) id list.  Loses the
            # held-out sets that have no cached chart, so the reference run
            # should pass --mapsets (or rely on the splits.py output).
            ids_sets = set(meta.loc[meta["beatmap_id"].isin(ids_set),
                                    "beatmapset_id"].dropna().tolist())
        charts = pd.Series(df["beatmap_id"].unique(), name="beatmap_id")
        chart_sets = charts.to_frame().merge(meta, on="beatmap_id", how="left")
        keep = set(charts[chart_sets["beatmapset_id"].isin(ids_sets)].astype(int))
        # charts explicitly requested but absent from meta are kept as-is
        keep |= ids_set & set(charts.astype(int))
        print(f"[filter] mode=mapset ids={len(ids_set)} "
              f"mapsets={len(ids_sets)} pairbest_chart_ids={len(keep)}", flush=True)
    else:
        keep = set(ids)
        print(f"[filter] mode=ids ids={len(keep)}", flush=True)

    out = df.loc[df["beatmap_id"].isin(keep)]
    n_rows = int(len(out))
    n_charts = int(out["beatmap_id"].nunique())
    n_segments = int(out["player_id"].nunique()) if "player_id" in out else -1
    n_players = (int(out["player_id_base"].nunique())
                 if "player_id_base" in out else -1)
    print(f"[filter] rows={n_rows} charts={n_charts} "
          f"player_segments={n_segments} players={n_players}", flush=True)

    ok = True
    if args.mode == "mapset" and not args.no_check:
        got = (n_rows, n_charts, n_segments)
        ref = (EXPECT_ROWS, EXPECT_CHARTS, EXPECT_PLAYER_SEGMENTS)
        ok = got == ref
        print(f"[check] rows/charts/player_segments {got} vs reference {ref}: "
              + ("OK" if ok else "MISMATCH"), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"[write] {args.out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
