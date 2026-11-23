#!/usr/bin/env python
"""Union several step0 play tables into one play table (dedupe + renumber).

This is the dataset-level sibling of scripts/step0_merge_snapshots.py.  The
randommerge table merges snapshots of the *same* dump series; this script merges
different datasets (1k, 10k, randommerge) into the `full` dataset.

Why play level and not pair level: step1/pairbest derives n_plays, playcount_cur
and the 365-day gap split from the whole play list.  Concatenating step1 tables
would double-count the plays that two datasets already share (1k is almost a
subset of 10k), freeze the pair counts at dataset scope, and cannot dedupe
identically-timed plays.  So the union happens here, on play rows, and step1
runs afterwards exactly as it does for 1k / 10k / randommerge.

Dedupe key: (player_id, beatmap_id, timestamp) -- one-second granularity makes
a play unique per (player, chart).  Duplicate keys keep the earliest source,
then the lowest loss; disagreements are counted in the report (`--strict`
aborts instead).  playcount_cur is recomputed on the merged timeline.

Output columns (same shape as step0_randommerge.parquet):

    player_id, beatmap_id, timestamp, playcount_cur, loss,
    source_first, source_last, n_sources

Optional `--meta` inputs are merged into one beatmap_meta table: dedupe by
beatmap_id, later sources win (they carry newer playcount/star/last_update).
That keeps `beatmap_meta_full.csv` self-describing for step0/chart_content;
step1 does not read meta.

Usage:
  python scripts/step0_union_datasets.py \
      --play data/processed/step0_1k.csv=1k \
      --play data/processed/step0_10k.parquet=10k \
      --play data/processed/step0_randommerge.parquet=randommerge \
      --meta data/processed/beatmap_meta_1k.csv=1k \
      --meta data/processed/beatmap_meta_10k.csv=10k \
      --meta data/processed/beatmap_meta_20261105.csv=randommerge \
      --out data/processed/step0_full.parquet \
      --meta-out data/processed/beatmap_meta_full.csv \
      --report data/processed/step0_full_union.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step0_merge_snapshots import COLS, KEY, merge_frames  # noqa: E402

INT32_MAX = int(np.iinfo(np.int32).max)


def parse_inputs(items: list[str]) -> list[tuple[str, Path]]:
    out = []
    for item in items:
        if "=" in item:
            path, tag = item.split("=", 1)
        else:
            path, tag = item, Path(item).stem
        out.append((tag, Path(path)))
    return out


def read_plays(tag: str, path: Path) -> pd.DataFrame:
    """Read one step0 play table, validate it and normalise dtypes/units."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=COLS)
    elif path.suffix in (".csv", ".txt"):
        df = pd.read_csv(path, usecols=COLS)
    else:
        raise SystemExit(f"unsupported play table format: {path}")
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    if df[COLS].isna().any().any():
        raise SystemExit(f"{path}: null values in the play table")
    df = df[COLS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].isna().any():
        raise SystemExit(f"{path}: unparseable timestamps")
    ns = df["timestamp"].astype("datetime64[ns]").astype("int64").to_numpy()
    subsec = int(np.abs(ns % 1_000_000_000).max())
    if subsec:
        raise SystemExit(
            f"{path}: timestamps are not whole seconds (max sub-second remainder "
            f"{subsec} ns); the (player, chart, timestamp) dedupe key would miss plays")
    for col in ("player_id", "beatmap_id", "playcount_cur"):
        if int(df[col].max()) > INT32_MAX:
            raise SystemExit(f"{path}: {col} exceeds int32; refusing to downcast")
        df[col] = df[col].astype("int32")
    df["loss"] = df["loss"].astype("float32")
    if not np.isfinite(df["loss"].to_numpy()).all():
        raise SystemExit(f"{path}: non-finite loss")
    if not (df["loss"] < 0).all():
        raise SystemExit(f"{path}: loss must be negative (log(1-ACC))")
    print(f"[union] {tag}: {len(df):,} plays / {df.player_id.nunique():,} players / "
          f"{df.beatmap_id.nunique():,} charts", flush=True)
    return df


def pair_keys(players: np.ndarray, beatmaps: np.ndarray, k: int) -> np.ndarray:
    return players.astype(np.int64) * np.int64(k) + beatmaps.astype(np.int64)


def pair_counts(df: pd.DataFrame, k: int) -> pd.Series:
    keys = pair_keys(df["player_id"].to_numpy(), df["beatmap_id"].to_numpy(), k)
    codes, uniq = pd.factorize(keys, sort=False)
    counts = np.bincount(codes, minlength=uniq.size)
    return pd.Series(counts, index=uniq, name="n_plays")


def merge_meta(items: list[tuple[str, Path]], out: Path | None) -> dict:
    """Dedupe beatmap meta by beatmap_id (later source wins) and report conflicts."""
    frames, ref_cols = [], None
    per_source = {}
    for tag, path in items:
        if not path.exists():
            raise SystemExit(f"missing meta input: {path}")
        df = pd.read_csv(path)
        if ref_cols is None:
            ref_cols = list(df.columns)
        elif list(df.columns) != ref_cols:
            raise SystemExit(f"{path}: columns differ from {ref_cols}")
        if df["beatmap_id"].duplicated().any():
            raise SystemExit(f"{path}: duplicate beatmap_id in meta table")
        df["_order"] = len(frames)
        frames.append(df)
        per_source[tag] = {"rows": int(len(df)), "path": str(path)}
        print(f"[meta] {tag}: {len(df):,} charts", flush=True)

    all_meta = pd.concat(frames, ignore_index=True)
    value_cols = [c for c in ref_cols if c != "beatmap_id"]
    conflicts = {}
    grouped = all_meta.groupby("beatmap_id", observed=True)[value_cols].nunique()
    for col in value_cols:
        n = int((grouped[col] > 1).sum())
        if n:
            conflicts[col] = n
    merged = (all_meta.sort_values("_order")
              .drop_duplicates(subset=["beatmap_id"], keep="last")
              .drop(columns=["_order"])
              .sort_values("beatmap_id")[ref_cols].reset_index(drop=True))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out, index=False)
        print(f"[meta] wrote {out} ({len(merged):,} charts)", flush=True)

    seen = (all_meta.groupby("beatmap_id", observed=True)["_order"].nunique().value_counts()
            .sort_index())
    return {
        "inputs": {tag: str(path) for tag, path in items},
        "per_source": per_source,
        "merged": {"charts": int(len(merged)), "columns": ref_cols,
                   "charts_seen_in_k_sources": {str(k): int(v) for k, v in seen.items()},
                   "value_conflicts": conflicts,
                   "duplicate_policy": "later --meta input wins"},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play", action="append", required=True, metavar="PATH=TAG",
                    help="step0 play table (parquet or csv) and its dataset label; repeatable")
    ap.add_argument("--meta", action="append", default=[], metavar="PATH=TAG",
                    help="optional beatmap meta table and label; repeatable")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--meta-out", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--strict", action="store_true",
                    help="abort if a duplicate play key disagrees on loss")
    args = ap.parse_args()

    t0 = time.time()
    plays = parse_inputs(args.play)
    metas = parse_inputs(args.meta)
    if metas and args.meta_out is None:
        raise SystemExit("--meta-out is required when --meta is given")

    frames, tags, per_source = [], [], {}
    for tag, path in plays:
        if not path.exists():
            raise SystemExit(f"missing play input: {path}")
        if tag in tags:
            raise SystemExit(f"duplicate source tag: {tag}")
        df = read_plays(tag, path)
        frames.append(df)
        tags.append(tag)
        per_source[tag] = {
            "path": str(path),
            "plays": int(len(df)),
            "players": int(df["player_id"].nunique()),
            "charts": int(df["beatmap_id"].nunique()),
            "t_min": str(df["timestamp"].min()),
            "t_max": str(df["timestamp"].max()),
        }

    k = max(int(df["beatmap_id"].max()) for df in frames) + 1
    counts = {tag: pair_counts(df, k) for tag, df in zip(tags, frames)}
    for tag in tags:
        per_source[tag]["pairs"] = int(len(counts[tag]))

    # duplicate plays / loss consistency between every pair of sources
    pairwise = {}
    for i, a in enumerate(tags):
        for j in range(i + 1, len(tags)):
            b = tags[j]
            m = frames[i].merge(frames[j], on=KEY, how="inner", suffixes=("_a", "_b"))
            diff = (m["loss_a"].astype("float64") - m["loss_b"].astype("float64")).abs()
            shared_pairs = np.intersect1d(counts[a].index.to_numpy(), counts[b].index.to_numpy())
            pairwise[f"{a}|{b}"] = {
                "duplicate_plays": int(len(m)),
                "loss_conflicts": int((diff > 1e-4).sum()),
                "loss_max_abs_diff": float(diff.max()) if len(m) else 0.0,
                "pairs_in_both": int(shared_pairs.size),
                "play_overlap_frac_of_smaller": round(
                    len(m) / max(min(len(frames[i]), len(frames[j])), 1), 6),
            }
            print(f"[union] {a}|{b}: {len(m):,} duplicate plays, "
                  f"{int((diff > 1e-4).sum()):,} loss conflicts, "
                  f"{shared_pairs.size:,} shared pairs", flush=True)

    merged, stats = merge_frames(frames, tags, prefix="source")
    conflicts = stats["loss_conflicts"]
    if conflicts and args.strict:
        print(f"[union] aborting: {conflicts} duplicate keys disagree on loss",
              file=sys.stderr)
        return 3
    if conflicts:
        print(f"[union] warning: {conflicts} duplicate keys disagree on loss; "
              f"kept earliest source / lowest loss", file=sys.stderr)

    keys = pair_keys(merged["player_id"].to_numpy(), merged["beatmap_id"].to_numpy(), k)
    n_pairs = int(np.unique(keys).size)
    n_play_keys = len(merged)
    if merged.duplicated(subset=KEY).any():
        raise SystemExit("[union] internal error: duplicate keys survived the merge")
    if int(merged["playcount_cur"].min()) != 1:
        raise SystemExit("[union] internal error: playcount_cur starts below 1")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".csv":
        merged.to_csv(args.out, index=False)
    elif args.out.suffix == ".parquet":
        merged.to_parquet(args.out, index=False)
    else:
        raise SystemExit(f"--out must end in .parquet or .csv, got {args.out}")
    print(f"[union] wrote {args.out} ({len(merged):,} plays / {n_pairs:,} pairs)",
          flush=True)

    n_sources = merged["n_sources"].to_numpy()
    union_counts = pair_counts(merged, k)

    # A pair that had exactly one play in every source that saw it, but >=2 after
    # the union, was *not* creatable from any single dataset.  These are genuine
    # distinct plays (identical plays were deduped), unlike randommerge's
    # snapshot merge where 0 pairs resurrected.
    resurrect = {}
    for tag in tags:
        cnt = counts[tag]
        u = union_counts.reindex(cnt.index).to_numpy()
        single = cnt.to_numpy() == 1
        newly = single & (u >= 2)
        single_everywhere = newly.copy()
        for other in tags:
            if other == tag:
                continue
            o = counts[other].reindex(cnt.index).fillna(0).to_numpy()
            single_everywhere &= (o == 0) | (o == 1)
        resurrect[tag] = {
            "pairs": int(len(cnt)),
            "single_play_pairs": int(single.sum()),
            "single_but_ge2_in_union": int(newly.sum()),
            "single_in_every_source_ge2_in_union": int(single_everywhere.sum()),
        }
        print(f"[union] {tag}: {newly.sum():,} single-play pairs gain a second play "
              f"(present in exactly one other source: {single_everywhere.sum():,})",
              flush=True)

    meta_report = merge_meta(metas, args.meta_out) if metas else None
    if meta_report is not None:
        union_charts = int(merged["beatmap_id"].nunique())
        meta_charts = int(meta_report["merged"]["charts"])
        # charts in the union that no meta table describes
        meta_ids = set(pd.read_csv(metas[0][1], usecols=["beatmap_id"])["beatmap_id"].tolist())
        for _, path in metas[1:]:
            meta_ids.update(pd.read_csv(path, usecols=["beatmap_id"])["beatmap_id"].tolist())
        have = merged["beatmap_id"].drop_duplicates()
        missing = int((~have.isin(meta_ids)).sum())
        meta_report["coverage_of_union_charts"] = {
            "union_charts": union_charts,
            "charts_in_merged_meta": meta_charts,
            "union_charts_without_meta": missing,
            "coverage": round((union_charts - missing) / union_charts, 6),
        }
        print(f"[meta] covers {union_charts - missing:,}/{union_charts:,} union charts",
              flush=True)

    report = {
        "method": "union of step0 play tables; dedupe on (player_id, beatmap_id, timestamp); "
                  "playcount_cur recomputed on the merged timeline",
        "inputs": {tag: str(path) for tag, path in plays},
        "merge_order": tags,
        "rules": {
            "dedupe_key": KEY,
            "duplicate_policy": stats["duplicate_play_rule"],
            "normalisation": "timestamp -> datetime64[ns] (whole seconds required); "
                             "player_id/beatmap_id/playcount_cur -> int32; loss -> float32",
            "playcount_cur": "rank of the play inside (player_id, beatmap_id) ordered by "
                             "timestamp, starting at 1 -- not copied from any source",
            "provenance": "source_first / source_last / n_sources over the --play inputs",
            "step1": "run step1/pairbest on this table; pair rules are unchanged",
        },
        "per_source": per_source,
        "pairwise": pairwise,
        "duplicates": {
            "rows_on_duplicate_keys": stats["rows_on_duplicate_keys"],
            "duplicate_keys": stats["duplicate_keys"],
            "loss_conflicts": conflicts,
        },
        "merged": {
            "plays": int(len(merged)),
            "players": int(merged["player_id"].nunique()),
            "charts": int(merged["beatmap_id"].nunique()),
            "pairs": n_pairs,
            "input_rows": int(sum(len(f) for f in frames)),
            "duplicate_rows_dropped": int(sum(len(f) for f in frames) - len(merged)),
            "retained_frac": round(len(merged) / sum(len(f) for f in frames), 6),
            "pairs_with_ge2_plays": int((union_counts >= 2).sum()),
            "t_min": str(merged["timestamp"].min()),
            "t_max": str(merged["timestamp"].max()),
            "plays_seen_in_k_sources": {
                str(k_): int(v) for k_, v in
                pd.Series(n_sources).value_counts().sort_index().items()},
        },
        "resurrected_pairs": resurrect,
        "beatmap_meta": meta_report,
        "checks": {
            "unique_play_keys": True,
            "loss_negative": bool((merged["loss"] < 0).all()),
            "playcount_starts_at_1": True,
            "all_sources_nonempty": all(v["plays"] > 0 for v in per_source.values()),
            "n_sources_within_bounds": bool(n_sources.min() >= 1 and n_sources.max() <= len(tags)),
        },
        "output": str(args.out),
        "meta_output": str(args.meta_out) if args.meta_out else None,
        "seconds": round(time.time() - t0, 1),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"[union] wrote {args.report}", flush=True)
    print(json.dumps(report["merged"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
