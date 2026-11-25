#!/usr/bin/env python3
"""step1_attach_judgements -- attach six-grade counts to step1 pairbest rows.

step1/pairbest selects, per (player segment, chart), the play with the lowest
loss (highest ACC).  This script joins that exact play's judgement counts from
step0_judge_full.parquet by (player_id, beatmap_id, timestamp).

It also checks the counts against
  * beatmap_meta.count_total (the chart's max combo / object count), and
  * the .osu object count when --osu-dir is given.

Output = the step1 schema + n_max/n300/n200/n100/n50/n_miss/n_total +
judge_src/judge_ok/count_ok/osu_ok/label_ok/count_total/osu_objects.
Rows with label_ok=False are kept by default so the caller can report them;
pass --drop-mismatch to drop them before writing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcr.cm3p_offline import parse_osu  # noqa: E402

COUNT_COLS = ["n_max", "n300", "n200", "n100", "n50", "n_miss"]


def osu_object_count(osu_dir: Path, bid: int) -> int | None:
    p = osu_dir / f"{bid}.osu"
    if not p.exists():
        return None
    try:
        osu = parse_osu(str(p))
    except Exception:
        return None
    return sum(1 for o in osu.objects if o.kind in ("circle", "hold"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--judge", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--osu-dir", type=Path, default=Path("data/osu_cache"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--drop-mismatch", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    step1 = pd.read_parquet(args.inp)
    if args.limit:
        step1 = step1.head(args.limit).copy()
    print(f"[attach] step1 rows {len(step1)}", flush=True)
    judge = pd.read_parquet(args.judge)
    print(f"[attach] judge rows {len(judge)}", flush=True)

    # step0 uses the original player id; step1 may have renumbered segments.
    # Join on player_id_base when step1 carries it.
    if "player_id_base" in step1.columns:
        key = ["player_id_base", "beatmap_id", "timestamp"]
        judge = judge.rename(columns={"player_id": "player_id_base"})
    else:
        key = ["player_id", "beatmap_id", "timestamp"]
    keep = key + COUNT_COLS + ["n_total", "judge_src", "judge_ok"]
    lack = set(keep) - set(judge.columns)
    if lack:
        raise SystemExit(f"{args.judge}: missing {sorted(lack)}")
    judge = judge[keep].copy()
    for c in ("beatmap_id",) + (("player_id_base",) if "player_id_base" in key else ("player_id",)):
        judge[c] = judge[c].astype("int64")
        step1[c] = step1[c].astype("int64")

    out = step1.merge(judge, on=key, how="left", validate="one_to_one")
    out["judge_ok"] = out["judge_ok"].eq(True)
    out["judge_src"] = out["judge_src"].fillna("missing")

    meta = pd.read_csv(args.meta)
    meta = meta[["beatmap_id", "count_total"]].copy()
    meta["beatmap_id"] = meta["beatmap_id"].astype("int64")
    out = out.merge(meta, on="beatmap_id", how="left", validate="many_to_one")
    out["count_total"] = out["count_total"].astype("Int64")
    out["count_ok"] = (out["n_total"] == out["count_total"]).fillna(False)

    # .osu object count: parse each unique chart once.
    n_charts = out["beatmap_id"].nunique()
    objects: dict[int, int | None] = {}
    if args.osu_dir and args.osu_dir.exists():
        for i, bid in enumerate(out["beatmap_id"].drop_duplicates().tolist(), 1):
            objects[int(bid)] = osu_object_count(args.osu_dir, int(bid))
            if i % 2000 == 0:
                print(f"[attach] parsed {i}/{n_charts} .osu", flush=True)
    else:
        print(f"[attach] no osu dir {args.osu_dir}; osu_ok not checked", flush=True)
    out["osu_objects"] = out["beatmap_id"].map(objects).astype("Int64")
    out["osu_ok"] = (out["n_total"] == out["osu_objects"]).fillna(False)
    out["label_ok"] = out["judge_ok"] & out["count_ok"] & out["osu_ok"]

    n = len(out)
    n_judge = int(out.judge_ok.sum())
    n_count = int(out.count_ok.sum())
    n_osu = int(out.osu_ok.sum())
    n_label = int(out.label_ok.sum())
    print(f"[attach] judge_ok {n_judge/n:.4%} | count_ok {n_count/n:.4%} | "
          f"osu_ok {n_osu/n:.4%} | label_ok {n_label/n:.4%}", flush=True)

    by_split = (out.groupby("split")
                .agg(rows=("loss", "size"), judge_ok=("judge_ok", "sum"),
                     count_ok=("count_ok", "sum"), osu_ok=("osu_ok", "sum"),
                     label_ok=("label_ok", "sum")).reset_index())
    for c in ("judge_ok", "count_ok", "osu_ok", "label_ok"):
        by_split[c + "_rate"] = by_split[c] / by_split["rows"]

    if args.drop_mismatch:
        before = len(out)
        out = out[out.label_ok].copy()
        print(f"[attach] dropped {before-len(out)} rows with label_ok=False", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".part")
    out.to_parquet(tmp, index=False)
    tmp.replace(args.out)

    report = {
        "in": str(args.inp),
        "judge": str(args.judge),
        "out": str(args.out),
        "rows_in": int(n),
        "rows_written": int(len(out)),
        "judge_ok": n_judge,
        "judge_ok_rate": float(n_judge / n) if n else None,
        "count_ok": n_count,
        "count_ok_rate": float(n_count / n) if n else None,
        "osu_ok": n_osu,
        "osu_ok_rate": float(n_osu / n) if n else None,
        "label_ok": n_label,
        "label_ok_rate": float(n_label / n) if n else None,
        "by_split": by_split.to_dict(orient="records"),
        "osu_dir": str(args.osu_dir),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=1, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
