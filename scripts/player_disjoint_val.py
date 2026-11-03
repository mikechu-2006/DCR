#!/usr/bin/env python
"""Player-disjoint validation table for CM3P rank signals.

Splits the *train* player clusters 80/20, fits the cross-fitted BT ceiling on
the 80% fit players, and evaluates fixed per-chart signals and the validation
NLL recorded by step2_cm3p_rank.py on the held-out 20% players.  This is the
cleanest "new player" comparison: neither the ceiling nor the model's selection
uses the held-out players' rows.

Usage:
  python scripts/player_disjoint_val.py \
      --signal-table results/cm3p_rank_1k/signals_p1.parquet \
      --signals msd_overall,sunny_star,daniel_star,star,notes_per_sec \
      --models cm3p_linear,cm3p_mlp_h64_L2,cm3p_linear_extra_l3,cm3p_mlp_extra_val \
      --val-frac 0.2 --val-seed 0 --folds 2 \
      --out results/cm3p_rank_1k/player_disjoint_val.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcr import rank_metrics as rm  # noqa: E402

LOG2 = rm.LOG2


def encode(df: pd.DataFrame, map_index: dict):
    item = df["beatmap_id"].map(map_index).to_numpy(np.int32)
    perf = (-df["loss"].astype(np.float64)).to_numpy()
    players = np.sort(df["player_id"].unique())
    pidx = {int(p): i for i, p in enumerate(players)}
    pi = df["player_id"].map(pidx).to_numpy(np.int32)
    tmp = pd.DataFrame({"pi": pi, "base": df["player_id_base"].to_numpy()})
    idx = tmp.groupby("pi").indices
    groups = [np.asarray(idx[i]) for i in range(len(players))]
    cl = (tmp.groupby("pi")["base"].first()
          .reindex(range(len(players))).to_numpy())
    return item, perf, groups, np.asarray(cl, dtype=np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default=str(ROOT / "data/processed/step1_1k_pairbest.parquet"))
    ap.add_argument("--signal-table", required=True)
    ap.add_argument("--signals", default="msd_overall,sunny_star,daniel_star,star,notes_per_sec")
    ap.add_argument("--models", default="cm3p_linear,cm3p_linear_extra,"
                                        "cm3p_mlp_h64_L1,cm3p_mlp_h64_L2,"
                                        "cm3p_mlp_h128_L1,cm3p_linear_extra_l3,"
                                        "cm3p_mlp_extra_val,ext_blend")
    ap.add_argument("--model-glob", default="results/cm3p_rank_1k/*.json")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--val-seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()

    step1 = pd.read_parquet(args.inp)
    if "player_id_base" not in step1.columns:
        step1["player_id_base"] = step1["player_id"]
    train = step1[step1["split"] == "train"].reset_index(drop=True)
    maps = np.sort(step1["beatmap_id"].unique())
    midx = {int(m): i for i, m in enumerate(maps)}
    item, perf, groups, clusters = encode(train, midx)
    M = len(maps)

    ids = np.unique(clusters)
    rng = np.random.default_rng(args.val_seed)
    n_val = max(1, int(round(args.val_frac * ids.size)))
    val_ids = set(rng.choice(ids, n_val, replace=False).tolist())
    fit_players = np.array([p for p, c in enumerate(clusters)
                            if int(c) not in val_ids])
    val_players = np.array([p for p, c in enumerate(clusters)
                            if int(c) in val_ids])
    fit_groups = [groups[p] for p in fit_players]
    val_groups = [groups[p] for p in val_players]
    fit_cl = clusters[fit_players]
    val_cl = clusters[val_players]
    tab_fit = rm.build_pair_table(fit_groups, item, perf, M)
    tab_val = rm.build_pair_table(val_groups, item, perf, M)
    print(f"[val] fit players {len(fit_players)} / val {len(val_players)}; "
          f"fit pairs {tab_fit.n_pairs:,}; val pairs {tab_val.n_pairs:,} "
          f"({tab_val.n_instances:,} comparisons)", flush=True)

    ceil = rm.cross_fit_bt(val_groups, item, perf, M, val_cl,
                           folds=args.folds, seed=args.val_seed,
                           fit_groups=fit_groups, fit_item=item,
                           fit_perf=perf, fit_clusters=fit_cl)
    print(f"[val ceiling] {ceil.nll:.5f} coverage={ceil.coverage:.3f} "
          f"n={ceil.n:,}", flush=True)

    sig = pd.read_parquet(args.signal_table).set_index("beatmap_id")
    rows = []
    for name in [s.strip() for s in args.signals.split(",") if s.strip()]:
        if name not in sig.columns:
            print(f"warn: {name} not in signal table")
            continue
        d = sig[name].reindex(maps).to_numpy(np.float64)
        if not np.isfinite(d).all():
            print(f"warn: {name} has missing charts")
            continue
        link = rm.fit_link(tab_val, d, kind="linear")
        nll = float(link.nll / tab_val.n_instances)
        rows.append({"signal": name, "kind": "fixed", "val_nll": nll,
                     "captured_val": (LOG2 - nll) / (LOG2 - ceil.nll),
                     "beta": float(link.beta)})

    for name in [s.strip() for s in args.models.split(",") if s.strip()]:
        p = Path(args.model_glob).parent / f"{name}.json"
        if not p.exists():
            print(f"warn: {p} not found")
            continue
        js = json.loads(p.read_text())
        v = js.get("val_nll_per_instance")
        if v is None:
            continue
        rows.append({"signal": name, "kind": "model", "val_nll": float(v),
                     "captured_val": (LOG2 - float(v)) / (LOG2 - ceil.nll),
                     "beta": None})

    out = pd.DataFrame(rows).sort_values("val_nll")
    print(out.to_string(index=False))
    meta = {"n_fit_players": int(len(fit_players)),
            "n_val_players": int(len(val_players)),
            "val_pairs": int(tab_val.n_pairs),
            "val_comparisons": int(tab_val.n_instances),
            "ceiling_nll": float(ceil.nll),
            "ceiling_coverage": float(ceil.coverage),
            "ceiling_n": int(ceil.n),
            "val_frac": float(args.val_frac), "val_seed": int(args.val_seed),
            "folds": int(args.folds)}
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"meta": meta, "signals": rows},
                               indent=2, ensure_ascii=False))
    outp.with_suffix(".csv").write_text(out.to_csv(index=False))
    print(f"wrote {outp} ({time.time() - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
