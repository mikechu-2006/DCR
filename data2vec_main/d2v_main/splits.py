"""Build pretrain/eval chart-ID lists for the main experiment.

Protocols
    all         pretrain on every cached .osu (transductive upper bound)
    complement  pretrain on charts outside the 6,828 pairbest charts
    holdout     complement + 80% of pairbest; eval on the star-stratified 20%

The holdout protocol is the honest "encoder never saw the eval chart" one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .charts import ChartStore
from .utils import repo_root


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--protocol", default="all", choices=["all", "complement", "holdout"])
    ap.add_argument("--osu-dir", default="data/osu_cache")
    ap.add_argument("--pairbest", default="data/processed/step1_1k_pairbest.parquet")
    ap.add_argument("--meta", default="data/processed/beatmap_meta_1k.csv")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def write_ids(path: Path, ids) -> None:
    path.write_text("\n".join(str(int(i)) for i in sorted(set(int(x) for x in ids))) + "\n")


def main(argv=None) -> int:
    args = parse_args(argv)
    root = repo_root()
    osu_dir = Path(args.osu_dir)
    if not osu_dir.is_absolute():
        osu_dir = root / osu_dir
    all_ids = ChartStore(osu_dir).ids
    pairbest = pd.read_parquet(root / args.pairbest)
    eval_ids = sorted(pairbest["beatmap_id"].unique().tolist())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.protocol == "all":
        pretrain, holdout = all_ids, eval_ids
    elif args.protocol == "complement":
        pretrain = [i for i in all_ids if i not in set(eval_ids)]
        holdout = eval_ids
    else:
        meta = pd.read_csv(root / args.meta)
        star = meta.set_index("beatmap_id")["star"].reindex(eval_ids)
        star = star.fillna(star.median())
        bins = pd.qcut(star, 5, labels=False, duplicates="drop")
        holdout = []
        for b in sorted(pd.unique(bins)):
            ids_b = [int(i) for i, bb in zip(eval_ids, bins) if bb == b]
            k = max(1, int(round(args.holdout_frac * len(ids_b))))
            holdout.extend(rng.choice(ids_b, size=min(k, len(ids_b)), replace=False).tolist())
        holdout_set = set(int(i) for i in holdout)
        pretrain = [i for i in all_ids if i not in holdout_set]
        holdout = sorted(holdout_set)
        star_p = star.loc[[i for i in pretrain if i in star.index]]
        star_h = star.loc[holdout]
        print(f"[holdout] star mean pretrain={star_p.mean():.3f} holdout={star_h.mean():.3f}")

    write_ids(out / "pretrain_ids.txt", pretrain)
    write_ids(out / "eval_ids.txt", holdout)
    summary = {
        "protocol": args.protocol,
        "n_all": len(all_ids),
        "n_pairbest": len(eval_ids),
        "n_pretrain": len(set(pretrain)),
        "n_eval": len(set(holdout)),
        "seed": args.seed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
