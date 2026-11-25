"""Build pretrain/eval chart-ID lists for the main experiment.

Protocols
    all             pretrain on every cached .osu (transductive upper bound)
    complement      pretrain on charts outside the 6,828 pairbest charts
    holdout         complement + 80% of pairbest; eval on the star-stratified 20%

The holdout protocol is the honest "encoder never saw the eval chart" one.

    mapset_holdout  Protocol D: hold out whole beatmapsets (songs) instead of
                    single charts.  Beatmapsets are shuffled with a fixed seed
                    and taken in order until the pairbest chart budget
                    int(holdout_frac * n_pairbest) is reached.  Eval is every
                    cached chart of a held-out set, pretrain is the remaining
                    cache.  No chart (or song) is shared between the two pools.

                    Defaults for this protocol: --pairbest
                    data/processed/step1_full_pairbest.parquet, --meta
                    data/processed/beatmap_meta_full.csv, --holdout-frac 0.25.
                    Reference run (seed 0, frac 0.25): 1,716 held-out sets,
                    5,236 pairbest eval charts, 5,462 eval cache charts,
                    16,487 pretrain cache charts.

The other protocols keep their historical defaults (1k pairbest / 1k meta /
frac 0.20), so re-running them stays bit-identical.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .charts import ChartStore
from .utils import repo_root

NL = chr(10)

# Defaults that predate mapset_holdout; kept per protocol so the existing
# all/complement/holdout invocations do not change behaviour.
_PROTOCOL_DEFAULTS = {
    "all": {
        "pairbest": "data/processed/step1_1k_pairbest.parquet",
        "meta": "data/processed/beatmap_meta_1k.csv",
        "holdout_frac": 0.2,
    },
    "complement": {
        "pairbest": "data/processed/step1_1k_pairbest.parquet",
        "meta": "data/processed/beatmap_meta_1k.csv",
        "holdout_frac": 0.2,
    },
    "holdout": {
        "pairbest": "data/processed/step1_1k_pairbest.parquet",
        "meta": "data/processed/beatmap_meta_1k.csv",
        "holdout_frac": 0.2,
    },
    "mapset_holdout": {
        "pairbest": "data/processed/step1_full_pairbest.parquet",
        "meta": "data/processed/beatmap_meta_full.csv",
        "holdout_frac": 0.25,
    },
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--protocol", default="all", choices=list(_PROTOCOL_DEFAULTS))
    ap.add_argument("--osu-dir", default="data/osu_cache")
    ap.add_argument("--pairbest", default=None,
                    help="pairbest parquet; default is protocol-specific "
                         "(1k pairbest, or full pairbest for mapset_holdout)")
    ap.add_argument("--meta", default=None,
                    help="beatmap meta csv with beatmap_id/beatmapset_id/star; "
                         "default is protocol-specific (meta_1k, or meta_full)")
    ap.add_argument("--holdout-frac", type=float, default=None,
                    help="holdout fraction; protocol default 0.20 (0.25 for "
                         "mapset_holdout).  For mapset_holdout this is a chart "
                         "budget: whole sets are added until the cumulative "
                         "pairbest chart count exceeds the budget.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    defaults = _PROTOCOL_DEFAULTS[args.protocol]
    if args.pairbest is None:
        args.pairbest = defaults["pairbest"]
    if args.meta is None:
        args.meta = defaults["meta"]
    if args.holdout_frac is None:
        args.holdout_frac = defaults["holdout_frac"]
    return args


def write_ids(path: Path, ids) -> None:
    ids = sorted(set(int(x) for x in ids))
    path.write_text(NL.join(str(i) for i in ids) + NL)


def _resolve(root: Path, path: str) -> Path:
    """Resolve an input path.

    Relative paths are interpreted from the repo root (historical behaviour),
    with a fallback to the current working directory when the repo-root path
    does not exist (e.g. --meta ../data/processed/... after cd data2vec_main).
    """
    p = Path(path)
    if p.is_absolute():
        return p
    at_root = root / p
    if at_root.exists() or not p.exists():
        return at_root
    return p.resolve()


def _run_mapset_holdout(args, root: Path, all_ids: list, pairbest: pd.DataFrame,
                        out: Path) -> int:
    """Protocol D split: whole beatmapsets are held out.

    Validated recipe (reference numbers for frac=0.25, seed=0: n_sets=1716,
    eval pairbest charts=5236, eval cache=5462, pretrain=16487):

        charts = unique beatmap_id of pairbest
        sub    = meta rows for those charts (drop_duplicates beatmap_id)
        vc     = sub.groupby("beatmapset_id").size()
        sets   = vc.sample(frac=1.0, random_state=seed).index
        order  = vc.loc[sets].cumsum()
        cut    = sets whose cumulative chart count <= int(frac * len(charts))
        eval     = cached charts whose beatmapset_id is in cut
        pretrain = remaining cached charts
    """
    charts = pairbest["beatmap_id"].unique()
    meta = pd.read_csv(_resolve(root, args.meta))
    need = {"beatmap_id", "beatmapset_id", "star"}
    missing = need - set(meta.columns)
    if missing:
        raise SystemExit(str(args.meta) + ": missing columns " + str(sorted(missing)))
    meta = meta.drop_duplicates("beatmap_id")

    sub = meta.loc[meta["beatmap_id"].isin(charts)]
    vc = sub.groupby("beatmapset_id").size()
    sets = vc.sample(frac=1.0, random_state=args.seed).index
    order = vc.loc[sets].cumsum()
    budget = int(args.holdout_frac * len(charts))
    cut = set(order[order <= budget].index)

    # map every cached chart to its beatmapset; charts without meta stay pretrain
    set_of = meta.set_index("beatmap_id")["beatmapset_id"]
    cache_sets = set_of.reindex(all_ids)
    in_eval = cache_sets.isin(cut).to_numpy()
    eval_cache = [i for i, m in zip(all_ids, in_eval) if m]
    pretrain = [i for i, m in zip(all_ids, in_eval) if not m]

    in_cut = sub["beatmapset_id"].isin(cut)
    eval_pair_charts = np.sort(sub.loc[in_cut, "beatmap_id"].to_numpy())
    star_of = meta.set_index("beatmap_id")["star"]
    summary = {
        "protocol": "mapset_holdout",
        "algorithm": (
            "charts = unique beatmap_id of pairbest; sub = meta rows for those "
            "charts (drop_duplicates beatmap_id); vc = sub.groupby("
            "beatmapset_id).size(); sets = vc.sample(frac=1.0, "
            "random_state=seed).index; order = vc.loc[sets].cumsum(); cut = "
            "sets whose cumulative chart count <= int(holdout_frac * "
            "len(charts)); eval_cache = ChartStore ids whose beatmapset_id is "
            "in cut; pretrain = remaining ChartStore ids"
        ),
        "seed": int(args.seed),
        "holdout_frac": float(args.holdout_frac),
        "chart_budget": int(budget),
        "n_cache": int(len(all_ids)),
        "n_pairbest": int(len(charts)),
        "n_pairbest_with_meta": int(len(sub)),
        "n_sets": int(len(cut)),
        "n_sets_total": int(vc.size),
        "n_eval_cache": int(len(eval_cache)),
        "n_pretrain": int(len(pretrain)),
        "n_eval_pairbest": int(len(eval_pair_charts)),
        "n_pretrain_pairbest": int(len(charts) - len(eval_pair_charts)),
        "star_mean_eval_pairbest": round(float(sub.loc[in_cut, "star"].mean()), 6),
        "star_mean_pretrain_pairbest": round(float(sub.loc[~in_cut, "star"].mean()), 6),
        "star_mean_eval_cache": round(float(star_of.reindex(eval_cache).mean()), 6),
        "star_mean_pretrain_cache": round(float(star_of.reindex(pretrain).mean()), 6),
        "pairbest": str(args.pairbest),
        "meta": str(args.meta),
        "osu_dir": str(args.osu_dir),
    }
    write_ids(out / "pretrain_ids.txt", pretrain)
    write_ids(out / "eval_ids.txt", eval_cache)
    # held-out beatmapsets + chart-level view of the eval pool (5,236 of the
    # 5,462 cached eval charts have pairbest rows).  eval_mapsets.txt is the
    # authoritative group list: 17 held-out sets have no cached chart, so they
    # cannot be recovered from the cache-level eval_ids.txt alone.
    write_ids(out / "eval_mapsets.txt", sorted(int(s) for s in cut))
    write_ids(out / "eval_pairbest_ids.txt", eval_pair_charts)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + NL)
    print(json.dumps(summary, indent=1))
    ref = (1716, 5236, 5462, 16487)
    got = (len(cut), len(eval_pair_charts), len(eval_cache), len(pretrain))
    if got != ref:
        print("[warn] reference numbers for frac=0.25 seed=0 are "
              "1716 / 5236 / 5462 / 16487 (n_sets / eval pairbest / eval cache / "
              "pretrain), got " + str(got), flush=True)
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    root = repo_root()
    osu_dir = Path(args.osu_dir)
    if not osu_dir.is_absolute():
        osu_dir = root / osu_dir
    all_ids = ChartStore(osu_dir).ids
    pairbest = pd.read_parquet(_resolve(root, args.pairbest))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.protocol == "mapset_holdout":
        return _run_mapset_holdout(args, root, all_ids, pairbest, out)

    eval_ids = sorted(pairbest["beatmap_id"].unique().tolist())
    rng = np.random.default_rng(args.seed)

    if args.protocol == "all":
        pretrain, holdout = all_ids, eval_ids
    elif args.protocol == "complement":
        pretrain = [i for i in all_ids if i not in set(eval_ids)]
        holdout = eval_ids
    else:
        meta = pd.read_csv(_resolve(root, args.meta))
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
