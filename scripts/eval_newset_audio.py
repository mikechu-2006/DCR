#!/usr/bin/env python
"""Experiment 2a: audio-on vs audio-off on new-beatmapset test pairs.

Reads the existing 1k P4 audio/legacy signal tables (no retraining) and
recomputes the paired Delta-NLL on test comparisons where both charts belong to
beatmapsets that never appear in the 6x6 supervised training charts.

Subsets
-------
all          every test comparison
new_set      both charts' set unseen in train
mixed        exactly one chart's set unseen
both_seen    both charts' sets seen
new_cross    both unseen, different sets
new_same     both unseen, same set
seen_any     at least one chart's set seen (mixed + both_seen)

For each subset we report n_instances, coverage, audio/legacy NLL, paired
Delta-NLL with a player-cluster bootstrap, and the full-test Pearson/Spearman
from the existing rank_eval.json.

Usage
-----
python scripts/eval_newset_audio.py \
  --split-prefix data/processed/step1_1k_pairbest_p4_nogap \
  --audio-dir results/p4_multiseed_audio/audio/p4 \
  --legacy-dir results/p4_multiseed_audio/legacy/p4 \
  --meta data/processed/beatmap_meta_1k.csv \
  --seeds 1,2,3,4,5 --bootstrap 400 \
  --out-dir results/audio_newset
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dcr import rank_metrics as rm  # noqa: E402

METHODS = [
    ("cm3p_linear", "CM3P linear"),
    ("cm3p_linear_extra", "CM3P + external linear"),
    ("cm3p_mlp_h64_L2", "CM3P MLP h64 L2"),
    ("cm3p_mlp_extra", "CM3P + external MLP"),
]
FIXED = [("sunny_star", "Sunny"), ("daniel_star", "Daniel"), ("star", "star")]


def _load_signals(path: Path, maps: np.ndarray, cols: list[str]) -> dict[str, np.ndarray]:
    df = pd.read_parquet(path).set_index("beatmap_id")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing {missing}")
    out = {}
    for c in cols:
        d = df[c].reindex(maps).to_numpy(np.float64)
        if not np.isfinite(d).all():
            raise SystemExit(f"{path}: {c} has non-finite values on test maps")
        out[c] = d
    return out


def _encode(df: pd.DataFrame, map_index: dict):
    item = df["beatmap_id"].map(map_index).to_numpy(np.int32)
    perf = (-df["loss"].astype(np.float64)).to_numpy()
    players = np.sort(df["player_id"].unique())
    p_idx = {int(p): i for i, p in enumerate(players)}
    pi = df["player_id"].map(p_idx).to_numpy(np.int32)
    tmp = pd.DataFrame({"pi": pi, "base": df["player_id_base"].to_numpy()})
    idx_map = tmp.groupby("pi").indices
    groups = [np.asarray(idx_map[i]) for i in range(len(players))]
    clusters = (tmp.groupby("pi")["base"].first()
                .reindex(range(len(players))).to_numpy())
    cidx, cluster_ids = pd.factorize(clusters.astype(np.int64), sort=True)
    return item, perf, groups, cidx, cluster_ids


def _subset_table(tab: rm.PairTable, mask: np.ndarray) -> rm.PairTable:
    keep = mask[tab.ua, tab.ub]
    ua = tab.ua[keep]
    ub = tab.ub[keep]
    nab = tab.n_ab[keep]
    nba = tab.n_ba[keep]
    n = int(round(float(nab.sum() + nba.sum())))
    return rm.PairTable(ua, ub, nab, nba, tab.n_items, n, 0)


def _nll_by_cluster_subset(groups, item, perf, g, cidx, n_clusters,
                           mask: np.ndarray):
    nll = np.zeros(n_clusters, dtype=np.float64)
    cnt = np.zeros(n_clusters, dtype=np.float64)
    for p, idx in enumerate(groups):
        l = int(len(idx))
        if l < 2:
            continue
        ii, jj = np.triu_indices(l, 1)
        a = item[idx][ii]
        b = item[idx][jj]
        swap = a > b
        ua = np.where(swap, b, a)
        ub = np.where(swap, a, b)
        keep = mask[ua, ub]
        if not keep.any():
            continue
        pa = perf[idx][ii][keep]
        pb = perf[idx][jj][keep]
        y = (pa > pb) ^ swap[keep]
        tie = pa == pb
        w = np.where(tie, 0.5, y.astype(np.float64))
        s = expit(g[ub[keep]] - g[ua[keep]])
        ll = w * np.log(np.clip(s, 1e-15, 1.0)) \
            + (1.0 - w) * np.log(np.clip(1.0 - s, 1e-15, 1.0))
        nll[cidx[p]] += -float(ll.sum())
        cnt[cidx[p]] += float(ll.size)
    return nll, cnt


def _eval_subset(tab: rm.PairTable, groups, item, perf, cidx, n_clusters,
                 mask: np.ndarray, d_audio: np.ndarray, d_legacy: np.ndarray,
                 bootstrap: int, rng: np.random.Generator):
    sub = _subset_table(tab, mask)
    out = {"n_pairs": int(sub.n_pairs), "n_instances": int(sub.n_instances)}
    if sub.n_pairs == 0 or sub.n_instances == 0:
        out.update({"nll_audio": None, "nll_legacy": None, "delta": None,
                    "ci": None, "p_delta_lt_0": None, "n_charts": 0,
                    "n_clusters": 0})
        return out
    charts = np.unique(np.r_[sub.ua, sub.ub])
    out["n_charts"] = int(charts.size)
    arms = {}
    for arm, d in (("audio", d_audio), ("legacy", d_legacy)):
        link = rm.fit_link(sub, d, kind="linear")
        g = link.transform(d)
        nll, cnt = _nll_by_cluster_subset(groups, item, perf, g, cidx,
                                          n_clusters, mask)
        arms[arm] = {"nll": float(link.nll / sub.n_instances),
                     "beta": float(link.beta),
                     "nll_by_cluster": nll, "n_by_cluster": cnt,
                     "rate": float(nll.sum() / cnt.sum())}
    out["nll_audio"] = arms["audio"]["rate"]
    out["nll_legacy"] = arms["legacy"]["rate"]
    out["delta"] = float(out["nll_audio"] - out["nll_legacy"])
    out["n_clusters"] = int((arms["audio"]["n_by_cluster"] > 0).sum())
    deltas = []
    for _ in range(int(bootstrap)):
        draw = rng.integers(0, n_clusters, n_clusters)
        mult = np.bincount(draw, minlength=n_clusters).astype(np.float64)
        na = float((arms["audio"]["nll_by_cluster"] * mult).sum())
        nb = float((arms["legacy"]["nll_by_cluster"] * mult).sum())
        ca = float((arms["audio"]["n_by_cluster"] * mult).sum())
        cb = float((arms["legacy"]["n_by_cluster"] * mult).sum())
        if ca <= 0 or cb <= 0:
            continue
        deltas.append(na / ca - nb / cb)
    d = np.asarray(deltas)
    out["ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] \
        if d.size else None
    out["p_delta_lt_0"] = float((d < 0).mean()) if d.size else None
    return out


def _load_corr(path: Path, keys: list[str]) -> dict:
    """Use the official rank_eval.json correlations (same pair set/ceiling)."""
    j = json.loads(path.read_text())
    out = {}
    for k in keys:
        s = j["signals"].get(k)
        if s is None:
            out[k] = {"pearson_bhat": None, "rho_bhat": None, "beta": None}
        else:
            out[k] = {"pearson_bhat": s.get("pearson_bhat"),
                      "rho_bhat": s.get("rho_bhat"),
                      "beta": s.get("link", {}).get("beta")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-prefix",
                    default="data/processed/step1_1k_pairbest_p4_nogap")
    ap.add_argument("--audio-dir", default="results/p4_multiseed_audio/audio/p4")
    ap.add_argument("--legacy-dir", default="results/p4_multiseed_audio/legacy/p4")
    ap.add_argument("--meta", default="data/processed/beatmap_meta_1k.csv")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/audio_newset")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.meta, usecols=["beatmap_id", "beatmapset_id"])
    setmap = meta.set_index("beatmap_id")["beatmapset_id"]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    all_rows = []
    for seed in seeds:
        split = Path(f"{args.split_prefix}_s{seed}.parquet")
        df = pd.read_parquet(split)
        train = df[df["split"] == "train"]
        test = df[df["split"] == "test"].reset_index(drop=True)
        train_charts = np.sort(train["beatmap_id"].unique())
        test_charts = np.sort(test["beatmap_id"].unique())
        maps = test_charts
        map_idx = {int(m): i for i, m in enumerate(maps)}
        M = int(maps.size)
        train_sets = set(int(x) for x in
                         setmap.reindex(train_charts).dropna().unique())
        sets = setmap.reindex(maps).to_numpy()
        if pd.isna(sets).any():
            raise SystemExit(f"{split}: test charts missing beatmapset_id")
        sets = sets.astype(np.int64)
        new = ~np.isin(sets, list(train_sets))

        item, perf, groups, cidx, cluster_ids = _encode(test, map_idx)
        tab = rm.build_pair_table(groups, item, perf, M)
        n_clusters = int(cluster_ids.size)

        a_sig = _load_signals(Path(args.audio_dir) / f"s{seed}" / "signals.parquet",
                              maps, [m for m, _ in METHODS] + [f for f, _ in FIXED])
        l_sig = _load_signals(Path(args.legacy_dir) / f"s{seed}" / "signals.parquet",
                              maps, [m for m, _ in METHODS] + [f for f, _ in FIXED])

        masks = {}
        masks["all"] = np.ones((M, M), dtype=bool)
        masks["new_set"] = np.outer(new, new)
        masks["mixed"] = np.outer(new, ~new) | np.outer(~new, new)
        masks["both_seen"] = np.outer(~new, ~new)
        masks["new_cross"] = masks["new_set"] & (sets[:, None] != sets[None, :])
        masks["new_same"] = masks["new_set"] & (sets[:, None] == sets[None, :])
        masks["seen_any"] = masks["mixed"] | masks["both_seen"]

        rng = np.random.default_rng(args.seed + seed)
        result = {"seed": seed, "split": str(split),
                  "n_train_charts": int(train_charts.size),
                  "n_test_charts": int(test_charts.size),
                  "n_train_sets": int(len(train_sets)),
                  "n_test_new_set_charts": int(new.sum()),
                  "n_test_new_set_frac": float(new.mean()),
                  "test_rows": int(len(test)),
                  "test_pairs": int(tab.n_pairs),
                  "test_instances": int(tab.n_instances),
                  "n_clusters": n_clusters,
                  "subsets": {}, "methods": {}}
        for name, mask in masks.items():
            sub = _subset_table(tab, mask)
            result["subsets"][name] = {
                "n_pairs": int(sub.n_pairs),
                "n_instances": int(sub.n_instances),
                "coverage": float(sub.n_instances / tab.n_instances)
                if tab.n_instances else None,
                "n_charts": int(np.unique(np.r_[sub.ua, sub.ub]).size)
                if sub.n_pairs else 0,
                "n_clusters": int((_nll_by_cluster_subset(
                    groups, item, perf, np.zeros(M), cidx, n_clusters, mask)[1] > 0).sum())
                if sub.n_pairs else 0,
            }
        # full-test correlations from the official rank_eval.json
        keys = [k for k, _ in METHODS] + [k for k, _ in FIXED]
        corr_audio = _load_corr(
            Path(args.audio_dir) / f"s{seed}" / "rank_eval.json", keys)
        corr_legacy = _load_corr(
            Path(args.legacy_dir) / f"s{seed}" / "rank_eval.json", keys)

        for key, label in METHODS + FIXED:
            result["methods"][key] = {"label": label, "subsets": {}}
            for name, mask in masks.items():
                r = _eval_subset(tab, groups, item, perf, cidx, n_clusters,
                                 mask, a_sig[key], l_sig[key], args.bootstrap, rng)
                r["corr_audio"] = corr_audio[key]
                r["corr_legacy"] = corr_legacy[key]
                result["methods"][key]["subsets"][name] = r
                if r["n_instances"]:
                    all_rows.append(dict(
                        seed=seed, method=key, label=label, subset=name,
                        n_instances=r["n_instances"], coverage=r["n_instances"] / tab.n_instances,
                        nll_audio=r["nll_audio"], nll_legacy=r["nll_legacy"],
                        delta=r["delta"], ci_lo=r["ci"][0], ci_hi=r["ci"][1],
                        p_delta_lt_0=r["p_delta_lt_0"],
                        pearson_audio=r["corr_audio"]["pearson_bhat"],
                        spearman_audio=r["corr_audio"]["rho_bhat"],
                        pearson_legacy=r["corr_legacy"]["pearson_bhat"],
                        spearman_legacy=r["corr_legacy"]["rho_bhat"]))
        out_dir.joinpath(f"subset_s{seed}.json").write_text(
            json.dumps(result, indent=1, ensure_ascii=False) + "\n")
        print(f"[s{seed}] test={len(test):,} charts={M:,} pairs={tab.n_pairs:,} "
              f"new_set_charts={result['n_test_new_set_charts']:,} "
              f"new_set_cmp={result['subsets']['new_set']['n_instances']:,} "
              f"({result['subsets']['new_set']['coverage']:.3f})", flush=True)

    rows = pd.DataFrame(all_rows)
    rows.to_csv(out_dir / "subset_summary.csv", index=False)
    # 5-seed mean per method/subset
    means = (rows.groupby(["method", "label", "subset"], sort=False)
             .agg(n_seeds=("seed", "nunique"),
                  coverage_mean=("coverage", "mean"),
                  delta_mean=("delta", "mean"),
                  delta_std=("delta", "std"),
                  seeds_negative=("delta", lambda x: int((x < 0).sum())),
                  ci_negative=("ci_hi", lambda x: int((x < 0).sum())),
                  nll_audio_mean=("nll_audio", "mean"),
                  nll_legacy_mean=("nll_legacy", "mean"))
             .reset_index())
    means.to_csv(out_dir / "subset_means.csv", index=False)
    print(means[means.subset.isin(["all", "new_set", "new_cross"])]
          .round(5).to_string(index=False), flush=True)
    print(f"[done] {time.time() - t0:.1f}s -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
