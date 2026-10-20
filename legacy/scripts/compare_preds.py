#!/usr/bin/env python
"""Paired comparison of two step2 prediction files -- "is this really better?".

The repository's own methodology note (docs/step2_capacity_dim_sweep.md section 7.4) is why this
exists: differences of 0.2-0.5% between step2 models are routinely smaller than the uncertainty
introduced by resampling, and deff ~ 5.4 means the nominal row count is not the sample size.
So every comparison here is PAIRED on identical test rows, and the interval is a CLUSTER
bootstrap -- by default resampling (player_id, beatmap_id) pairs, because plays are not
independent repeats (adjacent plays within a pair correlate at 0.93, section 6.3).

Usage:

    # baseline (linear, frozen CM3P) and the new model, both dumping their test predictions
    python scripts/step2_predict.py --tag 1k --models mf_dot --epochs 60 \
        --chart-content cm3p --content-dim 32 --content-reduce pca --save-preds
    python scripts/step2_cm3p_mlp.py --tag 1k --dim 16 --epochs 60 --save-preds
    python scripts/compare_preds.py data/processed/step2_1k_preds_loss.npz \
        data/processed/step2_1k_preds_loss.npz --keys pred_mf_dot pred_mlp

Row alignment is ASSERTED, never assumed: if the two files do not carry the same ids in the same
order, the script intersects them by pair key and re-sorts, so the comparison is still exact --
a silent misalignment would otherwise produce a perfectly plausible wrong number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def pair_key(player, beatmap):
    """Ordered-pair key; the +1 offset on the chart max keeps the encoding injective."""
    k = int(beatmap.max()) + 1
    return player.astype(np.int64) * np.int64(k) + beatmap.astype(np.int64)


def load_npz(path):
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def label(npz, key):
    """npz 里的 variant 是 numpy 0-d 字符串数组：str() 会打印 dtype，必须 .item()。"""
    v = npz.get(key)
    if v is None:
        return ""
    try:
        return str(v.item())
    except (AttributeError, ValueError):
        return str(v)


def rmse(p, y):
    return float(np.sqrt(((p - y) ** 2).mean()))


def cluster_sums(values, inv, n_clusters):
    return np.bincount(inv, weights=values, minlength=n_clusters)


def bootstrap_delta(pred_a, pred_b, y, cluster_inv, n_clusters, n_boot, seed):
    """Paired cluster bootstrap of RMSE(a) - RMSE(b), resampling whole clusters.

    Vectorised: summing per-cluster squared errors lets each replicate be a single bincount over
    the resampled cluster ids, instead of concatenating row index arrays (which dominated the
    runtime at ~270k rows x 2000 draws).
    """
    rng = np.random.default_rng(seed)
    ea = (pred_a - y) ** 2
    eb = (pred_b - y) ** 2
    sa = np.bincount(cluster_inv, weights=ea, minlength=n_clusters)
    sb = np.bincount(cluster_inv, weights=eb, minlength=n_clusters)
    cnt = np.bincount(cluster_inv, minlength=n_clusters).astype(np.float64)
    # Chunked: a full (n_boot, n_clusters) integer draw matrix is ~1.5 GB here (2000 x 183k),
    # and indexing sa/sb with it doubles that.  256 replicates at a time keeps it small.
    out = np.empty(n_boot)
    step = max(1, int(2e7 // max(n_clusters, 1)))
    for s in range(0, n_boot, step):
        e = min(s + step, n_boot)
        draws = rng.integers(0, n_clusters, size=(e - s, n_clusters))
        nt = cnt[draws].sum(1)
        out[s:e] = np.sqrt(sa[draws].sum(1) / nt) - np.sqrt(sb[draws].sum(1) / nt)
    return out


def paired_t(delta_sq, cluster_inv, n_clusters, cluster_unit):
    """t statistic of the paired squared-error difference, clustered when a cluster unit is used."""
    if cluster_unit == "row":
        se = delta_sq.std(ddof=1) / np.sqrt(delta_sq.size)
    else:
        cnt = np.bincount(cluster_inv, minlength=n_clusters).astype(np.float64)
        per = cluster_sums(delta_sq, cluster_inv, n_clusters) / cnt
        se = per.std(ddof=1) / np.sqrt(n_clusters)
    return float(delta_sq.mean() / se) if se > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a", type=Path, help="npz with prediction A (usually the new model)")
    ap.add_argument("file_b", type=Path, help="npz with prediction B (usually the baseline)")
    ap.add_argument("--keys", nargs=2, default=["pred_mlp", "pred_mf_dot"], metavar=("KA", "KB"))
    ap.add_argument("--cluster", default="pair", choices=["pair", "player", "chart", "row"],
                    help="bootstrap unit; 'row' = naive iid, reported only for contrast")
    ap.add_argument("--n-boot", type=int, default=400,
                    help="bootstrap replicates; 400 is plenty for a 95%% interval and the runtime "
                         "grows with n_boot x n_clusters (183k pairs at 1k)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="optional json output path")
    args = ap.parse_args()
    ka, kb = args.keys

    A, B = load_npz(args.file_a), load_npz(args.file_b)
    for d, name, key in ((A, args.file_a.name, ka), (B, args.file_b.name, kb)):
        if key not in d:
            raise SystemExit(f"{name}: no '{key}'; prediction keys present: "
                             f"{sorted(k for k in d if k.startswith('pred_'))}")

    pa, pb = A[ka].astype(np.float64), B[kb].astype(np.float64)
    ya, yb = A["y"].astype(np.float64), B["y"].astype(np.float64)

    # ---------------------------------------------------------------- alignment
    if "player_id" in A and "player_id" in B:
        ka_ids, kb_ids = pair_key(A["player_id"], A["beatmap_id"]), pair_key(B["player_id"], B["beatmap_id"])
        if ka_ids.shape == kb_ids.shape and np.array_equal(ka_ids, kb_ids):
            print(f"[align] {len(ya):,} test rows, identical ids in identical order")
            ids, y = ka_ids, ya
        else:
            common = np.intersect1d(ka_ids, kb_ids)
            if common.size == 0:
                raise SystemExit("the two files share no (player, chart) pair -- not comparable")
            print(f"[align] files differ ({ka_ids.size:,} vs {kb_ids.size:,} rows); comparing on the "
                  f"{common.size:,} pairs they share, sorted by pair key")
            ia = np.flatnonzero(np.isin(ka_ids, common))
            ib = np.flatnonzero(np.isin(kb_ids, common))
            ia, ib = ia[np.argsort(ka_ids[ia], kind="stable")], ib[np.argsort(kb_ids[ib], kind="stable")]
            if not np.array_equal(ka_ids[ia], kb_ids[ib]):
                raise SystemExit("pair sets do not match after intersection -- refusing to compare")
            ids, y = ka_ids[ia], ya[ia]
            pa, pb = pa[ia], pb[ib]
            if not np.allclose(y, yb[ib]):
                raise SystemExit("the shared pairs carry different labels in the two files")
            A["player_id"], A["beatmap_id"] = A["player_id"][ia], A["beatmap_id"][ia]
    else:
        raise SystemExit("at least one file has no player_id/beatmap_id: alignment cannot be "
                         "verified, and an unverified pairing is not a result")

    if args.cluster == "player":
        cl = A["player_id"]
    elif args.cluster == "chart":
        cl = A["beatmap_id"]
    elif args.cluster == "pair":
        cl = ids
    else:
        cl = np.arange(len(y))
    _, cluster_inv = np.unique(cl, return_inverse=True)
    n_clusters = int(cluster_inv.max()) + 1

    ra, rb = rmse(pa, y), rmse(pb, y)
    d = ra - rb
    boot = bootstrap_delta(pa, pb, y, cluster_inv, n_clusters, args.n_boot, args.seed)
    lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    delta_sq = (pa - y) ** 2 - (pb - y) ** 2
    t = paired_t(delta_sq, cluster_inv, n_clusters, args.cluster)

    verdict = ("A better (95% CI excludes 0)" if hi < 0 else
               "B better (95% CI excludes 0)" if lo > 0 else
               "indistinguishable at 95% (CI straddles 0)")
    res = {
        "rows": int(len(y)), "clusters": n_clusters, "cluster_unit": args.cluster,
        "n_boot": args.n_boot,
        "a": {"file": args.file_a.name, "key": ka, "rmse": round(ra, 6),
              "variant": label(A, "variant")},
        "b": {"file": args.file_b.name, "key": kb, "rmse": round(rb, 6),
              "variant": label(B, "variant")},
        "delta_rmse": round(d, 6), "delta_pct_of_b": round(100.0 * d / rb, 4),
        "ci95": [round(lo, 6), round(hi, 6)],
        "t_stat": round(t, 3), "verdict": verdict,
        "caveat": "deff ~ 5.4 (docs/step2_capacity_dim_sweep.md 6.3): the row-level CI is too "
                  "narrow, so --cluster row is reported only for contrast",
    }
    print(f"\nA = {res['a']['variant'] or ka}   ({args.file_a.name} :: {ka})")
    print(f"B = {res['b']['variant'] or kb}   ({args.file_b.name} :: {kb})")
    print(f"paired on {len(y):,} rows in {n_clusters:,} {args.cluster}-clusters, "
          f"{args.n_boot} bootstrap draws")
    print(f"RMSE   A = {ra:.6f}    B = {rb:.6f}    delta = {d:+.6f} ({res['delta_pct_of_b']:+.3f}%)")
    print(f"95% CI = [{lo:+.6f}, {hi:+.6f}]   t = {t:+.3f}")
    print(f"-> {verdict}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
