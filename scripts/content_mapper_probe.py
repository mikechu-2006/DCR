#!/usr/bin/env python
"""content_mapper_probe -- style and attribute probes on a frozen content table.

Two cheap, label-free diagnostics for the D2V attribute-channel programme.
Both use the SAME beatmapset-disjoint 70/30 split (numpy default_rng(seed=0)
over unique beatmapset_id values) so they are comparable and leakage-safe.

C3  mapper-ID probe
    Can a linear classifier recover the mapper (author style) from the content
    vector alone?  Top-K mappers by train chart count are kept; multinomial
    logistic regression (sklearn) is fitted on standardised content and scored
    on the set-disjoint test charts.  Reported: top-1 accuracy, majority
    baseline, macro-F1, per-class chance 1/K.  A K-dependence is expected: the
    vector is 512-d, so ~30 charts/mapper is enough to memorise.

C4  attribute decodability
    How much of each structured attribute is linearly present in the 512-d
    content?  Ridge regression content -> target, train-fit/test-R2, with the
    ridge alpha selected by GCV on the train charts.  This is an upper bound on
    what a downstream probe can read off; it does NOT imply the attribute is
    causally used.

Inputs (defaults):
    data/processed/chart_content_d2v_holdout.parquet   21,949 x 512 contract table
    data/processed/beatmap_meta_full.csv               beatmap_id, beatmapset_id, mapper_id
    data/processed/d2v_attr_audit_per_chart.parquet    per-chart attribute stats

Outputs (defaults, in --out-dir):
    mapper_probe_d2v8.json
    attr_decodability_d2v8.json

Usage:
    python scripts/content_mapper_probe.py
    python scripts/content_mapper_probe.py --content-path data/processed/chart_content_d2v_holdout.parquet --mode both
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = [
    "log1p_n_sv_changes",
    "kiai_frac",
    "log2_bpm_p50",
    "hold_frac_gt1s",
    "vol_resolved_p50",
    "hs_any_ext_frac",
]


# --------------------------------------------------------------------------- io
def load_content(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    df = pd.read_parquet(path)
    if "embedding" not in df.columns:
        raise SystemExit(f"{path}: no embedding column")
    ids = df["beatmap_id"].to_numpy(np.int64)
    dims = {np.asarray(e).shape for e in df["embedding"]}
    if len(dims) != 1:
        raise SystemExit(f"{path}: ragged embeddings {sorted(dims)[:3]}")
    X = np.stack(df["embedding"].to_numpy()).astype(np.float64)
    if not np.isfinite(X).all():
        raise SystemExit(f"{path}: non-finite embedding values")
    return ids, X, int(X.shape[1])


def load_meta(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path, usecols=["beatmap_id", "beatmapset_id", "mapper_id"])
    m = m.drop_duplicates("beatmap_id").set_index("beatmap_id")
    return m


def split_sets(content_ids: np.ndarray, set_ids: np.ndarray, meta: pd.DataFrame,
               test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Beatmapset-disjoint 70/30 split over the content charts."""
    sets = np.unique(set_ids)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(sets.size)
    n_test = int(round(test_frac * sets.size))
    test_mask_sets = np.zeros(sets.size, dtype=bool)
    test_mask_sets[perm[:n_test]] = True
    test_set_values = set(sets[test_mask_sets].tolist())
    is_test = np.array([int(s) in test_set_values for s in set_ids], dtype=bool)
    is_train = ~is_test
    # charts of the same set must never straddle the split
    tr_sets = set(set_ids[is_train].tolist())
    te_sets = set(set_ids[is_test].tolist())
    if tr_sets & te_sets:
        raise SystemExit(f"split leakage: {len(tr_sets & te_sets)} shared sets")
    info = {
        "n_charts": int(len(content_ids)),
        "n_beatmapsets": int(sets.size),
        "n_train_charts": int(is_train.sum()),
        "n_test_charts": int(is_test.sum()),
        "n_train_sets": int(len(tr_sets)),
        "n_test_sets": int(len(te_sets)),
        "train_chart_frac": float(is_train.mean()),
        "test_frac_requested": float(test_frac),
        "seed": int(seed),
        "split_unit": "beatmapset_id (set-disjoint)",
        "rng": "numpy.random.default_rng(seed).permutation(unique set ids); first round(test_frac*n) -> test",
    }
    return is_train, is_test, info


def standardise(X: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X[train_mask].mean(0)
    sd = X[train_mask].std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return (X - mu) / sd, mu, sd


# ------------------------------------------------------------------- C3 mapper
def _top_mappers(mapper: np.ndarray, train_mask: np.ndarray, k: int) -> list[int]:
    s = pd.Series(mapper[train_mask])
    s = s[s >= 0]
    vc = s.value_counts()
    return [int(v) for v in vc.index[:k]]


def _numpy_ridge_ovr(X: np.ndarray, y: np.ndarray, n_classes: int,
                     alpha: float = 1.0) -> np.ndarray:
    """Closed-form ridge one-vs-rest fallback (standardised X)."""
    Y = np.zeros((len(y), n_classes), dtype=np.float64)
    Y[np.arange(len(y)), y] = 1.0
    d = X.shape[1]
    A = X.T @ X + alpha * np.eye(d)
    B = X.T @ Y
    return np.linalg.solve(A, B)


def mapper_probe(Xs: np.ndarray, mapper: np.ndarray, is_train: np.ndarray,
                 is_test: np.ndarray, k: int, seed: int) -> dict:
    top = _top_mappers(mapper, is_train, k)
    top_idx = {m: i for i, m in enumerate(top)}
    keep = np.array([m in top_idx for m in mapper], dtype=bool)
    tr = is_train & keep
    te = is_test & keep
    y_tr = np.array([top_idx[m] for m in mapper[tr]], dtype=np.int64)
    y_te = np.array([top_idx[m] for m in mapper[te]], dtype=np.int64)
    X_tr, X_te = Xs[tr], Xs[te]
    n_classes = len(top)

    majority = int(np.bincount(y_tr, minlength=n_classes).argmax())
    majority_acc = float(np.mean(y_te == majority))

    use_sklearn = False
    pred = None
    n_iter = None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
        use_sklearn = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clf = LogisticRegression(
                max_iter=200, C=1.0, solver="lbfgs", tol=1e-4,
            )
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_te)
            n_iter = [int(v) for v in np.atleast_1d(clf.n_iter_)]
            warn_msgs = [str(w.message) for w in caught][:5]
        acc = float(np.mean(pred == y_te))
        f1_macro = float(f1_score(y_te, pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_te, pred, average="weighted", zero_division=0))
        acc_per_class = np.array(
            [np.mean(pred[y_te == c] == c) if np.any(y_te == c) else np.nan
             for c in range(n_classes)])
        model = {"kind": "sklearn.LogisticRegression",
                 "solver": "lbfgs", "max_iter": 200, "C": 1.0, "tol": 1e-4,
                 "n_iter": n_iter, "warnings": warn_msgs}
    except Exception as exc:  # pragma: no cover - fallback path
        use_sklearn = False
        W = _numpy_ridge_ovr(X_tr, y_tr, n_classes)
        pred = np.argmax(X_te @ W, axis=1)
        acc = float(np.mean(pred == y_te))
        f1_macro = float("nan")
        f1_weighted = float("nan")
        acc_per_class = np.array([np.nan] * n_classes)
        model = {"kind": "numpy ridge one-vs-rest", "alpha": 1.0,
                 "n_iter": None, "note": f"sklearn unavailable/failed: {exc}"}

    # sanity: trivial constant predictor R2-style baselines
    return {
        "task": "mapper-ID probe (C3)",
        "content_dim": int(X_tr.shape[1]),
        "top_k_mappers": int(k),
        "n_classes": int(n_classes),
        "n_train_charts": int(tr.sum()),
        "n_test_charts": int(te.sum()),
        "min_train_charts_per_mapper": int(np.bincount(y_tr, minlength=n_classes).min()),
        "max_train_charts_per_mapper": int(np.bincount(y_tr, minlength=n_classes).max()),
        "chance_accuracy": float(1.0 / n_classes),
        "majority_class_train_charts": int(np.bincount(y_tr, minlength=n_classes)[majority]),
        "top1_accuracy_test": acc,
        "top1_accuracy_lift_over_chance": float(acc - 1.0 / n_classes),
        "majority_baseline_test_accuracy": majority_acc,
        "top1_lift_over_majority": float(acc - majority_acc),
        "macro_f1_test": f1_macro,
        "weighted_f1_test": f1_weighted,
        "per_class_accuracy_mean": float(np.nanmean(acc_per_class)) if np.isfinite(acc_per_class).any() else float("nan"),
        "model": model,
        "seed": int(seed),
    }


# ------------------------------------------------------------ C4 decodability
def _targets(ids: np.ndarray, audit_path: Path) -> tuple[np.ndarray, list[str]]:
    a = pd.read_parquet(audit_path).set_index("beatmap_id")
    a = a.reindex(ids)
    cols = {
        "log1p_n_sv_changes": np.log1p(a["n_sv_changes"].to_numpy(np.float64)),
        "kiai_frac": a["kiai_frac"].to_numpy(np.float64),
        "log2_bpm_p50": np.log2(np.clip(a["bpm_p50"].to_numpy(np.float64), 1.0, 1000.0)),
        "hold_frac_gt1s": a["hold_frac_gt1s"].to_numpy(np.float64),
        "vol_resolved_p50": a["vol_resolved_p50"].to_numpy(np.float64),
        "hs_any_ext_frac": a["hs_any_ext_frac"].to_numpy(np.float64),
    }
    missing = [k for k, v in cols.items() if not np.isfinite(v).all()]
    if missing:
        raise SystemExit(f"{audit_path}: non-finite targets {missing}")
    Y = np.column_stack([cols[t] for t in TARGETS]).astype(np.float64)
    return Y, TARGETS


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _pearson(y: np.ndarray, pred: np.ndarray) -> float:
    yc = y - y.mean()
    pc = pred - pred.mean()
    den = np.sqrt(np.sum(yc * yc) * np.sum(pc * pc))
    return float(np.sum(yc * pc) / den) if den > 0 else float("nan")


def attr_decodability(Xs: np.ndarray, Y: np.ndarray, is_train: np.ndarray,
                      is_test: np.ndarray) -> dict:
    try:
        from sklearn.linear_model import RidgeCV
        alphas = np.logspace(-1, 4, 6)
        ridge_kind = "sklearn.RidgeCV(alpha by GCV on train charts)"
    except Exception:  # pragma: no cover
        RidgeCV = None
        alphas = np.array([1.0])
        ridge_kind = "closed-form ridge alpha=1.0"
    X_tr, X_te = Xs[is_train], Xs[is_test]
    out = {}
    for j, t in enumerate(TARGETS):
        y_tr, y_te = Y[is_train, j], Y[is_test, j]
        if RidgeCV is not None:
            reg = RidgeCV(alphas=alphas)
            reg.fit(X_tr, y_tr)
            alpha = float(reg.alpha_)
            pred_tr = reg.predict(X_tr)
            pred_te = reg.predict(X_te)
        else:  # pragma: no cover
            d = X_tr.shape[1]
            A = X_tr.T @ X_tr + 1.0 * np.eye(d)
            w = np.linalg.solve(A, X_tr.T @ y_tr)
            alpha = 1.0
            pred_tr = X_tr @ w
            pred_te = X_te @ w
        out[t] = {
            "n_train": int(y_tr.size),
            "n_test": int(y_te.size),
            "target_mean": float(y_tr.mean()),
            "target_std": float(y_tr.std()),
            "ridge_alpha": alpha,
            "r2_train": _r2(y_tr, pred_tr),
            "r2_test": _r2(y_te, pred_te),
            "pearson_test": _pearson(y_te, pred_te),
            "test_mean_predictor_r2": _r2(y_te, np.full_like(y_te, y_tr.mean())),
            "test_pred_std": float(np.std(pred_te)),
        }
    return {
        "task": "attribute decodability (C4)",
        "model": ridge_kind,
        "alphas_grid": [float(a) for a in alphas],
        "n_train_charts": int(is_train.sum()),
        "n_test_charts": int(is_test.sum()),
        "targets": out,
    }


# --------------------------------------------------------------------- driver
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--content-path", default="data/processed/chart_content_d2v_holdout.parquet")
    ap.add_argument("--meta", default="data/processed/beatmap_meta_full.csv")
    ap.add_argument("--audit", default="data/processed/d2v_attr_audit_per_chart.parquet")
    ap.add_argument("--out-dir", default="results/d2v_attr_diag")
    ap.add_argument("--mode", default="both", choices=["mapper", "attr", "both"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--top-mappers", type=int, default=50)
    ap.add_argument("--mapper-out", default=None)
    ap.add_argument("--attr-out", default=None)
    ap.add_argument("--tag", default="d2v8")
    return ap.parse_args(argv)


def _provenance(args) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            text=True).strip()
    except Exception:
        commit = "unknown"
    cp = Path(args.content_path)
    md5 = hashlib.md5(cp.read_bytes()).hexdigest() if cp.exists() else None
    return {"host": "local", "git_commit": commit, "content_md5": md5,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def main(argv=None) -> int:
    t0 = time.time()
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids, X, dim = load_content(Path(args.content_path))
    meta = load_meta(Path(args.meta))
    meta = meta.reindex(ids)
    if meta["beatmapset_id"].isna().any():
        n = int(meta["beatmapset_id"].isna().sum())
        raise SystemExit(f"{n} charts have no beatmapset_id in {args.meta}")
    if meta["mapper_id"].isna().any():
        n = int(meta["mapper_id"].isna().sum())
        raise SystemExit(f"{n} charts have no mapper_id in {args.meta}")
    set_ids = meta["beatmapset_id"].to_numpy(np.int64)
    mapper = meta["mapper_id"].to_numpy(np.int64)

    is_train, is_test, split_info = split_sets(ids, set_ids, meta,
                                               args.test_frac, args.seed)
    Xs, mu, sd = standardise(X, is_train)
    n_constant_cols = int(np.sum(sd == 1.0))
    print(f"[data] content={args.content_path} n={len(ids):,} dim={dim} md5={_provenance(args)['content_md5'][:8]}")
    print(f"[split] set-disjoint: train {split_info['n_train_charts']:,} charts / "
          f"{split_info['n_train_sets']:,} sets | test {split_info['n_test_charts']:,} charts / "
          f"{split_info['n_test_sets']:,} sets")
    print(f"[scale] standardised on train charts; {n_constant_cols} near-constant dims")

    prov = _provenance(args)
    manifest = {
        "content_path": str(args.content_path),
        "content_dim": dim,
        "n_charts": int(len(ids)),
        "meta": str(args.meta),
        "audit": str(args.audit),
        "split": split_info,
        "standardisation": "content z-scored with train-chart mean/std; sd<1e-8 -> 1.0",
        "n_near_constant_dims": n_constant_cols,
        "seed": int(args.seed),
        "provenance": prov,
    }

    if args.mode in ("mapper", "both"):
        t1 = time.time()
        res = mapper_probe(Xs, mapper, is_train, is_test, args.top_mappers, args.seed)
        res["manifest"] = manifest
        res["seconds"] = round(time.time() - t1, 1)
        path = Path(args.mapper_out or out_dir / f"mapper_probe_{args.tag}.json")
        path.write_text(json.dumps(res, indent=2))
        print(f"[C3] mapper top1={res['top1_accuracy_test']:.4f} "
              f"(chance {res['chance_accuracy']:.4f}, majority "
              f"{res['majority_baseline_test_accuracy']:.4f}) "
              f"n_test={res['n_test_charts']:,} -> {path} ({res['seconds']}s)")

    if args.mode in ("attr", "both"):
        t2 = time.time()
        Y, names = _targets(ids, Path(args.audit))
        res = attr_decodability(Xs, Y, is_train, is_test)
        res["manifest"] = manifest
        res["target_names"] = names
        res["seconds"] = round(time.time() - t2, 1)
        path = Path(args.attr_out or out_dir / f"attr_decodability_{args.tag}.json")
        path.write_text(json.dumps(res, indent=2))
        print(f"[C4] attribute decodability (test R2):")
        for t in TARGETS:
            r = res["targets"][t]
            print(f"  {t:20s} R2={r['r2_test']: .4f}  (train {r['r2_train']: .4f}, "
                  f"alpha={r['ridge_alpha']:g}, pearson={r['pearson_test']: .4f})")
        print(f"  -> {path} ({res['seconds']}s)")

    print(f"[done] {round(time.time() - t0, 1)}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
