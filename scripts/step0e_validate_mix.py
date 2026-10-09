#!/usr/bin/env python
"""Step 0e - may the freshly computed rows be MIXED with the published CM3P table?

The published OliBomby/CM3P-Embeddings-244K table was encoded WITH audio fusion.  A box that
only has .osu files can only encode WITHOUT audio (no audio tokens are emitted at all when
include_audio=False).  Those are two different input distributions, so concatenating them
without checking would inject a domain shift that has nothing to do with the chart -- and it
would be invisible in every downstream metric.

This script recomputes a fixed probe of charts that ARE in the published table, using exactly
the same code path as the new rows, and compares the result against the reference vectors
shipped in manifests/cm3p_probe_1k.npz.

    python scripts/step0e_validate_mix.py \
        --computed data/processed/chart_content_cm3p_probe.parquet \
        --probe-npz manifests/cm3p_probe_1k.npz

Decision rule printed at the end; nothing is written except an optional --json report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def nn_self(S: np.ndarray) -> np.ndarray:
    """Nearest neighbour of every row among the rows themselves (diagonal excluded)."""
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    return S.argmax(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-npz", default="manifests/cm3p_probe_1k.npz")
    ap.add_argument("--computed", required=True,
                    help="content table (step0c output) covering the probe ids")
    ap.add_argument("--json", default=None, help="optional path for a machine-readable report")
    args = ap.parse_args()

    ref = np.load(args.probe_npz)
    ids, R = ref["ids"].astype(np.int64), unit(np.asarray(ref["embeddings"], dtype=np.float64))
    print(f"probe: {len(ids)} charts, dim {R.shape[1]}  ({Path(args.probe_npz).name})")

    tab = pd.read_parquet(args.computed, columns=["beatmap_id", "embedding", "source"])
    have = {int(b): np.asarray(e, dtype=np.float64) for b, e in zip(tab.beatmap_id, tab.embedding)}
    missing = [int(i) for i in ids if int(i) not in have]
    keep = np.array([int(i) in have for i in ids])
    print(f"computed: {len(have):,} rows; probe coverage {int(keep.sum())}/{len(ids)}")
    if missing:
        print(f"  WARNING: {len(missing)} probe ids missing (first: {missing[:5]}) -- the verdict "
              f"below is computed on the rest")
    if keep.sum() < 32:
        raise SystemExit("fewer than 32 probe ids present; nothing to conclude")
    src = tab.source.mode().iloc[0]
    print(f"  probe rows came from source code {int(src)} "
          f"(0=hf-precomputed, 1=local-noaudio, 2=local-audio)")

    X = unit(np.stack([have[int(i)] for i in ids[keep]]))
    Rk = R[keep]

    cos_same = np.einsum("ij,ij->i", X, Rk)
    off = ~np.eye(len(Rk), dtype=bool)
    cos_diff_ref = Rk @ Rk.T
    cos_diff_new = X @ X.T
    mantel = float(np.corrcoef(cos_diff_ref[off], cos_diff_new[off])[0, 1])
    agree1 = float((nn_self(cos_diff_ref) == nn_self(cos_diff_new)).mean())
    # a shift-invariant view: after centring, are the two point clouds the same shape?
    Rc, Xc = Rk - Rk.mean(0), X - X.mean(0)
    procrustes = float(np.linalg.svd(Rc.T @ Xc, compute_uv=False).sum() /
                       np.sqrt((Rc ** 2).sum() * (Xc ** 2).sum()))

    print("\n--- same chart, recomputed vs published -------------------------------")
    print(f"  cosine  mean {cos_same.mean():.4f}   median {np.median(cos_same):.4f}   "
          f"min {cos_same.min():.4f}   p10 {np.percentile(cos_same, 10):.4f}")
    print("--- calibration: two DIFFERENT charts in the published space ----------")
    print(f"  cosine  mean {cos_diff_ref[off].mean():.4f}   (this is what 'not the same chart' "
          f"looks like; note how high it already is)")
    print(f"  cosine  mean in the recomputed space {cos_diff_new[off].mean():.4f}")
    print("--- does the recomputation preserve the geometry? ---------------------")
    print(f"  Mantel corr of pairwise cosines      {mantel:.4f}   (1.0 = identical geometry)")
    print(f"  nearest-neighbour agreement (recall@1) {agree1:.4f}")
    print(f"  orthogonal-Procrustes similarity     {procrustes:.4f}")

    if cos_same.mean() >= 0.99 and agree1 >= 0.95:
        verdict = ("MIX OK -- the no-audio path reproduces the published vectors to within "
                   "noise; concatenating the two tables is safe")
        code = "ok"
    elif cos_same.mean() >= 0.90 and mantel >= 0.80:
        verdict = ("MIX WITH CAUTION -- same chart is clearly recovered but the two clouds are "
                   "not identical; report the probe numbers alongside any result that mixes "
                   "them, or recompute everything on one side")
        code = "caution"
    else:
        verdict = ("DO NOT MIX -- the no-audio encoding is a different representation. Either "
                   "re-encode the published charts the same way (needs audio -> .osz) or keep "
                   "the two subsets in separate experiments")
        code = "do-not-mix"
    print(f"\nVERDICT [{code}]: {verdict}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "code": code, "verdict": verdict, "n_probe": int(keep.sum()),
            "cosine_same": {"mean": float(cos_same.mean()), "median": float(np.median(cos_same)),
                            "min": float(cos_same.min())},
            "cosine_different_reference": float(cos_diff_ref[off].mean()),
            "cosine_different_recomputed": float(cos_diff_new[off].mean()),
            "mantel": mantel, "recall_at_1": agree1, "procrustes": procrustes,
        }, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
