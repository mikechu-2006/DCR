"""Gauge-aware analysis of the learned level vectors.

For a chart c the model predicts  L = b0 + b_u + b_m + (P_u - C_c) . D_c .
So C_c is only identified *up to vectors orthogonal to D_c*: only C_c . D_c is meaningful.
This script therefore reports both
  * raw cosine similarity between the NM and DT vectors of the same map (only meaningful when
    the discrimination is shared), and
  * the gauge-invariant chart difficulty  s_c = b_m,c + (Pbar - C_c) . D_c .
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

PROC = Path("data/processed")


def main(path):
    d = np.load(path, allow_pickle=True)
    C, bm, P = d["C"], d["bm"], d["P"]
    bid, rate = d["beatmap_ids"].astype("int64"), d["rates"].astype("int64")
    D = d["D"] if "D" in d.files else None
    dg = d["d_global"] if "d_global" in d.files else np.zeros(C.shape[1])
    if D is None or not np.any(D):
        D = np.tile(dg, (len(C), 1))
    Pbar = P.mean(0)
    print(f"{Path(path).name}: charts={len(C)} dim={C.shape[1]} "
          f"shared_discrimination={bool(np.any(dg))}")
    s = bm + (Pbar[None, :] - C) * D
    s = s.sum(1)
    df = pd.DataFrame({"beatmap_id": bid, "rate": rate, "bm": bm, "s": s})
    nm = df[df.rate == 0].set_index("beatmap_id")
    dt = df[df.rate == 1].set_index("beatmap_id")
    common = nm.index.intersection(dt.index)
    print(f"  NM charts={len(nm)} DT charts={len(dt)} both={len(common)}")
    if len(common) > 10:
        i, j = nm.index.get_indexer(common), dt.index.get_indexer(common)
        cos = (C[i] * C[j]).sum(1) / (np.linalg.norm(C[i], axis=1) * np.linalg.norm(C[j], axis=1) + 1e-9)
        print("  cos(C_NM, C_DT): mean %.3f median %.3f p10 %.3f p90 %.3f" %
              (cos.mean(), np.median(cos), np.percentile(cos, 10), np.percentile(cos, 90)))
        print("  spearman(model difficulty NM, DT) = %.4f" %
              spearmanr(nm.loc[common].s, dt.loc[common].s).statistic)
        print("  DT-NM difficulty delta: mean %.3f (x%.2f loss)" %
              ((dt.loc[common].s - nm.loc[common].s).mean(), np.exp((dt.loc[common].s - nm.loc[common].s).mean())))
    star = pd.read_csv("data/interim/beatmap_difficulty_4k.csv")
    star["star"] = pd.to_numeric(star.star, errors="coerce")
    mods = pd.to_numeric(star.mods, errors="coerce").fillna(0).astype("int64")
    star["rate"] = np.where(mods & (64 | 512) != 0, 1, np.where(mods & 256 != 0, 2, 0))
    star = star[star.star > 0].sort_values("star").groupby(["beatmap_id", "rate"], as_index=False).star.max()
    mg = df.merge(star, on=["beatmap_id", "rate"], how="left")
    print("  spearman(model difficulty vs official star) = %.4f" %
          spearmanr(mg.s, mg.star).statistic)
    print("  spearman(b_m vs official star) = %.4f" % spearmanr(mg.bm, mg.star).statistic)
    df.to_csv(PROC / f"vector_analysis_{Path(path).stem}.csv", index=False)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
