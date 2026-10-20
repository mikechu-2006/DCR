"""How much does DT change the learned level vector? (chart-level model, same embedding space)"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

PROC = Path("data/processed")
d = np.load(PROC / "stage1_mirt_loss_v2_med.npz", allow_pickle=True)
C, D, bm = d["C"], d["D"], d["bm"]
bid, rate = d["beatmap_ids"].astype("int64"), d["rates"].astype("int64")
df = pd.DataFrame({"beatmap_id": bid, "rate": rate, "bm": bm})
nm = df[df.rate == 0].set_index("beatmap_id")
dt = df[df.rate == 1].set_index("beatmap_id")
common = nm.index.intersection(dt.index)
print(f"charts: NM={len(nm)} DT={len(dt)} both={len(common)}")


def cos(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)


i_nm = nm.loc[common].index.map(lambda x: list(nm.index).index(x))
i_dt = dt.loc[common].index.map(lambda x: list(dt.index).index(x))
Cn, Cd = C[i_nm], C[i_dt]
Dn, Dd = D[i_nm], D[i_dt]
cn = cos(Cn, Cd)
dn = cos(Dn, Dd)
print("cos(C_NM, C_DT): mean %.3f median %.3f p10 %.3f p90 %.3f" % (cn.mean(), np.median(cn), np.percentile(cn, 10), np.percentile(cn, 90)))
print("cos(D_NM, D_DT): mean %.3f median %.3f p10 %.3f p90 %.3f" % (dn.mean(), np.median(dn), np.percentile(dn, 10), np.percentile(dn, 90)))
print("cos(b_m):", round(float(np.corrcoef(nm.loc[common].bm, dt.loc[common].bm)[0, 1]), 3))
print("spearman(b_m NM, b_m DT):", round(float(spearmanr(nm.loc[common].bm, dt.loc[common].bm).statistic), 4))

# per-map DT-vs-NM difficulty delta vs map properties
bmaps = pd.read_csv("data/interim/beatmaps.csv").set_index("beatmap_id")
info = bmaps.reindex(common)
delta_bm = (dt.loc[common].bm.values - nm.loc[common].bm.values)
out = pd.DataFrame({"beatmap_id": common, "delta_bm": delta_bm,
                    "star": pd.to_numeric(info.star, errors="coerce").values,
                    "notes": pd.to_numeric(info.countTotal, errors="coerce").values,
                    "max_combo": pd.to_numeric(info.max_combo, errors="coerce").values})
out = out.dropna()
print("\nspearman(DT-NM b_m  vs  NM star):", round(float(spearmanr(out.delta_bm, out.star).statistic), 4))
print("spearman(DT-NM b_m  vs  note count):", round(float(spearmanr(out.delta_bm, out.notes).statistic), 4))
print("delta_bm: mean %.3f std %.3f  -> DT costs %.2fx more loss on average" % (
    out.delta_bm.mean(), out.delta_bm.std(), float(np.exp(out.delta_bm.mean()))))
out.to_csv(PROC / "dt_vs_nm_vectors.csv", index=False)
print("saved", PROC / "dt_vs_nm_vectors.csv")
