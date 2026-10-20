import pandas as pd, numpy as np
pd.set_option("display.width", 200)
bm = pd.read_csv("data/interim/beatmaps.csv")
mh = pd.read_csv("data/interim/mania_high.csv")
sc = pd.read_csv("data/interim/scores_detail.csv")
bm4 = set(bm[(bm.playmode == 3) & (bm["keys"] == 4)].beatmap_id)

mh4 = mh[mh.beatmap_id.isin(bm4)]
sc4 = sc[sc.beatmap_id.isin(bm4)]
print("mania_high 4K:", len(mh4), "| scores 4K:", len(sc4))
leg = pd.to_numeric(sc4.legacy_score_id, errors="coerce").dropna().astype("int64")
mh_ids = set(mh4.score_id.astype("int64"))
print("scores.legacy_score_id sample:", leg.head(3).tolist())
print("overlap legacy_score_id <-> mania_high.score_id:", leg.isin(mh_ids).sum(), "/", len(leg))
print("also check scores.id in mania_high:", sc4.id.astype('int64').isin(mh_ids).sum())
print("\nrank distribution:")
print("mania_high:", mh4["rank"].value_counts(normalize=True).round(4).to_dict())
print("scores    :", sc4["rank"].value_counts(normalize=True).round(4).to_dict())

# max vs median on multi-play cells (ScoreV2 acc from JSON)
for c in ["n_perfect","n_great","n_good","n_ok","n_meh","n_miss","max_total"]:
    sc4 = sc4.copy()
    sc4[c] = pd.to_numeric(sc4[c], errors="coerce")
sc4 = sc4[sc4.max_total > 0]
sc4["acc"] = (305*sc4.n_perfect + 300*sc4.n_great + 200*sc4.n_good + 100*sc4.n_ok + 50*sc4.n_meh)/(305*sc4.max_total)
g = sc4.groupby(["user_id","beatmap_id"]).acc
a = pd.DataFrame({"med": g.median(), "max": g.max(), "min": g.min(), "n": g.size()})
multi = a[a.n >= 2]
print("\nmulti-play cells:", len(multi), "of", len(a))
d = multi["max"] - multi["med"]
print("max-median acc gap: mean %.5f median %.5f p90 %.5f max %.5f" % (d.mean(), d.median(), d.quantile(.9), d.max()))
lg = lambda x: np.log10(np.clip(1-x, 1e-9, None))
print("log10-loss gap (max vs median): mean %.3f median %.3f p90 %.3f" % ((lg(multi['med'])-lg(multi['max'])).mean(), (lg(multi['med'])-lg(multi['max'])).median(), (lg(multi['med'])-lg(multi['max'])).quantile(.9)))
print("cells with acc==1 (all-MAX):", int((sc4.acc >= 1).sum()), "cells whose median==1:", int((a.med >= 1).sum()), "max==1:", int((a['max'] >= 1).sum()))
