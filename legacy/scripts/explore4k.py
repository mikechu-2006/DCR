import pandas as pd, numpy as np
pd.set_option("display.width", 220)
bm = pd.read_csv("data/interim/beatmaps.csv")
mh = pd.read_csv("data/interim/mania_high.csv")
sc = pd.read_csv("data/interim/scores_recent.csv")

bm4 = bm[(bm.playmode == 3) & (bm["keys"] == 4)].copy()
print("4K mania beatmaps in beatmaps.csv:", len(bm4))

mh4 = mh.merge(bm4[["beatmap_id", "max_combo", "countTotal", "star", "playcount"]], on="beatmap_id", how="inner")
sc4 = sc.merge(bm4[["beatmap_id", "max_combo", "countTotal", "star", "playcount"]], on="beatmap_id", how="inner")
print("\n--- mania_high 4K ---")
print("rows:", len(mh4), "users:", mh4.user_id.nunique(), "beatmaps:", mh4.beatmap_id.nunique())
pair = mh4.groupby(["user_id", "beatmap_id"]).size()
print("pairs:", len(pair), "density(pairs/(users*maps)):",
      round(len(pair) / (mh4.user_id.nunique() * mh4.beatmap_id.nunique()), 4))
vc = pair.value_counts().sort_index()
print("plays/pair: 1:%d 2:%d 3:%d 4:%d 5+:%d  max:%d" % (
    vc.get(1, 0), vc.get(2, 0), vc.get(3, 0), vc.get(4, 0), int(vc[vc.index >= 5].sum()), int(pair.max())))

print("\n--- scores 4K ---")
print("rows:", len(sc4), "users:", sc4.user_id.nunique(), "beatmaps:", sc4.beatmap_id.nunique())
pair2 = sc4.groupby(["user_id", "beatmap_id"]).size()
vc2 = pair2.value_counts().sort_index()
print("pairs:", len(pair2), "density:", round(len(pair2)/(sc4.user_id.nunique()*sc4.beatmap_id.nunique()), 4))
print("plays/pair: 1:%d 2:%d 3:%d 4:%d 5+:%d max:%d" % (
    vc2.get(1, 0), vc2.get(2, 0), vc2.get(3, 0), vc2.get(4, 0), int(vc2[vc2.index >= 5].sum()), int(pair2.max())))

print("\n--- overlap between the two sources (score_id <-> legacy_score_id) ---")
sc_ids = set(sc4.id.astype("int64"))
print("scores.id in mania_high.score_id:", mh4.score_id.astype("int64").isin(sc_ids).sum(), "/", len(mh4))

print("\n--- accuracy distribution (4K, scores) ---")
acc = pd.to_numeric(sc4.accuracy, errors="coerce")
print(acc.describe().to_string())
print("acc==1 exactly:", int((acc >= 1.0).sum()), "acc>0.9999:", int((acc > 0.9999).sum()),
      "acc>0.999:", int((acc > 0.999).sum()), "acc>0.99:", int((acc > 0.99).sum()))
print("1-acc quantiles:", (1-acc).quantile([0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.99]).to_dict())
print("log10(1-acc) quantiles:", np.log10((1-acc).replace(0, np.nan)).quantile([0.01, 0.05, 0.5, 0.95, 0.99]).to_dict())
print("\nrank dist:", sc4["rank"].value_counts().to_dict())
print("mods dist:", sc4.mods.value_counts().head(10).to_dict())
