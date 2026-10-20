import pandas as pd, numpy as np
pd.set_option("display.width", 220)
bm = pd.read_csv("data/interim/beatmaps.csv")
mh = pd.read_csv("data/interim/mania_high.csv")
st = pd.read_csv("data/interim/user_stats_mania.csv")
sc = pd.read_csv("data/interim/scores_recent.csv")
pc = pd.read_csv("data/interim/playcount.csv")

print("=== users ===")
print("stats(mania top1000):", len(st), "mania_high users:", mh.user_id.nunique(),
      "scores users:", sc.user_id.nunique(), "playcount users:", pc.user_id.nunique())
print("mania_high users in stats:", mh.user_id.isin(st.user_id).mean().round(4),
      "| scores users in stats:", sc.user_id.isin(st.user_id).mean().round(4))
print("rank_score_index range in stats:", st.rank_score_index.min(), st.rank_score_index.max())

print("\n=== mania_high: is it the full play history? ===")
per_user = mh.groupby("user_id").size().rename("high_rows")
cmp_ = st.set_index("user_id").join(per_user).dropna(subset=["high_rows"])
cmp_["playcount"] = pd.to_numeric(cmp_.playcount, errors="coerce")
cmp_["high_rows"] = cmp_.high_rows.astype(float)
print(cmp_[["playcount","high_rows"]].describe().to_string())
print("ratio high_rows/playcount:", (cmp_.high_rows/cmp_.playcount).describe().to_string())

print("\n=== mania_high rows by beatmap mode/keys ===")
j = mh.merge(bm[["beatmap_id","playmode","keys","star","max_combo","countTotal"]], on="beatmap_id", how="left")
print(j.groupby([j.playmode.fillna(-1), j["keys"]]).size().sort_values(ascending=False).head(12).to_string())
print("date range:", mh.date.min(), "->", mh.date.max())

print("\n=== scores table ===")
print("ruleset_id counts:", sc.ruleset_id.value_counts().to_dict())
print("passed counts:", sc.passed.value_counts().to_dict())
print("ended_at range:", sc.ended_at.min(), "->", sc.ended_at.max())
print("distinct users:", sc.user_id.nunique(), "beatmaps:", sc.beatmap_id.nunique())
pj = sc.merge(bm[["beatmap_id","playmode","keys"]], on="beatmap_id", how="left")
print("scores by mode/keys:")
print(pj.groupby([pj.playmode.fillna(-1), pj["keys"]]).size().sort_values(ascending=False).head(12).to_string())
print("mods top:", sc.mods.value_counts().head(8).to_dict())
print("pairs in scores:", sc.groupby(["user_id","beatmap_id"]).ngroups,
      "pairs>=2:", int((sc.groupby(["user_id","beatmap_id"]).size()>=2).sum()))
