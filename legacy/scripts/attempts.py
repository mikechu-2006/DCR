import pandas as pd, numpy as np
pd.set_option("display.width", 200)
bm = pd.read_csv("data/interim/beatmaps.csv")
sc = pd.read_csv("data/interim/scores_detail.csv")
pc = pd.read_csv("data/interim/playcount.csv")
bm4 = bm[(bm.playmode == 3) & (bm["keys"] == 4)]
s4 = sc.merge(bm4[["beatmap_id"]], on="beatmap_id", how="inner")
pop = s4.groupby("beatmap_id").user_id.nunique().sort_values(ascending=False)
top = set(pop.head(1000).index)
sub = s4[s4.beatmap_id.isin(top)]
cells = sub.groupby(["user_id", "beatmap_id"]).size().rename("n_scores").reset_index()
mg = cells.merge(pc, on=["user_id", "beatmap_id"], how="left")
mg["playcount"] = pd.to_numeric(mg.playcount, errors="coerce")
print("cells:", len(mg), "with playcount record:", int(mg.playcount.notna().sum()))
print("n_scores describe:", mg.n_scores.describe()[["mean","50%","max"]].round(2).to_dict())
print("playcount describe:", mg.playcount.describe()[["mean","50%","max"]].round(2).to_dict())
print("ratio playcount/n_scores:", (mg.playcount/mg.n_scores).describe()[["mean","50%","75%","max"]].round(2).to_dict())
print("\ncells where total attempts > recorded scores:", int((mg.playcount > mg.n_scores).sum()), "/", len(mg))
print("cells with playcount == 1:", int((mg.playcount == 1).sum()))
