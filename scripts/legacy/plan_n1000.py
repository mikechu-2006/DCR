import pandas as pd, numpy as np
pd.set_option("display.width", 220)
bm = pd.read_csv("data/interim/beatmaps.csv")
sc = pd.read_csv("data/interim/scores_recent.csv")
bm4 = bm[(bm.playmode == 3) & (bm["keys"] == 4)]
s4 = sc.merge(bm4[["beatmap_id", "star", "countTotal"]], on="beatmap_id", how="inner")
s4["accuracy"] = pd.to_numeric(s4.accuracy, errors="coerce")

x = s4[s4["rank"].isin(["X", "XH"])]
print("rank X rows:", len(x), "acc==1:", int((x.accuracy >= 1).sum()),
      "acc in [0.999,1):", int(((x.accuracy < 1) & (x.accuracy > 0.999)).sum()))
print("X acc quantiles:", x.accuracy.quantile([0, .05, .25, .5, .75, 1]).round(6).to_dict())
print("-> modern scores.accuracy uses 320-weighted (MAX=320) formula" if (x.accuracy < 1).mean() > 0.5 else "-> flat formula")

# top maps by distinct players
pop = s4.groupby("beatmap_id").user_id.nunique().sort_values(ascending=False)
top1000 = pop.head(1000).index
sub = s4[s4.beatmap_id.isin(top1000)]
pair = sub.groupby(["user_id", "beatmap_id"]).size()
print("\n=== top1000 maps x %d users ===" % sub.user_id.nunique())
print("rows:", len(sub), "users:", sub.user_id.nunique(), "maps:", sub.beatmap_id.nunique())
print("cells:", len(pair), "matrix:", sub.user_id.nunique()*1000, "density:", round(len(pair)/(sub.user_id.nunique()*1000), 4))
vc = pair.value_counts().sort_index()
print("plays/pair: 1:%d 2:%d 3:%d 4:%d 5+:%d" % (vc.get(1,0), vc.get(2,0), vc.get(3,0), vc.get(4,0), int(vc[vc.index>=5].sum())))
print("maps with <10 distinct players:", int((pop.head(1000) < 10).sum()), "| median distinct players/map:", int(pop.head(1000).median()))

# median-vs-max on the submatrix, in flat-acc space (loss = 1 - acc proxy)
g = sub.groupby(["user_id","beatmap_id"]).accuracy
agg = pd.DataFrame({"median": g.median(), "max": g.max(), "mean": g.mean(), "n": g.size()})
print("\ncells with median acc == 1.0:", int((agg["median"] >= 1).sum()), "/", len(agg))
print("cells with max acc == 1.0:", int((agg["max"] >= 1).sum()))
print("\nfirst 5 cells:")
print(agg.head().to_string())
