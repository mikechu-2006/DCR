import pandas as pd, numpy as np
pd.set_option("display.width", 250)
p = pd.read_parquet("data/processed/plays_4k.parquet")
m = p[p.source == "modern"].sample(8, random_state=7)
d = pd.read_csv("data/interim/scores_detail.csv", usecols=["id","accuracy"]).set_index("id")
m = m.assign(stored_acc=m.score_id.map(d.accuracy))
m["check_v2"] = m.acc_v2
m["diff"] = (m.stored_acc - m.check_v2).abs()
print(m[["score_id","total","c320","c300","c200","c100","c50","cmiss","acc_flat","acc_v2","stored_acc","diff","loss_v2"]].round(7).to_string(index=False))
print("\nmax |stored - acc_v2| over 200k modern rows:")
s = p[p.source == "modern"].sample(200000, random_state=1).assign(stored=lambda x: x.score_id.map(d.accuracy))
print(float((s.stored - s.acc_v2).abs().max()))
print("\nloss_v2 min/max:", p.loss_v2.min().round(4), p.loss_v2.max().round(4))
print("delta floor example (total=1000):", np.log(25/(305*1000)).round(4))
