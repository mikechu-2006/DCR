import pandas as pd, numpy as np
pd.set_option("display.width", 220)
d = pd.read_csv("data/interim/scores_detail.csv")
print("rows:", len(d), "cols:", list(d.columns))
for c in ["accuracy","n_perfect","n_great","n_good","n_ok","n_meh","n_miss","max_total"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
sub = d[(d.max_total > 0)].sample(min(300000, len(d)), random_state=0)
sub["acc320"] = (320*sub.n_perfect + 300*sub.n_great + 200*sub.n_good + 100*sub.n_ok + 50*sub.n_meh)/(320*sub.max_total)
sub["accflat"] = (300*(sub.n_perfect + sub.n_great) + 200*sub.n_good + 100*sub.n_ok + 50*sub.n_meh)/(300*sub.max_total)
for f in ["acc320","accflat"]:
    diff = (sub[f] - sub.accuracy).abs()
    print(f, "max|diff|:", round(diff.max(), 6), "mean:", round(diff.mean(), 8), "exact(<1e-6):", round((diff < 1e-6).mean(), 5))
print("\nchecks:")
print("judgement sums vs max_total equal:", int((sub[['n_perfect','n_great','n_good','n_ok','n_meh','n_miss']].sum(axis=1) == sub.max_total).mean()*100), "%")
print("acc==1 rows overall:", int((d.accuracy >= 1).sum()), "/", len(d))
print("legacy_score_id notnull:", int(d.legacy_score_id.notna().sum()))
