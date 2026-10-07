import pandas as pd, numpy as np
pd.set_option("display.width", 240)
d = pd.read_csv("data/interim/scores_detail.csv")
for c in ["accuracy","n_perfect","n_great","n_good","n_ok","n_meh","n_miss","max_total","total_score","max_combo"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
n = d[["n_perfect","n_great","n_good","n_ok","n_meh","n_miss"]]
d["sumj"] = n.sum(axis=1)
d["nz"] = (n > 0).sum(axis=1)

pure = d[(d.nz == 1)]
print("rows with a single judgement type:", len(pure))
for col in ["n_perfect","n_great","n_good","n_ok","n_meh","n_miss"]:
    s = pure[pure[col] > 0]
    if len(s):
        print(f"  only {col:10s} n={len(s):7d} accuracy: ", s.accuracy.describe()[["mean","min","max"]].round(6).to_dict())

two = d[(d.nz == 2) & (d.n_miss == 0) & (d.max_total == d.sumj)].head(200000)
print("\nsample rows (nz==2, no miss):")
print(two[["n_perfect","n_great","n_good","n_ok","n_meh","max_total","accuracy"]].head(8).to_string())

# least squares fit of weights
s = d[(d.max_total > 0) & (d.sumj == d.max_total) & (d.sumj > 0)].sample(200000, random_state=1)
A = s[["n_perfect","n_great","n_good","n_ok","n_meh","n_miss"]].to_numpy(float)
y = (s.accuracy.to_numpy(float) * s.max_total.to_numpy(float))
w, *_ = np.linalg.lstsq(A, y, rcond=None)
print("\nleast-squares weights (normalised to 300):", (w / w[1] * 300).round(3))
resid = A @ w - y
print("residual: mean|.|", np.abs(resid).mean().round(4), "rms", np.sqrt((resid**2).mean()).round(4), "scale(max_total) mean", s.max_total.mean().round(1))
pred = (A @ w) / s.max_total.to_numpy(float)
print("acc corr:", np.corrcoef(pred, s.accuracy)[0,1].round(6), "max|err|", np.abs(pred - s.accuracy).max().round(5))
print("\nexample mismatch rows:")
print(s.assign(pred=pred).head(5)[["n_perfect","n_great","n_good","n_ok","n_meh","n_miss","max_total","accuracy","pred"]].to_string())
