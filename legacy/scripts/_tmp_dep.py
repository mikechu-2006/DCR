import numpy as np, pandas as pd
cols = ["player_id", "beatmap_id", "timestamp", "playcount_cur", "loss"]
s1 = pd.read_parquet("data/processed/step1_1k.parquet", columns=cols)
s1["ts"] = pd.to_datetime(s1.timestamp).astype("int64") // 10**9
s1 = s1.sort_values(["player_id", "beatmap_id", "ts"], kind="stable")
print("train plays:", len(s1), " cells:", s1.groupby(["player_id","beatmap_id"]).ngroups)

g = s1.groupby(["player_id", "beatmap_id"], sort=False)
# 1. 同一 cell 内相邻两次的间隔
d = g.ts.diff()
gap = d.dropna() / 3600.0
print()
print("=== 同一 (player, chart) 相邻两次 play 的时间间隔（小时）===")
print(gap.describe(percentiles=[.1,.25,.5,.75,.9]).round(2).to_string())
for thr, lab in [(10/60,"<10min"), (1,"<1h"), (24,"<1d"), (24*7,"<1w"), (24*30,"<1mo")]:
    print(f"   {lab:>7}: {(gap<thr).mean()*100:5.1f}%")
print()
# 2. 相邻两次 loss 的相关 vs 间隔（session 效应 / 自相关）
sub = s1.copy()
sub["prev_loss"] = g.loss.shift(1)
sub["gap_h"] = sub.ts.diff() / 3600.0
sub.loc[g.ngroup().diff() != 0, "gap_h"] = np.nan   # 只在同一 cell 内
sub = sub.dropna(subset=["prev_loss","gap_h"])
print("=== corr(loss_t, loss_{t-1}) 按间隔分箱  —— 若 play 独立，应恒等于 cell 间相关 ===")
bins = [0,10/60,1,6,24,24*7,24*30,1e9]
lab  = ["<10min","10min-1h","1-6h","6-24h","1-7d","7-30d",">30d"]
sub["b"] = pd.cut(sub.gap_h, bins=bins, labels=lab)
r = sub.groupby("b", observed=True).apply(
    lambda x: pd.Series({"n": len(x), "corr": x.loss.corr(x.prev_loss),
                         "sd_loss": x.loss.std()}), include_groups=False)
print(r.round(4).to_string())
print()
# 3. 把"第几次"考虑进来后，cell 内还剩多少结构
print("=== cell 内残差：只扣 cell 均值 vs 扣 cell 均值+ln(顺序) ===")
tot_within = 0.0; tot_within2 = 0.0; dof = 0
for (u,mm), dd in s1.groupby(["player_id","beatmap_id"], sort=False):
    if len(dd) < 3: continue
    y = dd.loss.to_numpy()
    x = np.log(dd.playcount_cur.to_numpy(float))
    r1 = y - y.mean()
    A = np.c_[np.ones_like(x), x - x.mean()]
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = y - A @ beta
    tot_within += (r1**2).sum(); tot_within2 += (r2**2).sum(); dof += len(y) - 1
print(f"  n>=3 cells, within SS: 只扣均值 = {tot_within/dof:.4f}   再扣 ln(顺序)斜率 = {tot_within2/dof:.4f}")
print(f"  => ln(顺序) 解释了 cell 内方差的 {(1-tot_within2/tot_within)*100:.1f}%")
print()
print("=== 关键：同一 cell 内 loss 的'真实'抖动 vs 我当作噪声的 sigma^2 ===")
print(f"  cell 内残差方差（扣均值后）      = {tot_within/dof:.4f}")
print(f"  cell 内残差方差（再扣练习顺序）  = {tot_within2/dof:.4f}")
print("  这两者都不是'独立 play 的 Bernoulli 噪声'——若真是 Bernoulli，按 note 数 n 算")
print("  方差应是 p(1-p)/n ~ 1e-5 量级，实测大 4 个数量级 => 过度离散，且自相关（见上表）")
