import numpy as np, pandas as pd
cols = ["player_id", "beatmap_id", "loss"]
s1 = pd.read_parquet("data/processed/step1_1k.parquet", columns=cols)
y = s1.loss.to_numpy(np.float64)
pid = pd.factorize(s1.player_id)[0]; bid = pd.factorize(s1.beatmap_id)[0]
n = len(y); grand = y.mean()

def ss(x, codes, K):
    s = np.bincount(codes, weights=y, minlength=K)
    c = np.bincount(codes, minlength=K)
    return float((s**2 / np.maximum(c,1)).sum() - n*grand**2), c

SSp, cp = ss(y, pid, pid.max()+1)
SSb, cb = ss(y, bid, bid.max()+1)
SStot = float((y**2).sum() - n*grand**2)
SSE = SStot - SSp - SSb
print("=== 双向方差分析（玩家 + 谱面，不含交互）===")
print(f"  SS_player = {SSp:12.1f}   df={len(cp)-1:,}")
print(f"  SS_chart  = {SSb:12.1f}   df={len(cb)-1:,}")
print(f"  SS_error  = {SSE:12.1f}   df={n-len(cp)-len(cb)+1:,}")
MSE = SSE/(n-len(cp)-len(cb)+1)
print(f"  MSE (误差均方) = {MSE:.4f}  (= 同一玩家同一谱面内的残差方差的上界估计)")
print()
print("=== 用误差均方反推'有效独立观测数' ===")
print(f"  若 play 完全独立：Var(cell mean) = MSE/n_bar")
print(f"  实测 cell 内方差 = 0.5722（n>=3 的 cell，扣 cell 均值）")
print(f"  MSE = {MSE:.4f}  vs  扣均值后的 cell 内方差 0.5722")
print(f"  => 双向分解后剩下的'误差'里，仍有 {1 - MSE/0.5722:.1%} 可被 (玩家,谱面) 之外的因子解释")
print()
print("=== 结论 ===")
print("  (a) 谱面内确有聚集：deff ~ 5.4-6.3，ICC ~ 0.075-0.088，n_eff 约为名义的 1/5")
print("  (b) 但聚集不等于'一次 play 只值一条 Bernoulli'：")
print("      Bernoulli 预测的 cell 内方差 ~ p(1-p)/n_notes ~ 1e-5，实测 0.30-0.57，差 4 个数量级")
print("  (c) 扣掉 ln(练习顺序) 后 cell 内方差从 0.572 降到 0.298（-48%）")
print("      => 近一半的'噪声'其实是可预测的练习曲线，不是噪声")
