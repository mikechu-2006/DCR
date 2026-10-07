import numpy as np, pandas as pd
cols = ["player_id", "beatmap_id", "loss"]
s1 = pd.read_parquet("data/processed/step1_1k.parquet", columns=cols)
s0 = pd.read_csv("data/processed/step0_1k.csv", usecols=cols)
for nm, df in (("step1(train)", s1), ("step0(all)", s0)):
    g = df.groupby("beatmap_id").loss
    n = g.size(); m = g.mean(); v = g.var(ddof=1)
    multi = n >= 2
    # 谱面均值的观测方差
    obs_var_of_mean = m[multi].var(ddof=1)
    # 同谱面内的合并方差（= 若 play 独立时的 sigma^2）
    sig2 = float(((v[multi] * (n[multi]-1)).sum()) / ((n[multi]-1).sum()))
    # 独立假设下，谱面均值的期望方差 = E[sigma^2/n]
    exp_var_if_indep = float((sig2 / n[multi]).mean())
    # 设计效应 / 组内相关
    deff = obs_var_of_mean / exp_var_if_indep
    nbar = float(n[multi].mean())
    rho = (deff - 1) / (nbar - 1) if nbar > 1 else np.nan
    print(f"=== {nm} ===")
    print(f"  谱面数 {len(n):,}；每谱面 play 均值 {n.mean():.2f} / 中位 {n.median():.0f}")
    print(f"  谱面均值本身的观测方差        = {obs_var_of_mean:.4f}")
    print(f"  若 play 独立，应为 E[sigma^2/n] = {exp_var_if_indep:.4f}")
    print(f"  => 设计效应 deff              = {deff:.1f}")
    print(f"  => 组内相关 rho (由 deff 反推) = {rho:.4f}")
    print(f"  => 每张谱面的有效样本量 n_eff   = {nbar/deff:.2f}  (名义 {nbar:.1f})")
    print()
print("=== 这对'信息量'意味着什么 ===")
nbar = 34.35; deff = None
# 用 step1 的数字
g = s1.groupby("beatmap_id").loss; n = g.size(); v = g.var(ddof=1); m = g.mean()
multi = n >= 2
sig2 = float(((v[multi]*(n[multi]-1)).sum())/((n[multi]-1).sum()))
obs = m[multi].var(ddof=1); expv = float((sig2/n[multi]).mean()); deff = obs/expv
print(f"  deff = {deff:.1f} 意味着：674,436 条训练 play 关于'谱面'的信息量")
print(f"  只相当于 674,436 / {deff:.1f} = {674436/deff:,.0f} 条独立观测")
print(f"  而每张谱面要估 K=4 维向量 => 有效 plays/维 = {(674436/deff)/19636/4:.3f}")
