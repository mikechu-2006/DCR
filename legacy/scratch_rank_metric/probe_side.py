
import numpy as np, pandas as pd
from scipy.stats import rankdata
df=pd.read_parquet("data/processed/step1_1k_pairbest.parquet")
meta=pd.read_csv("data/processed/beatmap_meta_1k.csv")[["beatmap_id","star"]]
msd=pd.read_parquet("data/processed/minacalc_msd_1k.parquet")[["beatmap_id","msd_overall"]]
df=df.merge(meta,on="beatmap_id",how="left").merge(msd,on="beatmap_id",how="left")
df["perf"]=-df.loss.astype(float)
g=df.groupby("player_id")
# variance decomposition of loss (and perf)
m=g.loss.transform("mean")
between=g.loss.mean().var(ddof=1); within=g.loss.var(ddof=1).mean()
print("loss var total",df.loss.var(ddof=1),"between",between,"within",within,
      "between share",round(between/(between+within),4))
m2=g.perf.transform("mean"); b2=g.perf.mean().var(ddof=1); w2=g.perf.var(ddof=1).mean()
print("perf between share",round(b2/(b2+w2),4))
# player-level: mean loss vs mean star (skill vs map choice)
pl=g.agg(mean_loss=("loss","mean"),mean_star=("star","mean"),n=("loss","size"),
         mean_lognp=("n_plays",lambda s: np.log(s).mean()))
print("corr player mean_loss vs mean_star",round(pl.mean_loss.corr(pl.mean_star),4))
print("corr player mean_loss vs log n",round(pl.mean_loss.corr(pl.mean_lognp),4))
# within-player rho(log n_plays, perf)
def sp(x,y):
    rx=rankdata(x);ry=rankdata(y);rx=rx-rx.mean();ry=ry-ry.mean()
    den=np.sqrt((rx*rx).sum()*(ry*ry).sum()); return float((rx*ry).sum()/den) if den>0 else np.nan
rs=[];ns=[]
for pid,gg in g:
    if len(gg)<5: continue
    rs.append(sp(np.log(gg.n_plays.values.astype(float)),gg.perf.values)); ns.append(len(gg))
rs=np.array(rs);ns=np.array(ns)
print("within-player rho(log n_plays, perf) lw",round(float((rs*ns).sum()/ns.sum()),4),"n",len(rs))
# residualized per-player spearman for star/msd
maps=df[["beatmap_id","star","msd_overall"]].drop_duplicates().set_index("beatmap_id")
for name in ["star","msd_overall"]:
    d=df.beatmap_id.map(maps[name]).values.astype(float)
    res_raw=[];res_res=[];ls=[]
    for pid,gg in g:
        if len(gg)<5: continue
        p=gg.perf.values; ln=np.log(gg.n_plays.values.astype(float)); dd=d[gg.index.values]
        # residualize perf on log n_plays (with intercept)
        X=np.c_[np.ones(len(ln)),ln]; beta=np.linalg.lstsq(X,p,rcond=None)[0]; r=p-X@beta
        res_raw.append(sp(-dd,p)); res_res.append(sp(-dd,r)); ls.append(len(gg))
    res_raw=np.array(res_raw);res_res=np.array(res_res);ls=np.array(ls)
    print(name,"lw raw",round(float((res_raw*ls).sum()/ls.sum()),4),
          "lw resid-nplays",round(float((res_res*ls).sum()/ls.sum()),4),
          "mean change",round(float(np.mean(res_res-res_raw)),4))
