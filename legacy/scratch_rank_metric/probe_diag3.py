
import numpy as np, pandas as pd
from scipy.stats import rankdata
df=pd.read_parquet("data/processed/step1_1k_pairbest.parquet")
meta=pd.read_csv("data/processed/beatmap_meta_1k.csv")[["beatmap_id","star"]]
msd=pd.read_parquet("data/processed/minacalc_msd_1k.parquet")[["beatmap_id","msd_overall"]]
df=df.merge(meta,on="beatmap_id",how="left").merge(msd,on="beatmap_id",how="left")
df["perf"]=-df.loss.astype(float)
d=df[df.star.notna()&df.msd_overall.notna()].reset_index(drop=True)
maps=np.sort(d.beatmap_id.unique());M=len(maps);mid={m:i for i,m in enumerate(maps)}
d["mi"]=d.beatmap_id.map(mid).astype(np.int32)
players=np.sort(d.player_id.unique());P=len(players);pid={p:i for i,p in enumerate(players)}
d["pi"]=d.player_id.map(pid).astype(np.int32)
g=d.groupby("mi")
star=g.star.first().reindex(range(M)).values.astype(float)
msdv=g.msd_overall.first().reindex(range(M)).values.astype(float)
perf=d.perf.values;mir=d.mi.values
groups={k:np.asarray(v) for k,v in d.groupby("pi").indices.items()}
b0=np.load("scratch_rank_metric/bhat_half0.npy");b1=np.load("scratch_rank_metric/bhat_half1.npy")
bhat=np.load("scratch_rank_metric/bhat.npy")
def sp(x,y):
    rx=rankdata(x);ry=rankdata(y);rx=rx-rx.mean();ry=ry-ry.mean()
    den=np.sqrt((rx*rx).sum()*(ry*ry).sum());return float((rx*ry).sum()/den) if den>0 else np.nan
def perplayer(sig,mask_pi=None):
    rs=[];ls=[]
    for pi,idx in groups.items():
        if mask_pi is not None and pi%2!=mask_pi: continue
        if len(idx)<3: continue
        rr=sp(-sig[mir[idx]],perf[idx])
        if np.isfinite(rr): rs.append(rr);ls.append(len(idx))
    rs=np.array(rs);ls=np.array(ls)
    return float((rs*ls).sum()/ls.sum()),float(rs.mean()),len(rs)
# cross-fitted per-player spearman
cf_sig=np.where(np.arange(M)%1==0,0,0).astype(float)  # placeholder
cf=bhat.copy()
rs_even=b1;rs_odd=b0
cf=np.where(np.arange(M)>=0,0.0,0.0)
cf=np.empty(M)
cf[:]=np.nan
# map-level: use b1 for even players, b0 for odd; build a row-level signal
row_sig=np.empty(len(d))
for pi,idx in groups.items():
    row_sig[idx]= (b1 if pi%2==0 else b0)[mir[idx]]
rs=[];ls=[]
for pi,idx in groups.items():
    if len(idx)<3: continue
    rr=sp(-row_sig[idx],perf[idx])
    if np.isfinite(rr): rs.append(rr);ls.append(len(idx))
rs=np.array(rs);ls=np.array(ls)
ceil_lw=float((rs*ls).sum()/ls.sum());ceil_macro=float(rs.mean())
print("cross-fitted ceiling per-player spearman lw %.4f macro %.4f n %d"%(ceil_lw,ceil_macro,len(rs)))
print("in-sample ceiling 0.6802 / 0.6249")
for name,v in [("star",star),("msd",msdv),("random",np.random.default_rng(0).normal(size=M))]:
    lw,mac,n=perplayer(v)
    print(name,"lw %.4f macro %.4f | norm_lw %.3f norm_macro %.3f"%(
        lw,mac,(lw+0.0347)/(ceil_lw+0.0347),(mac+0.0274)/(ceil_macro+0.0274)))
# disattenuated corr with full bhat, reliability full via Spearman-Brown from 0.5823
rel_full=2*0.5823/(1+0.5823)
for name,v in [("star",star),("msd",msdv)]:
    r=pd.Series(v).corr(pd.Series(bhat),method="spearman")
    print(name,"rho_bhat %.4f disatten %.4f (rel_full %.3f)"%(r,r/np.sqrt(rel_full),rel_full))
