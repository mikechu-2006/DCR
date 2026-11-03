
import numpy as np, pandas as pd, json, time, sys
from scipy.stats import rankdata
from scipy.optimize import minimize
from scipy.special import expit
t0=time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]",*a); sys.stdout.flush()

df=pd.read_parquet("data/processed/step1_1k_pairbest.parquet")
meta=pd.read_csv("data/processed/beatmap_meta_1k.csv")[["beatmap_id","star","count_total","bpm"]]
msd=pd.read_parquet("data/processed/minacalc_msd_1k.parquet")[["beatmap_id","msd_overall"]]
df=df.merge(meta,on="beatmap_id",how="left").merge(msd,on="beatmap_id",how="left")
df["perf"]=-df.loss.astype(float)
d=df[df.star.notna()&df.msd_overall.notna()].reset_index(drop=True)
maps=np.sort(d.beatmap_id.unique());M=len(maps);mid={m:i for i,m in enumerate(maps)}
d["mi"]=d.beatmap_id.map(mid).astype(np.int32)
players=np.sort(d.player_id.unique());P=len(players);pid={p:i for i,p in enumerate(players)}
d["pi"]=d.player_id.map(pid).astype(np.int32)
g=d.groupby("mi")
cand={"star":g.star.first().reindex(range(M)).values.astype(float),
      "msd":g.msd_overall.first().reindex(range(M)).values.astype(float),
      "notes":g.count_total.first().reindex(range(M)).values.astype(float),
      "bpm":g.bpm.first().reindex(range(M)).values.astype(float)}
cand["random"]=np.random.default_rng(0).normal(size=M)
perf=d.perf.values;mir=d.mi.values
groups={k:np.asarray(v) for k,v in d.groupby("pi").indices.items()}

A=[];B=[];Y=[];PI=[]
for pi,idx in groups.items():
    l=len(idx)
    if l<2: continue
    ii,jj=np.triu_indices(l,1)
    a=mir[idx][ii];b=mir[idx][jj];sw=a>b
    A.append(np.where(sw,b,a).astype(np.int32));B.append(np.where(sw,a,b).astype(np.int32))
    Y.append(((perf[idx][ii]>perf[idx][jj])^sw).astype(np.int8))
    PI.append(np.full(len(ii),pi,dtype=np.int32))
A=np.concatenate(A);B=np.concatenate(B);Y=np.concatenate(Y).astype(float);PI=np.concatenate(PI)
N=len(A)
uk,inv,cnt=np.unique(A.astype(np.int64)*M+B,return_inverse=True,return_counts=True)
wins=np.bincount(inv,weights=Y,minlength=len(uk));nab=wins;nba=cnt-wins
UA=(uk//M).astype(np.int32);UB=(uk%M).astype(np.int32)
log("N",N,"U",len(uk))

def f(x):
    b=x-x.mean();u=b[UB]-b[UA];s=expit(u)
    nll=-np.sum(nab*np.log(np.clip(s,1e-15,1))+nba*np.log(np.clip(1-s,1e-15,1)))
    du=-nab*(1-s)+nba*s
    gr=np.bincount(UA,weights=-du,minlength=M)+np.bincount(UB,weights=du,minlength=M)
    return nll,gr-gr.mean()
r=minimize(f,np.zeros(M),jac=True,method="L-BFGS-B",options=dict(maxiter=400,gtol=1e-7,ftol=1e-14))
bhat=r.x-r.x.mean();nll_free=r.fun
log("BT free per-inst",round(nll_free/N,5),"bits",round(float(1-(nll_free/N)/np.log(2)),4),"nit",r.nit,"ok",bool(r.success))

def con(q,dv):
    b=q[0]+q[1]*dv;u=b[UB]-b[UA];s=expit(u)
    return -np.sum(nab*np.log(np.clip(s,1e-15,1))+nba*np.log(np.clip(1-s,1e-15,1)))
res={}
for k,v in cand.items():
    vv=(v-v.mean())/(v.std()+1e-12)
    rr=minimize(lambda q:con(q,vv),[0.0,0.1],method="Nelder-Mead",options=dict(xatol=1e-10,fatol=1e-8,maxiter=5000))
    nllc=rr.fun
    res[k]=dict(beta=float(rr.x[1]),nll=float(nllc/N),expl_dev=float(1-nllc/nll_free),
                rho_bhat=float(pd.Series(v).corr(pd.Series(bhat),method="spearman")),
                tau_bhat=float(pd.Series(v).corr(pd.Series(bhat),method="kendall")))
    log(k,{kk:round(vv,4) for kk,vv in res[k].items()})

def sp(x,y):
    rx=rankdata(x);ry=rankdata(y);rx=rx-rx.mean();ry=ry-ry.mean()
    den=np.sqrt((rx*rx).sum()*(ry*ry).sum());return float((rx*ry).sum()/den) if den>0 else np.nan
dc=bhat[mir]
for name,sig in [("ceiling_bhat",dc)]+[(k,cand[k][mir]) for k in cand]:
    rs=[];ls=[];ps=[]
    for pi,idx in groups.items():
        l=len(idx)
        if l<3: continue
        rr=sp(-sig[idx],perf[idx])
        if np.isfinite(rr): rs.append(rr);ls.append(l);ps.append(pi)
    rs=np.array(rs);ls=np.array(ls)
    res.setdefault("__perplayer__",{})[name]=dict(lw=float((rs*ls).sum()/ls.sum()),macro=float(rs.mean()),n=len(rs))
    log("perplayer",name,round(float((rs*ls).sum()/ls.sum()),4),round(float(rs.mean()),4))

# ceiling-normalized metrics
base=res["__perplayer__"]["random"]["lw"];top=res["__perplayer__"]["ceiling_bhat"]["lw"]
for k in cand:
    res[k]["norm_spear"]=float((res["__perplayer__"][k]["lw"]-base)/(top-base))
    res[k]["nll_bits"]=float(1-res[k]["nll"]/np.log(2))
log("ceiling",top,"random",base)
res["meta"]=dict(N=N,U=int(len(uk)),M=M,P=P,bt_free_nll_per_inst=float(nll_free/N),
                 bt_free_bits=float(1-(nll_free/N)/np.log(2)))
json.dump(res,open("scratch_rank_metric/probe_bt.json","w"),indent=1,default=float)
np.save("scratch_rank_metric/bhat.npy",bhat)
log("saved")
