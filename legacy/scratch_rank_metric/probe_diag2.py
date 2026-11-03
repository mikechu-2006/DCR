
import numpy as np, pandas as pd, json, time, sys
from scipy.stats import rankdata, spearmanr
from scipy.optimize import minimize
from scipy.special import expit
t0=time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]",*a); sys.stdout.flush()

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
A=[];B=[];Y=[];PI=[]
for pi,idx in groups.items():
    l=len(idx)
    if l<2: continue
    ii,jj=np.triu_indices(l,1)
    a=mir[idx][ii];b=mir[idx][jj];sw=a>b
    A.append(np.where(sw,b,a).astype(np.int32));B.append(np.where(sw,a,b).astype(np.int32))
    Y.append(((perf[idx][ii]>perf[idx][jj])^sw).astype(np.int8));PI.append(np.full(len(ii),pi,dtype=np.int32))
A=np.concatenate(A);B=np.concatenate(B);Y=np.concatenate(Y).astype(float);PI=np.concatenate(PI)
N=len(A);log("N",N)

def agg(mask):
    key=A[mask].astype(np.int64)*M+B[mask]; y=Y[mask]
    o=np.argsort(key,kind="stable");ks=key[o];ys=y[o]
    u,st,c=np.unique(ks,return_index=True,return_counts=True)
    w=np.add.reduceat(ys,st)
    return (u//M).astype(np.int32),(u%M).astype(np.int32),w,c-w

def fit_bt(mask,maxiter=300):
    ua,ub,nab,nba=agg(mask)
    def f(x):
        b=x-x.mean();u=b[ub]-b[ua];s=expit(u)
        nll=-np.sum(nab*np.log(np.clip(s,1e-15,1))+nba*np.log(np.clip(1-s,1e-15,1)))
        du=-nab*(1-s)+nba*s
        gr=np.bincount(ua,weights=-du,minlength=M)+np.bincount(ub,weights=du,minlength=M)
        return nll,gr-gr.mean()
    r=minimize(f,np.zeros(M),jac=True,method="L-BFGS-B",options=dict(maxiter=maxiter,gtol=1e-6,ftol=1e-13))
    return r.x-r.x.mean(),r

log("=== monotone-ish spline link (8 hat functions at deciles) ===")
ua,ub,nab,nba=agg(np.ones(N,dtype=bool))
for name,v in [("star",star),("msd",msdv)]:
    qs=np.quantile(v,np.linspace(0,1,8));qs=np.unique(qs)
    h=(qs[-1]-qs[0])/(len(qs)-1)+1e-9
    X=np.stack([np.maximum(0,1-np.abs(v-q)/h) for q in qs],axis=1)
    X=X-X.mean(0)
    def f(beta,lam=1e-4):
        b=X@beta;u=b[ub]-b[ua];s=expit(u)
        nll=-np.sum(nab*np.log(np.clip(s,1e-15,1))+nba*np.log(np.clip(1-s,1e-15,1)))+lam*np.sum(beta*beta)*N/1e6
        du=-nab*(1-s)+nba*s
        gr=X.T@(np.bincount(ua,weights=-du,minlength=M)+np.bincount(ub,weights=du,minlength=M))
        return nll,gr
    r=minimize(f,np.zeros(X.shape[1]),jac=True,method="L-BFGS-B",options=dict(maxiter=200,gtol=1e-6))
    log(name,"spline nll per inst",round(float(r.fun/N),5),"vs linear",round({"star":0.66085,"msd":0.64017}[name],5),"K",X.shape[1])

log("=== split-half reliability ===")
half=(PI%2)==0
b0,r0=fit_bt(half);b1,r1=fit_bt(~half)
np.save("scratch_rank_metric/bhat_half0.npy",b0);np.save("scratch_rank_metric/bhat_half1.npy",b1)
log("half0 nll",round(float(r0.fun/half.sum()),5),"half1 nll",round(float(r1.fun/(~half).sum()),5))
cov0=np.bincount(np.concatenate([A[half],B[half]]),minlength=M)
cov1=np.bincount(np.concatenate([A[~half],B[~half]]),minlength=M)
sel=(cov0>=20)&(cov1>=20)
log("maps>=20 both",int(sel.sum()),"split-half rho",round(float(spearmanr(b0[sel],b1[sel]).statistic),4),
    "tau",round(float(pd.Series(b0[sel]).corr(pd.Series(b1[sel]),method="kendall")),4))
cf=[]
for h,bo in [(half,b1),(~half,b0)]:
    u=bo[B[h]]-bo[A[h]];p=np.clip(expit(u),1e-12,1-1e-12)
    cf.append(-np.mean(Y[h]*np.log(p)+(1-Y[h])*np.log(1-p)))
cf=float(np.mean(cf));log("cross-fitted ceiling nll",round(cf,5),"bits",round(1-cf/np.log(2),4))
for name,nll in [("star",0.66085),("msd",0.64017)]:
    log(name,"captured(crossfit)",round((np.log(2)-nll)/(np.log(2)-cf),4))
def sp(x,y):
    rx=rankdata(x);ry=rankdata(y);rx=rx-rx.mean();ry=ry-ry.mean()
    den=np.sqrt((rx*rx).sum()*(ry*ry).sum());return float((rx*ry).sum()/den) if den>0 else np.nan
for h,bo in [(half,b1),(~half,b0)]:
    rs=[];ls=[]
    for pi in [p for p in groups if (p%2==0)==h]:
        idx=groups[pi]
        if len(idx)<3: continue
        rr=sp(-bo[mir[idx]],perf[idx])
        if np.isfinite(rr): rs.append(rr);ls.append(len(idx))
    rs=np.array(rs);ls=np.array(ls)
    log("crossfit per-player spearman",round(float((rs*ls).sum()/ls.sum()),4),round(float(rs.mean()),4),"n",len(rs))
json.dump(dict(crossfit_nll=cf,split_half_rho=float(spearmanr(b0[sel],b1[sel]).statistic)),open("scratch_rank_metric/probe_diag2.json","w"),indent=1)
log("done")
