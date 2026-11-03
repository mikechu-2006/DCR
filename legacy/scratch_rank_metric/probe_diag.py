
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
N=len(A)
uk,inv,cnt=np.unique(A.astype(np.int64)*M+B,return_inverse=True,return_counts=True)
wins=np.bincount(inv,weights=Y,minlength=len(uk));UA=(uk//M).astype(np.int32);UB=(uk%M).astype(np.int32)
log("N",N,"U",len(uk))

def fit_bt(mask):
    # aggregate counts on the subset of instances
    u,c=np.unique(A[mask].astype(np.int64)*M+B[mask],return_inverse=True,return_counts=True)
    w=np.bincount(c if False else inv[mask],weights=Y[mask],minlength=len(u))
    # careful: inv is for full uk; recompute
    u,c=np.unique(A[mask].astype(np.int64)*M+B[mask],return_inverse=True,return_counts=True)
    w=np.bincount(inv2:=inv[mask]*0 if False else np.zeros(len(A[mask]),dtype=np.int64),minlength=1) if False else None
    # simpler: recompute wins per u
    key=A[mask].astype(np.int64)*M+B[mask]
    order=np.argsort(key,kind="stable"); key_s=key[order]; y_s=Y[mask][order]
    u,c=np.unique(key_s,return_counts=True)
    w=np.bincount(np.searchsorted(u,key_s),weights=y_s,minlength=len(u))
    ua=(u//M).astype(np.int32);ub=(u%M).astype(np.int32)
    nab=w;nba=c-w
    def f(x):
        b=x-x.mean();uu=b[ub]-b[ua];s=expit(uu)
        nll=-np.sum(nab*np.log(np.clip(s,1e-15,1))+nba*np.log(np.clip(1-s,1e-15,1)))
        du=-nab*(1-s)+nba*s
        gr=np.bincount(ua,weights=-du,minlength=M)+np.bincount(ub,weights=du,minlength=M)
        return nll,gr-gr.mean()
    r=minimize(f,np.zeros(M),jac=True,method="L-BFGS-B",options=dict(maxiter=300,gtol=1e-6,ftol=1e-13))
    b=r.x-r.x.mean()
    return b,nll_per_instance(r.fun,key,cnt_src=None),r
def nll_per_instance(nll,key,cnt_src=None): return nll/len(key)

half=PI%2
for h in [0,1]:
    m=half==h
    b,_,r=fit_bt(m)
    np.save(f"scratch_rank_metric/bhat_half{h}.npy",b)
    log("half",h,"n_inst",int(m.sum()),"in-sample nll",round(float(r.fun/m.sum()),5),"ok",bool(r.success),"nit",r.nit)

b0=np.load("scratch_rank_metric/bhat_half0.npy");b1=np.load("scratch_rank_metric/bhat_half1.npy")
n0=np.bincount(PI[half==0],minlength=P);n1=np.bincount(PI[half==1],minlength=P)
c0=np.bincount(UA,weights=None,minlength=M) if False else None
# map coverage per half: count instances
cov0=np.bincount(np.concatenate([A[half==0],B[half==0]]),minlength=M)
cov1=np.bincount(np.concatenate([A[half==1],B[half==1]]),minlength=M)
sel=(cov0>=20)&(cov1>=20)
log("maps with >=20 in both halves",int(sel.sum()),"split-half rho",round(float(spearmanr(b0[sel],b1[sel]).statistic),4))

# cross-fitted NLL on each half
for h in [0,1]:
    m=half==h; b_other=b1 if h==0 else b0
    u=b_other[B[m]]-b_other[A[m]]
    nll=-np.mean(Y[m]*np.log(np.clip(expit(u),1e-12,1))+ (1-Y[m])*np.log(np.clip(1-expit(u),1e-12,1)))
    log("half",h,"cross-fitted nll",round(float(nll),5))

# per-player spearman with cross-fitted bhat
def sp(x,y):
    rx=rankdata(x);ry=rankdata(y);rx=rx-rx.mean();ry=ry-ry.mean()
    den=np.sqrt((rx*rx).sum()*(ry*ry).sum());return float((rx*ry).sum()/den) if den>0 else np.nan
for h in [0,1]:
    b_other=b1 if h==0 else b0
    ids=[pi for pi in groups if pi%2==h]
    rs=[];ls=[]
    for pi in ids:
        idx=groups[pi]
        if len(idx)<3: continue
        rr=sp(-b_other[mir[idx]],perf[idx])
        if np.isfinite(rr): rs.append(rr);ls.append(len(idx))
    rs=np.array(rs);ls=np.array(ls)
    log("half",h,"cross-fitted per-player spearman lw",round(float((rs*ls).sum()/ls.sum()),4),"macro",round(float(rs.mean()),4),"n",len(rs))

# candidate captured fraction vs cross-fitted ceiling
cf=[]
for h in [0,1]:
    m=half==h;b_other=b1 if h==0 else b0
    u=b_other[B[m]]-b_other[A[m]]
    cf.append(-np.mean(Y[m]*np.log(np.clip(expit(u),1e-12,1))+(1-Y[m])*np.log(np.clip(1-expit(u),1e-12,1))))
cf=float(np.mean(cf))
log("cross-fitted ceiling nll",round(cf,5),"bits saved",round(1-cf/np.log(2),4))
for name,v in [("star",star),("msd",msdv)]:
    z=v[A]-v[B]
    # fit beta on full data quickly using the constrained nll from probe_bt is already known
    log(name,"captured vs cross-fit",round((np.log(2)-({"star":0.66085,"msd":0.64017}[name]))/(np.log(2)-cf),4))
json.dump(dict(cross_fit_ceiling=cf,split_half_rho=float(spearmanr(b0[sel],b1[sel]).statistic),
               n_maps_both=int(sel.sum())),open("scratch_rank_metric/probe_diag.json","w"),indent=1)
log("done")
