
import numpy as np, pandas as pd, json, time, sys
from scipy.stats import rankdata
from scipy.optimize import minimize_scalar, minimize
from scipy.special import expit

t0=time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]",*a); sys.stdout.flush()

df = pd.read_parquet("data/processed/step1_1k_pairbest.parquet")
meta = pd.read_csv("data/processed/beatmap_meta_1k.csv")[["beatmap_id","star","count_total","bpm"]]
msd = pd.read_parquet("data/processed/minacalc_msd_1k.parquet")[["beatmap_id","msd_overall"]]
df = df.merge(meta,on="beatmap_id",how="left").merge(msd,on="beatmap_id",how="left")
df["perf"] = -df["loss"].astype(np.float64)
d = df[df.star.notna() & df.msd_overall.notna()].reset_index(drop=True)
maps=np.sort(d.beatmap_id.unique()); M=len(maps)
mid2idx={m:i for i,m in enumerate(maps)}; d["mi"]=d.beatmap_id.map(mid2idx).astype(np.int32)
players=np.sort(d.player_id.unique()); P=len(players)
pid2idx={p:i for i,p in enumerate(players)}; d["pi"]=d.player_id.map(pid2idx).astype(np.int32)
log("rows",len(d),"M",M,"P",P)

g=d.groupby("mi")
cand={"star":g.star.first().reindex(range(M)).values.astype(float),
      "msd":g.msd_overall.first().reindex(range(M)).values.astype(float),
      "notes":g.count_total.first().reindex(range(M)).values.astype(float),
      "bpm":g.bpm.first().reindex(range(M)).values.astype(float)}
rng=np.random.default_rng(0); cand["random"]=rng.normal(size=M)
perf=d.perf.values; mi_row=d.mi.values
groups={k:np.asarray(v) for k,v in d.groupby("pi").indices.items()}

def sp(x,y):
    rx=rankdata(x);ry=rankdata(y);rx=rx-rx.mean();ry=ry-ry.mean()
    den=np.sqrt((rx*rx).sum()*(ry*ry).sum())
    return float((rx*ry).sum()/den) if den>0 else np.nan

log("=== global row-level spearman(d, loss) — confounded by player skill ===")
for k,v in cand.items(): log(f"  {k:7s} {sp(v[mi_row],-perf):+.4f}")

# per-player spearman with pids tracked
log("=== per-player spearman(-d, perf) ===")
spear={}
for k,v in cand.items():
    drow=v[mi_row]; rhos=[];ls=[];pids=[]
    for pi,idx in groups.items():
        l=len(idx)
        if l<3: continue
        r=sp(-drow[idx],perf[idx])
        if np.isfinite(r): rhos.append(r);ls.append(l);pids.append(pi)
    rhos=np.array(rhos);ls=np.array(ls);pids=np.array(pids)
    z=np.arctanh(np.clip(rhos,-.999999,.999999))
    spear[k]=dict(pids=pids,values=rhos,ls=ls,
        l_weighted=float((rhos*ls).sum()/ls.sum()),
        l2_weighted=float((rhos*ls*ls).sum()/(ls*ls).sum()),
        macro=float(rhos.mean()),
        fisher_l3=float(np.tanh((z*(ls-3)).sum()/(ls-3).sum())))
    log(f"  {k:7s} lw={spear[k]['l_weighted']:+.4f} l2w={spear[k]['l2_weighted']:+.4f} "
        f"macro={spear[k]['macro']:+.4f} fisher={spear[k]['fisher_l3']:+.4f} n={len(rhos)}")

# pair instances
a_l=[];b_l=[];y_l=[];p_l=[]
for pi,idx in groups.items():
    l=len(idx)
    if l<2: continue
    ii,jj=np.triu_indices(l,1)
    a=mi_row[idx][ii];b=mi_row[idx][jj];sw=a>b
    aa=np.where(sw,b,a);bb=np.where(sw,a,b)
    y=(perf[idx][ii]>perf[idx][jj])^sw
    a_l.append(aa.astype(np.int32));b_l.append(bb.astype(np.int32))
    y_l.append(y.astype(np.int8));p_l.append(np.full(len(aa),pi,dtype=np.int32))
A=np.concatenate(a_l);B=np.concatenate(b_l);Y=np.concatenate(y_l).astype(np.float64);PI=np.concatenate(p_l)
N=len(A); log("pair instances",N)
uk,inv,cnt=np.unique(A.astype(np.int64)*M+B,return_inverse=True,return_counts=True)
wins=np.bincount(inv,weights=Y,minlength=len(uk)).astype(np.float64)
UA=(uk//M).astype(np.int32);UB=(uk%M).astype(np.int32)
log("unique pairs",len(uk),"singleton frac",round(float((cnt==1).mean()),3),
    "unanimous frac",round(float(((wins==0)|(wins==cnt)).mean()),3),
    "count q50/q90/q99/max",np.percentile(cnt,[50,90,99,100]).astype(int).tolist())

def nll_inst(z,beta):
    p=np.clip(expit(-beta*z),1e-12,1-1e-12)
    return -np.mean(Y*np.log(p)+(1-Y)*np.log(1-p))
def fit_beta(z):
    grid=np.concatenate([[0.0],np.geomspace(1e-4,1e3,120)])
    vals=[nll_inst(z,b) for b in grid]
    k=int(np.argmin(vals));lo=grid[max(0,k-1)];hi=grid[min(len(grid)-1,k+1)]
    r=minimize_scalar(lambda b:nll_inst(z,b),bounds=(lo,hi),method="bounded")
    return float(r.x),float(r.fun)

log("=== pairwise NLL (micro, beta fitted per system) ===")
pw={}
for k,v in cand.items():
    z=v[A]-v[B];beta,nll=fit_beta(z)
    p=np.clip(expit(-beta*z),1e-12,1-1e-12)
    li=-(Y*np.log(p)+(1-Y)*np.log(1-p))
    per_sum=np.bincount(PI,weights=li,minlength=P);per_n=np.bincount(PI,minlength=P)
    mask=per_n>=3
    macro=float(np.mean(per_sum[mask]/per_n[mask]))
    # unique-pair-equal nll
    zu=v[UA]-v[UB];pu=np.clip(expit(-beta*zu),1e-12,1-1e-12)
    ph=wins/cnt
    nll_u=float(-np.mean(ph*np.log(pu)+(1-ph)*np.log(1-pu)))
    pw[k]=dict(beta=beta,nll=nll,nll_unique=nll_u,macro_nll=macro,
               per_sum=per_sum,per_n=per_n,
               bits=float(1-nll/np.log(2)))
    log(f"  {k:7s} beta={beta:7.3f} nll={nll:.5f} nll_u={nll_u:.5f} macro={macro:.5f} bits={pw[k]['bits']:+.4f}")

# BT free fit
log("=== BT free fit ===")
nab=wins; nba=cnt-wins
def f(x):
    b=x-x.mean()
    u=b[UB]-b[UA];s=expit(u)
    nll=-np.sum(nab*np.log(np.clip(s,1e-15,1))+nba*np.log(np.clip(1-s,1e-15,1)))
    du=-nab*(1-s)+nba*s
    grad=np.bincount(UA,weights=-du,minlength=M)+np.bincount(UB,weights=du,minlength=M)
    return nll,grad-grad.mean()
r=minimize(f,np.zeros(M),jac=True,method="L-BFGS-B",options=dict(maxiter=300,gtol=1e-7,ftol=1e-13))
bhat=r.x-r.x.mean(); nll_free=r.fun
log("free nll",round(nll_free/N,5),"bits",round(float(1-(nll_free/N)/np.log(2)),4),"nit",r.nit,"ok",r.success)

log("=== constrained BT b=c+beta*d ===")
def con(params,dv):
    b=params[0]+params[1]*dv;u=b[UB]-b[UA];s=expit(u)
    return -np.sum(nab*np.log(np.clip(s,1e-15,1))+nba*np.log(np.clip(1-s,1e-15,1)))
con_res={}
for k,v in cand.items():
    vv=(v-v.mean())/(v.std()+1e-12)
    rr=minimize(lambda q:con(q,vv),[0.0,0.1],method="Nelder-Mead",
                options=dict(xatol=1e-10,fatol=1e-8,maxiter=4000))
    nllc=rr.fun
    con_res[k]=dict(beta=float(rr.x[1]),nll=float(nllc),expl_dev=float(1-nllc/nll_free),
                    rho_bhat=sp(v,bhat),
                    tau_bhat=float(pd.Series(v).corr(pd.Series(bhat),method="kendall")),
                    lr=float(2*(nllc-nll_free)))
    log(f"  {k:7s} beta={rr.x[1]:+.4f} nll={nllc:.0f} expl_dev={con_res[k]['expl_dev']:+.4f} "
        f"rho_bhat={con_res[k]['rho_bhat']:+.4f} tau={con_res[k]['tau_bhat']:+.4f}")

# ceiling per-player spearman using bhat
dceil=bhat[mi_row];rhos=[];ls=[];pids=[]
for pi,idx in groups.items():
    l=len(idx)
    if l<3: continue
    rr=sp(-dceil[idx],perf[idx])
    if np.isfinite(rr): rhos.append(rr);ls.append(l);pids.append(pi)
rhos=np.array(rhos);ls=np.array(ls)
log("CEILING per-player spearman(free bhat): lw=%.4f macro=%.4f"%(float((rhos*ls).sum()/ls.sum()),float(rhos.mean())))
con_res["__ceiling__"]=dict(l_weighted=float((rhos*ls).sum()/ls.sum()),macro=float(rhos.mean()))
# normalized spearman vs ceiling
for k in cand:
    base=spear["random"]["l_weighted"]; top=con_res["__ceiling__"]["l_weighted"]
    con_res[k]["norm_spear"]=float((spear[k]["l_weighted"]-base)/(top-base))
    con_res[k]["norm_spear_macro"]=float((spear[k]["macro"]-spear["random"]["macro"])/(con_res["__ceiling__"]["macro"]-spear["random"]["macro"]))
    log(f"  norm_spear {k:7s} {con_res[k]['norm_spear']:+.3f} macro {con_res[k]['norm_spear_macro']:+.3f}")

# n_plays confound
log("=== n_plays confound ===")
rr=[];ll=[]
for pi,idx in groups.items():
    l=len(idx)
    if l<5: continue
    rr.append(sp(np.log(d.n_plays.values[idx].astype(float)),perf[idx]));ll.append(l)
rr=np.array(rr);ll=np.array(ll)
log("mean within-player rho(log n_plays, perf)",round(float((rr*ll).sum()/ll.sum()),4))
log("corr rows log n_plays vs star",round(float(np.corrcoef(np.log(d.n_plays.values),cand["star"][mi_row])[0,1]),4))
log("corr rows perf vs log n_plays",round(float(np.corrcoef(perf,np.log(d.n_plays.values))[0,1]),4))
log("loss range",np.percentile(d.loss,[0,100]).round(3).tolist(),"acc range",
    (np.exp(np.percentile(d.loss,[0,100]))).round(4).tolist())

# bootstrap over players (micro NLL & lw spearman)
log("=== bootstrap CIs over players ===")
B=400;rng=np.random.default_rng(7);ci={}
for k in ["star","msd","notes","random"]:
    ps=pw[k]["per_sum"];pn=pw[k]["per_n"]
    sv=spear[k]["values"];sl=spear[k]["ls"];spids=spear[k]["pids"]
    vs=[];ss=[]
    for _ in range(B):
        s=rng.integers(0,len(spids),len(spids))
        vs.append(ps[spids[s]].sum()/pn[spids[s]].sum())
        ss.append(np.average(sv[s],weights=sl[s]))
    ci[k]=dict(nll=[float(np.percentile(vs,2.5)),float(np.percentile(vs,97.5))],
               lw_spear=[float(np.percentile(ss,2.5)),float(np.percentile(ss,97.5))])
    log(" ",k,{kk:[round(x,4) for x in vv] for kk,vv in ci[k].items()})

out=dict(spearman={k:{kk:(vv.tolist() if isinstance(vv,np.ndarray) and kk in("values","ls","pids") else vv)
                      for kk,vv in v.items()} for k,v in spear.items()},
         pairwise={k:{kk:(vv.tolist() if isinstance(vv,np.ndarray) else vv) for kk,vv in v.items()} for k,v in pw.items()},
         constrained=con_res,bt_free_bits=float(1-(nll_free/N)/np.log(2)),ci=ci,
         meta=dict(rows=len(d),M=M,P=P,N=N,unique_pairs=int(len(uk)),
                   singleton=float((cnt==1).mean()),unanimous=float(((wins==0)|(wins==cnt)).mean())))
json.dump(out,open("scratch_rank_metric/probe3.json","w"),indent=1,default=float)
np.save("scratch_rank_metric/bhat.npy",bhat)
log("done")
