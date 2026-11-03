
import numpy as np, pandas as pd, json, time
from scipy.stats import rankdata
from scipy.optimize import minimize_scalar, minimize
from scipy.special import expit, logit

t0=time.time()
df = pd.read_parquet("data/processed/step1_1k_pairbest.parquet")
meta = pd.read_csv("data/processed/beatmap_meta_1k.csv")[["beatmap_id","star","count_total","bpm"]]
msd = pd.read_parquet("data/processed/minacalc_msd_1k.parquet")[["beatmap_id","msd_overall"]]
df = df.merge(meta,on="beatmap_id",how="left").merge(msd,on="beatmap_id",how="left")
df["perf"] = -df["loss"].astype(np.float64)
ok = df.star.notna() & df.msd_overall.notna()
d = df[ok].reset_index(drop=True)
maps = np.sort(d.beatmap_id.unique()); M=len(maps)
mid2idx = {m:i for i,m in enumerate(maps)}
d["mi"] = d.beatmap_id.map(mid2idx).astype(np.int32)
players = np.sort(d.player_id.unique()); P=len(players)
pid2idx={p:i for i,p in enumerate(players)}
d["pi"]=d.player_id.map(pid2idx).astype(np.int32)

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

# global (row-level) spearman: the project's existing-style number, confounded by player skill
print("=== global row-level spearman(d, loss) [positive = harder=>higher loss] ===")
for k,v in cand.items():
    print(f"  {k:7s} {sp(v[mi_row], -perf):+.4f}")

# ---------- per-player spearman
print("=== per-player spearman (-d vs perf) ===")
spear={}
for k,v in cand.items():
    drow=v[mi_row]; rhos=[];ls=[]
    for pi,idx in groups.items():
        l=len(idx)
        if l<3: continue
        r=sp(-drow[idx],perf[idx])
        if np.isfinite(r): rhos.append(r);ls.append(l)
    rhos=np.array(rhos);ls=np.array(ls);z=np.arctanh(np.clip(rhos,-.999999,.999999))
    spear[k]=dict(l_weighted=float((rhos*ls).sum()/ls.sum()),
                  l2_weighted=float((rhos*ls*ls).sum()/(ls*ls).sum()),
                  macro=float(rhos.mean()),
                  fisher_l3=float(np.tanh(((z*(ls-3)).sum())/(ls-3).sum())),
                  values=rhos, ls=ls)
    print(f"  {k:7s} lw={spear[k]['l_weighted']:+.4f} l2w={spear[k]['l2_weighted']:+.4f} "
          f"macro={spear[k]['macro']:+.4f} fisher={spear[k]['fisher_l3']:+.4f}")

# ---------- pair instances
a_l=[];b_l=[];y_l=[];p_l=[]
for pi,idx in groups.items():
    l=len(idx)
    if l<2: continue
    ii,jj=np.triu_indices(l,1)
    a=mi_row[idx][ii];b=mi_row[idx][jj]
    sw=a>b
    aa=np.where(sw,b,a);bb=np.where(sw,a,b)
    y=(perf[idx][ii]>perf[idx][jj])^sw
    a_l.append(aa.astype(np.int32));b_l.append(bb.astype(np.int32));y_l.append(y.astype(np.int8))
    p_l.append(np.full(len(aa),pi,dtype=np.int32))
A=np.concatenate(a_l);B=np.concatenate(b_l);Y=np.concatenate(y_l).astype(np.float64);PI=np.concatenate(p_l)
print("instances",len(A))
uk,inv,cnt=np.unique(A.astype(np.int64)*M+B,return_inverse=True,return_counts=True)
wins=np.bincount(inv,weights=Y,minlength=len(uk))
UA=(uk//M).astype(np.int32);UB=(uk%M).astype(np.int32)
print("unique pairs",len(uk),"frac singleton",float((cnt==1).mean()),
      "frac unanimous",float(((wins==0)|(wins==cnt)).mean()))

def nll_inst(z,Y,beta):
    p=np.clip(expit(-beta*z),1e-12,1-1e-12)
    return -np.mean(Y*np.log(p)+(1-Y)*np.log(1-p))

def fit_beta(z,Y):
    grid=np.concatenate([[0.0],np.geomspace(1e-4,1e3,120)])
    vals=[nll_inst(z,Y,b) for b in grid]
    k=int(np.argmin(vals));lo=grid[max(0,k-1)];hi=grid[min(len(grid)-1,k+1)]
    r=minimize_scalar(lambda b:nll_inst(z,Y,b),bounds=(lo,hi),method="bounded")
    return float(r.x),float(r.fun)

print("=== pairwise NLL (micro per instance, beta fitted) ===")
pw={}
for k,v in cand.items():
    z=v[A]-v[B];beta,nll=fit_beta(z,Y)
    # per-player nll at global beta for bootstrap
    p=np.clip(expit(-beta*z),1e-12,1-1e-12)
    li=-(Y*np.log(p)+(1-Y)*np.log(1-p))
    per_sum=np.bincount(PI,weights=li,minlength=P);per_n=np.bincount(PI,minlength=P)
    pw[k]=dict(beta=beta,nll=nll,bits=float(1-nll/np.log(2)),
               per_sum=per_sum,per_n=per_n,
               macro=float(np.mean([per_sum[i]/per_n[i] for i in range(P) if per_n[i]>=3])))
    print(f"  {k:7s} beta={beta:7.3f} nll={nll:.5f} bits_saved={pw[k]['bits']:+.4f} macro_nll={pw[k]['macro']:.5f}")

print("=== BT free fit (MM/L-BFGS) ===")
nab=wins; nba=cnt-wins
def bt_nll_grad(x):
    b=x-x.mean()
    u=b[UB]-b[UA]
    s=expit(u)
    nll=-np.sum(nab*np.log(np.clip(s,1e-15,1))+nba*np.log(np.clip(1-s,1e-15,1)))
    du=-nab*(1-s)+nba*s
    grad=np.zeros(M); np.add.at(grad,UA,-du); np.add.at(grad,UB,du)
    grad=grad-grad.mean()
    return nll,grad
r=minimize(bt_nll_grad,np.zeros(M),jac=True,method="L-BFGS-B",
           options=dict(maxiter=500,gtol=1e-8,ftol=1e-12))
bhat=r.x-r.x.mean(); nll_free=r.fun
print("  free nll",nll_free,"N",len(A),"per-inst",nll_free/len(A),"bits_saved",
      float(1-(nll_free/len(A))/np.log(2)),"iters",r.nit)

print("=== constrained BT: b = c + beta*d ===")
def con_nll(params,dvec):
    c,beta=params; b=c+beta*dvec; u=b[UB]-b[UA]; s=expit(u)
    return -np.sum(nab*np.log(np.clip(s,1e-15,1))+nba*np.log(np.clip(1-s,1e-15,1)))
con={}
for k,v in cand.items():
    vv=(v-v.mean())/(v.std()+1e-12)
    rr=minimize(lambda p:con_nll(p,vv),[0.0,0.1],method="Nelder-Mead",
                options=dict(xatol=1e-9,fatol=1e-6,maxiter=2000))
    nllc=rr.fun
    con[k]=dict(c=rr.x[0],beta=rr.x[1],nll=nllc,
                expl_dev=float(1-nllc/nll_free),
                spearman_bhat=sp(v,bhat),kendall_bhat=float(pd.Series(v).corr(pd.Series(bhat),method="kendall")))
    print(f"  {k:7s} beta={rr.x[1]:+.4f} nll={nllc:.0f} expl_dev={con[k]['expl_dev']:+.4f} "
          f"rho_bhat={con[k]['spearman_bhat']:+.4f} tau_bhat={con[k]['kendall_bhat']:+.4f}")
print("  total signal: free vs uniform bits_saved",float(1-(nll_free/len(A))/np.log(2)))

# ---------- per-player spearman of the free-bhat ordering (ceiling in spearman units)
dceil=bhat[mi_row]; rhos=[];ls=[]
for pi,idx in groups.items():
    l=len(idx)
    if l<3: continue
    rr=sp(-dceil[idx],perf[idx])
    if np.isfinite(rr): rhos.append(rr);ls.append(l)
rhos=np.array(rhos);ls=np.array(ls)
print("CEILING per-player spearman(free bhat): lw=%.4f macro=%.4f"%(float((rhos*ls).sum()/ls.sum()),float(rhos.mean())))
con["__ceiling__"]=dict(l_weighted=float((rhos*ls).sum()/ls.sum()),macro=float(rhos.mean()))

# ---------- n_plays confound
print("=== n_plays confound ===")
# within-player: does more plays -> better perf?
rr=[];ll=[]
for pi,idx in groups.items():
    l=len(idx)
    if l<5: continue
    rr.append(sp(np.log(d.n_plays.values[idx].astype(float)),perf[idx]));ll.append(l)
rr=np.array(rr);ll=np.array(ll)
print("  mean within-player rho(log n_plays, perf) =",round(float((rr*ll).sum()/ll.sum()),4))
print("  mean within-player rho(log n_plays, star) =",
      round(float(np.mean([sp(np.log(d.n_plays.values[i].astype(float)),cand['star'][mi_row][i]) for i in list(groups.values())[:300]])),4))
print("  corr across rows log n_plays vs star",round(float(np.corrcoef(np.log(d.n_plays.values),cand['star'][mi_row])[0,1]),4))
print("  corr across rows perf vs log n_plays",round(float(np.corrcoef(perf,np.log(d.n_plays.values))[0,1]),4))

# ---------- bootstrap CIs over players for micro NLL and lw spearman
B=200; rng=np.random.default_rng(1); out={}
for k in ["star","msd","notes","random"]:
    ps=pw[k]["per_sum"];pn=pw[k]["per_n"]
    vals=[];svals=[]
    for _ in range(B):
        s=rng.integers(0,P,P)
        vals.append(ps[s].sum()/pn[s].sum())
        svals.append(np.average(spear[k]["values"][s][spear[k]["ls"][s]>=3] if False else spear[k]["values"],weights=spear[k]["ls"]))
    out[k]=dict(nll_ci=[float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))],
                spearman_ci=[float(np.percentile(svals,2.5)),float(np.percentile(svals,97.5))])
    print("CI",k,out[k])

json.dump({"spearman":{k:{kk:vv for kk,vv in v.items() if kk not in("values","ls")} for k,v in spear.items()},
           "pairwise":{k:{kk:vv for kk,vv in v.items() if kk not in("per_sum","per_n")} for k,v in pw.items()},
           "constrained":{k:{kk:float(vv) for kk,vv in v.items()} for k,v in con.items()},
           "bt_free_bits":float(1-(nll_free/len(A))/np.log(2)),
           "ci":out},open("scratch_rank_metric/probe2.json","w"),indent=1)
np.save("scratch_rank_metric/bhat.npy",bhat)
print("done",round(time.time()-t0,1))
