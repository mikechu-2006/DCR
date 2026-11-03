
import numpy as np, pandas as pd, json, time
from scipy.stats import rankdata
from scipy.optimize import minimize_scalar
from scipy.special import expit

t0=time.time()
df = pd.read_parquet("data/processed/step1_1k_pairbest.parquet")
meta = pd.read_csv("data/processed/beatmap_meta_1k.csv")[["beatmap_id","star","count_total","bpm"]]
msd = pd.read_parquet("data/processed/minacalc_msd_1k.parquet")[["beatmap_id","msd_overall"]]
df = df.merge(meta,on="beatmap_id",how="left").merge(msd,on="beatmap_id",how="left")
df["perf"] = -df["loss"].astype(np.float64)

print("rows",len(df),"players",df.player_id.nunique(),"maps",df.beatmap_id.nunique())
print("n_plays", {k:round(float(v),3) for k,v in df.n_plays.describe().to_dict().items()})
print("loss q:",np.percentile(df.loss,[0,1,5,50,95,99,100]).round(3))
print("star nan",round(float(df.star.isna().mean()),4),"msd nan",round(float(df.msd_overall.isna().mean()),4))

ok = df.star.notna() & df.msd_overall.notna()
d = df[ok].reset_index(drop=True)
print("rows on common map set",len(d))

maps = np.sort(d.beatmap_id.unique()); M=len(maps)
mid2idx = {m:i for i,m in enumerate(maps)}
d["mi"] = d.beatmap_id.map(mid2idx).astype(np.int32)
players = np.sort(d.player_id.unique()); P=len(players)
pid2idx={p:i for i,p in enumerate(players)}
d["pi"]=d.player_id.map(pid2idx).astype(np.int32)
print("M",M,"P",P)

g = d.groupby("mi")
cand = {
 "star": g.star.first().reindex(range(M)).values.astype(np.float64),
 "msd":  g.msd_overall.first().reindex(range(M)).values.astype(np.float64),
 "notes":g.count_total.first().reindex(range(M)).values.astype(np.float64),
 "bpm":  g.bpm.first().reindex(range(M)).values.astype(np.float64),
}
rng=np.random.default_rng(0)
cand["random"]=rng.normal(size=M)

perf = d.perf.values
mi_row = d.mi.values
groups = d.groupby("pi").indices
groups = {k:np.asarray(v) for k,v in groups.items()}
print("n players l>=3", sum(len(v)>=3 for v in groups.values()))

def spearman_pair(x,y):
    rx=rankdata(x); ry=rankdata(y)
    rx=rx-rx.mean(); ry=ry-ry.mean()
    den=np.sqrt((rx*rx).sum()*(ry*ry).sum())
    return float((rx*ry).sum()/den) if den>0 else np.nan

# ---------------- metric 1: per-player Spearman (-d vs perf)
spear={}
for name,dv in cand.items():
    drow=dv[mi_row]
    rhos=[];ls=[]
    for pi,idx in groups.items():
        l=len(idx)
        if l<3: continue
        r=spearman_pair(-drow[idx], perf[idx])
        if np.isfinite(r): rhos.append(r); ls.append(l)
    rhos=np.array(rhos); ls=np.array(ls)
    z=np.arctanh(np.clip(rhos,-0.999999,0.999999))
    zlw=(z*(ls-3)).sum()/(ls-3).sum()
    spear[name]={"l_weighted":float((rhos*ls).sum()/ls.sum()),
                 "macro":float(rhos.mean()),
                 "fisher_z_l3":float(np.tanh(zlw)),
                 "median":float(np.median(rhos)),
                 "n_players":int(len(rhos)),
                 "l_med":float(np.median(ls)),
                 "corr_rho_l":float(np.corrcoef(rhos,ls)[0,1])}
    print("SPEAR",name,{k:round(v,4) for k,v in spear[name].items()})
print("time spearman",round(time.time()-t0,1))

# ---------------- build pair instances (player x unordered pair)
# accumulate unique unordered pairs (a<b) counts and wins for a (perf_a>perf_b)
keys=[]; ywin=[]
n_inst=0
for pi,idx in groups.items():
    l=len(idx)
    if l<2: continue
    ii,jj=np.triu_indices(l,1)
    a=mi_row[idx][ii]; b=mi_row[idx][jj]
    # order a<b
    swap=a>b
    aa=np.where(swap,b,a); bb=np.where(swap,a,b)
    y=perf[idx][ii]>perf[idx][jj]
    y=np.where(swap,~y,y)
    keys.append(aa.astype(np.int64)*M+bb)
    ywin.append(y.astype(np.int8))
    n_inst+=len(aa)
keys=np.concatenate(keys); ywin=np.concatenate(ywin)
print("pair instances",n_inst,"unique",len(np.unique(keys)),"time",round(time.time()-t0,1))
uk, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
wins = np.bincount(inv, weights=ywin, minlength=len(uk)).astype(np.float64)
ua = (uk//M).astype(np.int32); ub=(uk%M).astype(np.int32)
print("unique pairs",len(uk),"count q",np.percentile(cnt,[0,50,90,99,100]))
print("p_hat hist",np.histogram(wins/cnt,bins=[0,.05,.2,.5,.8,.95,1.0001])[0])

# ---------------- metric 2: pairwise NLL with fitted beta (micro over instances)
# p(y=1 | z) = sigmoid(-beta*z), z = d[a]-d[b]
def nll_beta(z,y,beta):
    p=expit(-beta*z)
    p=np.clip(p,1e-12,1-1e-12)
    return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))

def fit_nll(z,y):
    # beta>=0 grid then refine
    grid=np.concatenate([[0.0],np.geomspace(1e-4,1e3,200)])
    vals=[nll_beta(z,y,b) for b in grid]
    k=int(np.argmin(vals))
    lo=grid[max(0,k-1)]; hi=grid[min(len(grid)-1,k+1)]
    r=minimize_scalar(lambda b:nll_beta(z,y,b),bounds=(lo,hi),method="bounded")
    return float(r.x), float(r.fun)

pair={}
for name,dv in cand.items():
    z=dv[ua]-dv[ub]
    beta,nll=fit_nll(z,ywin.astype(np.float64))
    # uniform baseline = log 2
    pair[name]={"beta":beta,"nll":nll,"nll_log2":float(np.log(2)),
                "bits_saved_vs_uniform":float(1-nll/np.log(2))}
    # per-player macro NLL: need per-player; recompute by player
    per=[]
    for pi,idx in groups.items():
        l=len(idx)
        if l<3: continue
        ii,jj=np.triu_indices(l,1)
        a=mi_row[idx][ii]; b=mi_row[idx][jj]
        yy=(perf[idx][ii]>perf[idx][jj])
        zz=dv[a]-dv[b]
        p=np.clip(expit(-beta*zz),1e-12,1-1e-12)
        per.append(-np.mean(yy*np.log(p)+(1-yy)*np.log(1-p)))
    pair[name]["macro_nll"]=float(np.mean(per))
    print("NLL",name,{k:round(v,5) for k,v in pair[name].items()})
print("time nll",round(time.time()-t0,1))

json.dump({"spearman":spear,"pairwise":pair},open("scratch_rank_metric/probe1.json","w"),indent=1)
np.savez("scratch_rank_metric/pairs.npz",uk=uk,cnt=cnt,wins=wins,ua=ua,ub=ub,cand=np.array([cand[k] for k in cand]),names=np.array(list(cand.keys())))
print("saved", round(time.time()-t0,1))
