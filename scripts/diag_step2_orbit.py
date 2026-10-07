import gc, sys, numpy as np, pandas as pd, resource
sys.path.insert(0, "scripts")
from diag_step2_feasible import rank_k_fit
print("start rss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024, flush=True)
mn = pd.read_parquet("data/processed/step1_1k.parquet", columns=["player_id","beatmap_id","loss"])
print("after read rss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024, flush=True)
k = int(mn.beatmap_id.max()) + 1 + 1
mn["pair"] = mn.player_id.astype(np.int64) * k + mn.beatmap_id.astype(np.int64)
up = np.unique(mn.pair.to_numpy())
tps = up[np.random.default_rng(20260928).random(len(up)) < 0.30]
tr = mn[~mn.pair.isin(set(tps.tolist()))].reset_index(drop=True)
del mn; gc.collect()
ent = np.unique(tr.player_id.to_numpy()); itm = np.unique(tr.beatmap_id.to_numpy())
ep = {v: i for i, v in enumerate(ent)}; ip = {v: i for i, v in enumerate(itm)}
u = np.ascontiguousarray(tr.player_id.map(ep).to_numpy(), dtype=np.int64)
m = np.ascontiguousarray(tr.beatmap_id.map(ip).to_numpy(), dtype=np.int64)
y = np.ascontiguousarray(tr.loss.to_numpy(), dtype=np.float64)
del tr; gc.collect()
print("n_ent", len(ent), "n_item", len(itm), "plays", len(y),
      "rss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024, flush=True)
f0 = rank_k_fit(u, m, y, len(ent), len(itm), 4, 1e-3, iters=25, seed=0)
D0 = f0["D"]; G0 = D0 @ D0.T
print("fit0 done rss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024, flush=True)
print("seed  rawD    Gram    orthdef  residAfterA", flush=True)
for sd in (1, 2):
    f1 = rank_k_fit(u, m, y, len(ent), len(itm), 4, 1e-3, iters=25, seed=sd)
    D1 = f1["D"]
    A, *_ = np.linalg.lstsq(D0, D1, rcond=None)
    od = float(np.linalg.norm(A @ A.T - np.eye(4)) / np.linalg.norm(np.eye(4)))
    rs = float(np.linalg.norm(D0 @ A - D1) / np.linalg.norm(D1))
    gc_ = float(np.corrcoef(G0.ravel(), (D1 @ D1.T).ravel())[0, 1])
    dr = float(np.corrcoef(D0.ravel(), D1.ravel())[0, 1])
    print(f"{sd:4d} {dr:7.4f} {gc_:7.4f} {od:8.4f} {rs:13.4f}", flush=True)
    del f1, D1, A; gc.collect()
