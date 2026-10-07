import numpy as np, glob, os
store = {}
for f in sorted(glob.glob("data/processed/step2_1k_*_loss.npz")):
    nm = os.path.basename(f).replace("step2_1k_", "").replace("_loss.npz", "")
    z = np.load(f)
    if int(z["P"].shape[1]) <= 4:
        store[nm] = {"D": z["D"].astype(np.float64), "ids": z["beatmap_ids"],
                     "dim": int(z["P"].shape[1])}
print("loaded", list(store), flush=True)

def stats(A, B, sub=4000):
    d = float(np.corrcoef(A.ravel(), B.ravel())[0, 1])
    n = float(np.corrcoef(np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1))[0, 1])
    i = np.random.default_rng(0).choice(A.shape[0], sub, replace=False)
    G1, G2 = A[i] @ A.T, B[i] @ B.T
    g = float(np.corrcoef(G1.ravel(), G2.ravel())[0, 1])
    del G1, G2
    M, *_ = np.linalg.lstsq(A, B, rcond=None)
    od = float(np.linalg.norm(M @ M.T - np.eye(A.shape[1])) / np.linalg.norm(np.eye(A.shape[1])))
    rf = float(np.linalg.norm(A @ M - B) / np.linalg.norm(B))
    return d, n, g, od, rf

print(f"{'pair':44s} {'dcorr':>8s} {'nrmcorr':>8s} {'gramcorr':>9s} {'orthdef':>8s} {'residA':>8s}")
pairs = [("mirt_F0_dim2", "mf_dot_F0_dim2"), ("mirt_F0_dim3", "mf_dot_F0_dim3"),
         ("mirt_F0_dim4", "mf_dot_F0_dim4"),
         ("mirt_F0_dim2", "mirt_exp_F0_dim2"), ("mirt_F0_dim3", "mirt_exp_F0_dim3"),
         ("mirt_F0_dim2", "mirt_F1_dim2_pd"), ("mirt_F0_dim3", "mirt_F1_dim3_pd"),
         ("mirt_F0_dim4", "mirt_F1_dim4_pd")]
for a, b in pairs:
    if a in store and b in store:
        r = stats(store[a]["D"], store[b]["D"])
        print(f"{a+' vs '+b:44s} {r[0]:+8.4f} {r[1]:+8.4f} {r[2]:+9.4f} {r[3]:8.4f} {r[4]:8.4f}",
              flush=True)
