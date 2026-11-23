"""T3: pool CM3P window vectors into per-chart difficulty features.

Variants (plan v5 section 6; v2 adds A0L/A2b/A2c)
    A0   mean over windows, re-normalised                      512
    A0L  A0 + log mean_norm + log n_win (metadata control)     514
    A1   per-dim mean + std (risk item)                       1024
    A2   A0 + mean/sd/q90/q99/max of z_t, logs                 519
    A2b  A0 + mean/sd/q75/q95/max of z_t, logs                 519
    A2c  A0 + mean/sd/q75/q90/q95/q99/max of z_t, logs         521
    A3   A2 + longest high-pressure run + longest rest run     521
    A4   difficulty-weighted mean (softmax over beta*z_t)      512
    A5   minacalc.aggregate_rows(W_t)                          512
    A6   a small TCN over the window sequence, trained on the rank NLL
         (written as a signal column, not a fixed feature matrix)

z_t = w . ((W_t - mu) / sd) uses the linear head of the A0 chart-level model.
The head is per-fold, not per-chart: a chart in fold k uses w_k; a chart that
was never in the fit's maps uses w_full.  A0/A0L/A1/A5 are label-free and can
be computed once; A2/A2b/A2c/A3/A4 are functions of (window, head), so they must
be rebuilt for every split seed / protocol.  The head bundle records the split
it was fit on (inp/chart_seed/lam/val_frac) so the per-seed features can be
audited.

P4 must not reuse a P2 head: P2 train rows include P4 test charts.  Fit a
separate head on the P4 train rows and use mode="full" (w_full) for every chart.
P2 still has the standard target-encoding dependence of the *training* charts'
features on the held-out fold's train rows; A2b vs A0 should be read with that
in mind (see the plan's diagnostic).

A0 is the T3 control: mean-then-normalise of the same canon window store that
step0c wrote. It is not required to reproduce the legacy chart table; all T3
variants share the same T1 crop, so A0 vs A2..A6 isolates pooling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

VARIANTS = ("a0", "a0l", "a1", "a2", "a2b", "a2c", "a3", "a4", "a5", "a6")
WIDTH = {"a0": 512, "a0l": 514, "a1": 1024, "a2": 519, "a2b": 519,
         "a2c": 521, "a3": 521, "a4": 512, "a5": 512, "a6": 512}


# --------------------------------------------------------------------- heads
@dataclass
class HeadBundle:
    maps: np.ndarray
    fold: np.ndarray
    w: np.ndarray                 # (K, D) linear head in standardised space
    w_full: np.ndarray            # (D,)
    mu: np.ndarray
    sd: np.ndarray
    content_dim: int
    chart_folds: int
    meta: dict

    @classmethod
    def load(cls, path: Path | str) -> "HeadBundle":
        z = np.load(path, allow_pickle=False)
        meta = {k: z[k].item() if getattr(z[k], "shape", ()) == () else z[k]
                for k in z.files if k not in
                ("maps", "fold", "w", "w_full", "mu", "sd")}
        maps = np.asarray(z["maps"], dtype=np.int64)
        fold = np.asarray(z["fold"], dtype=np.int64)
        if maps.size != fold.size:
            raise SystemExit(f"{path}: maps and fold lengths differ")
        return cls(maps=maps, fold=fold, w=np.asarray(z["w"], dtype=np.float32),
                   w_full=np.asarray(z["w_full"], dtype=np.float32),
                   mu=np.asarray(z["mu"], dtype=np.float32),
                   sd=np.asarray(z["sd"], dtype=np.float32),
                   content_dim=int(np.asarray(z["content_dim"]).item()),
                   chart_folds=int(np.asarray(z["chart_folds"]).item()),
                   meta={k: (v.item() if hasattr(v, "item") else v) for k, v in meta.items()})

    def __post_init__(self):
        self._pos = {int(b): i for i, b in enumerate(self.maps)}

    def weight_for(self, bid: int, mode: str = "fold") -> np.ndarray:
        """mode="fold": the chart's own fold head (P2); "full": w_full (P4)."""
        if mode == "full":
            return self.w_full
        if mode != "fold":
            raise SystemExit(f"unknown head mode {mode!r}; use fold|full")
        i = self._pos.get(int(bid))
        if i is None or self.fold[i] < 0:
            return self.w_full
        k = int(self.fold[i])
        return self.w[k] if k < self.w.shape[0] else self.w_full

    def z(self, W: np.ndarray, bid: int, mode: str = "fold") -> np.ndarray:
        Cs = (W.astype(np.float32) - self.mu) / self.sd
        return Cs @ self.weight_for(int(bid), mode=mode)


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for x in np.asarray(mask, dtype=bool):
        cur = cur + 1 if x else 0
        if cur > best:
            best = cur
    return best


# --------------------------------------------------------------------- pools
def pool_chart(variant: str, W: np.ndarray, bid: int,
               heads: Optional[HeadBundle] = None, *,
               beta: float = 5.0, a4_mode: str = "weighted",
               head_mode: str = "fold",
               mean_norm: Optional[float] = None) -> np.ndarray:
    """One chart's feature vector. W is (n_win, 512), model-unit vectors."""
    W = np.asarray(W, dtype=np.float32)
    if W.ndim != 2 or W.shape[0] == 0:
        raise SystemExit(f"chart {bid}: empty window matrix {W.shape}")
    n = W.shape[0]
    mean = W.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    mn = float(mean_norm) if mean_norm is not None else norm
    u = unit(mean)
    logs = [math.log(max(mn, 1e-8)), math.log(max(n, 1))]

    if variant == "a0":
        return u.astype(np.float32)
    if variant == "a0l":
        return np.concatenate([u, np.asarray(logs, dtype=np.float32)]).astype(np.float32)
    if variant == "a1":
        return np.concatenate([u, W.std(axis=0)]).astype(np.float32)
    if variant in ("a2", "a2b", "a2c", "a3", "a4"):
        if heads is None:
            raise SystemExit(f"{variant} needs --heads (the A0 linear head bundle)")
        if heads.mu.size != W.shape[1]:
            raise SystemExit(
                f"{variant}: the head bundle has content_dim={heads.mu.size} but the "
                f"window vectors have {W.shape[1]} dims; fit A0 on the same content")
        z = heads.z(W, bid, mode=head_mode).astype(np.float64)
        if variant == "a4":
            if a4_mode == "lse":
                Cs = (W - heads.mu) / heads.sd
                m = Cs.max(axis=0, keepdims=True)
                lse = m + np.log(np.exp(Cs - m).sum(axis=0, keepdims=True)) / beta
                return (lse[0] / 1.0).astype(np.float32)   # temperature already applied
            # w lives in the chart-standardised space; z can therefore have an
            # arbitrary scale.  Standardise z *within the chart* so beta=5 means
            # the same thing for every chart and the softmax does not saturate.
            zs = (z - z.mean()) / (z.std() + 1e-8)
            zz = beta * (zs - zs.max())
            wts = np.exp(zz)
            wts /= max(wts.sum(), 1e-12)
            return unit((wts[:, None] * W).sum(axis=0)).astype(np.float32)
        q = {p: float(np.quantile(z, p)) for p in (0.25, 0.75, 0.90, 0.95, 0.99)}
        if variant == "a2":
            stats = [float(z.mean()), float(z.std()), q[0.90], q[0.99], float(z.max())]
        elif variant == "a2b":
            stats = [float(z.mean()), float(z.std()), q[0.75], q[0.95], float(z.max())]
        elif variant == "a2c":
            stats = [float(z.mean()), float(z.std()), q[0.75], q[0.90],
                     q[0.95], q[0.99], float(z.max())]
        else:  # a3 keeps the A2 stats and adds the two run lengths
            stats = [float(z.mean()), float(z.std()), q[0.90], q[0.99], float(z.max())]
        f = np.concatenate([u, np.asarray(stats, dtype=np.float32),
                            np.asarray(logs, dtype=np.float32)])
        if variant == "a3":
            hi = z >= q[0.75]
            lo = z <= q[0.25]
            f = np.concatenate([f, np.asarray([_longest_run(hi), _longest_run(lo)],
                                              dtype=np.float32)])
        return f.astype(np.float32)
    if variant == "a5":
        # aggregate_rows(v) maps (k rows, n columns) -> (n,): here each window is
        # one row and each embedding dimension is a column, so the result is the
        # 512-d difficulty-weighted aggregation of the window set.
        import minacalc
        return np.asarray(minacalc.aggregate_rows(W), dtype=np.float32)
    raise SystemExit(f"unknown pooling variant {variant!r}")


# ------------------------------------------------------------------ features
def load_index(path: Path | str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for c in ("beatmap_id", "n_win", "mean_norm"):
        if c not in df:
            raise SystemExit(f"{path}: missing column {c}")
    return df


def build_feature_table(index: pd.DataFrame, win_dir: Path | str, variant: str,
                        heads: Optional[HeadBundle] = None, *,
                        beta: float = 5.0, a4_mode: str = "weighted",
                        head_mode: str = "fold", limit: int = 0,
                        log: Callable[[str], None] = print) -> pd.DataFrame:
    """One row per chart: beatmap_id + features (object array)."""
    if (variant in ("a2", "a2b", "a2c", "a3", "a4") and head_mode == "fold"
            and heads is not None and heads.chart_folds <= 1):
        log(f"[pool] WARNING: {variant} uses a P1 head bundle (chart_folds=1): z_t "
            f"for a chart saw that chart's own labels.  Fit A0 with --chart-folds 5 "
            f"for a chart-disjoint (P2) comparison.")
    ids, feats, missing = [], [], []
    win_dir = Path(win_dir)
    rows = index if not limit else index.head(limit)
    t0 = __import__("time").time()
    for k, r in enumerate(rows.itertuples(index=False)):
        bid = int(r.beatmap_id)
        p = win_dir / f"{bid}.npy"
        if not p.exists():
            missing.append(bid)
            continue
        W = np.load(p).astype(np.float32)
        f = pool_chart(variant, W, bid, heads, beta=beta, a4_mode=a4_mode,
                       head_mode=head_mode,
                       mean_norm=float(getattr(r, "mean_norm", float("nan")))
                       if np.isfinite(getattr(r, "mean_norm", float("nan"))) else None)
        expect = WIDTH.get(variant)
        if expect and f.size != expect:
            raise SystemExit(f"{bid}: {variant} produced {f.size} dims, want {expect}")
        ids.append(bid)
        feats.append(f)
        if (k + 1) % 2000 == 0:
            log(f"[pool] {variant}: {k + 1:,}/{len(rows):,} "
                f"({__import__('time').time() - t0:.0f}s)")
    if missing:
        log(f"[pool] WARNING: {variant}: {len(missing)} charts have no .npy "
            f"(e.g. {missing[:5]})")
    return pd.DataFrame({"beatmap_id": np.asarray(ids, dtype=np.int64),
                         "features": feats})


# ---------------------------------------------------------------------- A6
def _torch():
    import torch
    return torch


class TCNPooler:
    """2-layer dilated TCN over the window sequence -> 512-d pooled vector -> d."""

    def __init__(self, dim: int = 512, hid: int = 128):
        torch = _torch()
        nn = torch.nn
        self.net = nn.Module()
        self.net.c1 = nn.Conv1d(dim, hid, 3, padding=1)
        self.net.c2 = nn.Conv1d(hid, hid, 3, padding=2, dilation=2)
        self.net.out = nn.Linear(2 * hid, dim)
        self.net.head = nn.Linear(dim, 1)
        self.net = self.net

    def parameters(self):
        return self.net.parameters()

    def forward(self, x, mask):
        torch = _torch()
        m = mask.unsqueeze(-1)
        h = self.net.c1(torch.transpose(x, 1, 2))
        h = torch.nn.functional.gelu(h) * torch.transpose(m, 1, 2)
        h = self.net.c2(h)
        h = torch.nn.functional.gelu(h) * torch.transpose(m, 1, 2)
        h = torch.transpose(h, 1, 2)
        denom = m.sum(1).clamp(min=1.0)
        mean = (h * m).sum(1) / denom
        mx = h.masked_fill(~m.bool(), -1e9).amax(1)
        pooled = self.net.out(torch.cat([mean, mx], dim=-1))
        return self.net.head(pooled).squeeze(-1), pooled


def _encode_rows(df: pd.DataFrame, map_index: dict):
    item = df["beatmap_id"].map(map_index).to_numpy(np.int32)
    perf = (-df["loss"].astype(np.float64)).to_numpy()
    players = np.sort(df["player_id"].unique())
    p_idx = {int(p): i for i, p in enumerate(players)}
    pi = df["player_id"].map(p_idx).to_numpy(np.int32)
    tmp = pd.DataFrame({"pi": pi, "base": df["player_id_base"].to_numpy()})
    idx_map = tmp.groupby("pi").indices
    groups = [np.asarray(idx_map[i]) for i in range(len(players))]
    clusters = (tmp.groupby("pi")["base"].first()
                .reindex(range(len(players))).to_numpy())
    return item, perf, groups, np.asarray(clusters, dtype=np.int64)


def _val_clusters(clusters: np.ndarray, frac: float, seed: int):
    ids = np.unique(clusters)
    rng = np.random.default_rng(seed)
    n = max(1, int(round(frac * ids.size))) if frac > 0 else 0
    val = set(rng.choice(ids, n, replace=False).tolist()) if n else set()
    fit = np.array([i for i, c in enumerate(clusters) if int(c) not in val], np.int64)
    hold = np.array([i for i, c in enumerate(clusters) if int(c) in val], np.int64)
    return fit, hold


def _bt_loss_and_grad(d: np.ndarray, tab) -> tuple[float, np.ndarray]:
    from scipy.special import expit
    d = np.asarray(d, dtype=np.float64)
    mu, sd = d.mean(), d.std() + 1e-8
    z = (d - mu) / sd
    u = z[tab.ub] - z[tab.ua]
    s = expit(u)
    N = float(tab.n_instances)
    loss = -float(np.sum(tab.n_ab * np.log(np.clip(s, 1e-15, 1.0))
                         + tab.n_ba * np.log(np.clip(1.0 - s, 1e-15, 1.0)))) / N
    gu = (-tab.n_ab * (1.0 - s) + tab.n_ba * s) / N
    gz = np.zeros_like(z)
    np.add.at(gz, tab.ua, -gu)
    np.add.at(gz, tab.ub, gu)
    gd = (gz - gz.mean() - z * (gz * z).mean()) / sd
    return loss, gd.astype(np.float32)


def _bt_loss(d: np.ndarray, tab) -> float:
    return float(_bt_loss_and_grad(d, tab)[0])


def _pack_windows(windows: dict[int, np.ndarray], maps: np.ndarray):
    missing = [int(b) for b in maps if int(b) not in windows]
    if missing:
        raise SystemExit(f"{len(missing)} charts have no window .npy (e.g. {missing[:5]})")
    T = max(int(windows[int(b)].shape[0]) for b in maps)
    X = np.zeros((len(maps), T, 512), dtype=np.float16)
    M = np.zeros((len(maps), T), dtype=bool)
    for i, b in enumerate(maps):
        W = windows[int(b)]
        X[i, :W.shape[0]] = W.astype(np.float16)
        M[i, :W.shape[0]] = True
    return X, M


def _forward_all(model, X, M, chunk: int = 256) -> np.ndarray:
    torch = _torch()
    out = np.zeros(X.shape[0], dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X.shape[0], chunk):
            xb = torch.as_tensor(X[i:i + chunk].astype(np.float32))
            mb = torch.as_tensor(M[i:i + chunk], dtype=torch.bool)
            out[i:i + chunk] = model.forward(xb, mb)[0].numpy()
    return out


def train_a6_signal(step1: pd.DataFrame, windows: dict[int, np.ndarray],
                    heads: HeadBundle, *, epochs: int = 80, hid: int = 128,
                    lr: float = 1e-3, wd: float = 1e-4, val_frac: float = 0.2,
                    val_seed: int = 0, patience: int = 15, chunk: int = 256,
                    threads: int = 6, seed: int = 0, log=print) -> pd.DataFrame:
    """P2 (chart-disjoint) signal for the TCN variant; returns beatmap_id + signal."""
    torch = _torch()
    torch.set_num_threads(int(threads))
    from dcr import rank_metrics as rm

    maps = np.asarray(heads.maps, dtype=np.int64)
    if heads.chart_folds <= 1:
        raise SystemExit("A6 needs a chart-disjoint head bundle (chart_folds > 1)")
    map_idx = {int(b): i for i, b in enumerate(maps)}
    df = step1[step1["beatmap_id"].isin(set(maps.tolist()))].reset_index(drop=True)
    train = df[df["split"] == "train"].reset_index(drop=True)
    M = len(maps)
    X, Msk = _pack_windows(windows, maps)
    log(f"[a6] maps={M:,} T={X.shape[1]} charts={len(df):,} train_rows={len(train):,}")

    fold = np.asarray(heads.fold, dtype=np.int64)
    signal = np.full(M, np.nan, dtype=np.float64)
    for k in range(heads.chart_folds):
        torch.manual_seed(int(seed) + k)
        rows = train[train["beatmap_id"].isin(set(maps[fold != k].tolist()))]
        if rows.empty:
            raise SystemExit(f"a6 fold {k}: no training rows")
        item, perf, groups, clusters = _encode_rows(rows, map_idx)
        fit_p, val_p = _val_clusters(clusters, val_frac, val_seed + k)
        tab_fit = rm.build_pair_table([groups[p] for p in fit_p], item, perf, M)
        tab_val = (rm.build_pair_table([groups[p] for p in val_p], item, perf, M)
                   if val_p.size else None)
        model = TCNPooler(512, hid)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        best = (float("inf"), None, 0)
        for ep in range(1, epochs + 1):
            d = _forward_all(model, X, Msk, chunk)
            loss, gd = _bt_loss_and_grad(d, tab_fit)
            opt.zero_grad(set_to_none=True)
            for i in range(0, M, chunk):
                xb = torch.as_tensor(X[i:i + chunk].astype(np.float32))
                mb = torch.as_tensor(Msk[i:i + chunk], dtype=torch.bool)
                db, _ = model.forward(xb, mb)
                db.backward(torch.as_tensor(gd[i:i + chunk]))
            opt.step()
            vl = (_bt_loss(_forward_all(model, X, Msk, chunk), tab_val)
                  if tab_val is not None and tab_val.n_pairs else loss)
            if vl < best[0] - 1e-7:
                best = (vl, {kk: v.detach().clone()
                             for kk, v in model.net.state_dict().items()}, ep)
            elif ep - best[2] > patience:
                break
            if ep == 1 or ep % 10 == 0:
                log(f"[a6 fold {k} ep {ep}] train={loss:.5f} val={vl:.5f} best={best[0]:.5f}")
        if best[1] is not None:
            model.net.load_state_dict(best[1])
        d = _forward_all(model, X, Msk, chunk)
        z = (d - d.mean()) / (d.std() + 1e-8)
        signal[fold == k] = z[fold == k]
        log(f"[a6 fold {k}] rows={len(rows):,} best_ep={best[2]} val={best[0]:.5f}")
    return pd.DataFrame({"beatmap_id": maps, "signal": signal})
