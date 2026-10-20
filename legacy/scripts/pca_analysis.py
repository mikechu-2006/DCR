#!/usr/bin/env python
"""PCA_analysis: (谱面, 玩家) 中位数矩阵的主成分分解。

数据
----
输入是 step0_preprocess.py 产出的 play 表（player_id, beatmap_id, loss），
每一行是一次 play。也可以直接传 step1_{tag}.parquet；脚本不读 split 列，
会把整表一起聚合（要只看 train/test，请先过滤）。按 (beatmap_id, player_id)
分组取 loss 的中位数，得到 (谱面, 玩家) 矩阵 M：

    M[m, u] = median{ loss : (player, beatmap) = (u, m) }

1k 里它是 20,655 张谱面 x 990 名玩家，只有 4.2% 的 cell 至少有一次 play。
所以"完整矩阵"必须说明缺失 cell 怎么补。本脚本给两个口径：

  raw          缺失 cell 用该玩家的 cell 均值补，再按玩家中心化。
               等价于对稀疏中心化矩阵做 SVD；总方差 = 观测 cell 对玩家
               均值的离差平方和。这是"对完整矩阵做 PCA"的字面口径。

  interaction  先用交替均值拟合加性模型
                   loss_mp ~= c + a_p + b_m
               （只用观测 cell；a = 玩家水平，b = 谱面难度），缺失 cell
               用 c + a_p + b_m 补，再对残差 R = loss - c - a - b 做 SVD。
               总方差 = 观测 cell 的加性残差平方和。残差就是低秩模型
               （mf_dot / mirt）能加在加性基线之上的"交互项"。
               交替均值收敛后，观测残差按每个玩家和每张谱面的和都是 0，
               所以补零后的残差矩阵已经中心化，直接 SVD 就是 PCA。

为什么两个口径的 total_ss 都只在观测 cell 上求和：补出来的 cell 在中心化后
恰好是 0（raw 用玩家均值、interaction 用加性预测），对 Frobenius 范数没有
贡献。因此 cumulative_evr = 100% = "把观测 cell 的这部分方差全部解释掉"。

输出的 cumulative_evr_signal 把 total_ss 换成 total_ss - noise_ss（近似
"非噪声方差"）。它是上界：主成分也会拟合噪声，所以 k 很大时可能超过 100%。
noise_ss 的估计：同一 cell 内 play 的合并方差 sigma^2，乘上中位数的抽样方差
系数（n=1 时为 sigma^2，n>=2 时近似 (pi/2) * sigma^2 / n）。它包含 cell 内
真实的状态漂移，所以是"噪声"的上界、信号的下界。

输出
----
    results/pca_{tag}/pca_{tag}_spectrum.csv   每个维度的 singular value /
                                               explained variance / 累计 / RMSE
    results/pca_{tag}/pca_{tag}_summary.json   矩阵统计 + 阈值 + 有效秩
    results/pca_{tag}/pca_{tag}.png            累计解释方差 + scree

用法
----
    python scripts/pca_analysis.py --tag 1k
    python scripts/pca_analysis.py --plays data/processed/step0_10k.parquet --tag 10k
    python scripts/pca_analysis.py --tag 1k --min-cell-plays 2 --views interaction

缺失 cell 的处理是这份分析唯一的口径选择。默认两个口径都跑；如果只关心
"低秩交互能解释多少"，用 --views interaction。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds

VIEWS = ("raw", "interaction")
GRID = [1, 2, 3, 4, 5, 7, 10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768]
THRESHOLDS = (0.50, 0.80, 0.90, 0.95, 0.99)
# 中位数抽样方差 / 单次 play 方差的大样本系数：Var(median) ~= (pi/2) * sigma^2 / n
MEDIAN_FACTOR = np.pi / 2.0


# --------------------------------------------------------------------- 数据
def load_plays(args) -> tuple[Path, pd.DataFrame]:
    """读 step0 的 play 表；不给路径时按 tag 找 parquet/csv。"""
    path = Path(args.plays) if args.plays else None
    if path is None:
        for suf in (".parquet", ".csv"):
            cand = Path("data/processed") / f"step0_{args.tag}{suf}"
            if cand.exists():
                path = cand
                break
    if path is None or not path.exists():
        raise SystemExit(
            f"找不到 play 表：{args.plays or f'data/processed/step0_{args.tag}.[parquet|csv]'}；"
            f"用 --plays 指定（step0_preprocess.py 的输出）")

    cols = [args.chart_col, args.player_col, args.value_col]
    t0 = time.time()
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=cols)
    else:
        df = pd.read_csv(path, usecols=cols)
    print(f"[load] {path}  {len(df):,} plays  ({time.time() - t0:.1f}s)", flush=True)
    return path, df


def build_matrix(df: pd.DataFrame, args):
    """play 表 -> 每个 cell 的中位数稀疏矩阵 + cell 的 play 数 / 方差。"""
    chart, player, value = args.chart_col, args.player_col, args.value_col
    t0 = time.time()

    grp = df.groupby([chart, player], sort=True)[value]
    med = grp.median()
    size = grp.size()
    # 只有 n>=2 的 cell 有方差；n=1 的 NaN 后面按 0 处理
    var = grp.std(ddof=1) ** 2 if args.noise else None
    n_cells_all = len(med)
    n_plays_all = int(size.sum())
    print(f"[cells] {n_cells_all:,} cells  {n_plays_all:,} plays  "
          f"({time.time() - t0:.1f}s)", flush=True)

    # --min-cell-plays：丢掉 play 数不足的 cell（默认 1 = 全保留）
    keep = size.to_numpy() >= args.min_cell_plays
    if not keep.all():
        med, size = med[keep], size[keep]
        if var is not None:
            var = var[keep]
        print(f"[cells] --min-cell-plays {args.min_cell_plays}: kept "
              f"{len(med):,} / {n_cells_all:,} cells "
              f"({size.sum():,} / {n_plays_all:,} plays)", flush=True)
    if len(med) == 0:
        raise SystemExit("--min-cell-plays 把所有 cell 都删掉了")

    ci, charts = pd.factorize(med.index.get_level_values(0), sort=True)
    pi, players = pd.factorize(med.index.get_level_values(1), sort=True)
    shape = (len(charts), len(players))
    X = sparse.csr_matrix((med.to_numpy(np.float64), (ci, pi)), shape=shape)
    N = sparse.csr_matrix((size.to_numpy(np.float32), (ci, pi)), shape=shape)
    V = None
    if var is not None:
        V = sparse.csr_matrix(
            (np.nan_to_num(var.to_numpy(np.float64), nan=0.0), (ci, pi)), shape=shape)
    print(f"[matrix] {shape[0]:,} charts x {shape[1]:,} players  "
          f"cells={X.nnz:,}  density={X.nnz / (shape[0] * shape[1]):.3%}  "
          f"({time.time() - t0:.1f}s)", flush=True)

    # 可选的轴过滤（默认不删，保持"完整矩阵"）
    n0, m0, c0 = shape[0], shape[1], X.nnz
    if args.min_chart_cells > 1:
        rkeep = np.diff(X.indptr) >= args.min_chart_cells
        X, N = X[rkeep], N[rkeep]
        if V is not None:
            V = V[rkeep]
        charts = charts[rkeep]
    if args.min_player_cells > 1:
        ckeep = np.bincount(X.indices, minlength=X.shape[1]) >= args.min_player_cells
        X, N = X[:, ckeep], N[:, ckeep]
        if V is not None:
            V = V[:, ckeep]
        players = players[ckeep]
    X, N = X.tocsr(), N.tocsr()
    X.sort_indices()
    N.sort_indices()
    if V is not None:
        V = V.tocsr()
        V.sort_indices()
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise SystemExit("轴过滤把所有行/列都删掉了")
    if X.nnz != c0:
        print(f"[matrix] axis filter: {n0:,}x{m0:,} / {c0:,} cells -> "
              f"{X.shape[0]:,}x{X.shape[1]:,} / {X.nnz:,} cells", flush=True)

    sizes = N.data.astype(np.float64)
    q = np.percentile(sizes, [50, 75, 90, 95, 99])
    stats = dict(
        n_charts=int(X.shape[0]), n_players=int(X.shape[1]),
        n_cells=int(X.nnz), n_cells_total=int(n_cells_all),
        n_plays=int(sizes.sum()), n_plays_total=int(n_plays_all),
        density=float(X.nnz / (X.shape[0] * X.shape[1])),
        cell_plays_p50=float(q[0]), cell_plays_p75=float(q[1]),
        cell_plays_p90=float(q[2]), cell_plays_p95=float(q[3]),
        cell_plays_p99=float(q[4]),
        frac_single_play=float(np.mean(sizes == 1)),
    )
    return X, N, V, np.asarray(charts), np.asarray(players), stats


def estimate_noise(N, V) -> dict:
    """由 cell 内合并方差估计"中位数 cell 值"的噪声方差之和。"""
    if V is None:
        return {}
    n = N.data.astype(np.float64)
    multi = n >= 2
    if not multi.any():
        return {}
    dof = float(np.sum(n[multi] - 1.0))
    # 每个 cell 的方差贡献要乘以它的自由度 (n-1)，否则 n>2 的 cell 会被低估
    sigma2 = float(np.sum((n[multi] - 1.0) * V.data[multi]) / dof)
    # n=1: 中位数就是那一次 play，方差 = sigma^2；
    # n>=2: Var(median) ~= (pi/2) * sigma^2 / n。
    c = np.where(n == 1.0, 1.0, MEDIAN_FACTOR / np.maximum(n, 1.0))
    noise_ss = float(sigma2 * np.sum(c))
    return dict(sigma2=sigma2, sigma=float(np.sqrt(sigma2)), dof=dof,
                noise_ss=noise_ss)


# ------------------------------------------------------------------ 分解
def player_center(X) -> tuple[sparse.csr_matrix, np.ndarray]:
    """缺失 cell = 玩家均值，再按玩家中心化（缺失处恰好为 0）。"""
    m = X.shape[1]
    cnt = np.bincount(X.indices, minlength=m).astype(np.float64)
    mu = np.bincount(X.indices, weights=X.data, minlength=m) / np.maximum(cnt, 1.0)
    Xc = X.copy()
    Xc.data = X.data - mu[Xc.indices]
    return Xc, mu


def fit_additive(X, max_iter: int = 100, tol: float = 1e-10):
    """观测 cell 上的加性模型 loss = c + a_p + b_m（交替均值）。

    等价于对观测 cell 做无截距约束的最小二乘；a/b 的规范自由度由 c 吸收。
    返回 a (players), b (charts), c, 迭代数, 最后一步的变化量。
    """
    n, m = X.shape
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(X.indptr))
    cnt_c = np.diff(X.indptr).astype(np.float64)
    cnt_p = np.bincount(X.indices, minlength=m).astype(np.float64)
    c = float(X.data.mean())
    a = np.zeros(m)
    b = np.zeros(n)
    delta = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        b_new = np.bincount(rows, weights=X.data - c - a[X.indices], minlength=n)
        b_new /= np.maximum(cnt_c, 1.0)
        a_new = np.bincount(X.indices, weights=X.data - c - b_new[rows], minlength=m)
        a_new /= np.maximum(cnt_p, 1.0)
        delta = max(float(np.max(np.abs(b_new - b))),
                    float(np.max(np.abs(a_new - a))))
        a, b = a_new, b_new
        if delta < tol:
            break
    # 把残差均值挪进 c，残差在观测 cell 上的均值就是 0
    c += float(np.mean(X.data - c - a[X.indices] - b[rows]))
    return a, b, c, it, delta


def random_svd(M: sparse.csr_matrix, k: int, n_iter: int = 10, seed: int = 0):
    """无 sklearn 的 randomized SVD 兜底；只返回奇异值。"""
    rng = np.random.default_rng(seed)
    n, m = M.shape
    q = min(k + 10, m)
    Q = rng.standard_normal((m, q))
    Y = np.asarray(M @ Q)
    for _ in range(n_iter):
        Y, _ = np.linalg.qr(Y)
        Z = np.asarray(M.T @ Y)
        Z, _ = np.linalg.qr(Z)
        Y = np.asarray(M @ Z)
    Q, _ = np.linalg.qr(Y)
    B = np.asarray(M.T @ Q).T
    s = np.linalg.svd(B, full_matrices=False)[1]
    return s[:k], "randomized"


def compute_spectrum(M: sparse.csr_matrix, args):
    """返回 (singular values 降序, solver)。

    小矩阵（min(n,m) <= --full-max-dim）走精确 Gram 特征分解，拿完整谱；
    大矩阵用 ARPACK svds 取前 --components 个。
    """
    n, m = M.shape
    dim = min(n, m)
    if dim <= args.full_max_dim:
        G = (M.T @ M).toarray() if m <= n else (M @ M.T).toarray()
        ev = np.linalg.eigvalsh(G)[::-1]
        ev = np.clip(ev, 0.0, None)
        return np.sqrt(ev), "exact"

    k = int(args.components) if int(args.components) > 0 else 256
    k = min(k, dim - 1)
    if k < 1:
        raise SystemExit("--components 必须 >= 1（0 = 自动 256）")
    try:
        rng = np.random.default_rng(args.seed)
        v0 = rng.standard_normal(dim)
        _, s, _ = svds(M, k=k, which="LM", tol=args.svd_tol,
                       maxiter=args.svd_maxiter, v0=v0)
        return np.sort(s)[::-1], "svds"
    except Exception as exc:  # ArpackNoConvergence / 内存 / k 太大
        print(f"[warn] svds 失败（{type(exc).__name__}: {exc}），"
              f"改用 randomized SVD", flush=True)
        s, solver = random_svd(M, k, seed=args.seed)
        return np.sort(s)[::-1], solver


def k_for(cum: np.ndarray, thr: float):
    """累计曲线第一次 >= thr 的 k（1-based）；没到则 None。"""
    i = int(np.searchsorted(cum, thr, side="left"))
    return i + 1 if i < len(cum) else None


def analyze_view(name: str, M, total_ss: float, n_obs: int, args,
                 raw_total_ss: float | None = None, additive_ss: float = 0.0,
                 noise: dict | None = None) -> dict:
    """一个口径的完整谱表 + 汇总。"""
    if total_ss <= 0:
        raise SystemExit(f"{name}: total_ss=0（矩阵是常数？）")
    s, solver = compute_spectrum(M, args)
    eig = s ** 2
    cum = np.minimum(np.cumsum(eig), total_ss)
    evr = eig / total_ss
    cum_evr = cum / total_ss
    resid_ss = np.maximum(total_ss - cum, 0.0)
    rmse = np.sqrt(resid_ss / n_obs)

    # 交互口径额外给出"占原始矩阵方差"的累计曲线：加性 + 前 k 个交互
    if name == "interaction" and raw_total_ss:
        cum_of_raw = (additive_ss + cum) / raw_total_ss
    else:
        cum_of_raw = cum_evr

    full = solver == "exact"
    pr = float((eig.sum() ** 2) / np.sum(eig ** 2)) if float(np.sum(eig ** 2)) > 0 else None
    d = dict(
        view=name, solver=solver, n_components=int(len(s)),
        truncated=not full,
        total_ss=float(total_ss),
        top1=float(evr[0]) if len(evr) else 0.0,
        stable_rank=float(total_ss / eig[0]) if eig[0] > 0 else None,
        # participation ratio 只对完整谱有定义；截断谱的尾部没算，不能比较
        participation=pr if full else None,
        participation_topk=pr if not full else None,
        cum_at_k=float(cum_evr[-1]) if len(cum_evr) else 0.0,
        thresholds={f"{t:.0%}": k_for(cum_evr, t) for t in THRESHOLDS},
        thresholds_of_raw={f"{t:.0%}": k_for(cum_of_raw, t) for t in THRESHOLDS},
        grid={str(k): float(cum_evr[k - 1]) for k in GRID if k <= len(cum_evr)},
        # 下面这些只进 CSV，不进 JSON
        _s=s, _eig=eig, _evr=evr, _cum_evr=cum_evr, _cum=cum,
        _resid_ss=resid_ss, _rmse=rmse, _cum_of_raw=cum_of_raw,
    )
    if noise:
        signal_ss = max(total_ss - noise["noise_ss"], 0.0)
        d["signal_ss"] = float(signal_ss)
        d["noise_share"] = float(noise["noise_ss"] / total_ss) if total_ss else None
        d["_cum_signal"] = cum / signal_ss if signal_ss > 0 else np.zeros_like(cum)
    return d


# ------------------------------------------------------------------ 输出
def print_report(path, stats, noise, views, raw_total_ss, additive_ss, additive_info):
    print("\n" + "=" * 78)
    print(f"PCA_analysis  {path}")
    print("=" * 78)
    print(f"[matrix] {stats['n_charts']:,} charts x {stats['n_players']:,} players"
          f"  cells={stats['n_cells']:,} / {stats['n_charts'] * stats['n_players']:,}"
          f" = {stats['density']:.3%}")
    print(f"[cells]  plays={stats['n_plays']:,}  "
          f"p50={stats['cell_plays_p50']:.0f} p75={stats['cell_plays_p75']:.0f} "
          f"p90={stats['cell_plays_p90']:.0f} p95={stats['cell_plays_p95']:.0f} "
          f"p99={stats['cell_plays_p99']:.0f}  "
          f"single-play={stats['frac_single_play']:.1%}")
    if noise:
        print(f"[noise]  pooled within-cell sigma^2={noise['sigma2']:.4f} "
              f"(sigma={noise['sigma']:.4f}, dof={noise['dof']:,.0f})")
        share = f"{noise['noise_ss'] / raw_total_ss:.1%} of raw total SS"
        if "interaction" in views:
            share += (f", {noise['noise_ss'] / views['interaction']['total_ss']:.1%}"
                      f" of interaction SS")
        print(f"[noise]  expected median noise SS={noise['noise_ss']:,.1f}  ({share})")
    print(f"[add]    additive loss=c+a(player)+b(chart): "
          f"SS={additive_ss:,.1f} ({additive_ss / raw_total_ss:.2%} of raw)  "
          f"iters={additive_info[0]} delta={additive_info[1]:.2e}")
    if "raw" in views:
        print(f"[raw]    total_ss={raw_total_ss:,.1f}  "
              f"PC1={views['raw']['top1']:.2%}")
    if "interaction" in views:
        v = views["interaction"]
        print(f"[inter]  total_ss={v['total_ss']:,.1f}  "
              f"PC1={v['top1']:.2%}  "
              f"(interaction SS is {v['total_ss'] / raw_total_ss:.1%} of raw)")

    print("\n[thresholds]  k at which cumulative explained variance reaches ...")
    print(f"  {'view':12s} " + " ".join(f"{t:>7.0%}" for t in THRESHOLDS)
          + "   (fraction of own total_ss)")
    def fmt_k(x, v):
        return f">{v['n_components']}" if x is None else str(x)
    for name, v in views.items():
        vals = [fmt_k(v["thresholds"][f"{t:.0%}"], v) for t in THRESHOLDS]
        print(f"  {name:12s} " + " ".join(f"{x:>7s}" for x in vals)
              + f"   top{v['n_components']}={v['cum_at_k']:.2%}")

    print("\n[grid]  cumulative explained variance (%)")
    header = "  " + "k".rjust(6)
    for name in views:
        header += f"  {name[:12]:>12s}"
    if "interaction" in views:
        header += f"  {'int/of_raw':>12s}"
    print(header)
    for k in GRID:
        if k > min(len(v["_cum_evr"]) for v in views.values()):
            continue
        line = f"  {k:>6d}"
        for name in views:
            line += f"  {views[name]['_cum_evr'][k - 1] * 100:>12.2f}"
        if "interaction" in views:
            line += f"  {views['interaction']['_cum_of_raw'][k - 1] * 100:>12.2f}"
        print(line)

    if noise and "interaction" in views:
        v = views["interaction"]
        sig = max(v["total_ss"] - noise["noise_ss"], 0.0)
        print(f"\n[noise-adjusted] interaction: noise share ~{noise['noise_ss'] / v['total_ss']:.1%}; "
              f"non-noise SS={sig:,.1f}")
        for k in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            if k <= len(v["_cum_signal"]):
                print(f"  k={k:<4d} cum/non-noise = {v['_cum_signal'][k - 1]:.2%}")

    print("\n[note] 零填充 SVD 把缺失 cell 当成 0（raw）或无交互（interaction），")
    print("       所以这里的 EVR 是\"重建观测 cell 的方差\"，不是 held-out 预测增益；")
    print("       稀疏 + 噪声会让谱变平。加权 ALS 的样本内 rank-k 拟合更高但会过拟合，")
    print("       预测性 rank 扫描见 scripts/diag_step2_structure.py（--k-list）。")

    for name, v in views.items():
        pr = (f"{v['participation']:.1f}" if v["participation"] is not None
              else f"{v['participation_topk']:.1f} (top-{v['n_components']} only)")
        print(f"\n[{name}]  solver={v['solver']}  components={v['n_components']}  "
              f"top1={v['top1']:.3%}  stable_rank={v['stable_rank']:.1f}  "
              f"participation={pr}")
        if name == "interaction":
            print(f"  of raw total: k@50%={v['thresholds_of_raw']['50%']} "
                  f"k@80%={v['thresholds_of_raw']['80%']} "
                  f"k@90%={v['thresholds_of_raw']['90%']}")


def write_outputs(out_dir: Path, tag: str, args, stats, noise, views,
                  raw_total_ss, additive_ss, additive_info):
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, v in views.items():
        for i in range(v["n_components"]):
            rows.append(dict(
                view=name, k=i + 1,
                singular_value=v["_s"][i],
                eigenvalue=v["_eig"][i],
                evr=v["_evr"][i],
                cumulative_evr=v["_cum_evr"][i],
                cumulative_evr_of_raw=v["_cum_of_raw"][i],
                cumulative_evr_signal=(v.get("_cum_signal", v["_cum_evr"])[i]
                                       if "signal_ss" in v else np.nan),
                explained_ss=v["_cum"][i],
                residual_ss=v["_resid_ss"][i],
                rmse_observed=v["_rmse"][i],
            ))
    csv_path = out_dir / f"pca_{tag}_spectrum.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.10g")

    def clean(d):
        return {k: v for k, v in d.items() if not k.startswith("_")}

    summary = dict(
        tag=tag, plays=str(args.plays) if args.plays else None,
        args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        matrix=stats, noise=noise,
        additive=dict(ss=additive_ss, r2_of_raw=additive_ss / raw_total_ss,
                      iters=additive_info[0], last_delta=additive_info[1]),
        raw_total_ss=raw_total_ss,
        views={name: clean(v) for name, v in views.items()},
    )
    json_path = out_dir / f"pca_{tag}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    png_path = None
    if not args.no_plot:
        png_path = out_dir / f"pca_{tag}.png"
        try:
            make_plot(tag, stats, views, noise, png_path)
        except Exception as exc:  # 画图失败不应让分析结果丢掉
            print(f"[warn] 画图失败: {type(exc).__name__}: {exc}", flush=True)
            png_path = None
    return csv_path, json_path, png_path


def make_plot(tag, stats, views, noise, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    colors = {"raw": "tab:blue", "interaction": "tab:orange"}
    for name, v in views.items():
        k = np.arange(1, v["n_components"] + 1)
        c = colors.get(name)
        axes[0].plot(k, v["_cum_evr"] * 100, color=c,
                     label=f"{name} (PC1 {v['top1'] * 100:.1f}%)")
        # 精确谱的尾部是数值零；只画 EVR > 1e-6 的部分，避免 scree 出现假悬崖
        keep = v["_evr"] > 1e-6
        if keep.sum() == 0:
            keep = np.ones_like(v["_evr"], dtype=bool)
        axes[1].semilogy(k[keep], np.maximum(v["_evr"][keep] * 100, 1e-6),
                         color=c, label=name)
    if noise and "interaction" in views:
        frac = noise["noise_ss"] / views["interaction"]["total_ss"]
        axes[0].axhline(frac * 100, ls="--", color="tab:orange", alpha=0.6,
                        label=f"interaction noise share ~{frac * 100:.1f}%")
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right")
    axes[0].set_xlabel("number of components k")
    axes[0].set_ylabel("cumulative explained variance (%)")
    axes[0].set_title("cumulative")
    axes[1].set_xlabel("number of components k")
    axes[1].set_ylabel("per-component EVR (%) [log]")
    axes[1].set_title("scree")
    # 精确谱的最后几十个特征值只是数值零；截掉它们，避免 scree 出现假悬崖
    axes[1].set_ylim(bottom=1e-4)
    fig.suptitle(f"PCA of the {stats['n_charts']:,} x {stats['n_players']:,} "
                 f"median-loss matrix ({tag})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(
        description="PCA of the complete (chart, player) median-loss matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--plays", default=None,
                    help="step0 play 表 (.parquet/.csv)；默认 data/processed/step0_{tag}.*")
    ap.add_argument("--tag", default="1k", help="数据集标签")
    ap.add_argument("--out-dir", default=None, help="默认 results/pca_{tag}")
    ap.add_argument("--chart-col", default="beatmap_id")
    ap.add_argument("--player-col", default="player_id")
    ap.add_argument("--value-col", default="loss")
    ap.add_argument("--views", default="both",
                    help="raw,interaction 或 both")
    ap.add_argument("--min-cell-plays", type=int, default=1,
                    help="cell 至少要有几次 play 才进入矩阵（中位数仍按全部 play 算）")
    ap.add_argument("--min-chart-cells", type=int, default=1,
                    help="每张谱面至少要有多少个观测 cell（默认不删）")
    ap.add_argument("--min-player-cells", type=int, default=1,
                    help="每个玩家至少要有多少个观测 cell（默认不删）")
    ap.add_argument("--components", type=int, default=256,
                    help="大矩阵取前几个奇异值；小矩阵自动用精确全谱")
    ap.add_argument("--full-max-dim", type=int, default=2048,
                    help="min(n_charts,n_players) <= 该值时走精确 Gram 全谱")
    ap.add_argument("--svd-tol", type=float, default=1e-6)
    ap.add_argument("--svd-maxiter", type=int, default=5000)
    ap.add_argument("--additive-iter", type=int, default=100)
    ap.add_argument("--additive-tol", type=float, default=1e-10)
    ap.add_argument("--no-noise", dest="noise", action="store_false",
                    help="不估计 cell 中位数的噪声地板（默认会估计）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-output", action="store_true",
                    help="只打印汇总，不写 CSV/JSON/PNG")
    args = ap.parse_args()

    if args.views == "both":
        view_names = list(VIEWS)
    else:
        view_names = [v.strip() for v in args.views.split(",") if v.strip()]
    bad = set(view_names) - set(VIEWS)
    if bad or not view_names:
        raise SystemExit(f"--views 只能取 {VIEWS} 或 both；收到 {args.views!r}")

    t0 = time.time()
    path, df = load_plays(args)
    args.plays = str(path)  # 让 summary 记录实际读的表，而不是 None
    X, N, V, chart_ids, player_ids, stats = build_matrix(df, args)
    noise = estimate_noise(N, V) if args.noise else {}

    t1 = time.time()
    Xc, mu_p = player_center(X)
    raw_total_ss = float(np.sum(Xc.data ** 2))
    a, b, c, it, delta = fit_additive(X, args.additive_iter, args.additive_tol)
    rows = np.repeat(np.arange(X.shape[0], dtype=np.int64), np.diff(X.indptr))
    R = X.copy()
    R.data = X.data - c - a[R.indices] - b[rows]
    int_total_ss = float(np.sum(R.data ** 2))
    additive_ss = max(raw_total_ss - int_total_ss, 0.0)
    print(f"[fit] raw SS={raw_total_ss:,.1f}  interaction SS={int_total_ss:,.1f}  "
          f"additive R2={additive_ss / raw_total_ss:.2%}  ({time.time() - t1:.1f}s)",
          flush=True)

    views = {}
    for name in view_names:
        t2 = time.time()
        if name == "raw":
            views[name] = analyze_view("raw", Xc, raw_total_ss, X.nnz, args,
                                       noise=noise)
        else:
            views[name] = analyze_view("interaction", R, int_total_ss, X.nnz, args,
                                       raw_total_ss=raw_total_ss,
                                       additive_ss=additive_ss, noise=noise)
        print(f"[svd] {name}: solver={views[name]['solver']}  "
              f"components={views[name]['n_components']}  "
              f"({time.time() - t2:.1f}s)", flush=True)

    print_report(path, stats, noise, views, raw_total_ss, additive_ss, (it, delta))

    if args.no_output:
        print(f"\n[done] {time.time() - t0:.1f}s (--no-output: no files written)")
        return
    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / f"pca_{args.tag}"
    csv_path, json_path, png_path = write_outputs(
        out_dir, args.tag, args, stats, noise, views, raw_total_ss, additive_ss,
        (it, delta))
    print(f"\n[done] {time.time() - t0:.1f}s")
    print(f"  wrote {csv_path}")
    print(f"  wrote {json_path}")
    if png_path:
        print(f"  wrote {png_path}")


if __name__ == "__main__":
    main()
