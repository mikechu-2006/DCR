"""P0: 谱面张量（扁平存储）与训练表。

谱面用**变长扁平**存储，不 padding：20,655 张谱共 6.13M 个 0.5s 区间（median 250，
p99 816，最长 8671），padding 到最长要 8.6 GB，扁平只要 294 MB + 49 MB。

    diffs   (6, 2, T)        float32   按谱面顺序首尾相接
    pts     (2, T)           float32
    off     (Nc+1,)          int64     第 i 张谱占 [off[i], off[i+1])
    jack_*  (2, J)           事件按谱面顺序相接；joff[h] 形状 (Nc+1,)
             jack_itv = 该事件落在谱内的第几个区间（谱内相对下标）

按 batch 取数时用索引矩阵一次性 gather 成 (N, 6, 2, L)。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_SUFFIX = "_itv_1.000_0.930_v2.npz"
DIMS = ("stream", "js", "hs", "jackspeed", "cj", "tech")


def cache_key(osu_path) -> str:
    return hashlib.sha1(Path(osu_path).read_bytes()).hexdigest()[:16] + CACHE_SUFFIX


def build(parquet: str = "data/processed/step1_1k.parquet",
          out: str = "data/processed/charts_1k.npz",
          osu_dir: str = "data/osu_cache",
          cache_dir: str = ".minacalc_cache",
          limit: int = 0, verbose: bool = True) -> dict:
    """把 .minacalc_cache/*.npz 拼成扁平张量 + 变长偏移表。"""
    ids = sorted(pd.read_parquet(parquet, columns=["beatmap_id"])
                 ["beatmap_id"].unique().tolist())
    if limit:
        ids = ids[:limit]

    ns, miss = [], []
    for b in ids:
        f = Path(cache_dir) / cache_key(Path(osu_dir) / f"{b}.osu")
        if not f.exists():
            miss.append(int(b))
            continue
        with np.load(f, allow_pickle=False) as z:
            ns.append(int(z["diffs"].shape[-1]))
    if miss:
        raise SystemExit(f"{len(miss)} charts missing from {cache_dir}: {miss[:5]}")
    ns = np.asarray(ns, dtype=np.int64)
    keep = np.ones(len(ids), dtype=bool)

    off = np.zeros(len(ids) + 1, dtype=np.int64)
    np.cumsum(ns, out=off[1:])
    T = int(off[-1])
    diffs = np.empty((6, 2, T), dtype=np.float32)
    pts = np.empty((2, T), dtype=np.float32)
    jack_off = np.zeros((2, len(ids) + 1), dtype=np.int64)
    jitv_parts: list[list[np.ndarray]] = [[], []]
    jy_parts: list[list[np.ndarray]] = [[], []]

    for i, b in enumerate(ids):
        f = Path(cache_dir) / cache_key(Path(osu_dir) / f"{b}.osu")
        with np.load(f, allow_pickle=False) as z:
            a, c = off[i], off[i + 1]
            diffs[:, :, a:c] = z["diffs"].astype(np.float32)
            pts[:, a:c] = z["pts"].astype(np.float32)
            for h in range(2):
                itv = z[f"jack_itv{h}"].astype(np.int64)
                y = z[f"jack_y{h}"].astype(np.float32)
                # npz 里存的是谱内相对区间下标，但要防越界
                itv = np.clip(itv, 0, ns[i] - 1)
                jitv_parts[h].append(itv.astype(np.int32))
                jy_parts[h].append(y)
                jack_off[h, i + 1] = jack_off[h, i] + len(itv)
        if verbose and (i + 1) % 5000 == 0:
            print(f"  packed {i + 1}/{len(ids)}", flush=True)

    # 两手的事件拼在同一个数组里（先手 0 后手 1），所以手 1 的偏移要整体平移
    jack_off[1] += jack_off[0, -1]

    jitv = np.concatenate([np.concatenate(p) if p else np.zeros(0, np.int32)
                           for p in jitv_parts]) if len(ids) else np.zeros(0, np.int32)
    jy = np.concatenate([np.concatenate(q) if q else np.zeros(0, np.float32)
                         for q in jy_parts]) if len(ids) else np.zeros(0, np.float32)

    meta = {"n_charts": int(len(ids)), "n_itv_total": int(T),
            "n_jack_total": [int(jack_off[h, -1]) for h in range(2)],
            "mean_itv": float(ns.mean()), "max_itv": int(ns.max())}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, beatmap=np.asarray(ids, dtype=np.int64), n=ns, off=off,
             diffs=diffs, pts=pts, jack_off=jack_off, jack_itv=jitv, jack_y=jy,
             meta=np.array(json.dumps(meta)))
    if verbose:
        print(out, meta)
    return meta


@dataclass
class Plays:
    player_idx: np.ndarray      # (N,) int32  到 players 的序号
    chart_idx: np.ndarray       # (N,) int32  到 charts 的序号
    loss: np.ndarray            # (N,) float32
    split: np.ndarray           # (N,) <U7
    players: np.ndarray         # (P,) int64  原始 player_id


def load_plays(parquet: str = "data/processed/step1_1k.parquet",
               charts_npz: str = "data/processed/charts_1k.npz",
               split: str | None = "train") -> Plays:
    """行 -> (player_idx, chart_idx)。所有 split 共用同一套 player/chart 编号。"""
    with np.load(charts_npz, allow_pickle=False) as z:
        beatmaps = z["beatmap"]
    df = pd.read_parquet(parquet)
    players = np.asarray(sorted(df.player_id.unique().tolist()), dtype=np.int64)
    pmap = {int(v): i for i, v in enumerate(players)}
    cmap = {int(v): i for i, v in enumerate(beatmaps.tolist())}
    df = df[df.beatmap_id.isin(cmap)]
    if split is not None:
        df = df[df.split == split]
    return Plays(player_idx=np.array([pmap[int(v)] for v in df.player_id], dtype=np.int32),
                 chart_idx=np.array([cmap[int(v)] for v in df.beatmap_id], dtype=np.int32),
                 loss=df.loss.to_numpy(dtype=np.float32),
                 split=df.split.to_numpy(),
                 players=players)


class Charts:
    """扁平谱面张量的只读视图。"""

    def __init__(self, path: str = "data/processed/charts_1k.npz"):
        z = np.load(path, allow_pickle=False)
        self.beatmap = z["beatmap"]
        self.n = z["n"]
        self.off = z["off"]
        self.diffs = z["diffs"]          # (6, 2, T)
        self.pts = z["pts"]              # (2, T)
        self.jack_off = z["jack_off"]    # (2, Nc+1)
        self.jack_itv = z["jack_itv"]    # (J,)
        self.jack_y = z["jack_y"]        # (J,)
        self.meta = json.loads(str(z["meta"].item()))
        self.n_charts = int(len(self.beatmap))

    def gather(self, idx: np.ndarray, L: int | None = None):
        """-> diffs (N,6,2,L), pts (N,2,L), mask (N,L)。超长截断由调用方保证。"""
        n = self.n[idx]
        L = int(n.max()) if L is None else int(L)
        if L < n.max():
            raise ValueError(f"L={L} < max n={int(n.max())} in this batch")
        ar = np.arange(L, dtype=np.int64)
        flat = self.off[idx][:, None] + ar[None, :]
        mask = ar[None, :] < n[:, None]
        flat = np.minimum(flat, self.off[-1] - 1)
        d = self.diffs[:, :, flat].transpose(2, 0, 1, 3)     # (N,6,2,L)
        p = self.pts[:, flat].transpose(1, 0, 2)             # (N,2,L)
        d = d * mask[:, None, None, :]
        p = p * mask[:, None, :]
        return d, p, mask
