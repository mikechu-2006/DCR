#!/usr/bin/env python3
"""只跑玩家步：逐玩家精确求解 6 维技能，然后报告 train/test RMSE。

    python3 scripts/step2_player_step.py --players 40 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minacalc.data import Charts, Plays, load_plays          # noqa: E402
from minacalc.player_step import init_player, solve_player   # noqa: E402
from minacalc.torch_model import Theta, TorchScorer          # noqa: E402


def groups(pl: Plays):
    order = np.argsort(pl.player_idx, kind="stable")
    bounds = np.searchsorted(pl.player_idx[order], np.arange(len(pl.players) + 1))
    return order, bounds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plays", default="data/processed/step1_1k.parquet")
    ap.add_argument("--charts", default="data/processed/charts_1k.npz")
    ap.add_argument("--out", default="results/player_step")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--players", type=int, default=40, help="只解前 N 个玩家（0=全部）")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--lam", type=float, default=1e-3)
    args = ap.parse_args()

    dev = torch.device(args.device)
    charts = Charts(args.charts)
    tr = load_plays(args.plays, args.charts, split="train")
    te = load_plays(args.plays, args.charts, split="test")
    sc = TorchScorer(charts, device=dev)
    th = Theta().to(dev)

    otr, btr = groups(tr)
    ote, bte = groups(te)
    P = len(tr.players)
    sel = list(range(P if args.players <= 0 else min(args.players, P)))
    mu = torch.full((6,), 20.0, device=dev)

    X = np.full((P, 6), 20.0, dtype=np.float32)
    rows = []
    t0 = time.time()
    for j, p in enumerate(sel):
        idx = otr[btr[p]:btr[p + 1]]
        if len(idx) < 10:
            continue
        cidx = torch.as_tensor(tr.chart_idx[idx], dtype=torch.long)
        loss = torch.as_tensor(tr.loss[idx], dtype=torch.float32)
        x0 = init_player(sc, cidx, loss, th, dev, mu)
        x, info = solve_player(sc, idx, x0, cidx, loss, th, dev,
                               iters=args.iters, lam=args.lam)
        X[p] = x.cpu().numpy()
        rows.append({"player": int(tr.players[p]), "n": info["n"],
                     "rmse_tr": info["rmse"], "x": X[p].tolist(),
                     "active": info["active"].tolist()})
        if (j + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {j+1}/{len(sel)}  {el:.0f}s  "
                  f"({el/(j+1):.1f}s/player)  last rmse {info['rmse']:.3f}", flush=True)

    # 评估：这些玩家的 train/test 全部 play
    def rmse(pl, ol, bl, which):
        ss, n = 0.0, 0
        with torch.no_grad():
            for p in which:
                idx = ol[bl[p]:bl[p + 1]]
                if not len(idx):
                    continue
                cidx = torch.as_tensor(pl.chart_idx[idx], dtype=torch.long, device=dev)
                loss = torch.as_tensor(pl.loss[idx], dtype=torch.float32, device=dev)
                x = torch.as_tensor(X[p], dtype=torch.float32, device=dev).unsqueeze(0)
                pidx = torch.zeros(len(idx), dtype=torch.long, device=dev)
                out = sc.forward(x, pidx, cidx, th, L=int(sc.n[cidx].max()))
                tot = (out["S"] + out["T"]).clamp_min(1e-6)
                r = torch.log(tot) - torch.log(out["Max"].clamp_min(1e-9)) - loss
                ss += float((r * r).sum()); n += r.numel()
        return float(np.sqrt(ss / max(n, 1))), n

    rtr, ntr = rmse(tr, otr, btr, sel)
    rte, nte = rmse(te, ote, bte, sel)
    print(f"\nplayers {len(sel)}   train RMSE {rtr:.4f} (n={ntr:,})   "
          f"test RMSE {rte:.4f} (n={nte:,})", flush=True)
    print(f"X mean {X[sel].mean(0).round(2)}  Xstd {X[sel].std(0).round(2)}  "
          f"min {X[sel].min(0).round(1)}  max {X[sel].max(0).round(1)}", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", X)
    (out / "report.json").write_text(json.dumps(
        {"args": vars(args), "players": rows, "rmse_train": rtr,
         "rmse_test": rte, "n_train": ntr, "n_test": nte,
         "players_sel": [int(tr.players[p]) for p in sel]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
