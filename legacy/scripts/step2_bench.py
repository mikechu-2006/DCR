#!/usr/bin/env python3
"""量一次前向/反向的真实成本，用来更新算力估算。

    python3 scripts/step2_bench.py --device cuda --n-batches 6
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minacalc.data import Charts, Plays, load_plays          # noqa: E402
from minacalc.torch_model import Theta, TorchScorer          # noqa: E402
from scripts.step2_fit_skills import make_batches, residual  # noqa: E402


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plays", default="data/processed/step1_1k.parquet")
    ap.add_argument("--charts", default="data/processed/charts_1k.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-plays", type=int, default=4096)
    ap.add_argument("--batch-itv", type=int, default=3_000_000)
    ap.add_argument("--batch-ev", type=int, default=30_000_000)
    ap.add_argument("--n-batches", type=int, default=6)
    ap.add_argument("--sample-plays", type=int, default=120_000)
    args = ap.parse_args()

    dev = torch.device(args.device)
    charts = Charts(args.charts)
    tr = load_plays(args.plays, args.charts, split="train")
    rng = np.random.default_rng(0)
    if args.sample_plays and len(tr.loss) > args.sample_plays:
        keep = np.sort(rng.permutation(len(tr.loss))[:args.sample_plays])
        tr = Plays(tr.player_idx[keep], tr.chart_idx[keep], tr.loss[keep],
                   tr.split[keep], tr.players)
    batches = make_batches(tr, charts, args.batch_plays, args.batch_itv, args.batch_ev)
    print(f"plays {len(tr.loss):,}  batches {len(batches)}  device {dev}", flush=True)

    sc = TorchScorer(charts, device=dev)
    th = Theta().to(dev)
    X = torch.nn.Parameter(torch.full((len(tr.players), 6), 20.0, device=dev))

    # batches 按谱长排序，取开头几块会严重低估（L 最小）。均匀抽样才有代表性。
    idx = np.linspace(0, len(batches) - 1, args.n_batches).astype(int)
    sel = [batches[i] for i in idx]
    print("picked batches:", [int(b[0].numel()) for b in sel], flush=True)
    t_fwd = t_bwd = 0.0
    n_play = 0
    for pidx, cidx, loss, L in sel:
        pidx, cidx, loss = pidx.to(dev), cidx.to(dev), loss.to(dev)
        for _ in range(2):                                   # warmup
            out = sc.forward(X.detach(), pidx, cidx, th, L=L)
        sync(dev); t0 = time.time()
        out = sc.forward(X.detach(), pidx, cidx, th, L=L)
        sync(dev); t_fwd += time.time() - t0

        X.grad = None
        sync(dev); t0 = time.time()
        out = sc.forward(X, pidx, cidx, th, L=L)
        r = residual(out, loss)
        (r * r).mean().backward()
        sync(dev); t_bwd += time.time() - t0
        n_play += pidx.numel()
        print(f"  batch N={pidx.numel():5d} L={L:5d}  cum fwd {t_fwd:6.2f}s  cum bwd {t_bwd:6.2f}s",
              flush=True)

    print(f"\n== per-batch ==\n  fwd     {t_fwd/len(sel):6.3f} s"
          f"\n  fwd+bwd {t_bwd/len(sel):6.3f} s"
          f"\n  per-play fwd {(t_fwd/n_play)*1000:6.3f} ms")

    # 全零列占比（跳零列的收益来源）
    pidx, cidx, loss, L = sel[len(sel)//2]
    d, pts, mask = sc.gather(cidx.to(dev), L)
    x = X.detach()[pidx.to(dev)]
    d = d * torch.nn.functional.pad(d, (0, 0))  # noqa
    print(f"\n== 100-player step ==", flush=True)
    po = np.argsort(tr.player_idx, kind="stable")
    bounds = np.searchsorted(tr.player_idx[po], np.arange(len(tr.players)+1))
    rng2 = np.random.default_rng(1)
    tot_f = 0.0; tot_play = 0
    for rep in range(3):
        pick = np.sort(rng2.choice(len(tr.players), size=min(100, len(tr.players)),
                                   replace=False))
        keep = np.concatenate([po[bounds[p]:bounds[p+1]] for p in pick])
        sub = Plays(tr.player_idx[keep], tr.chart_idx[keep], tr.loss[keep],
                    tr.split[keep], tr.players)
        bs = make_batches(sub, charts, args.batch_plays, args.batch_itv, args.batch_ev)
        nn = 0
        for pi, ci2, lo2, L2 in bs:
            pi, ci2 = pi.to(dev), ci2.to(dev)
            with torch.no_grad():
                sc.forward(X.detach(), pi, ci2, th, L=L2)
            sync(dev); t0 = time.time()
            with torch.no_grad():
                sc.forward(X.detach(), pi, ci2, th, L=L2)
            sync(dev)
            tot_f += time.time() - t0; nn += pi.numel()
        tot_play += nn
    print(f"  100 players = {tot_play//3:,} plays   一步 fwd {tot_f/3:6.2f} s"
          f"   一次全玩家(9 步) ≈ {tot_f/3*9/60:.2f} min", flush=True)
    # 全量外推
    n_tr = 671105
    tot_fwd = (t_fwd / n_play) * n_tr
    tot_bwd = (t_bwd / n_play) * n_tr
    print(f"\n== full train (671,105 plays) ==\n"
          f"  one epoch forward      {tot_fwd:7.1f} s\n"
          f"  one epoch fwd+backward {tot_bwd:7.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
