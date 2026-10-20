#!/usr/bin/env python3
"""P2/P3: 块坐标下降拟合玩家 6 维技能 + 预测器参数 theta。

    # 冒烟（CPU，小样本）
    python3 scripts/step2_fit_skills.py --limit-plays 4000 --rounds 2 --device cpu

    # 全量（GPU）
    python3 scripts/step2_fit_skills.py --device cuda --rounds 3 --out results/fit_1k

目标（docs/step2_fit_skills.md §4.2）：

    r = ln(1 - score) - ln(1 - obs) = ln((S + T)/Max) - loss_obs
    L = mean(r^2) + lam_x * mean((X - mu)^2) + lam_theta * mean(du^2)

Phase A 只更新 X（Adam，minibatch）；Phase B1 只更新 theta.du1（c 6 个 + delta +
jack 幅度/斜率），Phase B2 只更新 theta.du2（区间疲劳 6 常数），都用 L-BFGS。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minacalc.data import Charts, Plays, load_plays      # noqa: E402
from minacalc.torch_model import Theta, TorchScorer  # noqa: E402


def make_batches(plays: Plays, charts: Charts, batch_plays: int,
                 batch_itv: int = 0, batch_ev: int = 0):
    """按谱面长度排序后分块，同时限制三个预算。

    * batch_plays —— 一块最多多少条 play
    * batch_itv   —— 这块的区间总数（控制 stamina 扫描的长度 L）
    * batch_ev    —— 这块的 **N x E** 元素数（E = 这块内最大 jack 事件数）。
      jack 的递推是逐事件串行的，长度由块内最长的那张谱决定；不限制的话，
      一张 3.4 万事件的谱会把整块拖成 (N,2,34626) 的巨型张量。
    """
    cidx_all = plays.chart_idx
    n_itv = charts.n[cidx_all]
    n_ev = ((charts.jack_off[0, cidx_all + 1] - charts.jack_off[0, cidx_all])
            + (charts.jack_off[1, cidx_all + 1] - charts.jack_off[1, cidx_all]))
    order = np.argsort(n_itv, kind="stable")
    out, cur = [], []
    cur_itv, cur_maxev = 0, 0
    for i in order:
        n = int(n_itv[i]); e = int(n_ev[i])
        too_big = False
        if cur and batch_ev:
            too_big = (len(cur) + 1) * max(cur_maxev, e) > batch_ev
        if cur and (len(cur) >= batch_plays
                    or (batch_itv and cur_itv + n > batch_itv) or too_big):
            out.append(cur)
            cur, cur_itv, cur_maxev = [], 0, 0
        cur.append(int(i))
        cur_itv += n
        cur_maxev = max(cur_maxev, e)
    if cur:
        out.append(cur)
    res = []
    for idx in out:
        idx = np.asarray(idx)
        cidx = plays.chart_idx[idx]
        res.append((torch.as_tensor(plays.player_idx[idx], dtype=torch.long),
                    torch.as_tensor(cidx, dtype=torch.long),
                    torch.as_tensor(plays.loss[idx], dtype=torch.float32),
                    int(charts.n[cidx].max())))
    return res


def residual(out, loss_obs):
    tot = (out["S"] + out["T"]).clamp_min(1e-6)
    return torch.log(tot) - torch.log(out["Max"].clamp_min(1e-9)) - loss_obs


def evaluate(sc, X, batches, theta, device, max_batches=0):
    n, ss = 0, 0.0
    with torch.no_grad():
        for bi, (pidx, cidx, loss, L) in enumerate(batches):
            if max_batches and bi >= max_batches:
                break
            out = sc.forward(X, pidx.to(device), cidx.to(device), theta, L=L)
            r = residual(out, loss.to(device))
            ss += float((r * r).sum())
            n += r.numel()
    return math.sqrt(ss / max(n, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plays", default="data/processed/step1_1k.parquet")
    ap.add_argument("--charts", default="data/processed/charts_1k.npz")
    ap.add_argument("--out", default="results/fit_1k")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-plays", type=int, default=4096)
    ap.add_argument("--batch-itv", type=int, default=3_000_000,
                    help="每块允许的总区间数（控制显存/内存）")
    ap.add_argument("--batch-ev", type=int, default=30_000_000,
                    help="每块允许的 N x (最大 jack 事件数)，控制 jack 张量规模")
    ap.add_argument("--limit-plays", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--epochs-a", type=int, default=3)
    ap.add_argument("--iters-b1", type=int, default=30)
    ap.add_argument("--iters-b2", type=int, default=15)
    ap.add_argument("--lr-x", type=float, default=1e-2)
    ap.add_argument("--lam-x", type=float, default=1e-3)
    ap.add_argument("--lam-theta", type=float, default=1.0,
                    help="theta 的 ridge（相对 MSE，1.0 约等于 1%% 的 MSE）")
    ap.add_argument("--init-x", type=float, default=20.0)
    ap.add_argument("--freeze-theta", action="store_true")
    ap.add_argument("--eval-test-batches", type=int, default=60,
                    help="评估 test 时最多用几个 batch（0=全部）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    charts = Charts(args.charts)
    tr = load_plays(args.plays, args.charts, split="train")
    te = load_plays(args.plays, args.charts, split="test")
    if args.limit_plays:
        keep = rng.permutation(len(tr.loss))[:args.limit_plays]
        tr = Plays(tr.player_idx[keep], tr.chart_idx[keep], tr.loss[keep],
                   tr.split[keep], tr.players)
    print(f"train {len(tr.loss):,}  test {len(te.loss):,}  "
          f"players {len(tr.players)}  charts {charts.n_charts:,}  device {device}",
          flush=True)

    sc = TorchScorer(charts, device=device)
    theta = Theta().to(device)
    P = len(tr.players)
    X = torch.nn.Parameter(torch.full((P, 6), float(args.init_x), device=device))
    mu = X.detach().clone()

    tr_batches = make_batches(tr, charts, args.batch_plays, args.batch_itv, args.batch_ev)
    te_batches = make_batches(te, charts, args.batch_plays, args.batch_itv, args.batch_ev)
    print(f"train batches {len(tr_batches)}  test batches {len(te_batches)}", flush=True)

    opt_x = torch.optim.Adam([X], lr=args.lr_x)

    def theta_objective():
        tot = 0.0
        n = 0
        for pidx, cidx, loss, L in tr_batches:
            out = sc.forward(X.detach(), pidx.to(device), cidx.to(device), theta, L=L)
            r = residual(out, loss.to(device))
            tot = tot + (r * r).sum()
            n += r.numel()
        return tot / n + args.lam_theta * theta.prior_penalty()

    history = []
    for rd in range(1, args.rounds + 1):
        t0 = time.time()
        # ---------------- Phase A ----------------
        for ep in range(args.epochs_a):
            for bi in rng.permutation(len(tr_batches)):
                pidx, cidx, loss, L = tr_batches[bi]
                opt_x.zero_grad(set_to_none=True)
                out = sc.forward(X, pidx.to(device), cidx.to(device), theta, L=L)
                r = residual(out, loss.to(device))
                # 注意：先验项也必须除以 n_batches，否则它被放大 168 倍，
                # 会把 X 死死钉在初值 20 上（实测 Xstd 只有 1.2）。
                obj = ((r * r).mean()
                       + args.lam_x * ((X - mu) ** 2).mean()) / len(tr_batches)
                obj.backward()
                torch.nn.utils.clip_grad_norm_([X], 10.0)
                opt_x.step()
        rmse_tr = evaluate(sc, X.detach(), tr_batches, theta, device)
        rmse_te = evaluate(sc, X.detach(), te_batches, theta, device,
                           args.eval_test_batches)
        print(f"[r{rd}] A: train {rmse_tr:.5f} test {rmse_te:.5f} "
              f"Xmean {X.detach().mean(0).cpu().numpy().round(2)} "
              f"Xstd {X.detach().std(0).cpu().numpy().round(2)} ({time.time()-t0:.0f}s)",
              flush=True)

        # ---------------- Phase B ----------------
        if not args.freeze_theta:
            for tag, par, iters in (("B1", theta.du1, args.iters_b1),
                                    ("B2", theta.du2, args.iters_b2)):
                opt = torch.optim.LBFGS([par], lr=1.0, max_iter=iters,
                                        history_size=10,
                                        line_search_fn="strong_wolfe",
                                        tolerance_grad=1e-7,
                                        tolerance_change=1e-10)

                def closure():
                    opt.zero_grad(set_to_none=True)
                    obj = theta_objective()
                    obj.backward()
                    return obj

                before = float(theta_objective().detach())
                opt.step(closure)
                after = float(theta_objective().detach())
                print(f"[r{rd}] {tag}: obj {before:.6f} -> {after:.6f}", flush=True)
            rmse_tr = evaluate(sc, X.detach(), tr_batches, theta, device)
            rmse_te = evaluate(sc, X.detach(), te_batches, theta, device,
                               args.eval_test_batches)
            print(f"[r{rd}] B: train {rmse_tr:.5f} test {rmse_te:.5f} "
                  f"theta {np.round(theta.vec().detach().cpu().numpy(), 4)}", flush=True)

        history.append({"round": rd, "rmse_train": rmse_tr, "rmse_test": rmse_te,
                        "theta": theta.vec().detach().cpu().numpy().tolist(),
                        "du1": theta.du1.detach().cpu().numpy().tolist(),
                        "du2": theta.du2.detach().cpu().numpy().tolist()})
        np.save(out_dir / f"X_iter{rd}.npy", X.detach().cpu().numpy())
        (out_dir / "history.json").write_text(json.dumps(
            {"args": vars(args), "history": history,
             "players": tr.players.tolist()}, indent=1))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
