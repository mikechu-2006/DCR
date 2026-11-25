#!/usr/bin/env python3
"""E1_fix F4: frozen D2V, K4 vs all-window 2x2 control.

The encoder is frozen and its per-window features are cached once.  Only the
shared ChartHead is trained, so the four conditions differ *only* in the
training/evaluation window policy:

    c0: train K=4 random windows, evaluate first 24 windows (current arm)
    c1: train all windows,           evaluate all windows
    c2: train K=4 random windows,   evaluate all windows
    c3: train all windows,          evaluate first 24 windows

The pair sampler and the 1M pair-view budget are shared across conditions.
This is a head-only reimplementation of the frozen arm, not a bitwise replay
of the original run (which used the old BT loss and early stopping).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data2vec_main"))

from d2v_main.charts import ChartStore, window_starts  # noqa: E402
from d2v_main.config import load_config  # noqa: E402
from d2v_main.model import Data2VecModel  # noqa: E402
from d2v_main.utils import resolve_device, seed_all  # noqa: E402
from scripts.step2_chart_encoder import (  # noqa: E402
    ChartCache, ChartHead, bt_loss, pair_targets, sample_rows, val_nll,
)

CONDITIONS = {
    "c0": ("k4", 24),
    "c1": ("all", 0),
    "c2": ("k4", 0),
    "c3": ("all", 24),
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-table", required=True)
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--osu-dir", default="data/osu_cache")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="data2vec_main/configs/audio_style_base.yaml")
    ap.add_argument("--window-sec", type=float, default=16.0)
    ap.add_argument("--frame-hz", type=int, default=100)
    ap.add_argument("--sigma-ms", type=float, default=8.0)
    ap.add_argument("--min-valid-sec", type=float, default=1.0)
    ap.add_argument("--players-per-batch", type=int, default=4)
    ap.add_argument("--charts-per-player", type=int, default=6)
    ap.add_argument("--windows-per-chart", type=int, default=4)
    ap.add_argument("--pair-views", type=int, default=1_000_000)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--wd-head", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--feature-cache", default=None)
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--conditions", default="c0,c1,c2,c3")
    ap.add_argument("--out-dir", required=True)
    return ap.parse_args(argv)


def load_split(args):
    df = pd.read_parquet(args.split_table)
    if args.split_col not in df.columns:
        raise SystemExit(f"{args.split_table}: no {args.split_col} column")
    for name in ("train", "val", "test"):
        if name not in set(df[args.split_col].unique()):
            raise SystemExit(f"{args.split_table}: no {name} rows")
    for name in ("train", "val", "test"):
        part = df[df[args.split_col] == name]
        if "label_ok" in part.columns:
            part = part[part.label_ok]
        if part.empty:
            raise SystemExit(f"{args.split_table}: empty {name} split")
        if name == "train":
            train = part.copy()
        elif name == "val":
            val = part.copy()
        else:
            test = part.copy()
    return train, val, test


@torch.inference_mode()
def build_features(model, cache, chart_ids, device, batch_windows=16):
    offs = [0]
    wvs, vos, nfs = [], [], []
    for k, c in enumerate(chart_ids):
        _, nf = cache.get(c)
        starts = window_starts(nf, cache.window_frames)
        if not starts:
            starts = [0]
        chart_wv, chart_vo = [], []
        for j in range(0, len(starts), batch_windows):
            x, v = cache.build(c, starts[j:j + batch_windows])
            x = torch.from_numpy(x).to(device)
            v = torch.from_numpy(v).to(device)
            t_out = (x.size(2) + 1) // 2
            vo = (v + 1) // 2
            vm = torch.arange(t_out, device=device)[None, :] < vo[:, None]
            f, _ = model.forward_student(x, vm, None)
            f = torch.nan_to_num(f.float(), nan=0.0, posinf=1e4, neginf=-1e4)
            w = vm.float().unsqueeze(-1)
            wv = (f * w).sum(1) / w.sum(1).clamp_min(1.0)
            chart_wv.append(wv.cpu().numpy())
            chart_vo.append(vo.cpu().numpy())
        wv = np.concatenate(chart_wv, axis=0).astype(np.float32)
        vo = np.concatenate(chart_vo, axis=0).astype(np.int32)
        wvs.append(wv); vos.append(vo); nfs.append(int(nf))
        offs.append(offs[-1] + len(wv))
        if (k + 1) % 500 == 0:
            print(f"[features] {k+1}/{len(chart_ids)} charts", flush=True)
    return {"chart_ids": np.asarray(chart_ids, np.int64),
            "offsets": np.asarray(offs, np.int64),
            "wv": np.concatenate(wvs, axis=0),
            "vo": np.concatenate(vos, axis=0),
            "n_frames": np.asarray(nfs, np.int32)}


def save_features(feat, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez(f, **feat)
    tmp.replace(path)


def load_features(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in ("chart_ids", "offsets", "wv", "vo", "n_frames")}
    if "meta" in z.files:
        out["meta"] = z["meta"]
    return out


def chart_vector(feat, pos, policy, rng=None, k=4, max_windows=0):
    a, b = int(feat["offsets"][pos]), int(feat["offsets"][pos + 1])
    n = b - a
    if n <= 0:
        return np.zeros(feat["wv"].shape[1], np.float32)
    if policy == "k4":
        take = min(k, n)
        idx = rng.choice(n, size=take, replace=False) if rng is not None \
            else np.arange(take)
    else:
        idx = np.arange(n)
    if max_windows and max_windows > 0:
        idx = idx[:max_windows]
    wv = feat["wv"][a:a + n][idx].astype(np.float64)
    vo = feat["vo"][a:a + n][idx].astype(np.float64)
    z = (wv * vo[:, None]).sum(0) / max(1.0, vo.sum())
    z = z / max(1e-12, np.linalg.norm(z))
    return z.astype(np.float32)


def dmap_for(feat, chart_ids, policy, rng, args, max_windows,
             head, device):
    """Chart difficulty d = head(pooled vector) for evaluation."""
    pos = {int(c): i for i, c in enumerate(feat["chart_ids"])}
    zs = [chart_vector(feat, pos[int(c)], policy, rng,
                       args.windows_per_chart, max_windows) for c in chart_ids]
    if not zs:
        return {}
    was = head.training
    head.eval()
    with torch.no_grad():
        z = torch.from_numpy(np.stack(zs)).to(device)
        d = head(z).cpu().numpy()
    head.train(was)
    return {int(c): float(v) for c, v in zip(chart_ids, d)}


def train_condition(args, feat, head, train, val, policy, eval_cap, device):
    rows, players = [], {}
    for _, r in train.iterrows():
        row = {"player": int(r["player_id"]), "chart": int(r["beatmap_id"]),
               "loss": float(r["loss"])}
        rows.append(row)
        players.setdefault(row["player"], []).append(row)
    if not any(len(v) >= 2 for v in players.values()):
        raise SystemExit("no train player has >=2 charts")
    pair_rng = np.random.default_rng(args.seed + 11)
    window_rng = np.random.default_rng(args.seed + 29)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr_head,
                            weight_decay=args.wd_head)
    pos = {int(c): i for i, c in enumerate(feat["chart_ids"])}
    pair_views = 0
    step = 0
    head.train()
    while pair_views < args.pair_views:
        batch_rows = sample_rows(players, SimpleNamespace(
            players_per_batch=args.players_per_batch,
            charts_per_player=args.charts_per_player), pair_rng)
        chart_ids = sorted({int(r["chart"]) for r in batch_rows})
        zs = []
        for c in chart_ids:
            zs.append(chart_vector(feat, pos[c], policy, window_rng,
                                   args.windows_per_chart, 0))
        z = torch.from_numpy(np.stack(zs)).to(device)
        d = head(z)
        pi_row, pj_row, py = pair_targets(batch_rows)
        row_chart = torch.tensor([chart_ids.index(int(r["chart"]))
                                  for r in batch_rows], dtype=torch.long)
        pi = row_chart[pi_row].to(device)
        pj = row_chart[pj_row].to(device)
        y = py.to(device)
        keep = pi != pj
        if int(keep.sum()) == 0:
            continue
        loss = bt_loss(d, pi[keep], pj[keep], y[keep])
        if not torch.isfinite(loss):
            print(f"[control] non-finite loss at step {step}", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(list(head.parameters()), 1.0)
        if not torch.isfinite(gn):
            print(f"[control] non-finite grad at step {step}", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        step += 1
        pair_views += int(keep.sum())
        if args.eval_every and step % args.eval_every == 0:
            vd = dmap_for(feat, sorted(val.beatmap_id.unique().tolist()),
                          "all", np.random.default_rng(args.seed + 99),
                          args, eval_cap, head, device)
            print(f"[control] {policy}/cap{eval_cap} step={step} "
                  f"pairs={pair_views} val_nll={val_nll(val, vd):.5f}", flush=True)
    return head, {"steps": step, "pair_views": pair_views}


def main(argv=None) -> int:
    args = parse_args(argv)
    seed_all(args.seed)
    device = resolve_device(args.device)
    train, val, test = load_split(args)
    store = ChartStore(args.osu_dir,
                       ids=sorted(pd.concat([train, val, test])
                                  .beatmap_id.unique().tolist()))
    cache = ChartCache(store, args.frame_hz,
                       int(round(args.window_sec * args.frame_hz)),
                       args.sigma_ms, int(round(args.min_valid_sec * args.frame_hz)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(vars(args), indent=1, ensure_ascii=False) + "\n")
    cache_path = Path(args.feature_cache) if args.feature_cache \
        else out_dir / "features.npz"
    expected_meta = np.asarray([args.window_sec, args.frame_hz,
                                args.sigma_ms, args.min_valid_sec], np.float64)
    if cache_path.exists() and not args.rebuild_features:
        feat = load_features(cache_path)
        if "meta" in feat and not np.allclose(feat["meta"], expected_meta):
            raise SystemExit(
                f"{cache_path}: feature cache meta {feat['meta'].tolist()} != "
                f"requested {expected_meta.tolist()}; rerun with --rebuild-features")
        print(f"[features] loaded {cache_path} ({len(feat['chart_ids'])} charts)",
              flush=True)
    else:
        cfg = load_config(args.config)
        cfg["data"]["window_sec"] = float(args.window_sec)
        cfg["data"]["frame_hz"] = int(args.frame_hz)
        cfg["data"]["gaussian_sigma_ms"] = float(args.sigma_ms)
        model = Data2VecModel(cfg).to(device).eval()
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck.get("model", ck), strict=True)
        model.requires_grad_(False)
        ids = sorted(pd.concat([train, val, test])
                     .beatmap_id.unique().tolist())
        feat = build_features(model, cache, ids, device)
        feat["meta"] = expected_meta
        save_features(feat, cache_path)
        print(f"[features] wrote {cache_path} ({len(ids)} charts)", flush=True)

    conds = [x.strip() for x in args.conditions.split(",") if x.strip()]
    bad = [c for c in conds if c not in CONDITIONS]
    if bad:
        raise SystemExit(f"unknown conditions: {bad}")
    all_metrics = {}
    for cond in conds:
        policy, eval_cap = CONDITIONS[cond]
        print(f"[control] {cond}: train={policy} eval_cap={eval_cap}", flush=True)
        seed_all(args.seed)
        head = ChartHead(int(feat["wv"].shape[1]), args.hid, args.dropout).to(device)
        head, info = train_condition(args, feat, head, train, val, policy,
                                     eval_cap, device)
        test_ids = sorted(test.beatmap_id.unique().tolist())
        dmap = dmap_for(feat, test_ids, "all",
                        np.random.default_rng(args.seed + 123), args, eval_cap,
                        head, device)
        pd.DataFrame({"beatmap_id": list(dmap.keys()),
                      "d": list(dmap.values())}).to_parquet(
            out_dir / f"{cond}_d_test.parquet", index=False)
        torch.save({"head": head.state_dict(), "policy": policy,
                    "eval_cap": eval_cap, "args": vars(args)},
                   out_dir / f"{cond}_head.pt")
        all_metrics[cond] = {"policy": policy, "eval_cap": eval_cap, **info,
                             "n_test_charts": len(test_ids)}
        (out_dir / "metrics.json").write_text(
            json.dumps(all_metrics, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(all_metrics, indent=1, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
