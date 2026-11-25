#!/usr/bin/env python3
"""step2_chart_encoder -- end-to-end chart encoder for the three behavior tasks.

Tasks
-----
rank    : d(chart) trained by within-player Bradley-Terry on ACC wins
judge   : rank + six-grade label-proportion aux head (per note event)
teacher : rank + K-dim reference-player signature aux head

The encoder is the D2V audio-style model (8-channel 100 Hz piano roll, 16 s
windows).  A chart vector is the valid-frame-weighted mean of its window
vectors, L2-normalised, then a small MLP head gives d.  The rank loss is

    q_ij = sigmoid(d_j - d_i),  y = 1[loss_i < loss_j]

for two charts i,j played by the same player segment.  The player ability
cancels in the difference, so no player embedding is needed for d.

This is a *research* trainer: the official metric is still
scripts/step2_rank_eval.py on the written d_test.parquet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data2vec_main"))

from d2v_main.config import load_config  # noqa: E402
from d2v_main.model import Data2VecModel  # noqa: E402
from d2v_main.charts import ChartStore, build_window, parse_chart, window_starts  # noqa: E402
from d2v_main.utils import git_rev, human_time, resolve_device, seed_all  # noqa: E402

COUNT_COLS = ["n_max", "n300", "n200", "n100", "n50", "n_miss"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_sha256(path: str) -> str:
    p = Path(path)
    return sha256_file(p) if p.exists() else "missing"


def set_deterministic(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


@contextmanager
def preserve_eval(*modules):
    """Run modules in eval mode, then restore their previous train/eval flags."""
    was = [m.training for m in modules]
    try:
        for m in modules:
            m.eval()
        yield
    finally:
        for m, flag in zip(modules, was):
            m.train(flag)


def snapshot_state(module: nn.Module) -> dict:
    """CPU clones, so a later optimizer step cannot mutate the snapshot."""
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------- data
class ChartCache:
    """Lazy .osu parse + window builder.  One process, one cache."""

    def __init__(self, store: ChartStore, frame_hz: int, window_frames: int,
                 sigma_ms: float, min_valid: int):
        self.store = store
        self.frame_hz = int(frame_hz)
        self.window_frames = int(window_frames)
        self.sigma_ms = float(sigma_ms)
        self.min_valid = int(min_valid)
        self._cache: dict[int, tuple] = {}

    def get(self, cid: int):
        cid = int(cid)
        hit = self._cache.get(cid)
        if hit is None:
            try:
                events, n_frames = parse_chart(str(self.store.path(cid)), self.frame_hz)
            except Exception:
                events, n_frames = tuple(), 0
            hit = (events, int(n_frames))
            self._cache[cid] = hit
        return hit

    def build(self, cid: int, starts: list[int]):
        events, n_frames = self.get(cid)
        xs, valid = [], []
        for s in starts:
            s = max(0, int(s))
            xs.append(build_window(events, s, self.window_frames,
                                   self.frame_hz, self.sigma_ms))
            valid.append(max(0, min(self.window_frames, n_frames - s)))
        return np.stack(xs), np.asarray(valid, dtype=np.int64)


# -------------------------------------------------------------------- heads
class ChartHead(nn.Module):
    def __init__(self, dim: int, hid: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hid, 1),
        )

    def forward(self, z):
        return self.net(self.norm(z)).squeeze(-1)


class EventHead(nn.Module):
    def __init__(self, dim: int, hid: int = 128, emb: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        self.col = nn.Embedding(4, emb)
        self.typ = nn.Embedding(2, emb)
        self.buc = nn.Embedding(8, emb)
        inp = dim + 3 * emb + 1
        self.net = nn.Sequential(
            nn.LayerNorm(inp), nn.Linear(inp, hid), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hid, 6),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, col, typ, buc, density):
        e = torch.cat([h, self.col(col), self.typ(typ), self.buc(buc),
                       density.unsqueeze(-1)], dim=-1)
        return self.net(e)


class SigHead(nn.Module):
    def __init__(self, dim: int, out_dim: int, hid: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hid, out_dim),
        )

    def forward(self, z):
        return self.net(self.norm(z))


# ---------------------------------------------------------------- utilities
def set_trainable(model: Data2VecModel, freeze: str, k: int) -> list[nn.Parameter]:
    model.requires_grad_(False)
    n_layers = len(model.student.layers)
    k = max(0, min(int(k), n_layers))
    if freeze == "none":
        for p in model.conv.parameters():
            p.requires_grad_(True)
        for p in model.student.parameters():
            p.requires_grad_(True)
    elif freeze == "topk":
        for p in model.student.layers[n_layers - k:].parameters():
            p.requires_grad_(True)
        for p in model.student.norm.parameters():
            p.requires_grad_(True)
    elif freeze == "all":
        pass
    else:
        raise ValueError(f"unknown freeze mode {freeze}")
    model.teacher.requires_grad_(False)
    model.student_head.requires_grad_(False)
    model.mask_emb.requires_grad_(False)
    return [p for p in model.parameters() if p.requires_grad]


def load_encoder(args, device: torch.device) -> tuple[Data2VecModel, int]:
    cfg = load_config(args.config)
    cfg["data"]["window_sec"] = float(args.window_sec)
    cfg["data"]["frame_hz"] = int(args.frame_hz)
    cfg["data"]["gaussian_sigma_ms"] = float(args.sigma_ms)
    dim = int(cfg["model"]["dim"])
    model = Data2VecModel(cfg)
    if args.init == "d2v":
        if not args.ckpt:
            raise SystemExit("--init d2v requires --ckpt")
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[encoder] load missing={missing} unexpected={unexpected}", flush=True)
    model.to(device)
    trainable = set_trainable(model, args.freeze, args.unfreeze_blocks)
    print(f"[encoder] init={args.init} freeze={args.freeze} k={args.unfreeze_blocks} "
          f"trainable={sum(p.numel() for p in trainable):,}", flush=True)
    return model, dim


def make_optimizer(model: Data2VecModel, head: nn.Module, args):
    enc = [p for p in model.parameters() if p.requires_grad]
    groups = []
    if enc:
        groups.append({"params": enc, "lr": args.lr_encoder,
                       "weight_decay": args.wd_encoder})
    groups.append({"params": list(head.parameters()), "lr": args.lr_head,
                   "weight_decay": args.wd_head})
    return torch.optim.AdamW(groups)


def bt_loss(d: torch.Tensor, pi: torch.Tensor, pj: torch.Tensor,
            y: torch.Tensor) -> torch.Tensor:
    if pi.numel() == 0:
        return d.sum() * 0.0
    # Keep the logistic loss in FP32.  Under bf16 autocast, sigmoid(8) rounds
    # to exactly 1.0, and the old `0 * log(0)` form returns NaN even when the
    # prediction direction is correct.
    logits = (d[pj] - d[pi]).float()
    target = y.to(dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")


def pair_targets(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All within-player pairs over the rows selected for one batch."""
    by_player: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        by_player.setdefault(int(r["player"]), []).append(i)
    pi, pj, yy = [], [], []
    for idx in by_player.values():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                li, lj = float(rows[i]["loss"]), float(rows[j]["loss"])
                y = 0.5 if li == lj else (1.0 if li < lj else 0.0)
                pi.append(i); pj.append(j); yy.append(y)
    return (torch.tensor(pi, dtype=torch.long),
            torch.tensor(pj, dtype=torch.long),
            torch.tensor(yy, dtype=torch.float32))


def counts_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[COUNT_COLS].to_numpy(np.float64)


def player_baselines(df: pd.DataFrame, counts: np.ndarray,
                     alpha: float = 5.0,
                     group_col: str = "player_id"):
    """Leave-one-chart-out six-grade baseline + skill, aligned to df rows."""
    g = counts.sum(axis=0)
    g = g / max(1.0, g.sum())
    tmp = pd.DataFrame({
        "player": df[group_col].to_numpy(),
        "loss": df["loss"].to_numpy(np.float64),
        "n": df[COUNT_COLS].sum(axis=1).to_numpy(np.float64),
    })
    for k, c in enumerate(COUNT_COLS):
        tmp[c] = counts[:, k]
    grp = tmp.groupby("player", sort=False)
    sums = grp[COUNT_COLS].transform("sum").to_numpy(np.float64)
    ns = grp["n"].transform("sum").to_numpy(np.float64)
    ls = grp["loss"].transform("sum").to_numpy(np.float64)
    nr = grp["loss"].transform("size").to_numpy(np.float64)
    den = (ns - tmp["n"].to_numpy() + alpha)[:, None]
    base = (sums - counts + alpha * g[None, :]) / np.maximum(den, 1e-9)
    base = base / np.maximum(base.sum(axis=1, keepdims=True), 1e-9)
    skill = (ls - tmp["loss"].to_numpy()) / np.maximum(nr - 1.0, 1.0)
    single = nr <= 1.0
    if single.any():
        skill[single] = float(np.mean(tmp["loss"].to_numpy()))
    mu, sd = float(skill.mean()), float(skill.std() + 1e-9)
    return base.astype(np.float32), ((skill - mu) / sd).astype(np.float32)


def shuffle_counts_within_player(df: pd.DataFrame, seed: int,
                                 group_col: str = "player_id") -> np.ndarray:
    # Unit 2x2 swaps preserve each chart's N and the player's six marginals.
    counts = counts_matrix(df).astype(np.int64)
    rng = np.random.default_rng(seed)
    for _, idx in df.groupby(group_col, sort=False).indices.items():
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) < 2:
            continue
        block = counts[idx].copy()
        n_swaps = max(20, 20 * len(idx))
        for _ in range(n_swaps):
            a, b = rng.choice(len(idx), size=2, replace=False)
            k, l = rng.choice(6, size=2, replace=False)
            if block[a, k] <= 0 or block[b, l] <= 0:
                continue
            block[a, k] -= 1; block[a, l] += 1
            block[b, l] -= 1; block[b, k] += 1
        counts[idx] = block
    return counts.astype(np.float64)


# ------------------------------------------------------------------ batches
def sample_rows(players: dict[int, list[dict]], args, rng) -> list[dict]:
    keys = [p for p, v in players.items() if len(v) >= 2]
    if not keys:
        raise SystemExit("no player has >=2 train charts")
    n = min(args.players_per_batch, len(keys))
    chosen = rng.choice(keys, size=n, replace=False)
    rows = []
    for p in chosen:
        cand = players[int(p)]
        m = min(args.charts_per_player, len(cand))
        take = rng.choice(len(cand), size=m, replace=False)
        rows.extend(cand[i] for i in take)
    return rows


def event_table(cache: ChartCache, cid: int, n_windows: int,
                hold_event: str):
    events, n_frames = cache.get(cid)
    if not events:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64),
                np.zeros(0, np.int64), np.zeros(0, np.int64),
                np.zeros(0, np.int64), np.zeros(0, np.float32),
                np.zeros(0, np.int64), np.zeros(0, np.int64))
    wf = cache.window_frames
    times = np.array([e.t for e in events], dtype=np.int64)
    order = np.argsort(times)
    st = times[order]
    win, frame, col, typ, buc, dens, win2, frame2 = [], [], [], [], [], [], [], []
    for e in events:
        head = int(e.t)
        tail = int(e.end) if e.end is not None else head
        tf = tail if hold_event == "tail" else head
        # "both" gathers onset and release; "head" and "tail" use one point.
        tf2 = tail if (hold_event == "both" and e.end is not None) else tf
        w1 = min(max(tf // wf, 0), n_windows - 1)
        f1 = min(max((tf - w1 * wf) // 2, 0), cache.window_frames // 2 - 1)
        win.append(w1); frame.append(f1)
        col.append(int(e.col))
        typ.append(1 if e.end is not None else 0)
        hl = int(e.end - e.t) if e.end is not None else 0
        buc.append(min(7, int(math.log2(hl + 1))) if hl > 0 else 0)
        lo = int(np.searchsorted(st, tf - 20, side="left"))
        hi = int(np.searchsorted(st, tf + 20, side="right"))
        dens.append(min(4.0, (hi - lo) / 32.0))
        if hold_event == "both" and tf2 != tf:
            w2 = min(max(tf2 // wf, 0), n_windows - 1)
            f2 = min(max((tf2 - w2 * wf) // 2, 0), cache.window_frames // 2 - 1)
        else:
            w2, f2 = w1, f1
        win2.append(w2); frame2.append(f2)
    return (np.asarray(win, np.int64), np.asarray(frame, np.int64),
            np.asarray(col, np.int64), np.asarray(typ, np.int64),
            np.asarray(buc, np.int64), np.asarray(dens, np.float32),
            np.asarray(win2, np.int64), np.asarray(frame2, np.int64))


def build_batch(rows: list[dict], cache: ChartCache, args, task: str,
                rng, chart_sig: dict[int, np.ndarray] | None = None,
                sig_map: dict[int, int] | None = None,
                mean_map: dict[int, float] | None = None):
    chart_ids = sorted({int(r["chart"]) for r in rows})
    cpos = {c: i for i, c in enumerate(chart_ids)}
    xs, valid, win_chart, offsets, n_windows = [], [], [], {}, {}
    for c in chart_ids:
        events, n_frames = cache.get(c)
        if task == "judge":
            starts = window_starts(n_frames, cache.window_frames)
            if args.max_windows > 0:
                starts = starts[: args.max_windows]
        else:
            hi = max(1, n_frames - cache.window_frames)
            starts = [int(rng.integers(0, hi + 1))
                      for _ in range(args.windows_per_chart)]
        if not starts:
            starts = [0]
        x, v = cache.build(c, starts)
        offsets[c] = sum(len(z) for z in xs)
        n_windows[c] = len(starts)
        xs.append(x); valid.append(v)
        win_chart.extend([cpos[c]] * len(starts))
    x = torch.from_numpy(np.concatenate(xs, axis=0))
    valid = torch.from_numpy(np.concatenate(valid, axis=0))
    win_chart = torch.tensor(win_chart, dtype=torch.long)

    pi_row, pj_row, py = pair_targets(rows)
    row_chart = torch.tensor([cpos[int(r["chart"])] for r in rows],
                             dtype=torch.long)
    # pair_targets returns row indices; d is indexed by unique chart position.
    pi = row_chart[pi_row] if pi_row.numel() else pi_row
    pj = row_chart[pj_row] if pj_row.numel() else pj_row
    keep = pi != pj
    pi, pj, py = pi[keep], pj[keep], py[keep]
    loss = torch.tensor([float(r["loss"]) for r in rows], dtype=torch.float32)
    baseline = torch.tensor(np.stack([r["baseline"] for r in rows]),
                            dtype=torch.float32) if "baseline" in rows[0] else None
    skill = torch.tensor([float(r.get("skill", 0.0)) for r in rows],
                         dtype=torch.float32)
    counts = torch.tensor(np.stack([r["counts"] for r in rows]),
                          dtype=torch.float32) if "counts" in rows[0] else None
    n_total = torch.tensor([float(r.get("n_total", 0.0)) for r in rows],
                           dtype=torch.float32)

    ev = None
    if task == "judge":
        ew, ef, ec, et, eb, ed, ew2, ef2 = [], [], [], [], [], [], [], []
        ev_chart = []
        for c in chart_ids:
            w, f, col, typ, buc, dens, w2, f2 = event_table(
                cache, c, n_windows[c], args.hold_event)
            if w.size == 0:
                continue
            base = offsets[c]
            ew.append(base + w); ef.append(f)
            ew2.append(base + w2); ef2.append(f2)
            ec.append(col); et.append(typ); eb.append(buc); ed.append(dens)
            ev_chart.append(np.full(w.size, cpos[c], np.int64))
        if ew:
            ev = {
                "win": torch.from_numpy(np.concatenate(ew)),
                "frame": torch.from_numpy(np.concatenate(ef)),
                "win2": torch.from_numpy(np.concatenate(ew2)),
                "frame2": torch.from_numpy(np.concatenate(ef2)),
                "col": torch.from_numpy(np.concatenate(ec)),
                "typ": torch.from_numpy(np.concatenate(et)),
                "buc": torch.from_numpy(np.concatenate(eb)),
                "dens": torch.from_numpy(np.concatenate(ed)),
                "chart": torch.from_numpy(np.concatenate(ev_chart)),
            }
        else:
            ev = {"chart": torch.zeros(0, dtype=torch.long)}

    sig = None
    if task == "teacher":
        k = args.teacher_dim
        tgt = np.full((len(chart_ids), k), np.nan, np.float32)
        for c in chart_ids:
            src = sig_map.get(c, c) if sig_map else c
            if args.aux == "mean":
                if mean_map is not None and src in mean_map:
                    tgt[cpos[c], 0] = float(mean_map[src])
            elif chart_sig and src in chart_sig:
                v = chart_sig[src]
                tgt[cpos[c], :len(v)] = v[:k]
        sig = torch.from_numpy(tgt)
    return {
        "x": x, "valid": valid, "win_chart": win_chart,
        "chart_ids": chart_ids, "cpos": cpos,
        "pair_i": pi, "pair_j": pj, "pair_y": py,
        "row_chart": row_chart, "row_loss": loss, "skill": skill,
        "baseline": baseline, "counts": counts, "n_total": n_total,
        "events": ev, "sig": sig, "n_rows": len(rows),
    }


def to_device(batch: dict, device: torch.device) -> dict:
    out = dict(batch)
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, dict) and v and torch.is_tensor(next(iter(v.values()))):
            out[k] = {kk: vv.to(device) for kk, vv in v.items()}
    return out


def encode(model: Data2VecModel, head: ChartHead, batch: dict):
    x = batch["x"]
    valid_in = batch["valid"]
    t_out = (x.size(2) + 1) // 2
    valid_out = (valid_in + 1) // 2
    vm = torch.arange(t_out, device=x.device)[None, :] < valid_out[:, None]
    out, _ = model.forward_student(x, vm, None)
    # A single extreme window must not poison the whole run; the D2V front-end
    # can produce inf/NaN on rare dense charts under bf16.
    out = torch.nan_to_num(out.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    w = vm.float().unsqueeze(-1)
    win_vec = (out * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
    n_charts = int(batch["win_chart"].max().item()) + 1 if batch["win_chart"].numel() else 1
    vec = torch.zeros(n_charts, win_vec.size(1), device=x.device, dtype=win_vec.dtype)
    den = torch.zeros(n_charts, device=x.device, dtype=win_vec.dtype)
    vec.index_add_(0, batch["win_chart"], win_vec * valid_out[:, None].float())
    den.index_add_(0, batch["win_chart"], valid_out.float())
    vec = vec / den.clamp_min(1.0)[:, None]
    z = F.normalize(vec, dim=-1)
    d = head(z)
    return out, vm, d, z


def judge_loss(event_logits, batch, class_weight, single_idx: int = -1):
    ev = batch["events"]
    if ev is None or "chart" not in ev or ev["chart"].numel() == 0:
        return event_logits.sum() * 0.0
    chart = ev["chart"]
    counts = batch["counts"]
    n_total = batch["n_total"].clamp_min(1.0)
    base = batch["baseline"].clamp_min(1e-9)
    losses = []
    for r in range(batch["n_rows"]):
        c = int(batch["row_chart"][r].item())
        mask = chart == c
        if not bool(mask.any()):
            continue
        logits = event_logits[mask] + torch.log(base[r])
        p = torch.softmax(logits, dim=-1)
        c_hat = p.sum(dim=0)
        if single_idx >= 0:
            rate = c_hat[single_idx] / c_hat.sum().clamp_min(1e-9)
            obs = counts[r, single_idx] / n_total[r]
            losses.append(-(obs * torch.log(rate.clamp_min(1e-9))
                            + (1.0 - obs) * torch.log((1.0 - rate).clamp_min(1e-9))))
        else:
            pred = c_hat / c_hat.sum().clamp_min(1e-9)
            obs = counts[r] / n_total[r]
            losses.append(-(class_weight * obs * torch.log(pred.clamp_min(1e-9))).sum())
    return torch.stack(losses).mean() if losses else event_logits.sum() * 0.0


def teacher_loss(s_hat, batch):
    tgt = batch["sig"]
    mask = torch.isfinite(tgt)
    if not bool(mask.any()):
        return s_hat.sum() * 0.0
    return F.mse_loss(s_hat[mask], tgt[mask])


def _val_baselines(train_df: pd.DataFrame, train_counts: np.ndarray,
                   val_df: pd.DataFrame, group_col: str,
                   alpha: float = 5.0):
    """Player priors for val rows from train counts only (no val leakage)."""
    g = train_counts.sum(axis=0)
    g = g / max(1.0, g.sum())
    tmp = pd.DataFrame({"u": train_df[group_col].to_numpy(),
                        "n": train_df[COUNT_COLS].sum(axis=1).to_numpy(np.float64),
                        "loss": train_df["loss"].to_numpy(np.float64)})
    for k, c in enumerate(COUNT_COLS):
        tmp[c] = train_counts[:, k]
    grp = tmp.groupby("u", sort=False)
    sums = grp[COUNT_COLS].sum()
    ns = grp["n"].sum()
    ls = grp["loss"].mean()
    out = np.tile(g, (len(val_df), 1)).astype(np.float32)
    for i, u in enumerate(val_df[group_col].to_numpy()):
        if u in sums.index:
            out[i] = ((sums.loc[u].to_numpy(np.float64) + alpha * g)
                      / max(1.0, float(ns.loc[u]) + alpha))
            out[i] /= max(1e-9, out[i].sum())
    return out


@torch.inference_mode()
def eval_event_proportions(model, head, event_head, cache, val_df, train_df,
                           train_counts, group_col, args, device, out_dir):
    """Held-out val event proportions and the manipulation-check metrics."""
    if event_head is None or val_df.empty:
        return None
    val = val_df.copy()
    full = []
    for c in sorted(val.beatmap_id.unique().tolist()):
        _, nf = cache.get(int(c))
        nw = len(window_starts(nf, cache.window_frames))
        if args.max_windows <= 0 or nw <= args.max_windows:
            full.append(int(c))
    val = val[val.beatmap_id.isin(set(full))].copy()
    if val.empty:
        return None
    base = _val_baselines(train_df, train_counts, val, group_col)
    rows = []
    with preserve_eval(model, head, event_head):
        for i in range(0, len(full), 8):
            chunk = full[i:i + 8]
            xs, valid, wc = [], [], []
            for c in chunk:
                _, nf = cache.get(c)
                starts = window_starts(nf, cache.window_frames)[:args.max_windows]                     if args.max_windows > 0 else window_starts(nf, cache.window_frames)
                x, v = cache.build(c, starts)
                xs.append(x); valid.append(v)
                wc.extend([len(xs) - 1] * len(starts))
            x = torch.from_numpy(np.concatenate(xs)).to(device)
            v = torch.from_numpy(np.concatenate(valid)).to(device)
            wc_t = torch.tensor(wc, dtype=torch.long, device=device)
            batch = {"x": x, "valid": v, "win_chart": wc_t,
                     "chart_ids": chunk, "cpos": {c: j for j, c in enumerate(chunk)}}
            out, _, _, _ = encode(model, head, batch)
            for j, c in enumerate(chunk):
                sub = val[val.beatmap_id == c]
                if sub.empty:
                    continue
                _, nf = cache.get(c)
                nw = len(window_starts(nf, cache.window_frames))
                if args.max_windows > 0:
                    nw = min(nw, args.max_windows)
                ew, ef, ec, et, eb, ed, ew2, ef2 = event_table(
                    cache, c, nw, args.hold_event)
                if ew.size == 0:
                    continue
                offset = wc.index(j)  # first encoded window of this chart
                ew_t = torch.from_numpy(ew).to(device)
                ef_t = torch.from_numpy(ef).to(device)
                ev = event_head(
                    out[offset + ew_t, ef_t],
                    torch.from_numpy(ec).to(device),
                    torch.from_numpy(et).to(device),
                    torch.from_numpy(eb).to(device),
                    torch.from_numpy(ed).to(device))
                for ridx in np.where(val.beatmap_id.to_numpy() == c)[0]:
                    r = int(ridx)
                    logits = ev + torch.log(torch.from_numpy(base[r]).to(device))
                    p = torch.softmax(logits, dim=-1).sum(dim=0).cpu().numpy()
                    pred = p / max(1e-12, p.sum())
                    cnt = val.iloc[r][COUNT_COLS].to_numpy(np.float64)
                    obs = cnt / max(1.0, cnt.sum())
                    pred = np.clip(pred.astype(np.float64), 1e-9, 1.0)
                    nll = float(-(obs * np.log(pred)).sum())
                    bnll = float(-(obs * np.log(np.clip(base[r], 1e-9, 1.0))).sum())
                    rec = {"beatmap_id": int(c),
                           "player_id": int(val.iloc[r]["player_id"]),
                           "n_events": int(ew.size),
                           "obs_n": float(cnt.sum()),
                           "event_nll": nll, "baseline_nll": bnll}
                    for k, name in enumerate(COUNT_COLS):
                        rec[f"obs_{name}"] = float(obs[k])
                        rec[f"pred_{name}"] = float(pred[k])
                    rows.append(rec)
    if not rows:
        return None
    ev_df = pd.DataFrame(rows)
    ev_df.to_parquet(out_dir / "events_val.parquet", index=False)
    qa = {"n_rows": int(len(ev_df)), "n_charts": int(ev_df.beatmap_id.nunique()),
          "event_nll": float(ev_df.event_nll.mean()),
          "baseline_nll": float(ev_df.baseline_nll.mean()),
          "delta_vs_baseline": float((ev_df.event_nll - ev_df.baseline_nll).mean()),
          "spearman": {}}
    for name in COUNT_COLS:
        a = ev_df[f"obs_{name}"].to_numpy()
        b = ev_df[f"pred_{name}"].to_numpy()
        qa["spearman"][name] = (float(np.corrcoef(a, b)[0, 1])
                                if a.std() > 0 and b.std() > 0 else None)
    (out_dir / "events_qa.json").write_text(
        json.dumps(qa, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(qa, indent=1, ensure_ascii=False), flush=True)
    return qa


# --------------------------------------------------------------------- eval
@torch.inference_mode()
def encode_charts(model, head, cache: ChartCache, chart_ids: list[int],
                  args, device: torch.device, batch_charts: int = 8) -> dict[int, float]:
    out = {}
    with preserve_eval(model, head):
        for i in range(0, len(chart_ids), batch_charts):
            chunk = chart_ids[i:i + batch_charts]
            xs, valid, wc = [], [], []
            for c in chunk:
                events, n_frames = cache.get(c)
                starts = window_starts(n_frames, cache.window_frames)
                if args.max_windows > 0:
                    starts = starts[: args.max_windows]
                if not starts:
                    starts = [0]
                x, v = cache.build(c, starts)
                xs.append(x); valid.append(v)
                wc.extend([len(xs) - 1] * len(starts))
            x = torch.from_numpy(np.concatenate(xs)).to(device)
            v = torch.from_numpy(np.concatenate(valid)).to(device)
            wc = torch.tensor(wc, dtype=torch.long, device=device)
            t_out = (x.size(2) + 1) // 2
            vo = (v + 1) // 2
            vm = torch.arange(t_out, device=device)[None, :] < vo[:, None]
            f, _ = model.forward_student(x, vm, None)
            f = torch.nan_to_num(f.float(), nan=0.0, posinf=1e4, neginf=-1e4)
            w = vm.float().unsqueeze(-1)
            wv = (f * w).sum(1) / w.sum(1).clamp_min(1.0)
            vec = torch.zeros(len(chunk), wv.size(1), device=device)
            den = torch.zeros(len(chunk), device=device)
            vec.index_add_(0, wc, wv * vo[:, None].float())
            den.index_add_(0, wc, vo.float())
            z = F.normalize(vec / den.clamp_min(1.0)[:, None], dim=-1)
            d = head(z).cpu().numpy()
            if not np.isfinite(d).all():
                print(f"[encoder] non-finite eval d: {int((~np.isfinite(d)).sum())}/{len(d)}",
                      flush=True)
                d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
            for c, val in zip(chunk, d):
                out[int(c)] = float(val)
    return out


def val_nll(df: pd.DataFrame, dmap: dict[int, float]) -> float:
    rows = df[df.beatmap_id.isin(dmap.keys())]
    vals = []
    for _, g in rows.groupby("player_id", sort=False):
        if len(g) < 2:
            continue
        b = g["beatmap_id"].to_numpy()
        loss = g["loss"].to_numpy(np.float64)
        d = np.array([dmap.get(int(x), np.nan) for x in b], np.float64)
        ok = np.isfinite(d)
        b, loss, d = b[ok], loss[ok], d[ok]
        if b.size < 2:
            continue
        ii, jj = np.triu_indices(b.size, 1)
        la, lb = loss[ii], loss[jj]
        y = np.where(la < lb, 1.0, np.where(la == lb, 0.5, 0.0))
        q = 1.0 / (1.0 + np.exp(-(d[jj] - d[ii])))
        q = np.clip(q, 1e-9, 1.0 - 1e-9)
        vals.append(-(y * np.log(q) + (1.0 - y) * np.log1p(-q)))
    return float(np.concatenate(vals).mean()) if vals else float("nan")


# ------------------------------------------------------------------- driver
def _resolve_score_checkpoint(args) -> Path:
    if args.checkpoint:
        p = Path(args.checkpoint)
        if p.is_dir():
            if not args.checkpoint_kind:
                raise SystemExit("--checkpoint is a directory; pass --checkpoint-kind best|last")
            p = p / f"{args.checkpoint_kind}.pt"
    elif args.run_dir and args.checkpoint_kind:
        p = Path(args.run_dir) / f"{args.checkpoint_kind}.pt"
    else:
        raise SystemExit("score-only needs --checkpoint, or --run-dir + --checkpoint-kind")
    if not p.exists():
        raise SystemExit(f"checkpoint not found: {p}")
    return p


def _eval_frame(args):
    df = pd.read_parquet(args.split_table)
    if args.split_col not in df.columns:
        raise SystemExit(f"{args.split_table}: no {args.split_col} column")
    store = ChartStore(args.osu_dir, ids=sorted(df.beatmap_id.unique().tolist()))
    df = df[df.beatmap_id.isin(set(store.ids))].copy()
    test = df[df[args.split_col] == "test"].copy()
    val = df[df[args.split_col] == "val"].copy()
    if test.empty:
        raise SystemExit("empty test split")
    for name in ("test", "val"):
        frame = test if name == "test" else val
        if "label_ok" in frame.columns:
            frame = frame[frame.label_ok].copy()
            if name == "test":
                test = frame
            else:
                val = frame
    cache = ChartCache(store, args.frame_hz,
                       int(round(args.window_sec * args.frame_hz)),
                       args.sigma_ms, int(round(args.min_valid_sec * args.frame_hz)))
    return test, val, cache


def _window_coverage(cache: ChartCache, chart_ids: list[int], max_windows: int):
    rows = []
    for c in chart_ids:
        _, n_frames = cache.get(c)
        starts_all = window_starts(n_frames, cache.window_frames)
        starts = starts_all[:max_windows] if max_windows > 0 else starts_all
        if not starts:
            starts = [0]
        used = sum(max(0, min(cache.window_frames, n_frames - s)) for s in starts)
        rows.append({"beatmap_id": int(c), "n_frames": int(n_frames),
                     "n_windows_all": int(len(starts_all)),
                     "n_windows_used": int(len(starts)),
                     "frames_used": int(used),
                     "frames_total": int(n_frames),
                     "coverage": float(used / n_frames) if n_frames else 0.0})
    return pd.DataFrame(rows)


def score_main(args) -> int:
    """Deterministic export from an explicit checkpoint; never trains."""
    if args.task != "rank":
        raise SystemExit("score-only currently supports --task rank only")
    set_deterministic(args.seed)
    device = resolve_device(args.device)
    ckpt = _resolve_score_checkpoint(args)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    ck_args = ck.get("args") or {}
    # Architecture must match the checkpoint.  Legacy checkpoints have no
    # args; fall back to CLI/config, which is why --config is required there.
    for key in ("config", "hid", "dropout", "window_sec", "frame_hz",
                "sigma_ms", "min_valid_sec"):
        if key in ck_args:
            setattr(args, key, ck_args[key])
    cfg = load_config(args.config)
    cfg["data"]["window_sec"] = float(args.window_sec)
    cfg["data"]["frame_hz"] = int(args.frame_hz)
    cfg["data"]["gaussian_sigma_ms"] = float(args.sigma_ms)
    model = Data2VecModel(cfg)
    model.load_state_dict(ck["model"], strict=True)
    if "head" not in ck:
        raise SystemExit(f"{ckpt}: missing head state")
    head = ChartHead(int(cfg["model"]["dim"]), args.hid, args.dropout)
    head.load_state_dict(ck["head"], strict=True)
    model.to(device).eval()
    head.to(device).eval()
    print(f"[score] checkpoint={ckpt} kind={ck.get('kind', 'unknown')} "
          f"step={ck.get('step')} pair_views={ck.get('pair_views')} "
          f"sha256={sha256_file(ckpt)[:16]}", flush=True)

    test, val, cache = _eval_frame(args)
    test_charts = sorted(test.beatmap_id.unique().tolist())
    val_charts = sorted(val.beatmap_id.unique().tolist()) if not val.empty else []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repeat = max(1, int(args.repeat_eval))
    d_test = None
    d_val = None
    max_diff = 0.0
    nll_first = nll_last = float("nan")
    for r in range(repeat):
        cur_test = encode_charts(model, head, cache, test_charts, args, device,
                                 batch_charts=args.eval_batch_charts)
        cur_val = (encode_charts(model, head, cache, val_charts, args, device,
                                 batch_charts=args.eval_batch_charts)
                   if val_charts else {})
        if d_test is not None:
            keys = sorted(set(d_test) & set(cur_test))
            if keys:
                max_diff = max(max_diff, max(abs(d_test[c] - cur_test[c]) for c in keys))
            nll_last = val_nll(test, cur_test)
        else:
            nll_first = val_nll(test, cur_test)
        d_test, d_val = cur_test, cur_val
    for name, dmap in (("d_test", d_test), ("d_val", d_val)):
        if dmap is not None:
            pd.DataFrame({"beatmap_id": list(dmap.keys()),
                          "d": list(dmap.values())}).to_parquet(
                out_dir / f"{name}.parquet", index=False)
    if not np.isfinite(nll_last):
        nll_last = nll_first
    coverage = _window_coverage(cache, test_charts, int(args.max_windows))
    coverage.to_parquet(out_dir / "window_coverage.parquet", index=False)
    out = {
        "checkpoint": str(ckpt),
        "checkpoint_kind": ck.get("kind"),
        "checkpoint_step": ck.get("step"),
        "checkpoint_sha256": sha256_file(ckpt),
        "checkpoint_best_val": ck.get("best_val"),
        "task": args.task,
        "split_table": str(args.split_table),
        "split_col": args.split_col,
        "max_windows": int(args.max_windows),
        "n_test_charts": int(len(test_charts)),
        "n_val_charts": int(len(val_charts)),
        "n_windows_all": int(coverage.n_windows_all.sum()),
        "n_windows_used": int(coverage.n_windows_used.sum()),
        "coverage_mean": float(coverage.coverage.mean()),
        "repeat_eval": int(repeat),
        "repeat_max_abs_diff": float(max_diff),
        "nll_test_first": float(nll_first),
        "nll_test_last": float(nll_last),
        "repeat_nll_diff": float(abs(nll_last - nll_first))
        if np.isfinite(nll_last) and np.isfinite(nll_first) else None,
        "git_rev": git_rev(ROOT),
        "config_sha256": config_sha256(args.config),
    }
    (out_dir / "score_metrics.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    (out_dir / "score_config.json").write_text(
        json.dumps(vars(args), indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=1, ensure_ascii=False), flush=True)
    if repeat > 1 and max_diff > args.eval_diff_tol:
        raise SystemExit(
            f"eval is not deterministic: max|d1-d2|={max_diff:.3g} > "
            f"{args.eval_diff_tol:.3g}; use --device cpu or a deterministic GPU setup")
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["rank", "judge", "teacher"], required=True)
    ap.add_argument("--init", choices=["d2v", "random"], default="d2v")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default="data2vec_main/configs/audio_style_base.yaml")
    ap.add_argument("--freeze", choices=["all", "topk", "none"], default="topk")
    ap.add_argument("--unfreeze-blocks", type=int, default=4)
    ap.add_argument("--split-table", required=True)
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--osu-dir", default="data/osu_cache")
    ap.add_argument("--window-sec", type=float, default=16.0)
    ap.add_argument("--frame-hz", type=int, default=100)
    ap.add_argument("--sigma-ms", type=float, default=8.0)
    ap.add_argument("--min-valid-sec", type=float, default=1.0)
    ap.add_argument("--windows-per-chart", type=int, default=4)
    ap.add_argument("--max-windows", type=int, default=24)
    ap.add_argument("--players-per-batch", type=int, default=4)
    ap.add_argument("--charts-per-player", type=int, default=6)
    ap.add_argument("--pair-views", type=int, default=2_000_000)
    ap.add_argument("--row-views", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--lr-encoder", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--wd-encoder", type=float, default=0.01)
    ap.add_argument("--wd-head", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--hid-event", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--judge-weight", type=float, default=0.0)
    ap.add_argument("--judge-class-weights", choices=["none", "sqrt_inv", "inv"],
                    default="sqrt_inv")
    ap.add_argument("--single-grade-col", default="n_miss")
    ap.add_argument("--teacher-signature", default=None)
    ap.add_argument("--teacher-weight", type=float, default=0.0)
    ap.add_argument("--teacher-dim", type=int, default=64)
    ap.add_argument("--aux", default="none",
                    choices=["none", "judge", "judge_shuffled", "single_grade",
                             "mean", "full", "shuffled"])
    ap.add_argument("--hold-event", choices=["head", "tail", "both"], default="tail")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--benchmark-views", type=int, default=0)
    ap.add_argument("--limit-train-rows", type=int, default=0)
    # Deterministic export / checkpoint selection (E1_fix)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--checkpoint", default=None,
                    help="explicit checkpoint file to score (legacy path)")
    ap.add_argument("--checkpoint-kind", choices=["best", "last"], default=None)
    ap.add_argument("--run-dir", default=None,
                    help="run directory containing best.pt/last.pt")
    ap.add_argument("--repeat-eval", type=int, default=1)
    ap.add_argument("--eval-diff-tol", type=float, default=1e-6)
    ap.add_argument("--eval-batch-charts", type=int, default=8)
    ap.add_argument("--export-checkpoint", choices=["last", "best"], default="last",
                    help="which checkpoint to export after training")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.score_only:
        return score_main(args)
    t0 = time.time()
    seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=1,
                                                    ensure_ascii=False) + "\n")

    df = pd.read_parquet(args.split_table)
    if args.split_col not in df.columns:
        raise SystemExit(f"{args.split_table}: no {args.split_col} column")
    if args.limit_train_rows:
        df = df.head(args.limit_train_rows).copy()
    store = ChartStore(args.osu_dir, ids=sorted(df.beatmap_id.unique().tolist()))
    keep = set(store.ids)
    df = df[df.beatmap_id.isin(keep)].copy()
    train = df[df[args.split_col] == "train"].copy()
    val = df[df[args.split_col] == "val"].copy()
    test = df[df[args.split_col] == "test"].copy()
    if test.empty:
        raise SystemExit("empty test split")
    if val.empty:
        val = train.sample(min(len(train), max(200, len(train) // 10)),
                           random_state=args.seed)
    print(f"[encoder] train {len(train)} val {len(val)} test {len(test)} "
          f"charts train={train.beatmap_id.nunique()} test={test.beatmap_id.nunique()}",
          flush=True)

    # A row without a valid six-grade label cannot train or be compared in the
    # judge arms; filter it for rank/teacher too so all arms share the same rows.
    if "label_ok" in train.columns:
        train = train[train.label_ok].copy()
    if "label_ok" in val.columns:
        val = val[val.label_ok].copy()
    if "label_ok" in test.columns:
        test = test[test.label_ok].copy()

    cache = ChartCache(store, args.frame_hz,
                       int(round(args.window_sec * args.frame_hz)),
                       args.sigma_ms, int(round(args.min_valid_sec * args.frame_hz)))
    if args.task == "judge":
        # The label-proportion loss sums over *all* notes.  A chart whose tail
        # falls outside max_windows would silently gather those notes into the
        # last window, so keep only fully covered charts in every judge arm.
        full, n_drop = [], 0
        for c in sorted(train.beatmap_id.unique().tolist()):
            _, nf = cache.get(c)
            nw = len(window_starts(nf, cache.window_frames))
            if args.max_windows <= 0 or nw <= args.max_windows:
                full.append(int(c))
            else:
                n_drop += 1
        before = len(train)
        train = train[train.beatmap_id.isin(set(full))].copy()
        print(f"[encoder] judge full-coverage charts {len(full)} "
              f"(dropped {n_drop} charts, {before-len(train)} rows)", flush=True)

    counts = None
    base = skill = None
    group_col = "player_id_base" if "player_id_base" in train.columns else "player_id"
    if args.task == "judge" or args.aux in ("judge", "judge_shuffled", "single_grade"):
        orig_counts = counts_matrix(train).astype(np.float64)
        # The baseline is always estimated from the real counts; only the
        # target distribution is permuted in the placebo arm.
        if args.aux == "judge_shuffled":
            counts = shuffle_counts_within_player(train, args.seed, group_col)
        else:
            counts = orig_counts
        base, skill = player_baselines(train, orig_counts, group_col=group_col)
        if args.task != "judge":
            raise SystemExit("judge aux is only implemented with --task judge")

    players: dict[int, list[dict]] = {}
    for i, (_, r) in enumerate(train.iterrows()):
        p = int(r["player_id"])
        row = {"player": p, "chart": int(r["beatmap_id"]),
               "loss": float(r["loss"])}
        if counts is not None:
            row["counts"] = counts[i]
            row["n_total"] = float(orig_counts[i].sum())
            row["baseline"] = base[i]
            row["skill"] = float(skill[i])
        players.setdefault(p, []).append(row)

    class_weight = None
    single_idx = -1
    if args.task == "judge":
        freq = orig_counts.sum(0) / max(1.0, orig_counts.sum())
        if args.judge_class_weights == "sqrt_inv":
            w = 1.0 / np.sqrt(np.maximum(freq, 1e-12))
        elif args.judge_class_weights == "inv":
            w = 1.0 / np.maximum(freq, 1e-12)
        else:
            w = np.ones(6)
        w = w / w.mean()
        class_weight = torch.tensor(w, dtype=torch.float32, device=device)
        if args.aux == "single_grade":
            single_idx = COUNT_COLS.index(args.single_grade_col)

    chart_sig = None
    sig_map = None
    mean_map = None
    teacher_dim = args.teacher_dim
    if args.task == "teacher":
        if not args.teacher_signature:
            raise SystemExit("--task teacher requires --teacher-signature")
        sig = pd.read_parquet(args.teacher_signature)
        cols = [c for c in sig.columns if c.startswith("s_")]
        cols = sorted(cols, key=lambda x: int(x.split("_")[1]))
        if not cols and args.aux != "mean":
            raise SystemExit("no s_* columns in teacher signature")
        if args.aux == "mean" and "teacher_mean_z" not in sig.columns:
            raise SystemExit("aux=mean requires teacher_mean_z in the signature")
        if args.aux == "mean":
            teacher_dim = 1
            mean_map = {int(r.beatmap_id): float(r.teacher_mean_z)
                        for _, r in sig.iterrows()}
        else:
            teacher_dim = min(args.teacher_dim, len(cols)) if args.teacher_dim else len(cols)
            chart_sig = {}
            for _, r in sig.iterrows():
                v = np.asarray([r[c] for c in cols[:teacher_dim]], np.float32)
                chart_sig[int(r["beatmap_id"])] = v
            if args.aux == "shuffled":
                ids = np.array(sorted(chart_sig.keys()))
                perm = rng.permutation(len(ids))
                sig_map = {int(a): int(b) for a, b in zip(ids, ids[perm])}
        args.teacher_dim = int(teacher_dim)
        print(f"[encoder] teacher signatures {len(sig)} dim={teacher_dim} aux={args.aux}",
              flush=True)

    model, dim = load_encoder(args, device)
    head = ChartHead(dim, args.hid, args.dropout).to(device)
    event_head = EventHead(dim, args.hid_event, dropout=args.dropout).to(device) \
        if args.task == "judge" else None
    sig_head = SigHead(dim, teacher_dim, args.hid, args.dropout).to(device) \
        if args.task == "teacher" else None
    opt = make_optimizer(model, head, args)
    if event_head is not None:
        opt.add_param_group({"params": list(event_head.parameters()),
                             "lr": args.lr_head, "weight_decay": args.wd_head})
    if sig_head is not None:
        opt.add_param_group({"params": list(sig_head.parameters()),
                             "lr": args.lr_head, "weight_decay": args.wd_head})

    train_charts = sorted(train.beatmap_id.unique().tolist())
    val_charts = sorted(val.beatmap_id.unique().tolist())
    test_charts = sorted(test.beatmap_id.unique().tolist())

    start_step = 0
    pair_views = 0
    row_views = 0
    best_val = float("inf")
    best_step = -1
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        head.load_state_dict(ck["head"])
        if event_head is not None and "event_head" in ck:
            event_head.load_state_dict(ck["event_head"])
        if sig_head is not None and "sig_head" in ck:
            sig_head.load_state_dict(ck["sig_head"])
        opt.load_state_dict(ck["opt"])
        start_step = int(ck.get("step", 0))
        pair_views = int(ck.get("pair_views", 0))
        row_views = int(ck.get("row_views", 0))
        best_val = float(ck.get("best_val", float("inf")))
        best_step = int(ck.get("best_step", -1))
        print(f"[encoder] resumed step={start_step} pair_views={pair_views}",
              flush=True)

    if args.freeze == "all":
        model.eval()
    else:
        model.train()
    head.train()
    if event_head is not None:
        event_head.train()
    if sig_head is not None:
        sig_head.train()

    config_hash = config_sha256(args.config)
    manifest_path = out_dir / "checkpoint_manifest.json"
    manifest = (json.loads(manifest_path.read_text())
                if manifest_path.exists() else {})

    def save_ckpt(kind: str) -> dict:
        path = out_dir / f"{kind}.pt"
        payload = {
            "format_version": 2,
            "kind": kind,
            "model": snapshot_state(model),
            "head": snapshot_state(head),
            "event_head": snapshot_state(event_head) if event_head is not None else None,
            "sig_head": snapshot_state(sig_head) if sig_head is not None else None,
            "opt": opt.state_dict() if kind == "last" else None,
            "step": step,
            "pair_views": pair_views,
            "row_views": row_views,
            "best_val": best_val,
            "best_step": best_step,
            "args": vars(args),
            "git_rev": git_rev(ROOT),
            "config_sha256": config_hash,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        atomic_torch_save(payload, path)
        rec = {"path": str(path), "step": step, "pair_views": pair_views,
               "val_nll": best_val if kind == "best" else None,
               "sha256": sha256_file(path)}
        manifest[kind] = rec
        manifest_path.write_text(json.dumps(manifest, indent=1,
                                           ensure_ascii=False) + "\n")
        return rec

    log_path = out_dir / "train_log.jsonl"
    log_f = open(log_path, "a", encoding="utf-8")
    bad_log = open(out_dir / "bad_batches.jsonl", "a", encoding="utf-8")
    step = start_step
    bad = 0
    bad_batches = 0
    bad_loss = 0
    bad_grad = 0
    t_start = time.time()
    amp = args.bf16 and device.type == "cuda"

    def finite_or_none(x) -> float | None:
        try:
            v = float(x)
        except Exception:
            return None
        return v if np.isfinite(v) else None

    def record_bad(stage: str, batch: dict, d: torch.Tensor,
                   rank: torch.Tensor | None = None,
                   aux: torch.Tensor | None = None,
                   gn: torch.Tensor | None = None) -> None:
        with torch.no_grad():
            pi, pj = batch["pair_i"], batch["pair_j"]
            diff = (d[pj] - d[pi]).abs() if pi.numel() else d.new_zeros(1)
            rec = {
                "step": step,
                "stage": stage,
                "charts": [int(x) for x in batch["chart_ids"][:8]],
                "n_windows": int(batch["x"].size(0)),
                "d_min": finite_or_none(d.min().detach().cpu()),
                "d_max": finite_or_none(d.max().detach().cpu()),
                "pair_diff_max": finite_or_none(diff.max().detach().cpu()),
                "rank_loss": finite_or_none(rank.detach().cpu())
                if rank is not None else None,
                "aux_loss": finite_or_none(aux.detach().cpu())
                if aux is not None else None,
                "grad_norm": finite_or_none(gn.detach().cpu())
                if gn is not None else None,
            }
        bad_log.write(json.dumps(rec) + "\n")
        bad_log.flush()

    while True:
        if pair_views >= args.pair_views and (args.row_views <= 0 or row_views >= args.row_views):
            break
        if args.max_steps and step >= args.max_steps:
            break
        rows = sample_rows(players, args, rng)
        batch = build_batch(rows, cache, args, args.task, rng,
                            chart_sig=chart_sig, sig_map=sig_map,
                            mean_map=mean_map)
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16) if amp else torch.autocast("cpu", enabled=False):
            out, vm, d, z = encode(model, head, batch)
            rank = bt_loss(d, batch["pair_i"], batch["pair_j"], batch["pair_y"])
            aux = d.sum() * 0.0
            if (args.task == "judge" and args.judge_weight > 0
                    and batch["events"] is not None and "win" in batch["events"]):
                ev = event_head(out[batch["events"]["win"], batch["events"]["frame"]],
                                batch["events"]["col"], batch["events"]["typ"],
                                batch["events"]["buc"], batch["events"]["dens"])
                if args.hold_event == "both":
                    ev2 = event_head(out[batch["events"]["win2"], batch["events"]["frame2"]],
                                     batch["events"]["col"], batch["events"]["typ"],
                                     batch["events"]["buc"], batch["events"]["dens"])
                    ev = 0.5 * (ev + ev2)
                aux = judge_loss(ev, batch, class_weight, single_idx)
            if args.task == "teacher" and args.teacher_weight > 0:
                aux = teacher_loss(sig_head(z), batch)
            loss = rank + args.judge_weight * aux if args.task == "judge" else \
                rank + args.teacher_weight * aux
        if not torch.isfinite(loss):
            bad_batches += 1
            bad_loss += 1
            record_bad("loss", batch, d, rank, aux)
            opt.zero_grad(set_to_none=True)
            if bad_batches % 50 == 1:
                print(f"[encoder] skip non-finite loss step={step} "
                      f"bad={bad_batches} charts={batch['chart_ids'][:6]}", flush=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
            + (list(event_head.parameters()) if event_head is not None else [])
            + (list(sig_head.parameters()) if sig_head is not None else []),
            args.grad_clip)
        if not torch.isfinite(gn):
            bad_batches += 1
            bad_grad += 1
            record_bad("grad", batch, d, rank, aux, gn)
            opt.zero_grad(set_to_none=True)
            if bad_batches % 50 == 1:
                print(f"[encoder] skip non-finite grad step={step} bad={bad_batches}", flush=True)
            continue
        opt.step()
        step += 1
        pair_views += int(batch["pair_i"].numel())
        row_views += int(batch["n_rows"])
        if step % args.log_every == 0:
            rec = {"step": step, "pair_views": pair_views, "row_views": row_views,
                   "loss": float(loss.detach().cpu()), "rank": float(rank.detach().cpu()),
                   "aux": float(aux.detach().cpu()), "grad_norm": float(gn),
                   "sec": round(time.time() - t_start, 1)}
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            print(f"[encoder] step={step} pairs={pair_views} rows={row_views} "
                  f"loss={rec['loss']:.4f} rank={rec['rank']:.4f} aux={rec['aux']:.4f} "
                  f"gn={gn:.2f} {human_time(time.time()-t_start)}", flush=True)
        if args.benchmark_views and step >= args.benchmark_views:
            break
        if args.eval_every and step % args.eval_every == 0:
            dval = encode_charts(model, head, cache, val_charts, args, device,
                                 batch_charts=8)
            vnll = val_nll(val, dval)
            print(f"[encoder] step={step} val_nll={vnll:.5f} pairs={pair_views}", flush=True)
            log_f.write(json.dumps({"step": step, "val_nll": vnll}) + "\n"); log_f.flush()
            if np.isfinite(vnll) and vnll < best_val - 1e-6:
                best_val = vnll
                best_step = step
                bad = 0
                save_ckpt("best")
            else:
                bad += 1
                if args.patience and bad >= args.patience:
                    print(f"[encoder] early stop at step {step}", flush=True)
                    break
        if args.save_every and step % args.save_every == 0:
            save_ckpt("last")

    # Always retain an explicit last checkpoint, even if the run ended between
    # save_every boundaries or before the first validation.
    save_ckpt("last")
    log_f.close()
    bad_log.close()

    export_path = out_dir / f"{args.export_checkpoint}.pt"
    if not export_path.exists():
        print(f"[encoder] {export_path} missing; falling back to last.pt", flush=True)
        export_path = out_dir / "last.pt"
    ck = torch.load(export_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    head.load_state_dict(ck["head"])
    if event_head is not None and ck.get("event_head") is not None:
        event_head.load_state_dict(ck["event_head"])
    if sig_head is not None and ck.get("sig_head") is not None:
        sig_head.load_state_dict(ck["sig_head"])
    model.eval()
    head.eval()
    dtest = encode_charts(model, head, cache, test_charts, args, device,
                          batch_charts=args.eval_batch_charts)
    dval = encode_charts(model, head, cache, val_charts, args, device,
                         batch_charts=args.eval_batch_charts)
    events_qa = None
    if args.task == "judge":
        events_qa = eval_event_proportions(model, head, event_head, cache, val,
                                           train, orig_counts, group_col, args,
                                           device, out_dir)
    for name, dmap in (("d_test", dtest), ("d_val", dval)):
        pd.DataFrame({"beatmap_id": list(dmap.keys()),
                      "d": list(dmap.values())}).to_parquet(
            out_dir / f"{name}.parquet", index=False)
    metrics = {
        "task": args.task, "aux": args.aux, "freeze": args.freeze,
        "init": args.init, "steps": step, "pair_views": pair_views,
        "row_views": row_views, "best_val_nll": best_val,
        "best_step": best_step,
        "n_train_charts": len(train_charts), "n_val_charts": len(val_charts),
        "n_test_charts": len(test_charts), "git_rev": git_rev(ROOT),
        "elapsed_sec": round(time.time() - t0, 1),
        "bad_batches": int(bad_batches),
        "bad_loss": int(bad_loss), "bad_grad": int(bad_grad),
        "checkpoint_export": str(export_path),
        "checkpoint_export_sha256": sha256_file(export_path),
        "checkpoint_best": str(out_dir / "best.pt")
        if (out_dir / "best.pt").exists() else None,
        "checkpoint_last": str(out_dir / "last.pt"),
        "eval_max_windows": int(args.max_windows),
        "eval_batch_charts": int(args.eval_batch_charts),
        "events_qa": events_qa,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1,
                                                     ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=1, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
