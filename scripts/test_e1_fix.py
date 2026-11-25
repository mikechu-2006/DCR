#!/usr/bin/env python
"""Synthetic self-checks for E1_fix F1/F2 (no data/ needed).

Run:  python scripts/test_e1_fix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.step2_chart_encoder import (  # noqa: E402
    ChartHead, atomic_torch_save, bt_loss, preserve_eval, snapshot_state,
)

CHECKS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def main() -> int:
    torch.manual_seed(0)
    z = torch.randn(16, 64)

    # P1: train-mode dropout is non-deterministic; preserve_eval fixes it and
    # restores the original flags.
    head = ChartHead(64, hid=32, dropout=0.5)
    head.train()
    a = head(z)
    b = head(z)
    check("train-mode dropout is non-deterministic",
          not torch.allclose(a, b), f"max={float((a-b).abs().max()):.3f}")

    with preserve_eval(head):
        c = head(z)
        d = head(z)
        check("preserve_eval makes head deterministic",
              torch.equal(c, d), f"max={float((c-d).abs().max()):.3f}")
    check("preserve_eval restores training flag", head.training)

    frozen = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
    frozen.eval()
    with preserve_eval(frozen):
        pass
    check("preserve_eval restores eval flag", not frozen.training)

    # P2: snapshots must not alias live parameters.
    snap = snapshot_state(frozen)
    before = snap["0.weight"].clone()
    with torch.no_grad():
        frozen[0].weight.add_(1.0)
    check("snapshot_state is detached from live weights",
          torch.equal(snap["0.weight"], before),
          f"max_drift={float((snap['0.weight']-frozen[0].weight).abs().max()):.3f}")

    # Atomic save leaves no .tmp and can be reloaded.
    tmpdir = Path("/tmp/e1fix_selftest")
    tmpdir.mkdir(exist_ok=True)
    p = tmpdir / "x.pt"
    atomic_torch_save({"x": 1}, p)
    check("atomic_torch_save writes a readable file",
          p.exists() and not Path(str(p) + ".tmp").exists()
          and torch.load(p, weights_only=False)["x"] == 1)

    # P3: extreme bf16 differences must be finite.
    for dtype in (torch.float32, torch.bfloat16):
        dvec = torch.tensor([0.0, 8.0], dtype=dtype)
        pi = torch.tensor([0], dtype=torch.long)
        pj = torch.tensor([1], dtype=torch.long)
        for yv, expected in ((1.0, 3.35e-4), (0.0, 8.0003), (0.5, 4.0003)):
            y = torch.tensor([yv], dtype=dtype)
            loss = bt_loss(dvec, pi, pj, y)
            check(f"bt_loss finite {dtype} y={yv}",
                  bool(torch.isfinite(loss)) and abs(float(loss) - expected) < 2e-3,
                  f"{float(loss):.6f}")
    d2 = torch.tensor([-1000.0, 1000.0], dtype=torch.bfloat16, requires_grad=True)
    loss = bt_loss(d2, torch.tensor([0]), torch.tensor([1]), torch.tensor([0.0]))
    loss.backward()
    check("bt_loss extreme gradient finite",
          bool(torch.isfinite(d2.grad).all()), f"{d2.grad.tolist()}")

    ok = all(v for _, v in CHECKS)
    print(f"\n{sum(v for _, v in CHECKS)}/{len(CHECKS)} checks passed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
