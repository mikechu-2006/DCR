#!/usr/bin/env python
"""Synthetic checks for the E2/E3 trainer pieces that are easy to get wrong.

Run:  python scripts/test_e2_e3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.step2_chart_encoder import (  # noqa: E402
    COUNT_COLS, event_table, judge_loss, shuffle_counts_within_player,
)

CHECKS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


class FakeCache:
    window_frames = 1600

    def __init__(self, events):
        self._events = events

    def get(self, cid):
        return self._events, 5000


def main() -> int:
    # 1. A hold judged on release must land on e.end, not on its onset.
    from types import SimpleNamespace
    ev = (SimpleNamespace(t=100, col=0, end=None),
          SimpleNamespace(t=200, col=1, end=3000))
    cache = FakeCache(ev)
    tail = event_table(cache, 1, 2, "tail")
    head = event_table(cache, 1, 2, "head")
    both = event_table(cache, 1, 2, "both")
    check("tail uses hold release", int(tail[0][1]) == 1 and int(tail[1][1]) == 700,
          f"win={int(tail[0][1])} frame={int(tail[1][1])}")
    check("head uses onset", int(head[0][1]) == 0 and int(head[1][1]) == 100,
          f"win={int(head[0][1])} frame={int(head[1][1])}")
    check("both has distinct head/tail positions",
          int(both[0][1]) != int(both[6][1]))

    # 2. Placebo swap preserves each chart N and the player's six marginals.
    rng = np.random.default_rng(0)
    n = 12
    cnt = rng.integers(0, 50, size=(n, 6)).astype(np.float64)
    cnt[:, 0] += 100
    df = pd.DataFrame({"player_id": 7, "player_id_base": 7,
                       "loss": rng.normal(size=n)})
    for k, c in enumerate(COUNT_COLS):
        df[c] = cnt[:, k]
    out = shuffle_counts_within_player(df, 0, "player_id_base")
    check("placebo preserves row N", np.array_equal(out.sum(1), cnt.sum(1)))
    check("placebo preserves player marginals", np.array_equal(out.sum(0), cnt.sum(0)))
    check("placebo changes the chart association", not np.array_equal(out, cnt))

    # 3. The single-grade branch is finite at the boundary rate=1.
    E = 20
    batch = {
        "events": {"chart": torch.zeros(E, dtype=torch.long)},
        "counts": torch.tensor([[0., 0., 0., 0., 0., 20.]]),
        "n_total": torch.tensor([20.]),
        "baseline": torch.tensor([[1/6] * 6]),
        "row_chart": torch.tensor([0]),
        "n_rows": 1,
    }
    logits = torch.zeros(E, 6, requires_grad=True)
    loss = judge_loss(logits, batch, torch.ones(6), single_idx=5)
    loss.backward()
    check("single-grade loss is finite at rate=1", bool(torch.isfinite(loss)))
    check("single-grade loss has finite grad", bool(torch.isfinite(logits.grad).all()))

    # 4. Six-grade branch is finite and differentiable.
    logits2 = torch.zeros(E, 6, requires_grad=True)
    loss2 = judge_loss(logits2, batch, torch.ones(6), single_idx=-1)
    loss2.backward()
    check("six-grade loss is finite", bool(torch.isfinite(loss2)))
    check("six-grade loss has finite grad", bool(torch.isfinite(logits2.grad).all()))

    ok = all(c for _, c in CHECKS)
    print(f"{sum(c for _, c in CHECKS)}/{len(CHECKS)} checks passed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
