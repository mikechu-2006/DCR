#!/usr/bin/env python
"""Synthetic alignment check for scripts/rank_paired_delta.py.

Run:  python scripts/test_rank_paired_delta.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcr import rank_metrics as rm  # noqa: E402
from scripts.step2_rank_eval import _save_ceiling  # noqa: E402

CHECKS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def run(cmd: list[str], cwd: Path) -> None:
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if p.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="e1fix_pair_") as td:
        d = Path(td)
        rng = np.random.default_rng(3)
        # Four player segments but only two player_id_base clusters: a
        # segment-level bootstrap would report 4 clusters; the base bootstrap
        # must report 2.
        players = [10, 11, 12, 13]
        bases = [100, 100, 200, 200]
        rows = []
        for p, b in zip(players, bases):
            for c in range(6):
                rows.append(dict(player_id=p, player_id_base=b,
                                 beatmap_id=1000 + c,
                                 loss=float(-rng.normal()), split="test"))
        inp = d / "in.parquet"
        pd.DataFrame(rows).to_parquet(inp, index=False)

        # Two signals on the same six charts.
        sig = d / "sig.parquet"
        pd.DataFrame({"beatmap_id": 1000 + np.arange(6),
                      "d_a": rng.normal(size=6),
                      "d_b": rng.normal(size=6)}).to_parquet(sig, index=False)

        df = pd.read_parquet(inp)
        item = (df.beatmap_id.to_numpy() - 1000).astype(np.int32)
        perf = -df.loss.to_numpy(np.float64)
        groups = [np.flatnonzero(df.player_id.to_numpy() == p) for p in players]
        clusters = np.array(bases, dtype=np.int64)
        ceil = rm.cross_fit_bt(groups, item, perf, 6, clusters, folds=2,
                               seed=0, min_l=1)
        cache = d / "ceiling.npz"
        _save_ceiling(cache, ceil, np.arange(1000, 1006))

        ev = d / "eval.json"
        run([sys.executable, str(ROOT / "scripts/step2_rank_eval.py"),
             "--in", str(inp), "--signal-table", str(sig),
             "--signals", "d_a,d_b", "--split", "test",
             "--ceiling-fit", "all", "--strict-pairs", "--no-cache",
             "--bootstrap", "20", "--out", str(ev)], ROOT)
        out = d / "paired.json"
        run([sys.executable, str(ROOT / "scripts/rank_paired_delta.py"),
             "--in", str(inp), "--a", str(sig), "--b", str(sig),
             "--a-col", "d_a", "--b-col", "d_b", "--split", "test",
             "--strict-pairs", "--ceiling-cache", str(cache),
             "--bootstrap", "20", "--out", str(out)], ROOT)

        e = json.loads(ev.read_text())["signals"]
        p = json.loads(out.read_text())
        check("strict pair count matches main table",
              p["n_pairs_strict"] == e["d_a"]["n_strict"],
              f"{p['n_pairs_strict']} vs {e['d_a']['n_strict']}")
        check("paired NLL A matches main strict NLL",
              abs(p["nll_a"] - e["d_a"]["nll_per_instance"]) < 1e-10,
              f"{p['nll_a']:.12f} vs {e['d_a']['nll_per_instance']:.12f}")
        check("paired NLL B matches main strict NLL",
              abs(p["nll_b"] - e["d_b"]["nll_per_instance"]) < 1e-10,
              f"{p['nll_b']:.12f} vs {e['d_b']['nll_per_instance']:.12f}")
        check("delta equals NLL(A)-NLL(B)",
              abs(p["delta"] - (p["nll_a"] - p["nll_b"])) < 1e-12)
        check("player bootstrap clusters by base, not segment",
              p["n_players"] == 4 and p["n_player_clusters"] == 2,
              f"players={p['n_players']} clusters={p['n_player_clusters']}")
        check("player and chart CIs are present",
              p["ci_player"] is not None and p["ci_chart"] is not None)

        # A fold that has not seen one chart must drop only the pairs that
        # touch that chart: 6 all-pairs -> 4 strict pairs here.
        rows2 = []
        for pid, base in ((20, 100), (21, 200)):
            for c in range(3):
                rows2.append(dict(player_id=pid, player_id_base=base,
                                  beatmap_id=2000 + c, loss=float(-rng.normal()),
                                  split="test"))
        inp2 = d / "in2.parquet"
        pd.DataFrame(rows2).to_parquet(inp2, index=False)
        sig2 = d / "sig2.parquet"
        pd.DataFrame({"beatmap_id": 2000 + np.arange(3),
                      "d_a": rng.normal(size=3),
                      "d_b": rng.normal(size=3)}).to_parquet(sig2, index=False)
        cache2 = d / "ceiling2.npz"
        np.savez_compressed(
            cache2,
            map_ids=np.arange(2000, 2003, dtype=np.int64),
            cluster_ids=np.array([100, 200], dtype=np.int64),
            fold_of_cluster=np.array([0, 1], dtype=np.int64),
            seen_fold_0=np.array([True, True, False]),
            seen_fold_1=np.array([True, True, True]),
            b_fold_0=np.zeros(3), b_fold_1=np.zeros(3),
            nll=np.float64(0.0), n=np.int64(4))
        out2 = d / "paired2.json"
        run([sys.executable, str(ROOT / "scripts/rank_paired_delta.py"),
             "--in", str(inp2), "--a", str(sig2), "--b", str(sig2),
             "--a-col", "d_a", "--b-col", "d_b", "--split", "test",
             "--strict-pairs", "--ceiling-cache", str(cache2),
             "--bootstrap", "10", "--out", str(out2)], ROOT)
        p2 = json.loads(out2.read_text())
        check("strict mask drops unseen-chart pairs",
              p2["n_pairs_all"] == 6 and p2["n_pairs_strict"] == 4,
              f"all={p2['n_pairs_all']} strict={p2['n_pairs_strict']}")

    ok = all(v for _, v in CHECKS)
    print(f"\n{sum(v for _, v in CHECKS)}/{len(CHECKS)} checks passed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
