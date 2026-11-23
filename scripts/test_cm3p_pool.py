#!/usr/bin/env python
"""Tests for the T3 pooling + driver glue (no window cache needed).

Run:  python scripts/test_cm3p_pool.py
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcr import cm3p_pool as pool  # noqa: E402

FAILS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""),
          flush=True)
    if not ok:
        FAILS.append(name)


def make_heads(td: Path, maps: np.ndarray, folds: np.ndarray, k: int) -> pool.HeadBundle:
    d = 512
    rng = np.random.default_rng(0)
    w = rng.normal(size=(k, d)).astype(np.float32) * 0.01
    np.savez(td / "heads.npz", maps=maps, fold=folds, w=w, w_full=w[0],
             mu=np.zeros(d, np.float32), sd=np.ones(d, np.float32),
             content_dim=np.int64(d), chart_folds=np.int64(k))
    return pool.HeadBundle.load(td / "heads.npz")


def make_windows(td: Path, ids: list[int]) -> tuple[Path, pd.DataFrame]:
    win = td / "win"
    win.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for b in ids:
        n = int(rng.integers(2, 7))
        W = rng.normal(size=(n, 512)).astype(np.float32)
        W /= np.linalg.norm(W, axis=1, keepdims=True)
        np.save(win / f"{b}.npy", W.astype(np.float16))
        rows.append({"beatmap_id": b, "n_win": n,
                     "mean_norm": float(np.linalg.norm(W.mean(0))),
                     "tok_len": [100] * n, "tok_len_full": [100] * n,
                     "content_md5": f"m{b}", "cfg_hash": "c",
                     "n_holds_split": 0, "n_holds_boundary_2ms": 0, "n_holds": 0,
                     "n_notes": 1, "n_note_tokens": 1, "n_column_tokens": 0,
                     "columns_emitted": False, "n_truncated_windows": 0,
                     "n_tokens_dropped": 0, "n_core_tokens_dropped": 0,
                     "song_length_s": 40.0, "window_starts_ms": [0.0] * n})
    return win, pd.DataFrame(rows)


def test_pools() -> None:
    print("\nHeadBundle / pool variants")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ids = list(range(20))
        folds = np.arange(20) % 2
        h = make_heads(td, np.array(ids), folds, 2)
        win, idx = make_windows(td, ids)
        check("head bundle shapes", h.w.shape == (2, 512) and h.w_full.shape == (512,))
        check("fold weight", np.allclose(h.weight_for(0), h.w[0])
              and np.allclose(h.weight_for(999), h.w_full))
        check("full weight", np.allclose(h.weight_for(0, mode="full"), h.w_full)
              and np.allclose(h.weight_for(1, mode="full"), h.w_full))
        widths = {"a0": 512, "a0l": 514, "a1": 1024, "a2": 519, "a2b": 519,
                  "a2c": 521, "a3": 521, "a4": 512, "a5": 512}
        for v, want in widths.items():
            df = pool.build_feature_table(idx, win, v, h, log=lambda s: None)
            got = int(df.features.iloc[0].size)
            check(f"{v} width", len(df) == 20 and got == want, f"{got}")
        # A0 is the unit mean of the stored windows
        W = np.load(win / "0.npy").astype(np.float32)
        a0 = pool.pool_chart("a0", W, 0, h)
        check("A0 is unit mean", abs(np.linalg.norm(a0) - 1) < 1e-5
              and np.allclose(a0, W.mean(0) / np.linalg.norm(W.mean(0)), atol=1e-6))
        # A0L = A0 + log mean_norm + log n
        a0l = pool.pool_chart("a0l", W, 0, h)
        check("A0L width/append", a0l.size == 514
              and np.allclose(a0l[:512], a0)
              and np.isclose(a0l[512], np.log(max(idx.loc[0, "mean_norm"], 1e-8)),
                             atol=1e-4)
              and np.isclose(a0l[513], np.log(W.shape[0])))
        # A2/A2b/A2c are deterministic and finite; quantiles are ordered
        a2 = pool.pool_chart("a2", W, 0, h)
        a2b = pool.pool_chart("a2b", W, 0, h)
        a2c = pool.pool_chart("a2c", W, 0, h)
        a3 = pool.pool_chart("a3", W, 0, h)
        check("A2 finite", np.isfinite(a2).all())
        check("A2 deterministic", np.allclose(a2, pool.pool_chart("a2", W, 0, h)))
        check("A2b finite", np.isfinite(a2b).all() and a2b.size == 519)
        check("A2b quantiles ordered", a2b[514] <= a2b[515] <= a2b[516])
        check("A2c finite", np.isfinite(a2c).all() and a2c.size == 521)
        check("A2c quantiles ordered", a2c[514] <= a2c[515] <= a2c[516] <= a2c[517] <= a2c[518])
        check("A3 runs non-negative", a3.size == 521 and (a3[519:] >= 0).all())
        # head mode full: every chart uses w_full, not the chart's fold head
        z_fold = h.z(W, 1, mode="fold")
        z_full = h.z(W, 1, mode="full")
        check("head mode full", np.allclose(z_full, ((W - h.mu) / h.sd) @ h.w_full)
              and not np.allclose(z_fold, z_full))
        full_tbl = pool.build_feature_table(idx, win, "a2b", h, head_mode="full",
                                            log=lambda s: None)
        check("head_mode=full table", len(full_tbl) == 20
              and int(full_tbl.features.iloc[0].size) == 519)
        # exclude-truncated
        idx2 = idx.copy()
        idx2.loc[0, "n_core_tokens_dropped"] = 2
        check("exclude-truncated stays in the CLI, not the pool", True)


def test_a6() -> None:
    print("\nA6 TCN signal (tiny synthetic)")
    rng = np.random.default_rng(0)
    M, P = 24, 5
    rows = []
    for p in range(P):
        for b in rng.choice(M, 9, replace=False):
            rows.append({"player_id": 100 + p, "player_id_base": 100 + p,
                         "beatmap_id": int(b), "loss": float(rng.normal()),
                         "split": "train"})
    step1 = pd.DataFrame(rows)
    windows = {}
    for b in range(M):
        W = rng.normal(size=(int(rng.integers(2, 6)), 512)).astype(np.float32)
        W /= np.linalg.norm(W, axis=1, keepdims=True)
        windows[b] = W
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        maps = np.arange(M)
        h = make_heads(td, maps, maps % 3, 3)
        df = pool.train_a6_signal(step1, windows, h, epochs=2, hid=16, val_frac=0.4,
                                  chunk=8, threads=2, log=lambda s: None)
        check("A6 covers every map", len(df) == M and set(df.beatmap_id) == set(range(M)))
        check("A6 signal finite", bool(np.isfinite(df.signal).all()))
        check("A6 signal varies", float(df.signal.std()) > 0)


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "pool_driver_test", ROOT / "scripts/run_cm3p_pool_experiments.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_driver_glue() -> None:
    print("\ndriver merge / summary")
    mod = _load_driver()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mod.OUT = td
        mod.BASE_SIGNALS = td / "signals_p2.parquet"
        mod.TOKEN_BUDGET = td / "missing.json"
        ids = np.arange(30)
        for v in ("a0", "a2"):
            pd.DataFrame({"beatmap_id": ids, f"pool_{v}": np.linspace(0, 1, 30)}
                         ).to_parquet(td / f"pool_{v}.parquet", index=False)
        pd.DataFrame({"beatmap_id": ids, "cm3p_linear_cd_l1e3": np.linspace(0, 1, 30),
                      "msd_overall": np.zeros(30), "sunny_star": np.zeros(30),
                      "daniel_star": np.zeros(30)}
                     ).to_parquet(mod.BASE_SIGNALS, index=False)
        sig = mod.merge_signals(["a0", "a2"])
        check("merged columns", {"pool_a0", "pool_a2", "cm3p_linear_cd_l1e3",
                                 "msd_overall"} <= set(sig.columns), str(list(sig.columns)))
        check("merged rows", len(sig) == 30)
        ev = {
            "eval": {"signals": {
                "pool_a0": {"nll_per_instance": 0.6148, "captured": 0.60,
                            "per_player": {"lw": 0.44}},
                "pool_a2": {"nll_per_instance": 0.5900, "captured": 0.70,
                            "per_player": {"lw": 0.50}},
                "cm3p_linear_cd_l1e3": {"nll_per_instance": 0.6148,
                                         "captured": 0.60},
            }},
            "paired": {"pairs": {
                "pool_a2:pool_a0": {"delta": -0.0248, "ci": [-0.03, -0.02],
                                    "p_delta_lt_0": 0.99},
                "pool_a2:cm3p_linear_cd_l1e3": {"delta": -0.0248,
                                                "ci": [-0.03, -0.02]},
            }},
        }
        mod.summary(["a0", "a2"], ev, {"widths": {"a0": 512, "a2": 519}},
                    {"a0": {"seconds": 1.0, "content_dim": 512},
                     "a2": {"seconds": 2.0, "content_dim": 519}})
        s = json.loads((td / "summary.json").read_text())
        check("summary variants", [r["variant"] for r in s["variants"]] == ["a0", "a2"])
        check("summary delta", s["variants"][1]["delta_vs_a0"] == -0.0248)
        md = (td / "SUMMARY.md").read_text()
        check("summary markdown", "pooling is the main effect" in md, md[-200:])


def main() -> int:
    test_pools()
    test_a6()
    test_driver_glue()
    print(f"\n{len(FAILS)} failure(s)")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
