#!/usr/bin/env python
"""Synthetic end-to-end smoke test: per-seed heads -> pool features -> linear_extra.

No real window cache is needed.  The script builds a tiny 48-chart / 48-player
dataset, runs the Part B driver stage by stage, then exercises the Part A
multiseed runner (--content-col / --assemble-only / --force-eval) on the same
tables.  It is intentionally small (a few seconds per model) but goes through
the real scripts, subprocess boundaries and parquet round-trips.

Run:
    python scripts/smoke_pool_randommerge.py [--keep]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        fh.write(" ".join(map(str, cmd)) + "\n\n")
        fh.flush()
        p = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), stdout=fh,
                           stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-40:])
        raise SystemExit(f"FAILED ({p.returncode}): {' '.join(map(str, cmd))}\n{tail}")


def make_data(td: Path) -> dict:
    rng = np.random.default_rng(7)
    charts = np.arange(1000, 1048, dtype=np.int64)
    players = np.arange(2000, 2048, dtype=np.int64)
    n_c, n_p = charts.size, players.size
    chart_eff = rng.normal(0, 0.5, n_c)
    player_eff = rng.normal(0, 0.5, n_p)
    rows = []
    for pi, p in enumerate(players):
        for ci, c in enumerate(charts):
            rows.append({"player_id": int(p), "player_id_base": int(p),
                         "beatmap_id": int(c),
                         "loss": float(1.0 + chart_eff[ci] - player_eff[pi]
                                       + rng.normal(0, 0.3))})
    df = pd.DataFrame(rows)
    # P2: 70/30 row split, per chart force >=10 test rows
    p2 = df.copy()
    split = np.empty(len(p2), dtype=object)
    for c, idx in p2.groupby("beatmap_id").indices.items():
        idx = rng.permutation(idx)
        n_test = max(10, int(round(0.30 * idx.size)))
        split[idx[:n_test]] = "test"
        split[idx[n_test:]] = "train"
    p2["split"] = split
    p2.to_parquet(td / "step1_smoke_p2_s99.parquet", index=False)
    # P4: 29 train players / 19 test players (all charts appear in both)
    train_players = set(players[:29].tolist())
    p4 = df.copy()
    p4["split"] = np.where(p4.player_id.isin(train_players), "train", "test")
    p4.to_parquet(td / "step1_smoke_p4_s99.parquet", index=False)
    # fixed + extra signals
    pd.DataFrame({
        "beatmap_id": charts,
        "msd_overall": chart_eff + rng.normal(0, 0.05, n_c),
        "sunny_star": -chart_eff + rng.normal(0, 0.05, n_c),
        "daniel_star": chart_eff + rng.normal(0, 0.05, n_c),
        "star": rng.normal(0, 0.5, n_c),
    }).to_parquet(td / "signals.parquet", index=False)
    # windows + index
    win = td / "win"
    win.mkdir()
    idx_rows = []
    for c in charts:
        n = int(rng.integers(3, 7))
        W = rng.normal(size=(n, 512)).astype(np.float32)
        W /= np.linalg.norm(W, axis=1, keepdims=True)
        np.save(win / f"{int(c)}.npy", W.astype(np.float16))
        idx_rows.append({"beatmap_id": int(c), "n_win": n,
                         "mean_norm": float(np.linalg.norm(W.mean(0))),
                         "n_core_tokens_dropped": 0, "columns_emitted": False,
                         "song_length_s": 60.0})
    pd.DataFrame(idx_rows).to_parquet(td / "index.parquet", index=False)
    print(f"[smoke] charts={n_c} players={n_p} rows={len(df)}", flush=True)
    return {"charts": charts, "players": players}


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""),
          flush=True)
    if not ok:
        raise SystemExit(f"smoke failed: {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the temp dir")
    args = ap.parse_args()
    td = Path(tempfile.mkdtemp(prefix="dcr_pool_smoke_"))
    print(f"[smoke] temp dir: {td}", flush=True)
    try:
        make_data(td)
        base = [PY, "scripts/run_pool_randommerge.py",
                "--feature-dir", td / "feat", "--out-dir", td / "pool",
                "--window-index", td / "index.parquet",
                "--window-dir", td / "win",
                "--signal-table", td / "signals.parquet",
                "--cache-dir", td / "cache",
                "--split-p2", td / "step1_smoke_p2",
                "--split-p4", td / "step1_smoke_p4",
                "--threads", "2", "--seeds", "99",
                "--min-l", "5", "--bootstrap", "40"]
        logs = td / "logs"
        for stage in ("shared", "features", "train", "eval", "summarize"):
            print(f"[smoke] driver --stage {stage}", flush=True)
            run(base + ["--stage", stage], logs / f"driver_{stage}.log")

        # feature widths
        for name, want in (("a0", 512), ("a0l", 514), ("a1", 1024)):
            d = pd.read_parquet(td / "feat" / "shared" / f"{name}.parquet",
                                columns=["features"])
            check(f"shared {name} width", int(d.features.iloc[0].size) == want)
        for name, want in (("a2", 519), ("a2b", 519), ("a2c", 521), ("a3", 521)):
            for proto in ("p2", "p4"):
                p = td / "feat" / proto / "s99" / f"{name}.parquet"
                d = pd.read_parquet(p, columns=["features"])
                check(f"{proto} {name} width", int(d.features.iloc[0].size) == want
                      and len(d) == 48)
        # head provenance
        z = np.load(td / "pool" / "heads" / "a0_heads_p2_s99.npz", allow_pickle=False)
        meta = {k: z[k].item() for k in z.files
                if k in ("inp", "chart_seed", "lam", "val_frac")}
        check("head provenance", Path(str(meta["inp"])).name
              == "step1_smoke_p2_s99.parquet"
              and int(meta["chart_seed"]) == 0 and float(meta["lam"]) == 1e-3
              and int(np.asarray(z["chart_folds"]).item()) == 5, str(meta))
        # signals + eval
        for proto in ("p2", "p4"):
            ev = json.loads((td / "pool" / proto / "s99" / "rank_eval.json").read_text())
            want = {f"pool_{v}" for v in
                    ("a0", "a0l", "a1", "a2", "a2b", "a2c", "a3")}
            check(f"{proto} eval has all variants", want <= set(ev["signals"]),
                  str(sorted(ev["signals"])))
            pdj = json.loads((td / "pool" / proto / "s99" / "paired_delta.json").read_text())
            check(f"{proto} paired has a2b-a0l",
                  "pool_a2b:pool_a0l" in pdj.get("pairs", {}),
                  str(list(pdj.get("pairs", {}))[:3]))
        check("summary files", all((td / "pool" / n).exists() for n in
                                   ("summary.csv", "summary_mean_std.csv",
                                    "summary_pairs.csv", "summary.md")))

        # Part A runner: --content-col + --assemble-only + --force-eval
        parta = td / "parta"
        pcmd = [PY, "scripts/run_p2_p4_multiseed.py",
                "--protocols", "p2", "--seeds", "99",
                "--models", "linear_extra,linear",
                "--split-p2", td / "step1_smoke_p2",
                "--split-p4", td / "step1_smoke_p4",
                "--out-dir", parta,
                "--content-path", td / "feat" / "shared" / "a0l.parquet",
                "--content-col", "features",
                "--signal-table", td / "signals.parquet",
                "--cache-dir", td / "cache",
                "--threads", "2", "--force-eval"]
        print("[smoke] Part A runner first pass (train + eval)", flush=True)
        run(pcmd, logs / "parta_run1.log")
        ev = json.loads((parta / "p2" / "s99" / "rank_eval.json").read_text())
        check("partA eval has both models",
              {"cm3p_linear_extra", "cm3p_linear"} <= set(ev["signals"]))
        sig = pd.read_parquet(parta / "p2" / "s99" / "signals.parquet")
        check("partA assembled both signals",
              {"cm3p_linear_extra", "cm3p_linear"} <= set(sig.columns))
        print("[smoke] Part A runner second pass (--assemble-only --force-eval)", flush=True)
        run(pcmd + ["--assemble-only"], logs / "parta_run2.log")
        ev = json.loads((parta / "p2" / "s99" / "rank_eval.json").read_text())
        check("partA rerun kept both models",
              {"cm3p_linear_extra", "cm3p_linear"} <= set(ev["signals"]))
        check("partA aggregate", (parta / "summary.csv").exists())
        print("\nSMOKE OK", flush=True)
        return 0
    finally:
        if not args.keep:
            import shutil
            shutil.rmtree(td, ignore_errors=True)
        else:
            print(f"[smoke] kept {td}")


if __name__ == "__main__":
    raise SystemExit(main())
