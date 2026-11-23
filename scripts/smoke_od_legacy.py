#!/usr/bin/env python
"""Synthetic smoke test for scripts/run_od_legacy.py (L0-L3).

Builds a tiny 40-chart / 40-player legacy-like content table and signal table,
then exercises the train -> eval -> summarize chain with the real scripts.
No real data or HPC is needed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
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


def make_data(td: Path) -> None:
    rng = np.random.default_rng(11)
    charts = np.arange(3000, 3040, dtype=np.int64)
    players = np.arange(4000, 4040, dtype=np.int64)
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
    p2 = df.copy()
    split = np.empty(len(p2), dtype=object)
    for c, idx in p2.groupby("beatmap_id").indices.items():
        idx = rng.permutation(idx)
        n_test = max(8, int(round(0.30 * idx.size)))
        split[idx[:n_test]] = "test"
        split[idx[n_test:]] = "train"
    p2["split"] = split
    p2.to_parquet(td / "step1_smoke_p2_s99.parquet", index=False)
    p4 = df.copy()
    p4["split"] = np.where(p4.player_id.isin(set(players[:24].tolist())),
                           "train", "test")
    p4.to_parquet(td / "step1_smoke_p4_s99.parquet", index=False)
    content = pd.DataFrame({
        "beatmap_id": charts,
        "embedding": [rng.normal(size=512).astype(np.float32) for _ in charts],
    })
    content.to_parquet(td / "legacy_content.parquet", index=False)
    od = np.clip(rng.normal(7.5, 0.7, n_c), 0, 10)
    signals = pd.DataFrame({
        "beatmap_id": charts,
        "msd_overall": chart_eff + rng.normal(0, 0.05, n_c),
        "sunny_star": -chart_eff + rng.normal(0, 0.05, n_c),
        "daniel_star": chart_eff + rng.normal(0, 0.05, n_c),
        "star": rng.normal(0, 0.5, n_c),
        "od": od,
        "log_n": np.log(rng.integers(3, 20, n_c)),
    })
    signals.to_parquet(td / "signals.parquet", index=False)
    print(f"[smoke] charts={n_c} players={n_p} rows={len(df)}", flush=True)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""),
          flush=True)
    if not ok:
        raise SystemExit(f"smoke failed: {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    td = Path(tempfile.mkdtemp(prefix="dcr_od_smoke_"))
    print(f"[smoke] temp dir: {td}", flush=True)
    try:
        make_data(td)
        base = [PY, "scripts/run_od_legacy.py",
                "--stage", "all",
                "--protocols", "p2,p4", "--seeds", "99",
                "--arms", "l0,l1,l2,l3",
                "--content-path", td / "legacy_content.parquet",
                "--signal-table", td / "signals.parquet",
                "--split-p2", td / "step1_smoke_p2",
                "--split-p4", td / "step1_smoke_p4",
                "--out-dir", td / "out", "--cache-dir", td / "cache",
                "--threads", "2", "--min-l", "5", "--bootstrap", "40"]
        run(base, td / "logs" / "all.log")
        for proto in ("p2", "p4"):
            d = td / "out" / proto / "s99"
            for arm in ("l0", "l1", "l2", "l3"):
                check(f"{proto} {arm} signal",
                      (d / "signals" / f"od_{arm}.parquet").exists())
            ev = json.loads((d / "rank_eval.json").read_text())
            want = {f"od_{a}" for a in ("l0", "l1", "l2", "l3")}
            check(f"{proto} eval has arms", want <= set(ev["signals"]),
                  str(sorted(ev["signals"])))
            for arm in ("l0", "l3"):
                s = ev["signals"][f"od_{arm}"]
                check(f"{proto} {arm} metrics",
                      s.get("nll_per_instance") is not None
                      and s.get("pearson_bhat") is not None
                      and s.get("rho_bhat") is not None)
            pdj = json.loads((d / "paired_delta.json").read_text())
            check(f"{proto} paired", "od_l3:od_l0" in pdj.get("pairs", {}))
        check("summary files", all((td / "out" / n).exists() for n in
                                   ("summary.csv", "summary_mean_std.csv",
                                    "summary_pairs.csv", "summary.md")))
        md = (td / "out" / "summary.md").read_text()
        check("summary md", "ΔPearson" in md and "L0" in md, md[-200:])
        print("\nSMOKE OK", flush=True)
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(td, ignore_errors=True)
        else:
            print(f"[smoke] kept {td}")


if __name__ == "__main__":
    raise SystemExit(main())
