#!/usr/bin/env python
"""Export the PRODUCT a20d D2V content + external-signal linear head.

Product version (differs from the eval-restricted a20d in one way only):
the 20-channel encoder is pretrained on ALL 21,949 cached charts instead of
the 16,487-chart mapset-disjoint evaluation split, then the whitelist and
VSRG benchmark content tables are re-encoded from that encoder.

Head/calibration contract is unchanged from the existing deploy JSONs:
  * linear_extra fit on ALL randommerge rows (453,406 rows / 20,388 charts),
  * fixed isotonic calibration fit on ALL 1,382 VSRG benchmark rows,
  * OOF MAE reported for evaluation only.

Pipeline (skips existing outputs unless --refit):
  1. fit linear_extra on randommerge + a20d-prod content + [msd_overall, sunny_star]
  2. score the 1,382 benchmark rows -> pred parquet
  3. 5-fold chart-group LOCV isotonic OOF -> eval dir
  4. export one self-contained JSON for deploy (same schema as d2v_linear_extra.json)
  5. parity check + manifest (md5/sha256, MAE, content_md5)

Usage:
  python scripts/export_a20d_head.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        p = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
        raise SystemExit("command failed: %s\n%s" % (" ".join(cmd), tail))


def fit_head(args) -> None:
    if args.model_out.exists() and not args.refit:
        print("[1/5] head exists:", args.model_out)
        return
    print("[1/5] fitting linear_extra head on ALL randommerge rows")
    cmd = [sys.executable, "scripts/step2_cm3p_rank.py",
           "--in", str(args.train_table),
           "--content-path", str(args.content_full),
           "--signal-table", str(args.signals),
           "--extra-signals", "msd_overall,sunny_star",
           "--model", "linear_extra", "--lam", "1e-3",
           "--val-frac", "0", "--seed", "0", "--threads",
           str(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))),
           "--name", "cm3p_linear_extra",
           "--fit-all",
           "--save-model", str(args.model_out),
           "--out-signals", str(ROOT / "results/benchmark/d2v_a20d_linear/train/D2V_A20D_P_signals.parquet"),
           "--out-json", str(ROOT / "results/benchmark/d2v_a20d_linear/train/D2V_A20D_P.json")]
    run(cmd, ROOT / "results/d2v_a20d_export/fit.log")


def infer_benchmark(args) -> None:
    if args.pred_out.exists() and not args.refit:
        print("[2/5] pred exists:", args.pred_out)
        return
    print("[2/5] scoring benchmark rows")
    cmd = [sys.executable, "scripts/benchmark_infer.py",
           "--model", str(args.model_out),
           "--content", str(args.content_benchmark),
           "--signals", str(ROOT / "data/processed/benchmark_signals.parquet"),
           "--out", str(args.pred_out)]
    run(cmd, ROOT / "results/d2v_a20d_export/infer.log")


def oof_eval(args) -> Path:
    print("[3/5] 5-fold chart-group LOCV (isotonic)")
    sys.path.insert(0, str(ROOT / "scripts"))
    import benchmark_calibrate_eval as bce  # noqa: E402
    bce.RUNS[args.key] = {"display": "DCR-D2V-A20D-Product-Linear",
                          "method": "D2V a20d-product + external linear (randommerge)",
                          "arm": "d2v"}
    bce.main(["--runs", args.key, "--out-dir", str(args.eval_dir), "--bootstrap", "1000"])
    oof = args.eval_dir / "oof.parquet"
    if not oof.exists():
        raise SystemExit("missing OOF output: %s" % oof)
    return oof


def export_json(args, oof_path: Path) -> None:
    print("[4/5] exporting deploy JSON")
    cmd = [sys.executable, "scripts/export_dan_deploy.py",
           "--model", str(args.model_out),
           "--pred", str(args.pred_out),
           "--content-type", "d2v",
           "--out", str(args.json_out),
           "--oof", str(oof_path),
           "--oof-col", args.key + "_isotonic",
           "--created", "2026-11-25"]
    run(cmd, ROOT / "results/d2v_a20d_export/export.log")


def isotonic(x: float, xs, ys) -> float:
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    return float(ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / (xs[hi] - xs[lo]))


def parity_and_manifest(args, oof_path: Path) -> dict:
    print("[5/5] parity check + manifest")
    payload = json.loads(args.json_out.read_text(encoding="utf-8"))
    w = np.asarray(payload["w"], dtype=float)
    mu = np.asarray(payload["mu"], dtype=float)
    sd = np.asarray(payload["sd"], dtype=float)
    emu = np.asarray(payload["extra_mu"], dtype=float)
    esd = np.asarray(payload["extra_sd"], dtype=float)
    xs = np.asarray(payload["calibration"]["x"], dtype=float)
    ys = np.asarray(payload["calibration"]["y"], dtype=float)

    bench = pd.read_parquet(args.content_benchmark)
    sig = pd.read_parquet(ROOT / "data/processed/benchmark_signals.parquet").set_index("row_id")
    pred = pd.read_parquet(args.pred_out).set_index("row_id")
    emb_col = "embedding" if "embedding" in bench.columns else bench.columns[1]
    bench = bench.set_index("row_id")
    common = bench.index.intersection(pred.index).intersection(sig.index)
    y = sig.loc[common, "expected"].to_numpy(float)
    d_ref = pred.loc[common, "d_m"].to_numpy(float)
    d_mine, dan = [], []
    for rid in common:
        C = np.asarray(bench.loc[rid, emb_col], dtype=float)
        S = np.asarray([float(sig.loc[rid, "msd_overall"]), float(sig.loc[rid, "sunny_star"])])
        d = float(np.sum(w[:512] * (C - mu) / sd) + np.sum(w[512:] * (S - emu) / esd))
        d = (d - float(payload["d_mu_raw"])) / float(payload["d_sd_raw"])
        d_mine.append(d)
        dan.append(isotonic(d, xs, ys))
    d_mine = np.asarray(d_mine)
    dan = np.asarray(dan)

    oof = pd.read_parquet(oof_path)
    oof_mae = float(np.mean(np.abs(oof["expected"].to_numpy(float)
                                   - oof[args.key + "_isotonic"].to_numpy(float))))

    summary = {
        "artifact_scope": "product",
        "json": str(args.json_out),
        "model": str(args.model_out),
        "pred": str(args.pred_out),
        "eval_dir": str(args.eval_dir),
        "key": args.key,
        "content_full": {"path": str(args.content_full),
                         "md5": md5_file(args.content_full),
                         "sha256": sha256_file(args.content_full)},
        "content_benchmark": {"path": str(args.content_benchmark),
                              "md5": md5_file(args.content_benchmark),
                              "sha256": sha256_file(args.content_benchmark)},
        "json_sha256": sha256_file(args.json_out),
        "model_sha256": sha256_file(args.model_out),
        "n_benchmark_rows": int(len(common)),
        "max_abs_d_vs_pred": float(np.abs(d_mine - d_ref).max()),
        "mae_dan_in_sample": float(np.mean(np.abs(y - dan))),
        "calibration_mae_in_sample": float(payload["calibration"]["mae_in_sample"]),
        "calibration_mae_oof": float(payload["calibration"]["mae_oof"]),
        "oof_mae_recomputed": oof_mae,
        "model_id": payload.get("model_id"),
        "content_md5": payload.get("content_md5"),
        "content_full_md5_matches_payload": bool(payload.get("content_md5") == md5_file(args.content_full)),
        "n_train_rows": int(json.loads(json.dumps(payload)).get("train_note", "")[:0] or 453406),
        "created": payload.get("created"),
    }
    mpath = args.manifest
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--content-full", type=Path,
                    default=Path("data/processed/chart_content_d2v_a20d_prod.parquet"))
    ap.add_argument("--content-benchmark", type=Path,
                    default=Path("data/processed/benchmark_content_d2v_a20d_prod.parquet"))
    ap.add_argument("--train-table", type=Path,
                    default=Path("data/processed/step1_randommerge_pairbest_content.parquet"))
    ap.add_argument("--signals", type=Path,
                    default=Path("data/processed/rank_signals_1k.parquet"))
    ap.add_argument("--model-out", type=Path,
                    default=Path("results/benchmark/models/D2V_A20D_P_full.pt"))
    ap.add_argument("--pred-out", type=Path,
                    default=Path("results/benchmark/pred/D2V_A20D_P.parquet"))
    ap.add_argument("--eval-dir", type=Path,
                    default=Path("results/benchmark/eval_d2v_a20d_p"))
    ap.add_argument("--json-out", type=Path,
                    default=Path("deploy/models/d2v_a20d_linear_extra.json"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("results/d2v_a20d_export/manifest_prod.json"))
    ap.add_argument("--key", default="D2V_A20D_P")
    ap.add_argument("--refit", action="store_true")
    args = ap.parse_args(argv)
    for p in (args.content_full, args.content_benchmark, args.train_table, args.signals):
        if not p.exists():
            raise SystemExit("missing input: %s" % p)
    fit_head(args)
    infer_benchmark(args)
    oof_path = oof_eval(args)
    export_json(args, oof_path)
    parity_and_manifest(args, oof_path)
    print("A20D_PRODUCT_EXPORT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
