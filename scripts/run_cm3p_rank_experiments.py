#!/usr/bin/env python
"""Run the CM3P rank-NLL experiment ladder on 1k and evaluate with the official runner.

Stages
  1. P1 (new pairs, same charts): M0 linear, M2 linear/extra/blend, M1 MLP sweep,
     best-config seed ensemble, M2 MLP+extra.
  2. P2 (new charts): 5-fold chart-disjoint linear and MLP.
  3. Robustness: the published audio CM3P table on its 4,042-chart subset (never
     mixed with the local table).
  4. Diagnostics: in-sample linear content ceiling.
  5. Merge every signal column into one table, then call scripts/step2_rank_eval.py
     with --split test --ceiling-fit all --common-maps --strict-pairs.

Usage:
  python scripts/run_cm3p_rank_experiments.py [--quick] [--force]

Outputs under results/cm3p_rank_1k/:
  <name>.parquet / <name>.json     per-model signal and metrics
  signals_1k.parquet               all model columns + external baselines
  models.csv / models.json         model-level train/val/test NLL
  rank_eval.json                   P1 official evaluation
  rank_eval_chart_disjoint.json    P2 official evaluation
  rank_eval_published.json         published-table subset evaluation
  summary.csv                      flattened captured / NLL / CIs
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cm3p_rank_1k"
STEP1 = ROOT / "data/processed/step1_1k_pairbest.parquet"
LOCAL = ROOT / "data/processed/chart_content_cm3p_local.parquet"
PUBLISHED = ROOT / "data/processed/chart_content_cm3p.parquet"
SIGNALS = ROOT / "data/processed/rank_signals_1k.parquet"
TRAIN = ROOT / "scripts" / "step2_cm3p_rank.py"
EVAL = ROOT / "scripts" / "step2_rank_eval.py"
BASELINES = ["msd_overall", "sunny_star", "daniel_star", "star", "notes_per_sec"]
CACHE = Path("/tmp/rank_cache")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], what: str) -> None:
    log(f"RUN {what}")
    p = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    if p.stdout:
        print(p.stdout, end="", flush=True)
    if p.returncode != 0:
        if p.stderr:
            print(p.stderr, end="", flush=True)
        raise RuntimeError(f"{what} failed (exit {p.returncode})")
    if p.stderr and "Warning" not in p.stderr and "OMP" not in p.stderr:
        print(p.stderr, end="", flush=True)


def model_cmd(name: str, args: list[str]) -> list[str]:
    return [sys.executable, str(TRAIN), "--name", name,
            "--out-signals", str(OUT / f"{name}.parquet"),
            "--out-json", str(OUT / f"{name}.json")] + args


def run_model(name: str, args: list[str], force: bool) -> dict:
    sig = OUT / f"{name}.parquet"
    js = OUT / f"{name}.json"
    if sig.exists() and js.exists() and not force:
        log(f"skip {name} (exists)")
        return json.loads(js.read_text())
    run(model_cmd(name, args), name)
    return json.loads((OUT / f"{name}.json").read_text())


def runner(signals: list[str], out_name: str, force: bool) -> dict:
    out = OUT / out_name
    if out.exists() and not force:
        log(f"skip runner {out_name} (exists)")
        return json.loads(out.read_text())
    cmd = [sys.executable, str(EVAL),
           "--in", str(STEP1),
           "--signal-table", str(OUT / "signals_1k.parquet"),
           "--signals", ",".join(signals),
           "--split", "test", "--ceiling-fit", "all",
           "--common-maps", "--strict-pairs",
           "--folds", "2", "--bootstrap", "400",
           "--cache-dir", str(CACHE), "--out", str(out)]
    run(cmd, f"rank_eval {out_name}")
    return json.loads(out.read_text())


def flatten(evals: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for ev, d in evals.items():
        for sig, r in d.get("signals", {}).items():
            if "error" in r:
                rows.append({"eval": ev, "signal": sig, "error": r["error"]})
                continue
            boot = r.get("bootstrap", {})
            rows.append({
                "eval": ev, "signal": sig,
                "n_maps": r["n_maps"], "n_instances": r["n_instances"],
                "n_instances_used": r.get("n_instances_used"),
                "nll_per_instance": r["nll_per_instance"],
                "bits_saved": r["bits_saved"], "captured": r["captured"],
                "captured_all": r.get("captured_all"),
                "nll_all_per_instance": r.get("nll_all_per_instance"),
                "ceiling_nll": r["ceiling_nll"], "ceiling_n": r["ceiling_n"],
                "ceiling_coverage": r["ceiling_coverage"],
                "lw": r["per_player"]["lw"], "macro": r["per_player"]["macro"],
                "fisher": r["per_player"]["fisher_l3"],
                "beta": r["link"]["beta"],
                "captured_ci": boot.get("captured"),
                "nll_ci": boot.get("nll"),
                "lw_ci": boot.get("lw"),
                "strict_pairs": r.get("strict_pairs"),
                "ceiling_fit": r.get("ceiling_fit"),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke run of every stage (not for results)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    quick = args.quick
    epochs = 3 if quick else 100
    patience = 5 if quick else 20
    hid = 16 if quick else 64

    # ---------------------------------------------------------------- model specs
    common = ["--val-frac", "0.2", "--val-seed", "0"]
    mlp_cfgs = {
        "cm3p_mlp_h64_L1": {"hid": hid, "depth": 1, "dropout": 0.0,
                             "wd": 1e-4, "lr": 3e-3},
        "cm3p_mlp_h64_L2": {"hid": hid, "depth": 2, "dropout": 0.1,
                             "wd": 1e-4, "lr": 3e-3},
        "cm3p_mlp_h128_L1": {"hid": 2 * hid, "depth": 1, "dropout": 0.1,
                              "wd": 1e-4, "lr": 3e-3},
    }
    if quick:
        mlp_cfgs = {"cm3p_mlp_h16_L1": {"hid": 16, "depth": 1, "dropout": 0.0,
                                         "wd": 1e-4, "lr": 1e-3}}

    def mlp_args(cfg, n_epochs, pat, seeds=1, model="mlp", extra=None):
        a = ["--model", model,
             "--hid", str(cfg["hid"]), "--depth", str(cfg["depth"]),
             "--dropout", str(cfg["dropout"]), "--wd", str(cfg["wd"]),
             "--lr", str(cfg["lr"]), "--epochs", str(n_epochs),
             "--patience", str(pat), "--seeds", str(seeds)] + common
        if extra:
            a += extra
        return a

    p1 = [
        ("cm3p_linear", ["--model", "linear", "--lam-grid",
                         "1e-2" if quick else "1e-3,1e-2,1e-1"] + common),
        ("cm3p_linear_extra", ["--model", "linear_extra", "--lam-grid", "1e-2"] + common),
        ("ext_blend", ["--model", "ext_blend", "--lam-grid", "1e-2"] + common),
    ]
    for nm, cfg in mlp_cfgs.items():
        p1.append((nm, mlp_args(cfg, epochs, patience)))
    mlp_specs = [(nm, mlp_args(cfg, epochs, patience)) for nm, cfg in mlp_cfgs.items()]

    model_js = {}
    for name, extra in p1:
        model_js[name] = run_model(name, extra, args.force)

    # best MLP by validation NLL (never by test)
    mlp_names = [n for n, _ in mlp_specs
                 if model_js.get(n, {}).get("val_nll_per_instance") is not None]
    best_name = min(mlp_names, key=lambda n: model_js[n]["val_nll_per_instance"])
    best_cfg = mlp_cfgs[best_name]
    best_epoch = int(model_js[best_name].get("best_epoch") or epochs)
    log(f"best MLP by val NLL: {best_name} "
        f"(val={model_js[best_name]['val_nll_per_instance']:.5f}, epoch={best_epoch})")

    if not quick:
        # seed ensemble and external residual: fix the epoch count from
        # validation, then train directly on all train rows (--val-frac 0)
        ens_args = mlp_args(best_cfg, best_epoch, 10 ** 9, seeds=3)
        ens_args += ["--val-frac", "0"]
        model_js["cm3p_mlp_ens"] = run_model("cm3p_mlp_ens", ens_args, args.force)
        ext_args = mlp_args(best_cfg, best_epoch, 10 ** 9, seeds=1,
                            model="mlp_extra")
        ext_args += ["--val-frac", "0"]
        model_js["cm3p_mlp_extra"] = run_model("cm3p_mlp_extra", ext_args, args.force)

    # in-sample content ceiling (linear; MLP ceiling is omitted because it overfits)
    ceil_args = ["--model", "linear", "--lam-grid", "1e-2",
                 "--content-ceiling"] + common
    model_js["cm3p_linear_ceilingcheck"] = run_model(
        "cm3p_linear_ceilingcheck", ceil_args, args.force)
    if not quick:
        ceil_extra = ["--model", "linear_extra", "--lam-grid", "1e-2",
                      "--content-ceiling"] + common
        model_js["cm3p_linear_extra_ceilingcheck"] = run_model(
            "cm3p_linear_extra_ceilingcheck", ceil_extra, args.force)

    # ------------------------------------------------------- chart-disjoint (P2)
    p2_names = []
    cd_lin = ["--model", "linear", "--chart-only", "--chart-folds",
              "2" if quick else "5", "--chart-seed", "0",
              "--val-frac", "0", "--lam", "1e-2"]
    model_js["cm3p_linear_cd"] = run_model("cm3p_linear_cd", cd_lin, args.force)
    p2_names.append("cm3p_linear_cd")
    cd_lin_extra = ["--model", "linear_extra", "--chart-only", "--chart-folds",
                    "2" if quick else "5", "--chart-seed", "0",
                    "--val-frac", "0", "--lam", "1e-2"]
    model_js["cm3p_linear_extra_cd"] = run_model(
        "cm3p_linear_extra_cd", cd_lin_extra, args.force)
    p2_names.append("cm3p_linear_extra_cd")
    if not quick:
        cd_mlp = mlp_args(best_cfg, best_epoch, 10 ** 9, seeds=1,
                          extra=["--chart-only", "--chart-folds", "5",
                                 "--chart-seed", "0", "--val-frac", "0"])
        model_js["cm3p_mlp_cd"] = run_model("cm3p_mlp_cd", cd_mlp, args.force)
        p2_names.append("cm3p_mlp_cd")

    # ------------------------------------------------ published-table robustness
    pub_name = None
    if not quick:
        pub_args = ["--model", "linear", "--content-path", str(PUBLISHED),
                    "--lam-grid", "1e-2"] + common
        model_js["cm3p_linear_pub"] = run_model("cm3p_linear_pub", pub_args, args.force)
        pub_name = "cm3p_linear_pub"

    # ------------------------------------------------------------ merge signals
    step1 = pd.read_parquet(STEP1, columns=["beatmap_id"])
    maps = np.sort(step1["beatmap_id"].unique())
    master = pd.DataFrame({"beatmap_id": maps})
    for name in list(model_js):
        p = OUT / f"{name}.parquet"
        if not p.exists():
            continue
        s = pd.read_parquet(p)
        if name not in s.columns:
            log(f"warn: {p} has no column {name}")
            continue
        master = master.merge(s[["beatmap_id", name]], on="beatmap_id", how="left")
    sig = pd.read_parquet(SIGNALS).set_index("beatmap_id")
    for c in BASELINES:
        master[c] = sig[c].reindex(master["beatmap_id"]).to_numpy()
    master.to_parquet(OUT / "signals_1k.parquet", index=False)
    log(f"wrote signals_1k.parquet {master.shape}")

    # --------------------------------------------------------------- evaluations
    p1_names = [n for n, _ in p1] + [n for n in ("cm3p_mlp_ens", "cm3p_mlp_extra")
                                     if n in model_js]
    p1_names = [n for n in p1_names if n in master.columns]
    p1_names += BASELINES
    p2_names = [n for n in p2_names if n in master.columns] + BASELINES
    evals = {}
    evals["p1"] = runner(p1_names, "rank_eval.json", args.force)
    if p2_names:
        evals["p2_chart_disjoint"] = runner(
            p2_names, "rank_eval_chart_disjoint.json", args.force)
    if pub_name:
        evals["published_subset"] = runner(
            [pub_name] + BASELINES, "rank_eval_published.json", args.force)

    summary = flatten(evals)
    summary.to_csv(OUT / "summary.csv", index=False)

    # model-level table + content-ceiling captured
    ceil = None
    for d in evals.values():
        for r in d.get("signals", {}).values():
            if "ceiling_nll" in r:
                ceil = float(r["ceiling_nll"])
                break
        if ceil is not None:
            break
    mrows = []
    for name, js in model_js.items():
        row = {"name": name, "model": js.get("model"),
               "content": Path(js.get("content_path", "")).name,
               "n_maps": js.get("n_maps"),
               "train_nll": js.get("train_nll_per_instance"),
               "val_nll": js.get("val_nll_per_instance"),
               "test_nll": js.get("test_nll_per_instance"),
               "test_bits": js.get("test_bits_saved"),
               "best_lam": js.get("best_lam"), "best_epoch": js.get("best_epoch"),
               "chart_folds": js.get("chart_folds"),
               "mlp_params": js.get("mlp_params"),
               "seconds": js.get("seconds")}
        if ceil is not None and js.get("test_nll_per_instance") is not None:
            row["captured_test"] = (float(np.log(2.0)) - js["test_nll_per_instance"]) / \
                (float(np.log(2.0)) - ceil)
        if js.get("content_ceiling_test_nll") is not None and ceil is not None:
            row["content_ceiling_nll"] = js["content_ceiling_test_nll"]
            row["captured_content"] = (float(np.log(2.0)) - js["content_ceiling_test_nll"]) / \
                (float(np.log(2.0)) - ceil)
        mrows.append(row)
    pd.DataFrame(mrows).to_csv(OUT / "models.csv", index=False)
    (OUT / "models.json").write_text(json.dumps(model_js, indent=2, ensure_ascii=False))

    print("\n=== summary ===")
    cols = ["eval", "signal", "nll_per_instance", "captured", "captured_all",
            "lw", "macro", "beta", "n_instances_used"]
    print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))
    log(f"done in {time.time() - t0:.0f}s; outputs in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
