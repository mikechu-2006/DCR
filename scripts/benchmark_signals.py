#!/usr/bin/env python
"""Compute benchmark-side external signals with the same versions as DCR.

  sunny_star  : osumania_map_analyser commit 448ed7c, headless Node entry
  msd_overall : minacalc-standalone calc_version 505, goal 0.93, rate 1.0

    python scripts/benchmark_signals.py \
        --manifest data/processed/benchmark_manifest.parquet \
        --out data/processed/benchmark_signals.parquet

Also re-computes 32 DCR charts and compares against rank_signals_1k.parquet;
that check is the contract that the benchmark-side numbers are on the same
scale as the training features.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_step0h():
    path = ROOT / "legacy/scripts/step0h_minacalc_msd.py"
    spec = importlib.util.spec_from_file_location("step0h", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_sunny(manifest: pd.DataFrame, extracted: Path, out_jsonl: Path) -> pd.DataFrame:
    node = shutil.which("node")
    if node is None:
        raise SystemExit("node not found; the headless Sunny entry needs Node.js")
    recs = []
    for _, row in manifest.iterrows():
        recs.append({"row_id": row["row_id"],
                     "osu_path": str(extracted / row["osu_path"])})
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n",
                         encoding="utf-8")
    proc = subprocess.run([node, str(ROOT / "scripts/benchmark_sunny.mjs"),
                           str(out_jsonl), str(ROOT)],
                          capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"node sunny failed ({proc.returncode}): {proc.stderr[-1000:]}")
    rows = []
    for line in proc.stdout.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default=".cache/benchmark/VSRG-DanEstimation-Benchmark")
    ap.add_argument("--manifest", default="data/processed/benchmark_manifest.parquet")
    ap.add_argument("--out", default="data/processed/benchmark_signals.parquet")
    ap.add_argument("--lib", default="external/minacalc-standalone/build/libminacalc.so")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--check-dcr", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    benchmark = Path(args.benchmark).resolve()
    extracted = benchmark / "extracted"
    manifest = pd.read_parquet(args.manifest)
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"[skip] {out} exists (use --force to recompute)", flush=True)
        return 0

    print(f"[sunny] {len(manifest)} charts", flush=True)
    sunny = run_sunny(manifest, extracted, Path("data/processed/benchmark_sunny_input.jsonl"))
    sunny = sunny.rename(columns={"error": "sunny_error"})
    if "row_id" not in sunny.columns:
        raise SystemExit("sunny output has no row_id")

    print(f"[msd] {len(manifest)} charts", flush=True)
    step0 = load_step0h()
    backend = step0.LibBackend(str(ROOT / args.lib), 1.0, 0.93)
    print(f"[msd] backend={backend.name} version={backend.version}", flush=True)

    def one(row):
        path = extracted / row["osu_path"]
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            keycount, rows, n_notes, n_holds = step0.parse_osu_mania(text)
            if keycount != 4:
                return {"row_id": row["row_id"], "msd_error": f"keycount={keycount}"}
            vals = backend.msd(rows)
            return {"row_id": row["row_id"],
                    **{f"msd_{k}": vals[k] for k in step0.SKILLS},
                    "msd_n_notes": n_notes, "msd_n_holds": n_holds,
                    "msd_hold_ratio": (n_holds / (n_notes + n_holds)) if (n_notes + n_holds) else 0.0,
                    "msd_error": None}
        except Exception as exc:  # noqa: BLE001
            return {"row_id": row["row_id"], "msd_error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        msd = pd.DataFrame(list(pool.map(one, [r for _, r in manifest.iterrows()])))

    sig = manifest[["row_id", "bid", "name", "pattern", "family", "expected", "rate"]].merge(
        sunny, on="row_id", how="left").merge(msd, on="row_id", how="left")
    sig["sunny_error"] = sig.get("sunny_error", pd.Series([None] * len(sig))).fillna("")
    sig["msd_error"] = sig.get("msd_error", pd.Series([None] * len(sig))).fillna("")
    out.parent.mkdir(parents=True, exist_ok=True)
    sig.to_parquet(out, index=False)
    meta = {
        "n_rows": int(len(sig)),
        "sunny_ok": int(sig["sunny_star"].notna().sum()),
        "msd_ok": int(sig["msd_overall"].notna().sum()),
        "sunny_error_sample": sig.loc[sig["sunny_star"].isna(), ["row_id", "name", "sunny_error"]].head(10).to_dict("records"),
        "msd_error_sample": sig.loc[sig["msd_overall"].isna(), ["row_id", "name", "msd_error"]].head(10).to_dict("records"),
        "msd_calc_version": int(backend.version),
        "msd_goal": 0.93,
        "msd_rate": 1.0,
    }

    # ---- DCR reconciliation -------------------------------------------------
    check = None
    n_check = int(args.check_dcr)
    if n_check > 0:
        ref = pd.read_parquet(ROOT / "data/processed/rank_signals_1k.parquet")
        ref = ref[["beatmap_id", "sunny_star", "msd_overall"]].dropna()
        cache = ROOT / "data/osu_cache"
        ref = ref[ref["beatmap_id"].apply(lambda b: (cache / f"{int(b)}.osu").exists())]
        ref = ref.sample(frac=1.0, random_state=0).head(n_check)
        dcr_manifest = pd.DataFrame({
            "row_id": [f"dcr:{int(b)}" for b in ref["beatmap_id"]],
            "osu_path": [str((cache / f"{int(b)}.osu").relative_to(ROOT)) for b in ref["beatmap_id"]],
        })
        # run_sunny expects paths relative to the extracted benchmark dir; make absolute
        recs = [{"row_id": r["row_id"], "osu_path": r["osu_path"]} for _, r in dcr_manifest.iterrows()]
        tmp = Path("/tmp/benchmark_sunny_dcr.jsonl")
        tmp.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        node = shutil.which("node")
        proc = subprocess.run([node, str(ROOT / "scripts/benchmark_sunny.mjs"),
                               str(tmp), str(ROOT)], capture_output=True, text=True, cwd=ROOT)
        sun = pd.DataFrame([json.loads(x) for x in proc.stdout.splitlines() if x.strip()])
        sun = sun.rename(columns={"error": "sunny_error"})
        msd_rows = []
        for _, r in dcr_manifest.iterrows():
            text = (ROOT / r["osu_path"]).read_text(encoding="utf-8", errors="replace")
            kc, rows, nn, nh = step0.parse_osu_mania(text)
            vals = backend.msd(rows)
            msd_rows.append({"row_id": r["row_id"], **{f"msd_{k}": vals[k] for k in step0.SKILLS}})
        msd_dcr = pd.DataFrame(msd_rows)
        merged = sun.merge(msd_dcr, on="row_id")
        merged["bid"] = merged["row_id"].str.replace("dcr:", "", regex=False).astype(int)
        merged = merged.merge(ref, left_on="bid", right_on="beatmap_id", how="left")
        d_sun = (merged["sunny_star_x"] - merged["sunny_star_y"]).abs()
        d_msd = (merged["msd_overall_x"] - merged["msd_overall_y"]).abs()
        check = {
            "n": int(len(merged)),
            "sunny_max_abs": float(d_sun.max()),
            "sunny_mean_abs": float(d_sun.mean()),
            "msd_max_abs": float(d_msd.max()),
            "msd_mean_abs": float(d_msd.mean()),
        }
        print(f"[check-dcr] n={check['n']} sunny max|d|={check['sunny_max_abs']:.3g} "
              f"msd max|d|={check['msd_max_abs']:.3g}", flush=True)
        meta["dcr_check"] = check

    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    print(f"[done] {out}: sunny_ok={meta['sunny_ok']}/{len(sig)} "
          f"msd_ok={meta['msd_ok']}/{len(sig)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
