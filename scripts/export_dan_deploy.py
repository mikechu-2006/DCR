#!/usr/bin/env python
"""Export a linear_extra checkpoint + full-VSRG isotonic calibration to one JSON.

The JSON is consumed by deploy/dan_predict.py.  The calibration is fit on ALL
1,382 benchmark rows (fixed deployment weights), not the 5-fold OOF split.

    python scripts/export_dan_deploy.py \
        --model results/benchmark/models/RM_L_full.pt \
        --pred results/benchmark/pred/RM_L.parquet \
        --content-type cm3p \
        --out deploy/models/cm3p_linear_extra.json \
        --oof results/benchmark/eval_rm/oof.parquet \
        --oof-col RM_L_isotonic
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--signals", default="data/processed/benchmark_signals.parquet")
    ap.add_argument("--content-type", required=True, choices=["cm3p", "d2v"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--oof", default=None)
    ap.add_argument("--oof-col", default=None)
    ap.add_argument("--created", default="2026-11-23")
    args = ap.parse_args(argv)

    payload = torch.load(args.model, map_location="cpu", weights_only=False)
    if payload.get("model") not in ("linear", "linear_extra", "ext_blend"):
        raise SystemExit(f"{args.model}: not a linear model ({payload.get('model')})")
    pred = pd.read_parquet(args.pred)[["row_id", "d_m"]]
    sig = pd.read_parquet(args.signals)[["row_id", "expected"]]
    df = pred.merge(sig, on="row_id", how="inner")
    if len(df) != 1382:
        raise SystemExit(f"expected 1,382 calibration rows, got {len(df)}")
    if df["d_m"].isna().any() or df["expected"].isna().any():
        raise SystemExit("calibration rows contain NaN")

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(df["d_m"].to_numpy(float), df["expected"].to_numpy(float))
    got = ir.predict(df["d_m"].to_numpy(float))
    mae_in = float(np.mean(np.abs(got - df["expected"].to_numpy(float))))

    oof_mae = None
    if args.oof and args.oof_col:
        oof = pd.read_parquet(args.oof)
        if args.oof_col not in oof.columns:
            raise SystemExit(f"{args.oof}: missing {args.oof_col}")
        oof_mae = float(np.mean(np.abs(
            oof["expected"].to_numpy(float) - oof[args.oof_col].to_numpy(float))))

    w = np.asarray(payload["w"], dtype=np.float64)
    xs = np.asarray(ir.X_thresholds_, dtype=np.float64)
    ys = np.asarray(ir.y_thresholds_, dtype=np.float64)
    h = hashlib.sha256()
    h.update(w.tobytes())
    h.update(xs.tobytes())
    h.update(ys.tobytes())
    h.update(str(payload.get("content_md5", "")).encode())
    model_id = h.hexdigest()[:16]

    out = {
        "schema_version": 1,
        "content_type": args.content_type,
        "model_type": payload["model"],
        "model_id": model_id,
        "content_dim": int(payload["content_dim"]),
        "extra_names": list(payload["extra_names"]),
        "mu": [float(x) for x in np.asarray(payload["mu"], dtype=np.float64)],
        "sd": [float(x) for x in np.asarray(payload["sd"], dtype=np.float64)],
        "extra_mu": [float(x) for x in np.asarray(payload["extra_mu"], dtype=np.float64)],
        "extra_sd": [float(x) for x in np.asarray(payload["extra_sd"], dtype=np.float64)],
        "w": [float(x) for x in w],
        "d_mu_raw": float(payload["d_mu_raw"]),
        "d_sd_raw": float(payload["d_sd_raw"]),
        "content_md5": payload.get("content_md5"),
        "train_git_commit": payload.get("git_commit"),
        "train_note": "randommerge --fit-all; 453,406 rows / 20,388 charts",
        "calibration": {
            "kind": "isotonic",
            "source": "VSRG benchmark, all 1,382 rows, full fit",
            "n": int(len(df)),
            "x": [float(x) for x in xs],
            "y": [float(x) for x in ys],
            "mae_in_sample": mae_in,
            "mae_oof": oof_mae,
        },
        "created": args.created,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[export] {path} model_id={model_id} iso_knots={len(xs)} "
          f"mae_in={mae_in:.4f} mae_oof={oof_mae}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
