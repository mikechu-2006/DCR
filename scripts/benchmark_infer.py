#!/usr/bin/env python
"""Score benchmark rows with a model saved by step2_cm3p_rank.py.

The model payload contains the content/extra normalization and the training
d_mu/d_sd.  We deliberately do NOT re-standardize on the benchmark set: that
would make each arm's scale depend on the benchmark composition.

    python scripts/benchmark_infer.py \
        --model results/benchmark/models/M1_L.pt \
        --content data/processed/benchmark_content_legacy.parquet \
        --signals data/processed/benchmark_signals.parquet \
        --out results/benchmark/pred/M1_L.parquet
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_rank_module():
    path = ROOT / "scripts/step2_cm3p_rank.py"
    spec = importlib.util.spec_from_file_location("cm3p_rank", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True,
                    help="one or more saved .pt models; predictions are averaged")
    ap.add_argument("--content", required=True)
    ap.add_argument("--signals", default="data/processed/benchmark_signals.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default="data/processed/benchmark_manifest.parquet")
    args = ap.parse_args(argv)

    content = pd.read_parquet(args.content)
    signals = pd.read_parquet(args.signals)
    content = content.drop_duplicates("row_id").set_index("row_id")
    signals = signals.drop_duplicates("row_id").set_index("row_id")
    common = content.index.intersection(signals.index)
    if len(common) == 0:
        raise SystemExit("no common row_id between content and signals")
    C = np.stack(content.loc[common, "embedding"].to_numpy()).astype(np.float32)
    print(f"[data] content={len(content):,} signals={len(signals):,} common={len(common):,}",
          flush=True)

    rank_mod = load_rank_module()
    preds = []
    for model_path in args.model:
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        mu = np.asarray(payload["mu"], dtype=np.float32)
        sd = np.asarray(payload["sd"], dtype=np.float32)
        Cs = (C - mu) / sd
        model_name = payload["model"]
        use_extra = model_name in ("linear_extra", "mlp_extra", "ext_blend")
        extra_names = (payload.get("extra_names") or []) if use_extra else []
        if extra_names:
            missing = [c for c in extra_names if c not in signals.columns]
            if missing:
                raise SystemExit(f"{args.signals}: missing extra signals {missing}")
            S = signals.loc[common, extra_names].to_numpy(dtype=np.float64)
            if not np.isfinite(S).all():
                raise SystemExit(f"non-finite extra signals on {int((~np.isfinite(S)).any(axis=1)).sum()} rows")
            em = np.asarray(payload["extra_mu"], dtype=np.float64)
            es = np.asarray(payload["extra_sd"], dtype=np.float64)
            S = (S - em) / es
            X = np.hstack([Cs, S]).astype(np.float32)
        else:
            X = Cs.astype(np.float32)

        if model_name in ("linear", "linear_extra", "ext_blend"):
            d_raw = X.astype(np.float64) @ np.asarray(payload["w"], dtype=np.float64)
        else:
            scorer = rank_mod.Scorer(X.shape[1], int(payload["hid"]),
                                     int(payload["depth"]), float(payload["dropout"]))
            scorer.model.load_state_dict(payload["state_dict"])
            scorer.model.eval()
            with torch.no_grad():
                d_raw = scorer.forward(torch.as_tensor(X, dtype=torch.float32)).cpu().numpy()
            d_raw = d_raw.astype(np.float64)

        d_mu = float(payload.get("d_mu_raw", 0.0))
        d_sd = float(payload.get("d_sd_raw", 1.0))
        d = (d_raw - d_mu) / (d_sd if d_sd else 1.0)
        preds.append(d)

    d_mean = np.mean(preds, axis=0)
    out = pd.DataFrame({"row_id": list(common), "d_m": d_mean})
    for i, p in enumerate(preds):
        out[f"d_m_{i}"] = p
    out = out.merge(content[["n_windows"]].reset_index(), on="row_id", how="left")
    out = out.merge(signals.reset_index()[["row_id", "family", "pattern", "expected", "rate"]],
                    on="row_id", how="left") if "family" in signals.columns else out
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"[done] {args.out}: {len(out)} rows, n_models={len(preds)}, "
          f"d_m mean={out.d_m.mean():+.4f} sd={out.d_m.std():.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
