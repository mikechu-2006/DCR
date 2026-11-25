#!/usr/bin/env python
"""build_meta_content -- chart-level metadata summary content table.

Builds a 12-d per-chart summary vector from the D2V structured-attribute audit
(data/processed/d2v_attr_audit_per_chart.parquet) and writes it in the
standard content-table contract:

    beatmap_id  : int64
    embedding   : float32 np.array of length 12
    n_windows   : 1            (one chart-level summary, not a window table)
    checksum_ok : True         (pure metadata, no audio checksum involved)
    source      : "meta_summary"
    d2v_rev     : "n/a"

The table is a metadata-only reference arm for the D2V attribute probes: it
contains no audio or D2V embedding, only the attributes the new channels are
meant to encode.  It covers all 21,949 osu!mania 4K charts the audit covers.

Features (order is part of the contract):
    log2_bpm_p50          log2(clip(bpm_p50, 1, 1000))
    log1p_n_red           log1p(n_red)
    log1p_n_sv_changes    log1p(n_sv_changes)
    kiai_frac             kiai_frac
    hold_frac_gt1s        hold_frac_gt1s
    log_hold_ms_p50       log(hold_ms_p50)
    hs_whistle_frac       hs_whistle_frac
    hs_finish_frac        hs_finish_frac
    hs_clap_frac          hs_clap_frac
    vol_resolved_p50_100  vol_resolved_p50 / 100
    log1p_notes_per_sec   log1p(notes_per_sec)
    log1p_span_s          log1p(span_s)

Missing values (only hold_ms_p50 is ever missing: charts with no holds) are
imputed with the median over the step1 training charts, so the imputed value is
the neutral point under the rank probe's train-set standardisation.  bpm_p50 is
clipped because the audit contains a handful of degenerate timing values
(0 BPM red lines / >20k BPM aliases) that would otherwise make log2 non-finite.

Usage:
    python scripts/build_meta_content.py
    python scripts/build_meta_content.py --out data/processed/chart_content_meta_summary.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "log2_bpm_p50",
    "log1p_n_red",
    "log1p_n_sv_changes",
    "kiai_frac",
    "hold_frac_gt1s",
    "log_hold_ms_p50",
    "hs_whistle_frac",
    "hs_finish_frac",
    "hs_clap_frac",
    "vol_resolved_p50_100",
    "log1p_notes_per_sec",
    "log1p_span_s",
]

BPM_CLIP = (1.0, 1000.0)


def _raw_features(df: pd.DataFrame) -> tuple[np.ndarray, list[dict]]:
    """Return the raw (pre-imputation) feature matrix and per-feature clip stats."""
    n = len(df)
    X = np.full((n, len(FEATURES)), np.nan, dtype=np.float64)
    notes: list[dict] = []
    bpm = df["bpm_p50"].to_numpy(np.float64)
    lo, hi = BPM_CLIP
    n_low = int((bpm < lo).sum())
    n_high = int((bpm > hi).sum())
    bpm_c = np.clip(bpm, lo, hi)
    notes.append({"feature": "log2_bpm_p50", "n_clipped_low": n_low,
                  "n_clipped_high": n_high, "clip": list(BPM_CLIP)})
    X[:, 0] = np.log2(bpm_c)
    X[:, 1] = np.log1p(df["n_red"].to_numpy(np.float64))
    X[:, 2] = np.log1p(df["n_sv_changes"].to_numpy(np.float64))
    X[:, 3] = df["kiai_frac"].to_numpy(np.float64)
    X[:, 4] = df["hold_frac_gt1s"].to_numpy(np.float64)
    X[:, 5] = np.log(df["hold_ms_p50"].to_numpy(np.float64))
    X[:, 6] = df["hs_whistle_frac"].to_numpy(np.float64)
    X[:, 7] = df["hs_finish_frac"].to_numpy(np.float64)
    X[:, 8] = df["hs_clap_frac"].to_numpy(np.float64)
    X[:, 9] = df["vol_resolved_p50"].to_numpy(np.float64) / 100.0
    X[:, 10] = np.log1p(df["notes_per_sec"].to_numpy(np.float64))
    X[:, 11] = np.log1p(df["span_s"].to_numpy(np.float64))
    return X, notes


def _train_chart_ids(step1_path: Path) -> set[int]:
    if not step1_path.exists():
        return set()
    s = pd.read_parquet(step1_path)
    if "split" not in s.columns:
        return set(int(x) for x in s["beatmap_id"].unique())
    return set(int(x) for x in s.loc[s["split"] == "train", "beatmap_id"].unique())


def build(audit_path: Path, out_path: Path, step1_path: Path) -> int:
    audit = pd.read_parquet(audit_path)
    if "beatmap_id" not in audit.columns:
        raise SystemExit(f"{audit_path}: no beatmap_id column")
    audit = audit.sort_values("beatmap_id").reset_index(drop=True)
    bmap = audit["beatmap_id"].to_numpy(np.int64)

    X, clip_notes = _raw_features(audit)

    train_ids = _train_chart_ids(step1_path)
    in_train = np.isin(bmap, np.fromiter(train_ids, dtype=np.int64,
                                        count=len(train_ids)))
    fill_stats = []
    n_filled_total = 0
    for j, name in enumerate(FEATURES):
        col = X[:, j]
        miss = ~np.isfinite(col)
        n_missing = int(miss.sum())
        med_train = float(np.nanmedian(col[in_train])) if in_train.any() else np.nan
        med_all = float(np.nanmedian(col)) if np.isfinite(col).any() else 0.0
        med = med_train if np.isfinite(med_train) else med_all
        if n_missing:
            col[miss] = med
            n_filled_total += n_missing
        fill_stats.append({"feature": name, "n_missing": n_missing,
                           "fill_value": med,
                           "fill_source": ("step1_train_median"
                                           if np.isfinite(med_train) else "all_median")})
    if not np.isfinite(X).all():
        bad = int((~np.isfinite(X)).sum())
        raise SystemExit(f"{bad} non-finite values after imputation")

    emb = [np.asarray(X[i], dtype=np.float32) for i in range(len(X))]
    out = pd.DataFrame({
        "beatmap_id": bmap,
        "embedding": pd.Series(emb, dtype=object),
        "n_windows": np.ones(len(X), dtype=np.int64),
        "checksum_ok": np.ones(len(X), dtype=bool),
        "source": "meta_summary",
        "d2v_rev": "n/a",
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    meta = {
        "path": str(out_path),
        "n_charts": int(len(out)),
        "dim": len(FEATURES),
        "features": FEATURES,
        "source": "meta_summary",
        "audit_path": str(audit_path),
        "step1_path": str(step1_path),
        "n_train_charts_for_imputation": int(in_train.sum()),
        "n_imputed_values": n_filled_total,
        "clip_notes": clip_notes,
        "fill_stats": fill_stats,
        "feature_stats": {
            name: {"mean": float(X[:, j].mean()), "std": float(X[:, j].std()),
                   "min": float(X[:, j].min()), "max": float(X[:, j].max())}
            for j, name in enumerate(FEATURES)
        },
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[out] {out_path} rows={len(out):,} dim={len(FEATURES)} "
          f"-> {out_path.stat().st_size / 1e6:.2f} MB")
    print(f"[impute] {n_filled_total} values, train charts={int(in_train.sum()):,}")
    for s in fill_stats:
        if s["n_missing"]:
            print(f"  {s['feature']}: filled {s['n_missing']} with "
                  f"{s['fill_value']:.4f} ({s['fill_source']})")
    print(f"{'feature':22s} {'mean':>9s} {'std':>9s} {'min':>9s} {'max':>9s}")
    for name, st in meta["feature_stats"].items():
        print(f"{name:22s} {st['mean']:9.4f} {st['std']:9.4f} "
              f"{st['min']:9.4f} {st['max']:9.4f}")
    print(f"[meta] {meta_path}")
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", default="data/processed/d2v_attr_audit_per_chart.parquet")
    ap.add_argument("--step1", default="data/processed/step1_1k_pairbest.parquet",
                    help="step1 pairbest table; only used to pick the imputation medians")
    ap.add_argument("--out", default="data/processed/chart_content_meta_summary.parquet")
    return ap.parse_args(argv)


def main() -> int:
    args = parse_args()
    return build(Path(args.audit), Path(args.out), Path(args.step1))


if __name__ == "__main__":
    raise SystemExit(main())
