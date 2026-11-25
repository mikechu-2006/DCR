#!/usr/bin/env python
"""Fit isotonic raw-score -> Reform Dan mappings from the anchor manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = ["Sunny", "DCR", "CM3P_ext"]
SPECS = {
    "Sunny": {"method": "Sunny", "model_id": "SunnyRework", "raw_column": "Sunny"},
    "DCR": {"method": "DCR", "model_id": "DCR_A20D_P", "raw_column": "DCR"},
    "CM3P_ext": {"method": "CM3P_ext", "model_id": "RM_L", "raw_column": "CM3P_ext"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fit_one(raw: np.ndarray, target: np.ndarray) -> tuple[dict, float]:
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip",
                             y_min=float(target.min()), y_max=float(target.max()))
    fitted = iso.fit_transform(raw, target)
    knots = {"x": [float(x) for x in iso.X_thresholds_],
             "y": [float(y) for y in iso.y_thresholds_]}
    return knots, float(np.mean(np.abs(fitted - target)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-scores", type=Path, default=ROOT / "data/reform/raw_scores.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data/reform/calibration.json")
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                    help="comma-separated raw score columns; default fits all supported methods")
    args = ap.parse_args()

    raw = pd.read_csv(args.raw_scores)
    anchors_path = ROOT / "data/reform/anchors.jsonl"
    anchors = pd.DataFrame(json.loads(line) for line in anchors_path.read_text().splitlines()
                           if line.strip())
    if "anchor_id" not in raw or "target" not in raw:
        raise SystemExit("raw score CSV needs anchor_id and target columns")
    if raw["anchor_id"].duplicated().any() or anchors["anchor_id"].duplicated().any():
        raise SystemExit("duplicate anchor_id in input")
    expected = anchors.set_index("anchor_id")["target"].sort_index()
    got = raw.set_index("anchor_id")["target"].sort_index()
    if not expected.index.equals(got.index) or not np.allclose(expected, got, rtol=0, atol=1e-8):
        raise SystemExit("raw score rows/targets do not exactly match the 117-anchor manifest")

    raw_meta_path = args.raw_scores.with_suffix(args.raw_scores.suffix + ".meta.json")
    raw_meta = (json.loads(raw_meta_path.read_text(encoding="utf-8"))
                if raw_meta_path.is_file() else {})
    requested = [s.strip() for s in args.methods.split(",") if s.strip()]
    methods: dict[str, dict] = {}
    target = raw["target"].to_numpy(dtype=float)
    for name in requested:
        if name not in SPECS:
            raise SystemExit(f"unknown method/head {name}; choose from {sorted(SPECS)}")
        spec = SPECS[name]
        col = spec["raw_column"]
        if col not in raw:
            raise SystemExit(f"raw score file missing {col}")
        values = raw[col].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            n_missing = int((~np.isfinite(values)).sum())
            raise SystemExit(f"{col} has {n_missing} missing/non-finite anchor scores")
        knots, mae = fit_one(values, target)
        model_id = spec["model_id"]
        if name == "DCR":
            model_id = raw_meta.get("model_ids", {}).get(name) or model_id
        entry = {"model_id": model_id, "raw_column": col,
                 "n_anchors": int(len(raw)), "n_distinct_raw_scores": int(np.unique(values).size),
                 "raw_min": float(values.min()), "raw_max": float(values.max()),
                 "fit_mae_on_anchors": mae, "calibration": knots}
        if spec["method"] == "CM3P_ext":
            methods.setdefault("CM3P_ext", {"default_model_id": "RM_L", "models": {}})
            methods["CM3P_ext"]["models"][spec["model_id"]] = entry
        else:
            methods[spec["method"]] = entry

    targets = anchors["target"].astype(float)
    payload = {
        "schema_version": 1,
        "target": "Reform Dan course label",
        "fit": "isotonic regression (increasing), equal weight per anchor, endpoint clipping",
        "n_anchors": int(len(raw)),
        "difficulty_range": [float(targets.min()), float(targets.max())],
        "tier_fraction_bounds": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "tier_labels": ["low", "low/mid", "mid", "mid/high", "high"],
        "anchor_manifest_sha256": sha256(anchors_path),
        "raw_scores_sha256": sha256(args.raw_scores),
        "raw_score_inputs_sha256": raw_meta.get("inputs_sha256", {}),
        "available_raw_scores": {name: int(raw[name].notna().sum())
                                  for name in ("Sunny", "DCR", "CM3P_ext")
                                  if name in raw},
        "methods": methods,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"out": str(args.out), "n_anchors": len(raw),
                      "fitted": {k: ({"default_model_id": v["default_model_id"],
                                      "models": list(v["models"])} if "models" in v else v["model_id"])
                                 for k, v in methods.items()}},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
