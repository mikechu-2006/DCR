#!/usr/bin/env python
"""Fit raw-score to Reform Dan mappings from the anchor manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import LinearConstraint, minimize
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = ["Sunny", "DCR", "CM3P_ext"]
SPLINE_DEGREE = 3
LOWER_INTERVALS = 4
PRIORITY_INTERVALS = 6
TAIL_INTERVALS = 5
SPLINE_MIN_SLOPE = 0.05
SPLINE_EXPORT_POINTS = 513
DEFAULT_SMOOTHNESS = 0.00005
DELTA_JOIN_ALLOWANCE = 0.5
TAIL_ANCHOR_WEIGHT = 0.5
TAIL_SMOOTHNESS_FACTOR = 0.25
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


def fit_one(raw: np.ndarray, target: np.ndarray,
            weights: np.ndarray) -> tuple[dict, float]:
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip",
                             y_min=float(target.min()), y_max=float(target.max()))
    fitted = iso.fit_transform(raw, target, sample_weight=weights)
    knots = {"x": [float(x) for x in iso.X_thresholds_],
             "y": [float(y) for y in iso.y_thresholds_]}
    return knots, float(np.mean(np.abs(fitted - target)))


def fit_spline_segment(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray,
                       breaks: np.ndarray, smoothness: float, *,
                       relaxed_after: float = 1.0,
                       end_range: tuple[float, float] = (1.0, 1.0),
                       start_slope: float | None = None) -> BSpline:
    """Fit a monotone spline in normalized coordinates starting from zero."""
    knots = np.pad(breaks, SPLINE_DEGREE, mode="edge")
    count = len(knots) - SPLINE_DEGREE - 1
    basis = BSpline(knots, np.eye(count), SPLINE_DEGREE)

    # Increasing coefficients keep the spline monotone. At the Greville sites
    # these coefficients reproduce the identity, giving a floor on its slope.
    greville = np.array([knots[i + 1:i + 1 + SPLINE_DEGREE].mean() for i in range(count)])
    linear_steps = np.diff(greville)
    min_steps = SPLINE_MIN_SLOPE * linear_steps
    max_steps = np.full_like(min_steps, end_range[1])
    free_steps = np.ones_like(min_steps, dtype=bool)
    if start_slope is not None:
        first_step = start_slope * linear_steps[0]
        min_steps[0] = max_steps[0] = first_step
        free_steps[0] = False
    remaining = end_range[0] - min_steps.sum()
    if remaining < 0:
        raise ValueError("slope at Delta cannot be continued to the upper endpoint")
    initial = min_steps.copy()
    initial[free_steps] += remaining * linear_steps[free_steps] / linear_steps[free_steps].sum()
    cumulative = np.tril(np.ones((count - 1, count - 1)))
    export_grid = np.linspace(0.0, 1.0, SPLINE_EXPORT_POINTS)
    observations = basis(scores)[:, 1:] @ cumulative
    curvature = basis(export_grid, nu=2)[:, 1:] @ cumulative
    row_scale = np.sqrt(weights / weights.sum())
    penalty_scale = np.sqrt((smoothness / len(export_grid)) *
                            np.where(export_grid > relaxed_after, TAIL_SMOOTHNESS_FACTOR, 1.0))
    design = np.vstack((observations * row_scale[:, None],
                        curvature * penalty_scale[:, None]))
    truth = np.r_[labels * row_scale, np.zeros(len(export_grid))]

    def objective(steps):
        residual = design @ steps - truth
        return float(residual @ residual), 2.0 * (design.T @ residual)

    result = minimize(objective, initial, method="SLSQP", jac=True,
                      bounds=list(zip(min_steps, max_steps)),
                      constraints=LinearConstraint(np.ones((1, len(initial))), *end_range),
                      options={"ftol": 1e-12, "maxiter": 200})
    if not result.success:
        raise ValueError(f"monotone spline fit failed: {result.message}")
    coefficients = np.r_[0.0, np.cumsum(result.x)]
    coefficients[-1] = np.clip(coefficients[-1], *end_range)
    return BSpline(knots, coefficients, SPLINE_DEGREE)


def fit_monotone_spline(raw: np.ndarray, target: np.ndarray, weights: np.ndarray,
                        smoothness: float, *, priority: float, delta: float,
                        epsilon: float) -> tuple[dict, float, dict]:
    """Fit through Delta first, then extend the same slope into the upper tail."""
    lower = target <= delta
    priority_rows = lower & (target >= priority)
    raw_min, raw_max = float(raw[lower].min()), float(raw.max())
    target_min, target_max = float(target.min()), float(target.max())
    join = float(raw[lower].max())
    lower_span, tail_span = join - raw_min, raw_max - join
    lower_height = delta - target_min
    if min(lower_span, tail_span, lower_height, target_max - delta) <= 0:
        raise ValueError("calibration needs distinct lower and upper score ranges")

    priority_start = (float(raw[priority_rows].min()) - raw_min) / lower_span
    lower_breaks = np.linspace(0.0, 1.0, LOWER_INTERVALS + PRIORITY_INTERVALS + 1)
    if 0.0 < priority_start < 1.0:
        lower_breaks = np.r_[np.linspace(0.0, priority_start, LOWER_INTERVALS + 1),
                             np.linspace(priority_start, 1.0, PRIORITY_INTERVALS + 1)[1:]]
    lower_fit = fit_spline_segment((raw[lower] - raw_min) / lower_span,
                                   (target[lower] - target_min) / lower_height,
                                   weights[lower], lower_breaks, smoothness,
                                   end_range=(1.0, 1.0 + DELTA_JOIN_ALLOWANCE / lower_height))
    join_target = target_min + lower_height * float(lower_fit(1.0))
    tail_height = target_max - join_target
    if tail_height <= 0:
        raise ValueError("the joined curve needs room above Delta")
    join_slope = float(lower_fit.derivative()(1.0)) * lower_height / lower_span

    # A high-labelled chart below the join keeps the lower prediction. It
    # cannot pull the lower fit or the continuation out of order.
    tail = ~lower & (raw > join)
    epsilon_raw = float(np.median(raw[target == epsilon]))
    epsilon_start = np.clip((epsilon_raw - join) / tail_span, 0.0, 1.0)
    tail_fit = fit_spline_segment((raw[tail] - join) / tail_span,
                                  (target[tail] - join_target) / tail_height,
                                  weights[tail], np.linspace(0.0, 1.0, TAIL_INTERVALS + 1),
                                  smoothness, relaxed_after=epsilon_start,
                                  start_slope=join_slope * tail_span / tail_height)

    grid = np.linspace(0.0, 1.0, SPLINE_EXPORT_POINTS)
    x = np.r_[raw_min + lower_span * grid[:-1], join + tail_span * grid]
    y = np.r_[target_min + lower_height * lower_fit(grid[:-1]), join_target + tail_height * tail_fit(grid)]
    x[-1], y[-1] = raw_max, target_max
    if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
        raise ValueError("exported spline calibration is not strictly increasing")
    mae = float(np.mean(np.abs(np.interp(raw, x, y) - target)))
    return {"x": x.tolist(), "y": y.tolist()}, mae, {
        "delta_join_raw": join, "delta_join_value": join_target,
        "epsilon_penalty_boundary_raw": epsilon_raw,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-scores", type=Path, default=ROOT / "data/reform/raw_scores.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data/reform/calibration.json")
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                    help="comma-separated raw score columns; default fits all supported methods")
    ap.add_argument("--fit", choices=("monotone-spline", "isotonic"), default="monotone-spline")
    ap.add_argument("--smoothness", type=float, default=DEFAULT_SMOOTHNESS,
                    help="curvature penalty for the monotone spline on normalized scores and labels")
    args = ap.parse_args()
    if not np.isfinite(args.smoothness) or args.smoothness <= 0:
        ap.error("--smoothness must be positive and finite")

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
    course_targets = anchors.groupby("difficulty_key")["target"].first()
    priority = float(course_targets["regular_8"])
    delta = float(course_targets["greek_delta"])
    epsilon = float(course_targets["greek_epsilon"])
    weights = np.where(target > epsilon, TAIL_ANCHOR_WEIGHT, 1.0)
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
        if args.fit == "monotone-spline":
            knots, mae, fit_bounds = fit_monotone_spline(
                values, target, weights, args.smoothness, priority=priority,
                delta=delta, epsilon=epsilon)
        else:
            knots, mae = fit_one(values, target, weights)
            fit_bounds = {}
        model_id = spec["model_id"]
        if name == "DCR":
            model_id = raw_meta.get("model_ids", {}).get(name) or model_id
        entry = {"model_id": model_id, "raw_column": col,
                 "n_anchors": int(len(raw)), "n_distinct_raw_scores": int(np.unique(values).size),
                 "raw_min": float(values.min()), "raw_max": float(values.max()),
                 "fit_mae_on_anchors": mae, "calibration": knots, **fit_bounds}
        if spec["method"] == "CM3P_ext":
            methods.setdefault("CM3P_ext", {"default_model_id": "RM_L", "models": {}})
            methods["CM3P_ext"]["models"][spec["model_id"]] = entry
        else:
            methods[spec["method"]] = entry

    targets = anchors["target"].astype(float)
    payload = {
        "schema_version": 1,
        "target": "Reform Dan course label",
        "fit": ("segmented penalized monotone cubic spline, endpoint clipping"
                if args.fit == "monotone-spline" else
                "isotonic regression (increasing), endpoint clipping"),
        "anchor_weights": {"through_epsilon": 1.0, "above_epsilon": TAIL_ANCHOR_WEIGHT},
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
    if args.fit == "monotone-spline":
        payload["fit_parameters"] = {
            "degree": SPLINE_DEGREE, "lower_intervals": LOWER_INTERVALS,
            "priority_intervals": PRIORITY_INTERVALS, "tail_intervals": TAIL_INTERVALS,
            "priority_from_target": priority, "delta_target": delta, "epsilon_target": epsilon,
            "delta_join_allowance": DELTA_JOIN_ALLOWANCE,
            "normalized_minimum_slope": SPLINE_MIN_SLOPE,
            "normalized_curvature_penalty": args.smoothness,
            "above_epsilon_curvature_factor": TAIL_SMOOTHNESS_FACTOR,
            "export_points_per_segment": SPLINE_EXPORT_POINTS,
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
