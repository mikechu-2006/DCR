"""Runtime helpers for Reform Dan anchor-based isotonic calibration."""
from __future__ import annotations

import json
import math
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Any


DEFAULT_CALIBRATION = Path(__file__).resolve().parents[1] / "data/reform/calibration.json"
TIER_LABELS = ("low", "low/mid", "mid", "mid/high", "high")


def load_calibration(path: str | Path = DEFAULT_CALIBRATION) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def isotonic_interpolate(raw_score: float, knots: dict[str, Any]) -> float:
    """Linearly interpolate monotone isotonic knots, clipping to endpoint values."""
    x = [float(v) for v in knots["x"]]
    y = [float(v) for v in knots["y"]]
    if not x or len(x) != len(y):
        raise ValueError("calibration needs equally sized, non-empty x/y knot arrays")
    value = float(raw_score)
    if not math.isfinite(value):
        raise ValueError("raw score must be finite")
    if len(x) == 1 or value <= x[0]:
        return y[0]
    if value >= x[-1]:
        return y[-1]
    lo, hi = 0, len(x) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x[mid] <= value:
            lo = mid
        else:
            hi = mid
    if x[hi] == x[lo]:
        return y[hi]
    return y[lo] + (y[hi] - y[lo]) * (value - x[lo]) / (x[hi] - x[lo])


def anchor_difficulty(raw_score: float, method: str, *, model_id: str | None = None,
                      calibration: dict[str, Any] | None = None,
                      calibration_path: str | Path = DEFAULT_CALIBRATION) -> float:
    """Convert one method's raw score to continuous Reform Dan difficulty.

    ``CM3P_ext`` uses the RM-L head. DCR and Sunny each have one curve.
    """
    config = calibration if calibration is not None else load_calibration(calibration_path)
    methods = config["methods"]
    key = str(method)
    if key not in methods:
        raise KeyError(f"unknown calibration method: {key}")
    entry = methods[key]
    if "models" in entry:
        chosen = model_id or entry.get("default_model_id")
        if chosen not in entry["models"]:
            raise KeyError(f"unknown {key} model_id: {chosen}")
        entry = entry["models"][chosen]
    elif model_id is not None and entry.get("model_id") not in (None, model_id):
        raise KeyError(f"{key} calibration has no model_id {model_id}")
    return float(isotonic_interpolate(raw_score, entry["calibration"]))


def five_tier(difficulty: float) -> str:
    """Return the five-tier label for the fractional part of a Dan difficulty."""
    value = float(difficulty)
    if not math.isfinite(value):
        raise ValueError("difficulty must be finite")
    decimal_value = Decimal(str(value))
    whole = decimal_value.to_integral_value(rounding=ROUND_FLOOR)
    fraction = decimal_value - whole
    index = min(4, int((fraction * 5).to_integral_value(rounding=ROUND_FLOOR)))
    return TIER_LABELS[index]


def anchored_result(raw_score: float, method: str, *, model_id: str | None = None,
                    calibration: dict[str, Any] | None = None,
                    calibration_path: str | Path = DEFAULT_CALIBRATION) -> dict[str, Any]:
    """Return both continuous difficulty and its five-tier display label."""
    value = anchor_difficulty(raw_score, method, model_id=model_id,
                              calibration=calibration, calibration_path=calibration_path)
    return {"difficulty": value, "tier": five_tier(value)}
