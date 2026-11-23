"""Two-function DCR -> Dan predictor for product packaging.

Public API (the two functions the packaging side needs):

    cm3p_to_dan(vector) -> float
    d2v_to_dan(vector)  -> float

Input vector contract
--------------------
Both functions take one flat sequence of 514 floats:

    [ C[0], ..., C[511], msd_overall, sunny_star ]

- C is the frozen 512-d content vector:
  * CM3P: OliBomby/CM3P@26cc694, audio off, crop legacy, 512-d.
  * D2V: data2vec audio-style holdout encoder, 512-d, L2-normalised.
- msd_overall: MinaCalc standalone calc_version 505, goal 0.93, rate 1.0.
- sunny_star: osumania_map_analyser commit 448ed7c.

The function standardizes C with the model's mu/sd, standardizes the two
external signals with extra_mu/extra_sd, computes the Bradley-Terry latent
d = w'[C, S], and applies the fixed isotonic calibration fitted on all 1,382
VSRG benchmark rows.  The returned float is the calibrated Dan estimate.

Output scale depends on the input chart population.  On the 1,382 benchmark rows the
median is about 14.0; on all 21,949 charts the median is about 2.76 because most charts
are easy.  The full-table histogram is NOT a star scale and does not mean the function
is broken; validate MAE only on the benchmark rows.

No third-party dependency is needed: only json, math, bisect.  This is the
reference for the TypeScript/JS port in the packaging repo.
"""
from __future__ import annotations

import json
import math
from bisect import bisect_left
from functools import lru_cache
from pathlib import Path
from typing import Sequence

CONTENT_DIM = 512
EXTRA_NAMES = ("msd_overall", "sunny_star")
VECTOR_DIM = CONTENT_DIM + len(EXTRA_NAMES)

_DIR = Path(__file__).resolve().parent
_MODEL_FILES = {
    "cm3p": _DIR / "models" / "cm3p_linear_extra.json",
    "d2v": _DIR / "models" / "d2v_linear_extra.json",
}


@lru_cache(maxsize=2)
def _load(kind: str) -> dict:
    path = _MODEL_FILES[kind]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run scripts/export_dan_deploy.py for {kind}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if int(cfg["content_dim"]) != CONTENT_DIM:
        raise ValueError(f"{path}: content_dim={cfg['content_dim']} != {CONTENT_DIM}")
    if tuple(cfg["extra_names"]) != EXTRA_NAMES:
        raise ValueError(f"{path}: extra_names={cfg['extra_names']} != {list(EXTRA_NAMES)}")
    if len(cfg["w"]) != VECTOR_DIM:
        raise ValueError(f"{path}: len(w)={len(cfg['w'])} != {VECTOR_DIM}")
    return cfg


def _isotonic(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    """Same piecewise-linear interpolation as sklearn IsotonicRegression.predict."""
    if not xs:
        raise ValueError("empty isotonic calibration")
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    i = bisect_left(xs, x)
    x0, x1 = float(xs[i - 1]), float(xs[i])
    y0, y1 = float(ys[i - 1]), float(ys[i])
    if x1 == x0:
        return y1
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _raw_d(kind: str, vector: Sequence[float]) -> float:
    """Bradley-Terry latent d before calibration (for parity tests)."""
    if len(vector) != VECTOR_DIM:
        raise ValueError(f"{kind}: vector must have {VECTOR_DIM} floats, got {len(vector)}")
    vals = [float(v) for v in vector]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{kind}: vector contains NaN/inf")

    cfg = _load(kind)
    mu, sd = cfg["mu"], cfg["sd"]
    extra_mu, extra_sd = cfg["extra_mu"], cfg["extra_sd"]
    w = cfg["w"]

    d = 0.0
    for i in range(CONTENT_DIM):
        d += w[i] * ((vals[i] - mu[i]) / sd[i])
    for j in range(len(EXTRA_NAMES)):
        d += w[CONTENT_DIM + j] * (
            (vals[CONTENT_DIM + j] - extra_mu[j]) / extra_sd[j])
    return (d - cfg["d_mu_raw"]) / cfg["d_sd_raw"]


def _predict(kind: str, vector: Sequence[float]) -> float:
    d = _raw_d(kind, vector)
    cal = _load(kind)["calibration"]
    return _isotonic(d, cal["x"], cal["y"])


def cm3p_to_dan(vector: Sequence[float]) -> float:
    """CM3P-512 + [msd_overall, sunny_star] -> calibrated Dan estimate."""
    return _predict("cm3p", vector)


def d2v_to_dan(vector: Sequence[float]) -> float:
    """D2V-512 + [msd_overall, sunny_star] -> calibrated Dan estimate."""
    return _predict("d2v", vector)


__all__ = ["cm3p_to_dan", "d2v_to_dan", "CONTENT_DIM", "EXTRA_NAMES", "VECTOR_DIM"]
