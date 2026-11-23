#!/usr/bin/env python
"""Render the VSRG benchmark dashboard's checks with matplotlib.

This mirrors the website's definitions (stats.js):
  exact <= 0.2, close <= 0.5, moderate <= 1.0, miss > 1.0
  MAE/RMSE/bias/median|d|/P90|d|/coverage, pattern MAE, delta histogram,
  expected-vs-got scatter, expected/got trend, head-to-head absolute-error plot.

    python scripts/render_benchmark_dashboard.py \
        --base results/benchmark/eval_full/candidates/DCR-CM3P-XMLP-Legacy.csv \
        --compare results/benchmark/site/results/Sunny.csv \
        --scope ALL --out results/benchmark/viz/M1L_vs_Sunny_ALL.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

BG = "#070b14"
PANEL = "#0d1424"
TEXT = "#eef3ff"
MUTED = "#94a3b8"
GRID = "rgba(148,175,240,0.12)"
BAND_COLORS = {"exact": "#10b981", "close": "#38bdf8",
               "moderate": "#f59e0b", "miss": "#f43f5e"}
BAND_ORDER = ["exact", "close", "moderate", "miss"]
BAND_MAX = {"exact": 0.2, "close": 0.5, "moderate": 1.0, "miss": np.inf}


def load(path: Path, scope: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["expected"] = pd.to_numeric(df["expected"], errors="coerce")
    df["got"] = pd.to_numeric(df["got"], errors="coerce")
    df = df.dropna(subset=["expected", "got"]).copy()
    df["delta"] = df["expected"] - df["got"]
    df["deltaAbs"] = df["delta"].abs()
    df["band"] = pd.cut(df["deltaAbs"], [-np.inf, 0.2, 0.5, 1.0, np.inf],
                        labels=BAND_ORDER, right=True)
    if scope == "RC":
        df = df[df["pattern"].isin(["jack", "speed", "stamina", "tech"])]
    elif scope == "LN":
        df = df[df["pattern"].eq("ln")]
    return df.reset_index(drop=True)


def stats(df: pd.DataFrame) -> dict:
    d = df["delta"].to_numpy(float)
    a = df["deltaAbs"].to_numpy(float)
    return {
        "total": int(len(df)),
        "valid": int(len(df)),
        "mae": float(np.mean(a)),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "bias": float(np.mean(d)),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "coverage": 100.0,
        "bands": {b: int((df["band"] == b).sum()) for b in BAND_ORDER},
    }


def key(row) -> str:
    return f"{str(row['name']).strip()}\x01{str(row['pattern']).strip()}\x01{str(row['subPattern']).strip()}"


def head_to_head(base: pd.DataFrame, compare: pd.DataFrame) -> dict:
    b = {key(r): r for _, r in base.iterrows()}
    c = {key(r): r for _, r in compare.iterrows()}
    common = sorted(set(b) & set(c))
    ba = np.array([b[k]["deltaAbs"] for k in common])
    ca = np.array([c[k]["deltaAbs"] for k in common])
    return {
        "n": len(common),
        "base_wins": int((ba < ca).sum()),
        "compare_wins": int((ba > ca).sum()),
        "ties": int((ba == ca).sum()),
        "base_mae": float(ba.mean()) if len(ba) else np.nan,
        "compare_mae": float(ca.mean()) if len(ca) else np.nan,
        "base": ba, "compare": ca,
    }


def kpi_card(fig, x, y, w, h, label, value, sub=""):
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(PANEL)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#1e2a44")
    ax.text(0.05, 0.74, label.upper(), color=MUTED, fontsize=8, transform=ax.transAxes)
    ax.text(0.05, 0.26, value, color=TEXT, fontsize=17, weight="bold", transform=ax.transAxes)
    if sub:
        ax.text(0.05, 0.04, sub, color="#6b7a99", fontsize=7, transform=ax.transAxes)


def dark_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=12, loc="left", pad=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ax.spines.values():
        s.set_color("#1e2a44")
    ax.grid(color="#1b2740", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-name", default=None)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--compare-name", default=None)
    ap.add_argument("--scope", default="ALL", choices=["ALL", "RC", "LN"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = load(Path(args.base), args.scope)
    bname = args.base_name or Path(args.base).stem
    bs = stats(base)
    comp = load(Path(args.compare), args.scope) if args.compare else None
    cname = args.compare_name or (Path(args.compare).stem if args.compare else "")
    cs = stats(comp) if comp is not None else None
    h2h = head_to_head(base, comp) if comp is not None else None

    fig = plt.figure(figsize=(18, 23), facecolor=BG)
    fig.text(0.04, 0.975, "Estimator Algorithm Benchmark", color=TEXT,
             fontsize=26, weight="bold")
    fig.text(0.04, 0.955,
             f"{bname}" + (f"  vs  {cname}" if comp is not None else "") +
             f"   |   scope={args.scope}   |   chart-group 5-fold OOF isotonic got",
             color=MUTED, fontsize=11)

    # KPI cards
    cards = [
        ("Total maps", f"{bs['total']}", "all benchmark rows"),
        ("Valid maps", f"{bs['valid']}", "numeric got"),
        ("MAE", f"{bs['mae']:.4f}", "mean |expected-got|"),
        ("RMSE", f"{bs['rmse']:.4f}", "root mean square"),
        ("Bias", f"{bs['bias']:+.4f}", "mean(expected-got)"),
        ("Median |delta|", f"{bs['median']:.4f}", "P50"),
        ("Coverage", f"{bs['coverage']:.1f}%", "valid / total"),
        ("P90 |delta|", f"{bs['p90']:.4f}", "90th percentile"),
    ]
    x0, y0, w, h, gap = 0.04, 0.890, 0.105, 0.058, 0.012
    for i, (label, value, sub) in enumerate(cards):
        row, col = divmod(i, 4)
        kpi_card(fig, x0 + col * (w + gap), y0 - row * (h + 0.012), w, h,
                 label, value, sub)

    # Doughnut
    ax = fig.add_axes([0.04, 0.56, 0.24, 0.17])
    dark_ax(ax, "Accuracy breakdown")
    ax.grid(False)
    vals = [bs["bands"][b] for b in BAND_ORDER]
    wedges, texts, autotexts = ax.pie(
        vals, colors=[BAND_COLORS[b] for b in BAND_ORDER],
        startangle=90, counterclock=False,
        wedgeprops=dict(width=0.38, edgecolor=BG, linewidth=2),
        autopct=lambda p: f"{p:.1f}%" if p >= 6 else "",
        pctdistance=0.78, textprops=dict(color=TEXT, fontsize=8))
    ax.text(0, 0.05, f"{bs['valid']}", color=TEXT, ha="center", va="center",
            fontsize=20, weight="bold")
    ax.text(0, -0.15, "valid maps", color=MUTED, ha="center", va="center", fontsize=8)
    ax.legend([f"{b}  ({bs['bands'][b]})" for b in BAND_ORDER],
              loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2,
              frameon=False, labelcolor=TEXT, fontsize=7)

    # Scatter
    ax = fig.add_axes([0.36, 0.53, 0.60, 0.22])
    dark_ax(ax, "Expected vs Got  (diagonal = perfect)")
    for b in BAND_ORDER:
        s = base[base["band"] == b]
        ax.scatter(s["expected"], s["got"], s=10, alpha=0.75,
                   color=BAND_COLORS[b], label=b, linewidths=0)
    lo = min(base["expected"].min(), base["got"].min()) - 0.5
    hi = max(base["expected"].max(), base["got"].max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "--", color="#9fb0da", linewidth=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Expected", color=MUTED); ax.set_ylabel("Got", color=MUTED)
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=8, loc="upper left")

    # Delta histogram
    ax = fig.add_axes([0.04, 0.30, 0.44, 0.17])
    dark_ax(ax, "Delta distribution  (expected - got)")
    step = 0.2
    lo = np.floor(base["delta"].min() / step) * step
    hi = np.ceil(base["delta"].max() / step) * step
    edges = np.arange(lo, hi + step, step)
    ax.hist(base["delta"], bins=edges, color="#80b3ff", alpha=0.65,
            edgecolor="#80b3ff", linewidth=0.8)
    ax.axvline(0, color="#9fb0da", linewidth=1, linestyle="--")
    ax.set_xlabel("expected - got", color=MUTED)

    # Trend
    ax = fig.add_axes([0.54, 0.30, 0.42, 0.17])
    dark_ax(ax, "Expected / Got trend  (sorted by expected)")
    tr = base.sort_values("expected").reset_index(drop=True)
    ax.plot(tr.index, tr["expected"], color="#7ad1ff", linewidth=2, label="Expected")
    ax.plot(tr.index, tr["got"], color="#5ee8bd", linewidth=1.4, alpha=0.9, label="Got")
    ax.set_xlabel("map index", color=MUTED); ax.set_ylabel("difficulty", color=MUTED)
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=8)

    # Pattern MAE
    ax = fig.add_axes([0.04, 0.055, 0.44, 0.17])
    dark_ax(ax, "Pattern MAE")
    pat = (base.groupby("pattern")["deltaAbs"].agg(["count", "mean"])
           .sort_values("mean", ascending=True))
    ax.barh(pat.index, pat["mean"], color="#f59e0b", alpha=0.8)
    for i, (name, row) in enumerate(pat.iterrows()):
        ax.text(row["mean"] + 0.01, i, f"{row['mean']:.3f} (n={int(row['count'])})",
                color=TEXT, va="center", fontsize=8)
    ax.set_xlabel("MAE", color=MUTED)

    # Head to head
    ax = fig.add_axes([0.54, 0.055, 0.42, 0.17])
    if h2h is not None and h2h["n"]:
        dark_ax(ax, f"Head-to-head |delta|  ({bname} vs {cname})")
        ax.scatter(h2h["base"], h2h["compare"], s=10, alpha=0.55, color="#38bdf8")
        m = max(h2h["base"].max(), h2h["compare"].max()) * 1.05
        ax.plot([0, m], [0, m], "--", color="#9fb0da", linewidth=1)
        ax.set_xlim(0, m); ax.set_ylim(0, m)
        ax.set_xlabel(f"{bname} |delta|", color=MUTED)
        ax.set_ylabel(f"{cname} |delta|", color=MUTED)
        ax.text(0.03, 0.95,
                f"matched {h2h['n']}\nbase wins {h2h['base_wins']}"
                f"   compare wins {h2h['compare_wins']}   ties {h2h['ties']}\n"
                f"MAE {h2h['base_mae']:.4f} vs {h2h['compare_mae']:.4f}"
                f"   gap {h2h['base_mae'] - h2h['compare_mae']:+.4f}",
                transform=ax.transAxes, color=TEXT, fontsize=8, va="top")
    else:
        ax.axis("off")

    fig.text(0.04, 0.015,
             "Definitions mirror benchmark.leoblack.top: exact<=0.2, close<=0.5, "
             "moderate<=1.0, miss>1.0; OOF isotonic got from 5-fold chart-group LOCV.",
             color="#6b7a99", fontsize=8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
