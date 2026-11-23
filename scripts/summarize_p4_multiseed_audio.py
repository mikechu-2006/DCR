#!/usr/bin/env python
"""summarize_p4_multiseed_audio -- combined old/legacy/audio table for the audio run.

Reads three run_p2_p4_multiseed summaries:

    old      results/p4_multiseed              (6,828 charts, old splits)
    legacy   results/p4_multiseed_audio/legacy (6,819 charts, new splits, audio-off)
    audio    results/p4_multiseed_audio/audio  (6,819 charts, new splits, audio-on)

The main statistic is the paired audio - legacy delta (same split seed, same
ceiling).  old - legacy and old - audio are also reported, but they are only
descriptive because the chart set and the split seeds differ.

Usage:
    python scripts/summarize_p4_multiseed_audio.py \
        --out-prefix results/p4_multiseed_audio/combined_summary
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = [
    ("cm3p_linear_extra", "CM3P + external 线性"),
    ("cm3p_mlp_extra", "CM3P + external MLP"),
    ("cm3p_linear", "CM3P 线性 (λ1e-3)"),
    ("cm3p_mlp_h64_L2", "CM3P MLP h64 L2"),
    ("sunny_star", "Sunny"),
    ("daniel_star", "Daniel"),
    ("star", "star"),
]
PROTOCOLS = [("p2", "P2（新 chart）"), ("p4", "P4（新玩家 + 新 chart）")]


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    df = pd.read_csv(path)
    need = {"protocol", "seed", "method", "signal", "nll", "pearson",
            "spearman", "ceiling_nll"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    return df


def fmt(mu: float, sd: float) -> str:
    return f"{mu:.5f} ± {sd:.5f}" if np.isfinite(mu) else "--"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default="results/p4_multiseed/summary.csv")
    ap.add_argument("--legacy", default="results/p4_multiseed_audio/legacy/summary.csv")
    ap.add_argument("--audio", default="results/p4_multiseed_audio/audio/summary.csv")
    ap.add_argument("--out-prefix", default="results/p4_multiseed_audio/combined_summary")
    args = ap.parse_args(argv)

    old, leg, aud = load(Path(args.old)), load(Path(args.legacy)), load(Path(args.audio))
    rows: list[dict] = []
    for proto, _ in PROTOCOLS:
        for sig, label in METHODS:
            so = old[(old.protocol == proto) & (old.signal == sig)].set_index("seed").sort_index()
            sl = leg[(leg.protocol == proto) & (leg.signal == sig)].set_index("seed").sort_index()
            sa = aud[(aud.protocol == proto) & (aud.signal == sig)].set_index("seed").sort_index()
            common = so.index.intersection(sl.index).intersection(sa.index)
            if len(common) == 0:
                continue
            d = (sa.loc[common, "nll"] - sl.loc[common, "nll"]).to_numpy()
            rows.append({
                "protocol": proto, "signal": sig, "method": label,
                "old_nll_mean": so.loc[common, "nll"].mean(),
                "old_nll_std": so.loc[common, "nll"].std(ddof=1),
                "legacy_nll_mean": sl.loc[common, "nll"].mean(),
                "legacy_nll_std": sl.loc[common, "nll"].std(ddof=1),
                "audio_nll_mean": sa.loc[common, "nll"].mean(),
                "audio_nll_std": sa.loc[common, "nll"].std(ddof=1),
                "paired_delta_audio_legacy_mean": float(d.mean()),
                "paired_delta_audio_legacy_std": float(d.std(ddof=1)),
                "n_better": int((d < 0).sum()), "n_seeds": int(len(d)),
                "legacy_minus_old_mean": sl.loc[common, "nll"].mean() - so.loc[common, "nll"].mean(),
                "audio_minus_old_mean": sa.loc[common, "nll"].mean() - so.loc[common, "nll"].mean(),
                "old_pearson_mean": so.loc[common, "pearson"].mean(),
                "legacy_pearson_mean": sl.loc[common, "pearson"].mean(),
                "audio_pearson_mean": sa.loc[common, "pearson"].mean(),
                "old_spearman_mean": so.loc[common, "spearman"].mean(),
                "legacy_spearman_mean": sl.loc[common, "spearman"].mean(),
                "audio_spearman_mean": sa.loc[common, "spearman"].mean(),
            })

    df = pd.DataFrame(rows)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")
    df.to_csv(csv_path, index=False)

    md = ["# 完整 audio-on 表：old / legacy / audio 三臂汇总", "",
          "old = results/p4_multiseed（6,828 图、旧划分）；legacy / audio = 新 6,819 图、"
          "同一批新划分。主判据是 **paired audio - legacy**；legacy - old / audio - old "
          "因图集与划分不同，只作描述性参照。", ""]
    for proto, title in PROTOCOLS:
        sub = df[df.protocol == proto]
        if sub.empty:
            continue
        md += ["## " + title, ""]
        co = old[(old.protocol == proto) & (old.signal == "cm3p_linear")].set_index("seed")
        cl = leg[(leg.protocol == proto) & (leg.signal == "cm3p_linear")].set_index("seed")
        ca = aud[(aud.protocol == proto) & (aud.signal == "cm3p_linear")].set_index("seed")
        md += [f"- ceiling NLL: old {co.ceiling_nll.mean():.5f} ± "
               f"{co.ceiling_nll.std(ddof=1):.5f}；legacy {cl.ceiling_nll.mean():.5f} ± "
               f"{cl.ceiling_nll.std(ddof=1):.5f}；audio {ca.ceiling_nll.mean():.5f} ± "
               f"{ca.ceiling_nll.std(ddof=1):.5f}（legacy/audio 同划分，应逐 seed 相同）", "",
               "| 方法 | old p4_multiseed | legacy（新划分） | audio（新表） | audio - legacy（配对） | seeds 变好 |",
               "|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            md.append(f"| {r.method} | {fmt(r.old_nll_mean, r.old_nll_std)} "
                      f"| {fmt(r.legacy_nll_mean, r.legacy_nll_std)} "
                      f"| {fmt(r.audio_nll_mean, r.audio_nll_std)} "
                      f"| {r.paired_delta_audio_legacy_mean:+.5f} ± "
                      f"{r.paired_delta_audio_legacy_std:.5f} "
                      f"| {r.n_better}/{r.n_seeds} |")
        md.append("")
    md += ["## 数据变更检查", "",
           "legacy - old = 新 6,819 图/新划分相对旧 6,828 图的移动；audio - old 是同一参考。",
           "", "| protocol | 方法 | legacy - old | audio - old |", "|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r.protocol} | {r.method} | {r.legacy_minus_old_mean:+.5f} "
                  f"| {r.audio_minus_old_mean:+.5f} |")
    md.append("")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"-> {csv_path} / {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
