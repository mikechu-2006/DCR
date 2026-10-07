"""Player-chart convergence filter.

判定一个 (玩家, 曲目) cell 是否"已经收敛"：把该 cell 的历次游玩按时间排序，
比较**末窗**与**前窗**的水平，以及末窗自身的离散度。

    delta   = mean(末窗 loss) - mean(前窗 loss)      # 负 = 后来越打越好（loss 下降 = ACC 上升）
    sd_tail = 末窗内 loss 的标准差                    # 末窗内是否稳定

收敛 = 末窗相对前窗没有持续漂移 (|delta| 小) 且末窗够紧 (sd_tail 小)，
另加最低样本量 / 时间跨度 / 练习深度门槛。

注意 loss = log(1-ACC) 是 ACC 的**递减**函数：loss 越小 = ACC 越高。
实测后窗普遍比前窗**低**（即后窗 ACC 更高 = 打得更好了，是一条正常的学习曲线，
见 docs/play_order_analysis.md）。所以判据对漂移取绝对值：不是要求"不再进步"，
而是要求"末段水平已经稳住、不再单向漂移"。

产出（--write）:
    data/processed/charts_v2_{tag}_conv.parquet  原 cells 表 + 轨迹特征 + conv_tier/converged
    data/processed/charts_v2_{tag}_conv.json     分层报告

用例:
    python scripts/convergence_filter.py --tag 1k --calibrate     # 打印分位数/存活率网格
    python scripts/convergence_filter.py --tag 1k --write         # 写表（全部 cell 行）
    python scripts/convergence_filter.py --tag 1k --write --primary-only-out   # 只写 primary 行
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

OUT = Path("data/processed")
RATE_NAME = {0: "NM", 1: "DT", 2: "HT"}
KEY_MUL = 100_000_000          # user_id * KEY_MUL + chart_id

TIER_NAMES = ["A", "B", "C", "none"]

# 两套口径：
#   abs = 绝对离散度阈值（末窗 sd <= max_sd_tail）+ 绝对水平差 <= max_adelta
#   se  = 尺度无关：末窗中位数的标准误 se_tail 足够小，且两窗之差不超过噪声 (|t| 检验)
#        se 口径对"难图/长图天然方差大"更公平，覆盖的曲目数明显更多。
DEFAULT_TIERS = {
    "abs": [
        {"name": "A", "min_n": 5, "min_span_days": 90, "min_attempts": 8,
         "max_adelta": 0.35, "max_sd_tail": 0.5},
        {"name": "B", "min_n": 4, "min_span_days": 30, "min_attempts": 5,
         "max_adelta": 0.6, "max_sd_tail": 0.8},
        {"name": "C", "min_n": 4, "min_span_days": 0, "min_attempts": 0,
         "max_adelta": 1.0, "max_sd_tail": 1.2},
    ],
    "se": [
        {"name": "A", "min_n": 5, "min_span_days": 90, "min_attempts": 8,
         "max_se_tail": 0.20, "max_t_delta": 2.0, "max_adelta": 0.6},
        {"name": "B", "min_n": 4, "min_span_days": 30, "min_attempts": 5,
         "max_se_tail": 0.30, "max_t_delta": 2.5, "max_adelta": 0.8},
        {"name": "C", "min_n": 4, "min_span_days": 0, "min_attempts": 0,
         "max_se_tail": 0.45, "max_t_delta": 3.0, "max_adelta": 1.2},
    ],
    # joint = abs 的「末窗够紧」∧ se 的「两窗之差不超过噪声」，
    # 堵住"末窗内部很紧、但整体相对前窗平移了"的漏洞（abs 单用会漏掉这种）。
    "joint": [
        {"name": "A", "min_n": 5, "min_span_days": 90, "min_attempts": 8,
         "max_sd_tail": 0.5, "max_adelta": 0.35, "max_t_delta": 2.5},
        {"name": "B", "min_n": 4, "min_span_days": 30, "min_attempts": 5,
         "max_sd_tail": 0.8, "max_adelta": 0.6, "max_t_delta": 3.0},
        {"name": "C", "min_n": 4, "min_span_days": 0, "min_attempts": 0,
         "max_sd_tail": 1.2, "max_adelta": 1.0, "max_t_delta": 4.0},
    ],
}


def cell_features(plays: pd.DataFrame, tail: int = 3) -> pd.DataFrame:
    """plays: user_id, chart_id, loss, year, doy -> 每个 cell 的时间轨迹特征。"""
    plays = plays.copy()
    plays["t"] = plays.year.to_numpy(np.float64) + (plays.doy.to_numpy(np.float64) - 1.0) / 365.0
    plays = plays.sort_values(["user_id", "chart_id", "t"], kind="stable")

    key = plays.user_id.to_numpy(np.int64) * KEY_MUL + plays.chart_id.to_numpy(np.int64)
    codes, uniq = pd.factorize(key, sort=False)
    L = plays.loss.to_numpy(np.float64)
    T = plays.t.to_numpy(np.float64)
    n = np.bincount(codes).astype(np.int64)
    pos = np.arange(len(codes), dtype=np.int64) - np.repeat(np.cumsum(n) - n, n)

    # 末窗 = 最后 tail 条；前窗 = 末窗之前同样多的成绩
    tail_n = np.minimum(tail, np.maximum(n - 1, 1))
    tn_row = np.repeat(tail_n, n)
    pn_row = np.repeat(n, n) - tn_row
    is_tail = pos >= pn_row
    is_prev = (pos < pn_row) & (pos >= np.maximum(pn_row - tn_row, 0))

    g = len(n)
    out = {"u": uniq, "n": n}
    for name, mask in (("tl", is_tail), ("pv", is_prev)):
        c = codes[mask]
        cnt = np.bincount(c, minlength=g)
        s = np.bincount(c, weights=L[mask], minlength=g)
        ss = np.bincount(c, weights=L[mask] ** 2, minlength=g)
        mean = s / np.maximum(cnt, 1)
        out[f"{name}_n"] = cnt
        out[f"{name}_mean"] = mean
        out[f"{name}_sd"] = np.sqrt(np.maximum(ss / np.maximum(cnt, 1) - mean ** 2, 0.0))

    out["tl_med"] = pd.Series(L[is_tail]).groupby(codes[is_tail]).median().to_numpy()
    out["tl_min"] = pd.Series(L[is_tail]).groupby(codes[is_tail]).min().to_numpy()
    out["tl_max"] = pd.Series(L[is_tail]).groupby(codes[is_tail]).max().to_numpy()
    ser = pd.Series(L, index=codes)
    out["l_max"] = ser.groupby(level=0).max().to_numpy()
    out["l_min"] = ser.groupby(level=0).min().to_numpy()
    out["l_std"] = ser.groupby(level=0).std().to_numpy()
    out["l_first"] = pd.Series(L).groupby(codes).first().to_numpy()
    out["l_last"] = pd.Series(L).groupby(codes).last().to_numpy()
    out["t_first"] = pd.Series(T).groupby(codes).min().to_numpy()
    out["t_last"] = pd.Series(T).groupby(codes).max().to_numpy()

    f = pd.DataFrame(out)
    f["chart_id"] = (f.u % KEY_MUL).astype(np.int64)
    f["user_id"] = (f.u // KEY_MUL).astype(np.int64)
    f["span"] = f.t_last - f.t_first
    f["delta"] = f.tl_mean - f.pv_mean
    f["sd_tail"] = f.tl_sd
    f["adelta"] = f.delta.abs()
    # 尺度无关口径：末窗中位数的标准误，以及"两窗之差 / 噪声"的近似 t 值
    f["se_tail"] = f.tl_sd / np.sqrt(np.maximum(f.tl_n, 1))
    f["se_delta"] = np.sqrt((f.tl_sd ** 2) / np.maximum(f.tl_n, 1)
                            + (f.pv_sd ** 2) / np.maximum(f.pv_n, 1))
    f["t_delta"] = f.adelta / np.maximum(f.se_delta, 1e-6)
    for c in f.columns:
        if c not in ("user_id", "chart_id", "n", "tl_n", "pv_n"):
            f[c] = f[c].astype(np.float32)
    return f.drop(columns=["u"])


FEATURE_COLS = ["n", "tl_n", "pv_n", "tl_mean", "pv_mean", "tl_sd", "pv_sd", "tl_med", "tl_min",
                "tl_max", "l_max", "l_min", "l_std", "l_first", "l_last", "t_first", "t_last",
                "span", "delta", "sd_tail", "adelta", "se_tail", "se_delta", "t_delta"]


def tier_codes(cols: dict, tiers: list[dict]) -> np.ndarray:
    """按 tiers 规格返回 0..len(tiers)-1 的层级编号，-1 = none。

    cols 需含 n / span(年) / attempts / adelta / sd_tail / se_tail / t_delta，
    规格里出现哪些阈值就检查哪些。
    """
    n = cols["n"]
    tier = np.full(len(n), -1, dtype=np.int8)
    ok = []
    for c in tiers:
        m = n >= c.get("min_n", 0)
        if "min_span_days" in c:
            m &= cols["span"] >= c["min_span_days"] / 365.0
        if "min_attempts" in c:
            m &= cols["attempts"] >= c["min_attempts"]
        if "max_adelta" in c:
            m &= cols["adelta"] <= c["max_adelta"]
        if "max_sd_tail" in c:
            m &= cols["sd_tail"] <= c["max_sd_tail"]
        if "max_se_tail" in c:
            m &= cols["se_tail"] <= c["max_se_tail"]
        if "max_t_delta" in c:
            m &= cols["t_delta"] <= c["max_t_delta"]
        ok.append(m)
    for i in range(len(tiers) - 1, -1, -1):        # 从最松到最严覆盖
        tier[ok[i]] = i
    return tier


def family_of(df: pd.DataFrame, attempts: np.ndarray, family: str, tiers: list[dict]) -> np.ndarray:
    cols = {"n": df["n"].to_numpy(), "span": df["span"].to_numpy(), "attempts": attempts,
            "adelta": df["adelta"].to_numpy(), "sd_tail": df["sd_tail"].to_numpy(),
            "se_tail": df["se_tail"].to_numpy(), "t_delta": df["t_delta"].to_numpy()}
    return tier_codes(cols, tiers)


class Accum:
    """流式累积分层统计（cells 表太大，不能一次性全读进内存）。"""

    @staticmethod
    def _new_stats() -> dict:
        return {"cells": 0, "users": set(), "charts": set(), "n": [], "attempts": [],
                "rate": {}, "star": [], "loss_med": [], "loss_tail": []}

    def __init__(self, tiers: list[dict]):
        self.tiers = tiers
        self.stats = {t["name"]: self._new_stats() for t in tiers}
        self.stats["none"] = self._new_stats()
        self.q = {"n": [], "span": [], "adelta": [], "sd_tail": [], "se_tail": [], "t_delta": [],
                  "gap_best_tail": []}

    def add(self, df: pd.DataFrame, tier: np.ndarray) -> None:
        prim = df.primary.to_numpy() if "primary" in df.columns else np.ones(len(df), bool)
        for i, name in enumerate([t["name"] for t in self.tiers]):
            m = prim & (tier == i)
            if m.any():
                self._add_one(self.stats[name], df, m)
        m = prim & (tier < 0)
        if m.any():
            self._add_one(self.stats["none"], df, m)
        m = prim & df["n"].notna().to_numpy()
        if m.any():
            self.q["n"].append(df["n"].to_numpy()[m])
            self.q["span"].append(df["span"].to_numpy()[m])
            self.q["adelta"].append(df["adelta"].to_numpy()[m])
            self.q["sd_tail"].append(df["sd_tail"].to_numpy()[m])
            self.q["se_tail"].append(df["se_tail"].to_numpy()[m])
            self.q["t_delta"].append(df["t_delta"].to_numpy()[m])
            self.q["gap_best_tail"].append((df["l_max"] - df["tl_mean"]).to_numpy()[m])

    @staticmethod
    def _add_one(s: dict, df: pd.DataFrame, m: np.ndarray) -> None:
        s["cells"] += int(m.sum())
        s["users"].update(df.user_id.to_numpy()[m].tolist())
        s["charts"].update(df.chart_id.to_numpy()[m].tolist())
        s["n"].append(df["n"].to_numpy()[m].astype(np.float32))
        if "attempts" in df.columns:
            s["attempts"].append(np.nan_to_num(df["attempts"].to_numpy()[m].astype(np.float32), nan=0.0))
        if "rate" in df.columns:
            v, c = np.unique(df.rate.to_numpy()[m], return_counts=True)
            for vv, cc in zip(v.tolist(), c.tolist()):
                s["rate"][int(vv)] = s["rate"].get(int(vv), 0) + cc
        if "star" in df.columns:
            s["star"].append(np.nan_to_num(df.star.to_numpy()[m].astype(np.float32), nan=np.nan))
        if "loss_med" in df.columns:
            s["loss_med"].append(np.nan_to_num(df.loss_med.to_numpy()[m].astype(np.float32), nan=np.nan))
        if "tl_med" in df.columns:
            s["loss_tail"].append(np.nan_to_num(df.tl_med.to_numpy()[m].astype(np.float32), nan=np.nan))

    @staticmethod
    def _cat(lst):
        return np.concatenate(lst) if lst else np.array([])

    def report(self, n_primary: int, cfg: dict) -> dict:
        rep = {"primary_cells": int(n_primary), "tiers": {}}
        for name in [t["name"] for t in self.tiers] + ["none"]:
            s = self.stats[name]
            rep["tiers"][name] = {
                "cells": s["cells"],
                "share_of_primary": round(s["cells"] / max(n_primary, 1), 4),
                "users": len(s["users"]),
                "charts": len(s["charts"]),
                "median_n": float(np.median(self._cat(s["n"]))) if s["n"] else None,
                "median_attempts": float(np.median(self._cat(s["attempts"]))) if s["attempts"] else None,
                "rate_mix": {RATE_NAME.get(k, str(k)): v for k, v in sorted(s["rate"].items())},
                "median_star": round(float(np.nanmedian(self._cat(s["star"]))), 4) if s["star"] else None,
                "loss_med_std": round(float(np.nanstd(self._cat(s["loss_med"]))), 4) if s["loss_med"] else None,
                "loss_tail_std": round(float(np.nanstd(self._cat(s["loss_tail"]))), 4) if s["loss_tail"] else None,
            }
        for k, v in self.q.items():
            v = self._cat(v)
            if v.size:
                rep[f"{k}_quantiles"] = {str(q): round(float(np.nanquantile(v, q)), 3)
                                         for q in (.1, .25, .5, .75, .9)}
        rep["config"] = cfg
        return rep

    def calibrate(self) -> None:
        prim = {k: self._cat(v) for k, v in self.q.items()}
        print(f"\nprimary cells with trajectories: {prim['n'].size}")
        print("\n|delta| = |mean(末窗)-mean(前窗)| 分位数:", {
            str(q): round(float(np.nanquantile(prim["adelta"], q)), 3) for q in (.1, .25, .5, .75, .9)})
        print("sd_tail 分位数:", {
            str(q): round(float(np.nanquantile(prim["sd_tail"], q)), 3) for q in (.1, .25, .5, .75, .9)})
        print("span(年) 分位数:", {
            str(q): round(float(np.nanquantile(prim["span"], q)), 3) for q in (.1, .25, .5, .75, .9)})
        print("末窗离个人最好 (l_max - tl_mean) 分位数:", {
            str(q): round(float(np.nanquantile(prim["gap_best_tail"], q)), 3) for q in (.1, .25, .5, .75, .9)})
        print("\n== 存活率网格（占 primary %）==")
        for tag, key, grid in (("abs: |delta|<=a, sd_tail<=s", "sd_tail", (0.3, 0.5, 0.8)),
                               ("se : t_delta<=t, se_tail<=s", "se_tail", (0.2, 0.3, 0.45))):
            print(f"\n-- {tag} --")
            print(f"{'min_n':>5} {'span_d':>6} {'att':>4} {'a/t':>5} {'s':>5} {'cells':>8} {'%prim':>7}")
            nn, sp, ad = prim["n"], prim["span"], prim["adelta"]
            sec = prim["t_delta"] if key == "se_tail" else prim["adelta"]
            for min_n in (4, 5):
                for span_d in (0, 30, 90, 180):
                    for cap in ((2.0, 2.5, 3.0) if key == "se_tail" else (0.2, 0.35, 0.5)):
                        for s_max in grid:
                            m = (nn >= min_n) & (sp >= span_d / 365) & (sec <= cap) & (prim[key] <= s_max)
                            print(f"{min_n:5d} {span_d:6d} {'-':>4} {cap:5.1f} {s_max:5.2f} "
                                  f"{int(m.sum()):8d} {m.mean() * 100:6.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1k")
    ap.add_argument("--plays", default=None)
    ap.add_argument("--cells", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tail", type=int, default=3)
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--min-span-days", type=int, default=90)
    ap.add_argument("--min-attempts", type=int, default=8)
    ap.add_argument("--max-adelta", type=float, default=0.35)
    ap.add_argument("--max-sd-tail", type=float, default=0.5)
    ap.add_argument("--criterion", choices=["abs", "se", "joint"], default="abs",
                    help="哪套口径写入 conv_tier / converged（两套都写进 conv_tier_se / conv_tier_abs）")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--primary-only-out", action="store_true",
                    help="输出只含 primary 行（内存/体积友好；默认输出全部行）")
    args = ap.parse_args()

    plays_p = Path(args.plays or OUT / f"plays_v2_{args.tag}.parquet")
    cells_p = Path(args.cells or OUT / f"charts_v2_{args.tag}_v3.parquet")
    out_p = Path(args.out or OUT / f"charts_v2_{args.tag}_conv.parquet")

    families = json.loads(json.dumps(DEFAULT_TIERS))
    families["joint"][0].update({"min_n": args.min_n, "min_span_days": args.min_span_days,
                                 "min_attempts": args.min_attempts, "max_adelta": args.max_adelta,
                                 "max_sd_tail": args.max_sd_tail})
    # CLI 覆盖 A 层（两套口径的 A 层同步覆盖绝对值；se 口径的 se_tail 阈值单独给）
    families["abs"][0].update({"min_n": args.min_n, "min_span_days": args.min_span_days,
                               "min_attempts": args.min_attempts, "max_adelta": args.max_adelta,
                               "max_sd_tail": args.max_sd_tail})
    tiers = families[args.criterion]
    cfg = {"tag": args.tag, "tail": args.tail, "criterion": args.criterion, "tiers": tiers,
           "tiers_all": families["se"] if args.criterion == "se" else families["abs"],
           "note": "delta = mean(last tail plays) - mean(previous tail plays); loss = log(1-ACC), "
                   "higher = better accuracy; 收敛 = 末窗相对前窗无漂移 + 末窗精度足够"}

    # ---- 1. cell 名单（只读 3 列） ----
    keys_df = pd.read_parquet(cells_p, columns=["user_id", "chart_id", "primary"])
    n_primary = int(keys_df.primary.sum())
    prim_keys = (keys_df.user_id.to_numpy(np.int64) * KEY_MUL
                 + keys_df.chart_id.to_numpy(np.int64))[keys_df.primary.to_numpy()]
    del keys_df
    print(f"cells: {n_primary} primary (of the table) | primary-only output: {args.primary_only_out}",
          flush=True)

    # ---- 2. primary cell 的轨迹特征 ----
    plays = pd.read_parquet(plays_p, columns=["user_id", "chart_id", "loss", "year", "doy"])
    pk = plays.user_id.to_numpy(np.int64) * KEY_MUL + plays.chart_id.to_numpy(np.int64)
    keep = np.isin(pk, prim_keys)
    plays = plays[keep]
    del pk, keep, prim_keys
    print(f"plays in primary cells: {len(plays)} (from {plays_p})", flush=True)
    feats = cell_features(plays, tail=args.tail)
    del plays
    print(f"trajectories: {len(feats)} cells, feature cols {list(feats.columns[2:])}", flush=True)

    # ---- 3. 流式扫描 cells 表 -> 打标 + 写出 ----
    pf = pq.ParquetFile(cells_p)
    acc = Accum(tiers)
    writer = None
    feat_key = feats.user_id.to_numpy(np.int64) * KEY_MUL + feats.chart_id.to_numpy(np.int64)
    order = np.argsort(feat_key)
    feat_key_sorted = feat_key[order]
    feats = feats.iloc[order].reset_index(drop=True)
    try:
        for batch in pf.iter_batches(batch_size=500_000):
            df = batch.to_pandas()
            if args.primary_only_out:
                df = df[df.primary].reset_index(drop=True)
            k = df.user_id.to_numpy(np.int64) * KEY_MUL + df.chart_id.to_numpy(np.int64)
            idx = np.searchsorted(feat_key_sorted, k)
            idx_c = np.clip(idx, 0, max(len(feat_key_sorted) - 1, 0))
            hit = (len(feat_key_sorted) > 0) & (feat_key_sorted[idx_c] == k)
            for c in FEATURE_COLS:
                vals = np.full(len(df), np.nan, dtype=np.float32)
                vals[hit] = feats[c].to_numpy()[idx_c[hit]]
                df[c] = vals
            attempts = (np.nan_to_num(df.attempts.to_numpy(np.float64), nan=0.0)
                        if "attempts" in df.columns else np.zeros(len(df)))
            for fam in ("abs", "se", "joint"):
                tl = families[fam]
                t = family_of(df, attempts, fam, tl)
                names = np.array([x["name"] for x in tl] + ["none"], dtype=object)
                df[f"conv_tier_{fam}"] = names[np.where(t < 0, len(tl), t)]
                if fam == args.criterion:
                    df["conv_tier"] = df[f"conv_tier_{fam}"]
                    df["converged"] = t >= 0
                    acc.add(df, t)
            if args.write:
                tbl = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_p, tbl.schema)
                writer.write_table(tbl)
                del tbl
            del df
    finally:
        if writer is not None:
            writer.close()

    # ---- 4. 报告 ----
    rep = acc.report(n_primary, cfg)
    if args.calibrate:
        acc.calibrate()
    print(json.dumps({k: v for k, v in rep.items() if k != "config"}, indent=2, ensure_ascii=False))
    if args.write:
        out_p.with_suffix(".json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        print("wrote", out_p, flush=True)


if __name__ == "__main__":
    main()
