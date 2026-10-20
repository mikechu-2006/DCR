#!/usr/bin/env python
r"""Step 2 (version: map_analyser) -- osumania_map_analyser 特征作为冻结谱面特征.

与 step2/minacalc_as_feature（L1）是**同一个模型**，只换了谱面特征的来源：

    L_hat = b0 + bu[u] + bm[m] + alpha . Z[m] + <P[u], C_unit[m]>
             \_____ 主效应 _____/   \_______ 交互 _______/

    Z[m]     = 冻结的谱面特征（zscore / pca / raw，只在训练谱面上拟合）
    alpha    = 每个特征一个系数，从 0 初始化 ⇒ epoch 0 严格等于同子集的 bias 基线
    bm[m]    = 每图自由截距（--feat-bm feat 时去掉，谱面截距完全由特征生成）
    P[u]     = 玩家潜向量，维度 = 特征维数

三个模型名（--models）:
    bias    -- b0 + bu + bm（同一谱面子集上的基线）
    fc_add  -- 只加特征主效应（没有 P、没有交互）
    fc      -- 特征主效应 + 点积交互

为什么要这个版本（docs/step2_map_analyser.md）:
    MinaCalc 的 D 只有 8 维，且丢掉 OD、LN 时长、键型分布、BPM/时长等信息。
    osumania_map_analyser 在一张表里给出 6 个 MinaCalc 版本 × 8 skillset、4 套估算器
    （Sunny/Daniel/Azusa/Roxy）+ Interlude + 键型分析 + 结构量，共 100+ 维。

前置资产:
    python scripts/step0i_map_analyser.py --tag 1k --ids data/processed/whitelist_1k.csv \
        --cache data/osu_cache --out data/processed/map_analyser_1k.parquet --jobs 4
    python main.py -c configs/exp_map_analyser_1k.yaml
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step2_predict as sp          # noqa: E402  （模型/指标/检查）
import step2_minacalc_as_feature as ml1  # noqa: E402  （复用 MSDModel / build_model）

MODELS = ("bias", "fc_add", "fc")
MODEL_TO_ML1 = {"bias": "bias", "fc_add": "msd_add", "fc": "msd"}

ETT_VERSIONS = ["068", "070", "072", "0723", "074", "075"]
ETT_SKILLS = ["overall", "stream", "jumpstream", "handstream",
              "stamina", "jackspeed", "chordjack", "technical"]
CORE_PATTERNS = ["stream", "chordstream", "jacks", "coordination", "density", "wildcard"]

# 特征分组：按列名前缀显式列出，避免把 id / 状态 / 常量列卷进来
GROUPS = {
    "struct": ["od", "n_notes", "n_holds", "ln_ratio", "duration_s", "n_rows",
               "chord_ratio", "notes_per_sec", "hold_ms_per_note", "bpm_median",
               "bpm_max", "n_timing_points", "n_breaks", "break_s"],
    "ett": [f"ett_{v}_{s}" for v in ETT_VERSIONS for s in ETT_SKILLS],
    "est": ["sunny_star", "sunny_variety", "sunny_acc_scalar", "sunny_spikiness",
            "sunny_switches", "daniel_star", "azusa_star", "azusa_numeric",
            "azusa_raw_numeric", "roxy_star", "roxy_raw_numeric", "interlude_star",
            "sunnywin_ln_star"],
            # roxy_numeric 全表 95% 为空（Roxy 只在高难区间给 numeric），不进特征
    "patt": (["patt_hb_ratio", "patt_sv_amount", "patt_n_clusters", "patt_n_specific"]
             + [f"patt_{p}_{k}" for p in CORE_PATTERNS
                for k in ("n", "amount_s", "bpm_max", "importance", "mult")]),
    # 时间曲线摘要：只有 step0i --with-graph 产出的表才有，默认不进 --feat-groups
    "graph": ["graph_mean", "graph_p50", "graph_p90", "graph_max", "graph_sd",
              "graph_spike", "graph_hot_frac"],
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="osumania_map_analyser 特征作为冻结谱面特征")
    # 与 step2_predict / step2_minacalc_as_feature 同名同义
    ap.add_argument("--tag", default="1k")
    ap.add_argument("--target", default="loss")
    ap.add_argument("--models", default="bias,fc")
    ap.add_argument("--dim", type=int, default=16,
                    help="交互空间维度；--feat-reduce pca 时是主成分数，raw/zscore 时自动等于特征数")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--features", default="F0", choices=["F0", "F1", "F2"])
    ap.add_argument("--entity", default="player", choices=["player", "player-year"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="step2")
    ap.add_argument("--proc-dir", type=Path, default=sp.PROC)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--train-path", type=Path, default=None)
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--p-drift", default="none", choices=["none", "logpc", "time"])
    ap.add_argument("--d-constraint", default="none", choices=["none", "nonneg", "simplex"])
    ap.add_argument("--d-init-std", type=float, default=None)
    ap.add_argument("--d-logit-wd", type=float, default=None)
    ap.add_argument("--c-logit-wd", type=float, default=None)
    ap.add_argument("--c-init-std", type=float, default=None)
    ap.add_argument("--drop-bias", default="none", choices=["none", "u", "m", "both"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save-preds", action="store_true")
    # 资产与特征
    ap.add_argument("--feat-path", type=Path, default=None,
                    help="特征表；默认 {proc_dir}/map_analyser_{tag}.parquet")
    ap.add_argument("--feat-groups", default="ett,est,patt,struct",
                    help=f"逗号分隔的特征组，可选 {','.join(GROUPS)}；all = 全部")
    ap.add_argument("--feat-cols", default=None,
                    help="显式列名（逗号分隔），给了就忽略 --feat-groups")
    ap.add_argument("--feat-tag", default="",
                    help="写进 variant 的短标签（默认由特征组拼出来）")
    ap.add_argument("--feat-reduce", default="zscore", choices=["raw", "zscore", "pca"])
    ap.add_argument("--feat-main", action="store_true",
                    help="加 alpha . Z[m] 主效应（L1 的加性项）；不加 = 只有点积交互")
    ap.add_argument("--feat-bm", default="free", choices=["free", "feat"],
                    help="feat = 去掉 bm[m]，谱面截距完全由特征生成")
    ap.add_argument("--no-est-missing", action="store_true",
                    help="不把 'Azusa/Roxy 判为 out-of-scope' 作为额外 0/1 特征")
    ap.add_argument("--feat-missing-median", action="store_true", default=True,
                    help="缺失值用训练谱面的列中位数填补（默认开）")
    return ap.parse_args(argv)


def resolve_columns(df: pd.DataFrame, groups: str, explicit: str | None) -> tuple[list, list]:
    """返回 (列名, 组名列表)。缺失的列直接报错，避免静默少特征。"""
    if explicit:
        cols = [c.strip() for c in explicit.split(",") if c.strip()]
        names = ["custom"]
    else:
        names = [g.strip() for g in groups.split(",") if g.strip()]
        if names == ["all"]:
            names = list(GROUPS)
        bad = [g for g in names if g not in GROUPS]
        if bad:
            raise SystemExit(f"--feat-groups 里的 {bad} 不支持；可选 {list(GROUPS)} 或 all")
        cols = []
        for g in names:
            for c in GROUPS[g]:
                if c not in cols:
                    cols.append(c)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"特征表缺列 {missing}（前 12 列是 {list(df.columns)[:12]}）")
    return cols, names


def load_feat_table(path: Path, cols: list, use_est_missing: bool) -> tuple:
    """返回 (ids int64[n], X float32[n, k], 额外 0/1 列, info)。一行一张谱面。"""
    if not path.exists():
        raise SystemExit(
            f"{path} 不存在 —— 先产出特征表：\n"
            f"  python scripts/step0i_map_analyser.py --tag <tag> --ids data/processed/whitelist_<tag>.csv \\\n"
            f"      --cache data/osu_cache --out {path} --jobs 4")
    df = pd.read_parquet(path)
    missing = [c for c in ["beatmap_id", *cols] if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} 缺列 {missing}")
    n_raw = len(df)
    df = df.drop_duplicates("beatmap_id", keep="first")
    X = df[cols].to_numpy(np.float64)
    extra_cols, extra_names = [], []
    if use_est_missing and "azusa_star" in cols:
        # Azusa/Roxy 对低难谱面返回 out-of-scope（NaN）。这本身是难度信号，单列保留。
        extra_cols.append(df["azusa_star"].isna().to_numpy(np.float64))
        extra_names.append("est_out_of_scope")
    info = {
        "path": str(path),
        "n_rows": int(len(df)),
        "n_rows_raw": int(n_raw),
        "cols": list(cols),
        "extra_cols": extra_names,
        "analyser_version": (str(df["analyser_version"].iloc[0])
                             if "analyser_version" in df.columns and len(df) else None),
        "analyser_commit": (str(df["analyser_commit"].iloc[0])
                            if "analyser_commit" in df.columns and len(df) else None),
        "missing_rate": {c: float(np.mean(~np.isfinite(X[:, i])))
                         for i, c in enumerate(cols)},
    }
    X = np.concatenate([X, np.stack(extra_cols, axis=1)], axis=1) if extra_cols else X
    print(f"[feat] {path.name}: {len(df):,} 张谱面, {X.shape[1]} 维 "
          f"(+{len(extra_names)} 个缺失指示), analyser {info['analyser_version']}", flush=True)
    return df["beatmap_id"].to_numpy(np.int64), X.astype(np.float32), info


def build_feat_features(ids, X, item_ids, dim, reduce_mode, cols) -> tuple:
    """对齐到训练词表、按训练谱面中位数补缺失、zscore/pca/raw、行单位范数。

    与 ml1.build_msd_features 的差别只有一处：这里**不丢**含 NaN 的行（资产表里
    Azusa/Roxy 的 out-of-scope 是普遍现象），改用训练谱面列中位数填补。
    """
    pos = pd.Index(ids).get_indexer(item_ids)
    if (pos < 0).any():
        raise SystemExit(f"{int((pos < 0).sum())} 张训练谱面没有特征行")
    Xa = np.ascontiguousarray(X[pos], dtype=np.float64)
    miss = ~np.isfinite(Xa)
    if miss.any():
        med = np.nanmedian(np.where(miss, np.nan, Xa), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xa = np.where(miss, med, Xa)
        print(f"[feat] 训练谱面上 {int(miss.sum()):,} 个缺失值用列中位数填补"
              f"（{int((miss.any(0)).sum())} 列受影响）", flush=True)
    if reduce_mode == "raw":
        Z, info = Xa, {"reduce": "raw"}
    elif reduce_mode == "zscore":
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-12
        Z = (Xa - mu) / sd
        info = {"reduce": "zscore", "mean": mu, "scale": sd, "fitted_on": "train_charts"}
    else:  # pca
        mu = Xa.mean(0)
        Xc = Xa - mu
        Vt = np.linalg.svd(Xc, full_matrices=False)[2]
        d = min(dim, Vt.shape[0])
        if d != dim:
            print(f"[feat] 只有 {d} 个主成分，--dim {dim} 被截断", flush=True)
        Z = Xc @ Vt[:d].T
        Z = Z / (Z.std(0) + 1e-12)
        info = {"reduce": "pca", "pca_mean": mu, "pca_components": Vt[:d],
                "fitted_on": "train_charts"}
    if reduce_mode != "pca" and dim != Z.shape[1]:
        dim = Z.shape[1]
        print(f"[feat] raw/zscore 下 --dim 自动设为特征数 {dim}", flush=True)
    std = np.ascontiguousarray(Z, dtype=np.float32)
    nrm = np.linalg.norm(std, axis=1, keepdims=True)
    unit = (std / np.maximum(nrm, 1e-12)).astype(np.float32)
    info.update({"dim": int(unit.shape[1]), "cols_used": list(cols),
                 "raw_row_norm_mean": float(nrm.mean()),
                 "raw_row_norm_min": float(nrm.min()),
                 "raw_row_norm_max": float(nrm.max())})
    print(f"[feat] 特征 {unit.shape[0]:,} x {unit.shape[1]}  reduce={reduce_mode}  "
          f"行范数(unit)=1  标准化前 |Z| 均值 {nrm.mean():.3f}", flush=True)
    return unit, std, info


def main():
    args = parse_args()
    t_start = time.time()
    out_dir = Path(args.out_dir) if args.out_dir else args.proc_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = sp.pick_device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    print(f"engine: torch {torch.__version__} on {device}, threads={torch.get_num_threads()}, "
          f"deterministic_algorithms={torch.are_deterministic_algorithms_enabled()}", flush=True)

    models_wanted = [s for s in args.models.split(",") if s]
    bad = [m for m in models_wanted if m not in MODELS]
    if bad:
        raise SystemExit(f"--models 里的 {bad} 不支持；可用 {', '.join(MODELS)}")
    feat_path = Path(args.feat_path) if args.feat_path \
        else Path(args.proc_dir) / f"map_analyser_{args.tag}.parquet"
    drop_bias = ml1.effective_drop_bias(args.drop_bias, args.feat_bm)
    # 复用 ml1.build_model / MSDModel 的接口名
    args.msd_main = bool(args.feat_main)
    args.msd_bm = args.feat_bm

    need_pc = args.features in ("F1", "F2") or args.p_drift == "logpc"
    need_ts = args.features == "F2" or args.entity == "player-year" or args.p_drift == "time"
    play_cols = sp.BASE_COLS + [args.target] \
        + (["playcount_cur"] if need_pc else []) + (["timestamp"] if need_ts else [])

    tr_all = sp.load_play_table(args.tag, "step1", play_cols, path=args.train_path,
                                proc_dir=args.proc_dir, optional_cols=(args.split_col,))

    # --------------------------------------------------- 特征覆盖率过滤（划分之前）
    raw = pd.read_parquet(feat_path, columns=["beatmap_id"])
    feat_ids = pd.to_numeric(raw["beatmap_id"], errors="coerce").dropna().astype(np.int64)
    have = set(feat_ids.tolist())
    n_all = len(tr_all)
    tr_all = tr_all[tr_all.beatmap_id.isin(have)].reset_index(drop=True)
    feat_drop = {"rows_dropped": int(n_all - len(tr_all)), "rows_total": int(n_all)}
    print(f"[feat] 有特征的谱面 {len(have):,} 张；play 表裁掉 {n_all - len(tr_all):,}/{n_all:,} 行",
          flush=True)
    if len(tr_all) == 0:
        raise SystemExit("裁到特征覆盖集之后没有 play 了")

    cols, group_names = resolve_columns(pd.read_parquet(feat_path), args.feat_groups,
                                        args.feat_cols)
    feat_ids, X, feat_meta = load_feat_table(feat_path, cols, not args.no_est_missing)
    all_cols = list(feat_meta["cols"]) + list(feat_meta["extra_cols"])

    # ------------------------------------------------------------------- split
    if args.split_col not in tr_all.columns:
        raise SystemExit(f"step1 表里没有 '{args.split_col}' 列 —— 划分应在 step1 完成")
    lab = tr_all[args.split_col].astype(str).to_numpy()
    unknown = sorted(set(lab.tolist()) - {"train", "test", "dropped"})
    if unknown:
        raise SystemExit(f"'{args.split_col}' 列出现未知标签 {unknown}")
    tr_mask, te_mask = lab == "train", lab == "test"
    train = tr_all[tr_mask].reset_index(drop=True)
    test = tr_all[te_mask].reset_index(drop=True)
    n_dropped = int((lab == "dropped").sum())
    if len(train) == 0 or len(test) == 0:
        raise SystemExit(f"train={len(train):,} / test={len(test):,} 行，无法训练或评估")

    split_spec = sp.read_split_spec(args.train_path, args.proc_dir, args.tag)
    k = int(max(int(train.beatmap_id.max()), int(test.beatmap_id.max()))) + 1
    row_keys = sp.pair_keys(tr_all.player_id.to_numpy(), tr_all.beatmap_id.to_numpy(), k)
    unit_kind = (split_spec or {}).get("unit", "pair")
    if unit_kind == "beatmap":
        ids_u = tr_all.beatmap_id.to_numpy()
        unit_tr, unit_te = np.unique(ids_u[lab != "test"]), np.unique(ids_u[te_mask])
    else:
        unit_tr = np.unique(row_keys[lab != "test"])
        unit_te = np.unique(row_keys[te_mask])
    units_all = np.union1d(unit_tr, unit_te)
    is_test = np.isin(units_all, unit_te)
    n_pairs, n_test_pairs = int(units_all.size), int(unit_te.size)
    cross_units = np.intersect1d(unit_tr, unit_te, assume_unique=True)
    test_pairs = np.unique(row_keys[te_mask])
    tr_pairs = np.unique(row_keys[tr_mask])
    expected_test_frac = float((split_spec or {}).get("actual_test_frac")
                               or (n_test_pairs / max(n_pairs, 1)))
    frac_tol = 0.02
    print(f"[split] train {len(train):,} 行 / test {len(test):,} 行 / dropped {n_dropped:,} 行 | "
          f"单位 {unit_kind}: train {unit_tr.size:,} / test {unit_te.size:,}"
          + (f" | step1: {split_spec.get('method')} train_frac={split_spec.get('train_frac')} "
             f"seed={split_spec.get('seed')}" if split_spec else " | 未找到 step1 报告 json"),
          flush=True)
    del row_keys, lab, tr_all
    gc.collect()

    chk = sp.Checklist()
    chk.check("pair_never_crosses_split", cross_units.size == 0,
              f"{cross_units.size} 个划分单位同时出现在两侧")
    tr_keys = sp.pair_keys(train.player_id.to_numpy(), train.beatmap_id.to_numpy(), k)
    te_keys = sp.pair_keys(test.player_id.to_numpy(), test.beatmap_id.to_numpy(), k)
    tr_in_test = np.isin(tr_keys, test_pairs)
    te_in_test = np.isin(te_keys, test_pairs)
    chk.check("train_and_test_disjoint",
              (not tr_in_test.any()) and bool(te_in_test.all()),
              f"train 落在留出对上的行 = {int(tr_in_test.sum()):,}; "
              f"test 落在留出对上的行 = {int(te_in_test.sum()):,}/{len(test):,}")
    if split_spec:
        frac_ok = abs(float(is_test.mean()) - expected_test_frac) <= frac_tol
        frac_detail = (f"观测 test 单位占比 {is_test.mean():.4%} vs step1 报告 "
                       f"{expected_test_frac:.2%}（容差 {frac_tol:.0%}）")
    else:
        frac_ok, frac_detail = True, f"未找到 step1 报告 json：观测占比 {is_test.mean():.4%}"
    chk.check("test_pair_frac", frac_ok, frac_detail)
    del tr_keys, te_keys, tr_in_test, te_in_test
    gc.collect()

    # ------------------------------------------------------------- entity / item
    def add_entity(df):
        if args.entity == "player-year":
            y = pd.to_datetime(df.timestamp).dt.year.to_numpy()
            return df.assign(entity=df.player_id.to_numpy(np.int64) * 100 + (y - 2000))
        return df.assign(entity=df.player_id.to_numpy(np.int64))

    train, test = add_entity(train), add_entity(test)
    ent_ids = np.unique(train.entity.to_numpy())
    item_ids = np.unique(train.beatmap_id.to_numpy())
    ents = {e: i for i, e in enumerate(ent_ids)}
    items = {m: i for i, m in enumerate(item_ids)}
    print(f"train vocab: {len(ents):,} entities, {len(items):,} items", flush=True)

    unit, std, feat_info = build_feat_features(feat_ids, X, item_ids,
                                              args.dim, args.feat_reduce, all_cols)
    args.dim = int(unit.shape[1])
    chart_feats = torch.as_tensor(unit, dtype=torch.float32, device=device)
    feat_std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
    print(f"[feat] P = ({len(ents):,}, {args.dim})；每图自由参数: "
          + ("0 (--feat-bm feat)" if args.feat_bm == "feat" else f"bm 一张 ({len(items):,})"),
          flush=True)

    t_ref = 0.0
    if args.p_drift == "time":
        t_ref = float(pd.to_datetime(train.timestamp).to_numpy(dtype="datetime64[D]")
                      .astype(np.float64).mean() / 365.25)

    def encode(df):
        e = sp.map_ids(df.entity.to_numpy(), ent_ids)
        i = sp.map_ids(df.beatmap_id.to_numpy(), item_ids)
        x1 = np.log(df.playcount_cur.to_numpy(np.float64)).astype(np.float32) if need_pc else None
        tau = ((pd.to_datetime(df.timestamp).dt.dayofyear.to_numpy() - 1) / 365.0).astype(np.float32) \
            if args.features == "F2" else None
        tv = None
        if args.p_drift == "time":
            days = pd.to_datetime(df.timestamp).to_numpy(dtype="datetime64[D]").astype(np.float64)
            tv = (days / 365.25 - t_ref).astype(np.float32)
        return e, i, x1, tau, tv, df[args.target].to_numpy(np.float32)

    tr_e, tr_i, tr_x, tr_tau, tr_t, tr_y = encode(train)
    te_e, te_i, te_x, te_tau, te_t, te_y = encode(test)
    tr_ok = (tr_e >= 0) & (tr_i >= 0)
    te_ok = (te_e >= 0) & (te_i >= 0)
    print(f"train rows in vocab = {int(tr_ok.sum()):,}/{len(train):,}   "
          f"test rows covered = {int(te_ok.sum()):,}/{len(test):,} ({te_ok.mean():.2%})", flush=True)

    if args.tag in sp.LOSS_RANGE:
        lo, hi = sp.LOSS_RANGE[args.tag]
        loss_ok = bool(np.isfinite(te_y).all() and te_y.min() >= lo - 1e-3 and te_y.max() <= hi + 1e-3)
        loss_detail = f"test({args.tag}) loss in [{te_y.min():.4f}, {te_y.max():.4f}], 期望 [{lo}, {hi}]"
    else:
        loss_ok = bool(np.isfinite(te_y).all() and (te_y <= 0).all())
        loss_detail = f"test({args.tag}) loss in [{te_y.min():.4f}, {te_y.max():.4f}]"
    chk.check("loss_in_range", loss_ok, loss_detail)
    chk.check("vocab_from_train_only",
              np.array_equal(ent_ids, np.unique(train.entity.to_numpy()))
              and np.array_equal(item_ids, np.unique(train.beatmap_id.to_numpy())),
              f"{ent_ids.size:,} entities / {item_ids.size:,} items == unique train ids")
    chk.check("determinism_selftest", sp.determinism_selftest(device),
              "same seed -> bitwise identical parameters")
    step = max(1, len(item_ids) // 64)
    probe = np.arange(len(item_ids))[::step][:64]
    pos = pd.Index(feat_ids).get_indexer(item_ids[probe])
    norms = chart_feats.norm(dim=1)
    chk.check("feat_aligned_with_item_ids",
              bool((pos >= 0).all()) and len(chart_feats) == len(item_ids)
              and bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-4)),
              f"{len(item_ids):,} 训练谱面; {len(probe)} 个探针 id 全部命中; "
              f"row norm in [{float(norms.min()):.4f}, {float(norms.max()):.4f}]")

    def dev(a):
        return torch.as_tensor(a, device=device)

    tr_e_t, tr_i_t, tr_y_t = dev(tr_e), dev(tr_i), dev(tr_y)
    tr_x_t = dev(tr_x) if tr_x is not None else None
    tr_tau_t = dev(tr_tau) if tr_tau is not None else None
    tr_t_t = dev(tr_t) if tr_t is not None else None
    te_e_t, te_i_t = dev(te_e[te_ok]), dev(te_i[te_ok])
    te_y_np = te_y[te_ok]
    te_x_t = dev(te_x[te_ok]) if te_x is not None else None
    te_tau_t = dev(te_tau[te_ok]) if te_tau is not None else None
    te_t_t = dev(te_t[te_ok]) if te_t is not None else None
    idx_tr = np.flatnonzero(tr_ok).astype(np.int64)
    del tr_e, tr_i, tr_x, tr_tau, tr_t, tr_y, te_e, te_i, te_x, te_tau, te_t, te_y
    gc.collect()

    # ------------------------------------------------------------------- 训练
    results, fitted, mech_by_model, degeneracy = {}, {}, {}, {}
    for kind in models_wanted:
        model = ml1.build_model(MODEL_TO_ML1[kind], args, device, len(ents), len(items),
                                args.dim, chart_feats, feat_std_t, drop_bias)
        opt, wd_of = sp.build_optimizer(model, args)
        mech_by_model[kind] = {"feat_main": args.feat_main, "feat_bm": args.feat_bm,
                               "drop_bias": drop_bias,
                               "weight_decay": {kk: wd_of[kk] for kk in wd_of},
                               "no_decay": sorted(model.no_decay)}
        rng_tr = np.random.default_rng(args.seed)
        t0 = time.time()
        for ep in range(args.epochs):
            perm = rng_tr.permutation(idx_tr.size)
            for s in range(0, perm.size, args.batch):
                b = torch.as_tensor(idx_tr[perm[s:s + args.batch]], device=device)
                x1 = tr_x_t[b] if tr_x_t is not None else None
                tau = tr_tau_t[b] if tr_tau_t is not None else None
                tt = tr_t_t[b] if tr_t_t is not None else None
                opt.zero_grad(set_to_none=True)
                pred = model.forward(tr_e_t[b], tr_i_t[b], x1, tau, tt)
                ((pred - tr_y_t[b]) ** 2).mean().backward()
                opt.step()
                model.gauge_fix()
            if args.eval_every and (ep + 1) % args.eval_every == 0:
                pv = model.predict(te_e_t, te_i_t, te_x_t, te_tau_t, te_t_t)
                print(f"  [{kind}] epoch {ep+1:3d}  test RMSE = "
                      f"{np.sqrt(((pv - te_y_np) ** 2).mean()):.5f}  ({time.time()-t0:.1f}s)",
                      flush=True)
        model.gauge_fix()
        pv = model.predict(te_e_t, te_i_t, te_x_t, te_tau_t, te_t_t)
        results[kind] = sp.report(pv, te_y_np, len(test))
        fitted[kind] = model
        degeneracy[kind] = sp.degeneracy_report(model)
        sp.warn_degenerate(kind, degeneracy[kind])
        alpha = model.np_array("alpha")
        extra = "" if alpha is None else f"  |alpha|={np.abs(alpha).sum():.4f}"
        print(f"{kind:8s} test rmse={results[kind]['rmse']:.5f} "
              f"cov={results[kind]['coverage']:.3f}{extra}", flush=True)

    chk.check("coverage_reported",
              all(("coverage" in r) and ("n_scored" in r) and r["coverage"] is not None
                  for r in results.values()),
              f"{len(results)} 个结果块带 coverage + n_scored")
    leak = np.isin(sp.pair_keys(train.player_id.to_numpy()[idx_tr],
                                train.beatmap_id.to_numpy()[idx_tr], k), test_pairs).sum()
    chk.check("no_test_leak", bool(leak == 0 and idx_tr.size == int(tr_ok.sum())),
              f"训练用行 = {idx_tr.size:,}；其中落在留出对上的 = {int(leak):,}")
    del test_pairs, tr_pairs
    gc.collect()

    # --------------------------------------------------------------------- save
    inv_e = {v: kk for kk, v in ents.items()}
    inv_i = {v: kk for kk, v in items.items()}
    feat_tag = args.feat_tag or ("+".join(group_names) if group_names != ["custom"] else "custom")
    reduce_tag = f"{args.feat_reduce}{args.dim}" \
        + ("" if not args.feat_main else "_main") \
        + ("" if args.feat_bm == "free" else "_bmfeat")
    variant = args.models.replace(",", "+") + f"_{feat_tag}_{reduce_tag}"
    if args.features != "F0":
        variant += f"_{args.features}"
    if args.entity != "player":
        variant += "_py"
    if args.p_drift != "none":
        variant += {"logpc": "_pd", "time": "_pdt"}[args.p_drift]
    if args.drop_bias != "none":
        variant += {"u": "_nobu", "m": "_nobm", "both": "_nobubm"}[args.drop_bias]
    feat_names = ([f"z_{c}" for c in all_cols] if args.feat_reduce in ("raw", "zscore")
                  else [f"pc{i + 1}" for i in range(args.dim)])
    entity_ids = np.array(list(inv_e.values()), dtype=np.int64)
    beatmap_ids = np.array(list(inv_i.values()), dtype=np.int32)
    npz_paths = {}
    for kind, model in fitted.items():
        path = out_dir / f"{args.out_prefix}_{args.tag}_{kind}_{args.features}{feat_tag}_{reduce_tag}_{args.target}.npz"
        extra = {}
        alpha = model.np_array("alpha")
        if alpha is not None:
            extra["alpha"] = alpha
            extra["feat_std"] = std
        C_eff = model.export_C()
        np.savez(path, entity_ids=entity_ids, beatmap_ids=beatmap_ids,
                 P=model.np_array("P", np.zeros((len(ents), args.dim), np.float32)),
                 V=model.np_array("V", np.zeros((len(ents), args.dim), np.float32)),
                 C=C_eff, D=model.export_D() if "D" in model.p else np.zeros_like(C_eff),
                 b0=model.np_array("b0"),
                 bu=model.np_array("bu", np.zeros(len(ents), np.float32)),
                 bm=model.np_array("bm", np.zeros(len(items), np.float32)),
                 feat_cols=np.array(feat_names),
                 **extra)
        npz_paths[kind] = path

    ok9, detail9 = True, []
    for kind, path in npz_paths.items():
        with np.load(path) as z:
            good = (len(z["entity_ids"]) == len(ents) and len(z["beatmap_ids"]) == len(items)
                    and z["C"].shape == (len(items), args.dim)
                    and z["D"].shape == z["C"].shape
                    and (("alpha" not in z.files) or z["alpha"].shape == (args.dim,)))
        ok9 = ok9 and bool(good)
        detail9.append(f"{kind}:{'ok' if good else 'BAD'}")
    chk.check("npz_matches_summary", ok9, " ".join(detail9))
    chk.require()

    preds_path = None
    if args.save_preds:
        slug = "".join(ch if ch.isalnum() or ch in "+-" else "_" for ch in variant)[:80]
        preds_path = out_dir / f"{args.out_prefix}_{args.tag}_preds_map_analyser_{slug}_{args.target}.npz"
        np.savez(preds_path, player_id=test.player_id.to_numpy()[te_ok],
                 beatmap_id=test.beatmap_id.to_numpy()[te_ok],
                 y=te_y_np, variant=variant,
                 **{f"pred_{kind}": fitted[kind].predict(
                     te_e_t, te_i_t, te_x_t, te_tau_t, te_t_t) for kind in fitted})
        print(f"wrote {preds_path.name} ({int(te_ok.sum()):,} test rows)")

    def jsonable(info: dict) -> dict:
        out = {}
        for kk, vv in info.items():
            if isinstance(vv, (np.floating, np.integer)):
                out[kk] = float(vv)
            elif isinstance(vv, np.ndarray):
                out[kk] = vv.round(6).tolist()
            elif isinstance(vv, (str, int, float, bool, type(None), list)):
                out[kk] = vv
            else:
                out[kk] = str(vv)
        return out

    feat_json = jsonable({**feat_meta, **feat_info})
    summary = {
        "variant": variant,
        "args": {kk: (str(v) if isinstance(v, Path) else v) for kk, v in vars(args).items()},
        "mechanism": {"feat_main": args.feat_main, "feat_bm": args.feat_bm,
                      "drop_bias": drop_bias, "by_model": mech_by_model},
        "engine": {"name": "torch", "version": torch.__version__, "device": str(device),
                   "threads": torch.get_num_threads(),
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "split": {"from": "step1", "col": args.split_col, "spec": split_spec,
                  "unit": unit_kind, "n_units": n_pairs, "n_test_units": n_test_pairs,
                  "observed_test_frac": float(is_test.mean()),
                  "rows": {"train": int(len(train)), "test": int(len(test)),
                           "dropped": n_dropped}},
        "sizes": {"train_plays": int(len(train)), "test_plays": int(len(test)),
                  "n_entities": len(ents), "n_items": len(items),
                  "train_rows_in_vocab": int(tr_ok.sum()),
                  "test_rows_covered": int(te_ok.sum())},
        "feat": {**feat_json, "groups": group_names,
                 "filtered_rows": feat_drop, "dim_used": int(args.dim),
                 "alpha": {kind: (None if fitted[kind].np_array("alpha") is None
                                  else fitted[kind].np_array("alpha").round(5).tolist())
                           for kind in fitted}},
        "results": {k: v for k, v in results.items()},
        "degeneracy": degeneracy,
        "preds_path": None if preds_path is None else str(preds_path),
        "checks": chk.items,
        "seconds": round(time.time() - t_start, 1),
    }
    merged_path = out_dir / f"{args.out_prefix}_{args.tag}_summary_{args.target}.json"
    merged = json.loads(merged_path.read_text()) if merged_path.exists() else {"runs": {}}
    merged["runs"][variant] = summary
    merged_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    print(f"\nwrote npz per model + {merged_path.name} ({summary['seconds']}s total)")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
