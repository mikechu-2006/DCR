#!/usr/bin/env python
"""Step 2 (version: minacalc_as_feature) -- MinaCalc MSD 作为冻结谱面特征 (L1).

方法（设计见 docs/step2_minacalc.md）:

    谱面侧 = MinaCalc 的 MSD 向量 D_m（冻结、不训练）；玩家侧仍然是可学的 P_u。
    与 step2/predict --chart-content cm3p 的唯一区别是"内容向量"换成 MSD，并多出一个
    线性主效应项:

        L_hat = b0 + bu[u] + bm[m] + alpha . D_std[m] + <P[u], C_unit[m]>
                 \\_____ 主效应 _____/   \\_______ 交互 _______/

      不加 --msd-main  : 去掉 alpha 项（= CM3P 那一套的 MSD 版，只有交互）
      --msd-bm  msd  : 去掉 bm[m]，谱面截距完全由 alpha . D_std[m] 生成（每图自由参数 = 0，
                       新谱面/无训练 play 的谱面也能拿到截距）
      --msd-bm free  : 保留 bm[m]（默认，与 step2/predict 的结构一致）

    alpha 从 0 初始化，所以任何配置在 epoch 0 都严格等于 step2/predict 的同名模型；
    加训后 alpha 的 L2 直接可读作"这些 MSD 维度解释了多少谱面截距"。

为什么这样接（docs/step2_minacalc.md 第 1 节）:
    * 谱面向量只依赖 .osu（不依赖成绩），所以它天然可迁移到 beatmap-random 划分与冷启动谱面；
    * 玩家侧不反解（R2 的 SSR 路线），而是留给模型学（R1），避免"用历史反解到测试局"的泄漏；
    * MSD 的 goal=0.93 是 Etterna 自己的 score 语义，不是 mania ACC，所以这里只当**相对**特征用：
      特征在训练谱面上标准化（--msd-reduce zscore/pca），绝对刻度交给 b0/bu/bm 去吸收。

与 step2/predict 的一致性（刻意的）:
    划分、词表、损失、指标、检查项、npz/summary 的字段都沿用 step2_predict.py：本文件 import
    它并复用 EmbeddingModel / report / Checklist / build_optimizer，只替换谱面特征的来源。
    这样 --models bias 就是**同一谱面子集上**的基线，两边数字可直接比。

    python scripts/step0h_minacalc_msd.py --tag 1k --cache data/osu_cache \
        --ids manifests/whitelist_1k.csv --lib build/libminacalc.so --out data/processed/minacalc_msd_1k.parquet
    python scripts/step2_minacalc_as_feature.py --tag 1k --models bias,msd --msd-main
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
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step2_predict as sp  # noqa: E402  （同目录，复用模型/指标/检查）

PROC = sp.PROC
BASE_COLS = sp.BASE_COLS
LOSS_RANGE = sp.LOSS_RANGE

# MinaCalc 的 8 个输出，顺序 = Skillset 枚举（NoteDataStructures.h）
SKILLS = ["overall", "stream", "jumpstream", "handstream",
          "stamina", "jackspeed", "chordjack", "technical"]
MSD_COLS = ["msd_" + s for s in SKILLS]
# 默认只用 7 个技能方向：overall 是它们自己的聚合（max(aggregate_skill, max)），留着会共线
DEFAULT_SKILLS = SKILLS[1:]
MODELS = ("bias", "msd_add", "msd")


# ------------------------------------------------------------------- MSD 资产读取
def load_msd_table(path: Path, skills: list) -> tuple:
    """(ids int64[n], X float32[n, k], info) —— 一行一张谱面。

    X 的列顺序 == skills（即 msd_<skill>）。重复 beatmap_id 只保留第一行；含 NaN/inf 的行
    直接丢掉（生产者那边算失败时可能写 NaN）。
    """
    if not path.exists():
        raise SystemExit(
            f"{path} 不存在 —— 先产出 MSD 表：\n"
            f"  python scripts/step0h_minacalc_msd.py --tag <tag> --cache data/osu_cache \\\n"
            f"      --ids manifests/whitelist_<tag>.csv --lib build/libminacalc.so --out {path}\n"
            f"  或把任一列名相同的表放到该路径（列: beatmap_id + msd_overall..msd_technical）")
    cols = ["msd_" + s for s in skills]
    df = pd.read_parquet(path)
    missing = [c for c in ["beatmap_id", *cols] if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} 缺少列 {missing}；实际列 = {list(df.columns)[:16]}")
    n_raw = len(df)
    df = df.drop_duplicates("beatmap_id", keep="first")
    X = df[cols].to_numpy(np.float64)
    ok = np.isfinite(X).all(1)
    if not ok.all():
        print(f"[msd] {int((~ok).sum()):,} 行含 NaN/inf，已丢弃", flush=True)
    df, X = df[ok], X[ok]
    info = {
        "path": str(path),
        "skills": list(skills),
        "n_rows": int(len(df)),
        "n_rows_raw": int(n_raw),
        "calc_version": int(df["calc_version"].iloc[0])
        if "calc_version" in df.columns and len(df) else None,
        "has_hold_ratio": bool("hold_ratio" in df.columns),
    }
    print(f"[msd] {path.name}: {len(df):,} charts, {len(skills)} 维, "
          f"calc_version={info['calc_version']}", flush=True)
    return df["beatmap_id"].to_numpy(np.int64), X.astype(np.float32), info


def build_msd_features(ids, X, item_ids, dim, reduce_mode) -> tuple:
    """把 MSD 对齐到训练词表顺序，然后标准化 / PCA / 原样，返回 (unit, std, info)。

    unit: 行单位范数，喂给交互项 <P[u], C_unit[m]>（与 CM3P 的约定一致，检查项会验证）
    std : 标准化但保留幅度，喂给主效应 alpha . D_std[m]（单位范数会把"这张图有多难"抹掉）
    变换只在**训练谱面**上拟合（与 align_content 的 pca 同约定）。
    """
    pos = pd.Index(ids).get_indexer(item_ids)
    if (pos < 0).any():
        raise SystemExit(f"{int((pos < 0).sum())} 张训练谱面没有 MSD 行；"
                         f"play 表必须先裁到有 MSD 的谱面（本脚本默认会裁）")
    Xa = np.ascontiguousarray(X[pos], dtype=np.float32)
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
            print(f"[msd] 只有 {d} 个主成分，--dim {dim} 被截断", flush=True)
        Z = Xc @ Vt[:d].T
        Z = Z / (Z.std(0) + 1e-12)
        info = {"reduce": "pca", "pca_mean": mu, "pca_components": Vt[:d],
                "fitted_on": "train_charts"}
    if reduce_mode != "pca" and dim != Z.shape[1]:
        raise SystemExit(f"--msd-reduce {reduce_mode} 得到 {Z.shape[1]} 维，"
                         f"与 --dim {dim} 不一致（zscore/raw 时 --dim 必须等于维数）")
    std = np.ascontiguousarray(Z, dtype=np.float32)
    nrm = np.linalg.norm(std, axis=1, keepdims=True)
    unit = (std / np.maximum(nrm, 1e-12)).astype(np.float32)
    info.update({"dim": int(unit.shape[1]),
                 "raw_row_norm_mean": float(nrm.mean()),
                 "raw_row_norm_min": float(nrm.min()),
                 "raw_row_norm_max": float(nrm.max())})
    print(f"[msd] 特征 {unit.shape[0]:,} x {unit.shape[1]}  reduce={reduce_mode}  "
          f"row-norm(unit)=1  标准化前 |D| 均值 {nrm.mean():.3f}", flush=True)
    return unit, std, info


# ------------------------------------------------------------------------ 模型
class MSDModel(sp.EmbeddingModel):
    """step2_predict.EmbeddingModel + 一个 MSD 线性主效应 alpha . D_std[m]。

    其余（b0/bu/bm/P/Pc/V/w1、p_drift、F1/F2、gauge_fix、export_*）全部原样继承，
    所以它就是"同一模型 + 一项谱面特征"，不是另写一个模型。
    """

    def __init__(self, *args, msd_main=None, **kwargs):
        super().__init__(*args, **kwargs)
        if msd_main is not None:
            dev = self.p["b0"].device
            self.register_buffer(
                "msd_main_feat",
                torch.as_tensor(msd_main, dtype=torch.float32, device=dev))
            self.p["alpha"] = nn.Parameter(
                torch.zeros(msd_main.shape[1], dtype=torch.float32, device=dev))
            self.active = [*self.active, "alpha"]

    def forward(self, u, m, x1=None, tau=None, t=None):
        out = super().forward(u, m, x1, tau, t)
        if "alpha" in self.p:
            out = out + (self.msd_main_feat[m] * self.p["alpha"]).sum(1)
        return out


def effective_drop_bias(user_drop_bias: str, msd_bm: str) -> str:
    """--msd-bm msd 要把 bm 拿掉：与用户已有的 --drop-bias 合并。"""
    if msd_bm == "free":
        return user_drop_bias
    return {"none": "m", "u": "both", "m": "m", "both": "both"}[user_drop_bias]


def build_model(name, args, device, n_ent, n_item, dim, unit, std, drop_bias):
    common = (n_ent, n_item, dim, args.features, args.seed, device, args.p_drift,
              args.d_constraint, args.d_init_std, drop_bias)
    if name == "bias":
        if args.msd_main or args.msd_bm == "msd":
            print("  [bias] MSD 开关对它无效（它没有任何谱面特征）；跑的就是 step2/predict "
                  "同一个基线，只是谱面子集被裁到有 MSD 的那些", flush=True)
        return sp.EmbeddingModel("bias", *common)
    if name == "msd_add":
        # 纯加性：谱面只有 alpha . D_std[m]，没有 P、没有交互
        return MSDModel("bias", *common, msd_main=std)
    if name == "msd":
        return MSDModel("mf_dot", *common, content="msd", chart_feats=unit,
                        msd_main=std if args.msd_main else None,
                        c_init_std=args.c_init_std)
    raise SystemExit(f"未知模型 {name!r}；--models 只接受 {', '.join(MODELS)}")


# ------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="MinaCalc MSD 作为冻结谱面特征 (L1)")
    # 与 step2_predict.py 同名同义
    ap.add_argument("--tag", default="1k")
    ap.add_argument("--target", default="loss")
    ap.add_argument("--models", default="bias,msd")
    ap.add_argument("--dim", type=int, default=16,
                    help="交互空间维度；--msd-reduce pca 时是主成分数，zscore/raw 时必须等于维数")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--features", default="F0", choices=["F0", "F1", "F2"])
    ap.add_argument("--entity", default="player", choices=["player", "player-year"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="step2")
    ap.add_argument("--proc-dir", type=Path, default=PROC)
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
    # MSD 专有
    ap.add_argument("--msd-path", type=Path, default=None,
                    help="MSD 表；默认 {proc_dir}/minacalc_msd_{tag}.parquet")
    ap.add_argument("--msd-skills", default=",".join(DEFAULT_SKILLS),
                    help="用哪些技能方向（默认 7 个，不含 overall）")
    ap.add_argument("--msd-reduce", default="zscore", choices=["raw", "zscore", "pca"])
    ap.add_argument("--msd-main", action="store_true",
                    help="加 alpha . D_std[m] 主效应（L1 的加性项）；不加 = 只有点积交互")
    ap.add_argument("--msd-bm", default="free", choices=["free", "msd"],
                    help="msd = 去掉 bm[m]，谱面截距完全由 MSD 生成")
    ap.add_argument("--allow-unversioned-msd", action="store_true",
                    help="MSD 表没有 calc_version 列时也继续（默认直接报错：263 与 5xx 不可比）")
    return ap.parse_args(argv)


# ------------------------------------------------------------------------ main
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
    skills = [s for s in args.msd_skills.split(",") if s]
    bad_sk = [s for s in skills if s not in SKILLS]
    if bad_sk:
        raise SystemExit(f"--msd-skills 里的 {bad_sk} 不在 {SKILLS}")
    msd_path = Path(args.msd_path) if args.msd_path \
        else Path(args.proc_dir) / f"minacalc_msd_{args.tag}.parquet"
    drop_bias = effective_drop_bias(args.drop_bias, args.msd_bm)

    need_pc = args.features in ("F1", "F2") or args.p_drift == "logpc"
    need_ts = args.features == "F2" or args.entity == "player-year" or args.p_drift == "time"
    cols = BASE_COLS + [args.target] \
        + (["playcount_cur"] if need_pc else []) + (["timestamp"] if need_ts else [])

    tr_all = sp.load_play_table(args.tag, "step1", cols, path=args.train_path,
                                proc_dir=args.proc_dir, optional_cols=(args.split_col,))

    # ------------------------------------------------------- MSD 覆盖率过滤
    # 与 step2/predict 的 content 过滤同位置、同理由：在划分**之前**裁，否则测试集会被换掉。
    msd_ids, msd_X, msd_meta = load_msd_table(msd_path, skills)
    if msd_meta["calc_version"] is None and not args.allow_unversioned_msd:
        raise SystemExit(f"{msd_path} 没有 calc_version 列。263 与 5xx 的 MSD 不可比 —— "
                         f"请用 scripts/step0h_minacalc_msd.py 产出，或显式 --allow-unversioned-msd")
    have = set(msd_ids.tolist())
    n_all = len(tr_all)
    tr_all = tr_all[tr_all.beatmap_id.isin(have)].reset_index(drop=True)
    msd_drop = {"rows_dropped": int(n_all - len(tr_all)), "rows_total": int(n_all)}
    print(f"[msd] 有 MSD 的谱面 {len(have):,} 张；play 表裁掉 {n_all - len(tr_all):,}/{n_all:,} 行"
          f"（train 与 test 两侧都裁）", flush=True)
    if len(tr_all) == 0:
        raise SystemExit("裁到 MSD 覆盖集之后没有 play 了")

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

    unit, std, msd_feat_info = build_msd_features(msd_ids, msd_X, item_ids,
                                                 args.dim, args.msd_reduce)
    args.dim = int(unit.shape[1])           # P 活在 MSD 空间里，--dim 跟随
    chart_feats = torch.as_tensor(unit, dtype=torch.float32, device=device)
    msd_std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
    print(f"[msd] P = ({len(ents):,}, {args.dim})；每图自由参数: "
          + ("0 (--msd-bm msd)" if args.msd_bm == "msd" else f"bm 一张 ({len(items):,})"),
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

    if args.tag in LOSS_RANGE:
        lo, hi = LOSS_RANGE[args.tag]
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
    pos = pd.Index(msd_ids).get_indexer(item_ids[probe])
    norms = chart_feats.norm(dim=1)
    chk.check("msd_aligned_with_item_ids",
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
        model = build_model(kind, args, device, len(ents), len(items), args.dim,
                            chart_feats, msd_std_t, drop_bias)
        opt, wd_of = sp.build_optimizer(model, args)
        mech_by_model[kind] = {"msd_main": args.msd_main, "msd_bm": args.msd_bm,
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
    msd_tag = f"_msd{args.msd_reduce}{args.dim}" \
        + ("" if not args.msd_main else "_main") \
        + ("" if args.msd_bm == "free" else "_bmmsd")
    variant = args.models.replace(",", "+") + msd_tag
    if args.features != "F0":
        variant += f"_{args.features}"
    if args.entity != "player":
        variant += "_py"
    if args.p_drift != "none":
        variant += {"logpc": "_pd", "time": "_pdt"}[args.p_drift]
    if args.drop_bias != "none":
        variant += {"u": "_nobu", "m": "_nobm", "both": "_nobubm"}[args.drop_bias]
    feat_names = ([f"msd_{s}" for s in skills] if args.msd_reduce in ("raw", "zscore")
                  else [f"msdpc{i + 1}" for i in range(args.dim)])
    entity_ids = np.array(list(inv_e.values()), dtype=np.int64)
    beatmap_ids = np.array(list(inv_i.values()), dtype=np.int32)
    npz_paths = {}
    for kind, model in fitted.items():
        path = out_dir / f"{args.out_prefix}_{args.tag}_{kind}_{args.features}{msd_tag}" \
                         f"{'' if args.entity == 'player' else '_py'}" \
                         f"{'' if args.p_drift == 'none' else ('_pd' if args.p_drift == 'logpc' else '_pdt')}" \
                         f"_{args.target}.npz"
        extra = {}
        alpha = model.np_array("alpha")
        if alpha is not None:
            extra["alpha"] = alpha
            extra["msd_std"] = std
        C_eff = model.export_C()
        np.savez(path, entity_ids=entity_ids, beatmap_ids=beatmap_ids,
                 P=model.np_array("P", np.zeros((len(ents), args.dim), np.float32)),
                 V=model.np_array("V", np.zeros((len(ents), args.dim), np.float32)),
                 C=C_eff, D=model.export_D() if "D" in model.p else np.zeros_like(C_eff),
                 b0=model.np_array("b0"),
                 bu=model.np_array("bu", np.zeros(len(ents), np.float32)),
                 bm=model.np_array("bm", np.zeros(len(items), np.float32)),
                 msd_cols=np.array(feat_names),
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
        preds_path = out_dir / f"{args.out_prefix}_{args.tag}_preds_minacalc_{slug}_{args.target}.npz"
        np.savez(preds_path, player_id=test.player_id.to_numpy()[te_ok],
                 beatmap_id=test.beatmap_id.to_numpy()[te_ok],
                 y=te_y_np, variant=variant,
                 **{f"pred_{kind}": fitted[kind].predict(
                     te_e_t, te_i_t, te_x_t, te_tau_t, te_t_t) for kind in fitted})
        print(f"wrote {preds_path.name} ({int(te_ok.sum()):,} test rows)")

    # 资产信息（path/skills/calc_version/n_rows）与特征变换信息（mean/scale/...）都要进 summary
    msd_json = jsonable({**msd_meta, **msd_feat_info})
    summary = {
        "variant": variant,
        "args": {kk: (str(v) if isinstance(v, Path) else v) for kk, v in vars(args).items()},
        "mechanism": {"msd_main": args.msd_main, "msd_bm": args.msd_bm,
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
        "msd": {**msd_json, "filtered_rows": msd_drop,
                "dim_used": int(args.dim),
                "alpha": {kind: (None if fitted[kind].np_array("alpha") is None
                                 else fitted[kind].np_array("alpha").round(5).tolist())
                          for kind in fitted}},
        "results": results,
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


def jsonable(info: dict) -> dict:
    """summary 里的 numpy 标量/数组要转成 python 原生类型，json 才写得出去。"""
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


if __name__ == "__main__":
    main()
