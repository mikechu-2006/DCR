#!/usr/bin/env python
"""Step 2 - prediction task:  (player_id, beatmap_id) -> loss.   [torch engine]

Design (legacy/docs/step2_prediction.md):

  * observation unit is the **play**; the label is step0's "loss".  No cell table, no
    per-cell medians, no aggregation of any kind.
  * **the split comes from step1.**  --train-path points at step1_{tag}, whose split
    column (--split-col) labels every row "train" / "test" / "dropped"; step2 draws no
    split of its own.
  * **train on the "train" rows** (step1-cleaned).  **"test" rows are only evaluated,
    never trained on**; "dropped" rows are ignored.  step1's two rules are a *training-side*
    treatment, so letting them also touch the test rows would condition on the treatment
    (legacy/docs/archive/cleaning_plan.md section 3.2).
  * **every metric is computed on the test rows** -- one result block per model, keyed by
    the model name.

Engine: **PyTorch**.  Forward, backward and the optimiser are autograd + torch.optim.Adam
(the reference numpy implementation -- explicit gradients + hand written Adam -- is kept at
legacy/scripts/step2_predict_numpy.py).  Seeding still uses numpy.random.default_rng(seed),
drawing the same values in the same order as that reference, so a torch run is directly
comparable with the documented numbers.  The nine assertions of legacy/docs/step2_prediction.md
section 10 run on every invocation (the npz/summary files are only written when they pass).

    L_hat = b0 + bu[u] + bm[m] + interaction
      bias    : interaction = 0
      mf_dot  : interaction = <P[u], C[m]>
      mirt    : interaction = <D[m] - P[u], C[m]>
      mirt_1  : same as mirt, but C = softplus(Z) > 0 (the sign-constrained variant)

    --features F1 adds  w1 * ln(playcount_cur)     (step1's own column, used as-is)
    --features F2 / --entity player-year additionally use
        P_eff = P[e] + (tau - 0.5) * V[e]          e = (player_id, calendar year)

    --p-drift logpc   adds a per-player, per-dimension drift rate Pc (n_ent, dim):
        P_eff = P[u] + Pc[u] * ln(playcount_cur)   x1 = ln(playcount_cur), Pc starts at 0
    --p-drift time    the same drift rate, but along REAL calendar time instead of practice
        count:  P_eff = P[u] + Pc[u] * t,  t = (timestamp - t_ref) / 365.25 days, so Pc is
        "how much this player's latent vector moves per year".  t_ref is the mean training
        timestamp and Pc starts at 0, so the model starts drift-free.

    --chart-content cm3p  replaces the free per-chart table C (a one-hot lookup) by a FROZEN
        content vector, and makes P live in that same space:
            C[m] = content[m]                       buffer, never a parameter, never in Adam
            P    : (n_ent, d) with d = --content-dim (--dim is forced to d)
        The interaction is still a plain dot product <P[u], C[m]>; the only per-chart free
        parameter left is bm.  Charts without a content row are dropped from BOTH tables
        BEFORE the split -- filtering after would silently move the test set.
    --d-constraint    constrains mirt's discrimination vector D (per chart):
        none    : D free (default; current behaviour)
        nonneg  : D = softplus(Z)                  -> D > 0, no sum constraint
        simplex : D = softmax(Z, dim=1)            -> D >= 0 and sum_i D_i = 1 exactly
      The constraint is a reparameterisation: Z are the parameters, the npz stores the
      *effective* D (plus D_raw = Z).  Adam's wd is a constant pull to Z=0, which for a
      softmax is the uniform-D symmetry sink, so a constrained run defaults to
      --d-logit-wd 0 and a spread Z init (--d-init-std).

    mirt_1 pins mirt's sign gauge.  P is player ability, D is chart difficulty and the label
    is a loss (smaller = better): the interaction must FALL as P rises and RISE as D rises,
    i.e. both coefficients need C >= 0.  Free mirt is invariant under (P, D, C) -> (-P, -D,
    -C) per latent dimension, so its C has an arbitrary sign.  mirt_1 removes that freedom
    by reparameterising C = softplus(Z) (exact, no clipping/projection): Z are the parameters,
    C is the effective per-chart table, the npz stores C plus C_raw = Z.  Z=0 maps to the
    constant C = softplus(0) = 0.693, so Adam's wd would erase cross-chart differences in C:
    the constrained C defaults to --c-logit-wd 0.  Adam's wd cannot regularise C itself
    (the parameter is Z), so mirt_1 adds the same L2 to the EFFECTIVE C (coefficient --wd):
    without it the exact scale gauge P,D -> lP,lD, C -> C/l lets P shrink and C grow for
    free, which overfits badly.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

PROC = Path("data/processed")
BASE_COLS = ["player_id", "beatmap_id"]
# Expected loss range of the *uncleaned* test table (step0), per tag -- checklist item 4.
# Values are step0's observed extremes; a tag without an entry is only checked for finiteness.
LOSS_RANGE = {"1k": (-12.4663, -0.2669), "10k": (-12.4663, -0.1358)}


# ----------------------------------------------------------------------------- io
def table_has_column(path: Path, col: str) -> bool:
    """表里有没有这一列（parquet 看 schema，csv 看表头）；读不出来就当有，让真正的读报错。"""
    try:
        if Path(path).suffix == ".parquet":
            import pyarrow.parquet as pq_
            return col in set(pq_.ParquetFile(path).schema.names)
        return col in set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return True


def load_play_table(tag: str, which: str, cols: list, path: Path | None = None,
                    proc_dir: Path = PROC, optional_cols: tuple = ()) -> pd.DataFrame:
    """which in {'step0','step1'} -> that play table (.parquet preferred).

    path 非空时直接读该文件（--train-path / --test-path）；否则在 proc_dir
    里按 {which}_{tag}.{parquet,csv} 找。optional_cols 里的列缺失时不报错（用于
    兼容没有 split 列的旧 step1 表）。
    """
    if path is not None:
        src = Path(path)
        if not src.exists():
            raise SystemExit(f"--{which}-path not found: {src}")
        want = list(cols) + [c for c in optional_cols if table_has_column(src, c)]
        df = (pd.read_parquet(src, columns=want) if src.suffix == ".parquet"
              else pd.read_csv(src, usecols=want))
    else:
        pq, csv = proc_dir / f"{which}_{tag}.parquet", proc_dir / f"{which}_{tag}.csv"
        src = pq if pq.exists() else (csv if csv.exists() else None)
        if src is None:
            raise SystemExit(f"neither {pq} nor {csv} exists")
        want = list(cols) + [c for c in optional_cols if table_has_column(src, c)]
        df = (pd.read_parquet(src, columns=want) if src.suffix == ".parquet"
              else pd.read_csv(src, usecols=want))
    print(f"[{which}] {src.name}: {len(df):,} rows, {df.player_id.nunique():,} players, "
          f"{df.beatmap_id.nunique():,} beatmaps", flush=True)
    return df


def read_split_spec(train_path: Path | None, proc_dir: Path, tag: str) -> dict | None:
    """读 step1 表旁边的报告 json 里的 split 定义（method/train_frac/seed/unit/实际比例）。

    找不到就当没有：调用方会退回 --test-pair-frac 做容差检查，只影响一处自检的基准。
    """
    base = Path(train_path) if train_path else proc_dir / f"step1_{tag}"
    for cand in (base.with_suffix(".json"), proc_dir / f"step1_{tag}.json"):
        try:
            if cand.exists():
                data = json.loads(cand.read_text(encoding="utf-8"))
                spec = data.get("split") if isinstance(data, dict) else None
                if isinstance(spec, dict):
                    return spec
        except (OSError, json.JSONDecodeError):
            continue
    return None


def pair_keys(player, beatmap, k: int) -> np.ndarray:
    """Ordered-pair key, int64.  k = beatmap_id.max()+1 makes the encoding injective."""
    return player.astype(np.int64) * np.int64(k) + beatmap.astype(np.int64)


def map_ids(values: np.ndarray, vocab_ids: np.ndarray) -> np.ndarray:
    """Index into a sorted unique vocab; -1 for ids the training rows never showed.

    Same semantics as pandas Series.map(dict).fillna(-1) in the reference implementation,
    without building a python dict over millions of keys.
    """
    v = values.astype(np.int64, copy=False)
    if vocab_ids.size == 0:
        return np.full(v.shape, -1, dtype=np.int64)
    pos = np.searchsorted(vocab_ids, v)
    pos[pos >= vocab_ids.size] = 0                      # clamp before the membership test
    ok = vocab_ids[pos] == v
    return np.where(ok, pos, np.int64(-1)).astype(np.int64)


# ------------------------------------------------------------------------- metrics
def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    return r


def spearman(pred: np.ndarray, y: np.ndarray) -> float:
    a, b = _ranks(pred), _ranks(y)
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def report(pred: np.ndarray, y: np.ndarray, n_total: int) -> dict:
    e = pred - y
    # loss -> ACC percentage points:  |exp(loss_pred) - exp(loss_true)| * 100.
    # A label shift s changes loss to log(s + exp(loss_step0)), so exp(label) is
    # ACC + s; the shift cancels in the difference and pp is still |dACC| * 100.
    dpp = np.abs(np.exp(pred.astype(np.float64)) - np.exp(y.astype(np.float64))) * 100.0
    return {
        "n_scored": int(len(y)),
        "coverage": round(len(y) / n_total, 6) if n_total else None,
        "rmse": round(float(np.sqrt((e ** 2).mean())), 6),
        "mae": round(float(np.abs(e).mean()), 6),
        "spearman": round(spearman(pred, y), 6),
        "pp_p50": round(float(np.percentile(dpp, 50)), 6),
        "pp_p90": round(float(np.percentile(dpp, 90)), 6),
        "pp_mean": round(float(dpp.mean()), 6),
        "pp_gt1_frac": round(float((dpp > 1).mean()), 6),
        "pp_gt5_frac": round(float((dpp > 5).mean()), 6),
    }


# --------------------------------------------------------------------------- model
class EmbeddingModel(nn.Module):
    """b0 + bu[u] + bm[m] + interaction, trained with autograd + torch.optim.Adam.

    Parameter set and initialisation mirror the numpy reference exactly: b0/bu/bm start at
    zero, P/C (and D for mirt) are drawn from default_rng(seed).normal(0, 0.05) in that
    order, V/w1 start at zero, and only the parameters a model actually uses are handed to
    the optimiser (bias never touches P/C/D).

    Kinds with a D vector: "mirt" (linear interaction <D-P, C>), "mirt_1" (the same
    interaction with the sign constraint C = softplus(Z) > 0) and "mirt_exp"
    (exponent/"ACC-space" interaction  ln( sum_i exp( (D_mi - P_ui) * c_mi ) ), c_m = softmax(C_m)).
    The logsumexp form is positive for any real D, so mirt_exp does NOT force D > 0 any more:
    D is free by default (pass --d-constraint nonneg to restore the softplus).

    mirt_1 exists because free mirt's C has an arbitrary sign: the whole model is invariant
    under (P, D, C) -> (-P, -D, -C) per dimension.  The label is a loss and P is ability, so
    the coefficient C on -P must be >= 0 (higher ability -> lower loss, higher difficulty D
    -> higher loss).  The constraint is exact (softplus reparameterisation), not a penalty.
    """

    def __init__(self, kind, n_ent, n_item, dim, features, seed=0, device=torch.device("cpu"),
                 p_drift="none", d_constraint="none", d_init_std=None, drop_bias="none",
                 content="none", chart_feats=None, c_init_std=None, c_constraint="auto"):
        super().__init__()
        if content != "none" and chart_feats is None:
            raise ValueError("content != 'none' needs chart_feats (n_item, dim)")
        if content != "none" and kind == "bias":
            raise ValueError("content has no effect on the bias model")
        if content != "none" and kind == "mirt_exp":
            raise ValueError(
                "mirt_exp 用 softmax(C) 当占比，但 --chart-content cm3p 时 C 是冻结的内容向量"
                "（单位范数）：softmax 之后接近均匀、而且不可训练，占比没有意义。"
                "请用 --chart-content none，或把占比改成独立的 per-chart 参数。")
        if content != "none" and kind == "mirt_1":
            raise ValueError(
                "mirt_1 约束的是自由 C 表（C = softplus(Z)）；--chart-content 时 C 是冻结的"
                "内容向量（可正可负），没有可优化的 logits。请用 --chart-content none。")
        if c_constraint == "auto":
            # mirt_1 的定义就是"mirt + C>=0"；其他 model 不启用（它们没有 -P 的方向问题）
            c_constraint = "nonneg" if kind == "mirt_1" else "none"
        if c_constraint not in ("none", "nonneg"):
            raise ValueError(f"unknown c_constraint {c_constraint!r}")
        if c_constraint != "none" and kind != "mirt_1":
            raise ValueError("c_constraint != 'none' is only implemented for mirt_1")
        self.kind, self.dim, self.features, self.content = kind, dim, features, content
        # 2026-10: mirt_exp 的交互已换成 logsumexp(...) 形式，exp 恒正、不再需要 D > 0，
        # 所以这里不再自动提升为 nonneg —— D 默认自由。要恢复旧行为请显式 --d-constraint nonneg。
        self.d_constraint = d_constraint
        self.c_constraint = c_constraint
        self.p_drift, self.d_init_std = p_drift, d_init_std
        self.n_ent, self.n_item = n_ent, n_item
        rng = np.random.default_rng(seed)

        def draw(shape, std=0.05):
            a = np.ascontiguousarray(rng.normal(0, std, shape).astype(np.float32))
            return torch.as_tensor(a, dtype=torch.float32, device=device)

        self.drop_bias = drop_bias
        self.p = {"b0": nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device))}
        if drop_bias not in ("u", "both"):      # b_u: per-entity main effect
            self.p["bu"] = nn.Parameter(torch.zeros(n_ent, dtype=torch.float32, device=device))
        if drop_bias not in ("m", "both"):      # b_m: per-item main effect
            self.p["bm"] = nn.Parameter(torch.zeros(n_item, dtype=torch.float32, device=device))
        self.p["P"] = nn.Parameter(draw((n_ent, dim)))
        if content == "none":
            # 若 C 走 softmax（mirt_exp），std=0.05 会让 softmax 几乎均匀，而均匀点正是
            # 对称汇（Jacobian 只有 ~1/dim），C 会原地不动 —— 所以默认给 0.5 的展开初始化。
            c_std = c_init_std if c_init_std is not None else (0.5 if kind == "mirt_exp" else 0.05)
            C0 = draw((n_item, dim), c_std)
            if self.c_constraint == "nonneg":
                # C = softplus(Z)：Z 从 softplus_inv(0.05)（+ 小噪声）出发，生效 C 约 0.05，
                # 与自由 mirt 的 C init 同尺度；C 恒正，梯度经 sigmoid(Z) 回传。
                with torch.no_grad():
                    C0.add_(float(np.log(np.expm1(0.05))))
            self.p["C"] = nn.Parameter(C0)
        else:
            # Frozen chart content.  A buffer: it never reaches Adam and never gets a
            # weight-decay pull, so the GL(K) gauge that made a *prior* on C meaningless
            # (diag_step2_gauge.py) has nothing to act on -- C is a constant here.
            self.register_buffer("C_feat", chart_feats.to(device=device, dtype=torch.float32))
        if kind in ("mirt", "mirt_1", "mirt_exp"):
            # simplex starts from a *spread* softmax (all charts identical at Z=0 would be a
            # symmetry sink: the softmax Jacobian is only ~1/dim there, so D never differentiates)
            std = d_init_std if d_init_std is not None else (0.5 if d_constraint == "simplex" else 0.05)
            self.p["D"] = nn.Parameter(draw((n_item, dim), std))
            if d_constraint == "nonneg":
                # start from D ~ 0.05 like the free model: Z = softplus_inv(0.05) + N(0, 0.05)
                with torch.no_grad():
                    self.p["D"].add_(float(np.log(np.expm1(0.05))))
        if p_drift in ("logpc", "time") and kind != "bias":
            # drift rate, same shape as P; starts at 0 so the model starts drift-free.
            # "logpc" drifts along ln(playcount_cur), "time" along calendar years.
            self.p["Pc"] = nn.Parameter(torch.zeros((n_ent, dim), dtype=torch.float32, device=device))
        if features == "F2":
            self.p["V"] = nn.Parameter(torch.zeros((n_ent, dim), dtype=torch.float32, device=device))
        if features in ("F1", "F2"):
            self.p["w1"] = nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device))

        # only the parameters this model actually uses (the rest are never handed to Adam)
        self.active = [k for k in ("b0", "bu", "bm") if k in self.p]
        if kind != "bias":
            self.active += ["P"]
            if content == "none":
                self.active += ["C"]        # with content there is no free chart table
        if kind in ("mirt", "mirt_1", "mirt_exp"):
            self.active += ["D"]
        if p_drift in ("logpc", "time") and kind != "bias":
            self.active += ["Pc"]
        if features == "F2":
            self.active += ["V"]
        if features in ("F1", "F2"):
            self.active += ["w1"]

        # 哪些参数不能吃 weight decay。Adam 的 wd 是"每步把参数往 0 拉 wd*param"的恒定力：
        # 对普通参数是正常正则，但对下面这些参数，0 恰好是退化点，这股利会把这个方向按死。
        #   C 的 logits -> 0: softmax 变成均匀占比（该点 Jacobian 只有 ~1/dim）；
        #   C 的 logits -> 0（softplus，mirt_1）: C 变成常数 softplus(0)=0.693，跨图差异被抹平；
        #   D          -> 0: 每张谱面难度都变成 e^0 = 1，难度信息被抹平。
        # 默认 wd=0，可用 --d-logit-wd / --c-logit-wd 覆盖。mirt 的线性 D 不在此列。
        self.c_is_logit = (kind == "mirt_exp" and content == "none")   # 公式里 C 过 softmax
        # mirt_exp 的交互只通过 (D - P) 依赖两者，所以 D->D+a, P->P+a 是精确的平坦方向：
        # 梯度与它正交，但 Adam 的逐参数归一化会让更新带出分量，两者的水平就无约束漂移。
        self.gauge_center_p = (kind == "mirt_exp")
        self.no_decay = set()
        if kind == "mirt_exp":
            if content == "none":
                self.no_decay.add("C")     # softmax logits：wd 拉向 0 = 占比均匀（对称汇）
            # D 与 P 是同一个差 (D-P) 的两半。给它们**不对称**的 wd 会让优化器把公共水平
            # 整个倒进没 wd 的那一半：实测 wd 只加在 P 上时，D 的均值漂到 -18.5、P 的
            # sd 只剩 0.055（差不变，但两个参数各自完全不可解释）。所以两者同处理 ——
            # 都不吃 wd（能力/难度的尺度不该被 L2 拉平），公共原点交给 gauge_fix()。
            self.no_decay.update(("D", "P"))
        elif d_constraint != "none":
            self.no_decay.add("D")
        if self.c_constraint != "none":
            # C = softplus(Z)：wd 拉向 Z=0，即所有谱面同一个 C=0.693 —— 常数无信息
            self.no_decay.add("C")

    @torch.no_grad()
    def gauge_fix(self):
        """钉死 D 与 P 的公共原点（mirt_exp 专用）。

        交互是 sc*(D - P)，只有差有意义 —— D->D+a, P->P+a 逐位不改变输出。这是纯规范
        自由度，但 Adam 是逐参数归一化的、更新并不正交于它，于是 D 和 P 的水平会无约束
        地漂开（实测 D sd 3.75 / P sd 0.19）。这里取 IRT 的常规约定：P 在每个潜维上对
        实体中心化，位移量同时从 D 减掉。模型输出逐位不变，只是让两者落在同一把尺子上。
        """
        if not getattr(self, "gauge_center_p", False) or "P" not in self.p or "D" not in self.p:
            return
        shift = self.p["P"].mean(dim=0, keepdim=True)
        self.p["P"].sub_(shift)
        self.p["D"].sub_(shift)

    def D_matrix(self, m):
        """Effective D for item ids m: the raw parameter, or its constrained transform."""
        Z = self.p["D"][m]
        if self.d_constraint == "simplex":
            return torch.softmax(Z, dim=1)          # D >= 0, sum_i D_i = 1
        if self.d_constraint == "nonneg":
            return F.softplus(Z)                    # D > 0
        return Z

    def C_matrix(self, m):
        """Effective C for item ids m: the raw table, or softplus(Z) >= 0 for mirt_1."""
        if self.content != "none":
            return self.C_feat[m]                   # frozen, never constrained
        Z = self.p["C"][m]
        if self.c_constraint == "nonneg":
            return F.softplus(Z)                    # C > 0 (exact reparameterisation)
        return Z

    def c_penalty(self):
        """0.5 * ||C_eff||^2 for mirt_1 (None when C is unconstrained).

        Adam's weight_decay acts on the *parameter*, but mirt_1's parameter is Z with
        C = softplus(Z).  Putting wd on Z would pull C to the constant softplus(0)=0.693
        (not to 0) and would leave the exact scale gauge (P,D,C) -> (lP,lD,C/l)
        unregularised: with P,D penalised and C not, the optimiser shrinks P and grows C
        while the loss is unchanged (observed: C sd 0.59 vs mirt's 0.12, and much worse
        test RMSE).  So the L2 is applied to the effective C, exactly like free mirt's
        wd on its C parameter.
        """
        if self.c_constraint == "none":
            return None
        C = F.softplus(self.p["C"])
        return 0.5 * (C * C).sum()

    def forward(self, u, m, x1=None, tau=None, t=None):
        p = self.p
        P = p["P"][u]
        if self.p_drift == "logpc":
            P = P + p["Pc"][u] * x1.unsqueeze(1)    # per-player drift along ln(playcount_cur)
        elif self.p_drift == "time":
            P = P + p["Pc"][u] * t.unsqueeze(1)     # per-player drift along calendar years
        if self.features == "F2":
            P = P + (tau - 0.5).unsqueeze(1) * p["V"][u]
        # materialise a (B,) vector: with every intercept dropped the model can be a bare scalar
        out = p["b0"][0] + torch.zeros(u.shape[0], dtype=torch.float32, device=u.device)
        if "bu" in p:
            out = out + p["bu"][u]
        if "bm" in p:
            out = out + p["bm"][m]
        if self.features in ("F1", "F2"):
            out = out + p["w1"][0] * x1
        if self.kind == "bias":
            return out
        C = self.C_matrix(m)
        if self.kind == "mf_dot":
            return out + (P * C).sum(1)
        D = self.D_matrix(m)
        if self.kind == "mirt_exp":
            return out + torch.logsumexp((D-P)*torch.softmax(C,dim=1), dim=1)
        if self.kind == "mirt":
            return out + ((D - P) * C).sum(1)
        if self.kind == "mirt_1":
            return out + ((D - P) * C).sum(1)
        return out + ((D - P) * C).sum(1)

    @torch.no_grad()
    def predict(self, u, m, x1=None, tau=None, t=None) -> np.ndarray:
        return self.forward(u, m, x1, tau, t).detach().to("cpu").numpy()

    def np_array(self, name, fallback=None):
        """Parameter as a numpy array; fallback (e.g. zeros_like) when the model has none."""
        if name in self.p:
            return self.p[name].detach().to("cpu").numpy()
        return fallback

    def export_C(self) -> np.ndarray:
        """Effective C (after the transform the formula applies) for the whole chart vocabulary.

        C 当 softmax 的 logits 时（mirt_exp），原始值只有"行内差异"有意义 —— softmax 对
        每行加常数不变，所以原始 C 的绝对数值不是任何东西。这里导出的是生效值 softmax(C)
        （即占比），原始 logits 另存为 C_raw，与 D / D_raw 的既有约定一致。
        mirt_1 同理：导出 softplus(Z)（满足 C>=0），原始 Z 存为 C_raw。
        """
        if self.content != "none":
            return self.C_feat.detach().to("cpu").numpy()
        Z = self.p["C"].detach()
        if self.c_is_logit:
            Z = torch.softmax(Z, dim=1)
        elif self.c_constraint == "nonneg":
            Z = F.softplus(Z)
        return Z.to("cpu").numpy()

    def chart_matrix(self) -> np.ndarray:       # 兼容旧调用点
        return self.export_C()

    def export_D(self) -> np.ndarray:
        """Effective D (after the constraint transform) for the whole chart vocabulary."""
        Z = self.p["D"].detach()
        if self.d_constraint == "simplex":
            Z = torch.softmax(Z, dim=1)
        elif self.d_constraint == "nonneg":
            Z = F.softplus(Z)
        return Z.to("cpu").numpy()


# ------------------------------------------------------------------------ checklist
class Checklist:
    """legacy/docs/step2_prediction.md section 10 -- run on every invocation."""

    def __init__(self):
        self.items = []

    def check(self, name, ok, detail=""):
        ok = bool(ok)
        self.items.append({"name": name, "ok": ok, "detail": detail})
        print(f"  [check] {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
        return ok

    def require(self):
        bad = [c["name"] for c in self.items if not c["ok"]]
        if bad:
            raise AssertionError(f"validation checklist failed: {', '.join(bad)}")


def determinism_selftest(device) -> bool:
    """Checklist item 8: the training path is fully seeded -- same seed -> bitwise equal."""

    def run_once():
        m = EmbeddingModel("mirt", 24, 17, 4, "F2", seed=0, device=device,
                           p_drift="logpc", d_constraint="simplex")
        opt = torch.optim.Adam([m.p[k] for k in m.active], lr=0.02, weight_decay=1e-4)
        r = np.random.default_rng(7)
        u = torch.as_tensor(r.integers(0, 24, 96), dtype=torch.long, device=device)
        i = torch.as_tensor(r.integers(0, 17, 96), dtype=torch.long, device=device)
        y = torch.as_tensor(r.normal(-3, 1, 96).astype(np.float32), device=device)
        x1 = torch.as_tensor(r.normal(1, 0.5, 96).astype(np.float32), device=device)
        tau = torch.as_tensor(r.random(96).astype(np.float32), device=device)
        rng = np.random.default_rng(3)
        for _ in range(3):
            for s in range(0, 96, 32):
                b = torch.as_tensor(rng.permutation(96)[s:s + 32], dtype=torch.long, device=device)
                opt.zero_grad(set_to_none=True)
                ((m.forward(u[b], i[b], x1[b], tau[b]) - y[b]) ** 2).mean().backward()
                opt.step()
        return {k: m.p[k].detach().clone() for k in m.p}

    a, b = run_once(), run_once()
    return all(torch.equal(a[k], b[k]) for k in a)


# ---------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1k")
    ap.add_argument("--target", default="loss")
    ap.add_argument("--label-shift", type=float, default=0.0,
                    help="step1 的 label shift s（仅用于 loss_in_range 自检与产物命名，"
                         "不改变训练）；标签为 log(s + exp(step0_loss)) = log(1+s-ACC)")
    ap.add_argument("--models", default="bias,mf_dot,mirt")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--features", default="F0", choices=["F0", "F1", "F2"])
    ap.add_argument("--entity", default="player", choices=["player", "player-year"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="step2")
    ap.add_argument("--proc-dir", type=Path, default=PROC,
                    help="输入表的目录（默认 data/processed）；--train-path 优先")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="产物目录（npz + summary json）；默认与 --proc-dir 相同")
    ap.add_argument("--train-path", type=Path, default=None,
                    help="step1 的输出表（train/test/dropped 都在它的 split 列里）；"
                         "默认 {proc_dir}/step1_{tag}.{parquet,csv}")
    ap.add_argument("--split-col", default="split",
                    help="step1 表里的划分标签列（train / test / dropped）")
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--p-drift", default="none", choices=["none", "logpc", "time"],
                    help="player-side drift: 'logpc' -> P_eff = P[u] + Pc[u]*ln(playcount_cur); "
                         "'time' -> P_eff = P[u] + Pc[u]*t with t in calendar years since the mean "
                         "training timestamp.  Pc starts at 0, so the model starts drift-free.")
    ap.add_argument("--chart-content", default="none", choices=["none", "cm3p"],
                    help="replace the free per-chart table C by a FROZEN content vector")
    ap.add_argument("--restrict-to-content", default="off", choices=["off", "on"],
                    help="filter both play tables to the charts that HAVE a content row, but do "
                         "not use the vectors.  This is how the 1-hot baseline is run on exactly "
                         "the same chart subset, so the two are comparable.")
    ap.add_argument("--content-path", default=str(PROC / "chart_content_cm3p.parquet"))
    ap.add_argument("--content-dim", type=int, default=64,
                    help="dimension of the frozen content vector -- and therefore of P")
    ap.add_argument("--content-reduce", default="pca", choices=["pca", "raw"],
                    help="pca: whiten to --content-dim on TRAIN charts only; raw: use the "
                         "table's own dimension (then --content-dim must equal it)")
    ap.add_argument("--d-constraint", default="none", choices=["none", "nonneg", "simplex"],
                    help="mirt's D: none (free) / nonneg (softplus, D>0) / "
                         "simplex (softmax, D>=0 and sum_i D_i = 1)")
    ap.add_argument("--d-logit-wd", type=float, default=None,
                    help="weight decay for D.  Default 0 whenever D's zero point is degenerate: "
                         "a constrained D (softmax/softplus logits) or mirt_exp's free D (the "
                         "difficulty scale -- wd drags every chart's difficulty to e^0 = 1).  "
                         "mirt's linear D keeps --wd")
    ap.add_argument("--c-logit-wd", type=float, default=None,
                    help="weight decay for C's raw logits when C is transformed: softmax(C) shares "
                         "(mirt_exp) or softplus(C)>=0 (mirt_1).  Default 0: Adam's wd is a "
                         "constant pull to Z=0, i.e. uniform shares / a constant C, which is "
                         "exactly the symmetry sink that stops C learning")
    ap.add_argument("--c-init-std", type=float, default=None,
                    help="std of C's init (softmax logits for mirt_exp, raw softplus logits for "
                         "mirt_1).  Default: 0.5 for mirt_exp (shares start spread out) vs 0.05 "
                         "for a plain C table / mirt_1")
    ap.add_argument("--drop-bias", default="none", choices=["none", "u", "m", "both"],
                    help="drop the main-effect intercepts: u = no b_u (per player), m = no b_m "
                         "(per chart), both = keep only the global b0")
    ap.add_argument("--d-init-std", type=float, default=None,
                    help="std of the raw D init; default: 0.05 (free/nonneg), 0.5 (simplex, to break "
                         "the uniform-D symmetry sink)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="torch device; 'auto' picks cuda when available, else cpu")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch.set_num_threads(); 0 = leave the default.  Reproducibility "
                         "across machines needs the same thread count (parallel reductions)")
    ap.add_argument("--save-preds", action="store_true",
                    help="also write {out_dir}/{prefix}_{tag}_preds_{target}.npz with the test "
                         "ids, the labels and every model's predictions, so paired bootstrap CIs "
                         "(scripts/compare_preds.py) need no re-run.  Off by default: it turns an "
                         "otherwise deterministic artifact set into one that depends on which "
                         "models were in --models")
    return ap.parse_args(argv)


# ------------------------------------------------------------------- chart content
def load_content_table(path) -> tuple:
    """(ids int64[n], E float32[n, d_raw]) from pipeline A's parquet."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{p} not found -- run scripts/step0c_chart_content.py first")
    df = pd.read_parquet(p, columns=["beatmap_id", "embedding"])
    ids = df["beatmap_id"].to_numpy(np.int64)
    E = np.stack([np.asarray(e, dtype=np.float32) for e in df["embedding"].to_numpy()])
    print(f"[content] {p.name}: {len(ids):,} charts, raw dim {E.shape[1]}", flush=True)
    return ids, E


def align_content(ids, E, item_ids, dim, reduce_mode):
    """Frozen content aligned to the training item order, then reduced.

    Alignment is the one step in this pipeline that fails silently: C[i] must belong to
    item_ids[i], so a missing chart is a hard error here rather than a wrong number later.
    The PCA basis is fitted on the TRAINING charts only (these rows), never on the table.
    """
    pos = pd.Index(ids).get_indexer(item_ids)
    if (pos < 0).any():
        raise SystemExit(f"{int((pos < 0).sum())} training charts have no content row; "
                         f"the play tables must be filtered to covered charts first")
    X = np.ascontiguousarray(E[pos], dtype=np.float32)          # (n_item, d_raw)
    if reduce_mode == "raw":
        if dim != X.shape[1]:
            raise SystemExit(f"--content-reduce raw needs --content-dim {X.shape[1]}, got {dim}")
        Z, info = X, {"reduce": "raw"}
    else:
        mu = X.mean(0)
        Xc = X - mu
        Vt = np.linalg.svd(Xc, full_matrices=False)[2]
        d = min(dim, Vt.shape[0])
        if d != dim:
            print(f"[content] only {d} components available, --content-dim {dim} truncated", flush=True)
        Z = Xc @ Vt[:d].T
        Z = Z / (Z.std(0) + 1e-12)                              # whiten
        info = {"reduce": "pca", "pca_mean": mu, "pca_components": Vt[:d],
                "pca_scale": Z.std(0), "fitted_on": "train_charts"}
    nrm = np.linalg.norm(Z, axis=1, keepdims=True)
    Z = (Z / np.maximum(nrm, 1e-12)).astype(np.float32)         # unit norm, like raw CM3P
    ev = float(np.var(Z, axis=0).sum())
    print(f"[content] {Z.shape[0]:,} x {Z.shape[1]}  mode={reduce_mode}  "
          f"row-norm=1  var-sum={ev:.3f}", flush=True)
    return Z, info


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    return torch.device(name)


def degeneracy_report(model) -> dict:
    """训练后体检：no_decay 参数有没有塌到退化点（0 是对称点 / 尺度退化点）。

    只报告不中断 —— 塌了打印 [warn] 并写进 summary 的 "degeneracy" 字段。
    """
    rep = {}
    if "D" in model.active:
        D = model.p["D"].detach()
        rep["D_sd_across_charts"] = round(float(D.std(0).mean() if D.dim() > 1 else D.std()), 6)
    if model.c_is_logit:
        with torch.no_grad():
            c = torch.softmax(model.p["C"].detach(), dim=1)
            d = c.shape[1]
            rep["C_norm_entropy"] = round(
                float((-(c * torch.log(c.clamp_min(1e-12))).sum(1)).mean() / np.log(d)), 6)
            rep["C_effective_components"] = round(
                float(np.exp(rep["C_norm_entropy"] * np.log(d))), 4)
    if model.content == "none" and "C" in model.p:
        # 生效 C（mirt_1 已是 softplus 之后的值）：符号是 mirt / mirt_1 对比的核心
        C = model.export_C()
        rep["C_neg_count"] = int((C < 0).sum())
        rep["C_numel"] = int(C.size)
        rep["C_neg_frac"] = round(float((C < 0).mean()), 6)
        rep["charts_with_neg_C"] = int((C < 0).any(axis=1).sum())
        rep["charts_total"] = int(C.shape[0])
        rep["C_min"] = round(float(C.min()), 6)
        rep["C_max"] = round(float(C.max()), 6)
    return rep


def warn_degenerate(kind, rep):
    bad = []
    if rep.get("D_sd_across_charts", 1.0) < 0.1:
        bad.append(f"D 跨谱面 sd={rep['D_sd_across_charts']:.4f}（难度趋同 = 全部退回 e^D=1）")
    if rep.get("C_norm_entropy", 0.0) > 0.95:
        bad.append(f"C 归一化熵={rep['C_norm_entropy']:.4f}（占比接近均匀 = C 基本没学）")
    if bad:
        print(f"  [warn] {kind}: 参数退化 —— " + "; ".join(bad)
              + "  → 检查该参数的 wd 是否为 0、初始化宽度是否够大", flush=True)


def build_optimizer(model, args):
    """Adam + 分组 weight decay；model.no_decay 里的参数默认 wd=0（见 EmbeddingModel.no_decay）。

    所有参数 wd 相同时走单组路径，与旧行为逐位一致；否则按 wd 的值分组。
    """
    wd_of = {}
    for k in model.active:
        if k in model.no_decay:
            ov = {"D": args.d_logit_wd, "C": args.c_logit_wd}.get(k)
            wd_of[k] = 0.0 if ov is None else float(ov)
        else:
            wd_of[k] = float(args.wd)
    if len(set(wd_of.values())) == 1:
        return torch.optim.Adam([model.p[k] for k in model.active],
                                lr=args.lr, weight_decay=args.wd), wd_of
    by_wd = {}
    for k in model.active:
        by_wd.setdefault(wd_of[k], []).append(model.p[k])
    return torch.optim.Adam([{"params": ps, "weight_decay": w} for w, ps in by_wd.items()],
                            lr=args.lr), wd_of


def main():
    args = parse_args()
    t_start = time.time()
    out_dir = Path(args.out_dir) if args.out_dir else args.proc_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    print(f"engine: torch {torch.__version__} on {device}, threads={torch.get_num_threads()}, "
          f"deterministic_algorithms={torch.are_deterministic_algorithms_enabled()}", flush=True)
    models_wanted = [s for s in args.models.split(",") if s]
    if (args.p_drift != "none" or args.d_constraint != "none"
            or "mirt_exp" in models_wanted or "mirt_1" in models_wanted):
        print(f"mechanism: p_drift={args.p_drift} (P_eff = P + Pc*ln(playcount_cur))  "
              f"d_constraint={args.d_constraint}  d_logit_wd={args.d_logit_wd}", flush=True)
    if "mirt_1" in models_wanted:
        print("note: mirt_1 = mirt with C = softplus(Z) >= 0 (exact reparameterisation).  The "
              "interaction is <D-P, C>; with C >= 0 higher ability P lowers the loss and higher "
              "difficulty D raises it.  C's wd defaults to 0 (--c-logit-wd), C_raw = Z is saved.",
              flush=True)
    if "mirt_exp" in models_wanted:
        print("note: mirt_exp = b0 + logsumexp_i( P_ui*softmax(C_m)_i + D_mi ).  logsumexp is "
              "positive for any D, so D is FREE by default (no softplus); the old "
              "ln(sum_i D_mi*exp(-(P_ui-C_mi))) form is what needed D > 0", flush=True)

    need_pc = args.features in ("F1", "F2") or args.p_drift == "logpc"
    need_ts = args.features == "F2" or args.entity == "player-year" or args.p_drift == "time"
    cols = BASE_COLS + [args.target] \
        + (["playcount_cur"] if need_pc else []) + (["timestamp"] if need_ts else [])

    # 只读 step1 的表：train/test/dropped 全在它的 split 列里（划分在 step1 完成）。
    tr_all = load_play_table(args.tag, "step1", cols, path=args.train_path,
                             proc_dir=args.proc_dir, optional_cols=(args.split_col,))

    # ------------------------------------------------- content availability filter
    # BEFORE the split, always.  Dropping charts after the split would change which pairs
    # are held out, so the run would no longer be comparable with the 1-hot baseline.
    content_ids = content_E = None
    content_drop = {"rows_dropped": 0, "rows_total": None}
    if args.chart_content != "none" or args.restrict_to_content == "on":
        content_ids, content_E = load_content_table(args.content_path)
        have = set(content_ids.tolist())
        n_all = len(tr_all)
        tr_all = tr_all[tr_all.beatmap_id.isin(have)].reset_index(drop=True)
        content_drop = {"rows_dropped": n_all - len(tr_all), "rows_total": n_all}
        print(f"[content] charts with a vector: {len(have):,} of {args.tag} whitelist; dropped "
              f"{n_all - len(tr_all):,}/{n_all:,} rows (train 与 test 两侧都裁)", flush=True)
        if len(tr_all) == 0:
            raise SystemExit("no plays left after the content filter")

    # ------------------------------------------------------------------- split
    # 划分在 step1 完成：这里只按 split 列取 train / test，dropped 忽略。
    if args.split_col not in tr_all.columns:
        raise SystemExit(
            f"step1 表里没有 '{args.split_col}' 列 —— train/test 划分应在 step1 完成。\n"
            f"  请重跑 step1（新版本会写这一列；若用 --resume 跳过了 step1，加 --force 强制重跑）")
    lab = tr_all[args.split_col].astype(str).to_numpy()
    unknown = sorted(set(lab.tolist()) - {"train", "test", "dropped"})
    if unknown:
        raise SystemExit(f"'{args.split_col}' 列出现未知标签 {unknown}"
                         f"（只允许 train/test/dropped）")
    tr_mask, te_mask = lab == "train", lab == "test"
    train = tr_all[tr_mask].reset_index(drop=True)     # 训练：step1 清洗过
    test = tr_all[te_mask].reset_index(drop=True)      # 评估：step1 未清洗（原始标签）
    n_dropped = int((lab == "dropped").sum())
    if len(train) == 0 or len(test) == 0:
        raise SystemExit(f"step1 表里 train={len(train):,} / test={len(test):,} 行，"
                         f"无法训练或评估")

    # 划分单位（pair 或 beatmap）；dropped 行也算 train 侧 —— 它们只是被清洗删掉
    split_spec = read_split_spec(args.train_path, args.proc_dir, args.tag)
    k = int(max(int(train.beatmap_id.max()), int(test.beatmap_id.max()))) + 1
    row_keys = pair_keys(tr_all.player_id.to_numpy(), tr_all.beatmap_id.to_numpy(), k)
    unit_kind = (split_spec or {}).get("unit", "pair")
    if unit_kind == "beatmap":
        ids = tr_all.beatmap_id.to_numpy()
        unit_tr, unit_te = np.unique(ids[lab != "test"]), np.unique(ids[te_mask])
    else:
        unit_tr = np.unique(row_keys[lab != "test"])
        unit_te = np.unique(row_keys[te_mask])
    units_all = np.union1d(unit_tr, unit_te)
    is_test = np.isin(units_all, unit_te)              # 仅用于统计 test 单位占比
    n_pairs, n_test_pairs = int(units_all.size), int(unit_te.size)
    cross_units = np.intersect1d(unit_tr, unit_te, assume_unique=True)
    test_pairs = np.unique(row_keys[te_mask])
    tr_pairs = np.unique(row_keys[tr_mask])
    expected_test_frac = float((split_spec or {}).get("actual_test_frac")
                               or (n_test_pairs / max(n_pairs, 1)))
    frac_tol = 0.02
    print(f"[split] step1 表的 '{args.split_col}' 列：train {len(train):,} 行 / "
          f"test {len(test):,} 行 / dropped {n_dropped:,} 行 | "
          f"单位 {unit_kind}: train 侧 {unit_tr.size:,} / test 侧 {unit_te.size:,}"
          + (f" | step1 报告: {split_spec.get('method')} "
             f"train_frac={split_spec.get('train_frac')} seed={split_spec.get('seed')}"
             if split_spec else " | 未找到 step1 报告 json（跳过比例比对）"), flush=True)
    del row_keys, lab, tr_all
    gc.collect()

    # ------------------------------------------------------------- checklist 1-3 (split)
    chk = Checklist()
    chk.check("pair_never_crosses_split", cross_units.size == 0,
              f"{cross_units.size} 个划分单位同时出现在两侧 "
              f"(train 侧 {unit_tr.size:,} / test 侧 {unit_te.size:,})")
    tr_keys = pair_keys(train.player_id.to_numpy(), train.beatmap_id.to_numpy(), k)
    te_keys = pair_keys(test.player_id.to_numpy(), test.beatmap_id.to_numpy(), k)
    tr_in_test = np.isin(tr_keys, test_pairs)
    te_in_test = np.isin(te_keys, test_pairs)
    chk.check("train_and_test_disjoint",
              (not tr_in_test.any()) and bool(te_in_test.all()),
              f"train rows on a held-out pair = {int(tr_in_test.sum()):,}; "
              f"test rows on a held-out pair = {int(te_in_test.sum()):,}/{len(test):,}")
    if split_spec:
        frac_ok = abs(float(is_test.mean()) - expected_test_frac) <= frac_tol
        frac_detail = (f"观测 test 单位占比 {is_test.mean():.4%} vs step1 报告 "
                       f"{expected_test_frac:.2%}（容差 {frac_tol:.0%}）")
    else:
        frac_ok = True
        frac_detail = (f"未找到 step1 报告 json：只记录观测 test 单位占比 "
                       f"{is_test.mean():.4%}（{n_test_pairs:,}/{n_pairs:,}）")
    chk.check("test_pair_frac", frac_ok, frac_detail)
    del tr_keys, te_keys, tr_in_test, te_in_test
    gc.collect()

    # ------------------------------------------------------------- entity / item ids
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

    # ----------------------------------------------------------- frozen content matrix
    chart_feats, content_info = None, None
    if content_ids is not None and args.chart_content == "none":
        print("[content] restricted to the covered charts, but the vectors are NOT used "
              "(--chart-content none): this is the 1-hot baseline on the same subset", flush=True)
    if content_ids is not None and args.chart_content != "none":
        Z, content_info = align_content(content_ids, content_E, item_ids,
                                        args.content_dim, args.content_reduce)
        args.dim = int(Z.shape[1])          # P lives in the content space, so --dim follows
        chart_feats = torch.as_tensor(Z, dtype=torch.float32, device=device)
        n_free = len(ents) * args.dim
        print(f"[content] P becomes ({len(ents):,}, {args.dim}) = {n_free:,} free player "
              f"parameters; per-chart free parameters left: bm only ({len(items):,})", flush=True)

    # ------------------------------------------------------- calendar-time drift axis
    # t is in years CE; t_ref is the MEAN TRAINING timestamp, so Pc is "how far this
    # player's latent vector moves per calendar year" and the model starts drift-free.
    t_ref = 0.0
    if args.p_drift == "time":
        t_ref = float(pd.to_datetime(train.timestamp).to_numpy(dtype="datetime64[D]")
                      .astype(np.float64).mean() / 365.25)
        yr = pd.to_datetime(train.timestamp).dt.year
        print(f"[time-drift] t_ref = {t_ref:.3f} yr CE (mean train timestamp, "
              f"{yr.min()}-{yr.max()}); Pc = latent displacement per YEAR", flush=True)

    def encode(df):
        e = map_ids(df.entity.to_numpy(), ent_ids)
        i = map_ids(df.beatmap_id.to_numpy(), item_ids)
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

    # ------------------------------------------------- checklist 4, 6, 8 (data + seeds)
    if args.tag in LOSS_RANGE:
        lo, hi = LOSS_RANGE[args.tag]
        note = ""
        if args.label_shift > 0:
            # Old range [lo, hi] for log(1-ACC); new label log(s + exp(old)).
            lo = math.log(args.label_shift + math.exp(lo))
            hi = math.log(args.label_shift + math.exp(hi))
            note = f" (mapped through label shift {args.label_shift:g})"
        loss_ok = bool(np.isfinite(te_y).all() and te_y.min() >= lo - 1e-3 and te_y.max() <= hi + 1e-3)
        loss_detail = (f"test({args.tag}) loss in [{te_y.min():.4f}, {te_y.max():.4f}], "
                       f"expected [{lo:.4f}, {hi:.4f}]{note}")
    else:
        loss_ok = bool(np.isfinite(te_y).all() and (te_y <= 0).all())
        loss_detail = f"test({args.tag}) loss in [{te_y.min():.4f}, {te_y.max():.4f}] (no per-tag range registered)"
    chk.check("loss_in_range", loss_ok, loss_detail)
    chk.check("vocab_from_train_only",
              np.array_equal(ent_ids, np.unique(train.entity.to_numpy()))
              and np.array_equal(item_ids, np.unique(train.beatmap_id.to_numpy())),
              f"{ent_ids.size:,} entities / {item_ids.size:,} items == unique train ids")
    chk.check("determinism_selftest", determinism_selftest(device),
              "same seed -> bitwise identical parameters on a synthetic training path")
    if chart_feats is not None:
        # C[i] must belong to item_ids[i].  A silent misalignment here would poison every
        # downstream number, so it is asserted rather than eyeballed.
        step = max(1, len(item_ids) // 64)
        probe = np.arange(len(item_ids))[::step][:64]
        pos = pd.Index(content_ids).get_indexer(item_ids[probe])
        norms = chart_feats.norm(dim=1)
        chk.check("content_aligned_with_item_ids",
                  bool((pos >= 0).all()) and len(chart_feats) == len(item_ids)
                  and bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-4)),
                  f"{len(item_ids):,} train charts; {len(probe)} probe ids all resolve; "
                  f"row norm in [{float(norms.min()):.4f}, {float(norms.max()):.4f}]")

    # ----------------------------------------------------------------------- tensors
    def dev(a):
        return torch.as_tensor(a, device=device)

    tr_e_t, tr_i_t = dev(tr_e), dev(tr_i)
    tr_y_t = dev(tr_y)
    tr_x_t = dev(tr_x) if tr_x is not None else None
    tr_tau_t = dev(tr_tau) if tr_tau is not None else None
    tr_t_t = dev(tr_t) if tr_t is not None else None
    te_e_t, te_i_t = dev(te_e[te_ok]), dev(te_i[te_ok])
    te_y_np = te_y[te_ok]
    te_y_keep = te_y_np.copy()      # 训练前 te_y 会被释放；--save-preds 还需要这份标签
    te_x_t = dev(te_x[te_ok]) if te_x is not None else None
    te_tau_t = dev(te_tau[te_ok]) if te_tau is not None else None
    te_t_t = dev(te_t[te_ok]) if te_t is not None else None
    idx_tr = np.flatnonzero(tr_ok).astype(np.int64)
    del tr_e, tr_i, tr_x, tr_tau, tr_t, tr_y, te_e, te_i, te_x, te_tau, te_t, te_y
    gc.collect()

    # ---------------------------------------------------------------------- training
    results, fitted, mech_by_model, degeneracy = {}, {}, {}, {}
    for kind in models_wanted:
        model = EmbeddingModel(kind, len(ents), len(items), args.dim, args.features,
                               args.seed, device, args.p_drift, args.d_constraint,
                               args.d_init_std, args.drop_bias,
                               content=args.chart_content, chart_feats=chart_feats,
                               c_init_std=args.c_init_std)
        if args.p_drift != "none" and kind == "bias":
            print("note: --p-drift has no effect on bias (it has no P vector)", flush=True)
        # 参数分组与 wd 策略全部交给 build_optimizer：model.no_decay 里的参数
        # （softmax/softplus 的 logits、mirt_exp 的难度尺度）默认 wd=0，其余用 --wd。
        opt, wd_of = build_optimizer(model, args)
        mech_by_model[kind] = {"d_constraint": model.d_constraint,
                               "c_constraint": model.c_constraint,
                               "c_eff_wd": (float(args.wd) if model.c_constraint != "none"
                                            else 0.0),
                               "d_logit_wd": wd_of.get("D", args.wd),
                               "c_logit_wd": wd_of.get("C", args.wd),
                               "weight_decay": {k: wd_of[k] for k in wd_of},
                               "no_decay": sorted(model.no_decay)}
        if model.no_decay:
            print(f"  [{kind}] no-decay 参数: "
                  + ", ".join(f"{k}={wd_of[k]:g}" for k in sorted(model.no_decay)), flush=True)
        # mirt_1: L2 on the EFFECTIVE C (softplus(Z)), because Adam's wd only sees Z.
        c_eff_wd = float(args.wd) if model.c_constraint != "none" else 0.0
        if c_eff_wd:
            print(f"  [{kind}] effective-C L2 = {c_eff_wd:g} (penalty on C, not on Z)",
                  flush=True)
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
                loss = ((pred - tr_y_t[b]) ** 2).mean()
                if c_eff_wd:
                    loss = loss + c_eff_wd * model.c_penalty()
                loss.backward()
                opt.step()
                model.gauge_fix()       # 钉死 D/P 的公共原点（只对 mirt_exp 生效）
            if args.eval_every and (ep + 1) % args.eval_every == 0:
                pv = model.predict(te_e_t, te_i_t, te_x_t, te_tau_t, te_t_t)
                print(f"  [{kind}] epoch {ep+1:3d}  test RMSE = "
                      f"{np.sqrt(((pv - te_y_np) ** 2).mean()):.5f}  ({time.time()-t0:.1f}s)",
                      flush=True)
        # 指标只算在 test 行上（train 行只用于拟合，从不进入这里）
        model.gauge_fix()
        pv = model.predict(te_e_t, te_i_t, te_x_t, te_tau_t, te_t_t)
        results[kind] = report(pv, te_y_np, len(test))
        fitted[kind] = model
        degeneracy[kind] = degeneracy_report(model)
        warn_degenerate(kind, degeneracy[kind])
        print(f"{kind:8s} test rmse={results[kind]['rmse']:.5f} "
              f"cov={results[kind]['coverage']:.3f}", flush=True)

    # ----------------------------------------------- checklist 5, 7 (coverage / leak)
    chk.check("coverage_reported",
              all(("coverage" in r) and ("n_scored" in r) and r["coverage"] is not None
                  for r in results.values()),
              f"{len(results)} result blocks carry coverage + n_scored")
    leak = np.isin(pair_keys(train.player_id.to_numpy()[idx_tr],
                             train.beatmap_id.to_numpy()[idx_tr], k), test_pairs).sum()
    chk.check("no_test_leak", bool(leak == 0 and idx_tr.size == int(tr_ok.sum())),
              f"training rows used = {idx_tr.size:,}; of those on a held-out pair = {int(leak):,}; "
              f"test tensors never enter the optimiser")
    del test_pairs, tr_pairs
    gc.collect()

    # ------------------------------------------------------------------------- save
    inv_e = {v: kk for kk, v in ents.items()}
    inv_i = {v: kk for kk, v in items.items()}
    variant = args.models.replace(",", "+")
    if args.features != "F0":
        variant += f"_{args.features}"
    if args.entity != "player":
        variant += f"_{args.entity.replace('player-year', 'py')}"
    if args.p_drift != "none":
        variant += {"logpc": "_pd", "time": "_pdt"}[args.p_drift]
    if args.chart_content != "none":
        variant += f"_cm3p{args.dim}" + ("" if args.content_reduce == "pca" else "raw")
    elif args.restrict_to_content == "on":
        variant += "_cov"
    if args.d_constraint != "none":
        variant += f"_d{args.d_constraint}"
    if args.d_init_std is not None:
        variant += f"_di{args.d_init_std}"
    if args.d_logit_wd is not None:
        variant += f"_dlw{args.d_logit_wd}"
    if args.drop_bias != "none":
        variant += {"u": "_nobu", "m": "_nobm", "both": "_nobubm"}[args.drop_bias]
    if args.dim != 16:
        variant += f"_dim{args.dim}"
    if args.label_shift:
        variant += f"_ls{args.label_shift:g}"
    entity_ids = np.array(list(inv_e.values()), dtype=np.int64)
    beatmap_ids = np.array(list(inv_i.values()), dtype=np.int32)
    npz_paths = {}
    for kind, model in fitted.items():
        path = out_dir / f"{args.out_prefix}_{args.tag}_{kind}_{args.features}" \
                      f"{'' if args.chart_content != 'none' else ('_cov' if args.restrict_to_content == 'on' else '')}" \
                      f"{'' if args.chart_content == 'none' else '_cm3p'}" \
                      f"{'' if args.dim == 16 else '_dim' + str(args.dim)}" \
                      f"{'' if not args.label_shift else '_ls' + format(args.label_shift, 'g')}" \
                      f"{'' if args.p_drift == 'none' else ('_pd' if args.p_drift == 'logpc' else '_pdt')}" \
                      f"{'' if args.d_constraint == 'none' else '_d' + args.d_constraint}" \
                      f"{'' if args.entity == 'player' else '_py'}" \
                      f"_{args.target}.npz"
        extra = {}
        if "Pc" in model.p:
            extra["Pc"] = model.np_array("Pc")
        if model.d_constraint != "none":
            extra["D_raw"] = model.np_array("D")        # logits, so the constraint is reproducible
        if model.c_is_logit or model.c_constraint != "none":
            extra["C_raw"] = model.np_array("C")        # logits；C 里存的是生效值
        C_eff = model.export_C()        # frozen content when --chart-content, else the table
        np.savez(path,
                 entity_ids=entity_ids,
                 beatmap_ids=beatmap_ids,
                 P=model.np_array("P"),
                 V=model.np_array("V", np.zeros_like(model.np_array("P"))),
                 C=C_eff,
                 # effective D (constraint applied); bias/mf_dot have no D -> zeros like C
                 D=model.export_D() if "D" in model.p else np.zeros_like(C_eff),
                 b0=model.np_array("b0"),
                 bu=model.np_array("bu", np.zeros(len(ents), np.float32)),
                 bm=model.np_array("bm", np.zeros(len(items), np.float32)),
                 **extra)
        npz_paths[kind] = path

    # ------------------------------------------------------------- checklist 9 (npz)
    ok9, detail9 = True, []
    for kind, path in npz_paths.items():
        with np.load(path) as z:
            good = (len(z["entity_ids"]) == len(ents) and len(z["beatmap_ids"]) == len(items)
                    and z["P"].shape == (len(ents), args.dim)
                    and z["C"].shape == (len(items), args.dim)
                    and z["D"].shape == z["C"].shape and z["V"].shape == z["P"].shape
                    and z["b0"].shape == (1,) and len(z["bu"]) == len(ents)
                    and len(z["bm"]) == len(items)
                    and (("Pc" not in z.files) or z["Pc"].shape == z["P"].shape)
                    and (("D_raw" not in z.files) or z["D_raw"].shape == z["C"].shape)
                    and (("C_raw" not in z.files) or z["C_raw"].shape == z["C"].shape))
        ok9 = ok9 and bool(good)
        detail9.append(f"{kind}:{'ok' if good else 'BAD'}")
    chk.check("npz_matches_summary", ok9, " ".join(detail9))

    okc, detailc = True, []
    for kind, path in npz_paths.items():
        if args.d_constraint == "none" or "D" not in np.load(path).files:
            continue
        with np.load(path) as z:
            D = z["D"]
            good = bool((D >= 0).all())
            if args.d_constraint == "simplex":
                good = good and bool(np.allclose(D.sum(1), 1.0, atol=1e-5))
                detailc.append(f"{kind}: D>=0, max|sum_i D_i - 1| = "
                               f"{np.abs(D.sum(1) - 1).max():.2e}")
            else:
                detailc.append(f"{kind}: D>0, min D = {D.min():.3e}")
        okc = okc and good
    if detailc:
        chk.check("d_constraint_holds", okc, " ".join(detailc))

    okc2, detailc2 = True, []
    for kind, path in npz_paths.items():
        if fitted[kind].c_constraint == "none":
            continue
        with np.load(path) as z:
            C = z["C"]
            good = bool((C >= 0).all())
            detailc2.append(f"{kind}: C>=0, min C = {C.min():.3e}, "
                            f"frac(C==0) = {float((C == 0).mean()):.2e}")
        okc2 = okc2 and good
    if detailc2:
        chk.check("c_constraint_holds", okc2, " ".join(detailc2))
    chk.require()

    # 可选：把测试行的预测落盘，供 scripts/compare_preds.py 做配对检验
    preds_path = None
    if args.save_preds:
        preds_path = out_dir / f"{args.out_prefix}_{args.tag}_preds_{args.target}.npz"
        np.savez(preds_path, player_id=test.player_id.to_numpy()[te_ok],
                 beatmap_id=test.beatmap_id.to_numpy()[te_ok],
                 y=te_y_keep, variant=variant,
                 **{f"pred_{kind}": fitted[kind].predict(
                     te_e_t, te_i_t, te_x_t, te_tau_t, te_t_t) for kind in fitted})
        print(f"wrote {preds_path.name} ({int(te_ok.sum()):,} test rows)")

    summary = {
        "variant": variant,
        # --train-path / --out-dir 等是 Path，json 里统一存字符串
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "mechanism": {"p_drift": args.p_drift, "d_constraint": args.d_constraint,
                      "d_init_std": args.d_init_std if args.d_init_std is not None
                      else (0.5 if args.d_constraint == "simplex" else 0.05),
                      "by_model": mech_by_model},
        "engine": {"name": "torch", "version": torch.__version__, "device": str(device),
                   "threads": torch.get_num_threads(),
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "split": {"from": "step1", "col": args.split_col, "spec": split_spec,
                  "unit": unit_kind, "n_units": n_pairs, "n_test_units": n_test_pairs,
                  "observed_test_frac": float(is_test.mean()),
                  "rows": {"train": int(len(train)), "test": int(len(test)),
                           "dropped": n_dropped}},
        "sizes": {"train_plays": int(len(train)), "test_plays": int(len(test)),
                  "dropped_rows": n_dropped,
                  "n_entities": len(ents), "n_items": len(items),
                  "train_rows_in_vocab": int(tr_ok.sum()),
                  "test_rows_covered": int(te_ok.sum())},
        "label": {"shift": args.label_shift,
                  "definition": "log(shift + exp(step0_loss)) = log(1+shift-ACC)"},
        "content": None if args.chart_content == "none" else {
            "mode": args.chart_content, "path": str(args.content_path),
            "dim": args.dim, "reduce": args.content_reduce,
            "n_charts_with_vector": int(len(content_ids)),
            "filtered_rows": content_drop,
        },
        "time_drift": None if args.p_drift != "time" else {
            "t_ref_year_ce": round(t_ref, 4), "unit": "years",
            "definition": "P_eff = P[u] + Pc[u] * (timestamp - t_ref) / 365.25d",
        },
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


if __name__ == "__main__":
    main()
