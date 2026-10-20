#!/usr/bin/env python
"""Step 2 (version: minacalc_L2) -- MinaCalc MSD 的 gap / IRT 模型 (L2).

方法（设计见 docs/step2_minacalc.md §6、docs/step2_minacalc_L2.md）:

    x(u, m) = a + bu[u] + bm[m] + sum_k w_k * (theta[u,k] - D[m,k])
                                     [+ sum_k v_k * relu(D[m,k] - theta[u,k])]
    ACC     = sigmoid(x)
    loss    = log(1 - ACC) = -softplus(x)            (--link logit，默认)
    loss    = x                                       (--link identity，消融用)

      D[m,k]     MinaCalc 的 MSD（冻结；按训练谱面逐技能 z-score）
      theta[u,k] 玩家技能向量（可学，K = 技能数；--theta-init chartmean 用"该玩家训练局
                 的平均谱面难度"初始化）
      w_k        softplus(w_raw) > 0：技能越高 loss 越小、谱面越难 loss 越大（单调）
      v_k        softplus(v_raw) > 0：只惩罚"打不动"（D > theta）的那一侧

三个模型（--models）:

    bias     b0 + bu + bm —— step2/predict 的同名基线，只是谱面子集被裁到有 MSD 的那些
    gap      L2 本体：共享 w 的 gap（theta 与 D 活在同一把尺子上）
    gapfree  消融：x = a + bu + bm + sum_k alpha_k*theta_k - sum_k beta_k*D_k
              （theta 与 D 各自一套权重，没有"同一把尺子"的约束）

为什么这样接:

  * loss = log(1-ACC) 与 sigmoid 的组合给出**精确**的链接函数，不用另拍一个
    "MSD 到 ACC"的映射；这一步把 L1 的加性/双线性换成 IRT 式的差，是 L2 的全部内容。
  * 点积 <P, C> 是加性的，表达不了"同一个 +1 难度，对高手无所谓、对新手致命"；
    gap 形式把"难度必须与能力比较"写进了公式。
  * theta 可学（R1），**不**反解 SSR（R2）——反解要用该玩家更早的成绩，
    在 pair-random 划分下很容易把测试局自己的信息漏进来。
  * D 只依赖 .osu（不依赖成绩），所以 beatmap-random 划分（整图留出）下"特征本身"依然可用 ——
    这是它相对可学 C_m 的结构性优势。注意：当前词表只从训练行建，未登录谱面会被 te_ok 过滤掉，
    要真正评测整图留出需要 content-only 回退（见 docs/step2_minacalc_L2.md §8.6）。
  * MSD 的 goal=0.93 是 Etterna 的 score 语义，不是 mania ACC：所以 D 只在**训练谱面**上
    标准化、只当相对量用，绝对刻度交给 b0/bu/bm（--gap-bm free 时）或由 -sum w_k*D_k 生成
    （--gap-bm msd，每图自由参数 = 0）。

与 step2/predict、step2/minacalc_as_feature 的一致性:

    划分、词表、损失、指标、检查项、npz/summary 的字段都沿用 step2_predict.py；MSD 资产
    读取复用 step2_minacalc_as_feature.load_msd_table。--models bias 就是**同一谱面子集上**
    的基线，两边数字可直接比。

用法:

    python scripts/step0h_minacalc_msd.py --tag 1k --cache data/osu_cache \
        --ids manifests/whitelist_1k.csv --lib build/libminacalc.so \
        --out data/processed/minacalc_msd_1k.parquet          # 资产（已产出可跳过）

    python scripts/step2_minacalc_L2.py --tag 1k --models bias,gap,gapfree \
        --epochs 60 --msd-reduce zscore --gap-mode linear --gap-bm free

已知坑（与 L1 相同，见 docs/step2_minacalc.md §5）:

    1. goal=0.93 不是 mania ACC，只当相对量；
    2. MinaCalc 的 NoteInfo 没有长按时长，LN 只算头部（表里 hold_ratio 可做子集分析）；
    3. OD 不在 MSD 里，而标签是 ACC 的函数 —— 本脚本暂未接入（需要 join beatmap_meta
       的 diff_overall，做成谱面主效应；见 L1 文档 §5.3）；
    4. msd_overall 是 7 个方向的聚合，默认不用（--msd-skills 可显式包含）。
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
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step2_predict as sp                 # noqa: E402  （同目录，复用模型/指标/检查）
import step2_minacalc_as_feature as mc1    # noqa: E402  （复用 MSD 资产读取）

PROC = sp.PROC
BASE_COLS = sp.BASE_COLS
LOSS_RANGE = sp.LOSS_RANGE

SKILLS = mc1.SKILLS
MSD_COLS = mc1.MSD_COLS
DEFAULT_SKILLS = mc1.DEFAULT_SKILLS
MODELS = ("bias", "gap", "gapfree")


# --------------------------------------------------------------------- 特征
def build_gap_features(ids, X, item_ids, skills, reduce_mode) -> tuple:
    """把 MSD 对齐到训练词表顺序，并做逐技能标准化。

    gap 模型里 theta 与 D 必须处在**同一把尺子**上：raw 也能跑（w_k 会吸收每维的量纲），
    但 theta 的初始化/尺度正则都是按"1 个单位为 1 个谱面难度 sd"设计的，所以默认 z-score。
    pca 不允许：theta 会失去"每个技能方向一项"的语义，hinge 项也没法解释。
    """
    pos = pd.Index(ids).get_indexer(item_ids)
    if (pos < 0).any():
        raise SystemExit(f"{int((pos < 0).sum())} 张训练谱面没有 MSD 行；"
                         f"play 表必须先裁到有 MSD 的谱面（本脚本默认会裁）")
    Xa = np.ascontiguousarray(X[pos], dtype=np.float32)
    if reduce_mode == "raw":
        Z = Xa
        info = {"reduce": "raw"}
        print("[gap] 注意：--msd-reduce raw 时 theta 的尺度正则（--theta-prior）按 raw 单位走",
              flush=True)
    else:
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-12
        Z = ((Xa - mu) / sd).astype(np.float32)
        info = {"reduce": "zscore", "mean": mu, "scale": sd}
    info.update({"dim": int(Z.shape[1]), "skills": list(skills),
                 "fitted_on": "train_charts"})
    print(f"[gap] 特征 {Z.shape[0]:,} x {Z.shape[1]}  reduce={reduce_mode}  "
          f"列均值|max|={np.abs(Z.mean(0)).max():.2e}  列 std 范围 "
          f"[{Z.std(0).min():.3f}, {Z.std(0).max():.3f}]", flush=True)
    return Z, info


def init_theta_chartmean(n_ent: int, e: np.ndarray, i: np.ndarray, D: np.ndarray) -> np.ndarray:
    """theta0[u] = 该玩家**训练局**所打谱面的平均难度向量（只用于初始化，不参与预测）。

    它把玩家直接放进"谱面难度"这把尺子里，比全零初始化收敛快；没有训练局的玩家保持 0。
    """
    K = D.shape[1]
    sums = np.zeros((n_ent, K), dtype=np.float64)
    cnt = np.zeros(n_ent, dtype=np.float64)
    step = 200_000
    for s in range(0, e.size, step):
        sl = slice(s, s + step)
        np.add.at(sums, e[sl], D[i[sl]])
        np.add.at(cnt, e[sl], 1.0)
    th = sums / np.maximum(cnt, 1.0)[:, None]
    th[cnt == 0] = 0.0
    print(f"[gap] theta 初始化 chartmean：{int((cnt > 0).sum()):,}/{n_ent:,} 个实体有训练局，"
          f"每维 std 均值 {th.std(0).mean():.3f}", flush=True)
    return th.astype(np.float32)


def inv_softplus(x: float) -> float:
    """softplus 的反函数：让 w 初始化在 x（而不是 0.693）。"""
    return float(np.log(np.expm1(x)))


# ----------------------------------------------------------------------- 模型
class GapModel(sp.EmbeddingModel):
    """b0 + bu + bm + gap（+ 单边 hinge），再过一个把 ACC 映射到 loss 的链接。

    继承 EmbeddingModel(kind="bias")：复用 b0/bu/bm、F1/F2、p_drift、drop_bias 与
    优化器的那套参数登记（self.p / self.active / self.no_decay），只额外挂三组参数：
    theta（玩家技能）、w_raw（共享权重）、v_raw（单边权重，可选）/ alpha+beta（gapfree）。
    """

    def __init__(self, n_ent, n_item, k, features, seed, device, p_drift, d_constraint,
                 d_init_std, drop_bias, chart_feats, link="logit", gap_mode="linear",
                 shared=True, theta_init=None, w_init=1.0):
        super().__init__("bias", n_ent, n_item, k, features, seed, device, p_drift,
                         d_constraint, d_init_std, drop_bias)
        self.register_buffer("D_feat", torch.as_tensor(
            chart_feats, dtype=torch.float32, device=device))
        self.k = int(k)
        self.link = link
        self.gap_mode = gap_mode
        self.shared = bool(shared)

        th0 = torch.zeros((n_ent, k), dtype=torch.float32) if theta_init is None \
            else torch.as_tensor(theta_init, dtype=torch.float32)
        self.p["theta"] = nn.Parameter(th0.clone().to(device))
        self.p["w_raw"] = nn.Parameter(torch.full((k,), inv_softplus(w_init),
                                                  dtype=torch.float32, device=device))
        self.active = [*self.active, "theta", "w_raw"]
        if not self.shared:
            self.p["alpha_raw"] = nn.Parameter(torch.full((k,), inv_softplus(w_init),
                                                          dtype=torch.float32, device=device))
            self.p["beta_raw"] = nn.Parameter(torch.full((k,), inv_softplus(w_init),
                                                         dtype=torch.float32, device=device))
            self.active = [*self.active, "alpha_raw", "beta_raw"]
        if self.gap_mode in ("hinge", "both"):
            self.p["v_raw"] = nn.Parameter(torch.full((k,), inv_softplus(0.5),
                                                      dtype=torch.float32, device=device))
            self.active = [*self.active, "v_raw"]

        # theta 的平移是纯规范自由度（theta += c，a -= sum w_k c_k），而 wd 会把 theta 拉向 0、
        # gauge_fix 又把它中心化 —— 两股力抵消不掉，只会把 theta 的尺度越压越小。
        # 所以 theta 不吃 wd，尺度交给 --theta-prior（默认 1e-2，把每维 std 拉向 1）。
        self.no_decay.add("theta")

    def _w(self):
        return F.softplus(self.p["w_raw"])

    def linear_predictor(self, u, m, x1=None, tau=None, t=None):
        """b0 + bu + bm (+F1/F2) + gap。forward 只在这上面套链接。"""
        out = super().forward(u, m, x1, tau, t)
        D = self.D_feat[m]
        th = self.p["theta"][u]
        if self.shared:
            out = out + (self._w() * (th - D)).sum(1)
        else:
            out = out + (F.softplus(self.p["alpha_raw"]) * th).sum(1) \
                      - (F.softplus(self.p["beta_raw"]) * D).sum(1)
        if "v_raw" in self.p:
            out = out + (F.softplus(self.p["v_raw"]) * F.relu(D - th)).sum(1)
        return out

    def forward(self, u, m, x1=None, tau=None, t=None):
        x = self.linear_predictor(u, m, x1, tau, t)
        return -F.softplus(x) if self.link == "logit" else x

    @torch.no_grad()
    def gauge_fix(self):
        """把 theta 在每个技能维上中心化，位移量折进 b0（模型输出逐位不变）。

        只有"差"有意义的模型都有一个平移规范；Adam 是逐参数归一化的、更新并不正交于它，
        所以不钉住的话 theta 的水平会无约束漂移，theta 的绝对值就没法读。
        """
        super().gauge_fix()
        if self.p["theta"].shape[0] < 2:
            return
        shift = self.p["theta"].mean(0, keepdim=True)
        if float(shift.abs().max()) == 0.0:
            return
        self.p["theta"].sub_(shift)
        if self.shared:
            gain = (self._w() * shift.squeeze(0)).sum()
        else:
            gain = (F.softplus(self.p["alpha_raw"]) * shift.squeeze(0)).sum()
        self.p["b0"].add_(gain)

    def theta_scale_penalty(self) -> torch.Tensor:
        """每维 std 拉向 1：把 theta 的尺度钉在"1 = 1 个谱面难度 sd"上。

        没有它，theta -> a*theta、w -> w/a 是精确的平坦方向：预测不变，但 theta 的数值
        不可解释，不同 seed 之间也没法比。
        """
        th = self.p["theta"]
        if th.shape[0] < 2:
            return torch.zeros((), device=th.device)
        return ((th.std(0, unbiased=False) - 1.0) ** 2).mean()

    def export_theta(self) -> np.ndarray:
        return self.p["theta"].detach().to("cpu").numpy()

    def export_weights(self) -> dict:
        out = {"w": F.softplus(self.p["w_raw"]).detach().to("cpu").numpy()
               if self.shared else None}
        if not self.shared:
            out["alpha"] = F.softplus(self.p["alpha_raw"]).detach().to("cpu").numpy()
            out["beta"] = F.softplus(self.p["beta_raw"]).detach().to("cpu").numpy()
        if "v_raw" in self.p:
            out["v"] = F.softplus(self.p["v_raw"]).detach().to("cpu").numpy()
        return out


def effective_drop_bias(user_drop_bias: str, gap_bm: str) -> str:
    """--gap-bm msd 要把 bm 拿掉（谱面截距完全由 -sum w_k*D_k 生成）。"""
    if gap_bm == "free":
        return user_drop_bias
    return {"none": "m", "u": "both", "m": "m", "both": "both"}[user_drop_bias]


def build_model(name, args, device, n_ent, n_item, k, D, theta0, drop_bias):
    common = (n_ent, n_item, k, args.features, args.seed, device, args.p_drift,
              args.d_constraint, args.d_init_std, drop_bias)
    if name == "bias":
        if args.gap_bm == "msd":
            print("  [bias] --gap-bm msd 对 bias 无意义（它没有任何谱面特征）；"
                  "这就是同一子集上的 step2/predict 基线", flush=True)
        return sp.EmbeddingModel("bias", *common)
    # gapfree 的定义就是"没有共享尺子"，所以它固定 shared=False；--gap-shared 只作用于 gap
    shared = (args.gap_shared == "on") and name == "gap"
    return GapModel(*common, chart_feats=D, link=args.link, gap_mode=args.gap_mode,
                    shared=shared,
                    theta_init=(theta0 if args.theta_init == "chartmean" else None),
                    w_init=args.w_init)


# ------------------------------------------------------------------------ CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="1k")
    ap.add_argument("--target", default="loss")
    ap.add_argument("--models", default="bias,gap")
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

    ap.add_argument("--msd-path", type=Path, default=None,
                    help="默认 <proc-dir>/minacalc_msd_<tag>.parquet")
    ap.add_argument("--msd-skills", default=",".join(DEFAULT_SKILLS))
    ap.add_argument("--msd-reduce", default="zscore", choices=["zscore", "raw"])
    ap.add_argument("--allow-unversioned-msd", action="store_true")

    ap.add_argument("--link", default="logit", choices=["logit", "identity"],
                    help="logit: ACC=sigmoid(x), loss=-softplus(x)（L2 本体）；"
                         "identity: 直接线性预测 loss（L1 式消融）")
    ap.add_argument("--gap-mode", default="linear", choices=["linear", "hinge", "both"],
                    help="linear: w*(theta-D)；hinge: 只惩罚 relu(D-theta)；both: 两者都加")
    # nargs="?"：YAML 里写 `gap_shared: on` 会被解析成 True，编排层会发一个裸 --gap-shared
    ap.add_argument("--gap-shared", nargs="?", const="on", default="on",
                    choices=["on", "off"],
                    help="on: 共享 w_k（theta 与 D 同一把尺子）；off: alpha_k/beta_k 各一套")
    ap.add_argument("--gap-bm", default="free", choices=["free", "msd"],
                    help="msd: 去掉 bm[m]，谱面截距完全由 -sum w_k*D_k 生成")
    ap.add_argument("--theta-init", default="chartmean", choices=["chartmean", "zero"])
    ap.add_argument("--theta-prior", type=float, default=1e-2,
                    help="把 theta 每维 std 拉向 1 的强度（0 = 关）")
    ap.add_argument("--w-init", type=float, default=1.0)
    return ap.parse_args(argv)


# ----------------------------------------------------------------------- main
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
    if len(skills) < 2:
        raise SystemExit("gap 模型至少要 2 个技能方向（theta 才有跨技能的比较）")
    k_skills = len(skills)

    msd_path = Path(args.msd_path) if args.msd_path \
        else Path(args.proc_dir) / f"minacalc_msd_{args.tag}.parquet"
    drop_bias = effective_drop_bias(args.drop_bias, args.gap_bm)

    need_pc = args.features in ("F1", "F2") or args.p_drift == "logpc"
    need_ts = args.features == "F2" or args.entity == "player-year" or args.p_drift == "time"
    cols = BASE_COLS + [args.target] \
        + (["playcount_cur"] if need_pc else []) + (["timestamp"] if need_ts else [])

    tr_all = sp.load_play_table(args.tag, "step1", cols, path=args.train_path,
                                proc_dir=args.proc_dir, optional_cols=(args.split_col,))

    # ------------------------------------------------------- MSD 覆盖率过滤
    # 与 step2/predict 的 content 过滤同位置、同理由：在划分**之前**裁，否则测试集会被换掉。
    msd_ids, msd_X, msd_meta = mc1.load_msd_table(msd_path, skills)
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

    D_std, gap_feat_info = build_gap_features(msd_ids, msd_X, item_ids, skills, args.msd_reduce)
    args.dim = int(D_std.shape[1])            # theta 活在 D 的空间里
    theta0 = None

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

    if args.theta_init == "chartmean":
        theta0 = init_theta_chartmean(len(ents), tr_e[tr_ok], tr_i[tr_ok], D_std)

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
    d_stats = (float(D_std.mean()), float(D_std.std(0).mean()))
    chk.check("gap_features_aligned",
              D_std.shape == (len(items), k_skills) and bool(np.isfinite(D_std).all())
              and abs(d_stats[1] - 1.0) < 0.05,
              f"D {D_std.shape} 训练谱面上每维 std 均值 {d_stats[1]:.4f}（z-score 应为 1）")

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

    D_t = dev(D_std)
    # ------------------------------------------------------------------- 训练
    results, fitted, mech_by_model, degeneracy = {}, {}, {}, {}
    for kind in models_wanted:
        model = build_model(kind, args, device, len(ents), len(items), args.dim,
                            D_t, theta0, drop_bias)
        opt, wd_of = sp.build_optimizer(model, args)
        mech_by_model[kind] = {"link": args.link if kind != "bias" else None,
                               "gap_mode": args.gap_mode if kind != "bias" else None,
                               "gap_shared": args.gap_shared if kind != "bias" else None,
                               "gap_bm": args.gap_bm, "drop_bias": drop_bias,
                               "theta_init": args.theta_init, "theta_prior": args.theta_prior,
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
                loss = ((pred - tr_y_t[b]) ** 2).mean()
                if args.theta_prior > 0 and hasattr(model, "theta_scale_penalty"):
                    loss = loss + args.theta_prior * model.theta_scale_penalty()
                loss.backward()
                opt.step()
                model.gauge_fix()
            if args.eval_every and (ep + 1) % args.eval_every == 0:
                pv = model.predict(te_e_t, te_i_t, te_x_t, te_tau_t, te_t_t)
                wtxt = ""
                if hasattr(model, "export_weights"):
                    ws = model.export_weights()
                    if ws.get("w") is not None:
                        wtxt = f"  w=[{', '.join(f'{v:.2f}' for v in ws['w'])}]"
                print(f"  [{kind}] epoch {ep+1:3d}  test RMSE = "
                      f"{np.sqrt(((pv - te_y_np) ** 2).mean()):.5f}{wtxt}  "
                      f"({time.time()-t0:.1f}s)", flush=True)
        model.gauge_fix()
        pv = model.predict(te_e_t, te_i_t, te_x_t, te_tau_t, te_t_t)
        results[kind] = sp.report(pv, te_y_np, len(test))
        fitted[kind] = model
        degeneracy[kind] = sp.degeneracy_report(model)
        sp.warn_degenerate(kind, degeneracy[kind])
        extra = ""
        if hasattr(model, "export_weights"):
            wts = model.export_weights()
            if wts.get("w") is not None:
                extra = f"  w=[{', '.join(f'{v:.2f}' for v in wts['w'])}]"
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
    gap_tag = f"_{args.link}_{args.gap_mode}" \
              + ("" if args.gap_shared == "on" else "_unshared") \
              + ("" if args.gap_bm == "free" else "_bmmsd") \
              + ("" if args.theta_init == "chartmean" else "_th0")
    variant = args.models.replace(",", "+") + gap_tag
    if args.features != "F0":
        variant += f"_{args.features}"
    if args.entity != "player":
        variant += "_py"
    if args.p_drift != "none":
        variant += {"logpc": "_pd", "time": "_pdt"}[args.p_drift]
    if args.drop_bias != "none":
        variant += {"u": "_nobu", "m": "_nobm", "both": "_nobubm"}[args.drop_bias]
    entity_ids = np.array(list(inv_e.values()), dtype=np.int64)
    beatmap_ids = np.array(list(inv_i.values()), dtype=np.int32)
    npz_paths = {}
    for kind, model in fitted.items():
        path = out_dir / f"{args.out_prefix}_{args.tag}_{kind}_{args.features}{gap_tag}" \
                         f"{'' if args.entity == 'player' else '_py'}" \
                         f"{'' if args.p_drift == 'none' else ('_pd' if args.p_drift == 'logpc' else '_pdt')}" \
                         f"_{args.target}.npz"
        extra = {"d_std": D_std}
        if hasattr(model, "export_theta"):
            extra["theta"] = model.export_theta()
            wts = model.export_weights()
            for name, arr in wts.items():
                if arr is not None:
                    extra[name] = arr
        np.savez(path, entity_ids=entity_ids, beatmap_ids=beatmap_ids,
                 P=model.np_array("P", np.zeros((len(ents), args.dim), np.float32)),
                 V=model.np_array("V", np.zeros((len(ents), args.dim), np.float32)),
                 C=model.export_C(), D=model.export_D() if "D" in model.p else np.zeros((len(items), args.dim), np.float32),
                 b0=model.np_array("b0"),
                 bu=model.np_array("bu", np.zeros(len(ents), np.float32)),
                 bm=model.np_array("bm", np.zeros(len(items), np.float32)),
                 msd_cols=np.array([f"msd_{s}" for s in skills]),
                 **extra)
        npz_paths[kind] = path

    ok9, detail9 = True, []
    for kind, path in npz_paths.items():
        with np.load(path) as z:
            good = (len(z["entity_ids"]) == len(ents) and len(z["beatmap_ids"]) == len(items)
                    and z["d_std"].shape == (len(items), args.dim)
                    and (("theta" not in z.files) or z["theta"].shape == (len(ents), args.dim))
                    and (("w" not in z.files) or z["w"].shape == (args.dim,)))
        ok9 = ok9 and bool(good)
        detail9.append(f"{kind}:{'ok' if good else 'BAD'}")
    chk.check("npz_matches_summary", ok9, " ".join(detail9))
    chk.require()

    preds_path = None
    if args.save_preds:
        slug = "".join(ch if ch.isalnum() or ch in "+-_" else "_" for ch in variant)[:80]
        preds_path = out_dir / f"{args.out_prefix}_{args.tag}_preds_minacalc_L2_{slug}_{args.target}.npz"
        np.savez(preds_path, player_id=test.player_id.to_numpy()[te_ok],
                 beatmap_id=test.beatmap_id.to_numpy()[te_ok],
                 y=te_y_np, variant=variant,
                 **{f"pred_{kind}": fitted[kind].predict(
                     te_e_t, te_i_t, te_x_t, te_tau_t, te_t_t) for kind in fitted})
        print(f"wrote {preds_path.name} ({int(te_ok.sum()):,} test rows)")

    gap_json = {}
    for kind, model in fitted.items():
        if not hasattr(model, "export_weights"):
            gap_json[kind] = None
            continue
        wts = {name: (None if arr is None else arr.round(5).tolist())
               for name, arr in model.export_weights().items()}
        th = model.export_theta()
        gap_json[kind] = {
            "weights": wts,
            "theta_mean_per_skill": th.mean(0).round(5).tolist(),
            "theta_std_per_skill": th.std(0).round(5).tolist(),
        }

    summary = {
        "variant": variant,
        "args": {kk: (str(v) if isinstance(v, Path) else v) for kk, v in vars(args).items()},
        "mechanism": {"by_model": mech_by_model},
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
        "msd": {**{kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                   for kk, vv in {**msd_meta, **gap_feat_info}.items()},
                "filtered_rows": msd_drop, "dim_used": int(args.dim)},
        "gap": gap_json,
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
    print(json.dumps({kind: {"rmse": r["rmse"], "coverage": r["coverage"]}
                      for kind, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
