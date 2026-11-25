#!/usr/bin/env python3
"""step2_teacher_signature -- build a K-dim reference-player signature per chart.

The teacher is the existing mf_dot model:

    loss_hat[u,m] = b0 + bu[u] + bm[m] + <P[u], C[m]>

We pick K reference players from the TRAIN block, predict each reference
player's loss on every train chart, centre within the chart and z-score:

    s_m[r] = (loss_hat[r,m] - mean_r' loss_hat[r',m]) / (sd_r' + eps)

The result is a fixed, rotation-identifiable summary of "who does relatively
better/worse on this chart".  The student only sees raw notes.

Inputs: a train table (player_id, beatmap_id, loss) and an mf_dot npz written
by scripts/step2_predict.py.  If --fit is given, the script invokes
step2_predict.py first and picks the *mf_dot*.npz it writes.
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def fit_mf_dot(args, out_dir: Path) -> Path:
    # Fit the teacher at the base-player level.  H4's val and test blocks are
    # dropped.  A player-disjoint 20% holdout of the train block gives an
    # honest teacher NLL (the previous row-random carve shared players, so its
    # test NLL was not a cold-player estimate).
    fit_table = args.train_table
    df = pd.read_parquet(args.train_table)
    if "split" in df.columns:
        df = df.copy()
        df["split"] = df["split"].replace({"val": "dropped", "test": "dropped"})
        if "player_id_base" in df.columns:
            df["player_id"] = df["player_id_base"].astype("int64")
        tr = df[df["split"] == "train"]
        # mf_dot's vocab is built from train only, so a player-disjoint test
        # set would have zero covered rows.  Hold out whole (player, chart)
        # pairs instead: a row-level carve can leave the other segment of the
        # same base-player/chart pair in train, which trips step2's
        # pair_never_crosses_split check.
        pairs = tr[["player_id", "beatmap_id"]].drop_duplicates()
        rem_b = pairs["beatmap_id"].value_counts().to_dict()
        rem_p = pairs["player_id"].value_counts().to_dict()
        elig = pairs[(pairs["beatmap_id"].map(rem_b) >= 2)
                     & (pairs["player_id"].map(rem_p) >= 2)]
        n_hold = min(len(elig), max(100, int(round(args.holdout_frac * len(pairs)))))
        cand = elig.sample(n=n_hold, random_state=args.seed)
        test_pairs = set()
        for _, row in cand.iterrows():
            b, p = int(row["beatmap_id"]), int(row["player_id"])
            if rem_b.get(b, 0) > 1 and rem_p.get(p, 0) > 1:
                test_pairs.add((p, b))
                rem_b[b] -= 1
                rem_p[p] -= 1
        if not test_pairs:
            raise SystemExit("could not carve a vocab-preserving teacher test set")
        pair_idx = pd.MultiIndex.from_frame(
            df[["player_id", "beatmap_id"]].astype("int64"))
        df.loc[pair_idx.isin(test_pairs), "split"] = "test"
        fit_table = out_dir / "teacher_fit_table.parquet"
        df.to_parquet(fit_table, index=False)
        print(f"[teacher] teacher_fit_table: train={int((df.split=='train').sum()):,} "
              f"test={int((df.split=='test').sum()):,} "
              f"test_pairs={len(test_pairs):,} dropped={int((df.split=='dropped').sum()):,}",
              flush=True)
    cmd = [sys.executable, str(ROOT / "scripts/step2_predict.py"),
           "--tag", args.tag,
           "--train-path", str(fit_table),
           "--models", "bias,mf_dot",
           "--dim", str(args.dim),
           "--epochs", str(args.epochs),
           "--seed", str(args.seed),
           "--out-dir", str(out_dir),
           "--out-prefix", "mf",
           "--save-preds",
           "--device", args.device]
    print("[teacher] fit:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    hits = sorted(glob.glob(str(out_dir / "*mf_dot*.npz")))
    if not hits:
        raise SystemExit(f"no mf_dot npz under {out_dir}")
    return Path(hits[-1])


def select_refs(players: np.ndarray, bu: np.ndarray, activity: np.ndarray,
                k: int, seed: int) -> np.ndarray:
    """Stratified by ability quintile x activity tertile, random within cell."""
    rng = np.random.default_rng(seed)
    ab = pd.qcut(bu, 5, labels=False, duplicates="drop")
    ac = pd.qcut(activity, 3, labels=False, duplicates="drop")
    cells = {}
    for p, a, c in zip(players, ab, ac):
        cells.setdefault((int(a), int(c)), []).append(int(p))
    keys = sorted(cells)
    chosen = []
    # Round-robin over cells until K, shuffling each cell once.
    for key in keys:
        rng.shuffle(cells[key])
    cursor = {key: 0 for key in keys}
    while len(chosen) < k:
        progressed = False
        for key in keys:
            i = cursor[key]
            if i < len(cells[key]):
                chosen.append(cells[key][i])
                cursor[key] = i + 1
                progressed = True
                if len(chosen) >= k:
                    break
        if not progressed:
            break
    if len(chosen) < k:
        raise SystemExit(f"only {len(chosen)} reference players available, need {k}")
    return np.asarray(chosen[:k], dtype=np.int64)


def holdout_rank_nll(preds_path: Path) -> dict:
    z = np.load(preds_path)
    pid = z["player_id"].astype(np.int64)
    y = z["y"].astype(np.float64)
    out = {}
    for kind in ("bias", "mf_dot"):
        key = f"pred_{kind}"
        if key not in z:
            continue
        p = z[key].astype(np.float64)
        vals = []
        for u in np.unique(pid):
            m = pid == u
            if int(m.sum()) < 2:
                continue
            ii, jj = np.triu_indices(int(m.sum()), 1)
            yy = (y[m][ii] < y[m][jj]).astype(np.float64)
            tie = y[m][ii] == y[m][jj]
            yy[tie] = 0.5
            logit = p[m][jj] - p[m][ii]
            q = 1.0 / (1.0 + np.exp(-logit))
            q = np.clip(q, 1e-9, 1.0 - 1e-9)
            vals.append(-(yy * np.log(q) + (1.0 - yy) * np.log1p(-q)))
        out[kind] = float(np.concatenate(vals).mean()) if vals else None
    if out.get("bias") is not None and out.get("mf_dot") is not None:
        out["delta_mf_dot_minus_bias"] = out["mf_dot"] - out["bias"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-table", required=True, type=Path)
    ap.add_argument("--mf-npz", type=Path, default=None)
    ap.add_argument("--fit", action="store_true", help="run step2_predict.py first")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reference-players", type=int, default=64)
    ap.add_argument("--min-pairs", type=int, default=30)
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                    help="player-disjoint teacher holdout for QA")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--json", required=True, type=Path)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.mf_npz
    if args.fit:
        npz_path = fit_mf_dot(args, out_dir)
    if npz_path is None or not Path(npz_path).exists():
        raise SystemExit("--mf-npz is required unless --fit is given")
    z = np.load(npz_path)
    entity_ids = z["entity_ids"].astype(np.int64)
    beatmap_ids = z["beatmap_ids"].astype(np.int64)
    P = z["P"].astype(np.float64)
    C = z["C"].astype(np.float64)
    bu = z["bu"].astype(np.float64)
    bm = z["bm"].astype(np.float64)
    b0 = float(np.asarray(z["b0"]).reshape(-1)[0])
    print(f"[teacher] npz {npz_path}: {len(entity_ids)} players, "
          f"{len(beatmap_ids)} charts, dim={P.shape[1]}", flush=True)

    train = pd.read_parquet(args.train_table)
    train = train[train["split"] == "train"] if "split" in train.columns else train
    if "player_id_base" in train.columns:
        train = train.assign(_pid=train["player_id_base"].astype("int64"))
    else:
        train = train.assign(_pid=train["player_id"].astype("int64"))
    activity = train.groupby("_pid").size()
    cand = np.array([int(p) for p in entity_ids
                     if int(p) in activity.index and int(activity.loc[int(p)]) >= args.min_pairs])
    if cand.size < args.reference_players:
        raise SystemExit(f"only {cand.size} candidates with >= {args.min_pairs} pairs")
    pos = {int(p): i for i, p in enumerate(entity_ids)}
    ref = select_refs(cand, bu[[pos[int(p)] for p in cand]],
                      np.array([int(activity.loc[int(p)]) for p in cand]),
                      args.reference_players, args.seed)
    ref_idx = np.array([pos[int(p)] for p in ref], dtype=np.int64)
    Pref = P[ref_idx]
    print(f"[teacher] reference players {len(ref)}; bu range "
          f"{bu[ref_idx].min():.3f}..{bu[ref_idx].max():.3f}", flush=True)

    charts = sorted(set(train["beatmap_id"].astype(int).tolist()) & set(beatmap_ids.tolist()))
    item_pos = {int(b): i for i, b in enumerate(beatmap_ids)}
    rows = []
    for b in charts:
        i = item_pos[b]
        pred = b0 + bu[ref_idx] + bm[i] + Pref @ C[i]
        centered = pred - pred.mean()
        sd = float(centered.std())
        s = centered / (sd + 1e-9)
        row = {"beatmap_id": int(b), "teacher_mean": float(pred.mean()),
               "teacher_sd": sd, "n_ref": int(len(ref))}
        for j, v in enumerate(s):
            row[f"s_{j}"] = float(v)
        rows.append(row)
    out = pd.DataFrame(rows)
    tm = out["teacher_mean"].to_numpy(np.float64)
    out["teacher_mean_z"] = (tm - tm.mean()) / (tm.std() + 1e-9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)

    # QA
    scols = [c for c in out.columns if c.startswith("s_")]
    corr = [float(np.corrcoef(out[c], out["teacher_mean"])[0, 1])
            for c in scols if out[c].std() > 0 and out["teacher_mean"].std() > 0]
    report = {
        "train_table": str(args.train_table),
        "mf_npz": str(npz_path),
        "out": str(args.out),
        "n_charts": int(len(out)),
        "n_ref": int(len(ref)),
        "ref_ids": [int(x) for x in ref],
        "ref_bu_min": float(bu[ref_idx].min()),
        "ref_bu_max": float(bu[ref_idx].max()),
        "signature_dim": len(scols),
        "signature_std_mean": float(out[scols].std().mean()),
        "teacher_mean_std": float(out["teacher_mean"].std()),
        "max_abs_corr_s_with_teacher_mean": float(np.max(np.abs(corr))) if corr else None,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    pred_glob = sorted(out_dir.glob("*_preds_loss.npz"))
    if pred_glob:
        report["teacher_holdout"] = holdout_rank_nll(pred_glob[-1])
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=1, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
