"""Validate the learned player side: does b_u / P track actual player skill?"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

INTERIM = Path("data/interim")
PROC = Path("data/processed")


def main(emb_path):
    emb = np.load(emb_path, allow_pickle=True)
    user_ids = emb["user_ids"].astype("int64")
    bu, P = emb["bu"], emb["P"]
    st = pd.read_csv(INTERIM / "user_stats_mania.csv")
    st["rank_score_index"] = pd.to_numeric(st.rank_score_index, errors="coerce")
    st["accuracy"] = pd.to_numeric(st.accuracy, errors="coerce")
    st["playcount"] = pd.to_numeric(st.playcount, errors="coerce")
    d = pd.DataFrame({"user_id": user_ids, "bu": bu}).merge(st, on="user_id", how="left")
    print(f"players: {len(d)}")
    out = {}
    for col in ["rank_score_index", "accuracy", "playcount", "rank"]:
        v = pd.to_numeric(d[col], errors="coerce")
        m = v.notna()
        rho = spearmanr(d.bu[m], v[m]).statistic
        out[f"spearman_bu_vs_{col}"] = float(rho)
        print(f"spearman(b_u vs {col}) = {rho:+.4f}")
    # norm of P: better players should need smaller |P|
    d["Pnorm"] = np.linalg.norm(P, axis=1)
    for col in ["rank_score_index", "accuracy"]:
        v = pd.to_numeric(d[col], errors="coerce")
        m = v.notna()
        print(f"spearman(|P| vs {col}) = {spearmanr(d.Pnorm[m], v[m]).statistic:+.4f}")
    (PROC / f"player_validation_{Path(emb_path).stem}.json").write_text(json.dumps(out, indent=2))
    print("saved")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(PROC / "stage1_mirt_loss_v2_med.npz"))
