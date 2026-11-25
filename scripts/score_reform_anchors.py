#!/usr/bin/env python
"""Compute model raw scores for the Reform Dan anchor manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import step2_cm3p_rank as rank_model  # noqa: E402


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _embedding_lookup(path: Path, ids: list[int]) -> tuple[dict[int, np.ndarray], list[int]]:
    table = pd.read_parquet(path)
    if "beatmap_id" not in table or "embedding" not in table:
        raise SystemExit(f"{path}: expected beatmap_id and embedding columns")
    table = table.drop_duplicates("beatmap_id", keep="last").set_index("beatmap_id")
    found = [bid for bid in ids if bid in table.index]
    vectors = {bid: np.asarray(table.at[bid, "embedding"], dtype=np.float32) for bid in found}
    return vectors, [bid for bid in ids if bid not in vectors]


def _score_benchmark_payload(payload: dict, C: np.ndarray,
                              signals: pd.DataFrame) -> np.ndarray:
    mu = np.asarray(payload["mu"], dtype=np.float32)
    sd = np.asarray(payload["sd"], dtype=np.float32)
    Cs = (C - mu) / sd
    names = payload.get("extra_names") or []
    if names:
        missing = [name for name in names if name not in signals]
        if missing:
            raise SystemExit(f"signal table missing {missing}")
        S = signals[names].to_numpy(dtype=np.float64)
        if not np.isfinite(S).all():
            raise SystemExit("extra signal has NaN/inf")
        S = (S - np.asarray(payload["extra_mu"], dtype=np.float64)) / \
            np.asarray(payload["extra_sd"], dtype=np.float64)
        X = np.hstack([Cs, S]).astype(np.float32)
    else:
        X = Cs.astype(np.float32)

    kind = payload["model"]
    if kind in ("linear", "linear_extra", "ext_blend"):
        d_raw = X.astype(np.float64) @ np.asarray(payload["w"], dtype=np.float64)
    else:
        scorer = rank_model.Scorer(X.shape[1], int(payload["hid"]),
                                   int(payload["depth"]), float(payload["dropout"]))
        scorer.model.load_state_dict(payload["state_dict"])
        scorer.model.eval()
        with torch.no_grad():
            d_raw = scorer.forward(torch.as_tensor(X, dtype=torch.float32)).cpu().numpy()
        d_raw = d_raw.astype(np.float64)
    d_mu, d_sd = float(payload.get("d_mu_raw", 0.0)), float(payload.get("d_sd_raw", 1.0))
    return (d_raw - d_mu) / (d_sd if d_sd else 1.0)


def _score_deploy_linear(payload: dict, C: np.ndarray,
                         signals: pd.DataFrame) -> np.ndarray:
    d = ((C.astype(np.float64) - np.asarray(payload["mu"], dtype=np.float64)) /
         np.asarray(payload["sd"], dtype=np.float64)) @ np.asarray(payload["w"][:C.shape[1]],
                                                                      dtype=np.float64)
    names = payload.get("extra_names") or []
    if names:
        S = signals[names].to_numpy(dtype=np.float64)
        d += ((S - np.asarray(payload["extra_mu"], dtype=np.float64)) /
              np.asarray(payload["extra_sd"], dtype=np.float64)) @ \
             np.asarray(payload["w"][C.shape[1]:], dtype=np.float64)
    return (d - float(payload.get("d_mu_raw", 0.0))) / \
        (float(payload.get("d_sd_raw", 1.0)) or 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cm3p-content", type=Path,
                    default=ROOT / "data/reform/chart_content_cm3p_local.parquet")
    ap.add_argument("--signals", type=Path, default=ROOT / "data/reform/signals.parquet")
    ap.add_argument("--d2v-content", type=Path, default=None,
                    help="optional beatmap_id/embedding table for the DCR method")
    ap.add_argument("--dcr-model", type=Path,
                    default=ROOT / "deploy/models/d2v_a20d_linear_extra.json")
    ap.add_argument("--rm-model", type=Path, default=ROOT / "results/benchmark/models/RM_L_full.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "data/reform/raw_scores.csv")
    args = ap.parse_args()

    anchor_map = pd.read_json(ROOT / "data/reform/feature_id_map.json")
    signals = pd.read_parquet(args.signals).set_index("row_id")
    anchor_map = anchor_map.set_index("anchor_id")
    if set(anchor_map.index) != set(signals.index):
        raise SystemExit("anchor id mismatch between feature map and signal table")
    anchor_map = anchor_map.loc[signals.index]
    ids = anchor_map["feature_beatmap_id"].astype(int).tolist()
    cm3p, cm3p_missing = _embedding_lookup(args.cm3p_content, ids)
    cm3p_have = [bid for bid in ids if bid in cm3p]
    out = pd.DataFrame({
        "anchor_id": anchor_map.index,
        "difficulty_key": anchor_map["difficulty_key"].to_numpy(),
        "target": anchor_map["target"].to_numpy(float),
        "feature_beatmap_id": ids,
        "Sunny": signals.loc[anchor_map.index, "sunny_star"].to_numpy(float),
        "DCR": np.nan,
        "CM3P_ext": np.nan,
    })

    if cm3p_have:
        rows = np.asarray([cm3p[bid] for bid in cm3p_have], dtype=np.float32)
        signal_rows = signals.loc[anchor_map.index[[ids.index(bid) for bid in cm3p_have]]]
        rm = torch.load(args.rm_model, map_location="cpu", weights_only=False)
        rm_scores = _score_benchmark_payload(rm, rows, signal_rows)
        score_at = {bid: i for i, bid in enumerate(ids)}
        for bid, val in zip(cm3p_have, rm_scores):
            out.loc[score_at[bid], "CM3P_ext"] = float(val)

    dcr_model_path = args.dcr_model
    dcr_model_id = None
    if args.d2v_content is not None and dcr_model_path.exists():
        d2v, _ = _embedding_lookup(args.d2v_content, ids)
        d2v_have = [bid for bid in ids if bid in d2v]
        if d2v_have:
            payload = json.loads(dcr_model_path.read_text(encoding="utf-8"))
            dcr_model_id = payload.get("model_id")
            rows = np.asarray([d2v[bid] for bid in d2v_have], dtype=np.float32)
            signal_rows = signals.loc[anchor_map.index[[ids.index(bid) for bid in d2v_have]]]
            values = _score_deploy_linear(payload, rows, signal_rows)
            score_at = {bid: i for i, bid in enumerate(ids)}
            for bid, val in zip(d2v_have, values):
                out.loc[score_at[bid], "DCR"] = float(val)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    report = {
        "n_anchors": int(len(out)), "sunny": int(out["Sunny"].notna().sum()),
        "cm3p_rm_l": int(out["CM3P_ext"].notna().sum()),
        "dcr": int(out["DCR"].notna().sum()), "cm3p_missing_feature_ids": cm3p_missing,
        "model_ids": {"Sunny": "SunnyRework", "CM3P_ext": "RM_L",
                      "DCR": dcr_model_id},
        "out": str(args.out),
        "inputs_sha256": {
            "anchor_feature_map": sha256_file(ROOT / "data/reform/feature_id_map.json"),
            "signals": sha256_file(args.signals),
            "cm3p_content": sha256_file(args.cm3p_content),
            "cm3p_content_meta": sha256_file(
                args.cm3p_content.with_suffix(args.cm3p_content.suffix + ".meta.json")),
            "rm_model": sha256_file(args.rm_model),
            "d2v_content": sha256_file(args.d2v_content) if args.d2v_content else None,
            "dcr_model": sha256_file(dcr_model_path),
        },
    }
    args.out.with_suffix(args.out.suffix + ".meta.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
