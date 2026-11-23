"""Encode charts with a trained student encoder -> chart content parquet.

Contract matches data/processed/chart_content_cm3p_local.parquet so that
scripts/step2_cm3p_rank.py / run_p2_p4_multiseed.py can consume it directly:
    beatmap_id, embedding (512-d float32, L2-normalised), n_windows,
    checksum_ok, source, d2v_rev
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .charts import ChartStore
from .config import load_config
from .data import ChartWindowDataset, collate_windows
from .model import Data2VecModel
from .utils import git_rev, repo_root, resolve_device


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--osu-dir", default=None)
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="d2v_main_audio_style")
    ap.add_argument("--index-cache", default=None, help="path to chart_index JSON")
    ap.add_argument("--no-index-cache", action="store_true")
    return ap.parse_args(argv)


@torch.no_grad()
def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.osu_dir:
        cfg["data"]["osu_dir"] = args.osu_dir
    device = resolve_device(args.device or str(cfg["train"].get("device", "auto")))
    ids = None
    if args.ids_file:
        ids = [int(x) for x in Path(args.ids_file).read_text().split() if x.strip()]
    if args.limit:
        ids = (ids or ChartStore(cfg["data"]["osu_dir"]).ids)[: args.limit]
    store = ChartStore(cfg["data"]["osu_dir"], ids=ids)
    ds = ChartWindowDataset(store, cfg, train=False, index_cache=args.index_cache,
                            use_index_cache=not args.no_index_cache)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_windows)
    model = Data2VecModel(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    n_windows: dict[int, int] = {}
    for batch in loader:
        x = batch["x"].to(device)
        valid_in = batch["valid_len"].to(device)
        t_out = (x.size(2) + 1) // 2
        valid_out = (valid_in + 1) // 2
        valid_mask = torch.arange(t_out, device=device)[None, :] < valid_out[:, None]
        out, _ = model.forward_student(x, valid_mask, None)
        w = valid_mask.float().unsqueeze(-1)
        emb = (out.float() * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        emb = emb.cpu().numpy().astype(np.float32)
        for cid, vec, ww in zip(batch["chart_id"].tolist(), emb, valid_out.tolist()):
            if ww <= 0:
                continue
            sums[cid] = sums.get(cid, 0) + vec * float(ww)
            counts[cid] = counts.get(cid, 0) + int(ww)
            n_windows[cid] = n_windows.get(cid, 0) + 1
    if not sums:
        raise SystemExit("no embeddings produced")
    rows = []
    rev = git_rev(repo_root())
    for cid in sorted(sums):
        vec = sums[cid] / max(1, counts[cid])
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        rows.append({
            "beatmap_id": int(cid),
            "embedding": vec.astype(np.float32),
            "n_windows": int(n_windows[cid]),
            "checksum_ok": True,
            "source": args.tag,
            "d2v_rev": rev,
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"[extract] wrote {len(rows)} charts -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
