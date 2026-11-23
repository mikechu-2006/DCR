#!/usr/bin/env python
"""CPU smoke test for the from-scratch audio-style data2vec pipeline.

Runs on 3 real charts with the tiny config (dim=64, 2 layers).  It checks the
parts that are easy to get silently wrong: piano-roll channels, padding-safe
span masking, student/teacher shapes, EMA movement, checkpoint round-trip and
the extracted parquet contract.

    cd data2vec_main && python scripts/smoke_test.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data2vec_main"))

from d2v_main import extract, index_cache, train  # noqa: E402
from d2v_main.charts import ChartStore, build_roll, parse_chart  # noqa: E402
from d2v_main.config import load_config  # noqa: E402
from d2v_main.data import ChartWindowDataset, collate_windows  # noqa: E402
from d2v_main.masking import mask_ratio, span_mask  # noqa: E402
from d2v_main.model import Data2VecModel, EMAState  # noqa: E402


def main() -> int:
    t0 = time.time()
    osu = ROOT / "data/osu_cache"
    all_ids = ChartStore(osu).ids
    ids = all_ids[:3]
    cfg = load_config(ROOT / "data2vec_main/configs/audio_style_smoke.yaml",
                      {"data": {"osu_dir": str(osu)}})
    store = ChartStore(osu, ids=ids)

    for cid in ids:
        events, n_frames = parse_chart(str(store.path(cid)), 100)
        assert len(events) > 0 and n_frames > 0, cid
        assert all(0 <= e.col < 4 for e in events), cid
    events, n_frames = parse_chart(str(store.path(ids[0])), 100)
    roll = build_roll(events, n_frames, 100, 8.0)
    assert roll.shape == (8, n_frames), roll.shape
    assert roll[:4].max() > 0 and roll[4:].max() > 0
    assert np.isfinite(roll).all()
    print(f"[ok] piano roll {roll.shape}, onset={int((roll[:4] > 0).sum())}, "
          f"hold={int((roll[4:] > 0).sum())}")

    # padding must never be masked
    valid = torch.tensor([10, 5])
    mask = span_mask(valid, 20, 0.5, 4)
    assert not bool(mask[0, 10:].any()) and not bool(mask[1, 5:].any())
    print(f"[ok] span mask padding-safe, ratio={float(mask_ratio(mask, valid)):.2f}")

    idx_path = ROOT / "data2vec_main/runs/smoke/index.json"
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx1 = index_cache.load(store, 100, idx_path, force=True)
    idx2 = index_cache.load(store, 100, idx_path)
    assert idx1 == idx2 and all(idx1[cid] > 0 for cid in ids)
    print(f"[ok] index cache {len(idx1)} charts")

    ds = ChartWindowDataset(store, cfg, train=True, use_index_cache=False)
    batch = collate_windows([ds[i] for i in range(len(ds))])
    model = Data2VecModel(cfg)
    x, vin = batch["x"], batch["valid_len"]
    t_out = (x.size(2) + 1) // 2
    vout = (vin + 1) // 2
    vm = torch.arange(t_out)[None, :] < vout[:, None]
    m = span_mask(vout, t_out, cfg["model"]["mask_prob"], cfg["model"]["mask_length"]) & vm
    out, layers = model.forward_student(x, vm, m)
    _, t_layers = model.forward_teacher(x, vm)
    target = model.target(t_layers, m)
    pred = model.predict(out, m)
    loss = torch.nn.functional.smooth_l1_loss(pred, target, beta=model.loss_beta)
    loss.backward()
    assert torch.isfinite(loss)
    assert target.shape == pred.shape and target.size(0) == int(m.sum())
    print(f"[ok] model forward/backward loss={float(loss.detach()):.4f} "
          f"params={model.parameter_count()}")

    ema = EMAState(model.teacher)
    before = [p.detach().clone() for p in model.teacher.parameters()]
    ema.update(model.teacher, model.student, 0.9)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.teacher.parameters()))
    assert all(torch.allclose(t, m.to(t.dtype)) for t, m in zip(model.teacher.parameters(), ema.master))
    print("[ok] fp32 EMA teacher moved and matches master")

    out_dir = ROOT / "data2vec_main/runs/smoke"
    shutil.rmtree(out_dir, ignore_errors=True)
    train.main(["--config", str(ROOT / "data2vec_main/configs/audio_style_smoke.yaml"),
                "--osu-dir", str(osu), "--out-dir", str(out_dir),
                "--limit-charts", "3", "--max-steps", "3", "--epochs", "2",
                "--batch-size", "1", "--num-workers", "0", "--device", "cpu",
                "--no-index-cache"])
    metrics = [json.loads(x) for x in (out_dir / "metrics.jsonl").read_text().splitlines()]
    assert metrics and all(np.isfinite(m["loss"]) for m in metrics)
    assert metrics[-1]["target_var"] > 0.01
    print(f"[ok] train 3 steps, last loss={metrics[-1]['loss']:.4f}")

    parquet = out_dir / "content.parquet"
    extract.main(["--config", str(ROOT / "data2vec_main/configs/audio_style_smoke.yaml"),
                  "--ckpt", str(out_dir / "checkpoints/last.pt"), "--out", str(parquet),
                  "--osu-dir", str(osu), "--limit", "3", "--batch-size", "2",
                  "--num-workers", "0", "--device", "cpu", "--no-index-cache"])
    df = pd.read_parquet(parquet)
    assert list(df.columns) == ["beatmap_id", "embedding", "n_windows", "checksum_ok",
                                "source", "d2v_rev"]
    assert len(df) == 3 and (df.n_windows > 0).all()
    norms = [float(np.linalg.norm(v)) for v in df.embedding]
    assert all(abs(n - 1.0) < 1e-4 for n in norms), norms
    print(f"[ok] extract {len(df)} charts, norms={[round(n, 4) for n in norms]}")
    print(f"SMOKE OK in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
