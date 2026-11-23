#!/usr/bin/env python
"""Encode benchmark charts with the frozen CM3P model (legacy or canon crop).

Row-keyed: one JSONL record per benchmark row_id.  This is deliberately not
keyed by beatmap_id because the benchmark contains rate variants that share a
bid but must keep separate content vectors.

    python scripts/benchmark_encode.py --crop legacy --shard 0 --num-shards 1
    python scripts/benchmark_encode.py --assemble --crop legacy
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_probe():
    path = ROOT / "scripts/cm3p_official_probe.py"
    spec = importlib.util.spec_from_file_location("cm3p_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def done_ids(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("vector") is not None:
                done.add(str(rec["row_id"]))
    return done


def run(args) -> int:
    import torch

    probe = load_probe()
    if args.threads:
        torch.set_num_threads(int(args.threads))
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    manifest = pd.read_parquet(args.manifest)
    if args.limit:
        manifest = manifest.head(int(args.limit))
    manifest = manifest.sort_values("row_id").reset_index(drop=True)
    shard = manifest.iloc[int(args.shard)::int(args.num_shards)].reset_index(drop=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.crop}_shard_{int(args.shard):02d}.jsonl"

    done = done_ids(out_path)
    todo = shard[~shard["row_id"].astype(str).isin(done)].reset_index(drop=True)
    print(f"[{args.crop}] shard {args.shard}/{args.num_shards}: "
          f"{len(shard)} assigned, {len(done)} done, {len(todo)} todo", flush=True)
    if todo.empty:
        return 0

    processor, model, device, dtype = probe.load_model(
        Path(args.model_dir), args.device, args.dtype, args.attn)
    snap = probe.processor_snapshot(processor)
    print("[processor] " + json.dumps(snap, ensure_ascii=False), flush=True)

    osu_root = Path(args.osu_root)
    t0 = time.time()
    n_ok = n_err = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for i, row in todo.iterrows():
            rid = str(row["row_id"])
            path = osu_root / str(row["osu_path"])
            rec = {"row_id": rid, "beatmap_id": str(row.get("bid", "")),
                   "name": row.get("name"), "pattern": row.get("pattern"),
                   "family": row.get("family"), "crop": args.crop,
                   "audio": "off", "pad_mode": args.pad_mode,
                   "max_length": int(args.max_length), "dtype": args.dtype,
                   "model_dir": str(args.model_dir),
                   "cm3p_rev": "OliBomby/CM3P@26cc694"}
            if not path.exists():
                rec["error"] = f"missing .osu: {path}"
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_err += 1
                continue
            try:
                res = probe.encode_chart(processor, model, path, None, device, dtype,
                                         args.pad_mode, int(args.win_batch),
                                         int(args.max_length), args.crop)
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"encode: {type(exc).__name__}: {exc}"
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_err += 1
                print(f"[fail] {rid}: {rec['error']}", flush=True)
                continue
            if res.get("error"):
                rec["error"] = res["error"]
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_err += 1
                continue
            vec = np.asarray(res.pop("vector"), dtype=np.float32)
            rec["vector"] = [round(float(x), 7) for x in vec]
            rec.update({k: v for k, v in res.items() if k != "tok_len"})
            rec["error"] = None
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_ok += 1
            if (n_ok + n_err) % 25 == 0:
                print(f"  {n_ok + n_err}/{len(todo)} ok={n_ok} err={n_err} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    print(f"[done] {out_path}: ok={n_ok} err={n_err} ({time.time() - t0:.0f}s)", flush=True)
    return 0


def assemble(args) -> int:
    import glob

    rows = []
    for path in sorted(glob.glob(str(Path(args.out_dir) / f"{args.crop}_shard_*.jsonl"))):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        raise SystemExit("no shard rows")
    recs = []
    for rec in rows:
        if rec.get("vector") is None:
            continue
        recs.append({
            "row_id": rec["row_id"],
            "bid": rec.get("beatmap_id"),
            "embedding": np.asarray(rec["vector"], dtype=np.float32),
            "n_windows": rec.get("n_win"),
            "n_truncated_windows": rec.get("n_truncated_windows"),
            "tok_mean": rec.get("tok_mean"),
            "crop": rec.get("crop"),
            "audio": rec.get("audio", "off"),
            "source": 1,
            "cm3p_rev": rec.get("cm3p_rev"),
            "encode_sec": rec.get("encode_sec"),
            "forward_sec": rec.get("forward_sec"),
        })
    df = pd.DataFrame(recs).drop_duplicates("row_id")
    out = Path(args.out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    meta = {
        "crop": args.crop,
        "n_rows": int(len(df)),
        "n_shards": len(glob.glob(str(Path(args.out_dir) / f"{args.crop}_shard_*.jsonl"))),
        "n_missing": int(len(pd.read_parquet(args.manifest)) - len(df)),
        "model_dir": str(args.model_dir),
        "config": {"crop": args.crop, "audio": "off", "pad_mode": args.pad_mode,
                   "max_length": args.max_length, "dtype": args.dtype},
        "cm3p_rev": "OliBomby/CM3P@26cc694",
    }
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    print(f"[assemble] {out}: {len(df)} rows", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/processed/benchmark_manifest.parquet")
    ap.add_argument("--osu-root", default=".cache/benchmark/VSRG-DanEstimation-Benchmark/extracted")
    ap.add_argument("--out-dir", default="results/benchmark/encode")
    ap.add_argument("--out-parquet", default=None)
    ap.add_argument("--crop", choices=["legacy", "canon"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model-dir", default=os.environ.get("CM3P_MODEL_DIR",
                                                          "data/raw/cm3p_model_26cc694"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--pad-mode", default="longest", choices=["longest", "max_length"])
    ap.add_argument("--win-batch", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--assemble", action="store_true")
    args = ap.parse_args(argv)
    if args.out_parquet is None:
        args.out_parquet = f"data/processed/benchmark_content_{args.crop}.parquet"
    if args.assemble:
        return assemble(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
