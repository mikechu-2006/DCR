#!/usr/bin/env python
"""Audio-on CM3P embeddings for the todo charts, using the config validated by
the official probe (scripts/cm3p_official_probe.py, mean cos 0.9975 vs the
published vectors).

Same pipeline as the probe:
    processor(beatmap=<.osu>, audio=<full song>, audio_sampling_rate=16000,
              padding=longest, truncation=longest_first, max_length=2000)
    per-window vector = normalize(beatmap_projection(pooled_output))
    chart vector      = normalize(mean(window vectors))

Each shard appends one JSON line per chart, INCLUDING the 512-d vector, so a
killed job loses at most the chart in flight.  Shards are independent; assemble
them into the repo's content-table schema with --assemble.

    # one array task per shard (i of N)
    python scripts/cm3p_embed_todo.py --shard 0 --num-shards 8 \
        --ids-csv data/processed/cm3p_ids_todo_2786.csv \
        --audio-dir data/todo_audio_hf --out-dir results/cm3p_todo_official/shards

    # assemble (login node; no model needed)
    python scripts/cm3p_embed_todo.py --assemble \
        --ids-csv data/processed/cm3p_ids_todo_2786.csv \
        --in-dir results/cm3p_todo_official/shards \
        --out-parquet data/processed/chart_content_cm3p_local_audio.parquet
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

AUDIO_EXTS = (".mp3", ".ogg", ".wav", ".flac", ".m4a")
SRC_LOCAL_AUDIO = 2
EMBED_DIM = 512


def log(msg: str) -> None:
    print(msg, flush=True)


def load_probe_module():
    path = Path(__file__).with_name("cm3p_official_probe.py")
    spec = importlib.util.spec_from_file_location("cm3p_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shard_rows(df: pd.DataFrame, shard: int, num_shards: int) -> pd.DataFrame:
    df = df.sort_values("beatmap_id").reset_index(drop=True)
    return df.iloc[shard::num_shards].reset_index(drop=True)


def read_done(path: Path) -> set[int]:
    done: set[int] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line from a killed job
            if rec.get("vector") is not None:
                done.add(int(rec["beatmap_id"]))
    return done


def run_embed(args) -> None:
    import torch

    probe = load_probe_module()
    if args.threads:
        torch.set_num_threads(args.threads)
    out_path = Path(args.out_dir) / f"shard_{args.shard:02d}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.ids_csv)
    df = df.drop_duplicates("beatmap_id")
    df = shard_rows(df, args.shard, args.num_shards)
    if args.limit:
        df = df.head(args.limit)
    done = read_done(out_path)
    if done:
        log(f"resume: {len(done)} chart(s) already in {out_path}")
    todo = df[~df.beatmap_id.isin(done)].reset_index(drop=True)
    log(f"shard {args.shard}/{args.num_shards}: {len(df)} chart(s), "
        f"{len(todo)} to compute")
    if todo.empty:
        return

    processor, model, device, dtype = probe.load_model(
        Path(args.model_dir), args.device, args.dtype, args.attn)
    snap = probe.processor_snapshot(processor)
    log("processor snapshot: " + json.dumps(snap, ensure_ascii=False))
    audio_dirs = [Path(p) for p in args.audio_dir.split(",") if p] if args.audio_dir else []
    osu_dir = Path(args.osu_dir)

    n_ok = n_err = 0
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as fh:
        for i, row in todo.iterrows():
            bid = int(row["beatmap_id"])
            setid = int(row["beatmapset_id"]) if "beatmapset_id" in todo.columns else -1
            rec = {"beatmap_id": bid, "beatmapset_id": setid,
                   "dtype": args.dtype, "pad_mode": args.pad_mode,
                   "crop": args.crop, "audio": "on",
                   "audio_loader": args.audio_loader,
                   "max_length": args.max_length,
                   "model_dir": str(args.model_dir)}
            osu_path = osu_dir / f"{bid}.osu"
            if not osu_path.exists():
                rec.update(error="no .osu in the cache")
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_err += 1
                log(f"[skip] {bid}: no .osu")
                continue
            if "checksum" in todo.columns:
                rec["checksum_ok"] = probe.md5(osu_path) == str(row["checksum"])
            audio_path = None
            for d in audio_dirs:
                audio_path = probe.find_audio(d, bid)
                if audio_path is not None:
                    break
            if audio_path is None:
                rec.update(error="no staged audio")
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_err += 1
                log(f"[skip] {bid}: no staged audio")
                continue
            try:
                if args.audio_loader == "ffmpeg":
                    audio = probe.decode_audio_ffmpeg(audio_path)
                else:
                    from transformers.audio_utils import load_audio
                    audio = np.asarray(load_audio(str(audio_path), sampling_rate=16000),
                                       dtype=np.float32)
            except Exception as exc:  # noqa: BLE001
                # the file may be a container libsndfile refuses; ffmpeg is the fallback
                try:
                    audio = probe.decode_audio_ffmpeg(audio_path)
                    rec["audio_loader"] = "ffmpeg-fallback"
                except Exception as exc2:  # noqa: BLE001
                    rec.update(error=f"audio decode: {type(exc).__name__}: {exc} / "
                                     f"ffmpeg: {type(exc2).__name__}: {exc2}")
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_err += 1
                    log(f"[fail] {bid}: decode")
                    continue
            rec["audio_file"] = audio_path.name
            rec["audio_bytes"] = audio_path.stat().st_size
            try:
                res = probe.encode_chart(processor, model, osu_path, audio, device,
                                         dtype, args.pad_mode, args.win_batch,
                                         args.max_length, args.crop)
            except Exception as exc:  # noqa: BLE001
                rec.update(error=f"encode: {type(exc).__name__}: {exc}")
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_err += 1
                log(f"[fail] {bid}: encode: {exc}")
                continue
            if res.get("error"):
                rec.update(error=res["error"])
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_err += 1
                log(f"[skip] {bid}: {res['error']}")
                continue
            vec = res.pop("vector")
            rec["vector"] = [round(float(x), 7) for x in vec]
            rec.update({k: v for k, v in res.items() if k != "tok_len"})
            rec["error"] = None
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_ok += 1
            if n_ok % 20 == 0 or n_ok < 5:
                el = time.time() - t0
                log(f"[{n_ok}/{len(todo)}] {bid} set={setid} win={rec['n_win']} "
                    f"tok={rec['tok_mean']:.0f} enc={rec['encode_sec']}s "
                    f"fwd={rec['forward_sec']}s ({el / max(n_ok, 1):.1f}s/chart)")
    log(f"shard {args.shard} done: ok={n_ok} err={n_err} "
        f"elapsed={time.time() - t0:.0f}s -> {out_path}")


def assemble(args) -> None:
    in_dir = Path(args.in_dir)
    recs: dict[int, dict] = {}
    for path in sorted(in_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("vector") is None or rec.get("error"):
                continue
            recs[int(rec["beatmap_id"])] = rec      # newest wins
    todo = pd.read_csv(args.ids_csv)
    ids = [int(x) for x in todo["beatmap_id"]]
    have = [i for i in ids if i in recs]
    log(f"assembled {len(have):,}/{len(ids):,} charts "
        f"({len(have) / max(len(ids), 1):.1%})")
    embs = [np.asarray(recs[i]["vector"], dtype=np.float32) for i in have]
    for e in embs:
        if e.shape != (EMBED_DIM,):
            raise SystemExit(f"bad embedding shape {e.shape}")
    checksum = {(int(r.beatmap_id)): recs[int(r.beatmap_id)].get("checksum_ok")
                for _, r in todo.iterrows() if int(r.beatmap_id) in recs}
    table = pd.DataFrame({
        "beatmap_id": np.asarray(have, dtype=np.int64),
        "embedding": embs,
        "n_windows": pd.array([int(recs[i]["n_win"]) for i in have], dtype="Int16"),
        "checksum_ok": pd.array([checksum.get(i) for i in have], dtype="boolean"),
        "source": np.asarray([SRC_LOCAL_AUDIO] * len(have), dtype=np.int8),
        "cm3p_rev": np.asarray(["OliBomby/CM3P@26cc694"] * len(have), dtype=object),
    })
    out = Path(args.out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    missing = sorted(set(ids) - set(have))
    meta = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ids_csv": str(args.ids_csv),
        "in_dir": str(in_dir),
        "n_target": len(ids),
        "n_rows": len(have),
        "coverage": len(have) / max(len(ids), 1),
        "n_missing": len(missing),
        "missing_sample": missing[:20],
        "source": SRC_LOCAL_AUDIO,
        "source_name": "local-audio",
        "config": {
            "audio": "on", "crop": args.crop, "pad_mode": args.pad_mode,
            "max_length": args.max_length, "dtype": args.dtype,
            "model": "OliBomby/CM3P@26cc694",
        },
        "probe_reference": "results/cm3p_probe_official/summary.json "
                           "(mean cos 0.99745, median 0.99809, 95/98 >= 0.99)",
    }
    (out.parent / (out.name + ".meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"wrote {out} ({len(have):,} rows)")
    log(f"missing {len(missing):,} chart(s): {missing[:10]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids-csv", default="data/processed/cm3p_ids_todo_2786.csv")
    ap.add_argument("--osu-dir", default="data/osu_cache")
    ap.add_argument("--audio-dir", default="data/todo_audio_hf")
    ap.add_argument("--model-dir", default=os.environ.get("CM3P_MODEL_DIR",
                                                          "data/raw/cm3p_model_26cc694"))
    ap.add_argument("--out-dir", default="results/cm3p_todo_official/shards")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--pad-mode", default="longest", choices=["longest", "max_length"])
    ap.add_argument("--win-batch", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=2000)
    ap.add_argument("--crop", default="legacy", choices=["legacy", "canon"])
    ap.add_argument("--audio-loader", default="librosa", choices=["librosa", "ffmpeg"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--in-dir", default="results/cm3p_todo_official/shards")
    ap.add_argument("--out-parquet",
                    default="data/processed/chart_content_cm3p_local_audio.parquet")
    args = ap.parse_args()
    if args.assemble:
        assemble(args)
    else:
        run_embed(args)


if __name__ == "__main__":
    main()
