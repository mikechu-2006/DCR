#!/usr/bin/env python
"""Probe the official CM3P embeddings: reproduce published 512-d vectors.

Target: the published table OliBomby/CM3P-Embeddings-244K (audio ON), filtered in
this repo to data/processed/chart_content_cm3p.parquet.  The question the probe
answers is: with the checkpoint's own processor configuration, can a local
re-encode reach cosine >= 0.99 against the published vector for a random sample
of charts?

What is reproduced (exactly the upstream extract_beatmap_embeddings.py path):

    processor(beatmap=<.osu>, audio=<full song>, audio_sampling_rate=16000,
              speed=1.0, padding=MAX_LENGTH, truncation=LONGEST_FIRST,
              max_length=2000)                  # checkpoint's processor_config
    model(input_ids, attention_mask, input_features)
    per-window vector = normalize(beatmap_projection(pooled_output))
    chart vector      = normalize(mean(per-window vectors))

Two deliberate implementation choices that do NOT change the math:

  * padding=LONGEST instead of MAX_LENGTH.  Padding is masked, so valid tokens
    see the same attention; this only avoids computing 2000 - len tokens of
    padding per window.  --pad-mode max_length restores the literal upstream
    padding (and is the control for a padding-sensitive bug).
  * windows are forwarded in mini-batches of --win-batch (same values, less RAM).

The audio files are staged by scripts/cm3p_probe_fetch_audio.py on a node with
network; this script never downloads anything, so it can run on an offline
compute node.

    python scripts/cm3p_official_probe.py \
        --sample results/cm3p_probe_official/sample.csv \
        --audio-dir /hpc2hdd/$USER/DCR/data/probe_audio \
        --model-dir data/raw/cm3p_model_26cc694 \
        --out results/cm3p_probe_official/probe.jsonl \
        --summary results/cm3p_probe_official/summary.json

--audio off skips the audio (and the audio_dir) and is the sanity check against
the local no-audio table: expect cosine ~0.99999 there, not against the published
table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AUDIO_EXTS = (".mp3", ".ogg", ".wav", ".flac", ".m4a")


def log(msg: str) -> None:
    print(msg, flush=True)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_sample(ids_csv: Path, n: int, seed: int, out_csv: Path | None) -> pd.DataFrame:
    df = pd.read_csv(ids_csv)
    need = {"beatmap_id", "beatmapset_id", "checksum"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{ids_csv}: expected columns {sorted(need)}")
    if "official_checksum_ok" in df.columns:
        n_all = len(df)
        df = df[df["official_checksum_ok"].fillna(False).astype(bool)]
        log(f"sample: {len(df)}/{n_all} rows have official_checksum_ok=True")
    # one chart per beatmapset keeps the number of .osz downloads equal to n
    df = df.sample(frac=1.0, random_state=seed)
    df = df.drop_duplicates("beatmapset_id", keep="first").head(n)
    df = df.sort_values("beatmap_id").reset_index(drop=True)
    if len(df) < n:
        raise SystemExit(f"only {len(df)} distinct beatmapsets available, wanted {n}")
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        log(f"sample written: {out_csv} ({len(df)} charts, seed={seed})")
    return df


def decode_audio_ffmpeg(path: Path, sampling_rate: int = 16000) -> np.ndarray:
    """Decode to mono float32 with ffmpeg (the newer transformers load_audio path)."""
    import shutil
    import subprocess
    # Prefer the static imageio-ffmpeg binary: a site /usr/bin/ffmpeg can break
    # under the cluster's LD_LIBRARY_PATH (libp11-kit/libffi symbol clash).
    exe = None
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg not found and imageio-ffmpeg unavailable")
    cmd = [exe, "-nostdin", "-v", "error", "-i", str(path),
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1",
           "-ar", str(sampling_rate), "pipe:1"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-500:])
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def find_audio(audio_dir: Path, beatmap_id: int) -> Path | None:
    for ext in AUDIO_EXTS:
        p = audio_dir / f"{beatmap_id}{ext}"
        if p.is_file():
            return p
    return None


def load_model(model_dir: Path, device: str, dtype_name: str, attn: str,
               trust_remote_code: bool = True):
    import torch
    from transformers import AutoModel, AutoProcessor

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[dtype_name]
    log(f"model: {model_dir}  device={device}  dtype={dtype_name}  attn={attn}")
    processor = AutoProcessor.from_pretrained(str(model_dir),
                                              trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code,
                                      torch_dtype=dtype, attn_implementation=attn)
    model.to(device)
    model.eval()
    return processor, model, device, dtype


def processor_snapshot(processor) -> dict:
    from transformers.utils import PaddingStrategy  # noqa: F401
    dk = getattr(processor, "default_kwargs", {}) or {}
    bk = dk.get("beatmap_kwargs", {}) or {}
    ak = dk.get("audio_kwargs", {}) or {}
    parser = getattr(processor, "beatmap_parser", None)
    tok = getattr(processor, "beatmap_tokenizer", None)
    return {
        "window_length_sec": bk.get("window_length_sec"),
        "window_stride_sec": bk.get("window_stride_sec"),
        "max_length": bk.get("max_length"),
        "padding": bk.get("padding"),
        "truncation": bk.get("truncation"),
        "audio_sampling_rate": ak.get("sampling_rate"),
        "audio_pad_to_multiple_of": ak.get("pad_to_multiple_of"),
        "audio_hop_length": ak.get("hop_length"),
        "audio_length_per_tok": ak.get("audio_length_per_tok"),
        "audio_max_source_positions": ak.get("max_source_positions"),
        "parser": {k: getattr(parser, k, None) for k in (
            "add_hitsounds", "add_kiai", "add_mania_sv", "add_snapping",
            "add_timing", "add_timing_points", "add_positions")},
        "tokenizer_max_time": getattr(tok, "max_time", None),
        "tokenizer_time_step": getattr(tok, "time_step", None),
        "tokenizer_add_cls": getattr(tok, "add_cls_token", None),
    }


def encode_chart(processor, model, osu_path: Path, audio: np.ndarray | None,
                 device: str, dtype, pad_mode: str, win_batch: int,
                 max_length: int, crop: str) -> dict:
    import torch
    from transformers.utils import PaddingStrategy
    from transformers.tokenization_utils_base import TruncationStrategy

    if crop == "canon":
        p = processor.beatmap_parser
        p.add_hitsounds = p.add_kiai = p.add_mania_sv = False
    elif crop == "legacy":
        p = processor.beatmap_parser
        p.add_hitsounds = p.add_kiai = p.add_mania_sv = True
    else:
        raise SystemExit(f"unknown --crop {crop}")

    pad = PaddingStrategy.MAX_LENGTH if pad_mode == "max_length" else PaddingStrategy.LONGEST
    t0 = time.time()
    enc = processor(beatmap=str(osu_path), audio=audio, audio_sampling_rate=16000,
                    speed=1.0, padding=pad, truncation=TruncationStrategy.LONGEST_FIRST,
                    max_length=max_length, return_tensors="pt")
    t_enc = time.time() - t0
    ids = enc["input_ids"]
    mask = enc["attention_mask"]
    feats = enc.get("input_features")
    n_win = int(ids.shape[0])
    if n_win == 0:
        return {"n_win": 0, "error": "no windows"}
    tok_len = mask.sum(dim=1).tolist()
    n_trunc = int(sum(1 for v in tok_len if v >= max_length))

    embs: list[np.ndarray] = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n_win, max(1, win_batch)):
            j = min(n_win, i + max(1, win_batch))
            kw = {"input_ids": ids[i:j].to(device),
                  "attention_mask": mask[i:j].to(device),
                  "return_loss": False}
            if feats is not None:
                kw["input_features"] = feats[i:j].to(device=device, dtype=dtype)
            out = model(**kw)
            embs.append(out.beatmap_embeds.detach().float().cpu().numpy())
    t_fwd = time.time() - t0
    win = np.concatenate(embs, axis=0)
    vec = win.mean(axis=0)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return {"n_win": n_win, "vector": vec, "tok_len": tok_len,
            "tok_max": int(max(tok_len)), "tok_mean": float(np.mean(tok_len)),
            "n_truncated_windows": n_trunc,
            "encode_sec": round(t_enc, 2), "forward_sec": round(t_fwd, 2),
            "audio_sec": (len(audio) / 16000.0) if audio is not None else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids-csv", default="data/processed/cm3p_ids_official_4042.csv",
                    help="id list to sample from (needs beatmap_id/beatmapset_id/checksum)")
    ap.add_argument("--sample", default=None,
                    help="fixed sample CSV; if given, --n/--seed are not used")
    ap.add_argument("--sample-out", default="results/cm3p_probe_official/sample.csv")
    ap.add_argument("--n", type=int, default=100, help="charts to sample (one per beatmapset)")
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--official-table", default="data/processed/chart_content_cm3p.parquet",
                    help="published vectors to compare against")
    ap.add_argument("--local-table", default=None,
                    help="optional second target (e.g. the local audio-off table)")
    ap.add_argument("--osu-dir", default="data/osu_cache")
    ap.add_argument("--audio-dir", default=None, help="staged audio; required with --audio on")
    ap.add_argument("--audio", choices=["on", "off"], default="on")
    ap.add_argument("--audio-loader", choices=["librosa", "ffmpeg"], default="librosa",
                    help="librosa = transformers 4.55 load_audio; ffmpeg = newer transformers path")
    ap.add_argument("--model-dir", default=os.environ.get("CM3P_MODEL_DIR",
                                                          "data/raw/cm3p_model_26cc694"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--attn", default="sdpa", help="attn implementation (sdpa/eager)")
    ap.add_argument("--pad-mode", default="longest", choices=["longest", "max_length"],
                    help="'longest' is masked-equivalent and cheaper; 'max_length' is literal upstream")
    ap.add_argument("--win-batch", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=2000)
    ap.add_argument("--crop", default="legacy", choices=["legacy", "canon"],
                    help="legacy = checkpoint defaults (hitsounds/kiai/SV on)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N sample rows")
    ap.add_argument("--out", default="results/cm3p_probe_official/probe.jsonl")
    ap.add_argument("--summary", default="results/cm3p_probe_official/summary.json")
    ap.add_argument("--save-vectors", default=None,
                    help="optional .npz with the reproduced vectors")
    ap.add_argument("--refresh", action="store_true", help="ignore existing --out rows")
    args = ap.parse_args()

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    osu_dir = Path(args.osu_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if args.sample:
        sample = pd.read_csv(args.sample)
        log(f"sample: {args.sample} ({len(sample)} charts)")
    else:
        sample = build_sample(Path(args.ids_csv), args.n, args.seed,
                              Path(args.sample_out) if args.sample_out else None)
    if args.limit:
        sample = sample.head(args.limit)
    sample = sample.reset_index(drop=True)
    for c in ("beatmap_id", "beatmapset_id"):
        if c in sample.columns:
            sample[c] = sample[c].astype("int64")

    official = pd.read_parquet(args.official_table)
    official = official.set_index("beatmap_id")
    local = None
    if args.local_table:
        local = pd.read_parquet(args.local_table).set_index("beatmap_id")

    done: dict[int, dict] = {}
    if out_path.exists() and not args.refresh:
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("error") is None and rec.get("cos_official") is not None:
                done[int(rec["beatmap_id"])] = rec
        if done:
            log(f"resume: {len(done)} chart(s) already in {out_path}")
    if args.refresh and out_path.exists():
        out_path.unlink()

    processor, model, device, dtype = load_model(Path(args.model_dir), args.device,
                                                 args.dtype, args.attn)
    snap = processor_snapshot(processor)
    log("processor snapshot: " + json.dumps(snap, ensure_ascii=False))

    mismatches = []
    if snap["window_length_sec"] not in (16, 16.0):
        mismatches.append(f"window_length_sec={snap['window_length_sec']} != 16")
    if snap["window_stride_sec"] not in (16, 16.0):
        mismatches.append(f"window_stride_sec={snap['window_stride_sec']} != 16")
    if snap["max_length"] not in (2000, None):
        mismatches.append(f"max_length={snap['max_length']} != 2000")
    if snap["parser"].get("add_snapping") not in (False, None):
        mismatches.append(f"add_snapping={snap['parser'].get('add_snapping')} (expected False)")
    if snap["tokenizer_add_cls"] not in (True, None):
        mismatches.append(f"add_cls_token={snap['tokenizer_add_cls']} (expected True)")
    if mismatches:
        log("WARNING: checkpoint processor config differs from the expected snapshot: "
            + "; ".join(mismatches))

    vectors: dict[int, np.ndarray] = {}
    n_ok = n_err = n_skip = 0
    t_start = time.time()
    with out_path.open("a", encoding="utf-8") as fh:
        for k, row in sample.iterrows():
            bid = int(row["beatmap_id"])
            setid = int(row["beatmapset_id"]) if "beatmapset_id" in sample.columns else -1
            if bid in done:
                continue
            osu_path = osu_dir / f"{bid}.osu"
            rec: dict = {"beatmap_id": bid, "beatmapset_id": setid,
                         "dtype": args.dtype, "pad_mode": args.pad_mode,
                         "crop": args.crop, "audio": args.audio,
                         "max_length": args.max_length,
                         "model_dir": str(args.model_dir)}
            if not osu_path.exists():
                rec.update(error="no .osu in the cache")
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_skip += 1
                log(f"[skip] {bid}: no .osu")
                continue
            if "checksum" in sample.columns:
                rec["checksum_ok"] = (md5(osu_path) == str(row["checksum"]))
            audio = None
            audio_path = None
            if args.audio == "on":
                if not args.audio_dir:
                    raise SystemExit("--audio on needs --audio-dir")
                audio_path = find_audio(Path(args.audio_dir), bid)
                if audio_path is None:
                    rec.update(error="no staged audio")
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_skip += 1
                    log(f"[skip] {bid}: no staged audio")
                    continue
                try:
                    if args.audio_loader == "ffmpeg":
                        audio = decode_audio_ffmpeg(audio_path)
                    else:
                        from transformers.audio_utils import load_audio
                        audio = np.asarray(load_audio(str(audio_path), sampling_rate=16000),
                                           dtype=np.float32)
                except Exception as exc:  # noqa: BLE001
                    rec.update(error=f"audio decode: {exc}")
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_err += 1
                    log(f"[fail] {bid}: audio decode: {exc}")
                    continue
                rec["audio_file"] = audio_path.name
                rec["audio_bytes"] = audio_path.stat().st_size
            try:
                res = encode_chart(processor, model, osu_path, audio, device, dtype,
                                   args.pad_mode, args.win_batch, args.max_length,
                                   args.crop)
            except Exception as exc:  # noqa: BLE001
                rec.update(error=f"encode: {type(exc).__name__}: {exc}")
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_err += 1
                log(f"[fail] {bid}: encode: {exc}")
                continue
            if res.get("error"):
                rec.update(error=res["error"])
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_skip += 1
                log(f"[skip] {bid}: {res['error']}")
                continue
            vec = res.pop("vector")
            vectors[bid] = vec
            off = official.loc[bid, "embedding"] if bid in official.index else None
            if off is not None:
                off = np.asarray(off, dtype=np.float32)
                rec["cos_official"] = float(vec @ off / np.linalg.norm(off))
            if local is not None and bid in local.index:
                lv = np.asarray(local.loc[bid, "embedding"], dtype=np.float32)
                rec["cos_local"] = float(vec @ lv / np.linalg.norm(lv))
            rec.update({k: v for k, v in res.items() if k != "tok_len"})
            rec["error"] = None
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_ok += 1
            c = rec.get("cos_official")
            log(f"[{n_ok}/{len(sample) - len(done)}] {bid} set={setid} "
                f"cos={c:.5f} " if c is not None else f"[{n_ok}] {bid} set={setid} "
                f"n_win={rec['n_win']} tok={rec['tok_mean']:.0f} "
                f"enc={rec['encode_sec']:.2f}s fwd={rec['forward_sec']:.2f}s")

    # ---- summary ------------------------------------------------------------
    rows = []
    if out_path.exists():
        rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    ok = [r for r in rows if r.get("cos_official") is not None]
    cos = np.array([r["cos_official"] for r in ok], dtype=np.float64) if ok else np.zeros(0)
    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_sample": int(len(sample)),
        "n_ok": int(len(ok)),
        "n_error": int(sum(1 for r in rows if r.get("error") is not None)),
        "cos": {
            "mean": float(cos.mean()) if len(cos) else None,
            "median": float(np.median(cos)) if len(cos) else None,
            "min": float(cos.min()) if len(cos) else None,
            "p05": float(np.quantile(cos, 0.05)) if len(cos) else None,
            "p10": float(np.quantile(cos, 0.10)) if len(cos) else None,
            "n_ge_0_99": int((cos >= 0.99).sum()) if len(cos) else 0,
            "n_ge_0_999": int((cos >= 0.999).sum()) if len(cos) else 0,
        },
        "config": {
            "audio": args.audio, "audio_loader": args.audio_loader,
            "crop": args.crop, "dtype": args.dtype,
            "pad_mode": args.pad_mode, "max_length": args.max_length,
            "win_batch": args.win_batch, "device": device,
            "model_dir": str(args.model_dir), "attn": args.attn,
            "official_table": args.official_table,
        },
        "processor": snap,
        "processor_mismatches": mismatches,
        "elapsed_sec": round(time.time() - t_start, 1),
        "out": str(out_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    if args.save_vectors:
        np.savez_compressed(args.save_vectors,
                            ids=np.array(list(vectors.keys()), dtype=np.int64),
                            vectors=np.stack([vectors[k] for k in vectors]) if vectors
                            else np.zeros((0, 512), dtype=np.float32))
    log("\n=== probe summary ===")
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log(f"summary written: {summary_path}")


if __name__ == "__main__":
    main()
