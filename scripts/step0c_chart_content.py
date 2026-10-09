#!/usr/bin/env python
"""Step 0c - chart content table (pipeline A):  beatmap_id -> CM3P embedding.

Design: docs/cm3p_step2_integration_plan.md section 5.  The transformer inference and
step2 are TWO INDEPENDENT PIPELINES that only talk through one parquet file:

    pipeline A (this script)   .osu corpus / HF precomputed  ->  chart_content_cm3p.parquet
    pipeline B (step2_predict) reads that parquet only; it never imports transformers
                               and never touches external/cm3p

Sources
  --source hf-precomputed
      OliBomby/CM3P-Embeddings-244K (one parquet, 244,114 rows).  Rows are filtered to
      mania (ModeInt==3) and 4K (Cs==4) and projected onto the beatmap_meta_{tag}
      whitelist.  Zero GPU cost -- this is the path the first runs use.
  --source osu-dir
      Run the CM3P beatmap tower on local .osu files.  Needs transformers + the CM3P
      checkout + the HF checkpoint, and a GPU to be practical.  Reuses upstream's own
      BeatmapFilesDataset so the windowing/averaging matches the published pipeline.

Embedding convention (matches upstream extract_beatmap_embeddings.py):
  one 512-d, L2-normalised vector per beatmap = mean over its 16-second windows,
  re-normalised to unit length.

Records that are NOT silently dropped
  checksum_ok=False means the .osu the encoder saw is a *later revision* than the
  2026-09-01 dump snapshot (beatmap_meta_{tag}.csv checksum).  They stay in the table and
  the flag is reported, never filtered out here.

Idempotent merge
  If the output already exists it is loaded first, and only whitelist ids that are still
  missing are fetched/computed (--refresh forces a full recompute).  On a collision the
  NEW row wins, so re-running with a better source upgrades rows in place.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("data/processed")
EMBED_DIM = 512
CM3P_REPO = "OliBomby/CM3P"

# source codes; the numbers are part of the on-disk contract, do not renumber
SRC_HF = 0
SRC_LOCAL_NOAUDIO = 1
SRC_LOCAL_AUDIO = 2
SRC_NAME = {SRC_HF: "hf-precomputed", SRC_LOCAL_NOAUDIO: "local-noaudio",
            SRC_LOCAL_AUDIO: "local-audio"}

COLUMNS = ["beatmap_id", "embedding", "n_windows", "checksum_ok", "source", "cm3p_rev"]


# --------------------------------------------------------------------------- helpers
def whitelist(tag: str) -> pd.DataFrame:
    """The 4K mania whitelist from step0's beatmap metadata (beatmap_id + checksum)."""
    path = PROC / f"beatmap_meta_{tag}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run scripts/step0_preprocess.py first")
    df = pd.read_csv(path, usecols=["beatmap_id", "checksum"])
    df["beatmap_id"] = df["beatmap_id"].astype("int64")
    df["checksum"] = df["checksum"].astype(str).str.lower()
    return df


def load_want(args) -> pd.DataFrame:
    """The (beatmap_id, checksum) set to cover: --ids-file if given, else the step0 whitelist."""
    if args.ids_file:
        df = from_ids_file(args.ids_file)
        print(f"[ids-file] {args.ids_file}: {len(df):,} beatmaps")
        return df
    return whitelist(args.tag)


def from_ids_file(path: str) -> pd.DataFrame:
    """(beatmap_id, checksum) from a manifest; header optional, extra columns ignored."""
    df = pd.read_csv(path, dtype=str)
    cols = {c.strip().lower(): c for c in df.columns}
    if "beatmap_id" not in cols:                     # no header row -> re-read it as data
        df = pd.read_csv(path, dtype=str, header=None, names=["beatmap_id", "checksum"])
        cols = {"beatmap_id": "beatmap_id", "checksum": "checksum"}
    out = pd.DataFrame({"beatmap_id": df[cols["beatmap_id"]].astype("int64")})
    out["checksum"] = (df[cols["checksum"]].astype(str).str.lower()
                       if "checksum" in cols else "")
    return out.drop_duplicates("beatmap_id").reset_index(drop=True)


def load_table(path: str) -> pd.DataFrame:
    """Read a content table produced by an earlier run (--source table)."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{p} not found")
    df = pd.read_parquet(p)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{p} is missing columns {missing}")
    print(f"[table] {p.name}: {len(df):,} rows")
    return df


def empty_table() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS})


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return empty_table()
    df = pd.read_parquet(path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns {missing}; refusing to merge")
    print(f"[merge] {path.name}: {len(df):,} existing rows")
    return df


def _pack(ids, embeddings, n_windows, checksum_ok, source, rev) -> pd.DataFrame:
    """Build a table whose 'embedding' column holds float32 arrays, not object lists."""
    emb = [np.asarray(e, dtype=np.float32) for e in embeddings]
    for e in emb:
        if e.shape != (EMBED_DIM,):
            raise SystemExit(f"unexpected embedding shape {e.shape}, want ({EMBED_DIM},)")
    return pd.DataFrame({
        "beatmap_id": np.asarray(ids, dtype=np.int64),
        "embedding": emb,
        "n_windows": pd.array(n_windows, dtype="Int16"),
        "checksum_ok": pd.array(checksum_ok, dtype="boolean"),
        "source": np.asarray(source, dtype=np.int8),
        "cm3p_rev": np.asarray(rev, dtype=object),
    })


# ------------------------------------------------------------------------ pipeline A
def from_hf_precomputed(src: Path, want: pd.DataFrame, rev: str, batch: int) -> pd.DataFrame:
    """Stream the 244K parquet, keep mania 4K rows that are on the whitelist."""
    import pyarrow.parquet as pq

    keep = set(want["beatmap_id"].tolist())
    csum = dict(zip(want["beatmap_id"], want["checksum"]))
    pf = pq.ParquetFile(src)
    print(f"[hf] {src.name}: {pf.metadata.num_rows:,} rows, {pf.metadata.num_row_groups} row groups")

    ids, embs, wins, oks, srcs, revs = [], [], [], [], [], []
    seen = 0
    for b in pf.iter_batches(batch_size=batch,
                             columns=["Id", "ModeInt", "Cs", "Checksum", "embedding"]):
        d = b.to_pandas()
        seen += len(d)
        d = d[(d["ModeInt"] == 3) & (d["Cs"] == 4)]
        if d.empty:
            continue
        d = d[d["Id"].isin(keep)]
        if d.empty:
            continue
        for bid, ck, e in zip(d["Id"].to_numpy(), d["Checksum"].to_numpy(), d["embedding"].to_numpy()):
            ids.append(int(bid))
            embs.append(e)
            wins.append(pd.NA)                       # the published table does not carry it
            oks.append(str(ck).lower() == csum.get(int(bid), ""))
            srcs.append(SRC_HF)
            revs.append(rev)
    print(f"[hf] scanned {seen:,} rows -> {len(ids):,} on the whitelist")
    if not ids:
        return empty_table()
    return _pack(ids, embs, wins, oks, srcs, revs)


def from_osu_dir(dirs, want: pd.DataFrame, rev: str, audio: bool, batch: int,
                 device: str) -> pd.DataFrame:
    """Run the CM3P beatmap tower on local .osu/.osz files (upstream dataset + model)."""
    root = Path(os.environ.get("CM3P_ROOT", "external/cm3p")).resolve()
    if not (root / "cm3p").is_dir():
        raise SystemExit(f"{root} not found -- git clone https://github.com/OliBomby/CM3P there")
    sys.path.insert(0, str(root))
    try:
        import torch
        from torch.utils.data import DataLoader
        from cm3p.processing_cm3p import CM3PProcessor
        from cm3p.modeling_cm3p import CM3PModel
        from utils.beatmap_files_dataset import BeatmapFilesDataset
    except Exception as exc:                                   # pragma: no cover
        raise SystemExit(
            "the osu-dir path needs transformers + torch (pip install -r "
            f"external/cm3p/requirements.txt); import failed with: {exc}")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[osu] device={device}  audio={audio}  dirs={[str(d) for d in dirs]}")
    processor = CM3PProcessor.from_pretrained(rev)
    model = CM3PModel.from_pretrained(rev, trust_remote_code=True, revision="main")
    # bf16 on CUDA is what upstream's extract_beatmap_embeddings.py used for the published
    # table, so it is also the faithful choice here; fp32 on CPU.
    mdt = torch.bfloat16 if device == "cuda" else torch.float32
    model = model.to(device=device, dtype=mdt).eval()
    print(f"[osu] model dtype {mdt}", flush=True)

    ds = BeatmapFilesDataset([str(d) for d in dirs], processor=processor,
                             include_audio=audio, include_beatmap=True, include_metadata=False)
    loader = DataLoader(ds, batch_size=batch, num_workers=0, drop_last=False)

    keep = set(want["beatmap_id"].tolist())
    csum = dict(zip(want["beatmap_id"], want["checksum"]))

    # map beatmap_id -> the .osu we are about to encode, so checksum_ok can be verified
    # against the 2026-09-01 dump snapshot instead of being asserted blindly
    meta = ds.metadata.reset_index()
    files: dict[int, Path] = {}
    for r in meta.itertuples():
        try:
            bid = int(r.Id)
        except (TypeError, ValueError):
            continue
        files[bid] = Path(str(getattr(r, "Path", "."))) / str(r.BeatmapSetFolder) / str(r.BeatmapFile)

    def md5_of(p: Path) -> str:
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    acc: dict[int, list] = {}
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    with torch.no_grad():
        for batch_i, b in enumerate(loader):
            if len(b.get("input_ids", [])) == 0:
                continue
            inputs = {"input_ids": b["input_ids"].to(device),
                      "attention_mask": b["attention_mask"].to(device)}
            if b.get("input_features") is not None:
                inputs["input_features"] = b["input_features"].to(device=device, dtype=dtype)
            out = model(**inputs, return_loss=False)
            emb = out.beatmap_embeds.detach().float().cpu().numpy()
            for i, bid in enumerate(b["beatmap_id"].tolist()):
                if bid is None or int(bid) not in keep:
                    continue
                acc.setdefault(int(bid), []).append(emb[i])
            if batch_i % 50 == 0:
                print(f"[osu] batch {batch_i}: {len(acc):,} beatmaps accumulated", flush=True)

    ids, embs, wins, oks, srcs, revs = [], [], [], [], [], []
    for bid, vecs in acc.items():
        m = np.mean(np.stack(vecs), axis=0)
        n = float(np.linalg.norm(m))
        want_ck = csum.get(bid, "")
        got_ck = ""
        if bid in files and files[bid].exists():
            got_ck = md5_of(files[bid])
        ids.append(bid)
        embs.append((m / n) if n > 0 else m)
        wins.append(len(vecs))
        oks.append((got_ck == want_ck) if want_ck else None)
        srcs.append(SRC_LOCAL_AUDIO if audio else SRC_LOCAL_NOAUDIO)
        revs.append(rev)
    print(f"[osu] {len(ids):,} beatmaps embedded")
    if not ids:
        return empty_table()
    return _pack(ids, embs, wins, oks, srcs, revs)


# ------------------------------------------------------------------------------ merge
def merge(base: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if new.empty:
        return base
    if base.empty:
        return new.reset_index(drop=True)
    both = base.merge(new, on="beatmap_id", how="inner", suffixes=("_old", "_new"))
    print(f"[merge] {len(both):,} collisions -> the new rows win")
    keep = base[~base["beatmap_id"].isin(set(new["beatmap_id"].tolist()))]
    return pd.concat([keep, new], ignore_index=True)


def write_atomic(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    df = df.sort_values("beatmap_id").reset_index(drop=True)
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)


# ------------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1k", help="which beatmap_meta_{tag}.csv is the whitelist")
    ap.add_argument("--source", default="hf-precomputed",
                    choices=["hf-precomputed", "osu-dir", "table"])
    ap.add_argument("--ids-file", default=None,
                    help="beatmap_id[,checksum] per line (header optional).  Overrides "
                         "beatmap_meta_{tag}.csv -- use this on a fresh clone, e.g. "
                         "manifests/cm3p_todo_1k.csv")
    ap.add_argument("--in", dest="in_path", default=None,
                    help="--source table: an already-computed content table to merge in")
    ap.add_argument("--infer-batch", type=int, default=8,
                    help="osu-dir only: beatmaps per forward pass (NOT --batch, which is rows)")
    ap.add_argument("--hf-path", default="data/raw/cm3p_embeddings/beatmap_embeddings.parquet")
    ap.add_argument("--osu-dir", action="append", default=None,
                    help="directory (or .osu/.osz file) to embed; repeatable; --source osu-dir")
    ap.add_argument("--audio", default="off", choices=["on", "off"],
                    help="osu-dir only: fuse audio.  MUST NOT be mixed with hf-precomputed rows "
                         "(they were computed with audio); the source column records it")
    ap.add_argument("--out", default=str(PROC / "chart_content_cm3p.parquet"))
    ap.add_argument("--refresh", action="store_true",
                    help="ignore the existing table and recompute every id")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--limit", type=int, default=0, help="debug: only this many ids")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    t0 = time.time()
    out = Path(args.out)

    want = load_want(args)
    print(f"target set: {len(want):,} beatmaps "
          f"(source: {args.ids_file or f'beatmap_meta_{args.tag}.csv'})")
    base = empty_table() if args.refresh else load_existing(out)
    if not base.empty:
        base = base[base["beatmap_id"].isin(set(want["beatmap_id"].tolist()))]
    missing = want[~want["beatmap_id"].isin(set(base["beatmap_id"].tolist()))]
    if args.limit:
        missing = missing.head(args.limit)
    print(f"to fetch: {len(missing):,}  (already present: {len(base):,})")

    if args.source == "table":
        # merge mode: the incoming table is the whole point, whether or not anything is
        # "missing" relative to the whitelist
        if not args.in_path:
            raise SystemExit("--source table needs --in PATH")
        new = load_table(args.in_path)
    elif len(missing) == 0:
        print("nothing to do; the table is already complete for this whitelist")
        new = empty_table()
    elif args.source == "hf-precomputed":
        src = Path(args.hf_path)
        if not src.exists():
            raise SystemExit(f"{src} not found -- see docs/cm3p_step2_integration_plan.md M1")
        new = from_hf_precomputed(src, missing, CM3P_REPO, args.batch)
    else:
        if not args.osu_dir:
            raise SystemExit("--source osu-dir needs at least one --osu-dir")
        new = from_osu_dir(args.osu_dir, missing, CM3P_REPO, args.audio == "on",
                           args.infer_batch, args.device)

    table = merge(base, new)
    write_atomic(table, out)

    # --------------------------------------------------------------------- report
    have = set(table["beatmap_id"].tolist())
    cov = len(have & set(want["beatmap_id"].tolist())) / len(want)
    print(f"\nwrote {out}  ({len(table):,} rows, {out.stat().st_size / 1e6:.1f} MB)")
    print(f"coverage of the target set: {len(have):,}/{len(want):,} = {cov:.2%}")
    by_src = table.groupby("source").size()
    for k, v in by_src.items():
        print(f"  source {int(k)} ({SRC_NAME.get(int(k), '?')}): {v:,}")
    bad = table["checksum_ok"].eq(False).sum()
    if bad:
        print(f"  checksum_ok=False: {int(bad):,} "
              f"({bad / len(table):.2%}) -- later .osu revision, kept and flagged")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
