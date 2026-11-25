#!/usr/bin/env python
"""Step 0c - chart content table (pipeline A):  beatmap_id -> CM3P embedding.

Design: legacy/docs/cm3p_step2_integration_plan.md section 5.  The transformer inference and
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
      checkout + the HF checkpoint, and a GPU to be practical.  The cache now holds
      *every* whitelist chart (scripts/step0d_fetch_osu.py, only .osu bodies -- no
      audio/images), so this path can cover the whole 21,949 instead of the 11,877 the
      published table happens to contain.

  --crop canon (default)
      T1 of the v5 plan: add_hitsounds/add_kiai/add_mania_sv = False.
      The kept flags are inherited from the checkpoint, NOT forced on: the real
      HF preprocessor config has add_snapping=False, add_cls_token=True and
      max_length=2000 (the upstream training yaml says True/False/4000).  --crop
      legacy turns the three payload flags on, leaving the rest unchanged.
  --windows on (default for --source osu-dir)
      T2: in addition to the chart table, write one float16 .npy per chart (time-ordered
      16 s windows) and .cache/cm3p/index.parquet.  Windows are batched by a
      BucketSampler (ceil(len/64)*64) with dynamic padding, so pad_efficiency and
      charts/s are reported; the old BeatmapFilesDataset path is --windows off.
  --mania-columns auto (default)
      Detects that upstream never emits [MANIA_COLUMN_*]: parsing_cm3p.py:501 writes
      Group.column while tokenization_cm3p.py:191 reads Group.mania_column, which the
      dataclass never sets.  The frozen checkpoint is therefore column-blind; 'on'
      installs the opt-in runtime patch (fixes the attribute and the 0/1-based column),
      which is out-of-distribution for the checkpoint.  Every index row records
      columns_emitted so the two representations can never be mixed silently.

  Merge guard
      A run writes <out>.meta.json with its cfg_hash/crop.  Merging canon rows into a
      table built with the legacy flags (or the reverse) is refused unless
      --allow-mixed-crop or --refresh is given.  --crop canon defaults --out to
      chart_content_cm3p_local_canon.parquet, leaving the legacy table untouched.

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
  NEW row wins, so re-running with a better source upgrades rows in place.  When the main
  table is already complete but --windows on is set, the run still goes through the window
  exporter: ready charts are skipped by the index, missing windows are filled in.

Chart id vs the id inside the .osu
  Upstream keys every dataset row by the `BeatmapID:` line *inside* the body, and a chart
  that was deleted and re-uploaded is served with the NEW id there (or with 0), so the row
  no longer matches the id the file was fetched for and would be dropped without a word.
  the --windows off path stages an id-normalised copy for exactly those files (the cache
  is never modified, and checksum_ok is still computed on the original file).  The T2 path
  does not need it: it passes the original file to the processor and keys the row by the
  file name, so the id line inside the body is never used.  If a legacy run reports fewer
  charts than it staged, look for the "[osu] N/M .osu name a different BeatmapID" line --
  the tail summary of scripts/slurm/cm3p_embed.slurm calls it "no .osu in the cache",
  which is only one of the two causes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dcr import cm3p_windows as cw  # noqa: E402

PROC = Path("data/processed")
EMBED_DIM = 512
CM3P_REPO = "OliBomby/CM3P"

# CM3P's modelling code decorates its head with @torch.compile(dynamic=True)
# (`compiled_head` in modeling_cm3p.py) and calls it whenever the model config sets
# `reference_compile` -- the published HF config does.  Compiling drags in
# TorchInductor, which shells out to `nvcc`; on a cluster where nvcc is not
# executable for the user that dies with
#     torch._inductor.exc.InductorError: PermissionError: [Errno 13] Permission denied: 'nvcc'
# Inference needs no compiler, and dynamo off evaluates the *same* math eagerly
# (the probe gate compares these vectors against the published ones, so a real
# difference would show up there).  setdefault: an explicit env still wins.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# source codes; the numbers are part of the on-disk contract, do not renumber
SRC_HF = 0
SRC_LOCAL_NOAUDIO = 1
SRC_LOCAL_AUDIO = 2
SRC_NAME = {SRC_HF: "hf-precomputed", SRC_LOCAL_NOAUDIO: "local-noaudio",
            SRC_LOCAL_AUDIO: "local-audio"}

COLUMNS = ["beatmap_id", "embedding", "n_windows", "checksum_ok", "source", "cm3p_rev"]


# --------------------------------------------------------------------------- helpers
def whitelist(tag: str, meta_dir: Path = PROC) -> pd.DataFrame:
    """The 4K mania whitelist from step0's beatmap metadata (beatmap_id + checksum)."""
    path = Path(meta_dir) / f"beatmap_meta_{tag}.csv"
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
    return whitelist(args.tag, args.meta_dir)


TRUEISH = ("true", "1", "1.0", "yes", "y", "t")


def from_ids_file(path: str) -> pd.DataFrame:
    """(beatmap_id, checksum) from a manifest; header optional, extra columns ignored.

    When the manifest carries a `cacheable` flag (the whitelist table that
    scripts/step0g_whitelist.py writes) only the true rows are kept, so an
    `--source osu-dir` run targets exactly the charts that are in the cache and the
    coverage report stays meaningful instead of counting charts that cannot exist.
    """
    df = pd.read_csv(path, dtype=str)
    cols = {c.strip().lower(): c for c in df.columns}
    if "beatmap_id" not in cols:                     # no header row -> re-read it as data
        df = pd.read_csv(path, dtype=str, header=None, names=["beatmap_id", "checksum"])
        cols = {"beatmap_id": "beatmap_id", "checksum": "checksum"}
    keep = None
    for flag in ("cacheable", "osu_cached", "cached"):
        if flag in cols:
            keep = df[cols[flag]].astype(str).str.strip().str.lower().isin(TRUEISH).to_numpy()
            print(f"[ids-file] {flag}: keeping {int(keep.sum())} of {len(df)} rows")
            break
    idx = df.index[keep] if keep is not None else df.index
    out = pd.DataFrame({"beatmap_id": df.loc[idx, cols["beatmap_id"]].astype("int64")})
    out["checksum"] = (df.loc[idx, cols["checksum"]].astype(str).str.lower()
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


def _locate_osu(dirs, ids: set[int]) -> dict[int, Path]:
    """beatmap_id -> .osu path, for the wanted charts only.

    The cache stores one `{beatmap_id}.osu` per chart, so this is one stat per wanted id;
    a directory that does not follow that convention is walked once instead.  Doing it
    here is what lets from_osu_dir hand upstream a corpus limited to the target set.
    """
    found: dict[int, Path] = {}
    for d in dirs:
        p = Path(d)
        if p.is_file():
            stem = p.stem.split(" ")[0]
            if stem.isdigit() and int(stem) in ids:
                found.setdefault(int(stem), p)
            continue
        if not p.is_dir():
            continue
        before = len(found)
        for bid in ids:
            q = p / f"{bid}.osu"
            if q.is_file():
                found.setdefault(bid, q)
        if len(found) == before:                  # convention does not apply -> walk once
            for q in p.rglob("*.osu"):
                stem = q.stem.split(" ")[0]
                if stem.isdigit() and int(stem) in ids:
                    found.setdefault(int(stem), q)
    return found


# Upstream keys every dataset row by the `BeatmapID:` line *inside* the .osu body
# (utils/beatmap_files_dataset.py: data['Id'] = _kv(meta, 'BeatmapID'), then
# item['beatmap_id'] = beatmap_metadata.name).  A chart that was deleted and re-uploaded is
# served with the NEW id in that line -- or with 0 when it was never submitted -- so the row
# no longer matches the id the file was fetched for, and step0c would silently drop it (the
# run reports one chart fewer than it staged, and the tail summary blames the cache).
# These two helpers let the staging step serve an id-normalised copy for exactly those files.
_OSU_ID_RE = re.compile(rb"^(\s*BeatmapID\s*:\s*)(-?\d+)", re.M)


def osu_beatmap_id(path: Path) -> int | None:
    """The BeatmapID the .osu body carries, or None when the line is absent."""
    try:
        m = _OSU_ID_RE.search(path.read_bytes())
    except OSError:
        return None
    return int(m.group(2)) if m else None


def relabel_beatmap_id(body: bytes, bid: int) -> bytes:
    """The body with its first `BeatmapID:` line rewritten to bid (nothing else changes)."""
    return _OSU_ID_RE.sub(lambda m: m.group(1) + str(bid).encode(), body, count=1)


# ---------------------------------------------------------------- T1/T2 helpers
def _git_commit(root: Path) -> str:
    try:
        import subprocess
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception:
        return ""


def _sidecar(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".meta.json")


def _check_merge_guard(out: Path, meta: dict | None, allow: bool) -> None:
    """Refuse to merge rows from a different token representation into a table."""
    if meta is None or not out.exists():
        return
    sc = None
    p = _sidecar(out)
    if p.exists():
        try:
            import json
            sc = json.loads(p.read_text())
        except Exception:
            sc = None
    if sc is None:
        crop = meta.get("crop", {})
        legacy = bool(crop.get("add_hitsounds")) and bool(crop.get("add_kiai")) \
            and bool(crop.get("add_mania_sv"))
        if legacy:
            print(f"[merge] {out.name} has no cfg sidecar; assuming the legacy flags "
                  f"(--crop legacy) -- OK", flush=True)
            return
        if allow:
            print(f"[merge] WARNING: {out.name} has no cfg sidecar and this run is "
                  f"{meta.get('crop')}; --allow-mixed-crop given", flush=True)
            return
        raise SystemExit(
            f"{out} exists without a cfg sidecar (it was built with the legacy flags). "
            "This run uses --crop canon, so merging would mix two representations. "
            "Use a new --out, delete the table, or pass --allow-mixed-crop after "
            "accepting the mixture.")
    if str(sc.get("cfg_hash")) != str(meta.get("cfg_hash")):
        msg = (f"{out} was built with cfg {sc.get('cfg_hash')} (crop "
               f"{sc.get('crop')}) but this run is cfg {meta.get('cfg_hash')} "
               f"(crop {meta.get('crop')})")
        if allow:
            print(f"[merge] WARNING: {msg}; --allow-mixed-crop given", flush=True)
            return
        raise SystemExit(msg + ". Use a new --out, --refresh, or --allow-mixed-crop.")


def _precheck_sidecar(out: Path, args, allow: bool) -> None:
    """Fail before the GPU work if the existing table uses a different crop."""
    if not out.exists() or args.refresh:
        return
    p = _sidecar(out)
    if not p.exists():
        if args.crop == "legacy" or allow:
            return
        raise SystemExit(
            f"{out} exists without a cfg sidecar (it was built with the legacy flags). "
            "A --crop canon run would merge two representations. Use a new --out, "
            "delete the table, or pass --allow-mixed-crop.")
    try:
        sc = json.loads(p.read_text())
    except Exception:
        return
    decision = ("add_hitsounds", "add_kiai", "add_mania_sv")
    want = {k: v for k, v in cw.CropSpec.from_name(args.crop).to_dict().items()
            if k in decision}
    got = {k: (sc.get("crop") or {}).get(k) for k in decision}
    if got != want or sc.get("mania_columns") != args.mania_columns:
        msg = (f"{out} was built with crop={sc.get('crop')} "
               f"mania_columns={sc.get('mania_columns')} (cfg {sc.get('cfg_hash')}) "
               f"but this run asks crop={want} mania_columns={args.mania_columns}")
        if allow:
            print(f"[merge] WARNING: {msg}; --allow-mixed-crop given", flush=True)
            return
        raise SystemExit(msg + ". Use a new --out, --refresh, or --allow-mixed-crop.")


def _core_audit(files: dict[int, Path], processor, snap, tids, crop,
                args, max_length: int) -> dict:
    """Prove on real charts that the crop removes no note/timing information."""
    n = int(args.core_audit)
    if n <= 0:
        return {"n": 0}
    legacy = cw.CropSpec.legacy()
    paths = sorted(files.items())[:n]
    removed = {"hitsound": 0, "kiai": 0, "sv": 0}
    n_notes = 0
    n_cols = 0
    n_core_drop = 0
    bad: list[str] = []
    skipped: list[str] = []
    for bid, path in paths:
        try:
            crop.apply(processor.beatmap_parser)
            ct_crop = cw.tokenize_chart(
                processor, path, window_length_sec=snap.window_length_sec,
                window_stride_sec=snap.window_stride_sec,
                min_window_length_sec=snap.min_window_length_sec,
                max_length=max_length, tids=tids, beatmap_id=bid,
                keep_tokens=True)
            legacy.apply(processor.beatmap_parser)
            ct_full = cw.tokenize_chart(
                processor, path, window_length_sec=snap.window_length_sec,
                window_stride_sec=snap.window_stride_sec,
                min_window_length_sec=snap.min_window_length_sec,
                max_length=max_length, tids=tids, beatmap_id=bid,
                keep_tokens=True)
        except Exception as exc:
            # a corrupt .osu is not a crop failure: the exporter will skip it too
            skipped.append(f"{bid}: {exc}")
            continue
        finally:
            crop.apply(processor.beatmap_parser)
        a = cw.audit_crop_pair(ct_full, ct_crop)
        for k, v in a.removed.items():
            removed[k] += v
        n_notes += ct_crop.n_notes
        n_cols += ct_crop.n_column_tokens
        n_core_drop += int(ct_crop.n_core_dropped)
        if not a.ok:
            bad.append(f"{bid}: " + "; ".join((a.problems + a.window_problems)[:3]))
    summary = {"n": len(paths), "n_bad": len(bad), "n_skipped": len(skipped),
               "removed": removed, "n_notes": n_notes,
               "n_column_tokens": n_cols, "columns_emitted": n_cols > 0,
               "n_core_tokens_dropped": n_core_drop}
    if skipped:
        print(f"[core] {len(skipped)} chart(s) could not be tokenised and were skipped "
              f"from the audit (they are skipped by the exporter too): "
              f"{skipped[:3]}", flush=True)
    if bad:
        for b in bad[:10]:
            print(f"[core] FAIL {b}", flush=True)
        raise SystemExit(
            "CORE AUDIT FAILED: the crop changed the note/timing stream. Refusing "
            "to export a window store that lost chart content.")
    if len(skipped) == len(paths):
        print("[core] WARNING: every audited chart failed to tokenise; the core audit "
              "verified nothing (the exporter will skip them)", flush=True)
    else:
        print(f"[core] {len(paths) - len(skipped)} charts: note/timing stream identical "
              f"after the crop; removed tokens hitsound={removed['hitsound']:,} "
              f"kiai={removed['kiai']:,} sv={removed['sv']:,}; "
              f"{n_notes:,} notes, {n_cols:,} column tokens", flush=True)
    if n_core_drop:
        print(f"[core] WARNING: {n_core_drop} core note token(s) in the audited "
              f"charts exceed max_length={max_length}; the export records them per "
              f"chart and --on-truncate decides (see the index).", flush=True)
    return summary


def _md5_of(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _encode_fn(model, device: str, torch_mod):
    def encode(ids, mask):
        with torch_mod.no_grad():
            out = model(input_ids=torch_mod.as_tensor(ids, device=device),
                        attention_mask=torch_mod.as_tensor(mask, device=device),
                        return_loss=False)
        return out.beatmap_embeds.detach().float().cpu().numpy()
    return encode


def _table_from_vectors(acc, wins, md5_map, csum, audio: bool, rev: str):
    ids, embs, wins_l, oks, srcs, revs = [], [], [], [], [], []
    for bid in sorted(acc):
        want = csum.get(int(bid), "")
        got = md5_map.get(int(bid), "")
        ids.append(int(bid))
        embs.append(np.asarray(acc[bid], dtype=np.float32))
        n = int(wins.get(int(bid), 0))
        wins_l.append(n if n > 0 else None)
        oks.append((got == want) if want else None)
        srcs.append(SRC_LOCAL_AUDIO if audio else SRC_LOCAL_NOAUDIO)
        revs.append(rev)
    if not ids:
        return empty_table()
    return _pack(ids, embs, wins_l, oks, srcs, revs)


def _export_windows_table(files, md5_map, csum, processor, snap, tids, crop, args,
                          rev: str, mdt, attn: str, cfg: str, encode,
                          torch_mod) -> tuple[pd.DataFrame, dict]:
    """T2: tokenise every chart, export .npy windows, build the chart table."""
    win_dir = Path(args.window_dir)
    index_path = Path(args.window_index)
    if args.refresh_windows:
        if index_path.exists():
            index_path.unlink()
        n_removed = 0
        if win_dir.exists():
            for p in win_dir.glob("*.npy"):
                p.unlink()
                n_removed += 1
        print(f"[win] --refresh-windows: cleared the index and {n_removed:,} .npy files",
              flush=True)
    existing = cw.load_window_index(index_path)
    hashes = cw.index_cfg_hashes(existing)
    if hashes and cfg not in hashes:
        raise SystemExit(
            f"[win] {index_path} was built with cfg {sorted(hashes)} but this run is "
            f"{cfg}. Mixing window representations is not allowed: pass "
            "--refresh-windows (rebuild) or use a different --window-dir/--window-index.")
    ready = (cw.index_ready(existing, cfg, md5_map, win_dir)
             if existing is not None else set())
    if ready:
        print(f"[win] resuming: {len(ready):,}/{len(files):,} charts already exported "
              f"under cfg {cfg}", flush=True)
    cfg_obj = cw.ExportConfig(
        win_dir=win_dir, index_path=index_path, max_length=snap.max_length,
        batch_size=int(args.window_batch), buffer_windows=int(args.window_buffer),
        seed=int(args.window_seed), dtype=args.window_dtype, cfg_hash=cfg,
        verify_all=(int(args.verify_windows) == 0),
        verify_first=max(1, int(args.verify_windows)),
        on_truncate=args.on_truncate)
    exporter = cw.WindowExporter(cfg_obj, encode, base_index=existing,
                                 log=lambda s: print(s, flush=True))
    acc: dict[int, np.ndarray] = {}
    wins: dict[int, int] = {}
    n_ready = n_failed = 0
    checked_columns = False
    order = sorted(files)
    t0 = time.time()
    for k, bid in enumerate(order):
        path = files[bid]
        if bid in ready:
            mat = np.load(win_dir / f"{bid}.npy").astype(np.float32)
            mean = mat.mean(axis=0)
            norm = float(np.linalg.norm(mean))
            acc[int(bid)] = (mean / norm) if norm > 0 else mean
            wins[int(bid)] = int(mat.shape[0])
            n_ready += 1
            continue
        try:
            ct = cw.tokenize_chart(
                processor, path, window_length_sec=snap.window_length_sec,
                window_stride_sec=snap.window_stride_sec,
                min_window_length_sec=snap.min_window_length_sec,
                max_length=snap.max_length, tids=tids, beatmap_id=bid,
                md5=md5_map.get(int(bid), ""))
        except Exception as exc:
            print(f"[win] WARNING: {bid} failed to tokenize: {exc}", flush=True)
            exporter.skipped.append((int(bid), str(exc)))
            n_failed += 1
            continue
        if not checked_columns and ct.n_note_tokens:
            checked_columns = True
            if ct.columns_emitted:
                print(f"[core] {bid}: [MANIA_COLUMN_*] tokens are PRESENT "
                      f"({ct.n_column_tokens:,} in {ct.n_windows} windows)", flush=True)
            elif args.mania_columns == "auto":
                print("[core] WARNING: no [MANIA_COLUMN_*] tokens are emitted. Upstream "
                      "stores Group.column but the tokenizer reads Group.mania_column "
                      "(parsing_cm3p.py:501 vs tokenization_cm3p.py:191), so the frozen "
                      "model is column-blind. Pass --mania-columns on to install the "
                      "opt-in parser patch (out-of-distribution for the checkpoint).",
                      flush=True)
            elif args.mania_columns == "on":
                raise SystemExit(
                    f"{bid}: --mania-columns on was requested but no column token "
                    "was emitted; the runtime patch did not take effect")
        wins[int(bid)] = ct.n_windows
        exporter.add_chart(ct)
        if (k + 1) % 200 == 0:
            el = time.time() - t0
            print(f"[win] tokenized {k + 1:,}/{len(order):,} charts "
                  f"({el:.0f}s, {(k + 1) / max(el, 1e-9):.1f} charts/s)", flush=True)
    res = exporter.finish()
    acc.update(res.per_chart)
    for bid, _reason in res.skipped:
        wins.pop(int(bid), None)
    table = _table_from_vectors(acc, wins, md5_map, csum, False, rev)
    print(f"[win] exported {res.n_charts:,} charts / {res.n_windows:,} windows in "
          f"{res.seconds:.0f}s ({res.n_charts / max(res.seconds, 1e-9):.2f} charts/s); "
          f"pad_efficiency={res.pad_efficiency:.3f}; parity cosine >= "
          f"{res.min_parity_cosine:.9f}", flush=True)
    if res.n_truncated_windows:
        print(f"[win] WARNING: {res.n_truncated_windows} window(s) exceeded "
              f"max_length={snap.max_length}; {res.n_core_tokens_dropped} core note "
              f"token(s) were dropped. The affected charts are flagged in the index "
              f"(n_core_tokens_dropped) and excluded from nothing by default.",
              flush=True)
    meta = {
        "tool": "step0c_chart_content/v4",
        "cfg_hash": cfg,
        "snapshot": snap.to_dict(),
        "crop": crop.to_dict(),
        "mania_columns": args.mania_columns,
        "columns_emitted": bool(res.columns_emitted),
        "model": rev,
        "dtype": str(mdt),
        "attn": attn,
        "windows": {"dir": str(win_dir), "index": str(index_path),
                    "batch": int(args.window_batch),
                    "buffer": int(args.window_buffer),
                    "dtype": args.window_dtype, "seed": int(args.window_seed),
                    "on_truncate": args.on_truncate},
        "export": res.summary(),
        "n_target": len(files), "n_ready": n_ready, "n_failed": n_failed,
        "n_no_windows": max(0, len(res.skipped) - n_failed),
    }
    cw.write_meta(index_path.with_suffix(".meta.json"), meta)
    return table, meta


def _dataset_table(files, csum, processor, crop, audio, batch, device,
                   rev, args, md5_map, forward_fn):
    """Legacy loader path (--windows off): upstream BeatmapFilesDataset."""
    from torch.utils.data import DataLoader
    from utils.beatmap_files_dataset import BeatmapFilesDataset
    farm = tempfile.TemporaryDirectory(prefix="cm3p_subset_")
    n_relabelled = 0
    for bid, src in files.items():
        dst = Path(farm.name) / f"{bid}.osu"
        try:
            inner = osu_beatmap_id(src)
            if inner is not None and inner != bid:
                dst.write_bytes(relabel_beatmap_id(src.read_bytes(), bid))
                n_relabelled += 1
            else:
                os.symlink(os.path.abspath(src), dst)
        except OSError as exc:
            print(f"[osu] WARNING: cannot stage {src}: {exc}", flush=True)
    if n_relabelled:
        print(f"[osu] {n_relabelled:,}/{len(files):,} .osu name a different BeatmapID "
              f"inside; staged id-normalised copies", flush=True)
    n_files = len(list(Path(farm.name).glob("*.osu")))
    print(f"[osu] corpus: {n_files:,} .osu staged for {len(files):,} wanted charts",
          flush=True)
    ds = BeatmapFilesDataset([farm.name], processor=processor,
                             include_audio=audio, include_beatmap=True,
                             include_metadata=False)
    loader = DataLoader(ds, batch_size=batch, num_workers=0, drop_last=False)
    acc: dict[int, list] = {}
    torch_mod = __import__("torch")
    t0 = time.time()
    with torch_mod.no_grad():
        for batch_i, b in enumerate(loader):
            if len(b.get("input_ids", [])) == 0:
                continue
            inputs = {"input_ids": b["input_ids"].to(device),
                      "attention_mask": b["attention_mask"].to(device)}
            if b.get("input_features") is not None:
                inputs["input_features"] = b["input_features"].to(device=device)
            emb = forward_fn(inputs)
            for i, bid in enumerate(b["beatmap_id"].tolist()):
                if bid is None or int(bid) not in files:
                    continue
                acc.setdefault(int(bid), []).append(emb[i])
            if batch_i % 10 == 0:
                el = time.time() - t0
                print(f"[osu] batch {batch_i + 1}: {len(acc):,}/{n_files:,} charts seen, "
                      f"{el:.0f}s elapsed ({(batch_i + 1) / max(el, 1e-9):.2f} batch/s)",
                      flush=True)
    vecs: dict[int, np.ndarray] = {}
    wins: dict[int, int] = {}
    for bid, vs in acc.items():
        m = np.mean(np.stack(vs), axis=0)
        n = float(np.linalg.norm(m))
        vecs[bid] = (m / n) if n > 0 else m
        wins[bid] = len(vs)
    table = _table_from_vectors(vecs, wins, md5_map, csum, audio, rev)
    print(f"[osu] {len(vecs):,} beatmaps embedded", flush=True)
    return table


def from_osu_dir(dirs, want, rev, audio, batch, device, args):
    """Run the CM3P beatmap tower on local .osu files (T1 crop + T2 windows)."""
    root = Path(os.environ.get("CM3P_ROOT", "external/cm3p")).resolve()
    if not (root / "cm3p").is_dir():
        raise SystemExit(f"{root} not found -- git clone https://github.com/OliBomby/CM3P there")
    sys.path.insert(0, str(root))
    try:
        import torch
        from cm3p.processing_cm3p import CM3PProcessor
        from cm3p.modeling_cm3p import CM3PModel
        from cm3p.parsing_cm3p import get_song_length, load_beatmap  # noqa: F401
    except Exception as exc:                                   # pragma: no cover
        hint = ""
        if "hydra" in str(exc):
            hint = ("\n  hydra-core: CM3P's config.py imports 'hydra' at module level "
                    "(it only builds\n  train.py's ConfigStore) and utils/mmrs_dataset.py "
                    "pulls it in, so the embed\n  path needs it even though Hydra itself "
                    "is never used:  pip install 'hydra-core>=1.3,<1.4'")
        raise SystemExit(
            "the osu-dir path needs the CM3P inference env (torch, transformers==4.55.0, "
            "hydra-core, slider) -- see requirements-cm3p.txt and legacy/docs/cm3p_hpc_slurm.md "
            f"section 1; import failed with: {exc}{hint}")

    # Some slider distributions treat mania's standard `endTime:hitSample`
    # payload as one integer and reject hold notes.  Match the local parser that
    # produced the existing CM3P table by splitting endTime from its hit sample.
    try:
        from datetime import timedelta
        from slider.beatmap import HoldNote
        try:
            probe = HoldNote._parse(None, None, 0, False, 0, ["1234:0:0:0:0:"])
            if probe.end_time != timedelta(milliseconds=1234):
                raise ValueError("unexpected mania hold end_time")
        except Exception:
            @classmethod
            def _parse_mania_hold(cls, position, time, hitsound, new_combo, combo_skip, rest):
                if not rest:
                    raise ValueError("missing end_time")
                try:
                    pieces = rest[0].split(":")
                    end_time, *sample = pieces
                    rest = [":".join(sample), *rest[1:]]
                    end_time = timedelta(milliseconds=int(end_time))
                except (IndexError, TypeError, ValueError) as exc:
                    raise ValueError("invalid mania end_time/hitSample") from exc
                if len(rest) > 1:
                    raise ValueError(f"extra data in mania hold note: {rest!r}")
                return cls(position, time, hitsound, end_time, new_combo, combo_skip, *rest)

            HoldNote._parse = _parse_mania_hold
            probe = HoldNote._parse(None, None, 0, False, 0, ["1234:0:0:0:0:"])
            if probe.end_time != timedelta(milliseconds=1234):
                raise SystemExit("failed to install mania hold-note parser compatibility")
            print("[osu] patched slider's mania endTime:hitSample parser", flush=True)
    except ImportError:
        pass

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[osu] device={device}  audio={audio}  dirs={[str(d) for d in dirs]}")
    processor = CM3PProcessor.from_pretrained(rev)
    parser_cls = type(processor.beatmap_parser)
    if args.mania_columns == "on":
        cw.install_mania_column_patch(parser_cls)
        print("[core] installed the opt-in mania-column patch (1-based column, "
              "Group.mania_column). The published checkpoint never saw these tokens: "
              "validate the embeddings before trusting a comparison.", flush=True)
    crop = cw.CropSpec.from_name(args.crop)
    crop.apply(processor.beatmap_parser)
    snap = cw.read_processor_snapshot(processor)
    crop = cw.CropSpec.read(processor.beatmap_parser)   # actual kept flags
    crop.check()
    cw.assert_processor_config(snap, strict=not args.allow_config_mismatch)
    tids = cw.token_ids(processor)
    print(f"[T0] beatmap kwargs: window={snap.window_length_sec:g}s "
          f"stride={snap.window_stride_sec:g}s max_length={snap.max_length} "
          f"truncation={snap.truncation} tokenizer.max_time={snap.tokenizer_max_time}ms "
          f"time_step={snap.tokenizer_time_step}ms add_cls={snap.tokenizer_add_cls}",
          flush=True)
    print(f"[T1] crop={crop.name}: hitsounds={crop.add_hitsounds} kiai={crop.add_kiai} "
          f"mania_sv={crop.add_mania_sv} timing={crop.add_timing} "
          f"timing_points={crop.add_timing_points} snapping={crop.add_snapping}",
          flush=True)
    if args.windows == "on" and audio:
        raise SystemExit(
            "--audio on is not supported by the T2 window export (it tokenises with "
            "audio=None, so the exported vectors would silently differ from the main "
            "table). Use --audio off, or --windows off for the legacy audio path.")

    attn = os.environ.get("CM3P_ATTN", "sdpa")
    try:
        model = CM3PModel.from_pretrained(rev, trust_remote_code=True, revision="main",
                                          attn_implementation=attn)
    except Exception as exc:                                   # noqa: BLE001
        print(f"[osu] attn_implementation={attn!r} rejected ({exc}); retrying without it",
              flush=True)
        model = CM3PModel.from_pretrained(rev, trust_remote_code=True, revision="main")
    mdt = torch.bfloat16 if device == "cuda" else torch.float32
    model = model.to(device=device, dtype=mdt).eval()
    print(f"[osu] model dtype {mdt}, attn={attn}", flush=True)

    keep = {int(b) for b in want["beatmap_id"].tolist()}
    csum = {int(b): c for b, c in zip(want["beatmap_id"], want["checksum"])}
    found = _locate_osu(dirs, keep)
    missing = sorted(keep - set(found))
    if missing:
        print(f"[osu] WARNING: {len(missing):,} wanted charts have no .osu under "
              f"{[str(d) for d in dirs]} (e.g. {missing[:5]})", flush=True)
    files: dict[int, Path] = dict(found)

    print(f"[osu] hashing {len(files):,} .osu files for content_md5", flush=True)
    t_h = time.time()
    md5_map = {bid: _md5_of(p) for bid, p in files.items()}
    print(f"[osu] hashed in {time.time() - t_h:.1f}s", flush=True)

    cfg_payload = {
        "tool": "step0c-cm3p-v4",
        "crop": crop.to_dict(),
        "snapshot": snap.to_dict(),
        "mania_columns": args.mania_columns,
        "columns_patch": cw.mania_column_patch_installed(parser_cls),
        "model": rev, "dtype": str(mdt), "attn": attn, "audio": bool(audio),
        # batching/seed are deliberately NOT part of the hash: windows are encoded
        # independently (no cross-window attention), so a cache can be reused after
        # changing --window-batch.  dtype and max_length change the vectors.
        "window": {"dtype": args.window_dtype, "max_length": int(snap.max_length)},
        "cm3p_commit": _git_commit(root),
        "torch": str(torch.__version__),
    }
    try:
        import transformers
        cfg_payload["transformers"] = str(transformers.__version__)
    except Exception:
        pass
    cfg = cw.cfg_hash(cfg_payload)

    if audio:
        print("[core] NOTE: the core audit tokenises with audio=None; the audio token "
              "prefix is not part of the audit, but the note/timing stream it checks "
              "is the same.", flush=True)
    audit_meta = _core_audit(files, processor, snap, tids, crop, args, snap.max_length)
    meta_base = {
        "tool": "step0c_chart_content/v4", "cfg_hash": cfg,
        "crop": crop.to_dict(), "snapshot": snap.to_dict(),
        "mania_columns": args.mania_columns,
        "columns_patch": cw.mania_column_patch_installed(parser_cls),
        "core_audit": audit_meta, "model": rev, "dtype": str(mdt), "attn": attn,
    }

    if args.windows == "on":
        encode = _encode_fn(model, device, torch)
        table, meta = _export_windows_table(
            files, md5_map, csum, processor, snap, tids, crop, args, rev, mdt,
            attn, cfg, encode, torch)
        meta = {**meta_base, **meta}
        print(f"[win] index: {args.window_index}  windows: {args.window_dir}",
              flush=True)
        return table, meta

    # ---------------------------------------------------------- legacy loader path
    def _chart_encode(inputs):
        with torch.no_grad():
            out = model(**inputs, return_loss=False)
        return out.beatmap_embeds.detach().float().cpu().numpy()
    table = _dataset_table(files, csum, processor, crop, audio, batch, device,
                           rev, args, md5_map, _chart_encode)
    meta = {**meta_base, "windows": None}
    return table, meta


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
    ap.add_argument("--meta-dir", type=Path, default=PROC,
                    help="beatmap_meta_{tag}.csv 所在目录（默认 data/processed）")
    ap.add_argument("--ids-file", default=None,
                    help="beatmap_id[,checksum] per line (header optional).  Overrides "
                         "beatmap_meta_{tag}.csv -- use this on a fresh clone, e.g. "
                         "manifests/whitelist_1k.csv (the whole whitelist) or "
                         "manifests/cm3p_todo_1k.csv (only what the HF table misses).  "
                         "Rows with cacheable=False are skipped when the column exists")
    ap.add_argument("--in", dest="in_path", default=None,
                    help="--source table: an already-computed content table to merge in")
    ap.add_argument("--infer-batch", type=int, default=8,
                    help="--windows off only: beatmaps per forward pass (NOT --batch, rows)")
    ap.add_argument("--crop", default="canon", choices=["canon", "legacy"],
                    help="T1: canon turns hitsounds/kiai/mania-SV off; legacy reproduces "
                         "the old table")
    ap.add_argument("--mania-columns", default="auto", choices=["auto", "on", "off"],
                    help="auto detects the upstream Group.column bug and leaves columns "
                         "off (checkpoint parity); on installs the opt-in patch (OOD for "
                         "the checkpoint); off silences the check")
    ap.add_argument("--windows", default="on", choices=["on", "off"],
                    help="T2: export per-window .npy + index (on) or use the legacy "
                         "BeatmapFilesDataset loader (off)")
    ap.add_argument("--window-dir", default=".cache/cm3p/win",
                    help="T2: directory for <beatmap_id>.npy")
    ap.add_argument("--window-index", default=".cache/cm3p/index.parquet",
                    help="T2: window index parquet")
    ap.add_argument("--window-batch", type=int, default=16,
                    help="T2: windows per forward pass")
    ap.add_argument("--window-buffer", type=int, default=4096,
                    help="T2: windows buffered before a bucket is flushed")
    ap.add_argument("--window-dtype", default="float16", choices=["float16", "float32"],
                    help="T2: on-disk window dtype (float16 = 218 MB for 21,949 charts)")
    ap.add_argument("--window-seed", type=int, default=0, help="T2: bucket shuffle seed")
    ap.add_argument("--refresh-windows", action="store_true",
                    help="T2: delete the window index and every .npy before rebuilding")
    ap.add_argument("--on-truncate", default="warn", choices=["warn", "error", "allow"],
                    help="T2: what to do when a window exceeds max_length and loses core "
                         "tokens (warn/error/allow)")
    ap.add_argument("--verify-windows", type=int, default=64,
                    help="T2: parity-check N charts by reading the .npy back; 0 = all")
    ap.add_argument("--core-audit", type=int, default=16,
                    help="T1: tokenise N charts with legacy+canon and prove the core "
                         "stream is identical (0 = off)")
    ap.add_argument("--allow-config-mismatch", action="store_true",
                    help="T0: run with the real preprocessor window/max_length even if "
                         "they are not 16/4000")
    ap.add_argument("--allow-mixed-crop", action="store_true",
                    help="allow merging rows from a different crop/cfg into --out")
    ap.add_argument("--hf-path", default="data/raw/cm3p_embeddings/beatmap_embeddings.parquet")
    ap.add_argument("--model-repo", default=CM3P_REPO,
                    help="CM3P model repo id or local from_pretrained directory")
    ap.add_argument("--osu-dir", action="append", default=None,
                    help="directory (or .osu/.osz file) to embed; repeatable; --source osu-dir")
    ap.add_argument("--audio", default="off", choices=["on", "off"],
                    help="osu-dir only: fuse audio.  MUST NOT be mixed with hf-precomputed rows "
                         "(they were computed with audio); the source column records it")
    ap.add_argument("--out", default=None,
                    help="output parquet (default: chart_content_cm3p.parquet for "
                         "hf-precomputed; chart_content_cm3p_local_canon.parquet for "
                         "osu-dir+canon; chart_content_cm3p_local.parquet for "
                         "osu-dir+legacy)")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore the existing table and recompute every id")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--limit", type=int, default=0, help="debug: only this many ids")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    t0 = time.time()
    if args.out is None:
        if args.source == "osu-dir":
            name = ("chart_content_cm3p_local_canon.parquet" if args.crop == "canon"
                    else "chart_content_cm3p_local.parquet")
        else:
            name = "chart_content_cm3p.parquet"
        args.out = str(PROC / name)
    out = Path(args.out)
    if args.source == "osu-dir":
        _precheck_sidecar(out, args, args.allow_mixed_crop)

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

    meta = None
    if args.source == "table":
        # merge mode: the incoming table is the whole point, whether or not anything is
        # "missing" relative to the whitelist
        if not args.in_path:
            raise SystemExit("--source table needs --in PATH")
        new = load_table(args.in_path)
    elif len(missing) == 0:
        if args.source == "osu-dir" and args.windows == "on":
            if not args.osu_dir:
                raise SystemExit("--source osu-dir needs at least one --osu-dir")
            target = want if not args.limit else want.head(args.limit)
            print(f"main table complete; exporting/verifying the T2 window store for "
                  f"{len(target):,} charts", flush=True)
            new, meta = from_osu_dir(args.osu_dir, target, args.model_repo,
                                     args.audio == "on", args.infer_batch,
                                     args.device, args)
        else:
            print("nothing to do; the table is already complete for this whitelist")
            new = empty_table()
    elif args.source == "hf-precomputed":
        src = Path(args.hf_path)
        if not src.exists():
            raise SystemExit(f"{src} not found -- see legacy/docs/cm3p_step2_integration_plan.md M1")
        new = from_hf_precomputed(src, missing, CM3P_REPO, args.batch)
    else:
        if not args.osu_dir:
            raise SystemExit("--source osu-dir needs at least one --osu-dir")
        new, meta = from_osu_dir(args.osu_dir, missing, args.model_repo, args.audio == "on",
                                 args.infer_batch, args.device, args)

    if meta is not None:
        _check_merge_guard(out, meta, args.allow_mixed_crop)
    table = merge(base, new)
    write_atomic(table, out)
    if meta is not None:
        cw.write_meta(_sidecar(out), {**meta, "out": str(out), "n_rows": int(len(table))})
        print(f"[cfg] {meta.get('cfg_hash')} crop={meta.get('crop')} "
              f"windows={meta.get('windows')}")

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
