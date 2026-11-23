#!/usr/bin/env python3
"""Export the DCR content table (md5 -> Reform Dan) consumed by the ManiaMapAnalyser plugin.

Two modes
---------
Build the binary table (needs the 512-d content vectors AND the two signal columns):

    python deploy/export_dcr_table.py \
        --content data/processed/chart_content_cm3p_local.parquet \
        --signals data/processed/rank_signals_1k.parquet

Only dump the two signal columns, when someone has the model but not this repo:

    python deploy/export_dcr_table.py --dump-signals signals_1k.csv

Output layout — must stay byte-compatible with the client parser
(`js/estimator/dcrContentTable.js`):

    header 16 B : "DCRT" | u16 format_version | u16 record_size | u32 count | u32 reserved
    record 28 B : md5[16] | f32 dan | f32 d | f32 sunny        (sorted by md5, binary search)

The forward pass is deliberately NOT re-implemented here: `deploy/dan_predict.py` is
imported and called, and `--model` repoints its module table. Any future change on the
model side therefore propagates to this exporter verbatim instead of silently diverging.

Exit codes: 0 ok, 1 bad usage/IO, 2 contract violation (wrong model for these vectors,
missing columns, non-finite values, ...).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

MAGIC = b"DCRT"
FORMAT_VERSION = 1
RECORD_SIZE = 28
MD5_SIZE = 16
HEADER = struct.Struct("<4sHHII")
RECORD = struct.Struct("<16sfff")

CONTENT_DIM = 512
EXTRA_NAMES = ("msd_overall", "sunny_star")
VECTOR_DIM = CONTENT_DIM + len(EXTRA_NAMES)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dan_predict as dp  # noqa: E402  (same directory, after sys.path tweak)


def die(msg: str, code: int = 2) -> "NoReturn":  # type: ignore[valid-type]
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dan_predict(model_path: Path, kind: str):
    """Point dan_predict at `model_path` and return (cfg, cm3p_to_dan|d2v_to_dan)."""
    cfg = json.loads(model_path.read_text(encoding="utf-8"))
    if int(cfg["content_dim"]) != CONTENT_DIM:
        die(f"{model_path.name}: content_dim={cfg['content_dim']} != {CONTENT_DIM}")
    if tuple(cfg["extra_names"]) != EXTRA_NAMES:
        die(f"{model_path.name}: extra_names={cfg['extra_names']} != {list(EXTRA_NAMES)}")
    if len(cfg["w"]) != VECTOR_DIM:
        die(f"{model_path.name}: len(w)={len(cfg['w'])} != {VECTOR_DIM}")
    if cfg.get("content_type") != kind:
        die(f"{model_path.name}: content_type={cfg.get('content_type')!r} != {kind!r}")

    if model_path.resolve() != dp._MODEL_FILES[kind].resolve():
        dp._MODEL_FILES[kind] = model_path
        dp._load.cache_clear()

    predict = {"cm3p": dp.cm3p_to_dan, "d2v": dp.d2v_to_dan}[kind]
    return cfg, predict


def dump_signals(signals_path: Path, out_path: Path) -> int:
    import pandas as pd

    if not signals_path.exists():
        die(f"{signals_path} not found")
    sig = pd.read_parquet(signals_path)
    missing = [c for c in ("beatmap_id", *EXTRA_NAMES) if c not in sig.columns]
    if missing:
        die(f"{signals_path.name}: missing column(s) {missing}; has {list(sig.columns)[:12]}")

    out = sig[["beatmap_id", *EXTRA_NAMES]].copy()
    out["beatmap_id"] = out["beatmap_id"].astype("int64")
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}  ({len(out):,} rows, {out_path.stat().st_size / 1024:.0f} KB)")
    for col in EXTRA_NAMES:
        s = out[col]
        print(f"  {col:12s} min {s.min():8.3f}  p50 {s.median():8.3f}  max {s.max():8.3f}  "
              f"nan {int(s.isna().sum())}")
    return 0


def build_table(args: argparse.Namespace) -> int:
    import numpy as np
    import pandas as pd

    model_path = Path(args.model)
    content_path = Path(args.content)
    signals_path = Path(args.signals)
    whitelist_path = Path(args.whitelist)
    for p in (model_path, content_path, signals_path, whitelist_path):
        if not p.exists():
            die(f"{p} not found")

    cfg = json.loads(model_path.read_text(encoding="utf-8"))
    kind = cfg["content_type"]
    module_cfg, predict = load_dan_predict(model_path, kind)
    cal = module_cfg["calibration"]

    # --- guard 1: the vectors must be the ones this model was fitted on -------------
    actual_md5 = file_md5(content_path)
    if actual_md5 != module_cfg["content_md5"]:
        if args.skip_md5_check:
            print(f"warning: content md5 mismatch ignored (--skip-md5-check): "
                  f"{actual_md5} != {module_cfg['content_md5']}", file=sys.stderr)
        else:
            die(f"{content_path.name} md5 {actual_md5} != model content_md5 "
                f"{module_cfg['content_md5']}\n"
                f"        this model was fitted on different vectors; pass the matching "
                f"parquet or --skip-md5-check if you know what you are doing")

    # --- load ---------------------------------------------------------------------------
    content = pd.read_parquet(content_path, columns=["beatmap_id", "embedding"])
    content["beatmap_id"] = content["beatmap_id"].astype("int64")
    if content["beatmap_id"].duplicated().any():
        die("content table has duplicate beatmap_id")

    sig = pd.read_parquet(signals_path, columns=["beatmap_id", *EXTRA_NAMES])
    sig["beatmap_id"] = sig["beatmap_id"].astype("int64")
    if sig["beatmap_id"].duplicated().any():
        die(f"{signals_path.name} has duplicate beatmap_id")

    wl = pd.read_csv(whitelist_path, usecols=["beatmap_id", "checksum"])
    wl["beatmap_id"] = wl["beatmap_id"].astype("int64")
    wl = wl.drop_duplicates("beatmap_id")
    if not wl["checksum"].astype(str).str.fullmatch(r"[0-9a-fA-F]{32}").all():
        die(f"{whitelist_path.name}: some checksum values are not 32-hex md5")

    merged = wl.merge(content, on="beatmap_id", how="inner").merge(sig, on="beatmap_id", how="inner")
    dropped = {
        "no_content": int(len(wl) - len(wl.merge(content[["beatmap_id"]], on="beatmap_id"))),
        "not_in_whitelist": int(len(content) - len(content.merge(wl[["beatmap_id"]], on="beatmap_id"))),
        "no_signals": int(len(wl.merge(content[["beatmap_id"]], on="beatmap_id"))
                          - len(merged)),
    }
    if args.limit:
        merged = merged.head(args.limit)
    if merged.empty:
        die("nothing to export: whitelist ∩ content ∩ signals is empty")

    # --- forward pass -------------------------------------------------------------------
    rows: list[dict] = []
    for rec in merged.itertuples(index=False):
        emb = rec.embedding
        if len(emb) != CONTENT_DIM:
            die(f"beatmap_id={rec.beatmap_id}: embedding dim {len(emb)} != {CONTENT_DIM}")
        msd, sunny = float(rec.msd_overall), float(rec.sunny_star)
        if not (math.isfinite(msd) and math.isfinite(sunny)):
            die(f"beatmap_id={rec.beatmap_id}: non-finite signal "
                f"(msd_overall={msd}, sunny_star={sunny})")
        vec = [float(v) for v in emb]
        vec.append(msd)
        vec.append(sunny)
        d = float(dp._raw_d(kind, vec))
        dan = float(dp._isotonic(d, cal["x"], cal["y"]))
        rows.append({"md5": str(rec.checksum).lower(), "dan": dan, "d": d, "sunny": sunny})

    # The shipped number must be exactly what the public API returns; spot-check a stride
    # of rows (private `_raw_d` is used above only to avoid running the forward pass twice).
    stride = max(1, len(rows) // 200)
    worst = 0.0
    for i in range(0, len(rows), stride):
        rec = merged.iloc[i]
        vec = [float(v) for v in rec.embedding] + [float(rec.msd_overall), float(rec.sunny_star)]
        worst = max(worst, abs(predict(vec) - rows[i]["dan"]))
    if worst > 1e-9:
        die(f"internal inconsistency: exporter vs dan_predict differ by {worst:.2e}")

    # --- write --------------------------------------------------------------------------
    rows.sort(key=lambda r: bytes.fromhex(r["md5"]))
    buf = bytearray(HEADER.pack(MAGIC, FORMAT_VERSION, RECORD_SIZE, len(rows), 0))
    for r in rows:
        buf += RECORD.pack(bytes.fromhex(r["md5"]), r["dan"], r["d"], r["sunny"])

    out_bin = Path(args.out_dir) / f"{args.out_name}.bin"
    out_manifest = Path(args.out_dir) / f"{args.out_name}.manifest.json"
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    out_bin.write_bytes(buf)

    dans = np.array([r["dan"] for r in rows], dtype=float)
    reform_version = args.reform_version or f"{cfg['model_id'][:8]}-{cfg['created']}"
    manifest = {
        "format": "dcr-content",
        "format_version": FORMAT_VERSION,
        "record_size": RECORD_SIZE,
        "count": len(rows),
        "table_bytes": len(buf),
        "sha256": hashlib.sha256(buf).hexdigest(),
        "built_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "stub": False,
        "reform_version": reform_version,
        "model": {
            "content_type": kind,
            "model_id": cfg["model_id"],
            "content_md5": cfg["content_md5"],
            "created": cfg.get("created"),
            "train_git_commit": cfg.get("train_git_commit"),
        },
        "source": {
            "content": content_path.name,
            "signals": signals_path.name,
            "whitelist": whitelist_path.name,
        },
        "whitelist_count": int(len(wl)),
        "coverage_of_whitelist": round(len(rows) / max(len(wl), 1), 6),
        "dropped": dropped,
        "distribution": {
            "dan_min": round(float(dans.min()), 4),
            "dan_p01": round(float(np.percentile(dans, 1)), 4),
            "dan_p25": round(float(np.percentile(dans, 25)), 4),
            "dan_p50": round(float(np.percentile(dans, 50)), 4),
            "dan_p75": round(float(np.percentile(dans, 75)), 4),
            "dan_p99": round(float(np.percentile(dans, 99)), 4),
            "dan_max": round(float(dans.max()), 4),
            "count_ge_18_5": int((dans >= 18.5).sum()),
        },
        "fields": {
            "dan": "Reform Dan, isotonic-calibrated (integer = mid of that dan)",
            "d": "raw Bradley-Terry latent (diagnostics only)",
            "sunny": "training-era sunny_star (diagnostics only)",
        },
    }
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")

    # --- read back ----------------------------------------------------------------------
    raw = out_bin.read_bytes()
    magic, ver, rec_size, count, _ = HEADER.unpack_from(raw, 0)
    if (magic, ver, rec_size, count) != (MAGIC, FORMAT_VERSION, RECORD_SIZE, len(rows)):
        die("round-trip check failed: header mismatch")
    if len(raw) != HEADER.size + count * RECORD_SIZE:
        die("round-trip check failed: size mismatch")
    keys = [raw[HEADER.size + i * RECORD_SIZE: HEADER.size + i * RECORD_SIZE + MD5_SIZE]
            for i in range(count)]
    if any(keys[i] > keys[i + 1] for i in range(count - 1)):
        die("round-trip check failed: records are not sorted by md5")

    print(f"wrote {out_bin}  ({len(buf):,} B, {len(rows):,} rows, "
          f"{len(buf) / 1024:.0f} KB)")
    print(f"wrote {out_manifest}")
    print(f"model            {kind} / {cfg['model_id']} (MAE OOF "
          f"{cal.get('mae_oof') if cal.get('mae_oof') is None else round(cal['mae_oof'], 4)})")
    print(f"coverage         {len(rows):,}/{len(wl):,} of 4K whitelist "
          f"({manifest['coverage_of_whitelist']:.2%})")
    print(f"dropped          no_content={dropped['no_content']:,} "
          f"no_signals={dropped['no_signals']:,} "
          f"content_outside_whitelist={dropped['not_in_whitelist']:,}")
    print(f"dan              min {dans.min():.2f}  p50 {np.percentile(dans, 50):.2f}  "
          f"max {dans.max():.2f}   >=18.5: {int((dans >= 18.5).sum())}")
    if (dans >= 18.5).any():
        print("warning: some rows are >= 18.5; MMA hides numeric values at that ceiling",
              file=sys.stderr)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Export the DCR md5->Dan table for the ManiaMapAnalyser plugin.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Output layout")[0].strip(),
    )
    ap.add_argument("--model", default=str(HERE / "models" / "cm3p_linear_extra.json"),
                    help="deploy model json (default: cm3p)")
    ap.add_argument("--content", default="data/processed/chart_content_cm3p_local.parquet",
                    help="512-d content vectors (parquet: beatmap_id, embedding)")
    ap.add_argument("--signals", default="data/processed/rank_signals_1k.parquet",
                    help="msd_overall + sunny_star per chart (parquet: beatmap_id, ...)")
    ap.add_argument("--whitelist", default="manifests/whitelist_1k.csv",
                    help="4K whitelist with checksum (beatmap_id, checksum)")
    ap.add_argument("--out-dir", default=str(HERE / "table"),
                    help="output directory (default: deploy/table)")
    ap.add_argument("--out-name", default="dcr_content", help="output basename")
    ap.add_argument("--reform-version", default=None,
                    help="version stamped into the manifest (default: <model_id>-<created>)")
    ap.add_argument("--limit", type=int, default=0, help="export only the first N rows (debug)")
    ap.add_argument("--skip-md5-check", action="store_true",
                    help="allow a content parquet whose md5 differs from the model's")
    ap.add_argument("--dump-signals", metavar="CSV", default=None,
                    help="only write beatmap_id,msd_overall,sunny_star to CSV and exit")
    args = ap.parse_args(argv)

    if args.dump_signals:
        return dump_signals(Path(args.signals), Path(args.dump_signals))
    return build_table(args)


if __name__ == "__main__":
    raise SystemExit(main())
