#!/usr/bin/env python
"""Build the VSRG-DanEstimation-Benchmark manifest for DCR.

One row per samples/data.csv row, with the .osu path, rate, chart group and
metadata needed by the rest of the benchmark pipeline.  This script never
inspects chart content beyond the .osu header lines (Mode/CircleSize/IDs/audio
name); it is not a place for per-chart tuning.

    python scripts/benchmark_manifest.py \
        --benchmark .cache/benchmark/VSRG-DanEstimation-Benchmark \
        --out data/processed/benchmark_manifest.parquet

The archive is expected at <benchmark>/samples/samples.7z.  If
<benchmark>/extracted does not exist, it is unpacked with bsdtar (the local
machine has no 7z binary).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

RATE_RE = re.compile(r"[xX]([0-9]+(?:\.[0-9]+)?)\s*$")
SECTION_RE = re.compile(r"^\[([^\]]+)\]$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_header(path: Path) -> dict:
    """Read only [General]/[Metadata] keys; stop at [HitObjects]."""
    out: dict[str, str] = {}
    section = ""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip().lstrip("\ufeff")
            m = SECTION_RE.match(line)
            if m:
                section = m.group(1)
                if section == "HitObjects":
                    break
                continue
            if section in ("General", "Metadata", "Difficulty") and ":" in line:
                key, value = line.split(":", 1)
                out[f"{section}.{key.strip()}"] = value.strip()
    return out


def safe_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_rate_text(text: str) -> float | None:
    """Parse 1.25x / x1.25 / _1_25x; ignore implausible values."""
    if not text:
        return None
    # audio filenames often use _1_10x for 1.10x
    for whole, frac in re.findall(r"(\d)_(\d{2,3})\s*[xX]", text):
        value = float(f"{whole}.{frac}")
        if 0.5 <= value <= 2.0:
            return value
    for value in re.findall(r"(\d+(?:\.\d+)?)\s*[xX]", text):
        rate = float(value)
        if 0.5 <= rate <= 2.0:
            return rate
    for value in re.findall(r"[xX]\s*(\d+(?:\.\d+)?)", text):
        rate = float(value)
        if 0.5 <= rate <= 2.0:
            return rate
    return None


def rate_from_row(name: str, header: dict) -> tuple[float, str]:
    """Return (rate, source).  Names win; .osu fields are a cross-check."""
    for source, text in (("name", name),
                         ("Metadata.Version", header.get("Metadata.Version", "")),
                         ("General.AudioFilename", header.get("General.AudioFilename", ""))):
        rate = parse_rate_text(text)
        if rate is not None:
            return rate, source
    return 1.0, "default"


def chart_group(row: pd.Series) -> str:
    bid = str(row.get("bid") or "").strip()
    if bid and bid.lower() != "nan":
        return f"bid:{bid}"
    name = re.sub(r"\s*(?:[xX]\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*[xX])\s*$", "", str(row["name"])).strip()
    name = re.sub(r"\s+", " ", name).casefold()
    return f"name:{name}"


def unpack(benchmark: Path) -> Path:
    extracted = benchmark / "extracted"
    if extracted.exists() and any(extracted.rglob("*.osu")):
        return extracted
    archive = benchmark / "samples" / "samples.7z"
    if not archive.exists():
        raise SystemExit(f"missing {archive}")
    extracted.mkdir(parents=True, exist_ok=True)
    subprocess.run(["bsdtar", "-xf", str(archive), "-C", str(extracted)], check=True)
    return extracted


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default=".cache/benchmark/VSRG-DanEstimation-Benchmark")
    ap.add_argument("--out", default="data/processed/benchmark_manifest.parquet")
    ap.add_argument("--meta-out", default=None)
    args = ap.parse_args(argv)

    benchmark = Path(args.benchmark).resolve()
    data_csv = benchmark / "samples" / "data.csv"
    if not data_csv.exists():
        raise SystemExit(f"missing {data_csv}")
    extracted = unpack(benchmark)

    df = pd.read_csv(data_csv, dtype=str, keep_default_na=False)
    needed = ["bid", "name", "pattern", "subPattern", "expected"]
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"{data_csv}: missing columns {missing_cols}")
    if len(df) != 1382:
        print(f"[warn] data.csv has {len(df)} rows, expected 1,382", flush=True)

    rows = []
    missing_files = []
    for i, row in df.iterrows():
        pattern = str(row["pattern"]).strip()
        name = str(row["name"]).strip()
        rel = Path(pattern) / f"{name}.osu"
        path = extracted / rel
        if not path.exists():
            # fallback: normalized basename search within the pattern directory
            target = re.sub(r"\s+", " ", name).strip().casefold()
            candidates = [p for p in (extracted / pattern).rglob("*.osu")
                          if re.sub(r"\s+", " ", p.stem).strip().casefold() == target]
            if not candidates:
                missing_files.append(str(rel))
                continue
            path = candidates[0]
        header = parse_header(path)
        mode = safe_int(header.get("General.Mode"))
        keys = safe_float(header.get("Difficulty.CircleSize"))
        rate, rate_source = rate_from_row(name, header)
        bid = str(row["bid"]).strip()
        if bid.lower() == "nan":
            bid = ""
        rows.append({
            "row_id": f"b{i:04d}",
            "bid": bid,
            "name": name,
            "pattern": pattern,
            "subPattern": str(row["subPattern"]).strip(),
            "expected": float(row["expected"]),
            "family": ("course" if pattern == "course" else
                       "ln" if pattern == "ln" else "rc"),
            "rate": rate,
            "rate_source": rate_source,
            "osu_path": str(path.relative_to(extracted)),
            "osu_md5": md5(path),
            "osu_sha256": sha256(path),
            "osu_mode": mode,
            "osu_keys": keys,
            "osu_beatmap_id": safe_int(header.get("Metadata.BeatmapID")),
            "osu_beatmapset_id": safe_int(header.get("Metadata.BeatmapSetID")),
            "audio_filename": header.get("General.AudioFilename", ""),
        })

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise SystemExit("no .osu files matched")
    out_df["chart_group"] = out_df.apply(chart_group, axis=1)
    out_df["row_id"] = [f"b{i:04d}" for i in range(len(out_df))]

    bad_mode = out_df[out_df["osu_mode"].fillna(-1).astype(int) != 3]
    bad_keys = out_df[out_df["osu_keys"].fillna(-1).astype(float) != 4.0]
    if len(bad_mode) or len(bad_keys):
        print(f"[warn] non-4K/Mode3 rows: mode={len(bad_mode)} keys={len(bad_keys)}",
              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)

    extra = sorted(p for p in extracted.rglob("*.osu")
                   if p.relative_to(extracted).as_posix() not in set(out_df["osu_path"]))
    meta = {
        "benchmark": str(benchmark),
        "benchmark_commit": subprocess.run(
            ["git", "-C", str(benchmark), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "data_csv_sha256": sha256(data_csv),
        "samples_7z_sha256": sha256(benchmark / "samples" / "samples.7z")
        if (benchmark / "samples" / "samples.7z").exists() else None,
        "n_rows": int(len(out_df)),
        "n_missing_files": int(len(missing_files)),
        "missing_files": missing_files[:20],
        "n_extra_osu": int(len(extra)),
        "extra_osu": [p.relative_to(extracted).as_posix() for p in extra[:20]],
        "family_counts": out_df["family"].value_counts().to_dict(),
        "pattern_counts": out_df["pattern"].value_counts().to_dict(),
        "rate_counts": out_df["rate"].value_counts().sort_index().to_dict(),
        "n_unique_chart_groups": int(out_df["chart_group"].nunique()),
        "n_with_bid": int((out_df["bid"] != "").sum()),
        "n_unique_bids": int(out_df.loc[out_df["bid"] != "", "bid"].nunique()),
    }
    meta_out = Path(args.meta_out) if args.meta_out else Path(str(out) + ".meta.json")
    meta_out.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {out}: {len(out_df):,} rows, {len(missing_files)} missing .osu, "
          f"{len(extra)} extra .osu", flush=True)
    print(json.dumps({k: meta[k] for k in ("family_counts", "n_with_bid",
                                           "n_unique_bids", "n_unique_chart_groups")},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
