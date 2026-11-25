#!/usr/bin/env python3
"""Inventory the eight Reform packs and construct the confirmed anchor manifest.

The input archives are the full beatmapsets listed in the Reform calibration
plan.  This script extracts only the original .osu files; audio remains inside
the source .osz archives.  It records manual split requirements rather than
silently substituting charts when an expected single-chart file is absent.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PACKS = {
    1079991: ("course_1st", "total"),
    887733: ("course_2nd", "total"),
    1156299: ("course_final", "total"),
    2234779: ("course_final_ext", "total"),
    2316976: ("jack", "individual"),
    2320002: ("speed", "individual"),
    2317095: ("stamina", "individual"),
    2316993: ("tech", "individual"),
}
GREEK = (
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
    "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
)
GREEK_LOOKUP = {name: i for i, name in enumerate(GREEK)}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_osu(data: bytes) -> dict:
    text = data.decode("utf-8-sig", errors="replace")
    sections: dict[str, list[str]] = defaultdict(list)
    section = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
        elif section:
            sections[section].append(line)
    values: dict[str, str] = {}
    for line in sections.get("General", []):
        if ":" in line:
            k, v = line.split(":", 1)
            values[f"General.{k.strip()}"] = v.strip()
    for line in sections.get("Metadata", []):
        if ":" in line:
            k, v = line.split(":", 1)
            values[f"Metadata.{k.strip()}"] = v.strip()
    for line in sections.get("Difficulty", []):
        if ":" in line:
            k, v = line.split(":", 1)
            values[f"Difficulty.{k.strip()}"] = v.strip()
    hit_objects = [line for line in sections.get("HitObjects", []) if line.strip()]
    timing_points = [line for line in sections.get("TimingPoints", []) if line.strip()]
    return {
        "text": text,
        "mode": values.get("General.Mode", ""),
        "keys": values.get("Difficulty.CircleSize", ""),
        "audio_filename": values.get("General.AudioFilename", ""),
        "beatmap_id": values.get("Metadata.BeatmapID", ""),
        "beatmapset_id": values.get("Metadata.BeatmapSetID", ""),
        "artist": values.get("Metadata.Artist", ""),
        "title": values.get("Metadata.Title", ""),
        "creator": values.get("Metadata.Creator", ""),
        "version": values.get("Metadata.Version", ""),
        "hit_objects": len(hit_objects),
        "timing_points": len(timing_points),
    }


def parse_difficulty(version: str) -> tuple[str, float] | tuple[None, None]:
    s = version.casefold()
    m = re.search(r"\bintro\s*[-_ ]*(\d+)(?:st|nd|rd|th)?\b", s)
    if m:
        n = int(m.group(1))
        return f"intro_{n}", n + 0.5
    m = re.search(r"\b(\d+)(?:st|nd|rd|th)\b", s)
    if m:
        n = int(m.group(1))
        return f"regular_{n}", n + 3.5
    for name, index in GREEK_LOOKUP.items():
        if re.search(rf"\b{name}\b", s):
            epsilon = GREEK_LOOKUP["epsilon"]
            target = 14.5 + index if index < epsilon else 19.5 + (index - epsilon)
            return f"greek_{name}", target
    return None, None


def safe_archive_path(name: str) -> str | None:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or not name.casefold().endswith(".osu"):
        return None
    return p.name


def inspect_archive(set_id: int, archive: Path, chart_root: Path) -> tuple[dict, list[dict]]:
    raw = archive.read_bytes()
    source_role, anchor_type = PACKS[set_id]
    record = {
        "beatmapset_id": set_id,
        "pack_role": source_role,
        "anchor_type": anchor_type,
        "archive": str(archive.relative_to(ROOT)),
        "archive_bytes": len(raw),
        "archive_sha256": sha256(raw),
        "download_source": f"https://txy1.sayobot.cn/beatmaps/download/full/{set_id}",
        "downloaded_at_utc": datetime.fromtimestamp(archive.stat().st_mtime, timezone.utc).isoformat(),
    }
    charts: list[dict] = []
    set_chart_dir = chart_root / str(set_id)
    set_chart_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        bad_member = zf.testzip()
        if bad_member:
            raise ValueError(f"{archive} has a CRC failure in {bad_member}")
        for member in zf.namelist():
            basename = safe_archive_path(member)
            if basename is None:
                continue
            data = zf.read(member)
            info = parse_osu(data)
            if info["beatmapset_id"] and info["beatmapset_id"].isdigit():
                actual_set = int(info["beatmapset_id"])
                if actual_set != set_id:
                    raise ValueError(f"archive {set_id} contains chart for set {actual_set}: {member}")
            try:
                beatmap_id = int(info["beatmap_id"])
            except ValueError:
                beatmap_id = None
            mode_ok = info["mode"] == "3" and info["keys"] == "4"
            if not mode_ok:
                continue
            chart_path = set_chart_dir / f"{beatmap_id or sha256(data)[:12]}.osu"
            chart_path.write_bytes(data)
            difficulty, target = parse_difficulty(info["version"])
            is_placeholder = "delete" in info["version"].casefold()
            is_old_speed5 = (set_id == 2320002 and difficulty == "regular_5"
                             and "old" in info["audio_filename"].casefold())
            include_anchor = not (is_placeholder or is_old_speed5)
            if is_placeholder:
                exclusion_reason = "delete_placeholder"
            elif is_old_speed5:
                exclusion_reason = "prior_speed_5_audio_variant; six-dan union is the only confirmed variant exception"
            else:
                exclusion_reason = None
            charts.append({
                **{k: v for k, v in info.items() if k != "text"},
                "beatmap_id": beatmap_id,
                "beatmapset_id": set_id,
                "pack_role": source_role,
                "anchor_type": anchor_type,
                "difficulty_key": difficulty,
                "target": target,
                "osu_path": str(chart_path.relative_to(ROOT)),
                "osu_md5": hashlib.md5(data).hexdigest(),
                "osu_sha256": sha256(data),
                "osu_bytes": len(data),
                "is_4k_mania": True,
                "included_in_anchor_manifest": include_anchor,
                "exclusion_reason": exclusion_reason,
            })
    record["n_4k_charts"] = len(charts)
    return record, charts


def expected_counts(difficulty_key: str) -> Counter:
    if difficulty_key == "regular_6":
        return Counter({"course_1st": 1, "course_2nd": 1,
                        "jack": 1, "speed": 2, "stamina": 1, "tech": 1})
    return Counter({"course": 1, "jack": 1, "speed": 1, "stamina": 1, "tech": 1})


def slot_for(chart: dict) -> str:
    if chart["anchor_type"] == "total":
        return chart["pack_role"]
    return chart["pack_role"]


def coverage_report(charts: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for chart in charts:
        if chart["difficulty_key"] and chart["included_in_anchor_manifest"]:
            groups[chart["difficulty_key"]].append(chart)
    report = []
    for key in sorted(groups, key=lambda k: (groups[k][0]["target"], k)):
        rows = groups[key]
        counts = Counter(slot_for(r) for r in rows)
        expected = expected_counts(key)
        if key == "regular_6":
            expected_by_slot = expected
            observed_by_slot = counts
        else:
            total_n = sum(counts[s] for s in ("course_1st", "course_2nd", "course_final", "course_final_ext"))
            expected_by_slot = expected
            observed_by_slot = Counter({"course": total_n})
            observed_by_slot.update({k: counts[k] for k in ("jack", "speed", "stamina", "tech")})
        missing = []
        extra = []
        for slot, need in expected_by_slot.items():
            actual = observed_by_slot[slot]
            if actual < need:
                missing.extend([slot] * (need - actual))
            elif actual > need:
                extra.extend([slot] * (actual - need))
        total_rows = [r for r in rows if r["anchor_type"] == "total"]
        report.append({
            "difficulty_key": key,
            "target": rows[0]["target"],
            "observed_count": len(rows),
            "observed_by_pack": dict(counts),
            "expected_by_slot": dict(expected_by_slot),
            "expected_anchor_count": 7 if key == "regular_6" else 5,
            "missing_slots": missing,
            "extra_slots": extra,
            "manual_split_source_beatmap_ids": [r["beatmap_id"] for r in total_rows] if missing else [],
            "complete": not missing and not extra,
        })
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", type=Path, default=ROOT / "data/reform/raw")
    ap.add_argument("--chart-dir", type=Path, default=ROOT / "data/reform/charts")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/reform")
    args = ap.parse_args()
    set_records, all_charts = [], []
    for set_id in PACKS:
        archive = args.archive_dir / f"{set_id}.osz"
        if not archive.is_file():
            raise SystemExit(f"missing full mapset archive: {archive}")
        set_record, charts = inspect_archive(set_id, archive, args.chart_dir)
        set_records.append(set_record)
        all_charts.extend(charts)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    charts_path = args.out_dir / "charts.jsonl"
    with charts_path.open("w", encoding="utf-8") as f:
        for chart in all_charts:
            f.write(json.dumps(chart, ensure_ascii=False) + "\n")
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "beatmapsets": set_records,
        "n_4k_charts": len(all_charts),
        "n_anchor_candidate_charts": sum(c["included_in_anchor_manifest"] for c in all_charts),
        "n_unmapped_difficulties": sum(c["difficulty_key"] is None for c in all_charts),
        "coverage": coverage_report(all_charts),
    }
    (args.out_dir / "coverage.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "beatmapsets": [{"id": r["beatmapset_id"], "charts": r["n_4k_charts"],
                         "archive_bytes": r["archive_bytes"], "sha256": r["archive_sha256"]}
                        for r in set_records],
        "n_4k_charts": len(all_charts),
        "anchor_candidates": sum(c["included_in_anchor_manifest"] for c in all_charts),
        "unmapped": report["n_unmapped_difficulties"],
        "groups": len(report["coverage"]),
        "incomplete_groups": [r["difficulty_key"] for r in report["coverage"] if not r["complete"]],
        "charts_manifest": str(charts_path.relative_to(ROOT)),
        "coverage_report": str((args.out_dir / "coverage.json").relative_to(ROOT)),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
