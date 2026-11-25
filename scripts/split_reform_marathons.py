#!/usr/bin/env python3
"""Create the missing four single-chart Reform anchors from Marathon maps.

Cuts are hand-curated in data/reform/manual_splits.json.  The script rebases
hit objects and timing points, carries active timing/SV/kiai state to each new
chart, records provenance, and compiles the full 5/7-anchor manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]


def split_sections(text: str) -> tuple[list[str], list[tuple[str, list[str]]]]:
    lines = text.lstrip("\ufeff").splitlines()
    header: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    name = None
    body: list[str] = []
    for line in lines:
        m = re.fullmatch(r"\s*\[([^]]+)\]\s*", line)
        if m:
            if name is not None:
                sections.append((name, body))
            else:
                header = body
            name, body = m.group(1), []
        else:
            body.append(line)
    if name is not None:
        sections.append((name, body))
    else:
        header = body
    return header, sections


def section_map(sections: list[tuple[str, list[str]]]) -> dict[str, list[str]]:
    return {name: body for name, body in sections}


def parse_hit_object(line: str) -> tuple[float, int, float | None, list[str]]:
    fields = line.split(",")
    if len(fields) < 5:
        raise ValueError(f"malformed HitObject: {line}")
    time = float(fields[2])
    obj_type = int(fields[3])
    end_time = None
    if obj_type & 128:
        if len(fields) < 6:
            raise ValueError(f"malformed hold object: {line}")
        end_time = float(fields[5].split(":", 1)[0])
    return time, obj_type, end_time, fields


def parse_timing_point(line: str) -> tuple[float, int, list[str]]:
    fields = line.split(",")
    if len(fields) < 8:
        raise ValueError(f"malformed TimingPoint: {line}")
    return float(fields[0]), int(fields[6]), fields


def fmt_ms(value: float) -> str:
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    return f"{value:.8f}".rstrip("0").rstrip(".")


def shift_timing_line(line: str, delta: float) -> tuple[float, str, int]:
    time, uninherited, fields = parse_timing_point(line)
    shifted = time - delta
    fields[0] = fmt_ms(shifted)
    return shifted, ",".join(fields), uninherited


def make_split_sections(source_text: str, lo: float, hi: float,
                        diff_key: str, part: int) -> tuple[str, int, int]:
    header, sections = split_sections(source_text)
    smap = section_map(sections)
    original_objects = [line for line in smap.get("HitObjects", []) if line.strip()]
    kept_objects = []
    for line in original_objects:
        start, obj_type, end, fields = parse_hit_object(line)
        if lo <= start < hi:
            if end is not None and end > hi:
                raise ValueError(f"hold crosses split end at {end} > {hi}: {line}")
            fields[2] = fmt_ms(start - lo)
            if end is not None:
                fields[5] = fmt_ms(end - lo) + ":" + fields[5].split(":", 1)[1]
            kept_objects.append(",".join(fields))
        elif start < lo and end is not None and end > lo:
            raise ValueError(f"hold crosses split start at {start} < {lo} < {end}: {line}")

    original_tps = [line for line in smap.get("TimingPoints", []) if line.strip()]
    active: dict[int, tuple[float, int, str, int]] = {}
    later: list[tuple[float, int, str, int]] = []
    for order, line in enumerate(original_tps):
        time, uninherited, _ = parse_timing_point(line)
        if time <= lo:
            # Keep the last active redline and inherited state at the cut.
            active[uninherited] = (time, order, line, uninherited)
        elif time < hi:
            shifted, new_line, ui = shift_timing_line(line, lo)
            later.append((shifted, order, new_line, ui))
    timing_rows: list[tuple[float, int, str, int]] = []
    for time, order, line, ui in active.values():
        shifted, new_line, ui = shift_timing_line(line, lo)
        fields = new_line.split(",")
        fields[0] = "0"
        timing_rows.append((0.0, order, ",".join(fields), ui))
    timing_rows.extend(later)
    timing_rows.sort(key=lambda x: (x[0], x[1]))
    if not timing_rows or not any(r[3] == 1 for r in timing_rows):
        raise ValueError(f"split {diff_key}/{part} has no active redline at start {lo}")

    events: list[str] = []
    for line in smap.get("Events", []):
        stripped = line.strip()
        if stripped.startswith("0,"):
            # Keep the source background; audio/model consumers use the chart
            # section, and the background line has no timing semantics.
            events.append(line)
        elif stripped.startswith("2,"):
            fields = stripped.split(",")
            if len(fields) >= 3:
                start, end = float(fields[1]), float(fields[2])
                if lo <= start and end <= hi:
                    fields[1] = fmt_ms(start - lo)
                    fields[2] = fmt_ms(end - lo)
                    events.append(",".join(fields))

    new_sections: list[tuple[str, list[str]]] = []
    for name, body in sections:
        if name == "Metadata":
            changed = []
            for line in body:
                if line.startswith("BeatmapID:"):
                    changed.append("BeatmapID:0")
                elif line.startswith("BeatmapSetID:"):
                    changed.append("BeatmapSetID:0")
                elif line.startswith("Version:"):
                    changed.append(f"Version:{diff_key} split {part}")
                elif line.startswith("Tags:"):
                    changed.append(line + " ReformAnchorSplit")
                else:
                    changed.append(line)
            new_sections.append((name, changed))
        elif name == "Events":
            new_sections.append((name, events))
        elif name == "TimingPoints":
            new_sections.append((name, [r[2] for r in timing_rows]))
        elif name == "HitObjects":
            new_sections.append((name, kept_objects))
        else:
            new_sections.append((name, body))

    out_lines = list(header)
    for name, body in new_sections:
        out_lines.extend(["", f"[{name}]"])
        out_lines.extend(body)
    return "\n".join(out_lines).rstrip() + "\n", len(original_objects), len(kept_objects)


def parse_breaks(text: str) -> list[tuple[int, int]]:
    _, sections = split_sections(text)
    events = section_map(sections).get("Events", [])
    breaks = []
    for line in events:
        s = line.strip()
        if s.startswith("2,"):
            fields = s.split(",")
            if len(fields) >= 3:
                breaks.append((int(float(fields[1])), int(float(fields[2]))))
    return breaks


def has_redline_after(text: str, when: int, window_ms: int = 2000) -> bool:
    _, sections = split_sections(text)
    tps = section_map(sections).get("TimingPoints", [])
    for line in tps:
        if not line.strip():
            continue
        time, ui, _ = parse_timing_point(line)
        if ui == 1 and when <= time <= when + window_ms:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/reform")
    ap.add_argument("--split-plan", type=Path, default=ROOT / "data/reform/manual_splits.json")
    args = ap.parse_args()
    data_dir = args.data_dir
    charts = read_jsonl(data_dir / "charts.jsonl")
    by_id = {int(c["beatmap_id"]): c for c in charts if c.get("beatmap_id") is not None}
    plan = json.loads(args.split_plan.read_text(encoding="utf-8"))
    split_dir = data_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    split_rows: list[dict] = []
    split_checks: list[dict] = []
    for item in plan["splits"]:
        source_id = int(item["source_beatmap_id"])
        if source_id not in by_id:
            raise SystemExit(f"manual split source not in chart manifest: {source_id}")
        source = by_id[source_id]
        if source["difficulty_key"] != item["difficulty_key"]:
            raise SystemExit(f"split difficulty mismatch for {source_id}: {source['difficulty_key']}")
        source_path = ROOT / source["osu_path"]
        source_text = source_path.read_text(encoding="utf-8-sig")
        breaks = set(parse_breaks(source_text))
        selected = [tuple(map(int, pair)) for pair in item["boundaries"]]
        ignored = [tuple(map(int, pair)) for pair in item.get("ignored_internal_breaks", [])]
        if any(b not in breaks for b in selected + ignored):
            raise SystemExit(f"manual boundary does not match a Break Period in {source_id}")
        for _, end in selected:
            if not has_redline_after(source_text, end):
                raise SystemExit(f"selected boundary at {end} has no following redline in {source_id}")
        for _, end in ignored:
            if has_redline_after(source_text, end):
                raise SystemExit(f"ignored break at {end} is followed by a redline in {source_id}")

        _, source_sections = split_sections(source_text)
        hit_lines = [l for l in section_map(source_sections).get("HitObjects", []) if l.strip()]
        object_bounds = [(parse_hit_object(l)[0], parse_hit_object(l)[2]) for l in hit_lines]
        t_end = max((end if end is not None else start) for start, end in object_bounds) + 1
        boundaries = sorted(selected)
        ranges = [(0.0, float(boundaries[0][0]))]
        for left, right in zip(boundaries, boundaries[1:]):
            ranges.append((float(left[1]), float(right[0])))
        ranges.append((float(boundaries[-1][1]), float(t_end)))

        part_counts = []
        for part, (lo, hi) in enumerate(ranges, start=1):
            osu_text, n_source, n_part = make_split_sections(
                source_text, lo, hi, item["difficulty_key"], part)
            if n_part == 0:
                raise SystemExit(f"empty split {source_id} part {part}")
            name = f"{source_id}_{item['difficulty_key']}_part{part}.osu"
            out = split_dir / name
            raw = osu_text.encode("utf-8")
            out.write_bytes(raw)
            row = {
                "anchor_id": f"split:{source_id}:{part}:{hashlib.md5(raw).hexdigest()}",
                "beatmap_id": None,
                "beatmapset_id": None,
                "source_beatmap_id": source_id,
                "source_beatmapset_id": source["beatmapset_id"],
                "source_osu_md5": source["osu_md5"],
                "split_start_ms": int(lo),
                "split_end_ms": int(hi),
                "split_part": part,
                "difficulty_key": item["difficulty_key"],
                "target": source["target"],
                "anchor_type": "individual",
                "anchor_slot": f"manual_split_{part}",
                "pack_role": "manual_split",
                "osu_path": str(out.relative_to(ROOT)),
                "osu_md5": hashlib.md5(raw).hexdigest(),
                "osu_sha256": hashlib.sha256(raw).hexdigest(),
                "osu_bytes": len(raw),
                "hit_objects": n_part,
                "is_manual_split": True,
                "split_policy": "rebase_hit_objects_and_timing_state",
            }
            split_rows.append(row)
            part_counts.append(n_part)
        if sum(part_counts) != len(hit_lines):
            raise SystemExit(f"source object loss/duplication for {source_id}: {sum(part_counts)} != {len(hit_lines)}")
        split_checks.append({
            "difficulty_key": item["difficulty_key"],
            "source_beatmap_id": source_id,
            "source_hit_objects": len(hit_lines),
            "split_hit_objects": part_counts,
            "break_boundaries": [list(b) for b in selected],
            "ignored_internal_breaks": [list(b) for b in ignored],
            "all_hit_objects_preserved": True,
        })

    direct = []
    for chart in charts:
        if not chart.get("included_in_anchor_manifest") or not chart.get("difficulty_key"):
            continue
        row = dict(chart)
        row.update({
            "anchor_id": f"osu:{chart['beatmapset_id']}:{chart['beatmap_id']}:{chart['osu_md5']}",
            "anchor_slot": chart["pack_role"],
            "is_manual_split": False,
        })
        direct.append(row)
    all_rows = direct + split_rows
    all_rows.sort(key=lambda r: (r["target"], r["anchor_type"], r["anchor_slot"], r["anchor_id"]))
    counts: Counter = Counter(r["difficulty_key"] for r in all_rows)
    coverage = []
    for key in sorted({r["difficulty_key"] for r in all_rows}, key=lambda k: min(r["target"] for r in all_rows if r["difficulty_key"] == k)):
        group = [r for r in all_rows if r["difficulty_key"] == key]
        expected = 7 if key == "regular_6" else 5
        expected_roles = Counter({"total": 2, "individual": 5}) if key == "regular_6" else Counter({"total": 1, "individual": 4})
        observed_roles = Counter(r["anchor_type"] for r in group)
        coverage.append({
            "difficulty_key": key,
            "target": group[0]["target"],
            "observed_anchor_count": counts[key],
            "expected_anchor_count": expected,
            "observed_roles": dict(observed_roles),
            "expected_roles": dict(expected_roles),
            "complete": counts[key] == expected and observed_roles == expected_roles,
        })
    if any(not c["complete"] for c in coverage):
        raise SystemExit("compiled anchor rows do not meet the 5/7-point contract; see coverage")

    anchors_path = data_dir / "anchors.jsonl"
    with anchors_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (data_dir / "raw_scores_template.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["anchor_id", "difficulty_key", "target", "Sunny",
                                               "DCR", "CM3P_ext"])
        writer.writeheader()
        for row in all_rows:
            writer.writerow({"anchor_id": row["anchor_id"], "difficulty_key": row["difficulty_key"], "target": row["target"]})
    with (data_dir / "split_charts.jsonl").open("w", encoding="utf-8") as f:
        for row in split_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema_version": 1,
        "n_difficulties": len(coverage),
        "n_direct_anchor_rows": len(direct),
        "n_manual_split_rows": len(split_rows),
        "n_anchor_rows": len(all_rows),
        "n_expected_anchor_rows": 117,
        "manual_split_checks": split_checks,
        "coverage": coverage,
        "manifest_sha256": hashlib.sha256(anchors_path.read_bytes()).hexdigest(),
    }
    (data_dir / "anchor_coverage.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "difficulty_groups": len(coverage),
        "direct_anchors": len(direct),
        "manual_splits": len(split_rows),
        "total_anchors": len(all_rows),
        "complete_groups": sum(c["complete"] for c in coverage),
        "anchor_manifest": str(anchors_path.relative_to(ROOT)),
        "coverage_report": str((data_dir / "anchor_coverage.json").relative_to(ROOT)),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
