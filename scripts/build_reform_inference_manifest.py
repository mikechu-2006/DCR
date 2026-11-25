#!/usr/bin/env python
"""Stage Reform Dan anchors under unique IDs for the existing CM3P/signal pipelines."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/reform"
SYNTHETIC_ID_BASE = 900_000_000
BEATMAP_ID = re.compile(rb"^(\s*BeatmapID\s*:\s*)(-?\d+)", re.M)


def main() -> None:
    anchors = [json.loads(line) for line in (DATA / "anchors.jsonl").read_text().splitlines()
               if line.strip()]
    synthetic = sorted((r for r in anchors if r.get("beatmap_id") is None),
                       key=lambda r: r["anchor_id"])
    synthetic_ids = {r["anchor_id"]: SYNTHETIC_ID_BASE + i
                     for i, r in enumerate(synthetic)}
    all_ids = [int(r["beatmap_id"]) if r.get("beatmap_id") is not None
               else synthetic_ids[r["anchor_id"]] for r in anchors]
    if len(all_ids) != len(set(all_ids)):
        raise SystemExit("anchor feature IDs are not unique")

    stage = DATA / "inference_osu"
    stage.mkdir(parents=True, exist_ok=True)
    map_rows = []
    signal_rows = []
    for row in anchors:
        feature_id = (int(row["beatmap_id"]) if row.get("beatmap_id") is not None
                      else synthetic_ids[row["anchor_id"]])
        source = ROOT / row["osu_path"]
        body = source.read_bytes()
        if not BEATMAP_ID.search(body):
            raise SystemExit(f"{source}: missing BeatmapID line")
        body = BEATMAP_ID.sub(lambda m: m.group(1) + str(feature_id).encode(), body, count=1)
        staged = stage / f"{feature_id}.osu"
        staged.write_bytes(body)
        checksum = hashlib.md5(body).hexdigest()

        map_rows.append({"anchor_id": row["anchor_id"], "feature_beatmap_id": feature_id,
                         "osu_path": str(staged.relative_to(ROOT)), "checksum": checksum,
                         "source_osu_path": row["osu_path"], "source_osu_md5": row["osu_md5"],
                         "difficulty_key": row["difficulty_key"], "target": float(row["target"]),
                         "anchor_type": row["anchor_type"], "is_manual_split": bool(row.get("is_manual_split"))})
        signal_rows.append({"row_id": row["anchor_id"], "osu_path": str(staged),
                            "bid": feature_id, "name": row["anchor_id"],
                            "pattern": row["difficulty_key"], "family": row.get("anchor_slot") or "course",
                            "expected": float(row["target"]), "rate": 1.0})

    pd.DataFrame(signal_rows).to_parquet(DATA / "signals_manifest.parquet", index=False)
    pd.DataFrame(map_rows).to_json(DATA / "feature_id_map.json", orient="records",
                                   indent=2, force_ascii=False)
    with (DATA / "cm3p_ids.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["beatmap_id", "checksum"])
        writer.writeheader()
        writer.writerows({"beatmap_id": r["feature_beatmap_id"], "checksum": r["checksum"]}
                         for r in map_rows)
    (DATA / "d2v_ids.txt").write_text(
        "\n".join(str(r["feature_beatmap_id"]) for r in map_rows) + "\n",
        encoding="utf-8")
    print(json.dumps({"anchors": len(anchors), "direct_ids": len(anchors) - len(synthetic),
                      "synthetic_ids": len(synthetic),
                      "signals_manifest": str(DATA / "signals_manifest.parquet"),
                      "cm3p_ids": str(DATA / "cm3p_ids.csv"),
                      "d2v_ids": str(DATA / "d2v_ids.txt"),
                      "staged_osu": len(list(stage.glob("*.osu")))}, indent=2))


if __name__ == "__main__":
    main()
