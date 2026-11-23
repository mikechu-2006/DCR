#!/usr/bin/env python3
"""Run ISOR v3.1 on DCR charts with DCR-local MinaCalc MSD injected.

The public ISOR entry point calls minacalc_bridge.calc(), which needs
msd.exe. On Linux that binary cannot run, so ISOR silently falls back to a
SR-only dp_msd and over-rates. This runner replaces the MSD lookup with the
DCR minacalc_msd_1k.parquet table (calc 505), so every chart gets a real MSD
vector. The resulting signal is ISOR v3.1 + DCR MSD 505, not necessarily
bit-identical to the benchmark CSV; G0 records the difference.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--charts", required=True)
    ap.add_argument("--msd", default=str(ROOT / "data/processed/minacalc_msd_1k.parquet"))
    ap.add_argument("--isor-root", default=str(ROOT / ".cache/methods/DanOverlay-v31"))
    ap.add_argument("--osu-dir", default=str(ROOT / "data/osu_cache"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def load_ids(path: str) -> list[int]:
    p = Path(path)
    if p.suffix in (".parquet", ".pq"):
        return sorted(pd.read_parquet(p, columns=["beatmap_id"])["beatmap_id"].astype("int64").unique().tolist())
    if p.suffix == ".csv":
        return sorted(pd.read_csv(p)["beatmap_id"].astype("int64").unique().tolist())
    ids = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(int(line.split()[0]))
    return sorted(set(ids))


def main() -> int:
    args = parse_args()
    ids = load_ids(args.charts)
    k, n = [int(x) for x in args.shard.split("/")]
    ids = [x for i, x in enumerate(ids) if i % n == k]
    if args.limit:
        ids = ids[: args.limit]

    sys.path.insert(0, str(Path(args.isor_root) / "src/08_isor_engine"))
    import isor_engine as ie  # noqa: E402

    msd = pd.read_parquet(args.msd).drop_duplicates("beatmap_id").set_index("beatmap_id")
    msd_map = {}
    for bid, row in msd.iterrows():
        d = {
            "overall": float(row.get("msd_overall", 0.0) or 0.0),
            "stream": float(row.get("msd_stream", 0.0) or 0.0),
            "jumpstream": float(row.get("msd_jumpstream", 0.0) or 0.0),
            "handstream": float(row.get("msd_handstream", 0.0) or 0.0),
            "stamina": float(row.get("msd_stamina", 0.0) or 0.0),
            "jackspeed": float(row.get("msd_jackspeed", 0.0) or 0.0),
            "chordjack": float(row.get("msd_chordjack", 0.0) or 0.0),
            "technical": float(row.get("msd_technical", 0.0) or 0.0),
            "duration_s": 0.0,
            "note_count": 0,
        }
        skills = ("overall", "stream", "jumpstream", "handstream",
                  "stamina", "jackspeed", "chordjack", "technical")
        if all(d[k] > 0 for k in skills):
            msd_map[int(bid)] = d

    def fake_msd(osu_path, rate=1.0, goal=0.96, mode="ssr"):
        try:
            bid = int(Path(osu_path).stem)
        except Exception:
            return None
        return msd_map.get(bid)

    ie.calculate_msd = fake_msd

    out = open(args.out, "w")
    t0 = time.time()
    ok = 0
    for i, bid in enumerate(ids, 1):
        p = Path(args.osu_dir) / (str(bid) + ".osu")
        rec = {"beatmap_id": int(bid), "method": "isor_v31", "version": "ec6ab5b",
               "raw": None, "valid": False, "reason": None, "error": None,
               "runtime_ms": None, "family": None, "dp_msd": None,
               "dp_sr": None, "ridge": None, "has_msd": int(bid in msd_map)}
        t = time.time()
        if not p.exists():
            rec["reason"] = "Missing"
            rec["error"] = "file_not_found"
        else:
            try:
                r = ie.analyze_beatmap_isor(str(p))
                if r.get("error"):
                    rec["reason"] = "RunnerError"
                    rec["error"] = str(r.get("error"))
                elif r.get("dp") is None:
                    rec["reason"] = "Invalid"
                    rec["error"] = "dp_null"
                else:
                    rec["raw"] = float(r["dp"])
                    rec["valid"] = True
                    rec["reason"] = "OK"
                    rec["family"] = r.get("family")
                    rec["dp_msd"] = r.get("dp_msd")
                    rec["dp_sr"] = r.get("dp_sr")
                    rec["ridge"] = r.get("ridge_correction")
                    ok += 1
            except Exception as exc:  # noqa: BLE001
                rec["reason"] = "RunnerError"
                rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["runtime_ms"] = int((time.time() - t) * 1000)
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if i % 50 == 0 or i == len(ids):
            dt = time.time() - t0
            print(f"[isor] shard {k}/{n} {i}/{len(ids)} ok={ok} {i/dt:.2f} charts/s", flush=True)
    out.close()
    print(f"[isor] wrote {args.out} ({len(ids)} rows, ok={ok})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
