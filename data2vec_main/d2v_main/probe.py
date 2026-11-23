"""Thin wrapper around the existing rank-NLL runners for a new content table.

P2: scripts/step2_cm3p_rank.py --chart-folds 5 (chart-disjoint)
P4: scripts/run_p2_p4_multiseed.py --protocols p4 (new players + new charts)

No metric code is reimplemented here: the point is to keep the DCR numbers
comparable to the CM3P baselines.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .utils import repo_root


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--content-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--protocol", default="p2", choices=["p2", "p4", "both"])
    ap.add_argument("--model", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = repo_root()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    content = str(Path(args.content_path).resolve())
    if args.protocol in ("p2", "both"):
        cmd = [py, "scripts/step2_cm3p_rank.py",
               "--in", "data/processed/step1_1k_pairbest.parquet",
               "--content-path", content,
               "--model", args.model,
               "--chart-folds", "5",
               "--epochs", str(args.epochs),
               "--threads", str(args.threads),
               "--out-signals", str(out / "p2_signals.parquet"),
               "--out-json", str(out / "p2_rank.json")]
        if args.dry_run:
            print(" ".join(cmd))
        else:
            subprocess.run(cmd, cwd=root, check=True)
            print(json.dumps(json.load(open(out / "p2_rank.json")), indent=1)[:2000])
    if args.protocol in ("p4", "both"):
        cmd = [py, "scripts/run_p2_p4_multiseed.py",
               "--protocols", "p4",
               "--seeds", "1,2,3,4,5",
               "--models", "cm3p_linear",
               "--content-path", content,
               "--out-dir", str(out / "p4_multiseed")]
        if args.quick:
            cmd.append("--quick")
        if args.dry_run:
            print(" ".join(cmd))
        else:
            subprocess.run(cmd, cwd=root, check=True)
            print(f"[probe] P4 results under {out / 'p4_multiseed'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
