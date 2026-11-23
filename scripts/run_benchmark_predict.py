#!/usr/bin/env python
"""Run benchmark inference for the 4 full models and the P4 seed ensembles.

    python scripts/run_benchmark_predict.py

Requires data/processed/benchmark_content_{legacy,canon}.parquet from the HPC
encoder.  Full-model predictions go to results/benchmark/pred; P4 ensembles
(average of the 5 seed-specific d_m) go to results/benchmark/pred_p4.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

CONFIGS = {
    "M1_L": ("legacy", "M1_L_full.pt"),
    "M1_C": ("canon", "M1_C_full.pt"),
    "M2_L": ("legacy", "M2_L_full.pt"),
    "M2_C": ("canon", "M2_C_full.pt"),
}


def run(cmd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        fh.write(" ".join(cmd) + "\n\n")
        fh.flush()
        p = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-20:])
        raise SystemExit(f"failed: {' '.join(cmd)}\n{tail}")


def main() -> int:
    for key, (arm, full_model) in CONFIGS.items():
        content = ROOT / f"data/processed/benchmark_content_{arm}.parquet"
        if not content.exists():
            raise SystemExit(f"missing {content}")
        out = ROOT / f"results/benchmark/pred/{key}.parquet"
        cmd = [PY, "scripts/benchmark_infer.py",
               "--model", f"results/benchmark/models/{full_model}",
               "--content", str(content.relative_to(ROOT)),
               "--signals", "data/processed/benchmark_signals.parquet",
               "--out", str(out.relative_to(ROOT))]
        print(f"[full] {key} <- {full_model}", flush=True)
        run(cmd, ROOT / f"results/benchmark/logs_infer_{key}_full.log")

        models = [f"results/benchmark/models/{key}_p4_s{i}.pt" for i in range(1, 6)]
        missing = [m for m in models if not (ROOT / m).exists()]
        if missing:
            raise SystemExit(f"{key}: missing P4 models {missing}")
        cmd = [PY, "scripts/benchmark_infer.py"]
        for m in models:
            cmd += ["--model", m]
        cmd += ["--content", str(content.relative_to(ROOT)),
                "--signals", "data/processed/benchmark_signals.parquet",
                "--out", f"results/benchmark/pred_p4/{key}.parquet"]
        print(f"[p4]   {key} <- 5 seeds", flush=True)
        run(cmd, ROOT / f"results/benchmark/logs_infer_{key}_p4.log")
    print("PRED_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
