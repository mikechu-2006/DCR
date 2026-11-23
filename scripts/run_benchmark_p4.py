#!/usr/bin/env python
"""Fit the 4 benchmark configurations on the 5 P4 split seeds.

This is the robustness arm: same methods/hyperparameters as the full-data
models, but trained on the P4 6x6 block of each split seed.  Models are saved so
benchmark_infer.py can ensemble them.

    python scripts/run_benchmark_p4.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

CONFIGS = {
    "M1_L": dict(content="data/processed/chart_content_cm3p_local.parquet",
                 model="mlp_extra", extra=True, name="cm3p_mlp_extra"),
    "M1_C": dict(content="data/processed/chart_content_cm3p_local_canon.parquet",
                 model="mlp_extra", extra=True, name="cm3p_mlp_extra"),
    "M2_L": dict(content="data/processed/chart_content_cm3p_local.parquet",
                 model="mlp", extra=False, name="cm3p_mlp_h64_L2"),
    "M2_C": dict(content="data/processed/chart_content_cm3p_local_canon.parquet",
                 model="mlp", extra=False, name="cm3p_mlp_h64_L2"),
}


def main() -> int:
    out_models = ROOT / "results/benchmark/models"
    out_sigs = ROOT / "results/benchmark/dcr_signals/p4"
    out_models.mkdir(parents=True, exist_ok=True)
    out_sigs.mkdir(parents=True, exist_ok=True)
    for key, cfg in CONFIGS.items():
        for seed in range(1, 6):
            split = ROOT / f"data/processed/step1_1k_pairbest_p4_nogap_s{seed}.parquet"
            model_path = out_models / f"{key}_p4_s{seed}.pt"
            sig_path = out_sigs / f"{key}_p4_s{seed}.parquet"
            json_path = out_sigs / f"{key}_p4_s{seed}.json"
            if model_path.exists() and sig_path.exists():
                print(f"[skip] {key} s{seed}", flush=True)
                continue
            cmd = [PY, "scripts/step2_cm3p_rank.py",
                   "--in", str(split.relative_to(ROOT)),
                   "--content-path", cfg["content"],
                   "--signal-table", "data/processed/rank_signals_1k.parquet",
                   "--model", cfg["model"], "--lam", "1e-2", "--hid", "64",
                   "--depth", "2", "--dropout", "0.1", "--lr", "3e-3",
                   "--epochs", "98", "--seed", "0", "--seeds", "1",
                   "--val-frac", "0", "--name", cfg["name"],
                   "--save-model", str(model_path.relative_to(ROOT)),
                   "--out-signals", str(sig_path.relative_to(ROOT)),
                   "--out-json", str(json_path.relative_to(ROOT)),
                   "--threads", "6"]
            if cfg["extra"]:
                cmd += ["--extra-signals", "msd_overall,sunny_star"]
            log = ROOT / f"results/benchmark/logs_{key}_p4_s{seed}.log"
            print(f"[fit] {key} s{seed}", flush=True)
            with log.open("w") as fh:
                fh.write(" ".join(cmd) + "\n\n")
                fh.flush()
                p = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                tail = "\n".join(log.read_text(errors="replace").splitlines()[-20:])
                raise SystemExit(f"{key} s{seed} failed:\n{tail}")
    print("P4_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
