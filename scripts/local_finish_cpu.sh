#!/bin/bash
# Finish the remaining CPU work LOCALLY (HPC i64m512u/debug were stuck on
# Priority / MaxJobsPerAccount): a21d set-P4 probe, VSRG benchmark
# randommerge fit+infer+isotonic calibration for a0d/a17d/a20d, C3/C4 for a21d.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/d2v_attr_run
mkdir -p "$OUT" results/benchmark_attr/pred results/benchmark_attr/models results/benchmark_attr/train
exec > >(tee -a "$OUT/local_finish.log") 2>&1
PY=python3
echo "=== local CPU finish start $(date)"
echo "=== [1/5] a21d + a20d set-P4 probe"
for spec in "a21d:data/processed/chart_content_d2v_a21d-mclean.parquet:results/d2v_attr_d_cached/a21d/setp4" "a20d_reprobe:data/processed/chart_content_d2v_a20d-mclean.parquet:results/d2v_attr_d_cached/a20d_reprobe/setp4"; do
  ARM=$(printf '%s' "$spec" | cut -d: -f1)
  C=$(printf '%s' "$spec" | cut -d: -f2)
  O=$(printf '%s' "$spec" | cut -d: -f3)
  echo "[probe] $ARM"
  $PY scripts/run_p2_p4_multiseed.py --protocols p4 --seeds 1 --models linear,linear_extra \
    --content-path "$C" --signal-table data/processed/rank_signals_1k.parquet \
    --split-p4 data/processed/step1_mclean25_p4sets_cached --out-dir "$O" --threads 8
done

echo "=== [2/5] benchmark fits + inference"
for ARM in a0d a17d a20d; do
  TAG=$(printf '%s' "$ARM" | tr '[:lower:]' '[:upper:]')
  mkdir -p "results/benchmark_attr/train/$ARM"
  echo "[fit] $ARM"
  $PY scripts/step2_cm3p_rank.py \
    --in data/processed/step1_randommerge_pairbest_content.parquet \
    --content-path "data/processed/chart_content_d2v_$ARM-all.parquet" \
    --signal-table data/processed/rank_signals_1k.parquet \
    --extra-signals msd_overall,sunny_star \
    --model linear_extra --lam 1e-3 --val-frac 0 --seed 0 --threads 8 \
    --name cm3p_linear_extra --fit-all \
    --save-model "results/benchmark_attr/models/$TAG-full.pt" \
    --out-signals "results/benchmark_attr/train/$ARM/signals.parquet" \
    --out-json "results/benchmark_attr/train/$ARM/train.json"
  echo "[infer] $ARM"
  $PY scripts/benchmark_infer.py \
    --model "results/benchmark_attr/models/$TAG-full.pt" \
    --content "data/processed/benchmark_content_$ARM.parquet" \
    --signals data/processed/benchmark_signals.parquet \
    --out "results/benchmark_attr/pred/$TAG.parquet"
done

echo "=== [3/5] isotonic 5-fold calibration"
for ref in RM_L D2V_L M1_L; do cp "results/benchmark/pred/$ref.parquet" "results/benchmark_attr/pred/$ref.parquet"; done
$PY scripts/benchmark_calibrate_eval.py \
  --pred-dir results/benchmark_attr/pred \
  --out-dir results/benchmark_attr/eval \
  --runs A0D,A17D,A20D,RM_L,D2V_L,M1_L --bootstrap 1000

echo "=== [4/5] C3/C4 for a21d"
$PY scripts/content_mapper_probe.py --content-path data/processed/chart_content_d2v_a21d-mclean.parquet \
  --mode both --tag a21d \
  --mapper-out results/d2v_attr_diag/mapper_probe_a21d.json \
  --attr-out results/d2v_attr_diag/attr_decodability_a21d.json

echo "=== [5/5] summary"
$PY - <<'PY'
import json
from pathlib import Path
import pandas as pd
lines = []
lines.append("# Local CPU finish: a21d + VSRG benchmark ablation")
lines.append("")
lines.append("## a21d set-P4 (seed 1, cached split)")
rows = []
for arm in ["a0d", "a17d", "a20d_reprobe", "a21d"]:
    p = Path("results/d2v_attr_d_cached") / arm / "setp4" / "summary.csv"
    if not p.exists():
        continue
    df = pd.read_csv(p).set_index("method")
    rows.append({"arm": arm,
                 "nll_linear": float(df.loc["CM3P 线性 (λ1e-3)", "nll"]),
                 "nll_ext": float(df.loc["CM3P + external 线性", "nll"]),
                 "captured_linear": float(df.loc["CM3P 线性 (λ1e-3)", "captured"])})
tab = pd.DataFrame(rows)
base = tab[tab.arm == "a0d"]
if not base.empty:
    b = base.iloc[0]
    tab["d_linear_vs_a0d"] = tab.nll_linear - b.nll_linear
    tab["d_ext_vs_a0d"] = tab.nll_ext - b.nll_ext
a20 = tab[tab.arm == "a20d_reprobe"]
if not a20.empty:
    r = a20.iloc[0]
    tab["d_linear_vs_a20d"] = tab.nll_linear - r.nll_linear
    tab["d_ext_vs_a20d"] = tab.nll_ext - r.nll_ext
lines.append("~~~")
lines.append(tab.to_markdown(index=False, floatfmt=".5f"))
lines.append("~~~")
lines.append("")
lines.append("## a21d C3/C4")
for name, path in [("a21d", "results/d2v_attr_diag/mapper_probe_a21d.json"),
                   ("a21d", "results/d2v_attr_diag/attr_decodability_a21d.json")]:
    if Path(path).exists():
        d = json.load(open(path))
        if "top1_accuracy_test" in d:
            lines.append("- C3 mapper top1: %.4f (majority %.4f)" % (d["top1_accuracy_test"], d["majority_baseline_test_accuracy"]))
        if "targets" in d:
            sel = {k: round(v.get("r2_test", float("nan")), 3) for k, v in d["targets"].items()
                   if k in ("kiai_frac", "log1p_n_sv_changes", "log2_bpm_p50", "vol_resolved_p50", "hs_any_ext_frac")}
            lines.append("- C4: " + json.dumps(sel, ensure_ascii=False))
lines.append("")
lines.append("## VSRG benchmark eval files")
for p in sorted(Path("results/benchmark_attr/eval").glob("*")):
    lines.append("- " + str(p))
text = chr(10).join(lines) + chr(10)
Path("results/d2v_attr_run/STATUS_LOCAL_FINISH.md").write_text(text)
print(text)
PY
echo "=== local CPU finish DONE $(date)"
