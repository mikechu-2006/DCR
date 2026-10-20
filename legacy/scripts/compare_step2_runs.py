#!/usr/bin/env python
"""把多次 step2 运行的 summary 拼成一张对比表（RMSE / Spearman / 覆盖）。

每次 `python main.py -c ...` 都会在 experiments_*/results/ 下写一份
`step2_{tag}_summary_{target}.json`，里面 runs[variant].results[model] 是测试指标。
本脚本只读这些 json，不重训、不改文件。

    python scripts/compare_step2_runs.py experiments_*/results/step2_1k_summary_loss.json
    python scripts/compare_step2_runs.py --csv /tmp/table.csv experiments_*/results/*summary*.json

口径：RMSE 是 loss 空间（与 step2/predict 一致）；coverage 是该模型在测试行上的覆盖；
n_scored 是真正参与打分的行数。**不同子集上的 RMSE 不可直接比** —— 看同一行里的
bias 基线是否相同，相同才说明子集一致。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    exp = path.parent.parent.name
    if "runs" not in data:
        data = {"runs": {path.stem: data}}
    rows = []
    for variant, run in data["runs"].items():
        feat = run.get("feat") or run.get("msd") or {}
        for model, res in (run.get("results") or {}).items():
            rows.append({
                "experiment": exp,
                "variant": variant,
                "model": model,
                "rmse": res.get("rmse"),
                "spearman": res.get("spearman"),
                "coverage": res.get("coverage"),
                "n_scored": res.get("n_scored"),
                "dim": run.get("sizes", {}).get("n_items") and feat.get("dim_used"),
                "groups": ",".join(feat.get("groups") or []) or feat.get("skills") or "",
                "seconds": run.get("seconds"),
                "path": str(path),
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description="汇总多个 step2 summary 的测试指标")
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--sort", default="rmse", choices=["rmse", "spearman", "experiment"])
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--model", default=None, help="只看某个模型名（如 bias / msd / fc）")
    args = ap.parse_args()

    rows = []
    for p in args.paths:
        if p.is_dir():
            rows += [r for f in sorted(p.glob("*summary*.json")) for r in load_rows(f)]
        else:
            rows += load_rows(p)
    if args.model:
        rows = [r for r in rows if r["model"] == args.model]
    if not rows:
        raise SystemExit("没有读到任何 runs")

    key = (lambda r: r["rmse"]) if args.sort == "rmse" else (
        (lambda r: -(r["spearman"] or 0)) if args.sort == "spearman"
        else (lambda r: (r["experiment"], r["model"])))
    rows.sort(key=key)

    cols = ["experiment", "variant", "model", "rmse", "spearman", "coverage", "n_scored", "dim", "seconds"]
    widths = {c: max(len(c), *(len(_fmt(r.get(c))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols))
    if args.csv:
        import csv
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


if __name__ == "__main__":
    main()
