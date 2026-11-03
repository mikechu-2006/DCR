#!/usr/bin/env python3
"""experiments/ 索引：把每个实验目录的 manifest 汇总成一张表。

用法
----
    python scripts/index_experiments.py                    # 写 experiments/INDEX.md
    python scripts/index_experiments.py --out /tmp/i.md    # 写到别处
    python scripts/index_experiments.py --json             # 同时写 INDEX.json
    python scripts/index_experiments.py --max-models 3     # 每个实验最多列几个模型

每个实验目录由 `python main.py -c configs/*.yaml` 生成，含 `config.yaml` /
`run.sh` / `logs/` / `artifacts/` / `results/` / `manifest.json`。本脚本
**只读 manifest.json**，不读产物，因此很快；它输出：

* 实验目录、创建时间、状态、配置路径；
* step1 / step2 的版本与 tag；
* `manifest.summary.step2` 里各 run / 模型的 test RMSE 与 Spearman；
* 失败实验的错误摘要。

`experiments/` 整体不入库（见 .gitignore），所以 INDEX.md 只在本机有效；需要长期
留档的结论请写进 docs/ 或 results/README.md。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "experiments"


def _num(value: object, nd: int = 4) -> str:
    """格式化数值；缺失时给 '—'。"""
    if isinstance(value, bool) or value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.{nd}f}"
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return str(value)


def _cell(text: object) -> str:
    """把任意值变成一行、可安全放进 markdown 表格的文本。"""
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _step_summary(manifest: dict) -> str:
    """'step1/pairbest + step2/predict' 这样的步骤摘要。"""
    parts = []
    for step in manifest.get("steps") or []:
        name = step.get("name", "?")
        version = step.get("version") or "?"
        status = step.get("status")
        text = f"{name}/{version}"
        if status and status != "ok":
            text += f"({status})"
        parts.append(text)
    return " + ".join(parts) or "—"


def _tag(manifest: dict) -> str:
    """配置里的 tag（1k / 10k / ...）。"""
    config = manifest.get("config") or {}
    tag = config.get("tag")
    return str(tag) if tag else "—"


def _metric_lines(manifest: dict, max_models: int) -> str:
    """step2 的 test 指标：每个 run 一行 'model RMSE/rho'。"""
    step2 = (manifest.get("summary") or {}).get("step2") or {}
    if not isinstance(step2, dict):
        return "—"
    chunks: list[str] = []
    for run_name, run in step2.items():
        results = (run or {}).get("results") if isinstance(run, dict) else None
        if not isinstance(results, dict) or not results:
            continue
        models = []
        for model_name, metrics in list(results.items())[:max_models]:
            if not isinstance(metrics, dict):
                continue
            models.append(f"{model_name} {_num(metrics.get('rmse'))}/{_num(metrics.get('spearman'))}")
        if len(results) > max_models:
            models.append(f"…(+{len(results) - max_models})")
        if models:
            chunks.append(f"{run_name}: " + "; ".join(models))
    return "<br>".join(_cell(c) for c in chunks) if chunks else "—"


def collect(root: Path, max_models: int) -> list[dict]:
    """扫描 root 下所有含 manifest.json 的实验目录。"""
    rows = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - 坏 manifest
            rows.append({
                "dir": path.name,
                "created": "—",
                "status": "unreadable",
                "steps": f"manifest 解析失败: {exc}",
                "config": "—",
                "tag": "—",
                "metrics": "—",
                "manifest": str(manifest_path),
            })
            continue
        rows.append({
            "dir": path.name,
            "created": str(manifest.get("created") or "—").replace("T", " "),
            "status": str(manifest.get("status") or "?"),
            "steps": _step_summary(manifest),
            "config": str(manifest.get("config_path") or "—"),
            "tag": _tag(manifest),
            "metrics": _metric_lines(manifest, max_models),
            "error": manifest.get("error"),
            "manifest": str(manifest_path),
        })
    return rows


def render(rows: list[dict]) -> str:
    """生成 markdown 索引。"""
    ok = sum(1 for r in rows if r["status"] == "ok")
    lines = [
        "# experiments/ 索引",
        "",
        "由 [`scripts/index_experiments.py`](../scripts/index_experiments.py) 生成：",
        "只读各实验目录的 `manifest.json`，不读产物。**本目录不入库**，索引只在本机有效；",
        "要长期留档的结论请写进 [docs/](../docs/) 或 [results/README.md](../results/README.md)。",
        "",
        f"共 **{len(rows)}** 个实验（ok **{ok}**）：",
        "",
        "| 实验目录 | 创建 | 状态 | tag | 步骤 | 配置 | step2 test 指标（RMSE / Spearman） |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        link = f"[{row['dir']}]({row['dir']}/)"
        lines.append(
            "| " + " | ".join([
                link,
                _cell(row["created"]),
                _cell(row["status"]),
                _cell(row["tag"]),
                _cell(row["steps"]),
                _cell(row["config"]),
                row["metrics"],
            ]) + " |"
        )
    lines += [
        "",
        "> 指标取自 `manifest.summary.step2[*].results`，是各实验 **test** 集上的",
        "> RMSE / Spearman。不同 tag（1k / 10k）、不同 step1 口径（pairbest / clean）、",
        "> 不同覆盖子集的数字**不能直接并排比较**；口径见 [docs/data.md](../docs/data.md)。",
        "",
    ]
    errors = [r for r in rows if r.get("error")]
    if errors:
        lines += ["## 失败实验", ""]
        for row in errors:
            lines.append(f"* `{row['dir']}`: `{_cell(row['error'])}`")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="实验目录的父目录（默认 experiments/）")
    parser.add_argument("--out", default=None, help="输出路径（默认 <root>/INDEX.md）")
    parser.add_argument("--json", action="store_true", help="同时写 <out>.json")
    parser.add_argument("--max-models", type=int, default=6, help="每个 run 最多列几个模型（默认 6）")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"没有实验目录: {root}", file=sys.stderr)
        return 1
    rows = collect(root, max_models=max(1, args.max_models))
    if not rows:
        print(f"{root} 下没有 manifest.json（先跑一次 main.py）", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else root / "INDEX.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rows), encoding="utf-8")
    print(f"写入 {out}（{len(rows)} 个实验）")

    if args.json:
        payload = [{k: v for k, v in row.items() if k != "manifest"} for row in rows]
        json_path = out.with_suffix(".json")
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"写入 {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
