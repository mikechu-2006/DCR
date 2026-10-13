"""实验目录：experiments_{时间戳}[_{name}]/ 的布局、日志与 manifest。

布局::

    experiments_20261012_153012_std_1k/
      config.yaml      本次实验的完整解析配置（含默认值，可直接重跑）
      manifest.json    每步：命令行、参数、输入输出指纹、耗时、状态、指标
      run.sh           等价命令序列（手工重放用）
      logs/            pipeline.log + 每步一份 00_step0_preprocess.log
      artifacts/       中间产物：step0 play 表 / beatmap_meta / step1 清洗表 / ...
      results/         实验结果：step2 的 npz 与 summary json
      figures/         图（分析类步骤输出到这里）
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from .paths import REPO_ROOT, display, expand, expand_deep, human_bytes, resolve_path, sha256_file

TS_FMT = "%Y%m%d_%H%M%S"
_SAFE = re.compile(r"[^0-9A-Za-z_.-]+")


def new_experiment_dir(root: Path, name: str = "", when: datetime | None = None) -> Path:
    """experiments_{时间戳}[_{name}]；同一秒重复运行自动加后缀。"""
    stamp = (when or datetime.now()).strftime(TS_FMT)
    suffix = _SAFE.sub("_", name or "").strip("_")
    base = root / (f"experiments_{stamp}" + (f"_{suffix}" if suffix else ""))
    candidate, i = base, 2
    while candidate.exists():
        candidate = root / f"{base.name}_{i}"
        i += 1
    return candidate


def _git_info() -> dict:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                                 text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
    }


class Experiment:
    """一次实验的目录 + 日志 + manifest。"""

    def __init__(self, directory: Path | str, cfg: Mapping, config_path: Path | None = None,
                 echo: bool = True, resume: bool = False):
        self.dir = Path(directory).resolve()
        self.cfg = dict(cfg)
        self.config_path = Path(config_path).resolve() if config_path else None
        self.echo = echo
        self.resume = resume
        self.artifacts = self.dir / "artifacts"
        self.results = self.dir / "results"
        self.logs = self.dir / "logs"
        self.figures = self.dir / "figures"
        self._log_fh = None
        self.manifest: dict[str, Any] = {
            "schema": 1,
            "experiment": self.dir.name,
            "dir": str(self.dir),
            "created": datetime.now().isoformat(timespec="seconds"),
            "updated": None,
            "status": "running",
            "config_path": display(self.config_path) if self.config_path else None,
            "config": self.cfg,
            "repo": {"root": str(REPO_ROOT), **_git_info()},
            "host": {
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "executable": sys.executable,
                "cpu_count": os.cpu_count(),
            },
            "steps": [],
            "summary": {},
        }
        if resume:
            self._load_manifest()

    def _load_manifest(self) -> None:
        """--resume：把上一次的 manifest 读回来，作为本次记录的基线。"""
        path = self.dir / "manifest.json"
        if not path.exists():
            return
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(prev, dict) or prev.get("schema") != 1:
            return
        self.manifest["created"] = prev.get("created") or self.manifest["created"]
        self.manifest["steps"] = list(prev.get("steps") or [])
        self.manifest["summary"] = prev.get("summary") or {}

    # ------------------------------------------------------------------ 目录与占位符
    def ensure(self) -> None:
        for d in (self.artifacts, self.results, self.logs, self.figures):
            d.mkdir(parents=True, exist_ok=True)
        if self._log_fh is None:
            self._log_fh = (self.logs / "pipeline.log").open("a", encoding="utf-8")

    def close(self) -> None:
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def mapping(self, extra: Mapping[str, Any] | None = None) -> dict:
        """占位符表：{tag} {exp} {artifacts} {results} {logs} {figures} {cache} {name}"""
        m = {
            "exp": str(self.dir),
            "artifacts": str(self.artifacts),
            "results": str(self.results),
            "logs": str(self.logs),
            "figures": str(self.figures),
            "repo": str(REPO_ROOT),
            "config": str(self.config_path) if self.config_path else "",
            "name": self.cfg.get("name", ""),
            "tag": self.cfg.get("tag", ""),
            "cache": str(resolve_path(self.cfg.get("cache_dir") or "data/processed")),
        }
        if extra:
            m.update({k: v for k, v in extra.items() if v is not None})
        return m

    def expand(self, value: Any, extra: Mapping[str, Any] | None = None) -> Any:
        return expand_deep(value, self.mapping(extra))

    def path(self, template: str | Path, extra: Mapping[str, Any] | None = None) -> Path:
        return resolve_path(expand(str(template), self.mapping(extra)))

    def relative(self, path: Path | str) -> str:
        p = Path(path)
        for root in (REPO_ROOT, self.dir):
            try:
                return p.relative_to(root).as_posix()
            except ValueError:
                continue
        return str(p)

    # ------------------------------------------------------------------------ 日志
    def log(self, msg: str = "") -> None:
        if self.echo:
            print(msg, flush=True)
        if self._log_fh is not None:
            self._log_fh.write(msg + "\n")
            self._log_fh.flush()

    def section(self, title: str) -> None:
        self.log("")
        self.log("=" * 78)
        self.log(title)
        self.log("=" * 78)

    # --------------------------------------------------------------------- 产物登记
    def fingerprint(self, path: Path, do_hash: bool = True, max_mb: int = 512) -> dict:
        p = Path(path)
        info: dict[str, Any] = {"path": str(p), "rel": self.relative(p)}
        try:
            st = p.stat()
            info["bytes"] = st.st_size if p.is_file() else None
            info["kind"] = "file" if p.is_file() else "dir"
            if p.is_dir():
                files = [f for f in p.rglob("*") if f.is_file()]
                info["files"] = len(files)
                info["bytes"] = sum(f.stat().st_size for f in files)
        except OSError:
            info["kind"] = "missing"
            return info
        if do_hash and p.is_file():
            info["sha256"] = sha256_file(p, max_mb * 1024 * 1024)
        return info

    # ------------------------------------------------------------------- manifest
    def record_step(self, entry: Mapping) -> None:
        """按 (name, version) 写入：同名 step 视为重跑，替换旧记录并累加 attempt。"""
        entry = dict(entry)
        steps = self.manifest["steps"]
        for i, old in enumerate(steps):
            if old.get("name") == entry.get("name") and old.get("version") == entry.get("version"):
                entry["attempt"] = int(old.get("attempt", 1)) + 1
                # --resume 跳过的 step 不该丢掉上一次的产出与指标
                for key in ("report", "outputs", "seconds", "exit_code", "started", "log"):
                    if not entry.get(key) and old.get(key):
                        entry[key] = old[key]
                steps[i] = entry
                break
        else:
            entry["attempt"] = 1
            steps.append(entry)
        self.flush()

    def set_summary(self, summary: Mapping) -> None:
        self.manifest["summary"] = dict(summary)
        self.flush()

    def finish(self, status: str, error: str | None = None) -> None:
        self.manifest["status"] = status
        if error:
            self.manifest["error"] = error
        self.flush()
        self.close()

    def flush(self) -> None:
        self.manifest["updated"] = datetime.now().isoformat(timespec="seconds")
        tmp = self.dir / "manifest.json.part"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False, default=str),
                       encoding="utf-8")
        tmp.replace(self.dir / "manifest.json")

    # ------------------------------------------------------------------- 快照与脚本
    def write_config_snapshot(self, yaml_text: str) -> None:
        (self.dir / "config.yaml").write_text(yaml_text, encoding="utf-8")

    def write_run_sh(self, env: Mapping | None = None) -> None:
        """由 manifest 里的步骤命令生成 run.sh（--resume 续跑后依然完整）。"""
        from .run import format_cmd
        lines = ["#!/usr/bin/env bash",
                 "# 由 main.py 生成：本实验执行过的命令（可手工重放，产物就地覆盖）",
                 "set -euo pipefail",
                 'cd "$(dirname "$0")/.."',
                 ""]
        for key, value in (env or {}).items():
            lines.append(f"export {key}={value!r}")
        if env:
            lines.append("")
        for entry in self.manifest.get("steps", []):
            cmd = entry.get("cmd")
            if isinstance(cmd, list) and cmd:
                lines.append(format_cmd([str(c) for c in cmd]))
        (self.dir / "run.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.chmod(self.dir / "run.sh", 0o755)

    def describe_outputs(self, paths: Iterable[Path]) -> str:
        parts = []
        for p in paths:
            try:
                size = p.stat().st_size if p.is_file() else None
            except OSError:
                size = None
            parts.append(f"{self.relative(p)} ({human_bytes(size)})")
        return ", ".join(parts)
