"""路径与文件小工具：仓库根定位、占位符展开、指纹。"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

#: 仓库根目录（本文件 = <repo>/dcr/paths.py）。所有相对路径都相对它解析，
#: 这样无论从哪个工作目录调用 main.py，行为都一致。
REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: Any, base: Path | None = None) -> Path:
    """把配置里的路径字符串解析成绝对路径（~ 与环境变量会被展开）。"""
    p = Path(os.path.expandvars(os.path.expanduser(str(value))))
    base = REPO_ROOT if base is None else base
    return p.resolve() if p.is_absolute() else (base / p).resolve()


class _KeepMissing(dict):
    """format_map 用：未知占位符原样保留，便于报错时看清是哪一个。"""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def expand(text: str, mapping: Mapping[str, Any]) -> str:
    """展开 {tag} / {exp} / {results} 之类的占位符。"""
    return str(text).format_map(_KeepMissing({k: str(v) for k, v in mapping.items()}))


def expand_deep(value: Any, mapping: Mapping[str, Any]) -> Any:
    """对 str / list / dict 递归展开占位符，其它类型原样返回。"""
    if isinstance(value, str):
        return expand(value, mapping)
    if isinstance(value, list):
        return [expand_deep(v, mapping) for v in value]
    if isinstance(value, tuple):
        return tuple(expand_deep(v, mapping) for v in value)
    if isinstance(value, dict):
        return {k: expand_deep(v, mapping) for k, v in value.items()}
    return value


def sha256_file(path: Path, max_bytes: int = 512 * 1024 * 1024) -> str | None:
    """文件指纹；超过 max_bytes 的文件跳过（返回 None），避免把大产物整个读一遍。"""
    try:
        if not path.is_file() or path.stat().st_size > max_bytes:
            return None
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def human_bytes(n: int | None) -> str:
    if n is None:
        return "-"
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{int(x)} B" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TB"


def display(path: Path | str) -> str:
    """尽量用相对仓库根的路径展示；在仓库外则用绝对路径。"""
    p = Path(path)
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(p)


def rel_link(target: Path, link: Path) -> None:
    """在 link 处创建指向 target 的相对符号链接（已存在则跳过）。"""
    if link.is_symlink() or link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.relpath(target, link.parent), link)
