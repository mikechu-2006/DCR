"""配置层：内置默认值 + YAML + --set 覆盖 + 校验。

一次实验 = 一个 YAML 文件。字段（全部可省略，缺省用下面的 DEFAULTS）::

    name: std_1k             # 实验名，附加到 experiments_{时间戳}_ 之后
    tag: 1k                  # 数据集标签，决定 step0_{tag} / step1_{tag} 等文件名
    python: python           # 解释器，默认与 main.py 相同
    cache_dir: data/processed  # 上游产物缓存目录（不跑上游时从这里取）
    osu_cache_dir: data/osu_cache  # .osu 谱面缓存（只有 .osu 正文，无音频/图片；跨实验共享）
    env: {OMP_NUM_THREADS: "4"}  # 追加的环境变量
    io:
      out_format: parquet    # step0 play 表格式: parquet | csv
      link_cache: true       # 复用缓存产物时，在实验目录里建一个相对符号链接
      hash: true             # 记录产物 sha256
      hash_max_mb: 512       # 超过这个大小的产物不哈希
    steps:                   # 只写要跑的 step；没写的 step 不跑
      step0: {version: preprocess, params: {dump_dir: ...}}
      step1: {version: clean, params: {acc_floor: 0.9}}
      step2: {version: predict, params: {models: [bias, mf_dot, mirt]}}
    extra_steps:             # 任意脚本也能挂进同一套编排
      - name: order_curve
        script: scripts/play_order_curve.py
        params: {tag: "{tag}", out-dir: "{results}"}
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .errors import DcrError
from .paths import REPO_ROOT, resolve_path

#: 内置默认值。实验 YAML 会与它深度合并，所以配置里只写「与默认不同」的部分。
DEFAULTS: dict[str, Any] = {
    "name": "",
    "tag": "1k",
    "python": None,
    "cache_dir": "data/processed",
    "osu_cache_dir": "data/osu_cache",
    "env": {},
    "io": {
        "out_format": "parquet",
        "link_cache": True,
        "hash": True,
        "hash_max_mb": 512,
    },
    "steps": {},
    "extra_steps": [],
}

#: 只写 params 不写 version 时使用的版本
STEP_DEFAULT_VERSION = {"step0": "preprocess", "step1": "clean", "step2": "predict"}

#: 主流程的固定顺序；extra_steps 排在它们后面
BUILTIN_ORDER = ("step0", "step1", "step2")


def load_config(path: str | Path) -> dict:
    """读取 YAML 配置（顶层必须是映射）。"""
    p = resolve_path(path)
    if not p.exists():
        raise DcrError(f"配置文件不存在: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DcrError(f"YAML 解析失败: {p}\n{exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise DcrError(f"配置顶层必须是映射，得到 {type(raw).__name__}: {p}")
    return dict(raw)


def deep_merge(base: Mapping, override: Mapping) -> dict:
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_override(text: str) -> tuple[str, Any]:
    """--set steps.step2.params.epochs=100 → ("steps.step2.params.epochs", 100)"""
    if "=" not in text:
        raise DcrError(f"--set 需要 KEY=VALUE 形式，收到: {text!r}")
    key, value = text.split("=", 1)
    key = key.strip()
    if not key:
        raise DcrError(f"--set 的 KEY 为空: {text!r}")
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        parsed = value
    return key, parsed


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    cur = cfg
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def resolve_config(raw: Mapping, overrides: Iterable[str] = ()) -> dict:
    """默认值 + 配置 + --set 覆盖 → 一份自洽的配置。"""
    cfg = deep_merge(DEFAULTS, dict(raw))
    for item in overrides:
        key, value = parse_override(item)
        set_dotted(cfg, key, value)
    _normalize(cfg)
    _validate(cfg)
    return cfg


def _normalize(cfg: dict) -> None:
    cfg["tag"] = str(cfg.get("tag", "1k"))
    cfg["name"] = "" if cfg.get("name") is None else str(cfg["name"])
    cfg["cache_dir"] = str(cfg.get("cache_dir") or "data/processed")
    cfg["osu_cache_dir"] = str(cfg.get("osu_cache_dir") or "data/osu_cache")
    if cfg.get("python") is not None:
        cfg["python"] = str(cfg["python"])
    if cfg.get("env") is None:
        cfg["env"] = {}
    if cfg.get("steps") is None:
        cfg["steps"] = {}
    if cfg.get("extra_steps") is None:
        cfg["extra_steps"] = []
    for key, value in list(cfg["steps"].items()):
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise DcrError(f"steps.{key} 必须是映射，得到 {type(value).__name__}")
        value = dict(value)
        if "version" not in value and key in STEP_DEFAULT_VERSION:
            value["version"] = STEP_DEFAULT_VERSION[key]
        cfg["steps"][key] = value


def _validate(cfg: dict) -> None:
    if not cfg["tag"]:
        raise DcrError("tag 不能为空")
    io = cfg["io"]
    if io.get("out_format") not in ("parquet", "csv"):
        raise DcrError(f"io.out_format 只能是 parquet 或 csv，得到 {io.get('out_format')!r}")
    if not isinstance(io.get("hash"), bool):
        raise DcrError("io.hash 必须是 true/false")
    if not isinstance(cfg["env"], Mapping):
        raise DcrError("env 必须是映射，如 {OMP_NUM_THREADS: 4}")
    for key, value in cfg["steps"].items():
        if key in STEP_DEFAULT_VERSION and value.get("version") is None:
            raise DcrError(f"steps.{key}.version 不能为空")
        params = value.get("params", {})
        if params is None:
            value["params"] = {}
        elif not isinstance(params, Mapping):
            raise DcrError(f"steps.{key}.params 必须是映射，得到 {type(params).__name__}")
        inputs = value.get("inputs", {})
        if inputs is None:
            value["inputs"] = {}
        elif not isinstance(inputs, Mapping):
            raise DcrError(f"steps.{key}.inputs 必须是映射（输入名 → 路径），得到 {type(inputs).__name__}")
    extras = cfg["extra_steps"]
    if not isinstance(extras, list):
        raise DcrError("extra_steps 必须是列表")
    for i, item in enumerate(extras):
        if not isinstance(item, Mapping):
            raise DcrError(f"extra_steps[{i}] 必须是映射")
        if not item.get("name"):
            raise DcrError(f"extra_steps[{i}] 缺少 name")
        if not item.get("script") and not item.get("cmd"):
            raise DcrError(f"extra_steps[{i}] ({item['name']}) 需要 script: 或 cmd:")


def dump_yaml(cfg: Mapping) -> str:
    return yaml.safe_dump(dict(cfg), sort_keys=False, allow_unicode=True, default_flow_style=False)


def cache_dir_of(cfg: Mapping) -> Path:
    return resolve_path(cfg.get("cache_dir") or "data/processed", REPO_ROOT)
