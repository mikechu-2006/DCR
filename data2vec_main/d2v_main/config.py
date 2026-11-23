"""Config loading: YAML on top of a small set of defaults."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "data": {
        "osu_dir": "data/osu_cache",
        "ids_file": None,
        "window_sec": 16.0,
        "frame_hz": 100,
        "gaussian_sigma_ms": 8.0,
        "min_valid_sec": 1.0,
    },
    "model": {
        "in_channels": 8,
        "conv_hidden": 256,
        "dim": 512,
        "layers": 12,
        "heads": 8,
        "ff_mult": 4,
        "dropout": 0.1,
        "top_k": 6,
        "mask_prob": 0.065,
        "mask_length": 10,
        "loss_beta": 1.0,
        "max_frames": 4096,
    },
    "train": {
        "batch_size": 16,
        "lr": 1.0e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "epochs": 20,
        "grad_clip": 1.0,
        "ema_start": 0.999,
        "ema_end": 0.9999,
        "ema_anneal_steps": 30000,
        "log_every": 50,
        "save_every": 2000,
        "num_workers": 4,
        "amp": True,
        "device": "auto",
    },
}


def _deep_update(base: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            patch = yaml.safe_load(fh) or {}
        if not isinstance(patch, dict):
            raise ValueError(f"{path}: YAML root must be a mapping")
        _deep_update(cfg, patch)
    if overrides:
        _deep_update(cfg, overrides)
    return cfg


def dump_config(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True)
