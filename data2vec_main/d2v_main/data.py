"""Datasets and collator for chart windows (on-the-fly piano rolls)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import index_cache as index_mod
from .charts import ChartError, ChartStore, build_window, parse_chart, window_starts


class ChartWindowDataset(Dataset):
    """Train: one random 16 s window per chart per epoch.

    Eval: every non-overlapping 16 s window (last one padded).
    The piano roll is built on the fly; parsed events are LRU-cached per worker.
    """

    def __init__(self, store: ChartStore, cfg: dict, train: bool,
                 index_cache: str | Path | None = None, use_index_cache: bool = True):
        self.store = store
        self.cfg = cfg
        self.train = train
        d = cfg["data"]
        self.frame_hz = int(d["frame_hz"])
        self.window_frames = int(round(float(d["window_sec"]) * self.frame_hz))
        self.sigma_ms = float(d["gaussian_sigma_ms"])
        self.min_valid = int(round(float(d["min_valid_sec"]) * self.frame_hz))
        self.rng = None
        self.items: list[tuple[int, int]] = []
        self.index: list[tuple[int, int]] = []
        bad = 0
        if use_index_cache:
            n_frames_map = index_mod.load(store, self.frame_hz, index_cache)
        else:
            n_frames_map = {}
            for cid in store.ids:
                try:
                    _, n_frames_map[cid] = parse_chart(str(store.path(cid)), self.frame_hz)
                except (ChartError, OSError, ValueError):
                    n_frames_map[cid] = -1
        for cid in store.ids:
            n_frames = int(n_frames_map.get(cid, -1))
            if n_frames < self.min_valid:
                bad += 1
                continue
            self.items.append((cid, n_frames))
            if not train:
                for start in window_starts(n_frames, self.window_frames):
                    self.index.append((cid, start))
        if not self.items:
            raise ValueError("no usable charts after filtering")
        if bad:
            print(f"[data] skipped {bad} unusable/short charts", flush=True)

    def __len__(self) -> int:
        return len(self.items) if self.train else len(self.index)

    def _rng(self) -> np.random.Generator:
        if self.rng is None:
            info = torch.utils.data.get_worker_info()
            seed = (int(self.cfg["seed"]) + 1) * 100003 + (info.id if info else 0)
            self.rng = np.random.default_rng(seed)
        return self.rng

    def __getitem__(self, i: int) -> dict:
        if self.train:
            cid, n_frames = self.items[i]
            start = window_starts(n_frames, self.window_frames, random_start=True,
                                  rng=self._rng())[0]
        else:
            cid, start = self.index[i]
            _, n_frames = parse_chart(str(self.store.path(cid)), self.frame_hz)
        events, _ = parse_chart(str(self.store.path(cid)), self.frame_hz)
        window = build_window(events, start, self.window_frames, self.frame_hz, self.sigma_ms)
        valid = max(0, min(self.window_frames, n_frames - start))
        return {
            "x": torch.from_numpy(window),
            "valid_len": int(valid),
            "chart_id": int(cid),
            "start": int(start),
        }


def collate_windows(batch: list[dict]) -> dict:
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "valid_len": torch.tensor([b["valid_len"] for b in batch], dtype=torch.long),
        "chart_id": torch.tensor([b["chart_id"] for b in batch], dtype=torch.long),
        "start": torch.tensor([b["start"] for b in batch], dtype=torch.long),
    }
