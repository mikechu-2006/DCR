"""Chart -> 100 Hz multi-channel piano roll.

This is the audio-style front-end of route B: the .osu hit objects are turned into
a dense 8-channel signal (4 onset + 4 hold) that a 1D conv encoder can consume,
instead of a discrete 8-bit token vocabulary.

Reuse note: the .osu parser lives in dcr/cm3p_offline.py (tested by the rest of
the repository).  We only depend on that module, not on transformers/CM3P.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dcr.cm3p_offline import mania_column, parse_osu  # noqa: E402

N_CHANNELS = 8
HOLD_OFFSET = 4


class ChartError(ValueError):
    """Raised when a file is not a usable 4K mania chart."""


@dataclass(frozen=True)
class NoteEvent:
    t: int          # onset frame at frame_hz
    col: int        # 0..3
    end: int | None # hold end frame (exclusive), None for taps


def _frame(ms: float | int | None, frame_hz: int) -> int:
    return int(round(float(ms) * frame_hz / 1000.0))


@lru_cache(maxsize=512)
def parse_chart(path: str, frame_hz: int = 100) -> tuple[tuple[NoteEvent, ...], int]:
    """Parse one .osu file into (events, n_frames). Cached per worker (bounded)."""
    osu = parse_osu(path)
    if osu.mode != 3:
        raise ChartError(f"{path}: mode={osu.mode}, not mania")
    if osu.circle_size != 4:
        raise ChartError(f"{path}: circle_size={osu.circle_size}, only 4K")
    events: list[NoteEvent] = []
    for obj in osu.objects:
        if obj.kind not in ("circle", "hold"):
            continue
        col = int(mania_column(osu, obj.x))
        t = _frame(obj.time_ms, frame_hz)
        end = None
        if obj.kind == "hold" and obj.end_ms is not None:
            end = _frame(obj.end_ms, frame_hz)
            if end <= t:
                end = t + 1
        events.append(NoteEvent(t=t, col=col, end=end))
    if not events:
        return tuple(), 0
    events.sort(key=lambda e: (e.t, e.col))
    n_frames = max((e.end if e.end is not None else e.t) for e in events) + 1
    return tuple(events), int(n_frames)


def gaussian_kernel(sigma_ms: float, frame_hz: int) -> np.ndarray:
    sigma = max(1e-3, sigma_ms * frame_hz / 1000.0)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2).astype(np.float32)
    return k / max(1e-6, float(k.max()))


def build_roll(events: tuple[NoteEvent, ...] | list[NoteEvent], n_frames: int,
               frame_hz: int = 100, sigma_ms: float = 8.0) -> np.ndarray:
    """(8, n_frames) float32 piano roll. Channels 0..3 onset, 4..7 hold."""
    n_frames = max(1, int(n_frames))
    roll = np.zeros((N_CHANNELS, n_frames), dtype=np.float32)
    if not events:
        return roll
    kernel = gaussian_kernel(sigma_ms, frame_hz)
    radius = len(kernel) // 2
    for ev in events:
        col = int(ev.col)
        if not 0 <= col < 4:
            continue
        t = int(ev.t)
        lo = max(0, t - radius)
        hi = min(n_frames, t + radius + 1)
        if hi > lo:
            klo = lo - (t - radius)
            roll[col, lo:hi] += kernel[klo:klo + (hi - lo)]
        if ev.end is not None:
            s = max(0, t)
            e = min(n_frames, int(ev.end))
            if e > s:
                np.maximum(roll[HOLD_OFFSET + col, s:e], 1.0,
                           out=roll[HOLD_OFFSET + col, s:e])
    np.clip(roll[:4], 0.0, 4.0, out=roll[:4])
    return roll


def build_window(events: tuple[NoteEvent, ...] | list[NoteEvent], start: int,
                 window_frames: int, frame_hz: int = 100,
                 sigma_ms: float = 8.0) -> np.ndarray:
    """Build only one window (much cheaper than rolling the whole chart)."""
    start = max(0, int(start))
    end = start + int(window_frames)
    roll = np.zeros((N_CHANNELS, int(window_frames)), dtype=np.float32)
    if not events:
        return roll
    kernel = gaussian_kernel(sigma_ms, frame_hz)
    radius = len(kernel) // 2
    for ev in events:
        col = int(ev.col)
        if not 0 <= col < 4:
            continue
        t = int(ev.t)
        if start - radius <= t < end + radius:
            pos = t - start
            lo = max(0, pos - radius)
            hi = min(window_frames, pos + radius + 1)
            if hi > lo:
                klo = lo - (pos - radius)
                roll[col, lo:hi] += kernel[klo:klo + (hi - lo)]
        if ev.end is not None:
            s = max(start, t)
            e = min(end, int(ev.end))
            if e > s:
                np.maximum(roll[HOLD_OFFSET + col, s - start:e - start], 1.0,
                           out=roll[HOLD_OFFSET + col, s - start:e - start])
    np.clip(roll[:4], 0.0, 4.0, out=roll[:4])
    return roll


def slice_window(roll: np.ndarray, start: int, window_frames: int):
    """Return (8, window_frames) padded window and the number of valid frames."""
    start = max(0, int(start))
    seg = roll[:, start:start + window_frames]
    valid = int(seg.shape[1])
    if valid == window_frames:
        return np.ascontiguousarray(seg), valid
    out = np.zeros((roll.shape[0], window_frames), dtype=np.float32)
    if valid:
        out[:, :valid] = seg
    return out, valid


def window_starts(n_frames: int, window_frames: int, stride_frames: int | None = None,
                  random_start: bool = False, rng: np.random.Generator | None = None) -> list[int]:
    if n_frames <= 0:
        return [0]
    if random_start:
        hi = max(0, n_frames - window_frames)
        rng = rng or np.random.default_rng()
        return [int(rng.integers(0, hi + 1))]
    stride = int(stride_frames or window_frames)
    return list(range(0, n_frames, stride))


class ChartStore:
    """beatmap_id -> .osu path, optionally restricted to an ID list."""

    def __init__(self, osu_dir: str | Path, ids: list[int] | None = None):
        osu_dir = Path(osu_dir)
        if not osu_dir.is_absolute():
            osu_dir = _ROOT / osu_dir
        if not osu_dir.exists():
            raise FileNotFoundError(osu_dir)
        paths = {}
        for p in osu_dir.glob("*.osu"):
            if p.stem.isdigit():
                paths[int(p.stem)] = p
        if ids is not None:
            keep = set(int(i) for i in ids)
            paths = {k: v for k, v in paths.items() if k in keep}
        if not paths:
            raise ValueError(f"no .osu files under {osu_dir}")
        self.osu_dir = osu_dir
        self.paths = dict(sorted(paths.items()))

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def ids(self) -> list[int]:
        return list(self.paths)

    def path(self, chart_id: int) -> Path:
        return self.paths[int(chart_id)]
