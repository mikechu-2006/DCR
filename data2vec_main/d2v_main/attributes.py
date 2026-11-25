"""Structured attribute channels for the route-B data2vec front-end.

The 8-channel piano roll (4 onset + 4 hold, data2vec_main/d2v_main/charts.py)
encodes only note onset/hold state.  This module adds typed attribute channels
that the .osu format already contains but the shared CM3P parser does not
expose: hold release, local BPM / beat and bar phase, effective scroll speed
(SV) and SV changes, kiai, and hitsound / sample-set / volume metadata.

Design notes
------------
* Frame grid, not a tokenizer: categorical fields become one Gaussian/spike
  channel per category (equivalent to a per-category embedding lookup), and
  continuous fields become one normalised value channel.  No sparse vocab.
* Global fixed normalisation only (log2 BPM/SV, /100 volume, sin/cos phase).
  No per-chart z-scoring, so absolute tempo/scroll information is preserved.
* Base 8 channels are NOT built here; charts.build_window stays the single
  source for channels 0..7, so the 8ch baseline is bit-identical.
* This module parses the .osu file itself (raw hitsound/volume/sample fields,
  the effects bitmask, hold ends).  It does not touch dcr/cm3p_offline.py,
  whose string-equality kiai transcription is kept for CM3P reproducibility.
* Semantics follow the upstream CM3P hitsound resolver where it matters for
  comparability: hitsound = field & 14 (normal bit dropped), per-note volume 0
  inherits the timing-point volume, HP lookup uses t + 5 ms.  One deliberate
  deviation: a timing-point sample set of 0 stays "auto" instead of being
  coerced to soft(2); an explicit ss_auto channel records raw auto notes.

Channel groups (order is fixed; append-only, base first):

    base        8   onset_0..3 + hold_0..3        (built by charts.py)
    release     4   release_0..3                  hold-end Gaussians per column
    tempo       5   bpm, beat_sin, beat_cos, bar_sin, bar_cos
    sv          2   sv, sv_change
    kiai        1   kiai
    meta        7   hs_whistle, hs_finish, hs_clap,
                    ss_soft, ss_drum, ss_auto, volume

Arms: A0 = base (8), A1 = +release (12), A2 = +tempo +sv +kiai (20),
A3 = +meta (27).
"""
from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .charts import gaussian_kernel  # noqa: E402

N_COLS = 4
HOLD_OFFSET = 4
BASE_CHANNELS = tuple(
    ["onset_0", "onset_1", "onset_2", "onset_3"]
    + ["hold_0", "hold_1", "hold_2", "hold_3"]
)

# canonical group order and per-group channel names (base is always present)
_GROUP_ORDER = ("base", "release", "tempo", "sv", "kiai", "meta")
_RELEASE_CHANNELS = tuple("release_%d" % c for c in range(N_COLS))
_TEMPO_CHANNELS = ("bpm", "beat_sin", "beat_cos", "bar_sin", "bar_cos")
_SV_CHANNELS = ("sv", "sv_change")
_KIAI_CHANNELS = ("kiai",)
_META_CHANNELS = ("hs_whistle", "hs_finish", "hs_clap",
                  "ss_soft", "ss_drum", "ss_auto", "volume")

HS_WHISTLE = 2
HS_FINISH = 4
HS_CLAP = 8
HP_QUERY_OFFSET_MS = 5.0   # upstream CM3P hitsound_point_at adds 5 ms


# --------------------------------------------------------------------- spec
@dataclass(frozen=True)
class ChannelSpec:
    """Ordered channel layout derived from cfg["data"]["channels"]."""

    groups: tuple[str, ...]
    names: tuple[str, ...]
    version: str = "attr_v1"
    shuffle: bool = False   # negative control: roll the attribute block per chart

    @property
    def n_channels(self) -> int:
        return len(self.names)

    def slice_of(self, group: str) -> slice:
        if group not in self.groups:
            raise KeyError("channel group %r not enabled" % group)
        start = 0
        for g in self.groups:
            n = len(_group_names(g))
            if g == group:
                return slice(start, start + n)
            start += n
        raise KeyError(group)

    def channel_index(self, name: str) -> int:
        return self.names.index(name)

    @property
    def attr_names(self) -> tuple[str, ...]:
        """Channel names of the attribute block (everything after base 8)."""
        return tuple(n for n in self.names if n not in BASE_CHANNELS)

    def attr_slice_of(self, group: str) -> slice:
        """Slice into the attribute block returned by build_attribute_window."""
        if group == "base":
            raise KeyError("base channels are built by charts.build_window")
        if group not in self.groups:
            raise KeyError("channel group %r not enabled" % group)
        start = 0
        for g in self.groups:
            if g == "base":
                continue
            n = len(_group_names(g))
            if g == group:
                return slice(start, start + n)
            start += n
        raise KeyError(group)

    def attr_index(self, name: str) -> int:
        return self.attr_names.index(name)

    @classmethod
    def base(cls) -> "ChannelSpec":
        return cls(groups=("base",), names=BASE_CHANNELS)

    @classmethod
    def from_config(cls, cfg: dict | None) -> "ChannelSpec":
        """Build the layout from cfg["data"]["channels"].

        Accepted shapes (missing => base only):
            channels: {version: attr_v1, groups: {release: {}, tempo: {...}, ...}}
            channels: "base"
        """
        data = (cfg or {}).get("data", {}) if isinstance(cfg, dict) else {}
        spec_cfg = data.get("channels")
        if spec_cfg is None or spec_cfg == "base":
            return cls.base()
        if not isinstance(spec_cfg, dict):
            raise ValueError("data.channels must be a mapping or 'base'")
        version = str(spec_cfg.get("version", "attr_v1"))
        shuffle = bool(spec_cfg.get("shuffle", False))
        groups_cfg = spec_cfg.get("groups") or {}
        groups = ["base"]
        for g in _GROUP_ORDER:
            if g == "base":
                continue
            flag = groups_cfg.get(g)
            if flag is None or flag is False:
                continue
            if isinstance(flag, dict) and flag.get("enabled") is False:
                continue
            groups.append(g)
        names: list[str] = []
        for g in groups:
            names.extend(_group_names(g, groups_cfg.get(g) or {}))
        return cls(groups=tuple(groups), names=tuple(names), version=version,
                   shuffle=shuffle)


def _group_names(group: str, cfg: dict | None = None) -> tuple[str, ...]:
    cfg = cfg or {}
    if group == "base":
        return BASE_CHANNELS
    if group == "release":
        return _RELEASE_CHANNELS
    if group == "tempo":
        names: list[str] = []
        if cfg.get("bpm", True):
            names.append("bpm")
        if str(cfg.get("beat_phase", "sin_cos")) == "sin_cos":
            names.extend(["beat_sin", "beat_cos"])
        if str(cfg.get("bar_phase", "sin_cos")) == "sin_cos":
            names.extend(["bar_sin", "bar_cos"])
        return tuple(names)
    if group == "sv":
        names = []
        if cfg.get("level", True):
            names.append("sv")
        if cfg.get("change_pulse", True):
            names.append("sv_change")
        return tuple(names)
    if group == "kiai":
        return _KIAI_CHANNELS
    if group == "meta":
        names = []
        if cfg.get("hitsound_presence", False):
            names.append("hs_presence")
        if cfg.get("hitsound", True):
            names.extend(["hs_whistle", "hs_finish", "hs_clap"])
        if cfg.get("sample_set", True):
            names.extend(["ss_soft", "ss_drum", "ss_auto"])
        if cfg.get("volume", True):
            names.append("volume")
        return tuple(names)
    raise KeyError("unknown channel group %r" % group)


# --------------------------------------------------------------- raw parsing
def _int(x, default: int = 0) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def _float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _clip_int(x, lo: int, hi: int, default: int = 0) -> int:
    """Parse an int and clip it into a plausible .osu range.

    The corpus contains malformed hit objects (verified on 21,949 cached
    charts): e.g. an object at type=5 (circle) whose field 5 is an end-time
    like 90568:0:0:0:0, and hitSound=99999.  Out-of-range values are treated
    as the field default instead of raising (or silently wrapping) later.
    """
    v = _int(x, default)
    if v < lo or v > hi:
        return default
    return v


def _sections(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    cur = None
    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        if line.startswith("[") and line.endswith("]"):
            cur = line[1:-1]
            out.setdefault(cur, [])
        elif cur is not None:
            out[cur].append(line)
    return out


def _kv(lines, key: str):
    for line in lines:
        s = line.strip()
        if s.startswith(key + ":"):
            return s.split(":", 1)[1].strip()
    return None


@dataclass
class ChartAttrs:
    """Everything the attribute channels need from one chart (numpy arrays)."""

    path: str
    # notes (sorted by time)
    t_ms: np.ndarray          # float64
    end_ms: np.ndarray        # float64, nan for taps
    col: np.ndarray           # int8
    kind: np.ndarray          # int8, 0 circle, 1 hold
    hitsound: np.ndarray      # int16 raw bitmask
    ss_raw: np.ndarray        # int8 raw sample set (0 = auto)
    vol_raw: np.ndarray       # int16 raw volume (0 = inherit)
    # timing points (sorted by offset)
    tp_off: np.ndarray        # float64
    tp_vol: np.ndarray        # int16
    tp_ss: np.ndarray         # int8
    tp_kiai: np.ndarray       # bool
    # red lines only (uninherited, mpb > 0)
    red_off: np.ndarray       # float64
    red_mpb: np.ndarray       # float64
    red_meter: np.ndarray     # int16
    # effective SV groups (CM3P normalisation; dcr.cm3p_offline.sv_groups)
    sv_off: np.ndarray        # float64
    sv_val: np.ndarray        # float64

    @property
    def n_notes(self) -> int:
        return int(self.t_ms.size)


@lru_cache(maxsize=512)
def parse_attributes(path: str | Path, frame_hz: int = 100) -> ChartAttrs:
    """Parse one .osu file into the raw attribute arrays (cached per worker).

    frame_hz is accepted for signature symmetry with parse_chart but the raw
    arrays are kept in milliseconds; frame conversion happens in the window
    builder.
    """
    return _parse_attributes(path, frame_hz)


def _parse_attributes(path: str | Path, frame_hz: int = 100) -> ChartAttrs:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    sec = _sections(text)

    # ---- timing points
    tp_off: list[float] = []
    tp_vol: list[int] = []
    tp_ss: list[int] = []
    tp_kiai: list[bool] = []
    tp_mpb: list[float] = []
    tp_meter: list[int] = []
    tp_uninherited: list[bool] = []
    for line in sec.get("TimingPoints", []):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        f = [x.strip() for x in line.split(",")]
        if len(f) < 2:
            continue
        off = _float(f[0])
        mpb = _float(f[1])
        # corpus contains malformed timing points (e.g. meter=99999, NaN offsets)
        if not (math.isfinite(off) and math.isfinite(mpb)):
            continue
        if abs(off) > 86_400_000.0:
            continue
        uninherited = (f[6] not in ("0", "0.0", "false", "False")) if len(f) >= 7 else mpb > 0
        effects = _int(f[7]) if len(f) > 7 else 0
        tp_off.append(off)
        tp_mpb.append(mpb)
        tp_meter.append(_clip_int(f[2], 1, 32, 4) if len(f) > 2 else 4)
        tp_ss.append(_clip_int(f[3], 0, 3, 0) if len(f) > 3 else 0)
        tp_vol.append(_clip_int(f[5], 0, 100, 100) if len(f) > 5 else 100)
        tp_uninherited.append(bool(uninherited))
        tp_kiai.append(bool(effects & 1))

    # ---- hit objects
    t_ms: list[int] = []
    end_ms: list[float] = []
    cols: list[int] = []
    kinds: list[int] = []
    hs: list[int] = []
    ss_raw: list[int] = []
    vol_raw: list[int] = []
    for line in sec.get("HitObjects", []):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        f = [x.strip() for x in line.split(",")]
        if len(f) < 4:
            continue
        x = _int(f[0])
        t = _int(f[2])
        flags = _int(f[3])
        if flags & 128:
            kind = 1
        elif flags & 1:
            kind = 0
        else:
            continue                     # sliders/spinners are illegal in mania
        hitsound = _clip_int(f[4], 0, 15, 0) if len(f) > 4 else 0
        parts = f[5].split(":") if len(f) > 5 and f[5] else []
        end = float("nan")
        if kind == 1:
            if parts:
                raw_end = _float(parts[0], float(t))
                end = raw_end if (raw_end > t and raw_end < t + 3_600_000) else float(t + 1)
            else:
                end = float(t + 1)
            parts = parts[1:]
        col = int(np.clip(x / 512.0 * 4, 0, 3))
        sample_set = _clip_int(parts[0], 0, 3, 0) if len(parts) > 0 else 0
        volume = _clip_int(parts[3], 0, 100, 0) if len(parts) > 3 else 0
        t_ms.append(t)
        end_ms.append(end)
        cols.append(col)
        kinds.append(kind)
        hs.append(hitsound)
        ss_raw.append(sample_set)
        vol_raw.append(volume)

    order = np.argsort(np.asarray(t_ms, dtype=np.int64), kind="stable") if t_ms else np.zeros(0, np.int64)
    t_arr = np.asarray(t_ms, dtype=np.float64)[order] if t_ms else np.zeros(0, np.float64)
    end_arr = np.asarray(end_ms, dtype=np.float64)[order] if t_ms else np.zeros(0, np.float64)
    col_arr = np.asarray(cols, dtype=np.int8)[order] if t_ms else np.zeros(0, np.int8)
    kind_arr = np.asarray(kinds, dtype=np.int8)[order] if t_ms else np.zeros(0, np.int8)
    hs_arr = np.asarray(hs, dtype=np.int16)[order] if t_ms else np.zeros(0, np.int16)
    ss_arr = np.asarray(ss_raw, dtype=np.int16)[order] if t_ms else np.zeros(0, np.int16)
    vol_arr = np.asarray(vol_raw, dtype=np.int16)[order] if t_ms else np.zeros(0, np.int16)

    off_arr = np.asarray(tp_off, dtype=np.float64)
    sort_tp = np.argsort(off_arr, kind="stable")
    off_arr = off_arr[sort_tp]
    vol_tp = np.asarray(tp_vol, dtype=np.int16)[sort_tp]
    ss_tp = np.asarray(tp_ss, dtype=np.int8)[sort_tp]
    kiai_tp = np.asarray(tp_kiai, dtype=bool)[sort_tp]
    mpb_tp = np.asarray(tp_mpb, dtype=np.float64)[sort_tp]
    meter_tp = np.asarray(tp_meter, dtype=np.int16)[sort_tp]
    uninh_tp = np.asarray(tp_uninherited, dtype=bool)[sort_tp]

    red_mask = uninh_tp & (mpb_tp > 0)
    red_off = off_arr[red_mask]
    red_mpb = mpb_tp[red_mask]
    red_meter = meter_tp[red_mask]

    sv_off, sv_val = _cm3p_sv_groups(
        off_arr, mpb_tp, meter_tp, uninh_tp, kiai_tp, t_arr, end_arr, kind_arr)

    return ChartAttrs(
        path=str(p),
        t_ms=t_arr, end_ms=end_arr, col=col_arr, kind=kind_arr,
        hitsound=hs_arr, ss_raw=ss_arr, vol_raw=vol_arr,
        tp_off=off_arr, tp_vol=vol_tp, tp_ss=ss_tp, tp_kiai=kiai_tp,
        red_off=red_off, red_mpb=red_mpb, red_meter=red_meter,
        sv_off=sv_off, sv_val=sv_val,
    )


def _cm3p_sv_groups(off: np.ndarray, mpb: np.ndarray, meter: np.ndarray,
                     uninherited: np.ndarray, kiai: np.ndarray,
                     note_t: np.ndarray, note_end: np.ndarray, kind: np.ndarray):
    """Exact effective SV groups via dcr.cm3p_offline.sv_groups.

    Building the small OsuMania wrapper guarantees the same normalisation as
    the CM3P legacy token stream (median-mpb BPM normalisation, group merging).
    """
    if off.size == 0:
        return np.zeros(0), np.zeros(0)
    from dcr.cm3p_offline import HitObject, OsuMania, TimingPoint, sv_groups
    tps = [TimingPoint(float(off[i]), float(mpb[i]), int(meter[i]),
                       bool(uninherited[i]), bool(kiai[i]))
           for i in range(off.size)]
    objects = []
    for i in range(note_t.size):
        end = None if not np.isfinite(note_end[i]) else int(note_end[i])
        objects.append(HitObject(int(note_t[i]), "hold" if int(kind[i]) == 1 else "circle",
                                 0, end))
    osu = OsuMania(path=Path("."), mode=3, circle_size=4, timing=tps, objects=objects,
                   n_unsupported=0, _red=[tp for tp in tps if tp.uninherited])
    osu._red_offsets = np.asarray([tp.offset_ms for tp in osu._red], dtype=np.float64)
    groups = sv_groups(osu, normalized=True, add_snapping=False)
    return (np.asarray([float(g.time) for g in groups], dtype=np.float64),
            np.asarray([float(g.scroll) for g in groups], dtype=np.float64))


def _unused_effective_sv_groups(off: np.ndarray, mpb: np.ndarray, uninherited: np.ndarray):
    """CM3P definition of effective SV groups (BPM-normalised scroll speed).

    Mirrors dcr.cm3p_offline.sv_groups(normalized=True) but works on arrays:
    red lines -> scroll 1.0; inherited -> clip(-100/mpb, 0.01, 10); then
    multiply by median_mpb / current_red_mpb.  A group is emitted when the
    normalised value changes (or at the first timing point).
    """
    if off.size == 0:
        return np.zeros(0), np.zeros(0)
    # median mpb: same "dominant BPM" rule as CM3P median_mpb (duration weighted)
    red = np.flatnonzero(uninherited & (mpb > 0))
    med = float(red and np.median(mpb[red]) or 120.0)
    # CM3P median_mpb is duration-weighted; use a robust duration-weighted mode
    if red.size:
        weights = np.diff(np.concatenate([off[red], [off[-1] + 10.0]]))
        weights = np.maximum(weights, 0.0)
        if weights.sum() > 0:
            best = {}
            for i, r in enumerate(red):
                key = round(float(mpb[r]), 6)
                best[key] = best.get(key, 0.0) + float(weights[i])
            med = max(best.items(), key=lambda kv: kv[1])[0]
    groups_off: list[float] = []
    groups_val: list[float] = []
    cur_mpb = med
    last = float("nan")
    n = off.size
    for i in range(n):
        if uninherited[i]:
            if mpb[i] > 0:
                cur_mpb = mpb[i]
            scroll = 1.0
        else:
            scroll = -100.0 / mpb[i] if mpb[i] != 0 else 1.0
            scroll = float(np.clip(scroll, 0.01, 10.0))
        if i == n - 1 or off[i + 1] > off[i]:
            norm = scroll * med / cur_mpb if cur_mpb else scroll
            if (not math.isnan(last)) and norm == last:
                continue
            groups_off.append(float(off[i]))
            groups_val.append(float(norm))
            last = norm
    return np.asarray(groups_off, dtype=np.float64), np.asarray(groups_val, dtype=np.float64)


# ------------------------------------------------------------- window builder
def _gauss_pulse(out: np.ndarray, ch: int, frame: int, window: int,
                 kernel: np.ndarray, amplitude: float = 1.0) -> None:
    radius = kernel.size // 2
    lo = max(0, frame - radius)
    hi = min(window, frame + radius + 1)
    if hi <= lo:
        return
    klo = lo - (frame - radius)
    out[ch, lo:hi] += amplitude * kernel[klo:klo + (hi - lo)]


def build_attribute_window(attrs: ChartAttrs, start: int, window_frames: int,
                           frame_hz: int = 100, spec: ChannelSpec | None = None,
                           sigma_ms: float = 8.0) -> np.ndarray:
    """Build the attribute block (all non-base channels) for one window.

    Returns (spec.n_channels - 8, window_frames) float32 in spec order.
    """
    spec = spec or ChannelSpec.from_config(
        {"data": {"channels": {"groups": {"release": {}, "tempo": {},
                                         "sv": {}, "kiai": {}, "meta": {}}}}})
    names = [n for n in spec.names if n not in BASE_CHANNELS]
    out = np.zeros((len(names), int(window_frames)), dtype=np.float32)
    idx = {name: i for i, name in enumerate(names)}
    window = int(window_frames)
    start = max(0, int(start))
    win_start_ms = start * 1000.0 / frame_hz
    win_end_ms = (start + window) * 1000.0 / frame_hz
    tms = np.arange(start, start + window, dtype=np.float64) * 1000.0 / frame_hz

    # ---- release: per-column Gaussian at hold end frames
    if "release_0" in idx:
        kernel = gaussian_kernel(sigma_ms, frame_hz)
        radius = kernel.size // 2
        hold = np.flatnonzero(attrs.kind == 1)
        if hold.size:
            ends = attrs.end_ms[hold]
            valid = np.isfinite(ends) & (ends >= attrs.t_ms[hold])
            hold = hold[valid]
            ends = ends[valid]
            end_frame = np.rint(ends * frame_hz / 1000.0).astype(np.int64)
            in_win = np.flatnonzero((end_frame >= start - radius) & (end_frame < start + window + radius))
            for j in in_win:
                ch = idx.get("release_%d" % int(attrs.col[hold[j]]))
                if ch is None:
                    continue
                _gauss_pulse(out, ch, int(end_frame[j] - start), window, kernel)
        for c in range(N_COLS):
            ch = idx.get("release_%d" % c)
            if ch is not None:
                np.clip(out[ch], 0.0, 1.0, out=out[ch])

    # ---- tempo: bpm, beat phase, bar phase (piecewise per red line)
    if "bpm" in idx or "beat_sin" in idx or "bar_sin" in idx:
        if attrs.red_off.size:
            ri = np.searchsorted(attrs.red_off, tms, side="right") - 1
            ri = np.clip(ri, 0, attrs.red_off.size - 1)
            mpb = attrs.red_mpb[ri]
            off = attrs.red_off[ri]
            meter = np.maximum(1.0, attrs.red_meter[ri].astype(np.float64))
            bpm = 60000.0 / np.maximum(1e-6, mpb)
            if "bpm" in idx:
                out[idx["bpm"]] = np.clip(np.log2(bpm / 120.0) / 2.0, -1.0, 1.0)
            beats = (tms - off) / np.maximum(1e-6, mpb)
            phase = 2.0 * math.pi * beats
            if "beat_sin" in idx:
                out[idx["beat_sin"]] = np.sin(phase)
                out[idx["beat_cos"]] = np.cos(phase)
            if "bar_sin" in idx:
                bar_phase = 2.0 * math.pi * (beats / meter)
                out[idx["bar_sin"]] = np.sin(bar_phase)
                out[idx["bar_cos"]] = np.cos(bar_phase)

    # ---- sv: normalized effective scroll speed + change pulses
    if "sv" in idx and attrs.sv_off.size:
        si = np.clip(np.searchsorted(attrs.sv_off, tms, side="right") - 1, 0, attrs.sv_off.size - 1)
        val = np.clip(attrs.sv_val[si], 1e-3, None)
        out[idx["sv"]] = np.clip(np.log2(val) / 2.0, -1.5, 1.5)
    if "sv_change" in idx and attrs.sv_off.size:
        lo = int(np.searchsorted(attrs.sv_off, win_start_ms, side="left"))
        hi = int(np.searchsorted(attrs.sv_off, win_end_ms, side="left"))
        frames = np.rint(attrs.sv_off[lo:hi] * frame_hz / 1000.0).astype(np.int64) - start
        ok = (frames >= 0) & (frames < window)
        out[idx["sv_change"], frames[ok]] = 1.0

    # ---- kiai: state of the last timing point at/before each frame
    if "kiai" in idx and attrs.tp_off.size:
        ti = np.searchsorted(attrs.tp_off, tms, side="right") - 1
        state = np.zeros(tms.size, dtype=np.float32)
        valid = ti >= 0
        state[valid] = attrs.tp_kiai[ti[valid]].astype(np.float32)
        out[idx["kiai"]] = state

    # ---- meta: per-note hitsound / sample set / volume pulses
    if any(n in idx for n in ("hs_presence", "hs_whistle", "hs_finish", "hs_clap",
                              "ss_soft", "ss_drum", "ss_auto", "volume")):
        lo = int(np.searchsorted(attrs.t_ms, win_start_ms, side="left"))
        hi = int(np.searchsorted(attrs.t_ms, win_end_ms, side="left"))
        for j in range(lo, hi):
            t = float(attrs.t_ms[j])
            f = int(round(t * frame_hz / 1000.0)) - start
            if f < 0 or f >= window:
                continue
            hs = int(attrs.hitsound[j]) & 14
            if hs and "hs_presence" in idx:
                out[idx["hs_presence"], f] = 1.0
            if hs & HS_WHISTLE and "hs_whistle" in idx:
                out[idx["hs_whistle"], f] = 1.0
            if hs & HS_FINISH and "hs_finish" in idx:
                out[idx["hs_finish"], f] = 1.0
            if hs & HS_CLAP and "hs_clap" in idx:
                out[idx["hs_clap"], f] = 1.0
            # resolve inheritance at the hitsound point (t + 5 ms, upstream rule)
            k = int(np.searchsorted(attrs.tp_off, t + HP_QUERY_OFFSET_MS, side="right")) - 1
            tp_ss = int(attrs.tp_ss[k]) if 0 <= k < attrs.tp_ss.size else 0
            tp_vol = int(attrs.tp_vol[k]) if 0 <= k < attrs.tp_vol.size else 100
            raw_ss = int(attrs.ss_raw[j])
            ss = raw_ss if raw_ss > 0 else tp_ss
            if raw_ss == 0 and "ss_auto" in idx:
                out[idx["ss_auto"], f] = 1.0
            if ss == 2 and "ss_soft" in idx:
                out[idx["ss_soft"], f] = 1.0
            elif ss == 3 and "ss_drum" in idx:
                out[idx["ss_drum"], f] = 1.0
            if "volume" in idx:
                raw_vol = int(attrs.vol_raw[j])
                vol = raw_vol if raw_vol > 0 else tp_vol
                val = float(np.clip(vol, 0, 100)) / 100.0
                if val > out[idx["volume"], f]:
                    out[idx["volume"], f] = val
    if spec.shuffle:
        # Negative control: roll the whole attribute block by a deterministic
        # per-chart offset.  Marginal statistics are preserved, but alignment
        # between attributes and note onsets is destroyed.
        off = int.from_bytes(hashlib.md5(str(attrs.path).encode()).digest()[:4], "big") % window
        out = np.roll(out, off, axis=1)
    return out


def full_channel_names(spec: ChannelSpec) -> list[str]:
    return list(spec.names)
