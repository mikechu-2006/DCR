"""Offline CM3P mania token accounting -- no transformers, no slider, no GPU.

Why this module exists
----------------------
T0/T1 of the CM3P crop plan need three numbers before anything touches a GPU:

  * what the beatmap tower actually tokenises (6 or 5 tokens per tap?),
  * how many tokens a window/chart costs, and
  * whether removing hitsounds / kiai / mania SV can remove *core* chart
    information (note identity, time, snapping, column, hold end, timing grid).

The published pipeline is not importable on a laptop (no transformers, no
slider, no network), but the .osu format is simple enough to reproduce the
*structural* token stream exactly for 4K mania.  Every rule below is a
transcription of external/cm3p/cm3p/parsing_cm3p.py and
tokenization_cm3p.py at commit 546f2e2; the places where that is not true are
called out in comments.  The output is therefore an accounting model, not a
tokenizer: ids/values are irrelevant, only the number and class of tokens.

The audit that matters
----------------------
audit_chart() builds the full (legacy) and cropped group streams and asserts

    [g.core() for g in full if g.kind not in ("kiai", "sv")]
        == [g.core() for g in crop]

i.e. after the three flags are switched off, the note/timing grid is *identical*
- same order, same times, same snapping, same columns, same hold ends.  The
only differences are the removed hitsound/volume pairs, kiai markers and mania
scroll-speed markers, which are counted exactly.  If that assertion ever fails,
the crop is losing chart content and the run must stop.

A core-information gap in upstream (found while writing this module)
-------------------------------------------------------------------
The parser stores the mania column on Group.column (parsing_cm3p.py:501)
but the tokenizer reads Group.mania_column (tokenization_cm3p.py:191).
The dataclass field is mania_column, so Group.mania_column stays None
and **no [MANIA_COLUMN_*] token is ever emitted**, in any commit of the
upstream repository.  The plan's "6 tokens per tap" is therefore 5 in the
running code, and the frozen CM3P model has never seen a mania column token.
This module reports both streams (columns=False = what the pipeline does,
columns=True = what the plan assumed / what a patched parser would emit)
so the difference is never silent.  dcr/cm3p_windows.py can install an
opt-in runtime patch that fixes the attribute *and* the 0-based/1-based
off-by-one its tokenizer has as well.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

# Token classes, in the order the tokenizer emits them inside a group.
EVENT_CORE = ("circle", "hold", "hold_end", "beat", "measure", "timing_point")
REMOVED_KINDS = ("kiai", "sv")

# Per-kind event-token name; only needed for human-readable audits.
KIND_LABEL = {
    "circle": "[CIRCLE]",
    "hold": "[HOLD_NOTE]",
    "hold_end": "[HOLD_NOTE_END]",
    "beat": "[BEAT]",
    "measure": "[MEASURE]",
    "timing_point": "[TIMING_POINT]",
    "kiai": "[KIAI]",
    "sv": "[SCROLL_SPEED_CHANGE]",
    "other": "[OTHER]",
}

# stable merge order at equal timestamps (parse_beatmap merges timing last,
# so timing groups come first; rank is only used for deterministic ordering).
_KIND_RANK = {"timing_point": 0, "measure": 0, "beat": 0,
              "kiai": 1, "sv": 2, "circle": 3, "hold": 3, "hold_end": 3,
              "other": 3}


@dataclass
class TimingPoint:
    offset_ms: float
    ms_per_beat: float
    meter: int
    uninherited: bool
    kiai: bool


@dataclass
class HitObject:
    time_ms: int
    kind: str                 # circle / hold / slider / spinner / other
    x: int = 0
    end_ms: Optional[int] = None


@dataclass
class OsuMania:
    path: Path
    mode: int = 3
    circle_size: int = 4
    timing: list[TimingPoint] = field(default_factory=list)
    objects: list[HitObject] = field(default_factory=list)
    n_unsupported: int = 0
    # cached red lines for O(log n) snapping (charts can have >20k timing points)
    _red: list = field(default_factory=list)
    _red_offsets: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n_notes(self) -> int:
        return len(self.objects)


@dataclass
class GroupTok:
    """One parser group, reduced to the information the tokenizer uses."""
    time: float
    kind: str
    snapping: Optional[int] = None
    column: Optional[int] = None
    scroll: Optional[float] = None
    hitsounds: int = 0

    def n_tokens(self) -> int:
        # event token + time-shift token, then one token per non-None feature
        n = 2
        if self.snapping is not None:
            n += 1
        if self.column is not None:
            n += 1
        if self.scroll is not None:
            n += 1
        n += 2 * self.hitsounds          # [HITSOUND_*] + [VOLUME_*] per entry
        return n

    def core(self) -> tuple:
        """Everything except the removable feature payload (hitsounds/kiai/sv)."""
        return (self.kind, round(self.time, 6), self.snapping, self.column,
                round(self.scroll, 6) if self.scroll is not None else None)

    def label(self) -> str:
        return KIND_LABEL.get(self.kind, self.kind)


# ------------------------------------------------------------------ .osu parse
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


def _kv(lines: Sequence[str], key: str) -> Optional[str]:
    for line in lines:
        s = line.strip()
        if s.startswith(key + ":"):
            return s.split(":", 1)[1].strip()
    return None


def parse_osu(path: Path | str) -> OsuMania:
    """Minimal .osu reader: mode, circle size, timing points, hit objects."""
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    sec = _sections(text)
    mode = 0
    m = _kv(sec.get("General", []), "Mode")
    if m:
        try:
            mode = int(m)
        except ValueError:
            mode = 0
    cs = 4
    c = _kv(sec.get("Difficulty", []), "CircleSize")
    if c:
        try:
            cs = int(float(c))
        except ValueError:
            cs = 4

    timing: list[TimingPoint] = []
    for line in sec.get("TimingPoints", []):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        f = [x.strip() for x in line.split(",")]
        if len(f) < 2:
            continue
        try:
            offset = float(f[0])
            mpb = float(f[1])
        except ValueError:
            continue
        meter = 4
        if len(f) > 2:
            try:
                meter = int(float(f[2]))
            except ValueError:
                meter = 4
        if len(f) >= 7 and f[6].strip() in ("0", "0.0", "false", "False"):
            uninherited = False
        else:
            uninherited = mpb > 0
        kiai = False
        if len(f) > 7:
            kiai = f[7].strip() in ("1", "1.0", "true", "True")
        timing.append(TimingPoint(offset, mpb, max(1, meter), uninherited, kiai))

    objects: list[HitObject] = []
    n_unsupported = 0
    for line in sec.get("HitObjects", []):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        f = [x.strip() for x in line.split(",")]
        if len(f) < 4:
            continue
        try:
            x = int(float(f[0]))
            t = int(float(f[2]))
            flags = int(f[3])
        except ValueError:
            continue
        end = None
        if flags & 128:                       # mania hold
            kind = "hold"
            if len(f) > 5 and f[5]:
                try:
                    end = int(float(f[5].split(":")[0]))
                except ValueError:
                    end = t
        elif flags & 1:
            kind = "circle"
        elif flags & 2:
            kind = "slider"
            n_unsupported += 1
            if len(f) > 5 and f[5]:
                try:
                    end = int(float(f[5].split(":")[0]))
                except ValueError:
                    end = t
        elif flags & 8:
            kind = "spinner"
            n_unsupported += 1
            if len(f) > 5 and f[5]:
                try:
                    end = int(float(f[5].split(":")[0]))
                except ValueError:
                    end = t
        else:
            kind = "other"
            n_unsupported += 1
        objects.append(HitObject(t, kind, x, end))
    objects.sort(key=lambda o: o.time_ms)
    red = [tp for tp in timing if tp.uninherited]
    return OsuMania(p, mode, cs, timing, objects, n_unsupported,
                    _red=red,
                    _red_offsets=np.asarray([tp.offset_ms for tp in red],
                                            dtype=np.float64))


# -------------------------------------------------------------------- helpers
def song_length_ms(osu: OsuMania) -> float:
    """parsing_cm3p.get_song_length for a beatmap (no audio)."""
    if osu.objects:
        last = osu.objects[-1]                # hit_objects() is time-sorted
        end = last.end_ms if last.end_ms is not None else last.time_ms
        return end + 0.999                    # + 0.000999 s
    if osu.timing:
        return osu.timing[-1].offset_ms + 10.0
    return 0.0


def _last_object_end_ms(osu: OsuMania) -> int:
    if not osu.objects:
        return 0
    last = osu.objects[-1]
    return int(last.end_ms if last.end_ms is not None else last.time_ms)


def median_mpb(osu: OsuMania) -> float:
    """parsing_cm3p.get_median_mpb_beatmap (osu! stable's dominant BPM)."""
    last_ms = _last_object_end_ms(osu)
    this = 0.0
    durations: dict[float, int] = {}
    for i in range(len(osu.timing) - 1, -1, -1):
        tp = osu.timing[i]
        offset = int(tp.offset_ms)
        if tp.uninherited:
            this = tp.ms_per_beat
        if this == 0 or offset > last_ms or (not tp.uninherited and i > 0):
            continue
        durations[this] = durations.get(this, 0) + int(last_ms - (0 if i == 0 else offset))
        last_ms = offset
    best, best_d = 0.0, 0
    for mpb, dur in durations.items():
        if dur > best_d:
            best, best_d = mpb, dur
    return best


def _uninherited_at(osu: OsuMania, t: float) -> Optional[TimingPoint]:
    """The red line in effect at t (clamped to the first one for t < first).

    bisect over the cached red-line offsets: SV-heavy charts can have >20,000
    timing points and 40,000 notes, so the naive scan is O(notes * timing).
    """
    red = osu._red
    if not red:
        return None
    i = int(np.searchsorted(osu._red_offsets, t, side="right")) - 1
    return red[max(0, i)]


def snapping_at(osu: OsuMania, t: float) -> Optional[int]:
    """parsing_cm3p._get_snapping: 0..16, 0 = unsnapped, None = no red line."""
    tp = _uninherited_at(osu, t)
    if tp is None or tp.ms_per_beat <= 0:
        return None
    beats = (t - tp.offset_ms) / tp.ms_per_beat
    for i in range(1, 17):
        if abs(beats - round(beats * i) / i) * tp.ms_per_beat < 2.0:
            return i
    return 0


def mania_column(osu: OsuMania, x: int) -> int:
    """parsing_cm3p._get_mania_column -> 0-based column."""
    return int(np.clip(x / 512.0 * osu.circle_size, 0, max(0, osu.circle_size - 1)))


# ---------------------------------------------------------------- group stream
def timing_groups(osu: OsuMania, *, add_timing_points: bool = True) -> list[GroupTok]:
    """parsing_cm3p.parse_timing: BEAT/MEASURE/TIMING_POINT every beat."""
    tps = [tp for tp in osu.timing if tp.uninherited]
    if not tps:
        return []
    out: list[GroupTok] = []
    last_ms = int(song_length_ms(osu))
    for i, tp in enumerate(tps):
        nxt = tps[i + 1] if i + 1 < len(tps) else None
        next_ms = (nxt.offset_ms - 10.0) if nxt is not None else last_ms
        start = tp.offset_ms
        time = start
        counter = 0
        delta = tp.ms_per_beat
        while time <= next_ms:
            if add_timing_points and counter == 0:
                kind = "timing_point"
            elif counter % max(1, tp.meter) == 0:
                kind = "measure"
            else:
                kind = "beat"
            out.append(GroupTok(int(time + 1e-5), kind))
            if not np.isfinite(delta) or delta <= 10.0:
                break
            counter += 1
            time = start + counter * delta
    return out


def kiai_groups(osu: OsuMania, *, add_snapping: bool = True) -> list[GroupTok]:
    """parsing_cm3p.parse_kiai: one group per on/off switch."""
    out: list[GroupTok] = []
    kiai = False
    for tp in osu.timing:
        if tp.kiai == kiai:
            continue
        out.append(GroupTok(int(tp.offset_ms + 1e-5), "kiai",
                            snapping=(snapping_at(osu, tp.offset_ms)
                                      if add_snapping else None)))
        kiai = tp.kiai
    return out


def sv_groups(osu: OsuMania, *, normalized: bool = True,
               add_snapping: bool = True) -> list[GroupTok]:
    """parsing_cm3p.parse_scroll_speeds: one group per effective change."""
    if not osu.timing:
        return []
    med = median_mpb(osu)
    mpb = med
    out: list[GroupTok] = []
    last = -1.0
    n = len(osu.timing)
    for i, tp in enumerate(osu.timing):
        if tp.uninherited:
            mpb = tp.ms_per_beat
            scroll = 1.0
        else:
            scroll = -100.0 / tp.ms_per_beat if tp.ms_per_beat != 0 else 1.0
        if i == n - 1 or osu.timing[i + 1].offset_ms > tp.offset_ms:
            norm = (scroll * med / mpb) if (normalized and mpb) else scroll
            if norm != last or last == -1:
                out.append(GroupTok(int(tp.offset_ms + 1e-5), "sv",
                                    snapping=(snapping_at(osu, tp.offset_ms)
                                              if add_snapping else None),
                                    scroll=norm))
            last = norm
    return out


def chart_groups(osu: OsuMania, *, hitsounds: bool, kiai: bool, sv: bool,
                 columns: bool, add_timing_points: bool = True,
                 add_snapping: bool = True) -> list[GroupTok]:
    """Every group parse_beatmap would emit for a 4K mania chart.

    add_snapping defaults to True (the upstream training yaml); the published
    HF preprocessor config sets it to False, and the caller must pass that.
    """
    snap = (lambda t: snapping_at(osu, t)) if add_snapping else (lambda t: None)
    notes: list[GroupTok] = []
    for ho in osu.objects:
        col = mania_column(osu, ho.x)
        hs = 1 if hitsounds else 0
        if ho.kind == "hold":
            notes.append(GroupTok(ho.time_ms, "hold", snap(ho.time_ms),
                                  col if columns else None, hitsounds=hs))
            end = ho.end_ms if ho.end_ms is not None else ho.time_ms
            notes.append(GroupTok(end, "hold_end", snap(end),
                                  col if columns else None))
        else:
            # circle / slider / spinner / other: the parser makes one group at
            # the object time in mania for everything except holds; sliders and
            # spinners are not legal in mania and are reported separately.
            notes.append(GroupTok(ho.time_ms, "circle", snap(ho.time_ms),
                                  col if columns else None, hitsounds=hs))
    groups = list(notes)
    if sv:
        groups += sv_groups(osu, add_snapping=add_snapping)
    if kiai:
        groups += kiai_groups(osu, add_snapping=add_snapping)
    groups += timing_groups(osu, add_timing_points=add_timing_points)
    groups.sort(key=lambda g: (g.time, _KIND_RANK.get(g.kind, 3)))
    return groups


# ------------------------------------------------------------------- windows
def window_bounds(song_length_ms: float, window_sec: float = 16.0,
                  min_window_sec: float = 1.0) -> list[tuple[float, float]]:
    """processing_cm3p.__call__: np.arange(0, song_length - min, stride)."""
    starts = np.arange(0.0, song_length_ms / 1000.0 - min_window_sec, window_sec)
    return [(float(s) * 1000.0, float(s + window_sec) * 1000.0) for s in starts]


def window_token_lengths(groups: Sequence[GroupTok],
                         windows: Sequence[tuple[float, float]],
                         add_cls_token: bool = False) -> list[int]:
    """Token count per window, including CLS/BOS/EOS (3 if add_cls else 2)."""
    groups = sorted(groups, key=lambda g: g.time)
    specials = 3 if add_cls_token else 2
    out: list[int] = []
    idx = 0
    n = len(groups)
    for start, end in windows:
        while idx < n and groups[idx].time < start:
            idx += 1
        j = idx
        toks = specials
        while j < n and groups[j].time < end:
            toks += groups[j].n_tokens()
            j += 1
        idx = j
        out.append(toks)
    return out


def window_counts(groups: Sequence[GroupTok],
                  windows: Sequence[tuple[float, float]],
                  add_cls_token: bool = False) -> list[dict[str, int]]:
    """Per-window counts of core / hitsound / kiai / sv / column tokens."""
    groups = sorted(groups, key=lambda g: g.time)
    specials = 3 if add_cls_token else 2
    out: list[dict[str, int]] = []
    idx = 0
    n = len(groups)
    for start, end in windows:
        while idx < n and groups[idx].time < start:
            idx += 1
        j = idx
        c = {"tokens": specials, "core": 0, "columns": 0, "hitsound": 0,
             "kiai": 0, "sv": 0, "timing": 0, "notes": 0, "note_tokens": 0}
        while j < n and groups[j].time < end:
            g = groups[j]
            c["tokens"] += g.n_tokens()
            if g.kind in REMOVED_KINDS:
                c[g.kind] += 1
            else:
                c["core"] += 1
                if g.kind in ("beat", "measure", "timing_point"):
                    c["timing"] += 1
                if g.kind in ("circle", "hold", "hold_end"):
                    c["notes"] += 1
            if g.column is not None:
                c["columns"] += 1
            if g.hitsounds:
                c["hitsound"] += 2 * g.hitsounds
            j += 1
        idx = j
        out.append(c)
    return out


def truncation_stats(lengths: Sequence[int], max_length: int) -> dict[str, int]:
    n_trunc = sum(1 for x in lengths if x > max_length)
    return {"n_windows": len(lengths), "n_truncated": n_trunc,
            "n_tokens_full": int(sum(lengths)),
            "n_tokens_used": int(sum(min(x, max_length) for x in lengths)),
            "n_tokens_dropped": int(sum(max(0, x - max_length) for x in lengths))}


def hold_split_stats(osu: OsuMania, window_ms: float) -> tuple[int, int, int]:
    """(holds crossing a window boundary, ends within 2 ms of a boundary, total)."""
    split = near = total = 0
    for ho in osu.objects:
        if ho.kind != "hold" or ho.end_ms is None:
            continue
        total += 1
        start, end = ho.time_ms, ho.end_ms
        if window_ms > 0 and int(start // window_ms) != int(end // window_ms):
            split += 1
        if window_ms > 0:
            r = end % window_ms
            if min(r, window_ms - r) <= 2.0:
                near += 1
    return split, near, total


# --------------------------------------------------------------------- audit
@dataclass
class CropAudit:
    beatmap_id: int
    n_notes: int
    n_groups_full: int
    n_groups_crop: int
    n_core: int
    n_columns: int
    core_match: bool
    token_delta: int
    expected_hitsound_tokens: int
    removed_hitsound_tokens: int
    removed_kiai_tokens: int
    removed_sv_tokens: int
    windows_full: int
    windows_crop: int
    len_full: int
    len_crop: int
    n_truncated_full: int
    n_truncated_crop: int
    n_core_dropped_crop: int
    problems: list[str] = field(default_factory=list)


def audit_chart(osu: OsuMania, *, window_sec: float = 16.0,
                min_window_sec: float = 1.0, max_length: int = 4000,
                columns: bool = False, add_snapping: bool = True,
                add_cls_token: bool = False) -> CropAudit:
    """Prove that turning hitsounds/kiai/sv off removes no core information."""
    full = chart_groups(osu, hitsounds=True, kiai=True, sv=True, columns=columns,
                        add_snapping=add_snapping)
    crop = chart_groups(osu, hitsounds=False, kiai=False, sv=False, columns=columns,
                        add_snapping=add_snapping)
    core_full = [g.core() for g in full if g.kind not in REMOVED_KINDS]
    core_crop = [g.core() for g in crop]
    windows = window_bounds(song_length_ms(osu), window_sec, min_window_sec)
    specials = 3 if add_cls_token else 2
    lf = window_token_lengths(full, windows, add_cls_token=add_cls_token)
    lc = window_token_lengths(crop, windows, add_cls_token=add_cls_token)
    n_trunc_full = sum(1 for x in lf if x > max_length)
    n_trunc_crop = sum(1 for x in lc if x > max_length)

    # core tokens lost to truncation: walk the cropped stream in order and
    # count core groups whose event+time token pair is not fully inside.
    n_core_dropped = 0
    for start, end in windows:
        pos = 1 + (1 if add_cls_token else 0)   # CLS+BOS before the first group
        for g in crop:
            if not (start <= g.time < end):
                continue
            if pos + 2 > max_length:
                n_core_dropped += 1
            pos += g.n_tokens()
            if pos >= max_length:
                break

    # removed tokens must be exactly the feature payload
    removed = {}
    for g in full:
        if g.kind in REMOVED_KINDS:
            removed[g.kind] = removed.get(g.kind, 0) + g.n_tokens()
        elif g.hitsounds:
            removed["hitsound"] = removed.get("hitsound", 0) + 2 * g.hitsounds
    full_tokens = sum(g.n_tokens() for g in full) + specials * len(windows)
    crop_tokens = sum(g.n_tokens() for g in crop) + specials * len(windows)
    problems = []
    if core_full != core_crop:
        problems.append("core group stream changed by the crop")
    if len(windows) != len(lf) or len(lf) != len(lc):
        problems.append("window count mismatch")
    if full_tokens - crop_tokens != sum(removed.values()):
        problems.append("removed-token count != hitsound+kiai+sv payload")
    if n_core_dropped:
        problems.append(f"{n_core_dropped} core group(s) truncated by max_length")
    return CropAudit(
        beatmap_id=-1, n_notes=osu.n_notes, n_groups_full=len(full),
        n_groups_crop=len(crop), n_core=len(core_crop),
        n_columns=sum(1 for g in crop if g.column is not None),
        core_match=core_full == core_crop,
        token_delta=full_tokens - crop_tokens,
        expected_hitsound_tokens=removed.get("hitsound", 0),
        removed_hitsound_tokens=removed.get("hitsound", 0),
        removed_kiai_tokens=removed.get("kiai", 0),
        removed_sv_tokens=removed.get("sv", 0),
        windows_full=len(windows), windows_crop=len(lc),
        len_full=full_tokens, len_crop=crop_tokens,
        n_truncated_full=n_trunc_full, n_truncated_crop=n_trunc_crop,
        n_core_dropped_crop=n_core_dropped, problems=problems)


# ---------------------------------------------------------------- summary
def pct_summary(values: Iterable[float]) -> dict[str, float]:
    a = np.asarray(list(values), dtype=np.float64)
    if a.size == 0:
        return {}
    return {"n": int(a.size), "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "max": float(a.max()),
            "min": float(a.min())}
