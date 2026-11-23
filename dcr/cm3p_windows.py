"""CM3P input crop + per-window export (T0/T1/T2 of the v5 plan).

Three things live here, in the order a run uses them:

T0  read_processor_snapshot / assert_processor_config
    The real window length, stride, max_length and tokenizer max_time come from
    the HF preprocessor config, not from this repository.  Everything downstream
    (token budget, padding, truncation) depends on them, so they are printed and
    asserted before a single chart is parsed.  tokenizer.max_time must cover the
    whole window or [TIME_SHIFT_*] clips and the window loses its timing.

T1  CropSpec
    The three flags of the plan -- add_hitsounds=False, add_kiai=False,
    add_mania_sv=False -- plus the flags inherited from the checkpoint (timing
    grid, snapping, CLS).  The real HF preprocessor config is
    add_snapping=False, add_cls_token=True, max_length=2000 (not the upstream
    training yaml's True/False/4000), so the inherited flags are read, never
    forced: adding a token the checkpoint never saw is a distribution change.
    CropSpec.canon() is the plan's setting, CropSpec.legacy() reproduces the
    old table (the three payload flags on).

    The mania column deserves its own paragraph: the upstream parser stores it
    on Group.column but the tokenizer reads Group.mania_column, so no
    [MANIA_COLUMN_*] token is ever emitted (see dcr/cm3p_offline.py).  The
    optional install_mania_column_patch() fixes the attribute and the
    0-based/1-based off-by-one at runtime; it is OFF by default because the
    frozen checkpoint never saw a column token.  Every export records
    columns_emitted so this can never silently change meaning.

T2  tokenize_chart / audit_crop_pair / export_windows
    tokenize_chart runs the real processor without padding or truncation, so it
    sees the *full* window and can prove that nothing but the three feature
    payloads is removed.  export_windows then pads dynamically, batches windows
    by ceil(len/64)*64 bucket (BucketSampler), stores one float16 .npy per chart
    in time order, and builds .cache/cm3p/index.parquet with the columns the
    plan specifies plus the core-information audit they make possible.

The chart-level embedding is still mean over windows, re-normalised, so the
main content table keeps its contract.  The parity check reads the .npy back
and compares it with the chart vector written by the *same* run (cosine >= 1 -
1e-6).  It is not a legacy-table parity check: A0 is the T3 control, and all
T3 variants share the same canon crop.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------- constants
DEFAULT_WINDOW_SEC = 16.0
DEFAULT_MAX_LENGTH = 4000
DEFAULT_MIN_WINDOW_SEC = 1.0
EMBED_DIM = 512
BUCKET_STEP = 64
PAD_ID = 0
CORE_EVENT_TOKENS = ("[CIRCLE]", "[HOLD_NOTE]", "[HOLD_NOTE_END]")
MANIA_COLUMN_TOKENS = tuple(f"[MANIA_COLUMN_{i}]" for i in range(1, 19))
REMOVED_PREFIXES = ("[HITSOUND_", "[VOLUME_", "[KIAI_", "[SCROLL_SPEED_")


# --------------------------------------------------------------------- hashing
def cfg_hash(payload: dict) -> str:
    """Short stable hash of a JSON-able payload (sorted keys, no whitespace)."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------- crop
@dataclass(frozen=True)
class CropSpec:
    """T1's three decision flags; the rest are inherited from the checkpoint.

    The published HF checkpoint's processor config is *not* the upstream
    training yaml: it has add_snapping=False and max_length=2000.  (The HPC
    login node was only the machine with transformers installed when this was
    read; HPC copies are disposable, the checkpoint config is the fact.)
    Forcing add_snapping=True would change the model input beyond T1 and break
    parity with the frozen checkpoint, so the kept flags default to None =
    "leave whatever the checkpoint says".  read() turns them into the actual
    booleans for the cfg hash and the index.
    """
    add_hitsounds: bool = False
    add_kiai: bool = False
    add_mania_sv: bool = False
    add_timing: Optional[bool] = None
    add_timing_points: Optional[bool] = None
    add_snapping: Optional[bool] = None

    @classmethod
    def canon(cls) -> "CropSpec":
        return cls()

    @classmethod
    def legacy(cls) -> "CropSpec":
        return cls(add_hitsounds=True, add_kiai=True, add_mania_sv=True)

    @classmethod
    def from_name(cls, name: str) -> "CropSpec":
        if name == "canon":
            return cls.canon()
        if name == "legacy":
            return cls.legacy()
        raise ValueError(f"unknown crop spec {name!r} (canon/legacy)")

    @property
    def name(self) -> str:
        if self.add_hitsounds and self.add_kiai and self.add_mania_sv:
            return "legacy"
        if not (self.add_hitsounds or self.add_kiai or self.add_mania_sv):
            return "canon"
        return "mixed"

    def to_dict(self) -> dict:
        return {"add_hitsounds": self.add_hitsounds, "add_kiai": self.add_kiai,
                "add_mania_sv": self.add_mania_sv, "add_timing": self.add_timing,
                "add_timing_points": self.add_timing_points,
                "add_snapping": self.add_snapping}

    def apply(self, parser) -> "CropSpec":
        """Set the decision flags; None means "keep the checkpoint's value"."""
        for k, v in self.to_dict().items():
            if v is not None:
                setattr(parser, k, bool(v))
        return self

    def check(self) -> None:
        # These are inherited from the checkpoint, never forced: adding a flag
        # the model was not trained with is a distribution change, not a crop.
        if self.add_timing is False:
            print("[crop] NOTE: the checkpoint has add_timing=False; the beat/measure "
                  "grid is not in the model input. T1 does not force it on (OOD).",
                  flush=True)
        if self.add_snapping is False:
            print("[crop] NOTE: the checkpoint has add_snapping=False; the model input "
                  "has no [SNAPPING_*] tokens. T1 does not force them on (OOD).",
                  flush=True)
        if self.add_timing_points is False:
            print("[crop] NOTE: the checkpoint has add_timing_points=False; the first "
                  "beat of a red line is a [MEASURE], not a [TIMING_POINT].", flush=True)

    @classmethod
    def read(cls, parser) -> "CropSpec":
        return cls(**{k: bool(getattr(parser, k)) for k in cls.__dataclass_fields__})


# ------------------------------------------------------------------ T0 snapshot
@dataclass
class ProcessorSnapshot:
    window_length_sec: float
    window_stride_sec: float
    min_window_length_sec: float
    max_length: int
    padding: str
    truncation: str
    tokenizer_max_time: int
    tokenizer_time_step: int
    tokenizer_add_cls: bool
    parser: dict

    def to_dict(self) -> dict:
        return {
            "window_length_sec": self.window_length_sec,
            "window_stride_sec": self.window_stride_sec,
            "min_window_length_sec": self.min_window_length_sec,
            "max_length": self.max_length,
            "padding": self.padding,
            "truncation": self.truncation,
            "tokenizer_max_time": self.tokenizer_max_time,
            "tokenizer_time_step": self.tokenizer_time_step,
            "tokenizer_add_cls": self.tokenizer_add_cls,
            "parser": dict(self.parser),
        }


def read_processor_snapshot(processor) -> ProcessorSnapshot:
    """Everything about the processor that changes the token stream."""
    bk = processor.default_kwargs["beatmap_kwargs"]
    tok = processor.beatmap_tokenizer
    return ProcessorSnapshot(
        window_length_sec=float(bk["window_length_sec"]),
        window_stride_sec=float(bk["window_stride_sec"]),
        min_window_length_sec=float(bk.get("min_window_length_sec", DEFAULT_MIN_WINDOW_SEC)),
        max_length=int(bk["max_length"]),
        padding=str(bk.get("padding", "")),
        truncation=str(bk.get("truncation", "")),
        tokenizer_max_time=int(getattr(tok, "max_time", -1)),
        tokenizer_time_step=int(getattr(tok, "time_step", -1)),
        tokenizer_add_cls=bool(getattr(tok, "add_cls_token", False)),
        parser=CropSpec.read(processor.beatmap_parser).to_dict(),
    )


def assert_processor_config(snap: ProcessorSnapshot, *,
                            window_sec: float = DEFAULT_WINDOW_SEC,
                            max_length: Optional[int] = None,
                            strict: bool = True) -> list[str]:
    """T0: the window/max_length the budget assumes must be the real ones.

    Also checks that the tokenizer's [TIME_SHIFT] range covers a whole window
    (otherwise the last seconds of every window are clipped into one token and
    the window silently loses its timing) and that windows do not overlap.
    """
    problems: list[str] = []
    if abs(snap.window_length_sec - window_sec) > 1e-9:
        problems.append(f"window_length_sec={snap.window_length_sec:g}, budget assumes "
                        f"{window_sec:g}")
    if snap.window_stride_sec != snap.window_length_sec:
        problems.append(f"window_stride_sec={snap.window_stride_sec:g} != "
                        f"window_length_sec={snap.window_length_sec:g} (overlapping "
                        "windows would duplicate notes)")
    if max_length is not None and snap.max_length != int(max_length):
        problems.append(f"max_length={snap.max_length}, caller assumes {max_length}")
    elif max_length is None and snap.max_length != DEFAULT_MAX_LENGTH:
        print(f"[T0] NOTE: max_length={snap.max_length} in the checkpoint config "
              f"(the plan's section-2 table assumed {DEFAULT_MAX_LENGTH}); the real "
              "value is used for truncation and the budget must be recomputed.",
              flush=True)
    if snap.tokenizer_max_time < int(round(snap.window_length_sec * 1000)):
        problems.append(f"tokenizer.max_time={snap.tokenizer_max_time}ms < window "
                        f"{snap.window_length_sec:g}s: [TIME_SHIFT] clips and the "
                        "window loses its timing")
    if snap.min_window_length_sec > snap.window_length_sec:
        problems.append(f"min_window_length_sec={snap.min_window_length_sec:g} > "
                        f"window {snap.window_length_sec:g}")
    if "longest_first" not in str(snap.truncation).lower():
        problems.append(f"truncation={snap.truncation!r}; T2 truncates the tail "
                        "(longest_first) and cannot promise the same for another "
                        "strategy")
    if problems:
        msg = ("T0 FAILED -- the real preprocessor config is not the one the token "
               "budget was computed with:\n  - " + "\n  - ".join(problems) +
               "\nRecompute the budget table (docs: section 2) or pass "
               "--allow-config-mismatch to run with the real values.")
        if strict:
            raise SystemExit(msg)
        print("[T0] WARNING: " + msg, flush=True)
    return problems


# ------------------------------------------------------- mania column detection
def install_mania_column_patch(parser_cls=None) -> bool:
    """Fix upstream's Group.column/Group.mania_column mismatch at runtime.

    The parser writes a 0-based column to Group.column, the tokenizer reads the
    never-set Group.mania_column and therefore emits nothing.  The patch sets
    mania_column to the 1-based column *after* the original _add_group ran, so
    the tokenizer's np.clip(x, 1, 18) is a no-op instead of collapsing column 0
    and 1 into the same token.

    This is opt-in: the published checkpoint never saw a column token, so
    feeding it columns is an out-of-distribution input.  See the docs.
    """
    if parser_cls is None:
        try:
            from cm3p.parsing_cm3p import CM3PBeatmapParser as parser_cls
        except Exception as exc:                       # pragma: no cover
            raise SystemExit(f"cannot import CM3PBeatmapParser: {exc}")
    if getattr(parser_cls, "_mania_column_patch", False):
        return True
    orig = parser_cls._add_group

    def _add_group(self, event_type, groups, time, *args, **kwargs):
        beatmap = kwargs.get("beatmap")
        pos = kwargs.get("pos")
        n0 = len(groups)
        out = orig(self, event_type, groups, time, *args, **kwargs)
        if (beatmap is not None and pos is not None
                and int(getattr(beatmap, "mode", -1)) == 3
                and n0 < len(groups)):
            g = groups[n0]
            if g.mania_column is None:
                g.mania_column = int(
                    self._get_mania_column(pos, int(beatmap.circle_size))) + 1
        return out

    parser_cls._add_group = _add_group
    parser_cls._mania_column_patch = True
    return True


def mania_column_patch_installed(parser_cls) -> bool:
    return bool(getattr(parser_cls, "_mania_column_patch", False))


# ------------------------------------------------------------------ token ids
@dataclass
class TokenIds:
    core: np.ndarray
    columns: np.ndarray

    def n_core(self, ids: np.ndarray) -> int:
        return int(np.isin(ids, self.core).sum())

    def n_columns(self, ids: np.ndarray) -> int:
        return int(np.isin(ids, self.columns).sum())


def token_ids(processor) -> TokenIds:
    tok = processor.beatmap_tokenizer
    core = np.asarray([int(tok.convert_tokens_to_ids(t)) for t in CORE_EVENT_TOKENS],
                      dtype=np.int64)
    cols = np.asarray([int(tok.convert_tokens_to_ids(t)) for t in MANIA_COLUMN_TOKENS],
                      dtype=np.int64)
    unk = getattr(tok, "unk_token_id", None)
    if unk is not None and (core == unk).any():
        raise SystemExit("a core event token is not in the vocabulary")
    return TokenIds(core=core, columns=cols)


# --------------------------------------------------------------- chart tokens
@dataclass
class ChartTokens:
    beatmap_id: int
    path: Path
    md5: str
    n_windows: int
    ids: list[np.ndarray]             # truncated to max_length, model input
    tok_len: list[int]
    tok_len_full: list[int]
    window_starts_ms: list[float]
    n_notes: int
    n_note_tokens: int
    n_column_tokens: int
    n_holds_split: int
    n_holds_near: int
    n_holds: int
    n_truncated_windows: int
    n_tokens_dropped: int
    n_core_dropped: int
    song_length_s: float
    tokens: Optional[list[list[str]]] = None    # untruncated, only for audits

    @property
    def columns_emitted(self) -> bool:
        return self.n_column_tokens > 0

    def index_row(self, cfg: str) -> dict:
        return {
            "beatmap_id": int(self.beatmap_id),
            "n_win": int(self.n_windows),
            "mean_norm": float("nan"),          # filled by the exporter
            "tok_len": [int(x) for x in self.tok_len],
            "tok_len_full": [int(x) for x in self.tok_len_full],
            "content_md5": self.md5,
            "cfg_hash": cfg,
            "n_holds_split": int(self.n_holds_split),
            "n_holds_boundary_2ms": int(self.n_holds_near),
            "n_holds": int(self.n_holds),
            "n_notes": int(self.n_notes),
            "n_note_tokens": int(self.n_note_tokens),
            "n_column_tokens": int(self.n_column_tokens),
            "columns_emitted": bool(self.columns_emitted),
            "n_truncated_windows": int(self.n_truncated_windows),
            "n_tokens_dropped": int(self.n_tokens_dropped),
            "n_core_tokens_dropped": int(self.n_core_dropped),
            "song_length_s": float(self.song_length_s),
            "window_starts_ms": [float(x) for x in self.window_starts_ms],
        }


def _split_holds(beatmap, window_ms: float) -> tuple[int, int, int]:
    """(crossing a boundary, end within 2 ms of a boundary, total holds)."""
    split = near = total = 0
    for ho in beatmap.hit_objects(stacking=False):
        end = getattr(ho, "end_time", None)
        if end is None:
            continue
        total += 1
        start_ms = int(ho.time.total_seconds() * 1000 + 1e-5)
        end_ms = int(end.total_seconds() * 1000 + 1e-5)
        if window_ms > 0 and int(start_ms // window_ms) != int(end_ms // window_ms):
            split += 1
        if window_ms > 0:
            r = end_ms % window_ms
            if min(r, window_ms - r) <= 2.0:
                near += 1
    return split, near, total


def build_chart_tokens(beatmap, raw_ids: Sequence[Sequence[int]],
                       starts_ms: Sequence[float], *, beatmap_id: int,
                       path: Path, md5: str, max_length: int,
                       window_length_ms: float, tids: TokenIds,
                       keep_tokens: bool = False) -> ChartTokens:
    """Pure bookkeeping on already-tokenised windows (testable without cm3p)."""
    ids_used: list[np.ndarray] = []
    lens: list[int] = []
    lens_full: list[int] = []
    n_note = n_col = n_trunc = n_drop = n_core_drop = 0
    toks: Optional[list[list[str]]] = [] if keep_tokens else None
    decode = None
    if keep_tokens:
        decode = getattr(beatmap, "_token_decode", None)
    for arr in raw_ids:
        full = np.asarray(arr, dtype=np.int32).reshape(-1)
        lens_full.append(int(full.size))
        if full.size > max_length:
            used = full[:max_length]
            dropped = full[max_length:]
            n_trunc += 1
            n_drop += int(dropped.size)
            n_core_drop += tids.n_core(dropped)
        else:
            used = full
        ids_used.append(used)
        lens.append(int(used.size))
        n_note += tids.n_core(used)
        n_col += tids.n_columns(used)
        if toks is not None:
            if decode is None:
                raise RuntimeError("keep_tokens=True needs beatmap._token_decode")
            toks.append(list(decode(full)))
    split, near, holds = _split_holds(beatmap, float(window_length_ms))
    return ChartTokens(
        beatmap_id=int(beatmap_id), path=Path(path), md5=md5,
        n_windows=len(ids_used), ids=ids_used, tok_len=lens,
        tok_len_full=lens_full, window_starts_ms=[float(x) for x in starts_ms],
        n_notes=int(len(beatmap.hit_objects(stacking=False))),
        n_note_tokens=int(n_note), n_column_tokens=int(n_col),
        n_holds_split=int(split), n_holds_near=int(near), n_holds=int(holds),
        n_truncated_windows=int(n_trunc), n_tokens_dropped=int(n_drop),
        n_core_dropped=int(n_core_drop),
        song_length_s=0.0, tokens=toks)


def tokenize_chart(processor, path: Path, *, window_length_sec: float,
                   window_stride_sec: float, min_window_length_sec: float,
                   max_length: int, tids: TokenIds, beatmap_id: int,
                   md5: str = "", keep_tokens: bool = False,
                   load_beatmap=None, get_song_length=None) -> ChartTokens:
    """Tokenise every window of one .osu with the real CM3P processor."""
    if load_beatmap is None or get_song_length is None:
        from cm3p.parsing_cm3p import load_beatmap as _lb, get_song_length as _gsl
        load_beatmap = load_beatmap or _lb
        get_song_length = get_song_length or _gsl
    b = load_beatmap(path)
    song_length = float(get_song_length(None, None, b))
    starts = np.arange(0.0, song_length - min_window_length_sec, window_stride_sec)
    # the processor's own windowing uses window_length_sec; with stride == window
    # the starts above are exactly the processor's batch_start_ms.
    enc = processor(beatmap=b, audio=None, speed=1.0,
                    padding="do_not_pad", truncation="do_not_truncate",
                    return_tensors=None)
    raw = enc["input_ids"] if "input_ids" in enc else []
    if len(raw) != len(starts):
        raise SystemExit(
            f"{path}: processor produced {len(raw)} windows but the window grid has "
            f"{len(starts)} (window={window_length_sec:g}s stride={window_stride_sec:g}s)")
    if keep_tokens:
        tok = processor.beatmap_tokenizer

        def _decode(arr):
            return tok.convert_ids_to_tokens([int(x) for x in arr])
        # attach a decoder without mutating the slider Beatmap class
        b = _WithDecode(b, _decode)
    ct = build_chart_tokens(
        b, raw, [float(s * 1000.0) for s in starts],
        beatmap_id=beatmap_id, path=Path(path), md5=md5, max_length=max_length,
        window_length_ms=window_length_sec * 1000.0, tids=tids,
        keep_tokens=keep_tokens)
    ct.song_length_s = song_length
    return ct


class _WithDecode:
    """Tiny proxy that adds a decoder to a slider Beatmap object."""
    def __init__(self, obj, decode):
        self._obj = obj
        self._token_decode = decode

    def __getattr__(self, name):
        return getattr(self._obj, name)


# ------------------------------------------------------------------- audit
@dataclass
class CropAuditResult:
    n_windows: int
    n_notes_full: int
    n_notes_crop: int
    n_cols_full: int
    n_cols_crop: int
    removed: dict
    problems: list[str]
    window_problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems and not self.window_problems


EVENT_TOKENS = frozenset({
    "[CIRCLE]", "[HOLD_NOTE]", "[HOLD_NOTE_END]", "[BEAT]", "[MEASURE]",
    "[TIMING_POINT]", "[KIAI_ON]", "[KIAI_OFF]", "[SCROLL_SPEED_CHANGE]",
    "[SPINNER]", "[SPINNER_END]", "[SLIDER_HEAD]", "[SLIDER_END]", "[REPEAT_END]",
    "[DRUMROLL]", "[DRUMROLL_END]", "[DENDEN]", "[DENDEN_END]",
    "[BEZIER_ANCHOR]", "[PERFECT_ANCHOR]", "[CATMULL_ANCHOR]", "[RED_ANCHOR]",
    "[LAST_ANCHOR]", "[NEW_COMBO]",
})
REMOVED_GROUP_EVENTS = frozenset({"[KIAI_ON]", "[KIAI_OFF]",
                                  "[SCROLL_SPEED_CHANGE]"})
SPECIAL_TOKENS = frozenset({"[CLS]", "[BOS]", "[EOS]"})


def _is_event_token(t: str) -> bool:
    return t in EVENT_TOKENS or t.endswith("_NEW_COMBO]")


def _group_kind(t: str) -> Optional[str]:
    if t.startswith("[KIAI_"):
        return "kiai"
    if t == "[SCROLL_SPEED_CHANGE]":
        return "sv"
    return None


def strip_removed(tokens: Sequence[str]) -> list[str]:
    """The token stream with the three crop payloads removed *as whole groups*.

    A kiai/SV group is [EVENT] [TIME_SHIFT] ([SNAPPING]) ([SCROLL_SPEED]), so
    dropping only the event token would leave its time shift behind and make the
    audit fail on a correct crop.  Groups are delimited by event tokens; note
    groups keep their time/snapping/column and lose only hitsound/volume.
    """
    out = []
    drop: Optional[str] = None
    for t in tokens:
        if t in SPECIAL_TOKENS:
            # [CLS]/[BOS]/[EOS] are window delimiters, not part of a kiai/SV
            # group.  A group may be the last thing before [EOS]; without this
            # the EOS would be swallowed and a correct crop would fail the audit.
            drop = None
            out.append(t)
            continue
        if _is_event_token(t):
            kind = _group_kind(t)
            if kind is not None:
                drop = kind
                continue
            drop = None
            out.append(t)
            continue
        if drop is not None:
            continue
        if t.startswith("[HITSOUND_") or t.startswith("[VOLUME_"):
            continue
        out.append(t)
    return out


def removed_counts(tokens: Sequence[str]) -> dict[str, int]:
    """Token counts for kiai/SV whole groups and hitsound/volume pairs."""
    out = {"hitsound": 0, "kiai": 0, "sv": 0}
    drop: Optional[str] = None
    for t in tokens:
        if t in SPECIAL_TOKENS:
            drop = None
            continue
        if _is_event_token(t):
            kind = _group_kind(t)
            if kind is not None:
                drop = kind
                out[kind] += 1
                continue
            drop = None
            continue
        if drop is not None:
            out[drop] += 1
            continue
        if t.startswith("[HITSOUND_") or t.startswith("[VOLUME_"):
            out["hitsound"] += 1
    return out


def audit_crop_pair(full: ChartTokens, crop: ChartTokens) -> CropAuditResult:
    """Prove that the crop only removes hitsound/volume, kiai and SV tokens.

    full is tokenised with CropSpec.legacy(), crop with CropSpec.canon(); both
    must keep every note and every timing group in the same order.  Comparing
    strip_removed(full_window) with crop_window is the strongest possible check:
    it catches a reordered token stream, a missing note and a wrong window
    assignment, not just a count mismatch.
    """
    problems: list[str] = []
    window_problems: list[str] = []
    if full.n_windows != crop.n_windows:
        problems.append(f"window count {full.n_windows} != {crop.n_windows}")
    if full.tokens is None or crop.tokens is None:
        problems.append("keep_tokens=False: token streams were not captured")
        return CropAuditResult(full.n_windows, full.n_note_tokens, crop.n_note_tokens,
                               full.n_column_tokens, crop.n_column_tokens, {}, problems,
                               window_problems)
    removed = {"hitsound": 0, "kiai": 0, "sv": 0}
    for i in range(min(full.n_windows, crop.n_windows)):
        a = full.tokens[i]
        b = crop.tokens[i]
        rc = removed_counts(a)
        for k, v in rc.items():
            removed[k] += v
        if strip_removed(a) != b:
            # report the first difference, with context
            sa = strip_removed(a)
            j = next((k for k in range(min(len(sa), len(b))) if sa[k] != b[k]),
                     min(len(sa), len(b)))
            window_problems.append(
                f"window {i}: stripped legacy != cropped at token {j}: "
                f"{sa[max(0, j - 2):j + 3]} vs {b[max(0, j - 2):j + 3]}")
    if full.n_note_tokens != crop.n_note_tokens:
        problems.append(f"note-event tokens {full.n_note_tokens} != {crop.n_note_tokens}")
    if full.n_column_tokens != crop.n_column_tokens:
        problems.append(f"column tokens {full.n_column_tokens} != {crop.n_column_tokens}")
    if full.n_holds_split != crop.n_holds_split:
        problems.append("hold split count changed")
    # Truncation is deliberately NOT a crop problem here: a very dense window can
    # exceed max_length on either side.  The exporter accounts for it per chart and
    # --on-truncate decides what to do, so a legitimately dense chart cannot block
    # the core audit.
    return CropAuditResult(full.n_windows, full.n_note_tokens, crop.n_note_tokens,
                           full.n_column_tokens, crop.n_column_tokens, removed,
                           problems, window_problems)


# ------------------------------------------------------------------ batching
@dataclass
class WindowItem:
    chart_key: int
    chart_seq: int
    win_idx: int
    ids: np.ndarray
    tok_len: int


def bucket_key(n: int, max_length: int) -> int:
    """ceil(len/64)*64, capped at max_length (longer windows were truncated)."""
    n = max(1, min(int(n), int(max_length)))
    return min(int(math.ceil(n / BUCKET_STEP) * BUCKET_STEP), int(max_length))


class BucketSampler:
    """Bucket windows by ceil(len/64)*64, shuffle inside the bucket.

    Same-chart windows keep their time order: items carry (chart_seq, win_idx)
    and every batch is re-sorted by that pair before it is stacked.  Batches are
    drawn from one bucket at a time, so the padded width is the bucket's width
    (or the batch max), never max_length.

    A buffer cap bounds memory for a 21,949-chart stream: when the buffer is
    full the largest bucket is flushed even if it is smaller than batch_size.
    """

    def __init__(self, batch_size: int = 16, buffer_windows: int = 4096,
                 max_length: int = DEFAULT_MAX_LENGTH, seed: int = 0):
        self.batch_size = max(1, int(batch_size))
        self.buffer_windows = max(self.batch_size, int(buffer_windows))
        self.max_length = int(max_length)
        self.rng = np.random.default_rng(seed)
        self.buckets: dict[int, list[WindowItem]] = {}
        self.n_buffered = 0
        self.n_added = 0
        self.n_batches = 0
        self.pad_num = 0.0
        self.pad_den = 0.0
        self._last_L = 0

    # ------------------------------------------------------------------ add
    def add(self, item: WindowItem) -> None:
        k = bucket_key(item.tok_len, self.max_length)
        self.buckets.setdefault(k, []).append(item)
        self.n_buffered += 1
        self.n_added += 1

    # ---------------------------------------------------------------- take
    def _take(self, key: int, n: int) -> list[WindowItem]:
        items = self.buckets[key]
        # randomise *which* windows enter the batch, then _pack restores the
        # within-chart time order.  (Shuffling after the selection would be a
        # no-op because _pack sorts again.)
        self.rng.shuffle(items)
        batch = items[:n]
        rest = items[n:]
        if rest:
            self.buckets[key] = rest
        else:
            del self.buckets[key]
        self.n_buffered -= len(batch)
        return batch

    def _pack(self, batch: list[WindowItem]) -> tuple[list[WindowItem], int]:
        batch.sort(key=lambda it: (it.chart_seq, it.win_idx))
        L = max(it.tok_len for it in batch)
        self.pad_num += sum(it.tok_len for it in batch)
        self.pad_den += len(batch) * L
        self.n_batches += 1
        self._last_L = L
        return batch, L

    def _largest_key(self) -> int:
        return max(self.buckets, key=lambda k: (len(self.buckets[k]), k))

    # --------------------------------------------------------------- public
    def pop_ready(self) -> Iterator[tuple[list[WindowItem], int]]:
        """Yield full batches while a bucket can fill one, or flush the largest
        bucket when the buffer cap is reached."""
        while self.buckets:
            full = [k for k, v in self.buckets.items() if len(v) >= self.batch_size]
            if full:
                # bucket order is random (seeded); all windows are still encoded once
                k = full[int(self.rng.integers(len(full)))]
                yield self._pack(self._take(k, self.batch_size))
                continue
            if self.n_buffered >= self.buffer_windows:
                k = self._largest_key()
                yield self._pack(self._take(k, len(self.buckets[k])))
                continue
            return

    def flush(self) -> Iterator[tuple[list[WindowItem], int]]:
        """Drain every remaining bucket, largest first."""
        for k in sorted(self.buckets.keys(), reverse=True):
            while self.buckets.get(k):
                yield self._pack(self._take(k, min(self.batch_size,
                                                   len(self.buckets[k]))))

    @property
    def pad_efficiency(self) -> float:
        return float(self.pad_num / self.pad_den) if self.pad_den else 0.0


def stack_batch(batch: Sequence[WindowItem], L: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.full((len(batch), L), PAD_ID, dtype=np.int64)
    mask = np.zeros((len(batch), L), dtype=np.int64)
    for i, it in enumerate(batch):
        n = min(it.tok_len, L)
        ids[i, :n] = it.ids[:n]
        mask[i, :n] = 1
    return ids, mask


# ------------------------------------------------------------------- export
@dataclass
class ExportConfig:
    win_dir: Path
    index_path: Path
    max_length: int = DEFAULT_MAX_LENGTH
    batch_size: int = 16
    buffer_windows: int = 4096
    seed: int = 0
    dtype: str = "float16"
    cfg_hash: str = ""
    embed_dim: int = EMBED_DIM
    verify_first: int = 64
    verify_all: bool = False
    on_truncate: str = "warn"          # warn / error / allow
    index_every: int = 200
    min_cosine: float = 1.0 - 1e-6


@dataclass
class ExportResult:
    n_charts: int
    n_windows: int
    n_batches: int
    pad_efficiency: float
    columns_emitted: bool
    n_truncated_windows: int
    n_core_tokens_dropped: int
    n_holds_split: int
    n_holds_boundary_2ms: int
    n_notes: int
    n_note_tokens: int
    n_column_tokens: int
    min_parity_cosine: float
    seconds: float
    skipped: list[tuple[int, str]] = field(default_factory=list)
    truncated_charts: list[int] = field(default_factory=list)
    per_chart: dict[int, np.ndarray] = field(default_factory=dict)   # unit, float32
    per_chart_norm: dict[int, float] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "n_charts": int(self.n_charts), "n_windows": int(self.n_windows),
            "n_batches": int(self.n_batches),
            "pad_efficiency": float(self.pad_efficiency),
            "columns_emitted": bool(self.columns_emitted),
            "n_truncated_windows": int(self.n_truncated_windows),
            "n_core_tokens_dropped": int(self.n_core_tokens_dropped),
            "n_holds_split": int(self.n_holds_split),
            "n_holds_boundary_2ms": int(self.n_holds_boundary_2ms),
            "n_notes": int(self.n_notes), "n_note_tokens": int(self.n_note_tokens),
            "n_column_tokens": int(self.n_column_tokens),
            "min_parity_cosine": float(self.min_parity_cosine),
            "seconds": round(float(self.seconds), 1),
            "charts_per_s": round(self.n_charts / max(self.seconds, 1e-9), 3),
            "skipped": len(self.skipped),
            "truncated_charts": len(self.truncated_charts),
        }


def _write_npy_atomic(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as fh:      # np.save(path) would append ".npy" to the name
        np.save(fh, arr)
    os.replace(tmp, path)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n) if n > 0 else v


class WindowExporter:
    """Encode charts window-by-window and write .npy + index rows.

    encode_fn(input_ids, attention_mask) -> (B, D) float array.  It is injected
    so the whole export path can be tested on CPU without transformers.
    """

    def __init__(self, config: ExportConfig,
                 encode_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 *, base_index: Optional[pd.DataFrame] = None,
                 log: Callable[[str], None] = print):
        self.cfg = config
        self.encode_fn = encode_fn
        self.log = log
        self.base_index = None
        if base_index is not None and not base_index.empty:
            self.base_index = base_index[base_index["cfg_hash"].astype(str)
                                        == str(config.cfg_hash)].copy()
        self.sampler = BucketSampler(config.batch_size, config.buffer_windows,
                                     config.max_length, config.seed)
        self.pending: dict[int, int] = {}
        self.vecs: dict[int, list[tuple[int, np.ndarray]]] = {}
        self.meta: dict[int, ChartTokens] = {}
        self.rows: dict[int, dict] = {}
        self.skipped: list[tuple[int, str]] = []
        self.truncated_ids: list[int] = []
        self.per_chart: dict[int, np.ndarray] = {}
        self.per_chart_norm: dict[int, float] = {}
        self._warned_truncation = False
        self.t0 = time.time()
        self.n_charts_done = 0
        self.n_windows_done = 0
        self.columns_emitted = False
        self.n_truncated = 0
        self.n_core_dropped = 0
        self.min_cos = 1.0
        self._seq = 0
        self._n_notes = self._n_note_tokens = self._n_columns = 0
        self._n_split = self._n_near = 0

    # ------------------------------------------------------------------ add
    def add_chart(self, ct: ChartTokens) -> None:
        if ct.n_windows == 0:
            self.skipped.append((ct.beatmap_id, "no windows"))
            return
        self.meta[ct.beatmap_id] = ct
        self.pending[ct.beatmap_id] = ct.n_windows
        self.vecs[ct.beatmap_id] = []
        self.columns_emitted = self.columns_emitted or ct.columns_emitted
        self.n_truncated += ct.n_truncated_windows
        self.n_core_dropped += ct.n_core_dropped
        self._n_notes += ct.n_notes
        self._n_note_tokens += ct.n_note_tokens
        self._n_columns += ct.n_column_tokens
        self._n_split += ct.n_holds_split
        self._n_near += ct.n_holds_near
        if ct.n_truncated_windows:
            self.truncated_ids.append(ct.beatmap_id)
        if ct.n_core_dropped and self.cfg.on_truncate == "error":
            raise SystemExit(
                f"chart {ct.beatmap_id}: {ct.n_core_dropped} core note tokens were "
                f"dropped by max_length={self.cfg.max_length}; raise --max-length or "
                "pass --on-truncate warn/allow")
        if ct.n_core_dropped and not self._warned_truncation:
            self._warned_truncation = True
            self.log(f"[win] WARNING: chart {ct.beatmap_id} exceeds max_length="
                     f"{self.cfg.max_length} and loses {ct.n_core_dropped} core note "
                     f"token(s); every affected chart is flagged in the index "
                     f"(n_core_tokens_dropped, n_truncated_windows)")
        for wi, arr in enumerate(ct.ids):
            self.sampler.add(WindowItem(ct.beatmap_id, self._seq, wi, arr, len(arr)))
        self._seq += 1
        self._drain()

    # --------------------------------------------------------------- batches
    def _drain(self, final: bool = False) -> None:
        it = self.sampler.flush() if final else self.sampler.pop_ready()
        for batch, L in it:
            self._run_batch(batch, L)
        if final:
            self._finalize_all()

    def _run_batch(self, batch: Sequence[WindowItem], L: int) -> None:
        ids, mask = stack_batch(batch, L)
        out = self.encode_fn(ids, mask)
        out = np.asarray(out, dtype=np.float32)
        if out.shape != (len(batch), self.cfg.embed_dim):
            raise SystemExit(f"encode_fn returned {out.shape}, want "
                             f"({len(batch)}, {self.cfg.embed_dim})")
        for it, vec in zip(batch, out):
            self.vecs[it.chart_key].append((it.win_idx, vec))
            self.pending[it.chart_key] -= 1
            self.n_windows_done += 1
            if self.pending[it.chart_key] == 0:
                self._finalize(it.chart_key)

    # -------------------------------------------------------------- finalize
    def _finalize(self, bid: int) -> None:
        ct = self.meta.pop(bid)
        vecs = sorted(self.vecs.pop(bid), key=lambda x: x[0])
        if len(vecs) != ct.n_windows:
            raise SystemExit(f"{bid}: got {len(vecs)} window vectors, want {ct.n_windows}")
        mat = np.stack([v for _, v in vecs]).astype(np.float32)
        mean = mat.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        unit = mean / norm if norm > 0 else mean
        store = mat.astype(np.dtype(self.cfg.dtype))
        out = Path(self.cfg.win_dir) / f"{bid}.npy"
        _write_npy_atomic(out, store)
        if self.cfg.verify_all or self.n_charts_done < self.cfg.verify_first:
            back = np.load(out).astype(np.float32)
            cos = float(np.dot(_unit(back.mean(axis=0)), _unit(mean)))
            self.min_cos = min(self.min_cos, cos)
            if cos < self.cfg.min_cosine:
                raise SystemExit(
                    f"{bid}: parity check failed: cosine(read-back mean, in-memory mean)="
                    f"{cos:.9f} < {self.cfg.min_cosine:.9f}; the window store is not "
                    "faithful (try --window-dtype float32)")
        row = ct.index_row(self.cfg.cfg_hash)
        row["mean_norm"] = norm
        self.rows[bid] = row
        self.per_chart[int(bid)] = unit.astype(np.float32)
        self.per_chart_norm[int(bid)] = norm
        self.n_charts_done += 1
        if self.n_charts_done % 50 == 0:
            el = time.time() - self.t0
            self.log(f"[win] {self.n_charts_done:,} charts, {self.n_windows_done:,} windows "
                     f"({el:.0f}s, {self.n_charts_done / el:.2f} charts/s, "
                     f"pad_eff={self.sampler.pad_efficiency:.3f})")
        if self.cfg.index_every and self.n_charts_done % self.cfg.index_every == 0:
            self.write_index()

    def _finalize_all(self) -> None:
        for bid in list(self.pending):
            if self.pending[bid] == 0:
                del self.pending[bid]           # already finalised when it hit 0
                continue
            raise SystemExit(f"{bid}: {self.pending[bid]} windows never encoded")
        self.write_index()

    # ----------------------------------------------------------------- index
    def write_index(self) -> None:
        if not self.rows:
            return
        df = pd.DataFrame(list(self.rows.values()))
        if self.base_index is not None and not self.base_index.empty:
            keep = self.base_index[~self.base_index["beatmap_id"].isin(df["beatmap_id"])]
            if not keep.empty:
                df = pd.concat([keep, df], ignore_index=True)
        path = Path(self.cfg.index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        df.sort_values("beatmap_id").reset_index(drop=True).to_parquet(tmp, index=False)
        os.replace(tmp, path)

    def finish(self) -> ExportResult:
        self._drain(final=True)
        el = time.time() - self.t0
        return ExportResult(
            n_charts=self.n_charts_done, n_windows=self.n_windows_done,
            n_batches=self.sampler.n_batches,
            pad_efficiency=self.sampler.pad_efficiency,
            columns_emitted=self.columns_emitted,
            n_truncated_windows=self.n_truncated,
            n_core_tokens_dropped=self.n_core_dropped,
            n_holds_split=self._n_split, n_holds_boundary_2ms=self._n_near,
            n_notes=self._n_notes, n_note_tokens=self._n_note_tokens,
            n_column_tokens=self._n_columns,
            min_parity_cosine=self.min_cos, seconds=el,
            skipped=self.skipped, truncated_charts=list(self.truncated_ids),
            per_chart=self.per_chart, per_chart_norm=self.per_chart_norm)


# ------------------------------------------------------------- index helpers
def load_window_index(path: Path) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def index_cfg_hashes(df: Optional[pd.DataFrame]) -> set[str]:
    if df is None or df.empty or "cfg_hash" not in df:
        return set()
    return set(df["cfg_hash"].astype(str).unique())


def index_ready(df: Optional[pd.DataFrame], cfg: str, md5: dict[int, str],
                win_dir: Path) -> set[int]:
    """beatmaps already exported under this cfg with the same source md5."""
    if df is None or df.empty:
        return set()
    d = df[df["cfg_hash"].astype(str) == cfg]
    out = set()
    for bid, got in zip(d["beatmap_id"].astype(int), d["content_md5"].astype(str)):
        if md5.get(int(bid)) == got and (Path(win_dir) / f"{int(bid)}.npy").exists():
            out.add(int(bid))
    return out


def write_meta(path: Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
