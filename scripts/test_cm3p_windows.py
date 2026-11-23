#!/usr/bin/env python
"""Tests for the CM3P crop/window-export core (T0/T1/T2).

Run:  python scripts/test_cm3p_windows.py

No transformers, no GPU, no CM3P checkout: the processor and the model are
fakes, and the tokenizer audit is driven by synthetic token ids.  The goal is
to pin the invariants the real run depends on:

  * the legacy->crop token diff removes ONLY hitsound/volume, kiai and SV,
    and leaves the note/timing stream identical (the core-information gate);
  * truncation is counted and never silently loses a note;
  * BucketSampler covers every window exactly once, never mixes far-apart
    lengths into one batch, and keeps a chart's windows in time order;
  * the .npy store round-trips to the chart mean (cosine >= 1 - 1e-6);
  * the index is resumable and merges with rows from an earlier run.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import types
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcr import cm3p_offline as off  # noqa: E402
from dcr import cm3p_windows as w  # noqa: E402

FAILS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""),
          flush=True)
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------------------- fakes
class FakeHold:
    def __init__(self, start_ms: int, end_ms: int):
        self.time = timedelta(milliseconds=start_ms)
        self.end_time = timedelta(milliseconds=end_ms)


class FakeBM:
    def __init__(self, holds=()):
        self._holds = list(holds)

    def hit_objects(self, stacking=False):
        return list(self._holds)


class FakeTok:
    def __init__(self):
        self.vocab = {"[PAD]": 0, "[CIRCLE]": 1, "[HOLD_NOTE]": 2,
                      "[HOLD_NOTE_END]": 3,
                      "[KIAI_ON]": 30, "[HITSOUND_0_2_1]": 40, "[VOLUME_50]": 41,
                      "[SCROLL_SPEED_1.00]": 50, "[SCROLL_SPEED_CHANGE]": 60,
                      "[TIME_SHIFT_0]": 61}
        for i in range(1, 19):
            self.vocab[f"[MANIA_COLUMN_{i}]"] = 9 + i
        self.id2 = {v: k for k, v in self.vocab.items()}
        self.unk_token_id = 999
        self.max_time = 16000
        self.time_step = 10
        self.add_cls_token = False

    def convert_tokens_to_ids(self, t):
        if isinstance(t, str):
            return self.vocab.get(t, self.unk_token_id)
        return [self.vocab.get(x, self.unk_token_id) for x in t]

    def convert_ids_to_tokens(self, ids):
        return [self.id2.get(int(i), "[UNK]") for i in ids]


class FakeProcessor:
    def __init__(self, raw, tok):
        self.raw = raw
        self.beatmap_tokenizer = tok

    def __call__(self, *, beatmap, audio=None, speed=1.0, padding=None,
                 truncation=None, return_tensors=None):
        return {"input_ids": self.raw}


TIDS = w.TokenIds(core=np.array([1, 2, 3]), columns=np.array([10, 11, 12, 13]))


def make_ct(bid, raw, keep_tokens=False, max_length=4000, holds=(), starts=None,
            window_ms=16000.0):
    bm = FakeBM(holds)
    if keep_tokens:
        tok = FakeTok()

        def dec(arr, _tok=tok):
            return _tok.convert_ids_to_tokens(arr)
        bm._token_decode = dec
    if starts is None:
        starts = [float(i * 16000) for i in range(len(raw))]
    return w.build_chart_tokens(bm, raw, starts, beatmap_id=bid,
                                path=Path(f"{bid}.osu"), md5=f"md5-{bid}",
                                max_length=max_length, window_length_ms=window_ms,
                                tids=TIDS, keep_tokens=keep_tokens)


# ------------------------------------------------------------------- tests
def test_crop_spec() -> None:
    print("\nCropSpec / cfg_hash")
    c = w.CropSpec.canon()
    leg = w.CropSpec.legacy()
    check("canon drops the three payloads", not (c.add_hitsounds or c.add_kiai
                                                 or c.add_mania_sv))
    check("canon inherits the kept flags", c.add_timing is None
          and c.add_snapping is None and c.add_timing_points is None)
    check("legacy turns all three on", leg.add_hitsounds and leg.add_kiai
          and leg.add_mania_sv)
    check("names", c.name == "canon" and leg.name == "legacy")
    parser = types.SimpleNamespace(add_hitsounds=True, add_kiai=True,
                                   add_mania_sv=True, add_timing=True,
                                   add_timing_points=True, add_snapping=False)
    c.apply(parser)
    check("apply does not force add_snapping on", parser.add_snapping is False
          and parser.add_hitsounds is False)
    r = w.CropSpec.read(parser)
    check("read captures the actual kept flags", r.add_snapping is False
          and r.add_hitsounds is False and r.to_dict()["add_timing"] is True)
    check("cfg_hash is stable and sensitive",
          w.cfg_hash({"a": 1}) == w.cfg_hash({"a": 1}) != w.cfg_hash({"a": 2}))


def test_t0() -> None:
    print("\nT0 config assertions")
    snap = w.ProcessorSnapshot(16.0, 16.0, 1.0, 4000, "longest", "longest_first",
                               16000, 10, False, {"add_timing": True})
    check("16/4000 accepted", w.assert_processor_config(snap, strict=True) == [])
    bad = w.ProcessorSnapshot(30.0, 30.0, 1.0, 8000, "longest", "longest_first",
                              16000, 10, False, {})
    problems = w.assert_processor_config(bad, max_length=4000, strict=False)
    check("30s/8000 reported", any("window_length_sec" in p for p in problems)
          and any("max_length" in p for p in problems))
    clipped = w.ProcessorSnapshot(16.0, 16.0, 1.0, 4000, "longest", "longest_first",
                                  8000, 10, False, {})
    problems = w.assert_processor_config(clipped, strict=False)
    check("tokenizer.max_time < window is caught",
          any("max_time" in p for p in problems))
    try:
        w.assert_processor_config(bad, strict=True)
        check("strict T0 raises", False)
    except SystemExit:
        check("strict T0 raises", True)


def test_buckets() -> None:
    print("\nBucketSampler")
    rng = np.random.default_rng(0)
    items = []
    for i in range(200):
        n = int(rng.integers(2, 4000))
        items.append(w.WindowItem(chart_key=i // 3, chart_seq=i // 3,
                                  win_idx=i % 3, ids=np.zeros(n, np.int32), tok_len=n))
    s = w.BucketSampler(batch_size=8, buffer_windows=64, max_length=4000, seed=0)
    for it in items:
        s.add(it)
    batches = list(s.pop_ready()) + list(s.flush())
    seen = [(it.chart_seq, it.win_idx) for b, _ in batches for it in b]
    check("every window exactly once", sorted(seen) == sorted(
        [(it.chart_seq, it.win_idx) for it in items]))
    check("batch size respected", all(len(b) <= 8 for b, _ in batches))
    check("chart order preserved", all(
        all(b[j].chart_seq < b[j + 1].chart_seq
            or (b[j].chart_seq == b[j + 1].chart_seq and b[j].win_idx < b[j + 1].win_idx)
            for j in range(len(b) - 1)) for b, _ in batches))
    # no batch mixes a tiny window with a near-4000 one by accident
    spread = max((L - min(it.tok_len for it in b) for b, L in batches), default=0)
    check("batches stay inside their bucket", spread < 128 + 64,
          f"max (padded - shortest) = {spread}")
    check("pad_efficiency is sane", 0.2 < s.pad_efficiency <= 1.0,
          f"{s.pad_efficiency:.3f}")
    check("buckets empty at the end", not s.buckets and s.n_buffered == 0)


def test_build_and_audit() -> None:
    print("\nbuild_chart_tokens / audit_crop_pair")
    bm = FakeBM([FakeHold(15800, 16000), FakeHold(1000, 2000)])
    raw = [np.array([1, 50, 1, 50], np.int32), np.array([1, 50], np.int32)]
    ct = make_ct(1, raw, holds=bm.hit_objects())
    check("windows and lengths", ct.n_windows == 2 and ct.tok_len == [4, 2])
    check("note tokens counted", ct.n_note_tokens == 3, str(ct.n_note_tokens))
    check("no column tokens when columns are off", ct.n_column_tokens == 0)
    check("hold split counted", ct.n_holds_split == 1 and ct.n_holds == 2,
          f"split={ct.n_holds_split}")
    check("hold end near a boundary counted", ct.n_holds_near == 1)

    # truncation: 6 tokens into max_length 4 loses 2 tail tokens
    ct2 = make_ct(2, [np.array([1, 2, 3, 1, 2, 3], np.int32)], max_length=4)
    check("truncation counted", ct2.n_truncated_windows == 1
          and ct2.tok_len == [4] and ct2.tok_len_full == [6]
          and ct2.n_tokens_dropped == 2 and ct2.n_core_dropped == 2)

    # token-level audit: full = crop + payload in exactly the right places
    tok = FakeTok()
    circle = tok.vocab["[CIRCLE]"]
    hs = tok.vocab["[HITSOUND_0_2_1]"]
    vol = tok.vocab["[VOLUME_50]"]
    col = tok.vocab["[MANIA_COLUMN_2]"]
    full = [circle, hs, vol, hs, vol]
    crop = [circle]
    f = make_ct(3, [np.array(full, np.int32)], keep_tokens=True)
    c = make_ct(4, [np.array(crop, np.int32)], keep_tokens=True)
    a = w.audit_crop_pair(f, c)
    check("legacy -> crop audit passes", a.ok, str(a.problems + a.window_problems))
    check("removed payload counted", a.removed["hitsound"] == 4)
    check("column tokens equal (0)", a.n_cols_full == a.n_cols_crop == 0)

    # a dropped note must fail the audit
    lost = make_ct(5, [np.array([], np.int32)], keep_tokens=True)
    a = w.audit_crop_pair(f, lost)
    check("lost note is caught", not a.ok)

    # kiai/SV groups must be removed with their TIME_SHIFT (not just the event)
    a = ["[CLS]", "[BOS]", "[CIRCLE]", "[TIME_SHIFT_0]", "[HITSOUND_0_2_1]",
         "[VOLUME_50]", "[KIAI_ON]", "[TIME_SHIFT_100]", "[SNAPPING_4]",
         "[SCROLL_SPEED_CHANGE]", "[TIME_SHIFT_200]", "[SNAPPING_2]",
         "[SCROLL_SPEED_1.00]", "[BEAT]", "[TIME_SHIFT_300]", "[EOS]"]
    b = ["[CLS]", "[BOS]", "[CIRCLE]", "[TIME_SHIFT_0]", "[BEAT]",
         "[TIME_SHIFT_300]", "[EOS]"]
    check("kiai/SV groups removed whole", w.strip_removed(a) == b,
          str(w.strip_removed(a)))
    rc = w.removed_counts(a)
    check("removed group counts", rc["kiai"] == 3 and rc["sv"] == 4
          and rc["hitsound"] == 2, str(rc))

    # a kiai/SV group immediately before [EOS] must not swallow the EOS
    a_eos = ["[BOS]", "[CIRCLE]", "[TIME_SHIFT_0]", "[KIAI_ON]",
             "[TIME_SHIFT_100]", "[EOS]"]
    b_eos = ["[BOS]", "[CIRCLE]", "[TIME_SHIFT_0]", "[EOS]"]
    check("kiai before EOS keeps EOS", w.strip_removed(a_eos) == b_eos,
          str(w.strip_removed(a_eos)))
    rc_eos = w.removed_counts(a_eos)
    check("EOS not counted as kiai payload", rc_eos["kiai"] == 2, str(rc_eos))

    # columns on both sides
    f2 = make_ct(6, [np.array([circle, col, hs, vol], np.int32)], keep_tokens=True)
    c2 = make_ct(7, [np.array([circle, col], np.int32)], keep_tokens=True)
    a = w.audit_crop_pair(f2, c2)
    check("column token preserved by the crop", a.ok and a.n_cols_crop == 1)


def test_exporter() -> None:
    print("\nWindowExporter / index / resume")
    rng = np.random.default_rng(0)
    charts = []
    for i in range(9):
        raw = [np.full(int(rng.integers(50, 400)), 1, np.int32) for _ in range(3)]
        charts.append(make_ct(100 + i, raw))

    def encode(ids, mask):
        out = []
        for r, m in zip(ids, mask):
            v = r[m > 0].astype(np.int64)
            e = np.bincount(v % 512, minlength=512).astype(np.float64)
            out.append(e / (np.linalg.norm(e) + 1e-12))
        return np.stack(out)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg = w.ExportConfig(win_dir=td / "win", index_path=td / "index.parquet",
                             batch_size=4, buffer_windows=8, cfg_hash="cfg1",
                             verify_first=64)
        ex = w.WindowExporter(cfg, encode, log=lambda s: None)
        for c in charts:
            ex.add_chart(c)
        res = ex.finish()
        check("all charts exported", res.n_charts == 9,
              f"{res.n_charts}")
        check("per-chart unit vectors", len(res.per_chart) == 9 and all(
            abs(np.linalg.norm(v) - 1) < 1e-6 for v in res.per_chart.values()))
        check("parity cosine", res.min_parity_cosine >= 1 - 1e-6,
              f"{res.min_parity_cosine:.9f}")
        check("npy count", len(list((td / "win").glob("*.npy"))) == 9)
        df = pd.read_parquet(td / "index.parquet")
        check("index rows", len(df) == 9 and set(df.beatmap_id) == set(range(100, 109)))
        check("index cfg", set(df.cfg_hash) == {"cfg1"})
        check("index tok_len length", all(len(x) == 3 for x in df.tok_len))
        check("mean_norm recorded", bool(np.isfinite(df.mean_norm).all()))
        ready = w.index_ready(df, "cfg1", {b: f"md5-{b}" for b in range(100, 109)},
                              td / "win")
        check("resume set", ready == set(range(100, 109)))
        ready2 = w.index_ready(df, "cfg2", {b: f"md5-{b}" for b in range(100, 109)},
                               td / "win")
        check("different cfg is not resumable", ready2 == set())

        # base_index merge: an old row for a chart that is not in this run
        base = df.copy()
        extra = base.iloc[[0]].copy()
        extra["beatmap_id"] = 999
        base = pd.concat([base, extra], ignore_index=True)
        ex2 = w.WindowExporter(cfg, encode, base_index=base, log=lambda s: None)
        raw = [np.full(60, 1, np.int32)]
        ex2.add_chart(make_ct(200, raw))
        ex2.finish()
        merged = pd.read_parquet(td / "index.parquet")
        check("base index rows kept", 999 in set(merged.beatmap_id))
        check("new row written", 200 in set(merged.beatmap_id))


def test_tokenize_chart() -> None:
    print("\ntokenize_chart (fake processor)")
    tok = FakeTok()
    proc = FakeProcessor([np.array([1, 10], np.int32), np.array([1], np.int32)], tok)
    bm = FakeBM()
    ct = w.tokenize_chart(
        proc, "fake.osu", window_length_sec=16.0, window_stride_sec=16.0,
        min_window_length_sec=1.0, max_length=4000, tids=TIDS, beatmap_id=7,
        md5="m", load_beatmap=lambda p: bm,
        get_song_length=lambda samples, sr, b: 33.0)
    check("window grid from song length", ct.n_windows == 2, str(ct.n_windows))
    check("ids/truncation", ct.tok_len == [2, 1] and ct.tok_len_full == [2, 1])
    check("columns counted", ct.n_column_tokens == 1)
    check("song length recorded", abs(ct.song_length_s - 33.0) < 1e-9)

    # a window-grid mismatch must be loud
    proc2 = FakeProcessor([np.array([1], np.int32)], tok)
    try:
        w.tokenize_chart(proc2, "fake.osu", window_length_sec=16.0,
                         window_stride_sec=16.0, min_window_length_sec=1.0,
                         max_length=4000, tids=TIDS, beatmap_id=7,
                         load_beatmap=lambda p: bm,
                         get_song_length=lambda samples, sr, b: 33.0)
        check("window mismatch raises", False)
    except SystemExit:
        check("window mismatch raises", True)


def test_mania_column_patch() -> None:
    print("\nmania column patch")
    class G:
        def __init__(self, event_type):
            self.event_type = event_type
            self.mania_column = None
            self.column = None

    class FakeParser:
        _mania_column_patch = False

        def _get_mania_column(self, pos, columns):
            return int(np.clip(pos[0] / 512 * columns, 0, columns - 1))

        def _add_group(self, event_type, groups, time, *, beatmap=None, pos=None,
                       **kwargs):
            groups.append(G(event_type))
            if pos is not None and getattr(beatmap, "mode", 0) == 3:
                groups[-1].column = self._get_mania_column(pos, int(beatmap.circle_size))

    class BM:
        mode = 3
        circle_size = 4

    w.install_mania_column_patch(FakeParser)
    p = FakeParser()
    groups = []
    p._add_group("circle", groups, 0, beatmap=BM(), pos=np.array([0, 192]))
    p._add_group("circle", groups, 0, beatmap=BM(), pos=np.array([256, 192]))
    p._add_group("circle", groups, 0, beatmap=BM(), pos=np.array([511, 192]))
    check("patch sets 1-based columns", [g.mania_column for g in groups] == [1, 3, 4],
          str([g.mania_column for g in groups]))
    check("original column still there", [g.column for g in groups] == [0, 2, 3])


def test_offline_audit() -> None:
    print("\noffline .osu accounting")
    text = """osu file format v14

[General]
Mode: 3

[Difficulty]
CircleSize: 4

[TimingPoints]
0,500,4,2,0,60,1,0
8000,-50,4,2,0,60,0,1

[HitObjects]
64,192,1000,1,0,0:0:0:0:
192,192,1500,1,0,0:0:0:0:
320,192,2000,128,0,17000:0:0:0:0:
448,192,17000,1,0,0:0:0:0:
"""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "1.osu"
        p.write_text(text)
        osu = off.parse_osu(p)
        check("parse notes", osu.n_notes == 4, str(osu.n_notes))
        check("parse timing", len(osu.timing) == 2)
        full = off.chart_groups(osu, hitsounds=True, kiai=True, sv=True, columns=True)
        crop = off.chart_groups(osu, hitsounds=False, kiai=False, sv=False, columns=True)
        core_full = [g.core() for g in full if g.kind not in off.REMOVED_KINDS]
        core_crop = [g.core() for g in crop]
        check("core stream identical", core_full == core_crop)
        check("columns present when asked", all(
            g.column is not None for g in crop
            if g.kind in ("circle", "hold", "hold_end")))
        a = off.audit_chart(osu, max_length=4000, columns=True)
        check("audit clean", a.core_match and not a.problems, str(a.problems))
        check("removed payload non-zero", a.token_delta > 0 and a.removed_kiai_tokens > 0)
        check("hold split", off.hold_split_stats(osu, 16000.0)[0] == 1)
        a2 = off.audit_chart(osu, max_length=8, columns=True)
        check("truncation flagged", a2.n_truncated_crop > 0
              and any("truncated" in x for x in a2.problems), str(a2.problems))


def _load_step0c():
    spec = importlib.util.spec_from_file_location(
        "step0c_test", ROOT / "scripts/step0c_chart_content.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_step0c_glue() -> None:
    print("\nstep0c sidecar guards / window-table glue")
    mod = _load_step0c()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        out = td / "table.parquet"
        out.write_bytes(b"x")                       # exists, but no sidecar
        args = types.SimpleNamespace(crop="canon", mania_columns="auto",
                                     refresh=False, allow_mixed_crop=False)
        try:
            mod._precheck_sidecar(out, args, False)
            check("canon into a sidecar-less table is refused", False)
        except SystemExit:
            check("canon into a sidecar-less table is refused", True)
        args_legacy = types.SimpleNamespace(crop="legacy", mania_columns="auto",
                                            refresh=False, allow_mixed_crop=False)
        mod._precheck_sidecar(out, args_legacy, False)
        check("legacy into a sidecar-less table is allowed", True)

        mod.cw.write_meta(mod._sidecar(out),
                          {"cfg_hash": "abc", "crop": w.CropSpec.canon().to_dict(),
                           "mania_columns": "auto"})
        try:
            mod._precheck_sidecar(out, args_legacy, False)
            check("crop change is refused", False)
        except SystemExit:
            check("crop change is refused", True)
        meta = {"cfg_hash": "def", "crop": w.CropSpec.canon().to_dict(),
                "mania_columns": "auto"}
        try:
            mod._check_merge_guard(out, meta, False)
            check("cfg change is refused", False)
        except SystemExit:
            check("cfg change is refused", True)
        mod._check_merge_guard(out, meta, True)
        check("--allow-mixed-crop overrides", True)

        # _export_windows_table: fake tokenizer + fake model
        snap = w.ProcessorSnapshot(16.0, 16.0, 1.0, 4000, "longest",
                                   "longest_first", 16000, 10, False,
                                   w.CropSpec.canon().to_dict())
        files = {1: td / "1.osu", 2: td / "2.osu"}
        md5_map = {1: "m1", 2: "m2"}
        csum = {1: "m1", 2: "m2"}

        def fake_tokenize(_proc, path, *, beatmap_id, md5, **_kw):
            raw = [np.full(80, 1, np.int32), np.full(200, 1, np.int32)]
            return make_ct(beatmap_id, raw, max_length=4000)
        old = mod.cw.tokenize_chart
        mod.cw.tokenize_chart = fake_tokenize
        try:
            args2 = types.SimpleNamespace(
                window_dir=td / "win", window_index=td / "index.parquet",
                refresh_windows=False, window_batch=4, window_buffer=8,
                window_seed=0, window_dtype="float16", verify_windows=4,
                on_truncate="warn", mania_columns="auto", audio="off")
            def encode(ids, mask):
                outv = []
                for r, m in zip(ids, mask):
                    e = np.bincount(r[m > 0] % 512, minlength=512).astype(np.float64)
                    outv.append(e / (np.linalg.norm(e) + 1e-12))
                return np.stack(outv)
            table, meta2 = mod._export_windows_table(
                files, md5_map, csum, None, snap, TIDS, w.CropSpec.canon(),
                args2, "rev", np.float32, "sdpa", "cfg-z", encode, None)
            check("window table covers both charts", len(table) == 2,
                  str(len(table)))
            check("window table n_windows", sorted(table.n_windows.tolist()) == [2, 2])
            check("window table ids", set(table.beatmap_id) == {1, 2})
            check("index written", (td / "index.parquet").exists())
            check("index meta written", (td / "index.meta.json").exists())
            m = json.loads((td / "index.meta.json").read_text())
            check("meta cfg", m["cfg_hash"] == "cfg-z" and m["export"]["n_charts"] == 2)
            idx = pd.read_parquet(td / "index.parquet")
            check("manifest flags", set(idx.columns) >= {
                "n_holds_split", "columns_emitted", "n_core_tokens_dropped"})
        finally:
            mod.cw.tokenize_chart = old


def test_from_osu_dir_end_to_end() -> None:
    """from_osu_dir with a fake CM3P package: T0/T1/T2 wiring, table, index."""
    print("\nfrom_osu_dir end-to-end (fake cm3p package)")
    import hashlib
    mod = _load_step0c()
    import torch

    class FakeProc:
        default_kwargs = {"beatmap_kwargs": {
            "max_length": 4000, "padding": "longest", "truncation": "longest_first",
            "window_length_sec": 16.0, "window_stride_sec": 16.0,
            "min_window_length_sec": 1.0}}

        def __init__(self):
            self.beatmap_tokenizer = FakeTok()
            self.beatmap_parser = types.SimpleNamespace(
                add_hitsounds=True, add_kiai=True, add_mania_sv=True,
                add_timing=True, add_timing_points=True, add_snapping=True)

        @classmethod
        def from_pretrained(cls, rev):
            return cls()

        def __call__(self, **kwargs):
            # emulate the parser: the SV token disappears with add_mania_sv=False
            first = [1, 10] + ([60, 61, 50]
                                if self.beatmap_parser.add_mania_sv else [])
            return {"input_ids": [np.array(first, np.int32),
                                  np.array([1, 2, 3], np.int32)]}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

        def to(self, device=None, dtype=None):
            return self

        def eval(self):
            return self

        def __call__(self, input_ids, attention_mask, return_loss=False):
            v = torch.zeros((input_ids.shape[0], 512))
            for i in range(input_ids.shape[0]):
                ids = input_ids[i][attention_mask[i].bool()].long()
                v[i] = torch.bincount(ids % 512, minlength=512).float()
            v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-9)
            return types.SimpleNamespace(beatmap_embeds=v)

    pkg = types.ModuleType("cm3p")
    pkg.__path__ = []
    proc_mod = types.ModuleType("cm3p.processing_cm3p")
    proc_mod.CM3PProcessor = FakeProc
    model_mod = types.ModuleType("cm3p.modeling_cm3p")
    model_mod.CM3PModel = FakeModel
    parse_mod = types.ModuleType("cm3p.parsing_cm3p")
    parse_mod.load_beatmap = lambda p: FakeBM()
    parse_mod.get_song_length = lambda samples, sr, b: 33.0
    old = {k: sys.modules.get(k) for k in
           ("cm3p", "cm3p.processing_cm3p", "cm3p.modeling_cm3p", "cm3p.parsing_cm3p")}
    sys.modules.update({"cm3p": pkg, "cm3p.processing_cm3p": proc_mod,
                        "cm3p.modeling_cm3p": model_mod,
                        "cm3p.parsing_cm3p": parse_mod})
    try:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ids = [101, 102]
            for b in ids:
                (td / f"{b}.osu").write_text("osu file format v14\n")
            csum = {b: hashlib.md5((td / f"{b}.osu").read_bytes()).hexdigest()
                    for b in ids}
            want = pd.DataFrame({"beatmap_id": ids,
                                 "checksum": [csum[b] for b in ids]})
            args = types.SimpleNamespace(
                mania_columns="auto", crop="canon", allow_config_mismatch=False,
                core_audit=2, windows="on", window_dir=td / "win",
                window_index=td / "index.parquet", refresh_windows=False,
                window_batch=2, window_buffer=4, window_seed=0,
                window_dtype="float16", verify_windows=4, on_truncate="warn",
                audio="off", infer_batch=2)
            table, meta = mod.from_osu_dir([str(td)], want, mod.CM3P_REPO, False,
                                           2, "cpu", args)
            check("table rows", len(table) == 2, str(len(table)))
            check("table n_windows", sorted(table.n_windows.tolist()) == [2, 2])
            check("table checksum_ok", bool(table.checksum_ok.fillna(False).all()))
            check("window index exists", (td / "index.parquet").exists())
            check("index meta exists", (td / "index.meta.json").exists())
            check("meta columns_emitted", meta["columns_emitted"] is True)
            check("meta export", meta["export"]["n_charts"] == 2)
            check("meta crop", meta["crop"]["add_hitsounds"] is False)
            check("npy files", len(list((td / "win").glob("*.npy"))) == 2)

            # main(): a normal run, then the "main table complete, export windows"
            # branch on the same table.
            csv = td / "ids.csv"
            csv.write_text("beatmap_id,checksum\n"
                           + "".join(f"{b},{csum[b]}\n" for b in ids))
            out = td / "out.parquet"
            argv = ["step0c", "--source", "osu-dir", "--ids-file", str(csv),
                    "--osu-dir", str(td), "--out", str(out), "--crop", "canon",
                    "--windows", "on", "--window-dir", str(td / "win"),
                    "--window-index", str(td / "index.parquet"),
                    "--core-audit", "1", "--mania-columns", "auto",
                    "--device", "cpu", "--window-batch", "2",
                    "--window-buffer", "4", "--verify-windows", "2"]
            old_argv = sys.argv
            sys.argv = argv
            try:
                mod.main()
            finally:
                sys.argv = old_argv
            check("main wrote table", out.exists())
            t1 = pd.read_parquet(out)
            check("main table rows", len(t1) == 2 and set(t1.beatmap_id) == set(ids),
                  str(len(t1)))
            check("main sidecar", (td / "out.parquet.meta.json").exists())
            sys.argv = argv
            try:
                mod.main()
            finally:
                sys.argv = old_argv
            t2 = pd.read_parquet(out)
            check("T2-only run keeps rows", len(t2) == 2, str(len(t2)))
    finally:
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_budget_guards() -> None:
    print("\noffline budget guards")
    spec = importlib.util.spec_from_file_location(
        "budget_test", ROOT / "scripts/cm3p_token_budget.py")
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)
    body = """osu file format v14

[General]
Mode: 3

[Difficulty]
CircleSize: 4

[TimingPoints]
0,500,4,2,0,60,1,0

[HitObjects]
64,192,1000,1,0,0:0:0:0:
"""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "1.osu"
        p.write_text(body)
        r = tb._chart_stats(str(p), 16.0, 1.0, 4000, False)
        check("normal chart accounted", "error" not in r and r["n_notes"] == 1)
        p2 = Path(td) / "2.osu"
        p2.write_text(body.replace("1000,1,0,0:0:0:0:", "20000000,1,0,0:0:0:0:"))
        r2 = tb._chart_stats(str(p2), 16.0, 1.0, 4000, False)
        check("implausible song rejected", "error" in r2, str(r2.get("error")))
        p3 = Path(td) / "3.osu"
        p3.write_text(body.replace("0,500,4,2,0,60,1,0", "0,nan,4,2,0,60,1,0"))
        import signal
        def _timeout(signum, frame):
            raise TimeoutError("chart_groups did not terminate")
        old = signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(5)
        try:
            osu = off.parse_osu(p3)
            g = off.chart_groups(osu, hitsounds=True, kiai=True, sv=True,
                                 columns=False)
            check("non-finite timing terminates", len(g) < 1000)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


def main() -> int:
    test_crop_spec()
    test_t0()
    test_buckets()
    test_build_and_audit()
    test_exporter()
    test_tokenize_chart()
    test_mania_column_patch()
    test_offline_audit()
    test_step0c_glue()
    test_from_osu_dir_end_to_end()
    test_budget_guards()
    print(f"\n{len(FAILS)} failure(s)")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
