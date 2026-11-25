#!/usr/bin/env python
"""Self-check for the structured attribute channels (CPU, synthetic + real charts).

Run from data2vec_main/:  python scripts/test_attribute_channels.py

Covers the parts that are easy to get silently wrong:
  * channel spec counts/order for A0/A1/A2/A3,
  * release pulse placement and per-column exclusivity, window edges,
  * BPM / beat / bar phase values and red-line resets,
  * kiai effects bitmask (effects=9 regression),
  * volume / sample-set inheritance and the t+5ms hitsound-point rule,
  * SV parity with dcr.cm3p_offline.sv_groups,
  * zero-init conv1 expansion is function-preserving at step 0,
  * dataset integration (x shape and finiteness).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data2vec_main"))
sys.path.insert(0, str(ROOT))

from dcr.cm3p_offline import parse_osu, sv_groups  # noqa: E402
from d2v_main.attributes import (  # noqa: E402
    ChannelSpec, build_attribute_window, parse_attributes,
)
from d2v_main.charts import ChartStore, build_window, parse_chart  # noqa: E402
from d2v_main.config import load_config  # noqa: E402
from d2v_main.data import ChartWindowDataset, collate_windows  # noqa: E402
from d2v_main.checkpoint import load_model_state_expanded  # noqa: E402
from d2v_main.model import Data2VecModel  # noqa: E402

FRAME_HZ = 100
WINDOW = 1600


def _osu(notes, timing=None):
    """Minimal 4K mania .osu text. notes=(x,y,time,flags,hitSound,hitsample)."""
    if timing is None:
        timing = ["0,500,4,2,0,60,1,0"]
    lines = ["osu file format v14", "", "[General]", "Mode: 3", "",
             "[Difficulty]", "CircleSize:4", "", "[TimingPoints]"]
    lines += timing
    lines += ["", "[HitObjects]"]
    lines += [",".join(str(v) for v in n) for n in notes]
    return "\n".join(lines) + "\n"


def _tmp_osu(text: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix="d2v_attr_"))
    p = d / "123456.osu"
    p.write_text(text, encoding="utf-8")
    return p


def _spec(*groups, **sub):
    cfg = {"data": {"channels": {"version": "test", "groups": {g: {} for g in groups}}}}
    for g, v in sub.items():
        cfg["data"]["channels"]["groups"][g] = v
    return ChannelSpec.from_config(cfg)


def test_specs():
    assert _spec().n_channels == 8
    s12 = _spec("release")
    s20 = _spec("release", "tempo", "sv", "kiai")
    s27 = _spec("release", "tempo", "sv", "kiai", "meta")
    assert s12.n_channels == 12 and s20.n_channels == 20 and s27.n_channels == 27, (
        s12.n_channels, s20.n_channels, s27.n_channels)
    assert s20.names[8:12] == ("release_0", "release_1", "release_2", "release_3")
    assert "bpm" in s20.names and "beat_sin" in s20.names and "bar_cos" in s20.names
    assert "sv" in s20.names and "sv_change" in s20.names and "kiai" in s20.names
    assert s27.names[-1] == "volume"
    # legacy config files must derive the matching counts
    for arm, n in (("a12", 12), ("a20", 20), ("a27", 27)):
        cfg = load_config(str(ROOT / ("data2vec_main/configs/audio_style_%s.yaml" % arm)))
        assert ChannelSpec.from_config(cfg).n_channels == n, arm
    print("[ok] channel specs 8/12/20/27")


def test_release():
    # hold col0 t=1000 end=2000; hold col3 t=1000 end=1599+0 (inside first window)
    notes = [
        (64, 192, 1000, 128, 0, "2000:0:0:0:60:"),
        (448, 192, 1000, 128, 0, "1599:0:0:0:60:"),
    ]
    p = _tmp_osu(_osu(notes))
    attrs = parse_attributes(p, FRAME_HZ)
    spec = _spec("release")
    w = build_attribute_window(attrs, 0, WINDOW, FRAME_HZ, spec)
    assert w.shape == (4, WINDOW)
    rel = w[spec.attr_slice_of("release")]
    assert rel[0].size == WINDOW
    f_end_0 = int(round(2000 * FRAME_HZ / 1000.0))   # 200
    assert rel[0, f_end_0] > 0.99, float(rel[0, f_end_0])
    assert abs(float(rel[1].max())) < 1e-9 and abs(float(rel[2].max())) < 1e-9
    assert rel[3, int(round(1599 * FRAME_HZ / 1000.0))] > 0.99
    # second window: the first hold end is outside; nothing for col0
    w2 = build_attribute_window(attrs, WINDOW, WINDOW, FRAME_HZ, spec)
    assert float(w2[spec.attr_slice_of("release")][0].max()) == 0.0
    print("[ok] release pulses per column + window edges")


def test_tempo():
    # red line 120 BPM at 0, switch to 240 BPM at 5 s; meter 4
    timing = ["0,500,4,2,0,60,1,0", "5000,250,4,2,0,60,1,0"]
    p = _tmp_osu(_osu([(64, 192, 100, 1, 0, "0:0:0:0:60:")], timing=timing))
    attrs = parse_attributes(p, FRAME_HZ)
    spec = _spec("tempo")
    w = build_attribute_window(attrs, 0, WINDOW, FRAME_HZ, spec)
    bpm = w[spec.attr_index("bpm")]
    sin = w[spec.attr_index("beat_sin")]
    cos = w[spec.attr_index("beat_cos")]
    bar_sin = w[spec.attr_index("bar_sin")]
    bar_cos = w[spec.attr_index("bar_cos")]
    assert abs(float(bpm[0]) - 0.0) < 1e-6, bpm[0]              # 120 BPM
    assert abs(float(bpm[600]) - 0.5) < 1e-5, bpm[600]          # 240 BPM
    assert abs(float(sin[0])) < 1e-6 and abs(float(cos[0]) - 1.0) < 1e-6
    # t=1000 ms at 120 BPM -> 2 beats -> sin 0 cos 1
    assert abs(float(sin[100])) < 1e-5 and abs(float(cos[100]) - 1.0) < 1e-5
    # bar phase with meter 4: t=2000 ms -> 4 beats -> bar phase 1
    assert abs(float(bar_sin[200])) < 1e-5 and abs(float(bar_cos[200]) - 1.0) < 1e-5
    print("[ok] bpm/beat/bar phase and red-line reset")


def test_kiai_effects_bitmask():
    # effects=9 => kiai bit set + omit barline: must be ON (regression for the
    # string-equality bug in dcr/cm3p_offline.py)
    timing9 = ["0,500,4,2,0,60,1,0", "2000,500,4,2,0,60,1,9"]
    p = _tmp_osu(_osu([(64, 192, 100, 1, 0, "0:0:0:0:60:")], timing=timing9))
    attrs = parse_attributes(p, FRAME_HZ)
    spec = _spec("kiai")
    w = build_attribute_window(attrs, 0, WINDOW, FRAME_HZ, spec)
    k = w[spec.attr_index("kiai")]
    assert float(k[100]) == 0.0 and float(k[300]) == 1.0, (k[100], k[300])
    timing8 = ["0,500,4,2,0,60,1,0", "2000,500,4,2,0,60,1,8"]
    p8 = _tmp_osu(_osu([(64, 192, 100, 1, 0, "0:0:0:0:60:")], timing=timing8))
    a8 = parse_attributes(p8, FRAME_HZ)
    w8 = build_attribute_window(a8, 0, WINDOW, FRAME_HZ, spec)
    assert float(w8[spec.attr_index("kiai")].max()) == 0.0
    print("[ok] kiai effects bitmask (effects=9 regression)")


def test_volume_and_sample_set():
    # TP volume 60 / sample set 3 (drum); second TP at 1003 ms with volume 30.
    timing = ["0,500,4,3,0,60,1,0", "1003,500,4,3,0,30,1,0"]
    notes = [
        (64, 192, 1000, 1, 0, "0:0:0:0:0:"),      # raw 0,0 -> inherit: ss drum, vol .30 after +5ms
        (128, 192, 1500, 1, 2, "2:0:0:40:"),      # raw ss 2 (soft), raw vol 40 -> .40
    ]
    p = _tmp_osu(_osu(notes, timing=timing))
    attrs = parse_attributes(p, FRAME_HZ)
    spec = _spec("meta")
    w = build_attribute_window(attrs, 0, WINDOW, FRAME_HZ, spec)
    vol = w[spec.attr_index("volume")]
    ss_soft = w[spec.attr_index("ss_soft")]
    ss_drum = w[spec.attr_index("ss_drum")]
    ss_auto = w[spec.attr_index("ss_auto")]
    f0 = int(round(1000 * FRAME_HZ / 1000.0))
    f1 = int(round(1500 * FRAME_HZ / 1000.0))
    # t+5ms lookup lands on the TP at 1003 -> volume 30 = 0.30
    assert abs(float(vol[f0]) - 0.30) < 1e-6, vol[f0]
    assert abs(float(vol[f1]) - 0.40) < 1e-6, vol[f1]
    assert float(ss_drum[f0]) == 1.0 and float(ss_auto[f0]) == 1.0
    assert float(ss_soft[f1]) == 1.0 and float(ss_auto[f1]) == 0.0
    print("[ok] volume/sample-set inheritance and t+5ms HP rule")


def test_sv_parity():
    timing = ["0,500,4,2,0,60,1,0", "2000,-200,4,2,0,60,0,0"]
    p = _tmp_osu(_osu([(64, 192, 100, 1, 0, "0:0:0:0:60:"),
                       (64, 192, 3000, 1, 0, "0:0:0:0:60:")], timing=timing))
    attrs = parse_attributes(p, FRAME_HZ)
    osu = parse_osu(p)
    ref = sv_groups(osu, normalized=True, add_snapping=False)
    ref_t = np.asarray([float(g.time) for g in ref])
    ref_v = np.asarray([float(g.scroll) for g in ref])
    assert np.allclose(attrs.sv_off, ref_t, atol=1e-6), (attrs.sv_off, ref_t)
    assert np.allclose(attrs.sv_val, ref_v, atol=1e-6), (attrs.sv_val, ref_v)
    spec = _spec("sv")
    w = build_attribute_window(attrs, 0, WINDOW, FRAME_HZ, spec)
    sv = w[spec.attr_index("sv")]
    chg = w[spec.attr_index("sv_change")]
    assert abs(float(chg[200]) - 1.0) < 1e-6
    assert np.isfinite(sv).all()
    print("[ok] SV parity with dcr.cm3p_offline.sv_groups")


def test_malformed_fields():
    # Reproduce the HPC failure: malformed hit objects observed in the corpus.
    #  - type=5 (circle) with field 5 = an end-time-like value 90568:0:0:0:0
    #  - hitSound=99999 (out of bitmask range)
    notes = [
        (192, 192, 90480, 5, 99999, "90568:0:0:0:0"),
        (64, 192, 1000, 1, 0, "0:0:0:0:60:"),
    ]
    # also a malformed timing point: meter=99999 and volume=99999
    timing = ["0,500,99999,2,0,60,1,0", "2000,500,4,2,0,99999,1,9"]
    p = _tmp_osu(_osu(notes, timing=timing))
    attrs = parse_attributes(p, FRAME_HZ)
    assert attrs.n_notes == 2
    assert int(attrs.hitsound[0]) == 0, int(attrs.hitsound[0])
    assert int(attrs.ss_raw[0]) == 0 and int(attrs.vol_raw[0]) == 0
    assert int(attrs.tp_vol.max()) <= 100 and int(attrs.red_meter.max()) <= 32
    spec = _spec("release", "meta")
    w = build_attribute_window(attrs, 0, WINDOW, FRAME_HZ, spec)
    assert w.shape[0] == spec.attr_names.__len__() and np.isfinite(w).all()
    print("[ok] malformed hit objects are sanitized (int8/int16 overflow regression)")


def test_hitsound_presence():
    # a21 = a20 + one binary "has extra hitsound" channel; no type/sample/volume.
    timing = ["0,500,4,2,0,60,1,0"]
    notes = [
        (64, 192, 1000, 1, 0, "0:0:0:0:60:"),       # plain (0 == normal)
        (128, 192, 2000, 1, 8, "0:0:0:0:60:"),      # clap -> presence
        (192, 192, 3000, 1, 6, "0:0:0:0:60:"),      # whistle+finish -> presence
        (256, 192, 4000, 1, 1, "0:0:0:0:60:"),      # explicit normal bit -> still plain
    ]
    p = _tmp_osu(_osu(notes, timing=timing))
    attrs = parse_attributes(p, FRAME_HZ)
    cfg = load_config(str(ROOT / "data2vec_main/configs/audio_style_a21.yaml"))
    spec = ChannelSpec.from_config(cfg)
    assert spec.n_channels == 21 and spec.names[-1] == "hs_presence", spec.names
    w = build_attribute_window(attrs, 0, WINDOW, FRAME_HZ, spec)
    pres = w[spec.attr_index("hs_presence")]
    f = [int(round(t * FRAME_HZ / 1000.0)) for t in (1000, 2000, 3000, 4000)]
    assert pres[f[0]] == 0.0 and pres[f[3]] == 0.0
    assert pres[f[1]] == 1.0 and pres[f[2]] == 1.0
    assert abs(float(pres.sum()) - 2.0) < 1e-6, float(pres.sum())
    assert not any(n.startswith("hs_whistle") for n in spec.names)
    print("[ok] a21 hs_presence: binary channel only, no type/volume/sample-set")


def test_shuffle_control():
    osu = ROOT / "data/osu_cache"
    cid = ChartStore(osu).ids[0]
    a = parse_attributes(ChartStore(osu).path(cid), FRAME_HZ)
    base_cfg = {"data": {"channels": {"groups": {"release": {}, "tempo": {}, "sv": {},
                                                  "kiai": {}, "meta": {}}}}}
    shuf_cfg = {"data": {"channels": {"groups": {"release": {}, "tempo": {}, "sv": {},
                                                  "kiai": {}, "meta": {}},
                                     "shuffle": True}}}
    plain = build_attribute_window(a, 0, WINDOW, FRAME_HZ, ChannelSpec.from_config(base_cfg))
    shuf1 = build_attribute_window(a, 0, WINDOW, FRAME_HZ, ChannelSpec.from_config(shuf_cfg))
    shuf2 = build_attribute_window(a, 0, WINDOW, FRAME_HZ, ChannelSpec.from_config(shuf_cfg))
    assert np.array_equal(shuf1, shuf2), "shuffle must be deterministic"
    assert not np.array_equal(plain, shuf1)
    # rolling only permutes frames: the multiset of per-frame channel sums is unchanged
    assert np.allclose(np.sort(plain.sum(axis=0)), np.sort(shuf1.sum(axis=0)), atol=1e-4)
    print("[ok] shuffle negative control is deterministic and alignment-breaking")


def test_zero_init_expansion():
    ckpt_path = ROOT / "data2vec_main/runs/smoke/checkpoints/last.pt"
    if not ckpt_path.exists():
        print("[skip] zero-init expansion: no smoke checkpoint (run scripts/smoke_test.py)")
        return
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg8 = ckpt["cfg"]
    cfg8["data"]["channels"] = None
    cfg8["model"]["in_channels"] = 8
    base = Data2VecModel(cfg8)
    base.load_state_dict(ckpt["model"])
    base.eval()
    import copy
    cfg12 = copy.deepcopy(cfg8)
    cfg12["model"]["in_channels"] = 12
    cfg12["data"]["channels"] = {"version": "t", "groups": {"release": {}}}
    exp = Data2VecModel(cfg12)
    info = load_model_state_expanded(exp, ckpt, expand=True)
    assert info["expanded"] and info["new_in"] == 12
    exp.eval()
    # smoke cfg has max_frames=512, so use 800 input frames (-> 400 output frames)
    n_in = 800
    x8 = torch.randn(2, 8, n_in)
    x12 = torch.cat([x8, torch.randn(2, 4, n_in)], dim=1)
    valid = torch.ones(2, (n_in + 1) // 2, dtype=torch.bool)
    with torch.no_grad():
        o8, _ = base.forward_student(x8, valid, None)
        o12, _ = exp.forward_student(x12, valid, None)
    assert torch.allclose(o8, o12, atol=1e-6), float((o8 - o12).abs().max())
    # after one backward step the new columns get non-zero gradients
    x12.requires_grad_(False)
    out, _ = exp.forward_student(x12, valid, torch.zeros_like(valid))
    out.float().pow(2).mean().backward()
    g = exp.conv.conv1.weight.grad
    assert g is not None and float(g[:, 8:, :].abs().sum()) > 0
    print("[ok] zero-init conv1 expansion is function-preserving; new cols get grads")


def test_dataset_integration():
    osu = ROOT / "data/osu_cache"
    ids = ChartStore(osu).ids[:3]
    cfg = load_config(str(ROOT / "data2vec_main/configs/audio_style_a20.yaml"),
                      {"data": {"osu_dir": str(osu)}})
    store = ChartStore(osu, ids=ids)
    ds = ChartWindowDataset(store, cfg, train=False, use_index_cache=False)
    batch = collate_windows([ds[i] for i in range(min(3, len(ds)))])
    x = batch["x"]
    assert x.shape[1:] == (20, WINDOW), x.shape
    assert bool(x.isfinite().all())
    print("[ok] dataset integration: x = (B, 20, 1600) finite")


def main() -> int:
    test_specs()
    test_release()
    test_tempo()
    test_kiai_effects_bitmask()
    test_volume_and_sample_set()
    test_sv_parity()
    test_malformed_fields()
    test_hitsound_presence()
    test_shuffle_control()
    test_zero_init_expansion()
    test_dataset_integration()
    # base 8ch path untouched
    osu = ROOT / "data/osu_cache"
    cid = ChartStore(osu).ids[0]
    events, n_frames = parse_chart(str(ChartStore(osu).path(cid)), FRAME_HZ)
    base = build_window(events, 0, WINDOW, FRAME_HZ, 8.0)
    assert base.shape == (8, WINDOW) and np.isfinite(base).all()
    print("[ok] base 8ch path unchanged")
    print("ATTR TESTS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
