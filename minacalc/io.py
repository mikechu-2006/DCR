"""逐区间难度：C++ 侧导出（export_itv）与 Python 侧读取。

C++ 工具见 minacalc/cpp/export_itv.cpp，编译：

    g++ -std=c++17 -O2 -I external/minacalc-standalone \
        minacalc/cpp/export_itv.cpp \
        external/minacalc-standalone/MinaCalc/MinaCalc.cpp \
        -o external/minacalc-standalone/build/export_itv

导出的是 Calc::base_adj_diff（与 goal 无关）与 itv_points，正是 score() 需要的两个数组。
"""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
from pathlib import Path

import numpy as np

from .core import ChartData

MAGIC = 0x5449434D
HEADER_FMT = "<4I2f"
HEADER_SIZE = struct.calcsize(HEADER_FMT)   # 24
DIMS_ORDER = ("stream", "js", "hs", "jackspeed", "cj", "tech")
DEFAULT_EXE = Path("external/minacalc-standalone/build/export_itv")
FORMAT_VERSION = 2      # 缓存键版本：v2 起 npz 里也存 jack 事件


def load_itv(path, name=None) -> ChartData:
    """读取 export_itv 的二进制输出。"""
    raw = Path(path).read_bytes()
    magic, ver, keycount, n = struct.unpack("<4I", raw[:16])
    if magic != MAGIC:
        raise ValueError(f"bad magic 0x{magic:08x} in {path}")
    if ver not in (1, 2):
        raise ValueError(f"unsupported format version {ver}")
    rate, goal = struct.unpack("<2f", raw[16:24])
    n_float = 12 * n + 2 * n
    arr = np.frombuffer(raw, dtype="<f4", count=n_float, offset=HEADER_SIZE)
    off = HEADER_SIZE + 4 * n_float
    diffs = arr[: 12 * n].reshape(2, 6, n).transpose(1, 0, 2).astype(float)
    pts = arr[12 * n:].reshape(2, n).astype(float)

    jack = None
    if ver >= 2:
        nj = np.frombuffer(raw, dtype="<u4", count=2, offset=off)
        off += 8
        jack = []
        for h in range(2):
            m = int(nj[h])
            itv = np.frombuffer(raw, dtype="<i4", count=m, offset=off)
            off += 4 * m
            y = np.frombuffer(raw, dtype="<f4", count=m, offset=off)
            off += 4 * m
            jack.append((itv.astype(np.int64), y.astype(float)))

    return ChartData(name or Path(path).stem, float(rate), 0.5, pts, diffs,
                     "minacalc-itv",
                     {"version": int(ver), "keycount": int(keycount),
                      "goal": float(goal), "numitv": int(n),
                      "n_jack": None if jack is None else [int(len(j[0])) for j in jack]},
                     jack=jack)


def export_itv(osu_path, out_path, rate=1.0, goal=0.93, exe=DEFAULT_EXE) -> Path:
    """调用 C++ 工具导出一张图的逐区间难度。"""
    exe = Path(exe)
    if not exe.exists():
        raise FileNotFoundError(f"exporter not built: {exe}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(exe), str(osu_path), str(out_path),
                    f"{rate:.6f}", f"{goal:.6f}"], check=True,
                   stdout=subprocess.DEVNULL)
    return out_path


def load_chart_itv(osu_path, rate=1.0, goal=0.93, exe=DEFAULT_EXE,
                   cache_dir=".minacalc_cache", use_cache=True) -> ChartData:
    """导出 + 缓存。缓存键 = .osu 内容哈希 + rate + goal + 格式版本。"""
    osu_path = Path(osu_path)
    key = (hashlib.sha1(osu_path.read_bytes()).hexdigest()[:16]
           + f"_itv_{rate:.3f}_{goal:.3f}_v{FORMAT_VERSION}")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    npz = cache / f"{key}.npz"
    if use_cache and npz.exists():
        z = np.load(npz, allow_pickle=False)
        meta = json.loads(str(z["meta"].item()))
        jack = None
        if "jack_itv0" in z:
            jack = [(z["jack_itv0"], z["jack_y0"]),
                    (z["jack_itv1"], z["jack_y1"])]
        return ChartData(osu_path.stem, meta["rate"], meta["interval"],
                         z["pts"], z["diffs"], meta["source"], meta, jack=jack)
    tmp = cache / f"{key}.bin"
    export_itv(osu_path, tmp, rate=rate, goal=goal, exe=exe)
    ch = load_itv(tmp, name=osu_path.stem)
    tmp.unlink(missing_ok=True)
    extra = {}
    if ch.jack is not None:
        extra = {"jack_itv0": ch.jack[0][0], "jack_y0": ch.jack[0][1],
                 "jack_itv1": ch.jack[1][0], "jack_y1": ch.jack[1][1]}
    np.savez_compressed(npz, pts=ch.pts, diffs=ch.diffs,
                        meta=np.array(json.dumps(
                            {**ch.meta, "rate": ch.rate, "interval": ch.interval,
                             "source": ch.source})), **extra)
    return ch
