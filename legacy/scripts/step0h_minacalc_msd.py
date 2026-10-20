#!/usr/bin/env python
"""Step 0h -- MinaCalc MSD 谱面资产:  .osu -> (beatmap_id, msd_overall..msd_technical).

这是 step2/minacalc_as_feature 的**上游资产**（只依赖 .osu，不依赖任何成绩），所以它属
step0 系：跑一次、进 cache_dir，之后 step2 只读表。谱面集合默认取 step0/whitelist 的
manifests/whitelist_{tag}.csv，.osu 正文来自 step0d 的共享缓存 data/osu_cache/{id}.osu。

后端（二选一，推荐第一个）:

  --lib build/libminacalc.so      ctypes 直连 C API（kangalio/minacalc-standalone）
                                  接口见其 API.h:
                                    CalcHandle* create_calc();
                                    MsdForAllRates calc_msd(CalcHandle*, const NoteInfo*, size_t);
                                    Ssr calc_ssr(CalcHandle*, NoteInfo*, size_t, float rate, float goal);
                                    int calc_version();
                                  构建:
                                    git clone https://github.com/kangalio/minacalc-standalone
                                    cd minacalc-standalone
                                    g++ -std=c++17 -O2 -fPIC -c MinaCalc/MinaCalc.cpp API.cpp
                                    g++ -shared -o libminacalc.so *.o
  --calc-bin path/to/cli          子进程后端，协议（stdin 谱面 / stdout 8 个数）:
                                    每行 "<row_time_seconds> <notes_bitmask>"
                                    输出一行 8 个浮点, 顺序 = overall stream jumpstream
                                    handstream stamina jackspeed chordjack technical

口径（会写进表里，供 step2 校验）:
  * 4K mania: 列 = floor(x * keycount / 512)，越界夹紧；同一毫秒的多个物件 OR 成一个 bitmask。
  * 长按只取**头部**当一个 tap（MinaCalc 的 NoteInfo 没有时长维度，这是已知失真）。
    长按占比另存 hold_ratio，便于 step2 侧做子集分析。
  * 默认 rate=1.0（step0 的总体只有 NM 的 1.0x）；--goal 默认 0.93，即社区口径的 MSD。
  * calc_version() 一定写进表：263 与 5xx 不可比，step2 会拒绝没有版本列的表。

用法:
    python scripts/step0h_minacalc_msd.py --tag 1k --cache data/osu_cache \
        --ids manifests/whitelist_1k.csv --lib build/libminacalc.so \
        --out data/processed/minacalc_msd_1k.parquet --jobs 8
    # 先小跑 200 张看看：
    python scripts/step0h_minacalc_msd.py ... --limit 200
"""
from __future__ import annotations

import argparse
import ctypes
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SKILLS = ["overall", "stream", "jumpstream", "handstream",
          "stamina", "jackspeed", "chordjack", "technical"]
MSD_COLS = ["msd_" + s for s in SKILLS]


# ------------------------------------------------------------------ C API 绑定
class NoteInfo(ctypes.Structure):
    _fields_ = [("notes", ctypes.c_uint), ("rowTime", ctypes.c_float)]


class Ssr(ctypes.Structure):
    _fields_ = [(s, ctypes.c_float) for s in SKILLS]


class MsdForAllRates(ctypes.Structure):
    # 0.7 .. 2.0, 步长 0.1，共 14 档
    _fields_ = [("msds", Ssr * 14)]


def rate_index(rate: float) -> int:
    return int(min(max(round((rate - 0.7) / 0.1), 0), 13))


class LibBackend:
    """ctypes 后端：每个线程一个 CalcHandle（calc 对象是有状态的，不能共享）。"""

    name = "minacalc-standalone (lib)"

    def __init__(self, lib_path: str, rate: float, goal: float):
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.calc_version.restype = ctypes.c_int
        self.lib.create_calc.restype = ctypes.c_void_p
        self.lib.destroy_calc.argtypes = [ctypes.c_void_p]
        self.lib.calc_msd.restype = MsdForAllRates
        self.lib.calc_msd.argtypes = [ctypes.c_void_p, ctypes.POINTER(NoteInfo), ctypes.c_size_t]
        self.lib.calc_ssr.restype = Ssr
        self.lib.calc_ssr.argtypes = [ctypes.c_void_p, ctypes.POINTER(NoteInfo),
                                      ctypes.c_size_t, ctypes.c_float, ctypes.c_float]
        self.version = int(self.lib.calc_version())
        self.rate, self.goal = float(rate), float(goal)
        self._tls = threading.local()

    def _handle(self):
        h = getattr(self._tls, "h", None)
        if h is None:
            h = self.lib.create_calc()
            self._tls.h = h
        return h

    def msd(self, rows: list) -> dict:
        n = len(rows)
        arr = (NoteInfo * n)(*[NoteInfo(int(bits), float(t)) for t, bits in rows])
        if abs(self.goal - 0.93) < 1e-9:
            out = self.lib.calc_msd(self._handle(), arr, n)
            ssr = out.msds[rate_index(self.rate)]
        else:
            ssr = self.lib.calc_ssr(self._handle(), arr, n, self.rate, self.goal)
        return {s: float(getattr(ssr, s)) for s in SKILLS}


class CliBackend:
    """子进程后端（协议见模块 docstring），给不想编 .so 的场景用。"""

    name = "external CLI"

    def __init__(self, argv: list, rate: float, goal: float):
        import subprocess
        self.argv, self.rate, self.goal = [str(a) for a in argv], float(rate), float(goal)
        self._subprocess = subprocess
        probe = self._subprocess.run(self.argv + ["--version"], capture_output=True, text=True)
        self.version = int((probe.stdout or "0").strip().split()[-1] or 0)

    def msd(self, rows: list) -> dict:
        payload = "".join(f"{t:.6f} {bits}\n" for t, bits in rows)
        p = self._subprocess.run(self.argv + ["--keycount", "4", "--rate", str(self.rate),
                                              "--goal", str(self.goal)],
                                 input=payload, capture_output=True, text=True, timeout=120)
        if p.returncode != 0:
            raise RuntimeError((p.stderr or "").strip()[:200] or f"exit {p.returncode}")
        vals = [float(x) for x in p.stdout.split()]
        if len(vals) < len(SKILLS):
            raise RuntimeError(f"后端只回了 {len(vals)} 个数")
        return dict(zip(SKILLS, vals))


# --------------------------------------------------------------------- .osu 解析
def parse_osu_mania(text: str) -> tuple:
    """(keycount, rows, n_notes, n_holds)；rows = [(time_seconds, bitmask), ...] 已排序。"""
    keycount, mode = 4, None
    section = ""
    per_ms = {}
    n_notes = n_holds = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("["):
            section = line.strip("[]")
            continue
        if section == "General" and line.startswith("Mode"):
            mode = int(line.split(":", 1)[1].strip() or 0)
        elif section == "Difficulty" and line.startswith("CircleSize"):
            keycount = int(float(line.split(":", 1)[1].strip()))
        elif section == "HitObjects":
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                x, t, typ = int(parts[0]), int(parts[2]), int(parts[3])
            except ValueError:
                continue
            is_tap, is_hold = bool(typ & 1), bool(typ & 128)
            if not (is_tap or is_hold):
                continue                      # spinner / 其它物件，mania 里不该有
            col = min(max(x * keycount // 512, 0), keycount - 1)
            per_ms[t] = per_ms.get(t, 0) | (1 << col)
            n_holds += int(is_hold)
            n_notes += int(is_tap)
    if mode is not None and mode != 3:
        raise ValueError(f"不是 mania 谱面 (Mode={mode})")
    rows = sorted((t / 1000.0, bits) for t, bits in per_ms.items())
    return keycount, rows, n_notes, n_holds


# ------------------------------------------------------------------------ 主流程
def read_ids(args) -> list:
    if args.ids:
        df = pd.read_csv(args.ids)
        col = "beatmap_id" if "beatmap_id" in df.columns else df.columns[0]
        return [int(v) for v in df[col].tolist()]
    if args.meta:
        df = pd.read_csv(args.meta)
        for col in ("beatmap_id", "id"):
            if col in df.columns:
                return [int(v) for v in df[col].tolist()]
        raise SystemExit(f"{args.meta} 里找不到 beatmap_id 列")
    if args.cache:
        return sorted(int(p.stem) for p in Path(args.cache).glob("*.osu") if p.stem.isdigit())
    raise SystemExit("需要 --ids / --meta / --cache 之一来确定谱面集合")


def main():
    ap = argparse.ArgumentParser(description="osu!mania 4K .osu -> MinaCalc MSD 表")
    ap.add_argument("--tag", default="1k", help="只用于日志与默认路径")
    ap.add_argument("--ids", default=None, help="CSV，需含 beatmap_id（如 manifests/whitelist_{tag}.csv）")
    ap.add_argument("--meta", default=None, help="step0 的 beatmap_meta 表（含 beatmap_id）")
    ap.add_argument("--cache", default="data/osu_cache", help=".osu 缓存目录（step0d 的产物）")
    ap.add_argument("--out", required=True, help="输出 parquet")
    ap.add_argument("--lib", default=None, help="libminacalc.so/.dylib/.dll（ctypes 后端）")
    ap.add_argument("--calc-bin", default=None, nargs="+", help="外部 CLI 后端（argv，可带空格分隔的参数）")
    ap.add_argument("--rate", type=float, default=1.0, help="music rate（表里只存这一档）")
    ap.add_argument("--goal", type=float, default=0.93,
                    help="目标得分比例；0.93 = 社区口径的 MSD，其它值走 calc_ssr")
    ap.add_argument("--keycount", type=int, default=4, help="只处理这个键位（默认 4）")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（冒烟用）")
    ap.add_argument("--state", default=None, help="断点续跑状态 json（默认 <out>.state.json）")
    ap.add_argument("--min-notes", type=int, default=1)
    args = ap.parse_args()

    if not args.lib and not args.calc_bin:
        raise SystemExit("需要 --lib 或 --calc-bin 之一作为 MinaCalc 后端")
    backend = (LibBackend(args.lib, args.rate, args.goal) if args.lib
               else CliBackend(args.calc_bin, args.rate, args.goal))
    print(f"[backend] {backend.name}  calc_version={backend.version}  "
          f"rate={args.rate} goal={args.goal}", flush=True)

    ids = read_ids(args)
    if args.limit:
        ids = ids[:args.limit]
    state_path = Path(args.state or (str(args.out) + ".state.json"))
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": {}, "failed": {}}
    todo = [i for i in ids if str(i) not in state["done"] and str(i) not in state["failed"]]
    print(f"[plan] {len(ids):,} 张待处理, 已缓存 {len(ids) - len(todo):,}, 本轮到 {len(todo):,}", flush=True)

    cache = Path(args.cache)
    lock = threading.Lock()
    t0 = time.time()
    counters = {"ok": 0, "skip": 0, "err": 0}

    def work(bid: int):
        path = cache / f"{bid}.osu"
        if not path.exists():
            return bid, None, "missing .osu"
        try:
            keycount, rows, n_notes, n_holds = parse_osu_mania(path.read_text(encoding="utf-8", errors="replace"))
            if keycount != args.keycount:
                return bid, None, f"keycount={keycount}"
            if len(rows) < args.min_notes:
                return bid, None, f"notes={len(rows)}"
            vals = backend.msd(rows)
            rec = {"beatmap_id": bid, **{f"msd_{s}": vals[s] for s in SKILLS},
                   "calc_version": backend.version, "keycount": keycount,
                   "n_notes": n_notes, "n_holds": n_holds,
                   "hold_ratio": (n_holds / (n_notes + n_holds)) if (n_notes + n_holds) else 0.0,
                   "rate": args.rate, "goal": args.goal}
            return bid, rec, None
        except Exception as exc:                      # noqa: BLE001 - 逐张记录，不中断整批
            return bid, None, f"{type(exc).__name__}: {exc}"

    rows_out = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for bid, rec, err in pool.map(work, todo):
            with lock:
                if rec is not None:
                    rows_out.append(rec)
                    state["done"][str(bid)] = 1
                    counters["ok"] += 1
                elif err is None:
                    counters["skip"] += 1
                else:
                    state["failed"][str(bid)] = err
                    counters["err"] += 1
                n = counters["ok"] + counters["skip"] + counters["err"]
                if n % 200 == 0:
                    print(f"  {n:,}/{len(todo):,}  ok={counters['ok']:,} "
                          f"err={counters['err']:,}  ({time.time()-t0:.0f}s)", flush=True)
            if len(rows_out) and len(rows_out) % 2000 == 0:
                state_path.write_text(json.dumps(state))

    # 续跑时把上一轮的 done 记录也读回来（只留本 tag 需要的 id）
    if rows_out or len(state["done"]) > len(rows_out):
        cache_rows = []
        done_keys = set(state["done"])
        if done_keys and Path(args.out).exists():
            prev = pd.read_parquet(args.out)
            cache_rows.append(prev[prev.beatmap_id.astype(str).isin(done_keys - {str(r["beatmap_id"]) for r in rows_out})])
        if cache_rows:
            rows_out.extend(pd.concat(cache_rows).to_dict("records"))
    df = pd.DataFrame(rows_out)
    if len(df):
        df = df.drop_duplicates("beatmap_id").sort_values("beatmap_id").reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    state_path.write_text(json.dumps(state, indent=0))
    ns = {k: int(v) for k, v in df["n_notes"].agg(["sum", "mean", "max"]).items()} if len(df) else {}
    print(f"[done] {args.out}: {len(df):,} 张谱面  ok={counters['ok']:,} "
          f"skip={counters['skip']:,} err={counters['err']:,}  ({time.time()-t0:.0f}s)")
    if len(df):
        print(f"[done] msd_overall 均值 {df.msd_overall.mean():.2f} / 最高 "
              f"{df.msd_overall.max():.2f}；hold_ratio 均值 {df.hold_ratio.mean():.3f}")
    errs = list(state["failed"].items())[:5]
    if errs:
        print(f"[warn] {len(state['failed']):,} 张失败，例如 " + "; ".join(f"{k}: {v}" for k, v in errs))


if __name__ == "__main__":
    main()
