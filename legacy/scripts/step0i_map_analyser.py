#!/usr/bin/env python
"""Step 0i -- osumania_map_analyser 谱面特征资产.

用 external/osumania_map_analyser（LeoBlackMT/osumania_map_analyser，插件 2.1.0）的**纯计算
模块**逐图算特征，产出一张 beatmap_id -> 特征 的表，供 step2/map_analyser 当冻结谱面特征。

相对 step0h（MinaCalc MSD）多了什么：
  * 6 个 Etterna/MinaCalc 版本 × 8 个 skillset（0.68 / 0.70 / 0.72 / 0.72.3 / 0.74 / 0.75）；
  * 4 套估算器：Sunny（含 ppMetrics：variety / accScalar / spikiness / switches）、
    Daniel、Azusa（numeric）、Roxy（numeric）、SunnyWindow 的 LN 星数、Interlude 星数；
  * 键型分析（Interlude RC/LN 算法）：ModeTag（RC/LN/HB/Mix/SV）、LN 比例、HB 行比例、
    SV 用量、6 类核心键型的覆盖时长 / 最高 BPM / importance / 加权 rating multiplier；
  * 结构量：OD、BPM、时长、排版行数、和弦比、note/s、长按时长、break。

实现方式是 Node harness（scripts/map_analyser_harness.mjs）：本脚本只负责切分 id、并发跑
harness、合并 NDJSON -> parquet。**只依赖 .osu**（不依赖成绩），所以属于 step0 资产。

用法:
    python scripts/step0i_map_analyser.py --tag 1k --ids data/processed/whitelist_1k.csv \
        --cache data/osu_cache --out data/processed/map_analyser_1k.parquet --jobs 4
    # 冒烟：
    python scripts/step0i_map_analyser.py --tag 1k --ids data/processed/whitelist_1k.csv \
        --out /tmp/mma.parquet --limit 50 --jobs 2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HARNESS = ROOT / "scripts" / "map_analyser_harness.mjs"
DEFAULT_PLUGIN = ROOT / "external" / "osumania_map_analyser" / "ManiaMapAnalyser by Leo_Black"


def read_ids(args) -> list:
    if args.ids:
        df = pd.read_csv(args.ids)
        col = "beatmap_id" if "beatmap_id" in df.columns else df.columns[0]
        return [int(v) for v in df[col].tolist()]
    if args.cache:
        return sorted(int(p.stem) for p in Path(args.cache).glob("*.osu") if p.stem.isdigit())
    raise SystemExit("需要 --ids 或 --cache 之一来确定谱面集合")


def plugin_info(plugin_dir: Path) -> dict:
    info = {"plugin": "LeoBlackMT/osumania_map_analyser", "plugin_dir": str(plugin_dir)}
    meta = plugin_dir / "metadata.txt"
    if meta.exists():
        for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("Version:"):
                info["analyser_version"] = line.split(":", 1)[1].strip()
    repo = plugin_dir.parent
    try:
        info["analyser_commit"] = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                  # noqa: BLE001 - 缺 git 不影响出表
        info["analyser_commit"] = None
    return info


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def main():
    ap = argparse.ArgumentParser(description="osu!mania .osu -> osumania_map_analyser 特征表")
    ap.add_argument("--tag", default="1k", help="只用于日志与默认路径")
    ap.add_argument("--ids", default=None, help="CSV，需含 beatmap_id（如 data/processed/whitelist_{tag}.csv）")
    ap.add_argument("--cache", default="data/osu_cache", help=".osu 缓存目录（step0d 的产物）")
    ap.add_argument("--out", required=True, help="输出 parquet")
    ap.add_argument("--jobs", type=int, default=4, help="并行 Node 进程数")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（冒烟用）")
    ap.add_argument("--keycount", type=int, default=4, help="只保留这个键位（默认 4）")
    ap.add_argument("--min-notes", type=int, default=1)
    ap.add_argument("--harness", default=str(DEFAULT_HARNESS))
    ap.add_argument("--plugin-dir", default=str(DEFAULT_PLUGIN))
    ap.add_argument("--node", default="node")
    ap.add_argument("--skip-ett", action="store_true", help="不算 6 个 MinaCalc 版本（调试用）")
    ap.add_argument("--with-graph", action="store_true",
                    help="额外算 Sunny 的时间曲线摘要（graph_mean/p50/p90/max/sd/spike/hot_frac）")
    ap.add_argument("--only-graph", action="store_true",
                    help="只算 Sunny 时间曲线（几十 ms/张），配合 --merge-into 给已有表补列")
    ap.add_argument("--merge-into", default=None,
                    help="把本轮产出按 beatmap_id 并到这张已有表上（左表列优先），写到 --out")
    ap.add_argument("--state", default=None, help="断点续跑状态 json（默认 <out>.state.json）")
    ap.add_argument("--resume", action="store_true", help="复用 <out> 里已有的行")
    args = ap.parse_args()

    if not Path(args.harness).exists():
        raise SystemExit(f"harness 不存在: {args.harness}")
    if not Path(args.plugin_dir).is_dir():
        raise SystemExit(f"plugin dir 不存在: {args.plugin_dir}\n"
                         f"  git clone https://github.com/LeoBlackMT/osumania_map_analyser.git external/osumania_map_analyser")

    ids = read_ids(args)
    out = Path(args.out)
    state_path = Path(args.state or (str(out) + ".state.json"))
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": {}, "failed": {}}
    have = set(int(k) for k in state["done"])
    n_have = sum(1 for i in ids if i in have)
    todo = [i for i in ids if i not in have]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[plan] {len(ids):,} 张谱面；已完成 {n_have:,}；本轮 {len(todo):,}；"
          f"jobs={args.jobs}", flush=True)

    cache = Path(args.cache)
    work = out.parent / f"_{out.stem}.work"
    work.mkdir(parents=True, exist_ok=True)
    for old in work.glob("part_*.ndjson"):
        old.unlink()

    # 按 id 切成 jobs 份（轮转分配，长谱面不会全落在一个进程里）
    shards = [todo[k::args.jobs] for k in range(args.jobs) if todo[k::args.jobs]]
    procs, part_paths = [], []
    t0 = time.time()
    for k, chunk in enumerate(shards):
        lst, part = work / f"ids_{k:02d}.tsv", work / f"part_{k:02d}.ndjson"
        with lst.open("w", encoding="utf-8") as fh:
            for bid in chunk:
                p = (cache / f"{bid}.osu").resolve()
                if p.exists():
                    fh.write(f"{bid}\t{p}\n")
        argv = [args.node, "--no-warnings", args.harness,
                "--list", str(lst), "--out", str(part),
                "--plugin-dir", str(Path(args.plugin_dir).resolve())]
        if args.skip_ett:
            argv.append("--skip-ett")
        if args.with_graph:
            argv.append("--with-graph")
        if args.only_graph:
            argv.append("--only-graph")
        log = (work / f"log_{k:02d}.txt").open("w", encoding="utf-8")
        procs.append((subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT), log, len(chunk)))
        part_paths.append(part)
    print(f"[run] {len(shards)} 个 Node 进程，日志在 {work}/log_*.txt", flush=True)

    total = sum(len(c) for c in shards)
    while any(p.poll() is None for p, _, _ in procs):
        time.sleep(10)
        done = sum(count_lines(p) for p in part_paths)
        dt = max(time.time() - t0, 1e-9)
        print(f"  {done:,}/{total:,}  {done / dt:.2f} charts/s  "
              f"eta {(total - done) / max(done / dt, 1e-9) / 60:.1f} min", flush=True)
    codes = [p.returncode for p, _, _ in procs]
    for _, log, _ in procs:
        log.close()
    if any(c != 0 for c in codes):
        tails = []
        for k in range(len(shards)):
            txt = (work / f"log_{k:02d}.txt").read_text(encoding="utf-8", errors="replace")
            tails.append(f"--- shard {k} ---\n" + "\n".join(txt.splitlines()[-8:]))
        raise SystemExit("Node harness 失败（exit " + str(codes) + "）\n" + "\n".join(tails))

    # ------------------------------------------------------------------ 合并
    frames = []
    for p in part_paths:
        if p.exists() and p.stat().st_size:
            frames.append(pd.read_json(p, lines=True, dtype=False))
    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(new) == 0:
        raise SystemExit("没有产出任何行")
    new = new.drop_duplicates("beatmap_id", keep="first").sort_values("beatmap_id")
    bad = new[new.get("ok", 0) != 1]
    good = new[new.get("ok", 0) == 1].copy()
    for bid, msg in zip(bad.get("beatmap_id", []), bad.get("error", [])):
        state["failed"][str(int(bid))] = str(msg)
    for bid in good["beatmap_id"].tolist():
        state["done"][str(int(bid))] = 1
        state["failed"].pop(str(int(bid)), None)

    if args.resume and out.exists():
        prev = pd.read_parquet(out)
        prev = prev[~prev.beatmap_id.isin(good.beatmap_id)]
        good = pd.concat([prev, good], ignore_index=True)

    good = good.drop(columns=[c for c in good.columns if c.startswith("err_")], errors="ignore")
    info = plugin_info(Path(args.plugin_dir))
    good["analyser_version"] = info.get("analyser_version")
    good["analyser_commit"] = info.get("analyser_commit")
    good["rate"] = 1.0
    good["goal"] = 0.93
    good = good.sort_values("beatmap_id").reset_index(drop=True)
    if args.merge_into:
        base = pd.read_parquet(args.merge_into)
        add = good.drop(columns=[c for c in good.columns if c in base.columns and c != "beatmap_id"],
                        errors="ignore")
        before = len(base)
        good = base.merge(add, on="beatmap_id", how="left")
        print(f"[merge] {Path(args.merge_into).name}: {before:,} 行 + "
              f"{add.shape[1] - 1} 列 -> {len(good):,} 行, 新列命中 "
              f"{int(good[add.columns[1]].notna().sum()) if add.shape[1] > 1 else 0:,}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    good.to_parquet(out, index=False)
    state.update({"n_ids": len(ids), "n_rows": int(len(good)), "n_failed": len(state["failed"]),
                  "seconds": round(time.time() - t0, 1), **info})
    state_path.write_text(json.dumps(state, indent=0))

    n_cols = len([c for c in good.columns if c not in ("beatmap_id", "status", "patt_category",
                                                       "patt_mode_tag", "analyser_version",
                                                       "analyser_commit")])
    print(f"[done] {out}: {len(good):,} 张谱面 × {n_cols} 个数值列  "
          f"ok={len(good):,} failed={len(state['failed']):,}  ({time.time() - t0:.0f}s)")
    print(f"[done] 版本 {info.get('analyser_version')} @ {str(info.get('analyser_commit'))[:12]}")
    if len(state["failed"]):
        print("[warn] 失败样例: " + json.dumps(dict(list(state["failed"].items())[:5]), ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
