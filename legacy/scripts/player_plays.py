#!/usr/bin/env python3
"""Player play records ∩ the 4K chart whitelist.

The two dumps in this repo (2026-09-01 mania top-1000 / top-10000) only cover the
*sampled* players, so a player outside that sample (e.g. 17598247) has no rows on
disk at all.  This script takes play records exported from the player's own client
and intersects them with manifests/whitelist_1k.csv (21,949 4K charts).

Two input kinds, freely mixable (rows are merged and de-duplicated):

  --scores-db   osu!(stable) scores.db -- raw binary, parsed here.
                Every score carries the beatmap MD5, and the whitelist has a
                'checksum' column, so the join needs no beatmap IDs.
  --api-json    JSON arrays written by scripts/fetch_player_scores.py
                (raw osu! API v2 best / recent payloads, joined by id).

    python scripts/player_plays.py --user 17598247 \
        --scores-db data/player/17598247/scores.db

Outputs (--out-dir, default data/processed/player_{user}/):

  plays.csv            every score, flattened (source / mods / counts / time)
  whitelist_hits.csv   the intersection, one row per chart, with star & difficulty
  report.json          counts, coverage, star histogram, mod distribution

Definitions:

  * "played" = at least one score exists for the chart; several mod variants of the
    same chart collapse into one row in whitelist_hits.csv.
  * The whitelist is mod-agnostic -- only the chart itself matters here.
  * stable scores.db is the local personal-best table: it has no failed plays.
"""
from __future__ import annotations

import argparse
import json
import struct
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# osu! stable mod bits (only what a score row can carry)
MOD_BITS = {
    1 << 0: "NF", 1 << 1: "EZ", 1 << 2: "TD", 1 << 3: "HD", 1 << 4: "HR",
    1 << 5: "SD", 1 << 6: "DT", 1 << 7: "RX", 1 << 8: "HT", 1 << 9: "NC",
    1 << 10: "FL", 1 << 11: "AT", 1 << 12: "SO", 1 << 13: "AP", 1 << 14: "PF",
    1 << 15: "4K", 1 << 16: "5K", 1 << 17: "6K", 1 << 18: "7K", 1 << 19: "8K",
    1 << 20: "FI", 1 << 21: "RN", 1 << 22: "CN", 1 << 23: "TP", 1 << 24: "9K",
    1 << 25: "CO", 1 << 26: "1K", 1 << 27: "3K", 1 << 28: "2K", 1 << 29: "V2",
}
MOD_TARGET_PRACTICE = 1 << 23

MODE_NAMES = {0: "osu", 1: "taiko", 2: "catch", 3: "mania"}
MODE_IDS = {v: k for k, v in MODE_NAMES.items()}

TICKS_EPOCH = 621_355_968_000_000_000  # .NET ticks at 1970-01-01
STAR_BINS = [0, 2, 3, 4, 5, 6, 7, 8, 100]
STAR_LABELS = ["<2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8+"]


class ScoresDbError(RuntimeError):
    """scores.db 结构不符合预期（版本过旧 / 文件截断 / 不是 scores.db）。"""


# --------------------------------------------------------------------------- stable binary
def _exact(f, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise ScoresDbError(
            f"scores.db 提前结束（偏移 {f.tell() - len(data)}，还差 {n - len(data)} 字节）")
    return data


def _u8(f) -> int:
    return _exact(f, 1)[0]


def _u16(f) -> int:
    return struct.unpack("<H", _exact(f, 2))[0]


def _i32(f) -> int:
    return struct.unpack("<i", _exact(f, 4))[0]


def _i64(f) -> int:
    return struct.unpack("<q", _exact(f, 8))[0]


def _text(f) -> str | None:
    """osu! stable string: 0x00 = null, 0x0b + ULEB128 length + UTF-8 bytes."""
    marker = _u8(f)
    if marker == 0x00:
        return None
    if marker != 0x0B:
        raise ScoresDbError(f"字符串标记异常：{marker:#04x}（偏移 {f.tell() - 1}）")
    length = 0
    shift = 0
    while True:
        byte = _u8(f)
        length |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            break
        if shift > 28:
            raise ScoresDbError("字符串长度前缀异常")
    if not length:
        return ""
    return _exact(f, length).decode("utf-8", "replace")


def mods_to_names(bits: int | None) -> str:
    if not bits:
        return "NM"
    names = [name for bit, name in MOD_BITS.items() if bits & bit]
    return "+".join(names) if names else f"0x{bits:x}"


def _ticks_to_iso(ticks: int | None) -> str | None:
    if not ticks or ticks <= TICKS_EPOCH:
        return None
    try:
        moment = datetime(1970, 1, 1) + timedelta(microseconds=(ticks - TICKS_EPOCH) // 10)
    except OverflowError:
        return None
    return moment.isoformat(timespec="seconds")


def read_stable_scores_db(path: Path) -> tuple[int, list[dict]]:
    rows: list[dict] = []
    with open(path, "rb") as f:
        version = _i32(f)
        n_beatmaps = _i32(f)
        if not 0 <= n_beatmaps <= 5_000_000:
            raise ScoresDbError(f"谱面组数异常：{n_beatmaps}（文件可能不是 scores.db）")
        for _ in range(n_beatmaps):
            group_md5 = _text(f)
            n_scores = _i32(f)
            if not 0 <= n_scores <= 100_000:
                raise ScoresDbError(f"成绩数异常：{n_scores}")
            for _ in range(n_scores):
                mode = _u8(f)
                _score_version = _i32(f)
                score_md5 = _text(f) or group_md5
                player = _text(f)
                _replay_md5 = _text(f)
                n300, n100, n50 = _u16(f), _u16(f), _u16(f)
                geki, katu, miss = _u16(f), _u16(f), _u16(f)
                total = _i32(f)
                combo = _u16(f)
                perfect = _u8(f)
                mods = _i32(f)
                _text(f)                    # reserved, empty
                ticks = _i64(f)
                _i32(f)                     # reserved, -1
                _sid = _i64(f)              # online score id
                if mods & MOD_TARGET_PRACTICE:
                    _exact(f, 8)            # target-practice accuracy
                rows.append({
                    "source": "stable:scores.db",
                    "player": player,
                    "beatmap_id": None,
                    "beatmap_md5": (score_md5 or "").lower() or None,
                    "mode": MODE_NAMES.get(mode, str(mode)),
                    "mods": mods_to_names(mods),
                    "mods_bits": mods,
                    "total_score": total,
                    "max_combo": combo,
                    "perfect": bool(perfect),
                    "n300": n300, "n100": n100, "n50": n50,
                    "geki": geki, "katu": katu, "miss": miss,
                    "pp": None,
                    "accuracy": None,
                    "rank": None,
                    "played_at": _ticks_to_iso(ticks),
                })
        tail = f.read()
    if tail:
        raise ScoresDbError(
            f"解析完成后还剩 {len(tail)} 字节：这个 scores.db 的结构比本脚本支持的旧，"
            "把文件放到 data/player/<user>/ 并告诉我，我来适配")
    return version, rows


# --------------------------------------------------------------------------- osu! API v2 JSON
def _api_mode(score: dict) -> str:
    if score.get("mode") in MODE_IDS:
        return str(score["mode"])
    ruleset = score.get("ruleset_id")
    if isinstance(ruleset, int):
        return MODE_NAMES.get(ruleset, str(ruleset))
    return "mania"


def _api_mods(score: dict) -> str:
    mods = score.get("mods")
    if isinstance(mods, str):
        return mods or "NM"
    if isinstance(mods, list) and mods:
        return "+".join(str(m) for m in mods)
    return "NM"


def read_api_json(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("scores", "data", "best", "recent"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            raise SystemExit(f"{path}: 顶层是对象且没有 scores/data/best/recent 数组")
    rows: list[dict] = []
    for score in payload:
        beatmap = score.get("beatmap") or {}
        beatmap_id = beatmap.get("id") or score.get("beatmap_id")
        checksum = (beatmap.get("checksum") or score.get("beatmap_md5") or "").strip().lower()
        user = score.get("user")
        rows.append({
            "source": f"api:{path.stem}",
            "player": user.get("username") if isinstance(user, dict) else None,
            "beatmap_id": int(beatmap_id) if beatmap_id else None,
            "beatmap_md5": checksum or None,
            "mode": _api_mode(score),
            "mods": _api_mods(score),
            "mods_bits": None,
            "total_score": score.get("total_score") or score.get("score"),
            "max_combo": score.get("max_combo"),
            "perfect": score.get("perfect"),
            "n300": None, "n100": None, "n50": None,
            "geki": None, "katu": None, "miss": None,
            "pp": score.get("pp"),
            "accuracy": score.get("accuracy"),
            "rank": score.get("rank"),
            "played_at": score.get("ended_at") or score.get("started_at"),
        })
    return rows


# --------------------------------------------------------------------------- intersection
def load_whitelist(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"白名单不存在：{path}")
    wl = pd.read_csv(path)
    for column in ("beatmap_id", "checksum"):
        if column not in wl.columns:
            raise SystemExit(f"{path}: 缺少 {column} 列（现有列：{list(wl.columns)}）")
    wl["beatmap_id"] = wl["beatmap_id"].astype("int64")
    wl["checksum"] = wl["checksum"].astype(str).str.strip().str.lower()
    if wl["beatmap_id"].duplicated().any():
        raise SystemExit(f"{path}: beatmap_id 有重复行")
    return wl


def load_meta(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    meta = pd.read_csv(path)
    keep = [c for c in ("beatmap_id", "version", "star", "bpm",
                        "max_combo", "count_total", "approved") if c in meta.columns]
    return meta[keep].drop_duplicates("beatmap_id")


def analyse(user, plays: pd.DataFrame, wl: pd.DataFrame,
            meta: pd.DataFrame | None, sources: list[dict]):
    plays = plays.copy()
    plays["beatmap_id"] = plays["beatmap_id"].astype("Int64")
    plays["beatmap_md5"] = plays["beatmap_md5"].astype("string").str.lower()
    plays["played_at"] = pd.to_datetime(plays["played_at"], utc=True, errors="coerce")

    md5_to_id = dict(zip(wl["checksum"], wl["beatmap_id"]))
    # API rows already carry ids; stable rows get theirs from the whitelist checksum
    plays["beatmap_id"] = plays["beatmap_id"].fillna(plays["beatmap_md5"].map(md5_to_id))
    plays["beatmap_id"] = plays["beatmap_id"].astype("Int64")

    has_id = plays["beatmap_id"].notna()
    plays["chart_key"] = plays["beatmap_id"].astype("string")
    plays.loc[~has_id, "chart_key"] = "md5:" + plays.loc[~has_id, "beatmap_md5"].fillna("?")

    wl_ids = set(wl["beatmap_id"])
    hits = plays[has_id & plays["beatmap_id"].isin(wl_ids)]

    hit_charts = int(hits["beatmap_id"].nunique())
    played_charts = int(plays["chart_key"].dropna().nunique())

    per_chart = (hits.groupby("beatmap_id")
                 .agg(n_scores=("total_score", "size"),
                      best_score=("total_score", "max"),
                      last_played=("played_at", "max"),
                      mods=("mods", lambda s: "+".join(sorted({m for v in s.dropna()
                                                               for m in str(v).split("+") if m}))))
                 .reset_index())
    per_chart["beatmap_id"] = per_chart["beatmap_id"].astype("int64")
    per_chart = per_chart.merge(wl, on="beatmap_id", how="left")
    if per_chart.empty:
        per_chart = pd.DataFrame(columns=["beatmap_id", "n_scores", "best_score",
                                          "last_played", "mods", "checksum", "beatmapset_id"])
    if meta is not None and not per_chart.empty:
        per_chart = per_chart.merge(meta, on="beatmap_id", how="left")

    star_hist: dict[str, int] = {}
    top_charts = []
    if "star" in per_chart.columns:
        stars = pd.cut(per_chart["star"], bins=STAR_BINS, labels=STAR_LABELS, right=False)
        star_hist = {str(k): int(v) for k, v in stars.value_counts().sort_index().items()}
        top = per_chart.dropna(subset=["star"]).nlargest(10, "star")
        for row in top.itertuples():
            top_charts.append({
                "beatmap_id": int(row.beatmap_id),
                "star": round(float(row.star), 2),
                "version": getattr(row, "version", None),
                "playcount": int(row.n_scores),
            })

    by_mode = plays["mode"].value_counts(dropna=False).to_dict()
    mods_hist = hits["mods"].value_counts().head(12).to_dict() if not hits.empty else {}
    times = plays["played_at"].dropna()

    report = {
        "user": user,
        "whitelist": {"charts": int(len(wl))},
        "sources": sources,
        "plays": {
            "rows": int(len(plays)),
            "distinct_charts": played_charts,
            "by_mode": {str(k): int(v) for k, v in by_mode.items()},
            "first_played": times.min().isoformat() if len(times) else None,
            "last_played": times.max().isoformat() if len(times) else None,
        },
        "intersection": {
            "charts": hit_charts,
            "share_of_whitelist": round(hit_charts / len(wl), 6) if len(wl) else None,
            "share_of_played": round(hit_charts / played_charts, 6) if played_charts else None,
            "star_histogram": star_hist,
            "mods": {str(k): int(v) for k, v in mods_hist.items()},
            "hardest": top_charts,
        },
        "not_in_whitelist_charts": played_charts - hit_charts,
    }
    return report, per_chart, plays


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True, help="osu! user id（只用于命名与报告）")
    ap.add_argument("--scores-db", type=Path, action="append", default=[],
                    help="osu!(stable) scores.db，可重复")
    ap.add_argument("--api-json", type=Path, action="append", default=[],
                    help="fetch_player_scores.py 落盘的 best/recent JSON，可重复")
    ap.add_argument("--whitelist", type=Path, default=Path("manifests/whitelist_1k.csv"),
                    help="白名单表（默认 manifests/whitelist_1k.csv）")
    ap.add_argument("--meta", type=Path, default=Path("data/processed/beatmap_meta_1k.csv"),
                    help="谱面元数据（星数/难度名；默认 data/processed/beatmap_meta_1k.csv）")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="输出目录（默认 data/processed/player_{user}）")
    args = ap.parse_args()

    if not args.scores_db and not args.api_json:
        raise SystemExit("至少给一个输入：--scores-db 或 --api-json（先看 docs/player_plays.md）")
    out_dir = args.out_dir or Path(f"data/processed/player_{args.user}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    sources: list[dict] = []
    for path in args.scores_db:
        if not path.exists():
            raise SystemExit(f"找不到 {path}")
        try:
            version, part = read_stable_scores_db(path)
        except ScoresDbError as exc:
            raise SystemExit(f"{path}: {exc}") from None
        rows += part
        sources.append({"kind": "stable_scores_db", "path": str(path), "db_version": version,
                        "rows": len(part),
                        "distinct_md5": len({r["beatmap_md5"] for r in part})})
    for path in args.api_json:
        if not path.exists():
            raise SystemExit(f"找不到 {path}")
        part = read_api_json(path)
        rows += part
        sources.append({"kind": "osu_api_json", "path": str(path), "rows": len(part),
                        "distinct_beatmaps": len({r["beatmap_id"] for r in part
                                                  if r["beatmap_id"]})})

    if not rows:
        raise SystemExit("输入里没有任何成绩")

    plays = pd.DataFrame(rows)
    wl = load_whitelist(args.whitelist)
    meta = load_meta(args.meta)
    report, per_chart, plays = analyse(args.user, plays, wl, meta, sources)

    plays_out = plays.drop(columns=["mods_bits"]).sort_values("played_at", na_position="last")
    plays_out.to_csv(out_dir / "plays.csv", index=False, encoding="utf-8")
    if "star" in per_chart.columns:
        per_chart = per_chart.sort_values("star", ascending=False, na_position="last")
    else:
        per_chart = per_chart.sort_values("n_scores", ascending=False)
    per_chart.to_csv(out_dir / "whitelist_hits.csv", index=False, encoding="utf-8")
    report["whitelist"]["path"] = str(args.whitelist)
    report["outputs"] = {"plays": str(out_dir / "plays.csv"),
                         "hits": str(out_dir / "whitelist_hits.csv")}
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    p = report["plays"]
    it = report["intersection"]
    print(f"玩家 {args.user}")
    for src in sources:
        extra = (f"{src['distinct_md5']:,} 个 MD5" if "distinct_md5" in src
                 else f"{src['distinct_beatmaps']:,} 张谱面")
        print(f"  来源    {src['path']}  ({src['rows']:,} 行 / {extra})")
    print(f"  白名单  {report['whitelist']['charts']:,} 张 4K 谱面")
    modes = ", ".join(f"{k}:{v}" for k, v in p["by_mode"].items())
    print(f"  玩过    {p['distinct_charts']:,} 张（去重；模式 {modes}）")
    print(f"  交集    {it['charts']:,} 张"
          f"（白名单 {it['share_of_whitelist']:.2%}，玩过 {it['share_of_played']:.2%}）")
    if p["first_played"]:
        print(f"  时间    {p['first_played']} → {p['last_played']}")
    if it["charts"] and it["star_histogram"]:
        print("  星数    " + "  ".join(f"{k}:{v}" for k, v in it["star_histogram"].items()))
    if it["hardest"]:
        print("  最难    " + ", ".join(f"{c['star']}★ {c['version']}" for c in it["hardest"][:5]))
    print(f"\n  -> {out_dir / 'whitelist_hits.csv'}")
    print(f"  -> {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
