"""Parse osu! data.ppy.sh SQL dumps into CSV tables.

Usage:
  python scripts/parse_dump.py <dump_dir> <out_dir> [table ...]
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

CSV_FIELDSIZE = csv.field_size_limit(10 ** 9)


def iter_insert_lines(path: Path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("INSERT INTO"):
                yield line


def rows_simple(line: str):
    """Rows whose fields contain no commas inside quoted strings."""
    body = line[line.index("VALUES") + 6:].strip()
    body = body[1:]                       # drop leading '('
    if body.endswith(");"):
        body = body[:-2]
    elif body.endswith(")"):
        body = body[:-1]
    for row in body.split("),("):
        yield [v[1:-1] if len(v) >= 2 and v[0] == "'" and v[-1] == "'" else v
               for v in row.split(",")]


def parse_careful(line: str):
    """Generic MySQL INSERT VALUES parser (handles commas/escapes in strings)."""
    i = line.index("VALUES") + 6
    n = len(line)
    row, field = [], []
    in_str = esc = has = quoted = False
    while i < n:
        c = line[i]
        if in_str:
            if esc:
                field.append(c); esc = False
            elif c == "\\":
                esc = True
            elif c == "'":
                in_str = False; quoted = True
            else:
                field.append(c)
        elif c == "'":
            in_str = True; has = True
        elif c == ",":
            row.append(("".join(field) if quoted else ("NULL" if not has else "".join(field))))
            field = []; has = quoted = False
        elif c == "(":
            row, field = [], []; has = quoted = False
        elif c == ")":
            row.append(("".join(field) if quoted else ("NULL" if not has else "".join(field))))
            yield row
            field = []; has = quoted = False
        elif c == ";":
            return
        elif c in " \t\r\n":
            pass
        else:
            field.append(c); has = True
        i += 1


def w(name: str, out: Path, header, rows):
    p = out / name
    with open(p, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        n = 0
        for r in rows:
            wr.writerow(r)
            n += 1
    print(f"wrote {p} rows={n}", flush=True)


def num(v):
    return None if v in ("NULL", "") else v


def parse_beatmaps(dump: Path, out: Path):
    def gen():
        for line in iter_insert_lines(dump / "osu_beatmaps.sql"):
            for r in parse_careful(line):
                # 0 beatmap_id 1 beatmapset_id 2 user_id 3 filename 4 checksum 5 version
                # 6 total_length 7 hit_length 8 countTotal 9 countNormal 10 countSlider
                # 11 countSpinner 12 diff_drain 13 diff_size 14 diff_overall 15 diff_approach
                # 16 playmode 17 approved 18 last_update 19 difficultyrating 20 max_combo
                # 21 playcount 22 passcount ... 27 bpm
                if len(r) < 23:
                    continue
                yield [r[0], r[1], r[3], r[5], r[8], r[13], r[16], r[17], r[19], r[20], r[21], r[22]]
    w("beatmaps.csv", out,
      ["beatmap_id", "beatmapset_id", "filename", "version", "countTotal", "keys",
       "playmode", "approved", "star", "max_combo", "playcount", "passcount"], gen())


def parse_mania_high(dump: Path, out: Path):
    def gen():
        for line in iter_insert_lines(dump / "osu_scores_mania_high.sql"):
            for r in rows_simple(line):
                # 0 score_id 1 beatmap_id 2 user_id 3 score 4 maxcombo 5 rank 6 count50
                # 7 count100 8 count300 9 countmiss 10 countgeki 11 countkatu 12 perfect
                # 13 enabled_mods 14 date 15 pp 16 replay 17 hidden 18 country
                yield [r[0], r[1], r[2], r[5], r[6], r[7], r[8], r[9], r[10], r[11],
                       r[12], r[13], r[14], num(r[15])]
    w("mania_high.csv", out,
      ["score_id", "beatmap_id", "user_id", "rank", "count50", "count100", "count300",
       "countmiss", "countgeki", "countkatu", "perfect", "enabled_mods", "date", "pp"], gen())


MOD_RE = re.compile(r'\\"acronym\\": \\"([A-Za-z0-9]+)\\"')


def parse_scores(dump: Path, out: Path):
    n_tail = 7  # pp, legacy_score_id, legacy_total_score, started_at, ended_at, unix_updated_at, build_id

    def gen():
        for line in iter_insert_lines(dump / "scores.sql"):
            body = line[line.index("VALUES") + 6:].strip()
            body = body[1:]
            if body.endswith(");"):
                body = body[:-2]
            for row in body.split("),("):
                parts = row.split(",")
                if len(parts) < 12 + n_tail:
                    continue
                head = parts[:12]
                tail = parts[-n_tail:]
                data = ",".join(parts[12:-n_tail])
                mods = "+".join(MOD_RE.findall(data))
                # 0 id 1 user_id 2 ruleset_id 3 beatmap_id 4 has_replay 5 preserve 6 ranked
                # 7 rank 8 passed 9 accuracy 10 max_combo 11 total_score 12 data 13.. tail
                yield [head[0], head[1], head[2], head[3],
                       head[7].strip("'"), head[8], head[9], head[10], head[11],
                       num(tail[0]), mods, tail[4].strip("'"), tail[3].strip("'"), tail[5]]
    w("scores_recent.csv", out,
      ["id", "user_id", "ruleset_id", "beatmap_id", "rank", "passed", "accuracy",
       "max_combo", "total_score", "pp", "mods", "ended_at", "started_at", "unix_updated_at"], gen())


def parse_playcount(dump: Path, out: Path):
    def gen():
        for line in iter_insert_lines(dump / "osu_user_beatmap_playcount.sql"):
            for r in rows_simple(line):
                yield r[:3]
    w("playcount.csv", out, ["user_id", "beatmap_id", "playcount"], gen())


def parse_stats(dump: Path, out: Path):
    def gen():
        for line in iter_insert_lines(dump / "osu_user_stats_mania.sql"):
            for r in rows_simple(line):
                # 0 user_id ... 7 accuracy 8 playcount 16 rank 23 country 24 rank_score 25 rank_score_index
                yield [r[0], r[7], r[8], r[16], r[23], r[24], r[25], r[27]]
    w("user_stats_mania.csv", out,
      ["user_id", "accuracy", "playcount", "rank", "country", "rank_score",
       "rank_score_index", "accuracy_new"], gen())


TABLES = {
    "beatmaps": parse_beatmaps,
    "mania_high": parse_mania_high,
    "scores": parse_scores,
    "playcount": parse_playcount,
    "stats": parse_stats,
}

if __name__ == "__main__":
    dump_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    which = sys.argv[3:] or list(TABLES)
    for name in which:
        print(f"== parsing {name}", flush=True)
        TABLES[name](dump_dir, out_dir)
    print("done", flush=True)
