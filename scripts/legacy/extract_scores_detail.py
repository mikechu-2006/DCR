"""Extract full score detail (incl. JSON statistics) from scores.sql."""
from __future__ import annotations
import csv, json, re, sys
from pathlib import Path

csv.field_size_limit(10 ** 9)
ESC = re.compile(r"\\(.)")
ESCMAP = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "b": "\b", "Z": "\x1a"}


def unescape(s: str) -> str:
    return ESC.sub(lambda m: ESCMAP.get(m.group(1), m.group(1)), s)


COLS = ["id", "user_id", "ruleset_id", "beatmap_id", "rank", "passed", "accuracy", "max_combo",
        "total_score", "pp", "legacy_score_id", "legacy_total_score", "mods", "n_perfect",
        "n_great", "n_good", "n_ok", "n_meh", "n_miss", "max_total", "ended_at"]
N_TAIL = 7

def main(dump: Path, out_path: Path):
    n = 0
    bad = 0
    with open(dump, encoding="utf-8", errors="replace") as f, open(out_path, "w", newline="", encoding="utf-8") as fo:
        wr = csv.writer(fo)
        wr.writerow(COLS)
        for line in f:
            if not line.startswith("INSERT INTO"):
                continue
            body = line[line.index("VALUES") + 6:].strip()
            body = body[1:]
            if body.endswith(");"):
                body = body[:-2]
            for row in body.split("),("):
                parts = row.split(",")
                if len(parts) < 12 + N_TAIL:
                    continue
                head = parts[:12]
                tail = parts[-N_TAIL:]
                raw = ",".join(parts[12:-N_TAIL]).strip()
                if raw.startswith("'") and raw.endswith("'"):
                    raw = raw[1:-1]
                try:
                    data = json.loads(unescape(raw))
                except Exception:
                    bad += 1
                    continue
                mods = "+".join(m["acronym"] for m in data.get("mods", []))
                st = data.get("statistics", {}) or {}
                mx = data.get("maximum_statistics", {}) or {}
                max_total = sum(v for k, v in mx.items() if k != "legacy_combo_increase")
                wr.writerow([head[0], head[1], head[2], head[3], head[7].strip("'"), head[8],
                             head[9], head[10], head[11], tail[0], tail[1], tail[2], mods,
                             st.get("perfect", 0), st.get("great", 0), st.get("good", 0),
                             st.get("ok", 0), st.get("meh", 0), st.get("miss", 0),
                             max_total, tail[4].strip("'")])
                n += 1
    print("rows:", n, "bad json:", bad, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
