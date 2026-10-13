#!/usr/bin/env python
"""Step 0f - player-voted beatmap tags (osu! "user tags") -> chart_user_tags_{tag}.parquet

Where this data lives
---------------------
osu!(lazer) lets a player vote for tags on the result screen after a C or better; the votes
land in osu-web's beatmap_tags table (beatmap_id, tag_id, user_id).  A tag is *displayed* once
it has >= 5 votes (config BEATMAP_TAGS_MIN_VOTES_DISPLAY), but the raw counts - including tags
with a single vote - are exposed on the public beatmapset page:

    GET https://osu.ppy.sh/beatmapsets/{beatmapset_id}
      -> <script id="json-beatmapset" ...>{...}</script>
           beatmaps[].top_tag_ids = [{"tag_id": 53, "count": 482}, ...]
           related_tags           = [{"id": 53, "name": "expression/difficulty spike",
                                      "ruleset_id": null, "description": "..."}, ...]

No account, no OAuth, no API key.  The data is NOT in any data.ppy.sh dump (the dumps ship 13
tables and none of them is beatmap_tags/tags), so it has to be crawled per beatmapset.

Outputs (--out-dir, default data/processed)
  chart_user_tags_{tag}.parquet    beatmap_id, tag_id, tag_name, count      (long format)
  chart_user_tag_vocab_{tag}.csv   tag_id, name, ruleset_id, description
Raw pages are kept in --work-dir (jsonl, one line per beatmapset) so a re-run resumes.

Examples
  python scripts/step0f_fetch_tags.py --tag 1k --limit 20          # smoke test
  python scripts/step0f_fetch_tags.py --tag 1k --workers 6 --max-rps 4
  python scripts/step0f_fetch_tags.py --tag 1k --refresh           # ignore the page cache
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import threading
import time
import urllib.request
from pathlib import Path

import pandas as pd

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
BLOB_RE = re.compile(r'<script id="json-beatmapset"[^>]*>(.*?)</script>', re.S)
URL = "https://osu.ppy.sh/beatmapsets/{sid}"

PRINT_LOCK = threading.Lock()


def log(msg: str) -> None:
    with PRINT_LOCK:
        print(msg, flush=True)


class RateLimiter:
    """One request every 1/max_rps seconds, shared by all workers."""

    def __init__(self, max_rps: float) -> None:
        self.interval = 0.0 if max_rps <= 0 else 1.0 / max_rps
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            sleep = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval
        if sleep:
            time.sleep(sleep)


def fetch_page(sid: int, limiter: RateLimiter, timeout: float, retries: int) -> dict | None:
    """One beatmapset page -> {"related_tags": [...], "beatmaps": {id: top_tag_ids}}."""
    last = None
    for attempt in range(retries):
        limiter.wait()
        try:
            req = urllib.request.Request(URL.format(sid=sid), headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                html = resp.read().decode("utf-8", "replace")
            m = BLOB_RE.search(html)
            if not m:
                last = "no json-beatmapset blob"
                raise ValueError(last)
            blob = json.loads(m.group(1))
            return {
                "related_tags": blob.get("related_tags") or [],
                "beatmaps": {b["id"]: (b.get("top_tag_ids") or []) for b in blob.get("beatmaps") or []},
            }
        except Exception as exc:                        # noqa: BLE001 - retried below
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(min(30.0, 2.0 ** attempt))
    log(f"[step0f] set {sid}: giving up ({last})")
    return None


def read_cache(path: Path) -> dict[int, dict]:
    cache: dict[int, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue                                # torn last line
                rec["beatmaps"] = {int(k): v for k, v in (rec.get("beatmaps") or {}).items()}
                cache[int(rec["beatmapset_id"])] = rec
    return cache


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="1k", help="dataset tag; decides the defaults")
    ap.add_argument("--meta", type=Path, default=None,
                    help="beatmap_meta_{tag}.csv whitelist (default data/processed/beatmap_meta_{tag}.csv)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--work-dir", type=Path, default=None,
                    help="page cache (default data/interim/step0f_{tag})")
    ap.add_argument("--limit", type=int, default=None, help="only the N most played beatmapsets")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-rps", type=float, default=4.0, help="global request rate cap (0 = unlimited)")
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--refresh", action="store_true", help="ignore the cached pages")
    ap.add_argument("--min-count", type=int, default=1,
                    help="drop tag votes below this count (default 1 = keep everything the page shows)")
    args = ap.parse_args()

    meta_path = args.meta or Path(f"data/processed/beatmap_meta_{args.tag}.csv")
    work = args.work_dir or Path(f"data/interim/step0f_{args.tag}")
    work.mkdir(parents=True, exist_ok=True)
    cache_path = work / "pages.jsonl"

    meta = pd.read_csv(meta_path)
    sets: dict[int, list[int]] = {}
    played: dict[int, int] = {}
    for row in meta.itertuples():
        sets.setdefault(int(row.beatmapset_id), []).append(int(row.beatmap_id))
        played[int(row.beatmapset_id)] = max(played.get(int(row.beatmapset_id), 0), int(row.playcount))

    order = sorted(sets, key=lambda s: -played[s])
    if args.limit:
        order = order[:args.limit]

    cache = {} if args.refresh else read_cache(cache_path)
    todo = [s for s in order if s not in cache]
    print(f"[step0f] {meta_path}: {len(meta)} beatmaps in {len(sets)} beatmapsets; "
          f"cached {len(order) - len(todo)}, to fetch {len(todo)}", flush=True)

    limiter = RateLimiter(args.max_rps)
    t0 = time.time()
    done = 0
    failed: list[int] = []
    with open(cache_path, "a", encoding="utf-8") as fh, cf.ThreadPoolExecutor(args.workers) as ex:
        futures = {ex.submit(fetch_page, sid, limiter, args.timeout, args.retries): sid for sid in todo}
        for fut in cf.as_completed(futures):
            sid = futures[fut]
            page = fut.result()
            done += 1
            if page is None:
                failed.append(sid)
            else:
                rec = {"beatmapset_id": sid, **page}
                cache[sid] = rec
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            if done % 25 == 0 or done == len(todo):
                rate = done / max(time.time() - t0, 1e-9)
                log(f"[step0f] {done}/{len(todo)} pages  {rate:.1f}/s  failed {len(failed)}  "
                    f"eta {(len(todo) - done) / max(rate, 1e-9) / 60:.1f} min")

    # ---- completeness: every whitelisted beatmapset/beatmap gets a verdict ----
    missing_sets = [s for s in order if s not in cache]
    (work / "missing_sets.txt").write_text("".join(f"{s}\n" for s in missing_sets))
    vocab: dict[int, dict] = {}
    rows = []
    cov = []
    for sid in order:
        rec = cache.get(sid)
        fetched = 1 if rec is not None else 0
        bmap = (rec or {}).get("beatmaps") or {}
        for t in (rec or {}).get("related_tags", []):
            vocab[int(t["id"])] = t
        for bid in sets[sid]:
            votes = bmap.get(bid)
            if fetched and votes is not None:
                for t in votes:
                    if int(t["count"]) >= args.min_count:
                        rows.append((bid, int(t["tag_id"]), int(t["count"])))
            cov.append((bid, sid, fetched, int(votes is not None), len(votes or []),
                        sum(int(t["count"]) for t in (votes or []))))
    covdf = pd.DataFrame(cov, columns=["beatmap_id", "beatmapset_id", "set_fetched",
                                       "beatmap_present", "n_tags", "n_votes"])
    cout = args.out_dir / f"chart_user_tag_coverage_{args.tag}.csv"
    covdf.to_csv(cout, index=False)

    tags = pd.DataFrame(rows, columns=["beatmap_id", "tag_id", "count"])
    if len(tags):
        tags["tag_name"] = tags.tag_id.map(lambda i: vocab.get(i, {}).get("name", f"tag_{i}"))
        tags = tags[["beatmap_id", "tag_id", "tag_name", "count"]].sort_values(
            ["beatmap_id", "count", "tag_id"], ascending=[True, False, True])
    out = args.out_dir / f"chart_user_tags_{args.tag}.parquet"
    tags.to_parquet(out, index=False)

    vdf = pd.DataFrame(
        [{"tag_id": i, "name": v.get("name"), "ruleset_id": v.get("ruleset_id"),
          "description": (v.get("description") or "").strip()} for i, v in sorted(vocab.items())]
    ) if vocab else pd.DataFrame(columns=["tag_id", "name", "ruleset_id", "description"])
    vout = args.out_dir / f"chart_user_tag_vocab_{args.tag}.csv"
    vdf.to_csv(vout, index=False)

    n_maps = tags.beatmap_id.nunique() if len(tags) else 0
    print(f"[step0f] tag votes {len(tags)} rows | tagged beatmaps {n_maps} / {len(meta)} "
          f"({n_maps / max(len(meta), 1) * 100:.1f}%) | vocabulary {len(vdf)}")
    print(f"[step0f] sets fetched {len(order) - len(missing_sets)}/{len(order)} | "
          f"missing sets {len(missing_sets)} | beatmaps absent from a fetched page "
          f"{int((covdf.beatmap_present == 0).sum())} | failed this run {len(failed)}")
    if missing_sets:
        print(f"[step0f] INCOMPLETE - see {work / 'missing_sets.txt'} and re-run to retry")
    if len(tags):
        top = tags.groupby("tag_name")["count"].agg(["sum", "size"]).sort_values("sum", ascending=False)
        print("[step0f] top tags by votes:")
        for name, r in top.head(15).iterrows():
            print(f"          {name:28s} votes={int(r['sum']):6d}  beatmaps={int(r['size'])}")
    print(f"[step0f] wrote {out}")
    print(f"[step0f] wrote {vout}")
    print(f"[step0f] wrote {cout}")


if __name__ == "__main__":
    main()
