#!/usr/bin/env python3
"""Fetch one player's public scores through the osu! API v2 (client credentials).

Only the read-only 'public' scope is used -- no password, no session cookie.
Create an OAuth application first (any name, redirect URL can be
http://localhost:727/ because the client-credentials flow never redirects):

    https://osu.ppy.sh/home/account/edit#oauth

Then:

    export OSU_CLIENT_ID=12345
    export OSU_CLIENT_SECRET=...        # 不要写进 git / 不要贴到聊天里
    python scripts/fetch_player_scores.py --user 17598247

Writes into --out-dir (default data/player/{user}/):

    api_user.json     /users/{user}          (username, stats)
    api_best.json     /users/{user}/scores/best?mode=mania   (paginated)
    api_recent.json   /users/{user}/scores/recent?mode=mania (last 100, incl. fails)

Coverage caveat: 'best' is the personal-best table (paginated, up to
--max-best-pages * 100 entries) and 'recent' is the last 100 plays.  That is NOT
the complete list of every map ever played -- the only complete local source is
the client database (see docs/player_plays.md).  Feed these files to
scripts/player_plays.py --api-json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = "https://osu.ppy.sh/api/v2"
TOKEN_URL = "https://osu.ppy.sh/oauth/token"
PAGE = 100


def _token(client_id: str, client_secret: str) -> str:
    body = urlencode({
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials",
        "scope": "public",
    }).encode()
    request = Request(TOKEN_URL, data=body, headers={
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    })
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise SystemExit(f"取 token 失败：HTTP {exc.code} {detail}") from None
    token = payload.get("access_token")
    if not token:
        raise SystemExit(f"取 token 失败：{payload}")
    return token


def _get(path: str, token: str, params: dict | None = None) -> object:
    url = API + path
    if params:
        url += "?" + urlencode(params)
    request = Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    })
    for attempt in range(4):
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code == 429:
                wait = int(exc.headers.get("Retry-After") or "5") + 1
                print(f"  限速，{wait}s 后重试（{path}）")
                time.sleep(wait)
                continue
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise SystemExit(f"GET {path} -> HTTP {exc.code} {detail}") from None
        except TimeoutError:
            if attempt == 3:
                raise
            time.sleep(3)
    raise SystemExit(f"GET {path} 失败：重试次数用完")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True, help="user id 或用户名")
    ap.add_argument("--mode", default="mania", choices=["osu", "taiko", "fruits", "mania"])
    ap.add_argument("--max-best-pages", type=int, default=10,
                    help="best 最多翻几页（每页 100 条，默认 10）")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="输出目录（默认 data/player/{user}/）")
    args = ap.parse_args()

    client_id = os.environ.get("OSU_CLIENT_ID", "").strip()
    client_secret = os.environ.get("OSU_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise SystemExit("先设置 OSU_CLIENT_ID / OSU_CLIENT_SECRET（见文档，不要贴到聊天里）")

    out_dir = args.out_dir or Path(f"data/player/{args.user}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("取 token ...")
    token = _token(client_id, client_secret)

    print(f"读用户 {args.user} ...")
    user = _get(f"/users/{args.user}", token)
    if not isinstance(user, dict) or "id" not in user:
        raise SystemExit(f"用户不存在或接口返回异常：{str(user)[:200]}")
    username = user.get("username")
    print(f"  {username} (id={user['id']})")
    (out_dir / "api_user.json").write_text(
        json.dumps(user, ensure_ascii=False, indent=2), encoding="utf-8")

    best: list[dict] = []
    seen: set = set()
    for page in range(max(1, args.max_best_pages)):
        batch = _get(f"/users/{args.user}/scores/best", token, {
            "mode": args.mode, "limit": PAGE, "offset": page * PAGE,
        })
        if not isinstance(batch, list) or not batch:
            break
        fresh = [s for s in batch if s.get("id") not in seen]
        for score in fresh:
            seen.add(score.get("id"))
        best += fresh
        print(f"  best 第 {page + 1} 页：{len(batch)} 条（累计 {len(best)}）")
        if len(batch) < PAGE or not fresh:
            break
        time.sleep(1.2)
    (out_dir / "api_best.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"读最近成绩 ...")
    recent = _get(f"/users/{args.user}/scores/recent", token, {
        "mode": args.mode, "limit": PAGE, "include_fails": 1,
    })
    recent = recent if isinstance(recent, list) else []
    (out_dir / "api_recent.json").write_text(
        json.dumps(recent, ensure_ascii=False, indent=2), encoding="utf-8")

    maps = {s.get("beatmap", {}).get("id") for s in best + recent if s.get("beatmap")}
    print(f"\n{username}: best {len(best):,} 条, recent {len(recent):,} 条, "
          f"去重谱面 {len(maps):,} 张")
    print(f"  -> {out_dir / 'api_best.json'}")
    print(f"  -> {out_dir / 'api_recent.json'}")
    print("\n下一步：python scripts/player_plays.py --user "
          f"{user['id']} --api-json {out_dir / 'api_best.json'} "
          f"--api-json {out_dir / 'api_recent.json'}")


if __name__ == "__main__":
    main()
