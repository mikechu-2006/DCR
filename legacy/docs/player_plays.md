# 玩家游玩记录 ∩ 4K 白名单

给**不在 dump 抽样里的玩家**（例如 17598247）算交集：他玩过的谱面里，有多少在
`manifests/whitelist_1k.csv`（21,949 张 4K 谱面）里。

## 1. 为什么需要玩家自己提供

`data/raw` 里的两个 dump 是 **2026-09-01 的 mania top-1000 / top-10000 抽样**：
只包含被抽中的玩家。抽查 17598247，三张表在 1k 和 10k 里都是 **0 行**：

```bash
D1=data/raw/extracted/2026_09_01_performance_mania_top_1000
grep -c -E '\([0-9]+,[0-9]+,17598247,' $D1/osu_scores_mania_high.sql   # 0
grep -c -E '\(17598247,'                 $D1/osu_user_beatmap_playcount.sql  # 0
grep -c -E '\(17598247,'                 $D1/osu_user_stats_mania.sql        # 0
```

（10k dump 的三张表同样为 0；`scores.sql` 里唯一一次命中是时间戳
`1759824791` 的巧合，不是 user_id。）

所以磁盘上没有这个玩家的任何成绩，交集必须由玩家本人补充数据。三条路都可以，
**推荐 A**（最完整、最简单）。

## 2. 文件放哪

统一放到：

```
data/player/17598247/
```

`data/**` 已在 `.gitignore` 里（`*.db` 也被单独忽略），这些文件只留在本机，
不会进 git、不会上传。

## 3. 方案 A（推荐）：osu!(stable) 的 `scores.db`

stable 的 `scores.db` 是本地**个人最佳成绩表**：每条成绩都带**谱面 MD5**，
而白名单里有 `checksum` 列，所以不需要 beatmap id 就能对齐。

1. **完全退出 osu!**（客户端运行时文件被占用）。
2. 打开 osu! 文件夹：游戏内 **Options → General → Open osu! folder**
   （或者按下表自己找）。
3. 把里面的 **`scores.db`** 复制到 `data/player/17598247/`。
4. 顺手确认一下文件大小：正常是几 MB 量级（几百 KB 到几十 MB 都正常）。

| 系统 | 典型路径 |
|---|---|
| Windows | `C:\Users\<你>\AppData\Local\osu!\scores.db` |
| macOS | 游戏内 Open osu! folder（Wineskin 包内 `drive_c/osu!/scores.db`） |
| Linux (Wine / Lutris) | `<prefix>/drive_c/users/<你>/AppData/Local/osu!/scores.db` |

文件里只有：谱面 MD5、你的玩家名、replay MD5、连击/判定数、mods、时间戳。
**没有**账号密码、没有邮箱、没有登录凭据。

## 4. 方案 B：osu!(lazer) 的 `client.realm`

lazer 的成绩存在 Realm 数据库里，谱面带在线 id，信息更全（连失败局都有）。

1. **完全退出 lazer**（Realm 运行时会加锁）。
2. 游戏内 **Settings → General → Open osu! folder**，或按下表找。
3. 复制 **`client.realm`** 到 `data/player/17598247/`。

| 系统 | 路径 |
|---|---|
| Windows | `%APPDATA%\osu\client.realm` |
| macOS | `~/Library/Application Support/osu/client.realm` |
| Linux | `~/.local/share/osu/client.realm` |
| Flatpak | `~/.var/app/sh.ppy.osu/data/osu/client.realm` |

注意：Realm 文件需要 schema 才能读，我会先拿副本试解析（只读，不写回）；
如果版本对不上，再退回方案 C。**不要把唯一的 client.realm 覆盖掉**，先复制。

## 5. 方案 C：osu! API v2（不传客户端文件）

只读公开成绩（`public` scope），不用给密码，也不用给网页 cookie。

1. 打开 <https://osu.ppy.sh/home/account/edit#oauth>，点 **New OAuth Application**：
   - Name：随便填（例如 `dcr-local`）
   - Redirect URL：填 `http://localhost:727/`（client credentials 流程不会跳转，
     只是表单必填）
2. 记下 **Client ID** 和 **Client Secret**（只显示一次）。
3. 在本机跑（secret 只进环境变量，不要贴到聊天里、不要提交）：

```bash
cd /home/mikechu/Documents/Projects/DCR
export OSU_CLIENT_ID=你的ClientID
read -rs OSU_CLIENT_SECRET && export OSU_CLIENT_SECRET   # 粘贴后回车，不回显
python scripts/fetch_player_scores.py --user 17598247
```

脚本会写 `data/player/17598247/api_user.json`、`api_best.json`、`api_recent.json`。

**覆盖范围**：`best` 是个人最佳（每页 100 条，默认最多翻 10 页），`recent` 是最近
100 条（含失败）。这**不是**「所有玩过的谱面」，所以报告里会把 API 口径单独标注。
要完整清单还是走 A / B。

## 6. 跑交集

拿到文件后：

```bash
# A：stable
python scripts/player_plays.py --user 17598247 \
    --scores-db data/player/17598247/scores.db

# B/C：API JSON（可以同时给多个文件，和 scores.db 混用也行）
python scripts/player_plays.py --user 17598247 \
    --api-json data/player/17598247/api_best.json \
    --api-json data/player/17598247/api_recent.json
```

产物在 `data/processed/player_17598247/`：

| 文件 | 内容 |
|---|---|
| `plays.csv` | 摊平后的每条成绩（来源、MD5、mods、判定数、时间） |
| `whitelist_hits.csv` | **交集**：按谱面去重，附星数、难度名、成绩次数、最好成绩 |
| `report.json` | 计数、覆盖率、按星数分布、mods 分布、最难的 10 张 |

终端会直接打印一行结论，例如：

```
玩家 17598247
  来源    data/player/17598247/scores.db  (26,481 行 / 11,331 个 MD5)
  白名单  21,949 张 4K 谱面
  玩过    11,331 张（去重；模式 mania:11284, osu:47）
  交集    1,234 张（白名单 5.62%，玩过 10.89%）
```

## 7. 口径

* **玩过 = 至少有一条成绩**；同一张谱面的不同 mod 会收敛成一张（`whitelist_hits.csv`
  一行），`plays.csv` 保留每一条。
* **白名单与 mod 无关**，这里只看谱面本身；mod 分布只是附带统计。
* stable `scores.db` 不含失败局（lazer 的 Realm 有，API 的 `recent` 也有）。
* 谱面按 **MD5 → checksum** 对齐；对不上的成绩会留在 `plays.csv`，
  计入「玩过但不在白名单」的计数，不会丢。

## 8. 安全

* **不要提供** osu! 密码、`osu_session` cookie、或任何网页登录态——那等价于账号控制权。
* OAuth 的 Client Secret 只用于换 token；只需要 `public` 只读权限，
  随时可以在设置页删除该应用，token 立即失效。
* `scores.db` / `client.realm` 只在本机读取；脚本不会写回原文件，
  也不会把内容发到任何地方。
