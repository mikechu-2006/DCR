# Step 0 — Preprocessing

**Goal.** Turn the raw osu!mania dump into two tables:

1. **the play table** — the smallest possible play-level table:

   ```
   player_id, beatmap_id, timestamp, playcount_cur, loss
   ```

   exactly one row per qualifying play. Nothing else — no mods, no rate, no
   metadata, no aggregation. Everything downstream (cleaning, convergence tiers,
   weighting, modelling) starts from this file.

2. **the beatmap metadata table** — a lookup keyed by `beatmap_id`:

   ```
   beatmap_id -> beatmapset_id, mapper_id, version, star, bpm, max_combo,
                 count_total, diff_overall, diff_drain, hit_length,
                 total_length, playcount, passcount, approved, last_update,
                 checksum
   ```

   one row per 4K mania beatmap. It is a **side artifact**: it is not joined into
   the play table, and the play table keeps its five columns. Downstream code
   joins it on `beatmap_id` when it needs chart properties.

**Status.** Specification + implemented. `scripts/step0_preprocess.py` produced
`step0_1k.csv` + `beatmap_meta_1k.csv` (≈ 75 s) and `step0_10k.parquet` +
`beatmap_meta_10k.csv` (375 s); all numbers in §7 come from those two runs.

**Mod policy in one line.** Two stages. First *forget the mods that do not change the
chart* (`MR`, `SD`, `PF`) — canonicalise them to "off" — then keep the play iff what
is left is exactly `CL`. `NF` is chart-neutral too but is **rejected** rather than
forgotten, because it is the one such mod that changes *which plays are recorded*
(§9 **D1**). So `CL`, `CL+SD+PF` and `CL+MR` are the same observation; `CL+NF` is not
in the file at all.

---

## 1. Where this step sits

| Step | Name | Input | Output | In this document |
|---|---|---|---|---|
| **0** | **preprocessing** | raw dump | `step0_{tag}.csv` (plays) **+** `beatmap_meta_{tag}.csv` (beatmap lookup) | **yes** |
| 1 | data cleaning | step-0 CSV | filtered cells (junk plays, `playcount ≤ 2`, `n_scores ≤ 2`, convergence tiers, weights) | no |
| 2+ | modelling | cleaned cells | stage-1 latent vectors, stage-2 difficulty regression | no |

### Non-goals of step 0

Step 0 must **not** do any of the following — these belong to step 1 or later:

* cell-level filtering of any kind (no ACC floor, no play-count floor, no record-count floor);
* aggregation (no per-cell medians, no `loss_med`);
* convergence / drift judgement;
* **any column beyond the five in the play table** — chart properties live in the
  separate metadata table (§6.2) and are never carried on a play row;
* any *derived* chart quantity (no `star / count_total`, no note-density, no
  rank-based tier, no per-mod star). The metadata table stores what the dump
  stores; features are step-1 work.

The only filters step 0 applies are *definitional*: what counts as a valid
observation at all (4K mania, classic scoring, no non-classic mods, non-empty
judgement counts). The one deliberate exception is the rejection of `NF` (§9 D1),
which is a **population** choice rather than a chart choice — a `CL+NF` play is
chart-equivalent to `CL` but is not the same observation. It is called out explicitly
here because it is the single place step 0 filters on something other than the chart.

### Why step 0 is separate

Step 0 defines the **population**. Step 1 selects a **subset** of it. Keeping them
apart means the cleaning rules can be re-tuned, compared and ablated without
re-deriving `loss` and `playcount_cur` every time — and it makes the "the test set
must not be selected by the treatment under test" rule trivially satisfiable
(step 0 is the common denominator of every configuration).

---

## 2. Inputs

All paths and switches are arguments, never hard-coded (§10).

| Source table | File in the dump | Used for |
|---|---|---|
| `osu_beatmaps` | `osu_beatmaps.sql` | 4K mania whitelist (`playmode = 3`, `diff_size = 4`) **and** the beatmap metadata table (§6.2) |
| `scores` (modern) | `scores.sql` | play records + judgement counts + mods + `ended_at` |
| `osu_scores_mania_high` (legacy) | `osu_scores_mania_high.sql` | play records + judgement counts + mod bitmask + `date` |

### 2.1 The three dumps are not nested

Three dumps share the snapshot date 2026-09-01 and differ in *which players* were
sampled — and, as it turns out, in **how much of the database was shipped with
them**. They are not supersets of one another, and the difference is not a matter of
degree.

| table | `1k` | `10k` | `random` |
|---|---|---|---|
| `scores` (modern) | **910 MB** | — | — |
| `osu_scores_mania_high` (legacy) | 370 MB | **2.44 GB** | — |
| `osu_user_beatmap_playcount` | 78 MB | 597 MB | 161 MB |
| `osu_user_stats_mania` | 0.2 MB (1,000 users) | 2.1 MB (10,000) | 1.8 MB (10,000) |
| `sample_users` (ids **and usernames**) | **28 KB (1,000)** | — | — |
| `osu_beatmaps` | **56 MB** | — | — |
| `osu_beatmapsets` | 46 MB | — | — |
| `osu_beatmap_difficulty` | 1.03 GB | — | — |
| `osu_beatmap_difficulty_attribs` | 1.18 GB | — | — |
| `osu_beatmap_failtimes` | 142 MB | — | — |
| `osu_beatmap_performance_blacklist`, `osu_counts`, `osu_difficulty_attribs` | present | — | — |
| **tables shipped** | **13** | **3** | **2** |

| Tag | Dump | `scores.sql`? | `osu_beatmaps.sql`? | Output |
|---|---|---|---|---|
| `1k` | `2026_09_01_performance_mania_top_1000` | **yes** (plus legacy) | **yes** | `step0_1k.csv` + `beatmap_meta_1k.csv` |
| `10k` | `2026_09_01_performance_mania_top_10000` | **no** — legacy only | **no** | `step0_10k.parquet` + `beatmap_meta_10k.csv` |
| `random` | `2026_09_01_performance_mania_random_10000` | **no** | **no** | *not used* |

**`random_10000` has no score table at all.** It ships only
`osu_user_beatmap_playcount` and `osu_user_stats_mania` — two aggregate tables. There
is no play-level data of any kind: no judgement counts, no timestamps, no mods, no
beatmap ids of *plays* (only of counts). `loss` cannot be computed, and neither can
`playcount_cur`. So step 0 is not merely harder on that dump — **it is impossible**,
which is why the dump is unused in this project. What it can support is a
per-`(user, beatmap)` play-count table and per-user aggregates, i.e. a
population/composition study, never a learning curve.

> **2026-10-04**: the `random_10000` dump was deleted from disk
> (`data/raw/extracted_random/` and the `interim_random/` scratch, ~300 MB total).
> Nothing in the pipeline or in any analysis script reads it.

**`10k` ships the legacy score table but not `scores`.** Consequences, in order of
how much they cost:

* **No `osu_beatmaps`** → no chart properties at all. No `star`, `bpm`,
  `count_total`, `max_combo`, OD/HP, `approved`, `last_update`, `checksum`. This is
  the one gap that is *fully* recoverable, because `osu_beatmaps` is keyed by
  `beatmap_id` and not by user: the same table is in the 1k dump at the same snapshot
  date. `--beatmap-src` is exactly that workaround, and it is why
  `beatmap_meta_10k.csv` is byte-identical to `beatmap_meta_1k.csv`. The metadata
  table is a property of the dump date and the keys whitelist, **not** of which
  players were sampled.
* **No `scores`** → no modern rows, therefore no `legacy_score_id` column, therefore
  the cross-source dedup (§9 D3) is a no-op and every 10k play comes from the legacy
  table alone. It also means the 10k pool cannot see the modern table's coverage of
  recent plays, and the `CL`-vs-lazer distinction (§9 D1) does not exist there — but
  that costs nothing, because legacy rows are classic by construction.
* **No `sample_users`** → the 10,000 players are anonymous `user_id`s. There is no
  username column anywhere in the 10k dump, so no display names, no join to any
  external per-player source, and no way to confirm from the dump itself what the
  sampling criterion was. (The 1k dump does ship the list, 1,000 rows with
  `username`.)

**`1k` is the only dump with a beatmap side.** It is therefore the only dump on which
a *difficulty* question can be asked at all, and the only one that can be extended
with per-mod difficulty (`osu_beatmap_difficulty`) or a failure-position distribution
(`osu_beatmap_failtimes`).

### 2.2 The filters the dumps were taken with

Each `.sql` file carries its own `-- WHERE:` header. Those filters are part of the
data, not an implementation detail:

| table | filter |
|---|---|
| `scores` (1k only) | `user_id IN sample_users AND preserve = 1 AND ranked = 1 AND ruleset_id = 3` |
| `osu_scores_mania_high` (1k, 10k) | `user_id IN sample_users` |
| `osu_user_stats_mania` (all three) | `user_id IN sample_users` |
| `osu_beatmaps` (1k only) | names `sample_beatmaps_mania`, which is **not shipped** — and did not restrict the mode |

Three things follow.

**The modern table is a play history, not a leaderboard.** The `preserve = 1 AND
ranked = 1` filter looks like it should narrow the table to one best score per
mod combination, and it does not. Two checks:

* A single `(user_id, beatmap_id, mod-set) = (34376284, 3204545, CL)` key holds
  **53 rows** — consecutive plays on `2026-08-02 … 2026-08-26`, ACC climbing
  0.778 → 0.910 with non-monotone dips. That is a learning curve, not a set of
  personal bests, so the filter is not a "keep the best" rule.
* `ranked = 1` does not mean "on a ranked beatmap" either: **25.87% of the modern
  `CL` rows are on `approved = 4` (loved) maps**, essentially the same share as the
  legacy table's 24.91%. So the flag is not a beatmap-status filter.

What the filter *does* exclude cannot be determined from the dump, and that is the
honest answer: the modern table is a history of *submitted, preserved* scores, some
scores are missing, and the dump gives no way to enumerate which. The practical
consequence is only that "modern" should not be assumed complete either — see §3.2,
where the recorded/total ratio is 0.286 regardless of source.

**`osu_beatmaps` is dumped for all four rulesets.** The 235,061 rows break down as
152,347 osu!std + 38,562 taiko + 13,201 catch + 30,951 mania. This is why step 0
needs its own `playmode = 3 AND diff_size = 4` filter at all, and why that filter
selects 21,949 of the 30,951 mania rows — the remaining 9,002 are 5K–18K key modes
(7K alone is 5,785), which the project excludes by definition.

**`ruleset_id = 3` is redundant for the legacy table but not for `scores`.** The
legacy table is mania-only by name; the modern table is shared across rulesets, so
the filter is what makes it usable. Step 0 re-checks it (`modern_wrong_ruleset`) as a
guard against a future dump that drops the filter — on these dumps it never fires.

### 2.3 What is not recoverable, and what is impossible

This is the honest inventory. It is ordered by how badly each gap blocks work.

| Missing | Where | Why it cannot be reconstructed |
|---|---|---|
| **Failed / abandoned attempts** | *every* dump | The legacy table's `rank` is `enum('A','B','C','D','S','SH','X','XH')` — **`'F'` is not in the enum**, so a failure is unstorable at the schema level, not merely absent. The modern table stores `passed` and it is `1` for all 1,531,094 4K rows. |
| **Per-play failure position** | *every* dump | `osu_beatmap_failtimes` is aggregated per **beatmap** (`type ∈ {fail, exit}` × `p1..p100`), with `beatmap_id` not unique and no user column — it is a histogram over *all* players. `osu_user_beatmap_playcount` is per `(user, beatmap)` with no timestamps. `osu_user_stats_mania` is per **user** totals (`fail_count`, `exit_count`). Nowhere is there a per-play note position. |
| **Any play-level data at all** | `random` | No score table. |
| **Chart properties** | `10k`, `random` | No `osu_beatmaps`. Recoverable for `10k` by borrowing the 1k table (same date); not recoverable for `random` without also borrowing. |
| **Usernames** | `10k`, `random` | No `sample_users`. Only `user_id` survives. |
| **Play start time** | `1k` modern | `started_at` is `NULL` in the dumped rows; `ended_at` is present. Legacy has only `date` (end time). So "how long was this attempt" is not derivable. |
| **Unranked and non-preserved modern scores** | `1k` | The `preserve = 1 AND ranked = 1` filter is applied *before* the dump is written. Nothing downstream can undo it. |
| **Non-mania plays by the sampled players** | all | The score tables are mania-specific (`osu_scores_mania_high` by name, `scores` by `ruleset_id = 3`). A player's osu!std history is not in the dump, so cross-mode practice is invisible. |
| **Note count, if you needed it** | `10k`, `random` | The modern JSON carries `maximum_statistics` (e.g. `perfect: 814`), which *is* the note count, and the legacy table has no equivalent. In practice this costs nothing because `osu_beatmaps.count_total` is available — but only via the 1k dump. |

**What the two score tables are, side by side.** They are not two copies of one
thing; they are two different tables that happen to overlap:

| | modern (`scores`) | legacy (`osu_scores_mania_high`) |
|---|---|---|
| present in | `1k` only | `1k` and `10k` |
| rows on 4K (1k) | 1,531,094 | 1,918,989 |
| rows after the mod rule (1k) | 1,026,950 | 1,399,505 |
| rows the *other* table does not have | 100,163 (lazer-only) | 372,665 |
| time column | `ended_at` (and `started_at`, `NULL`) | `date` |
| mods | JSON `data.mods[].acronym`, **`CL` explicit** | `enabled_mods` bitmask, classic implicit |
| submission source | stable **and** lazer | stable only |
| classic marker | `CL` token ⟺ `legacy_score_id` non-`NULL` | by construction |
| join key to the other | `legacy_score_id` → `score_id` | `score_id` |
| rows per `(user, beatmap)`, canonical-`CL` only | 1.38 | 1.62 |
| max rows per `(user, beatmap)`, canonical-`CL` only | 54 | **125** |
| beatmap status of its rows | 74.1% ranked · 25.9% loved | 75.1% ranked · 24.9% loved |
| failure representable | `passed = 0` — but never occurs | **no** (`rank` enum has no `'F'`) |
| judgement weights | `statistics` keys (`perfect`/`great`/`good`/`ok`/`meh`) | `countgeki`/`count300`/`countkatu`/`count100`/`count50` |
| extra columns not in the other | `pp`, `build_id`, `has_replay`, `accuracy`, `maximum_statistics` | `replay`, `hidden`, `country_acronym` |

The practical consequence is §9 D3: the two overlap heavily (1,430,724 of the modern
4K rows carry a `legacy_score_id` that exists in the legacy table) but neither is a
superset of the other in rows. The legacy table reaches further back and holds more
rows per cell; the modern table is the only source for lazer plays and the only one
that spells out `CL`. **Both must be read, and the overlap must be removed by the
`legacy_score_id` join.**

**Not used by step 0:** `osu_user_beatmap_playcount`, `osu_beatmap_failtimes`,
`osu_beatmap_difficulty`, `osu_beatmap_difficulty_attribs`, `osu_beatmapsets`,
`osu_user_stats_mania`.

That is deliberate and it is what makes this step cheap: `playcount_cur` is defined
entirely from the play records themselves, so no external counter is joined in, and
the metadata table reads exactly one more table than the whitelist already needed.
The two tables that were considered and rejected for `playcount_cur` are discussed
in §9 **D2**; the four rejected for the metadata table are in §9 **D5**.

---

## 3. What the dump actually contains

Worth stating precisely, because it fixes what `playcount_cur` can and cannot mean.

### 3.1 Only **passed** plays are stored

| Evidence | Result |
|---|---|
| `osu_scores_mania_high.rank` over all 3,463,843 rows of the 1k legacy table | `S` 2,346,010 · `A` 490,933 · `X` 436,700 · `B` 99,062 · `SH` 59,726 · `C` 12,605 · `D` 11,035 · `XH` 7,772 — **no `F` at all** |
| the legacy `rank` **column type** | `enum('A','B','C','D','S','SH','X','XH')` — **`'F'` is not in the enum**, so a failure is unstorable at the schema level, not merely unobserved |
| `scores.passed` over all 2,650,870 rows of the 1k modern table | `1` for **100%** of rows |

So neither table contains a failed play, and neither table has any field describing
*where in the map* a play ended. The legacy half of that statement is structural: no
amount of re-reading the dump can produce a failed legacy play, because the type
system has no room for one. See §2.3 for the full inventory of what is missing.

### 3.2 The record set is a minority of the actual playing

`osu_user_beatmap_playcount.playcount` counts **pass + fail + exit** as one play each.
Comparing the two for 4K mania (1k dump):

| Quantity | Value |
|---|---|
| recorded plays / total plays, median over (player, beatmap) pairs | **0.286** |
| pass rate implied by `osu_beatmaps` (`passcount / playcount`) | 0.365 |
| decomposition of `playcount` over 21,933 4K maps | pass 36.5% · fail 12.6% · exit 42.6% |

Concrete example — player `566276`, beatmap `1048296`:

| | value |
|---|---|
| total plays (`playcount`) | 12 |
| recorded rows | 3 |
| plays with no record | 9 |

The three records are `2017-01-26` (rank B, DT), `2023-04-11` (S) and `2025-03-01`
(S). The player played the map 12 times between 2017 and 2025; 9 of those plays
(failed, or quit and restarted) left no trace at all.

Note that there are **two** independent reasons a play leaves no row, and they are
easy to conflate. The first is the one above: the play did not finish, so nothing was
submitted. The second applies only to the modern table, which was dumped with
`preserve = 1 AND ranked = 1` (§2.2) — a filter whose exact effect is not
determinable from the dump, but which certainly removes *some* submitted scores. So
a successful modern play can also be absent, and not for a reason the data explains.
That is part of why the legacy table holds more rows per cell (1.62 vs 1.38
canonical-`CL` rows per `(player, beatmap)`, with a much longer tail: 125 vs 54).

Neither table is a complete history. The legacy table is the denser of the two and
reaches further back; the modern table is the only one that can see lazer plays. Both
must be read, and neither can be read as "every attempt" (§9 D2).

### 3.3 Consequence

> **`playcount_cur` counts recorded successful plays, not plays.**
> It is the position of a play within the *observed* history of that
> (player, beatmap) pair — not within the player's real history on that map.
> On this data the observed history is roughly a third of the real one.

Step 0 deliberately takes the observed count. Recovering the real one would require
attributing the unobserved plays to gaps between records, which the dump does not
support — see §9 **D2** for what was tried and why it was dropped.

---

## 4. Definitions

### 4.1 `loss`

Classic (stable) scoring — `MAX` and `300` weigh the same — plus one pseudo-`250`
judgement so that a full-combo perfect play still has `1 − ACC > 0`:

```
ACC  = (300·(c300 + cMAX) + 200·c200 + 100·c100 + 50·c50 + 250)
       / (300 · (total + 1))
loss = log(1 − ACC)
```

with `total = cMAX + c300 + c200 + c100 + c50 + cmiss`.

Sign convention: `loss` is **decreasing** in ACC. ACC 0.99 → −4.605; ACC 0.90 →
−2.303. Better play = smaller (more negative) `loss`.

Column mapping per source:

| Judgement | classic weight | legacy `osu_scores_mania_high` | modern `scores.data.statistics` |
|---|---|---|---|
| MAX | 300 | `countgeki` | `perfect` |
| 300 | 300 | `count300` | `great` |
| 200 | 200 | `countkatu` | `good` |
| 100 | 100 | `count100` | `ok` |
| 50 | 50 | `count50` | `meh` |
| miss | 0 | `countmiss` | `miss` |

The modern table's own `accuracy` column must **not** be used: it is ScoreV2
(`MAX = 305`). Verified on row `2057059876`: stored `accuracy = 0.868409`; ScoreV2
recomputation `(305·300 + 300·264 + 200·208 + 100·31 + 50·4)/(305·814) = 0.86840`;
classic recomputation `214100/244200 = 0.87674`. Step 0 uses the classic one.

Rows with `total == 0` are dropped.

### 4.2 `timestamp`

The play's **end** time, `YYYY-MM-DD HH:MM:SS`, taken from the source table
(modern `ended_at`, legacy `date`). Emitted so that the ordering that produced
`playcount_cur` stays auditable, and so downstream steps can compute spans, gaps and
calendar drift without going back to the dump.

Full second-level resolution is present in both source tables. The older pipeline
reduced it to `(year, day-of-year)`; that was a parser limitation, not a property of
the dump, and step 0 does not inherit it. The worked example below is the reason it
matters: two of the five plays share a day and are 2 minutes apart.

### 4.3 `playcount_cur`

> **Definition.** Within one (player, beatmap) pair, order the recorded plays by time.
> For a given row, `playcount_cur` is **the number of recorded plays up to and
> including this one** — i.e. its 1-based rank. It is an integer, and it starts at 1
> for the earliest recorded play of the pair.

```
playcount_cur = 1, 2, 3, … within each (player_id, beatmap_id) cell
```

Equivalently: `playcount_cur` is the cumulative count of successful plays recorded
for that pair as of this play.

Properties that follow directly:

* the minimum in any cell is 1 and the values are contiguous with no gaps;
* the maximum in a cell equals the cell size `n`;
* it is *not* comparable across cells — a 3 in a cell with `n = 3` (the player's
  last recorded play there) and a 3 in a cell with `n = 50` are different situations;
* it understates the real play count by construction (§3.3).

Worked example — player `89545`, beatmap `777356` (a 5-record cell):

| timestamp | `playcount_cur` | `loss` |
|---|---|---|
| 2019-11-07 12:11:37 | 1 | −3.199165 |
| 2019-11-17 12:15:14 | 2 | −3.499919 |
| 2025-05-15 11:59:50 | 3 | −2.859518 |
| 2025-05-15 12:02:10 | 4 | −3.078706 |
| 2025-06-02 10:20:59 | 5 | −3.082001 |

Note the last two rows: two plays two minutes apart. `playcount_cur` still separates
them, which is why step 0 orders on the full timestamp and not on a date (§5).

---

## 5. Algorithm

```
 1. Read the 4K mania whitelist
       playmode == 3 AND diff_size == 4          (21,949 beatmaps in the 1k dump)
       read from --beatmap-src, which defaults to --dump-dir

 2. MODERN SOURCE  (skip entirely if scores.sql is absent — the 10k case)
      for each row of scores.sql with beatmap_id in the whitelist:
          tokens = set of mod acronyms in data.mods
          canon  = tokens - {MR, SD, PF}        # neutral mods canonicalised
                                                # to "off"                    (D1)
          KEEP iff canon == {CL}                # NF is NOT neutral, so {CL,NF}
                                                # survives to here and is rejected
          total  = sum(data.statistics.values())
          drop if total == 0
          loss   = classic ACC formula (§4.1)
          ts     = ended_at
          sid    = id
          remember legacy_score_id for step 4

 3. LEGACY SOURCE
      for each row of osu_scores_mania_high.sql with beatmap_id in the whitelist:
          mods = decode(enabled_mods)          # bits -> modern token vocabulary
                 NF=1 SD=32 PF=16384 MR=1073741824
                 a mask with any *other* bit set decodes to None
                 SD is folded away when PF is present (stable sets both)   (D1)
          canon = ({"CL"} | mods) - {MR, SD, PF}
                 # legacy rows are classic by construction, so 'CL' is implicit;
                 # the same canonicalisation then applies; a mask carrying the NF
                 # bit yields {CL,NF} and is rejected                       (D1)
          KEEP iff mods is not None and canon == {CL}
          total = countgeki + count300 + countkatu + count100 + count50 + countmiss
          drop if total == 0
          loss  = classic ACC formula (§4.1)
          ts    = date
          sid   = score_id

 4. DEDUPLICATE ACROSS SOURCES                                     (D3)
      the modern table is a superset of the legacy table for the 1k dump
      (1,430,724 of 1,430,731 modern rows carry a legacy_score_id that exists in
      osu_scores_mania_high, and all 1,034,788 legacy (user, beatmap) pairs appear
      in the modern table), so every play would otherwise be counted twice
      -> drop every legacy row whose score_id is in the modern legacy_score_id set

 5. ORDER
      sort by (user_id, beatmap_id, ts, sid)     — sid breaks ties inside a second

 6. COUNT
      playcount_cur = 1-based cumulative count inside each (user_id, beatmap_id) cell

 7. WRITE  (play table)
      player_id, beatmap_id, timestamp, playcount_cur, loss
      CSV, or parquet when --out ends in .parquet (the 10k case, 13 M rows)

 8. BEATMAP METADATA                                                  (§6.2)
      re-read osu_beatmaps.sql, keep the same whitelist as step 1
          playmode == 3 AND diff_size == 4
      project the fixed field list and write beatmap_id -> metadata
      one row per whitelisted beatmap, sorted by beatmap_id
```

No filtering beyond steps 1–4. In particular a play with ACC 0.23 is kept — it is a
pass, and whether it is a "junk play" is a step-1 decision. Step 8 is a projection,
not a filter: it adds no rows to and removes none from the play table, and it is
independent of steps 2–7 (it can be regenerated on its own).

Steps 2 and 3 run **two stages over one token vocabulary** (§9 D1): first
canonicalise the neutral mods away, then test membership. They differ only in how
the mods are read — the modern table spells them out, the legacy table packs them
into a bitmask. Each kept row's mod set is printed as a per-source breakdown at the
end of every run (§11), both as the source spelled it and after canonicalisation;
that pair of printouts is the check that the two encodings agree, and that the
collapse to `CL` is total.

### Notes on ordering

* Timestamps are **full `YYYY-MM-DD HH:MM:SS`** in both sources, and step 0 uses
  them. (The older pipeline reduced them to `(year, day-of-year)`; that was a parser
  limitation, not a property of the dump. The sample cell in §4.2 shows two plays on
  the same day that must be ordered separately.)
* `score_id` / `id` is monotone with submission order, so it is a sound tie-break
  for plays sharing a timestamp. Ties are rare but not impossible.
* The definition says "before this play *started*". Ordering by end time is
  equivalent in practice (plays do not overlap); `started_at` is frequently `NULL`
  in the modern table and legacy has no start time at all.

---

## 6. Outputs

Two files. They share one key — `beatmap_id` — and nothing else.

### 6.1 Play table — `data/processed/step0_{tag}.csv`

Single CSV, one row per qualifying play, no index:

```csv
player_id,beatmap_id,timestamp,playcount_cur,loss
89545,777356,2019-11-07 12:11:37,1,-3.199165
89545,777356,2019-11-17 12:15:14,2,-3.499919
89545,777356,2025-05-15 11:59:50,3,-2.859518
89545,777356,2025-05-15 12:02:10,4,-3.078706
89545,777356,2025-06-02 10:20:59,5,-3.082001
...
```

| Column | Type | Notes |
|---|---|---|
| `player_id` | int32 | the dump's `user_id` |
| `beatmap_id` | int32 | the osu! beatmap id (no rate suffix — §9 D1) |
| `timestamp` | string | `YYYY-MM-DD HH:MM:SS`, play end time (§4.2) |
| `playcount_cur` | int32 | ≥ 1, contiguous `1..n` inside each cell |
| `loss` | float32/float64 | `log(1 − ACC)` ≤ 0 |

Rows are written in `(player_id, beatmap_id, timestamp)` order, so the file is
also usable as a streaming input and needs no separate sort downstream. `loss` is
written at full precision — do not round, it is a log and loses resolution near 0.
`timestamp` is written in the source format, not as an epoch number, so the file
stays greppable.

**Post-conditions to assert** after writing:

1. no duplicate `(player_id, beatmap_id, playcount_cur)`;
2. within each cell, `playcount_cur` is exactly `1..n` with no gaps;
3. `playcount_cur >= 1` and `loss < 0` everywhere;
4. `timestamp` non-decreasing within each cell;
5. row count == number of plays that survived steps 1–4.

### 6.2 Beatmap metadata table — `data/processed/beatmap_meta_{tag}.csv`

Single CSV, one row per 4K mania beatmap, keyed by `beatmap_id`, no index:

```csv
beatmap_id,beatmapset_id,mapper_id,version,star,bpm,max_combo,count_total,diff_overall,diff_drain,hit_length,total_length,playcount,passcount,approved,last_update,checksum
222593,19194,566276,Another,5.07085,175,1176,1096,8,8,79,79,22002,7334,1,2024-06-26 07:53:44,a0ff...
257524,20794,1241097,Insane,4.50725,170,987,912,8,8,64,64,41387,12045,1,2024-06-26 07:53:44,4c1d...
```

| Column | Type | Source field (`osu_beatmaps`) | Notes |
|---|---|---|---|
| `beatmap_id` | int32 | `beatmap_id` | primary key |
| `beatmapset_id` | int32 | `beatmapset_id` | join key if set-level metadata is ever added |
| `mapper_id` | int32 | `user_id` | the mapper's user id, **not** a player id |
| `version` | string | `version` | difficulty name (`"Another"`, `"4K Hard"`, …) |
| `star` | float32 | `difficultyrating` | **no-mod** star rating for the map's own mode |
| `bpm` | float32 | `bpm` | base BPM, **no rate mod applied** |
| `max_combo` | int32 | `max_combo` | |
| `count_total` | int32 | `countTotal` | for mania this is the note count |
| `diff_overall` | float32 | `diff_overall` | mania OD |
| `diff_drain` | float32 | `diff_drain` | mania HP drain |
| `hit_length` | int32 | `hit_length` | seconds, drain time |
| `total_length` | int32 | `total_length` | seconds, full length |
| `playcount` | int64 | `playcount` | global plays on this map, all rulesets' players |
| `passcount` | int64 | `passcount` | global passes on this map |
| `approved` | int8 | `approved` | raw status code, see below |
| `last_update` | string | `last_update` | `YYYY-MM-DD HH:MM:SS`, server time |
| `checksum` | string | `checksum` | md5 of the `.osu` file — identifies the *revision* |

Rows are sorted by `beatmap_id`. `version` and `checksum` are the only string
columns that can contain a comma or a quote, so the CSV must be written with
standard quoting (it is).

**`approved` is emitted raw.** The dump stores an integer status, and the code
table is *not* re-derived here: mapping `1 → ranked` in step 0 would silently
encode a guess about a field whose semantics have changed across schema versions.
The observed value distribution on the 1k dump is in §7; step 1 can decide on the
mapping. This is the same reasoning as §9 **D1** — do not bake interpretation into
preprocessing.

**Post-conditions to assert** after writing:

1. `beatmap_id` is unique and strictly increasing;
2. `star > 0`, `count_total > 0`, `bpm > 0` for every row;
3. `hit_length <= total_length`;
4. every `beatmap_id` in the play table is present in the metadata table;
5. row count == number of 4K mania beatmaps in the whitelist.

Check 4 is the only cross-file invariant. It holds because both files are built
from the same whitelist, and it is worth asserting because a mismatch means the two
halves of step 0 were generated from different dumps.

---

## 7. Measured results

Both dumps, produced by `scripts/step0_preprocess.py` reading the raw SQL directly.
The commands are in §10. Wall time moves by a few seconds between runs with
disk-cache state.

| | 1k | 10k |
|---|---|---|
| runtime | ≈ 75 s | 375 s |
| play table | `step0_1k.csv` (67 MB) | `step0_10k.parquet` (180 MB) |
| metadata table | `beatmap_meta_1k.csv` (3.1 MB) | `beatmap_meta_10k.csv` (identical) |

### 1k — mod policy funnel

| Stage | Rows |
|---|---|
| 4K mania beatmaps (whitelist) | 21,949 |
| modern `scores` rows on 4K | 1,531,094 |
| → dropped: not canonical-`CL` (incl. every `NF` row) | −504,144 |
| → **modern rows kept** | **1,026,950** (67.1%) |
| legacy `osu_scores_mania_high` rows on 4K | 1,918,989 |
| → dropped: not canonical-`CL` (incl. every `NF` row) | −519,484 |
| → **legacy rows kept** | **1,399,505** (72.9%) |
| → dropped: already in the modern table | −1,026,840 |
| → **legacy rows kept** | **372,665** |
| **final play rows** | **1,399,615** (modern 1,026,950 + legacy 372,665) |
| **final metadata rows** | **21,949** (= the whitelist) |

Note the shape of the legacy funnel: it keeps 72.9% of its 4K rows on the mod rule
and then throws away 73.4% of *those* as duplicates. The legacy table is a broad,
redundant history; almost everything it has to say about these 990 players is
already in the modern table.

### 1k — kept rows by mod set

Printed by the script on every run (§11). Legacy counts are after the cross-source
deduplication, so they sum to the play table's legacy half. "As spelled" is the
source's own vocabulary — the modern table's token set, or the legacy bitmask
decoded into the same tokens; "canonical" is what survives after `MR/SD/PF` are
dropped (`NF` rows are rejected, not dropped into the canonical set).

| mod set (as spelled) | modern rows | share | legacy rows | share |
|---|---|---|---|---|
| `CL` | 898,304 | 87.47% | 339,363 | 91.06% |
| `CL+MR` | 106,922 | 10.41% | 32,329 | 8.68% |
| `CL+PF` | 9,798 | 0.95% | 506 | 0.14% |
| `CL+SD` | 9,432 | 0.92% | 305 | 0.08% |
| `CL+MR+SD` | 1,350 | 0.13% | 91 | 0.02% |
| `CL+MR+PF` | 1,144 | 0.11% | 71 | 0.02% |
| **total** | **1,026,950** | | **372,665** | |
| *(after canonicalisation)* | *`CL` 100.000%* | | *`CL` 100.000%* | |

Two spellings that used to be in this table are now gone from it: `CL+NF`
(6,694 modern / 303 legacy rows) and `CL+MR+NF` (314 / 5). Those rows are
**rejected**, not folded — see §9 D1. Their removal is the difference between this
table and the previous run's.

The canonical block is the point: six distinct spellings on each side, one
observation. The previous policy — an explicit list of five allowed subsets — kept
`CL`, `CL+MR`, `CL+NF`, `CL+SD`, `CL+PF` but silently dropped the *combinations*
`CL+MR+NF` / `CL+MR+SD` / `CL+MR+PF`, which are 2,808 modern + 167 legacy rows.
Enumerating subsets does not scale in `|neutral mods|`; canonicalising does, and it
is the only form in which the rule can be stated without a table.

The two sources disagree sharply about how common the fail mods are: `SD`/`PF`
are **7–11× rarer in the legacy table** than in the modern one. The legacy table is
populated for older plays and the modern one for recent ones, so this is a
time-trend in mod usage, not a parsing asymmetry — but it does mean the two sources
are not exchangeable populations, which matters for §9 D3's dedup logic and for any
analysis that compares "old" with "new" plays.

### 1k — what canonicalisation folds into the `CL` bucket

This is the honest accounting for the mod policy. `MR`, `SD` and `PF` do not change
the judgement weights, so they do not change ACC, and `MR` is inert. But `SD` and
`PF` are *fail* mods and they select on the outcome: canonicalising them away puts
their rows into the same bucket as plain `CL`, and the table below is what that
bucket now contains. Counts are over **all** canonical-`CL` 4K plays in both sources
(pre-dedup, 2,426,455 rows), because the play table carries no mod column and cannot
be split this way.

| mod set | rows | mean ACC | mean `loss` | `loss` range | ACC ≥ 0.9999 |
|---|---|---|---|---|---|
| `CL` | 2,135,866 | 0.98412 | −5.674 | −12.47 … −0.267 | 5.9% |
| `CL+MR` | 246,169 | 0.98274 | −5.437 | −10.91 … −0.387 | 6.5% |
| `CL+SD` | 19,169 | **0.99615** | **−7.002** | −12.41 … −0.470 | 7.9% |
| `CL+MR+SD` | 2,791 | 0.99642 | −7.329 | −11.42 … −0.634 | 12.0% |
| `CL+PF` | 20,101 | **0.99971** | **−8.512** | −12.47 … **−4.836** | 19.4% |
| `CL+MR+PF` | 2,359 | **0.99980** | **−8.798** | −10.99 … −5.919 | 32.1% |

Read the extremes:

* **`PF` is pinned to the ACC ceiling.** Perfect fails the player on any judgement
  below a 300, and in mania both `MAX` and `300` weigh 300 — so a *recorded* PF play
  has zero 200/100/50/miss by construction. Verified on the raw legacy rows: **all
  11,518 PF rows have zero `countkatu`/`count100`/`count50`/`countmiss`** (11,426
  have some 300s, 92 are all-MAX). The worst PF play in the file is ACC 0.992; the
  best is 0.999996. A range of 0.8 pp, mean 0.9997.
* **`SD` is nearly as bad**: fail-on-miss leaves zero misses but permits 200/100/50,
  so mean ACC 0.9962.

**Why this is acceptable.** The selection is not *introduced* by keeping these rows —
it is already in the `CL` bucket, which is 88% of the file. A player who drops a
note in a plain `CL` attempt restarts by hand, and the abandoned attempt leaves no
record at all (§3.1); only the clean run is stored. A `CL+PF` play is the same
observation with the restart automated. So the two are the same *kind* of data point,
which is exactly why canonicalising them together is coherent rather than merely
convenient. `SD`/`PF` also do not make the `CL` bucket *cleaner* in any way that
matters here, because `CL` is already selected to be a pass.

**What is still true.** The `loss` range is the range of "what a recorded play can
score", not of "what a player can score" — every row here is a *pass*, so the left
tail is truncated. Any statistic that depends on the extremes — a min/max, a robust
scale, a trimmed mean, a histogram binning — is affected. This is a step-1 concern,
and §8.13 explains why step 0 cannot help with it.

**`NF` is the mod that used to be folded here and is now rejected.** NoFail lets a
player *finish* a map they would otherwise have failed, so NF rows are negatively
selected: mean ACC **0.833** (1k) / **0.825** (10k), and the largest `loss` in the
raw 1k data (−0.0002, ACC 0.0002, i.e. closest to zero) is an NF play. `SD` and `PF` delete a play, so they
can only ever *remove* rows that plain `CL` already has an analogue for; `NF` *adds*
rows that have no `CL` analogue at all — a failed run that would otherwise be
invisible. That asymmetry is the whole reason for the split policy: `SD`/`PF` are
folded, `NF` is dropped. §9 D1 has the decision and §8.13 the evidence.

Removing `NF` also cleans up the file's extremes: the 1k maximum `loss` moves from
−0.0002 (ACC 0.0002) to **−0.2669** (ACC 0.234), and the minimum ACC from 0.0002 to
**0.234**. The project's earlier practice-curve estimate (ΔACC ≈ 0.0134·ln n) was
fitted before any of these rows were admitted; it is worth re-checking on this file,
since it now no longer carries the 0.6% low-ACC injection that used to be there.

### 1k — play table product

| Quantity | Value |
|---|---|
| rows | 1,399,615 |
| players | 990 |
| beatmaps | 20,655 |
| (player, beatmap) cells | 866,616 |
| `loss` mean / sd | −5.6457 / 2.1435 |
| `loss` min / max | −12.4663 / −0.2669 |
| implied mean ACC | 0.9965 |
| `playcount_cur` mean / median / max | 2.253 / 1 / 125 |
| `timestamp` range | 2013-02-14 11:10:49 … 2026-08-31 18:50:54 |
| post-conditions (§6.1) | dup 0 · contiguous `1..n` ✔ · `playcount_cur ≥ 1` ✔ · `loss < 0` ✔ |

For reference, earlier configurations of the same file:

| whitelist | rows | beatmaps | cells | `loss` mean / sd |
|---|---|---|---|---|
| `{CL}`, `{CL,MR}` | 1,376,918 | 20,623 | 856,327 | −5.6101 / 2.1345 |
| five explicit subsets | 1,403,956 | 20,655 | 867,636 | −5.6257 / 2.1514 |
| canonicalise `MR/SD/PF`, keep `NF` | 1,406,931 | 20,663 | 868,503 | −5.6294 / 2.1532 |
| **canonicalise `MR/SD/PF`, reject `NF`** | **1,399,615** | **20,655** | **866,616** | **−5.6457 / 2.1435** |

The mean and sd barely move across the four configurations — the whole spread is
~2% of rows — but the **extremes** do: the `NF`-rejecting and `NF`-keeping
configurations differ by 7,316 rows and by the maximum `loss` (−0.2669 vs −0.0002). The
comparison that matters is against the five-subset list, the policy this one
replaced: it kept `CL+NF` but silently dropped the `CL+MR+{SD,PF}` combinations.
Enumerating subsets does not scale in `|neutral mods|`; canonicalising does.

`playcount_cur` distribution (share of **rows**):

| `playcount_cur` | rows | share of rows |
|---|---|---|
| 1 | 866,616 | 61.92% |
| 2 | 224,266 | 16.02% |
| 3 | 104,375 | 7.46% |
| 4 | 60,896 | 4.35% |
| 5 | 38,542 | 2.75% |
| 6 | 25,934 | 1.85% |
| 10 | 7,214 | 0.52% |
| 20 | 853 | 0.06% |
| 50 | 30 | 0.00% |
| 100 | 3 | 0.00% |
| ≤ 3 | — | **85.40%** |

Do not confuse that with the cell-level figure. Because every cell has exactly one
row with `playcount_cur = 1`, the 866,616 rows above are also the 866,616 cells. The
real split is:

| Cells by size | cells | share of cells | rows | share of rows |
|---|---|---|---|---|
| exactly 1 record | 642,350 | **74.12%** | 642,350 | 45.89% |
| ≥ 2 records | 224,266 | 25.88% | 757,265 | 54.11% |

So three quarters of the cells contribute a single row and no order information at
all, while the other quarter carries all of the within-cell ordering.

### Beatmap metadata product (both dumps)

`data/processed/beatmap_meta_1k.csv` — 21,949 rows × 17 columns, 3.1 MB.

| Quantity | Value |
|---|---|
| rows | 21,949 (= the 4K whitelist) |
| distinct `beatmapset_id` | 6,857 |
| distinct `mapper_id` | 1,399 |
| `star` min / median / max | 0.054 / 3.142 / 10.369 |
| `star < 3` | 10,135 = **46.2%** |
| `bpm` min / median / max | 31.8 / 175.0 / 666.0 |
| `count_total` min / median / max | 2 / 1,098 / 63,907 |
| `hit_length` min / median / max (s) | 2 / 125 / 5,029 |
| `last_update` range | 2014-03-10 16:33:15 … 2026-08-30 21:31:41 |
| `approved` | `1`: 19,187 · `3`: 59 · `4`: 2,703 — no other value |
| post-conditions (§6.2) | dup 0 · sorted ✔ · `star/count_total/bpm > 0` ✔ · `hit_length ≤ total_length` ✔ · **0 played beatmaps missing** ✔ |

The `approved` distribution is the whole observed range: on this dump only `1`,
`3` and `4` occur. That is consistent with the osu-web code table
(`1 = ranked`, `3 = qualified`, `4 = loved`) and **inconsistent** with the older
stable enum, which is exactly why §6.2 refuses to map it.

Coverage against the play table:

| Quantity | 1k | 10k |
|---|---|---|
| beatmaps in the play table | 20,655 | 21,899 |
| of those present in the metadata table | **100%** | **100%** |
| whitelist beatmaps with zero plays in the play table | 1,294 | 50 |
| `star` median, played beatmaps | 3.243 | 3.144 |
| `approved` over played beatmaps | `1`: 17,952 · `3`: 19 · `4`: 2,684 | `1`: 19,166 · `3`: 30 · `4`: 2,703 |

The unplayed whitelist beatmaps are kept deliberately: the metadata table is a
property of the *dump*, not of the plays, so it stays valid if step 1 narrows the
population. The 10k pool covers 1,244 more maps than the 1k pool, which is why the
played-subset `star` median sits at the whitelist median there (3.144 vs 3.142) but
above it for the 1k pool (3.243) — the top-1000 players do not touch every 4K map,
and the ones they skip are on average easier.

Two rows in the whitelist are degenerate and worth knowing about before trusting
`count_total`: the minimum is **2 notes** and the minimum `hit_length` is **2 s**.
These are real dump entries (abandoned/stub uploads that still carry `playmode = 3`,
`diff_size = 4`), not a parsing bug. Step 1 should floor on `count_total`, not on
whitelist membership.

### 10k — legacy-only results

`2026_09_01_performance_mania_top_10000`, run with `--beatmap-src` pointed at the
1k dump (the 10k dump has no `osu_beatmaps.sql`, §2). Legacy-only: there is no
`modern` branch at all, so `--beatmap-src` affects only the whitelist and the
metadata table.

| Quantity | Value |
|---|---|
| 4K rows in `osu_scores_mania_high` | 16,899,734 |
| → dropped: not canonical-`CL` (incl. every `NF` row) | −3,540,379 |
| → **kept** | **13,359,355** (79.0%) |
| deduplication | none (no modern table) |
| **final play rows** | **13,359,355** |
| players | 9,812 |
| beatmaps | 21,899 |
| (player, beatmap) cells | 7,784,519 |
| `loss` mean / sd | −4.7708 / 1.8800 |
| `loss` min / max | −12.4663 / −0.1358 |
| implied mean ACC | 0.9915 |
| `playcount_cur` mean / median / max | 2.629 / 1 / **527** |
| `timestamp` range | 2013-02-14 11:03:44 … 2026-08-31 17:00:34 |
| cells with exactly one record | 5,641,713 = **72.47%** |
| rows with `playcount_cur = 1` | 7,784,519 = 58.27% |
| runtime | 375 s |

Mod breakdown (legacy only, post-filter; every surviving mask reduces to `CL`):

| mod set (as spelled) | rows | share |
|---|---|---|
| `CL` | 12,136,008 | 90.84% |
| `CL+MR` | 1,026,189 | 7.68% |
| `CL+PF` | 106,034 | 0.79% |
| `CL+SD` | 59,513 | 0.45% |
| `CL+MR+PF` | 21,962 | 0.16% |
| `CL+MR+SD` | 9,649 | 0.07% |
| **total** | **13,359,355** | |
| *(after canonicalisation)* | *`CL` 100.000%* | |

Two spellings from the previous run are gone here too: `CL+NF` (73,578 rows) and
`CL+MR+NF` (3,999), rejected by §9 D1.

Two things to carry forward from this table:

* **The 10k pool is not a scaled-up 1k pool.** It is easier (mean ACC 0.9915 vs
  0.9965), it has 10× the players, and its `playcount_cur` runs to 527 instead of
  125 — a much longer practice tail. Any claim calibrated on 1k must be re-checked
  here.
* **The mod-mix differs between the dumps too**: on the kept rows `SD`/`PF` make up
  1.48% of the 10k table but 1.62% of the 1k table, and the mod filter drops 21.0%
  of the 10k legacy 4K rows against 27.1% of the 1k legacy 4K rows — the wider
  player pool uses fewer `DT`/`HD` plays. Since the added mod sets are
  outcome-selected (§7), the size of that contamination is **not** the same in the
  two dumps.

---

## 8. Known limitations

Read this section before using the file.

**Scope of this section: these are limitations of the *data*, not of the *build*.**
None of items 1–13 is a gap in generating the file. Step 0's transformation is
**total and exact**: every output value is either read straight from the dump
(`player_id`, `beatmap_id`, `timestamp`), a deterministic function of the stored
judgements (`loss`), or a cumulative count over the rows that survive the
definitional filters (`playcount_cur`). Nothing is imputed, smoothed or
approximated, and no row is dropped for want of a field. What is missing is missing
from the **dump**, so these items describe the population the file is a faithful
render of — they are not reasons it cannot be built. The distinction matters most
for the failure data: the dump records only passes (§3.1), but that does not touch
`playcount_cur`, which is defined as the index among *recorded successful* plays and
never consults failures (§4.3). Even with a perfect record of every failed attempt,
`playcount_cur` would be computed identically.

Concretely, the items below split into three kinds — all three are "no":

| kind | items | why it is not a build gap |
|---|---|---|
| facts the definition never reads — failure counts, plays before the first record | 6 | `playcount_cur` counts *recorded successes* only; a perfect failure log would change nothing (§4.3) |
| properties of the **population**, not of the pipeline — lower-bound `playcount_cur`, 74% single-record cells, the long `loss` tail, non-attempt plays, outcome selection | 1, 2, 3, 4, 5, 13 | every value is exact; what it *means* is narrower than it looks |
| the input was pre-filtered or is a snapshot before we received it — modern `preserve=1 AND ranked=1`, 10k has no modern table, `star` is a 2026 value, global `playcount`, `mapper_id` namespace, unplayed whitelist maps, no per-mod star | 7, 8, 9, 10, 11, 12 | the dump is what it is; step 0 renders it exactly rather than guessing |

The third row is the only one that can bite downstream, and it is an *input* fact:
the population is smaller than "every `CL` play these players ever made". See
§2.2–§2.3 for the filters, and §9 D1 for the two **deliberate** policy exclusions
(`NF` rejected, `MR`/`SD`/`PF` folded) — those are choices, not gaps.

1. **`playcount_cur` is not a play count.** It counts recorded *successful* plays.
   The observed history is about a third of the real one (§3.2). Two consequences:
   * every value is a *lower bound* on the real number of plays up to that point,
     and the gap grows with the true count;
   * the bias is not uniform — players who fail/restart a lot have more unobserved
     plays than players who clear maps first try, so `playcount_cur` compresses
     exactly the population where practice depth differs most.
   It should be treated as an **ordinal** feature ("which attempt is this, within
   what we can see"), not as a measure of practice volume.

2. **74.1% of cells have exactly one record** (72.2% in 10k), so for most of the file
   `playcount_cur ≡ 1`. Any model that relies on this column is really relying on
   the 25.9% of cells that have ≥ 2 records. Split every evaluation by cell size.

3. **It is not comparable across cells.** `playcount_cur = 3` in a 3-record cell is
   "the player's last known play on this map"; in a 50-record cell it is "very
   early". Normalising by cell size (or using `playcount_cur / n`) is a step-1
   decision, not a step-0 one.

4. **`loss` has a long left tail.** `mean = −5.65, sd = 2.14`, driven by the
   `1/(1 − ACC)` sensitivity near a full combo: 1 pp of ACC wobble is worth
   0.10 `loss` at ACC 0.90 but 1.00 at ACC 0.99. Any loss-space statistic is
   dominated by high-ACC plays. This is deliberate (that is where the resolution
   is), but it means step-1's cleaning has to be done in loss units.

5. **Step 0 keeps plays that are not "real attempts".** A pass at ACC 0.23, a
   sight-read, a play on a 1★ map — all present. That is by design (§1), but it
   means the raw CSV is *not* a training set.

6. **Plays before the first record are invisible.** A player's first recorded play
   always gets `playcount_cur = 1`, even if they had already failed the map five
   times. There is no way to recover that from the dump (§9 D2).

7. **`timestamp` is server time, and comes from two different columns.** Modern rows
   use `scores.ended_at`, legacy rows use `osu_scores_mania_high.date`; both are the
   osu! server's clock, so any "same day" or "same hour" grouping is server-day, not
   the player's local day. The two columns also have slightly different meanings
   (score submission time vs. the legacy table's date field), which is harmless for
   ordering inside a cell — a cell's rows always come from one source after
   deduplication — but should be remembered before comparing timestamps across
   sources.

8. **The metadata table is a snapshot, not a history.** `star`, `playcount`,
   `passcount`, `approved` and `max_combo` are the values *as of the dump date*
   (2026-09-01), attached to plays that go back to 2013. A 2015 play on a map that
   was remapped in 2020 carries the 2020 star rating and the 2020 `checksum`. Only
   `last_update` reveals that the map changed at all. This is unavoidable — the
   dump has no per-date difficulty history — but it means `star` is a *current*
   covariate, not a contemporaneous one, and any model that treats it as a stable
   chart property is making an assumption step 0 cannot check.

9. **`playcount` / `passcount` are global, not pool-local.** They count every play
   on the map by every player of any skill level, not the 990 players in this dump.
   They are popularity proxies, not practice-volume measures — do not read
   `playcount` as "how much this map was played by the population under study".

10. **`mapper_id` is a user id from the same namespace as `player_id`.** They are
    not the same column and must not be joined. A mapper who is also one of the 990
    players will appear as both, which is a plausible leakage path for step 1.

11. **The metadata table contains maps that the play table does not** (1,294 in 1k,
    50 in 10k), and two of them are degenerate stubs (2 notes, 2 s). Filtering on
    `count_total` is step-1 work; step 0 emits them.

12. **Per-mod difficulty is not in the table.** Only the no-mod `star` is. Since
    step 0 keeps only 1.0× plays (§9 D1) that is sufficient *for this population*,
    but it means the file cannot be reused for a rate-modelled variant without
    going back to `osu_beatmap_difficulty`. The full per-mod ratings already exist
    in `data/legacy/interim/beatmap_difficulty_4k.csv`.

13. **The population is selected on the outcome, and the play table cannot say by
    how much.** This is the sharpest limitation in this document.

    Every row in the file is a *pass* (§3.1) — there is no such thing as a failed
    attempt in the dump. So the population is already conditioned on "the player
    got to the end", and the only question the mod policy can change is *how* the
    surviving attempts were selected.

    Under the mod policy (§9 D1) `MR`/`SD`/`PF` are canonicalised into the `CL`
    bucket and `NF` is rejected. They select on the outcome in three different ways:

    * `SD` and `PF` **delete** the play unless it is clean. A recorded `CL+PF` play
      has no 200/100/50/miss at all — verified on all 11,518 PF rows — so its ACC is
      pinned between 0.992 and 1.000 (§7). A recorded `CL+SD` play has no miss but
      may have 200s. But this only *automates* a manual restart, which leaves no
      record either (§3.1), so it is folded rather than dropped.
    * `NF` does the opposite: it **keeps** a play that would otherwise have been
      abandoned, so NF rows are the low-ACC tail (mean ACC 0.83 against 0.98 for
      `CL`). There is no `CL` analogue to fold them into, so they are excluded.
    * `MR` is inert.

    **Why the `SD`/`PF` folding is not a reason to change the policy.** The selection
    is not created by those rows; it is already present in the `CL` bucket, which is
    88% of the file. A player who drops a note in a plain `CL` attempt restarts by
    hand, and the abandoned attempt leaves no record at all (§3.1) — only the clean
    run is stored. A `CL+PF` play is that same observation with the restart
    automated, so folding it into `CL` groups like with like. Keeping it *separate*
    would suggest a distinction the data does not support.

    **What is still true, and what step 1 must handle.**

    a. **`loss`'s range is the range of "what a recorded play can score", not "what
       a player can score".** Every row is a pass, so the left tail is truncated by
       construction: the file's maximum `loss` is −0.2669 (ACC 0.234) and its
       minimum −12.4663 (ACC ≈ 0.999996). Any statistic that depends on the extremes
       — a min/max, a robust scale, a trimmed mean, a histogram binning — moves
       without any change in player behaviour. Cleaning is done in `loss` units
       (§8.4), so this is squarely a step-1 decision.

    b. **`NF` is excluded at the source.** The excluded set is small but strongly
       left-tail-concentrated. On the deduplicated rows (`scripts/probe_nf_acc.py`),
       NF would be **0.52%** of the 1k play table and **0.58%** of the 10k one, yet
       it accounts for:

       | slice | 1k: NF share | 10k: NF share | vs base rate |
       |---|---|---|---|
       | ACC ≤ 0.10 | **100%** (48 of 48) | **100%** (394 of 394) | ~180× |
       | ACC ≤ 0.40 | 81.0% (252 of 311) | 88.6% (2,121 of 2,394) | ~155× |
       | ACC ≤ 0.63 | 62.1% (683 of 1,100) | 67.3% (7,663 of 11,392) | ~117× |
       | ACC ≤ 0.90 | 15.7% (4,205 of 26,739) | 12.3% (47,295 of 385,582) | 21–30× |
       | ACC ≤ 0.99 | 1.16% | 0.93% | 1.6–2.2× |

       So the tail NF creates is real and it is the *only* source of it: **without NF
       there is no row below ACC 0.10 at all**, and the minimum ACC in the file is
       now 0.234 (1k) / 0.127 (10k) instead of 0.0002 / 0.00003. But the effect is
       confined to the deep tail. By ACC ≤ 0.99 NF is down to 1% of the slice, and
       **42.5% (1k) / 39.0% (10k) of NF rows are at ACC ≥ 0.90** — i.e. the majority
       of NF plays are ordinary-quality plays where the player simply had the mod on,
       not rescued failures. NF's mode sits in the 0.90–0.95 band, not in the tail.
       The practical reading: excluding NF matters for anything that looks at the
       extreme left tail (min/max, robust scale, `loss`-space trimming) and is a
       rounding error for everything else. §9 D1 has the decision and the rejected
       alternative.

       The project's earlier practice-curve estimate (ΔACC ≈ 0.0134·ln n) was fitted
       on a file that did *not* carry these rows, so it is unaffected by this
       decision — but the file it is re-checked against is now the clean one.

    c. **The play table carries no mod column, so the exclusion is not reversible
       from the output.** That is by design (§9 D5, and the spec is explicit about
       the five columns). Recovering the NF rows means re-scanning the dump *and*
       lifting the rejection (`REJECTED_CHART_NEUTRAL_MODS` emptied, then
       `--mod-whitelist CL,CL+NF`) — a deliberate code change, not a flag. §13.10
       discusses why a `neutral_mods` column cannot bring them back on its own.

---

## 9. Design decisions

Each decision is recorded with the alternative that was rejected and why. All are
switchable (§10).

### D1 — mod policy: canonicalise `MR`/`SD`/`PF`, reject `NF` and everything else

* **Chosen.** Two stages, in this order:

  1. **Canonicalise.** Drop the chart-neutral mods from the row's mod set, so they
     read as "not enabled". Default list: `MR`, `SD`, `PF`.
  2. **Whitelist.** Keep the row iff what remains is exactly `CL` (configurable via
     `--mod-whitelist`).

  | mod | meaning | treatment | why |
  |---|---|---|---|
  | `MR` | Mirror | **canonicalise** | mirrors the columns; no difficulty change at all |
  | `SD` | SuddenDeath | **canonicalise** | fails on a miss; no judgement change, and only automates a manual restart |
  | `PF` | Perfect | **canonicalise** | fails on a non-300; no judgement change, and only automates a manual restart |
  | `NF` | NoFail | **reject** | chart-neutral but *not* observation-neutral — see below |

  Nothing else survives. In particular `DT/NC/HT` (different rate ⇒ different chart),
  `HD/FL/EZ/HR/FI/RD/IN/…` (different read/play semantics) and ScoreV2 stay out.

* **Why canonicalise rather than enumerate.** An earlier form of this decision was a
  list of five allowed token sets, `{CL} {CL,MR} {CL,NF} {CL,SD} {CL,PF}`. It is the
  same rule for the common cases but it is the wrong shape, for three reasons:

  1. **It does not compose.** `CL+MR+SD` is a real, chart-equivalent 1.0× play, and
     the five-set list silently dropped it (1,350 modern + 91 legacy rows in 1k, plus
     `CL+MR+PF` 1,144 + 71). Any fix means adding `{CL,MR,SD}`, `{CL,MR,PF}`, … — the
     list grows as 2^|neutral| and every omission is invisible.
  2. **It states the policy twice.** With an enumerated list, "MR is chart-neutral" is
     encoded in several places; adding a mod to `--neutral-mods` requires editing
     `--mod-whitelist` too, and forgetting to is a silent bug. Canonicalising states
     it once, and the two flags stay orthogonal.
  3. **It obscures the claim being made.** The list says "these are the acceptable
     combinations". The intent is "the chart is what matters, and these mods do not
     touch it". The second is what downstream code needs to know.

* **Why `MR`/`SD`/`PF` are genuinely the same observation.** None of them touches the
  judgement weights, so none of them touches ACC — and `loss` is a function of ACC
  alone (§4.1). `beatmap_id` therefore still identifies the chart, no `rate` column is
  needed, and the earlier project's `chart_id = beatmap_id·4 + rate` does not appear
  here.

  For `SD`/`PF` there is a stronger argument. They do not merely leave ACC alone; they
  only *automate* what a player does by hand — restart when the run is not clean. A
  hand-restarted attempt leaves **no record at all** in the dump (§3.1), so the
  plain-`CL` rows are already selected for "the player got to the end cleanly enough to
  keep". A `CL+PF` row is the same kind of observation with the restart mechanised, not
  a new kind. Folding it into `CL` groups like with like; keeping it separate would
  assert a distinction the data cannot support.

* **Why `NF` is rejected instead.** `NF` is chart-neutral by the same test — it does
  not touch the judgement weights — but it is the one mod that is not
  *observation*-neutral, and the difference is measurable:

  | | `SD`/`PF` | `NF` |
  |---|---|---|
  | effect on which plays are recorded | only **removes** plays (a non-clean run is not stored) | only **adds** plays (a run that would have been abandoned is completed and stored) |
  | does the `CL` bucket already contain this observation? | **yes** — the manual restart is unrecorded, so `CL` is selected the same way | **no** — an abandoned run leaves nothing, so there is no `CL` analogue |
  | mean ACC of its rows | 0.996 (`SD`) / 0.9997 (`PF`) | **0.83** |
  | rows in 1k / 10k | 19,169 / 165,547 (`SD`), 20,101 / 127,996 (`PF`) | 7,316 / 77,577 |

  The `NF` tail is real and it is the *only* source of it: on the deduplicated rows
  **100% of the ACC ≤ 0.10 rows are `NF`** (48 of 48 in 1k, 394 of 394 in 10k), and
  without `NF` the file's minimum ACC is 0.234 (1k) / 0.127 (10k) instead of 0.0002 /
  0.00003. The full breakdown is in §8.13b.

  So `NF` rows are a **different population**, not a relabelling of the `CL` bucket,
  and step 0's job is to define the population. Rather than fold them in and leave
  step 1 to undo it, they are excluded at the source. The cost is explicit and small:
  7,316 rows (0.52%) in 1k and 77,577 (0.58%) in 10k.

  Note that this is a **policy** choice, not a claim that `NF` plays are invalid. It is
  the one place step 0 filters on something other than the chart definition, and the
  reason is that the alternative (canonicalise it, then let step 1 try to identify it)
  is not available: the play table carries no mod column (§8.13c), so a folded-in `NF`
  row would be permanently indistinguishable.

* **The legacy encoding trap.** `osu_scores_mania_high` stores mods as a bitmask,
  and osu!stable **always sets the SuddenDeath bit together with the Perfect
  bit**. So Perfect is `SD|PF` = 16416, not 16384:

  | legacy mask | rows (1k, 4K) | decoded as | canonical | kept? |
  |---|---|---|---|---|
  | `0` | 1,237,562 | `CL` | `CL` | ✔ |
  | `1073741824` (`MR`) | 139,247 | `CL+MR` | `CL` | ✔ |
  | `16416` (`SD\|PF`) | 10,303 | `CL+PF` | `CL` | ✔ |
  | `32` (`SD`) | 9,737 | `CL+SD` | `CL` | ✔ |
  | `1` (`NF`) | 6,995 | `CL+NF` | `CL+NF` | ✘ |
  | `16384` (`PF` alone) | **0** | — never occurs | — | — |

  The decoder folds `SD` away when `PF` is present. Under the default policy this is
  *subsumed* by canonicalisation (`{SD,PF}` and `{PF}` both reduce to nothing), but it
  is kept because it is a fact about the bitmask rather than about the policy: with
  `--neutral-mods MR,SD` it is the difference between reading 16416 as `CL+PF` (right)
  and as `CL+SD+PF` (wrong).

* **`CL` is a real marker in the modern table, not decoration.** The modern table does
  *not* always spell out `CL` — 100,163 of its 1,531,094 4K rows (6.5%) have no `CL`
  token, and **every one of them has `legacy_score_id = NULL`**, i.e. it is a lazer
  ScoreV2 score rather than a stable one. Conversely only 146 of the 1,430,931
  `CL`-bearing rows lack a `legacy_score_id`. So `CL` present ⟺ the score came
  through the classic pipeline ⟺ the legacy table would have stored it, and requiring
  `CL` is exactly the rule that makes the modern half comparable with the legacy half.
  This also means **modern `mods=[]` is not the same play as legacy `enabled_mods=0`**:
  the former is lazer-without-Classic (dropped), the latter is stable-with-no-mods
  (spelled `CL` in the modern table, kept). See `scripts/probe_modern_cl.py`.

* **One policy, two encodings.** `--mod-whitelist` + `--neutral-mods` are the single
  source of truth, and both sources run through the *same* two stages. They differ
  only in how the mods are read: the modern table spells them out as JSON acronyms,
  the legacy table packs them into a bitmask. Two independent filters is how the
  halves drift apart. `--legacy-mod-bits` maps acronym → bit; the legacy mask is
  decoded into the same token vocabulary and then canonicalised identically.

  `REJECTED_CHART_NEUTRAL_MODS` (currently just `NF`) is a third, non-configurable
  list: it exists so that "chart-neutral but deliberately excluded" is a named concept
  rather than an absence, and the script refuses to start if it overlaps either of the
  other two. Without that guard, re-adding `NF` to `--neutral-mods` would silently
  undo this decision — the exact class of bug this document keeps recording.

* **Rejected alternative — a bitmask union (`mask & ~(SD|PF|MR) == 0`).** Tempting
  for the legacy side: it is one comparison and it *is* the canonicalisation rule
  expressed directly in bits. It is still rejected, for a different reason than
  before: it **fails open**. Any bit the mask does not name is silently ignored, so
  `64` (DoubleTime) would decode to an empty token set and be mistaken for a plain
  `CL` play. The implemented decoder returns `None` for a mask carrying an unnamed
  bit, and `None` is a rejection. (This is not hypothetical — the first implementation
  of this change had exactly that bug and kept 882,894 legacy rows instead of 372,665,
  i.e. it let every DT/HD/HT play through. The per-mod breakdown printed by §11 is what
  caught it.) A second reason: the union is expressible only in bit space, so the
  modern half would still need the token-set form, and the two halves would no longer
  share one test.

* **Rejected alternative — derive the legacy bit table from `--mod-whitelist`
  alone.** Would require the bits to be inferred from acronyms, which puts a guess
  about stable's encoding into the code. Keeping `--legacy-mod-bits` separate and
  explicit means a wrong bit assignment is a visible config error, not a silent
  semantic one.

* **Rejected alternative — canonicalise `NF` too (the previous policy).** This was the
  behaviour until the `NF` breakdown in §8.13b was measured. The argument for it is
  that `NF` is chart-neutral and step 0 should not filter (§1). The argument against,
  which won, is that `NF` is the only mod that *adds* rows with no `CL` analogue, so
  canonicalising it does not group like with like — it merges two populations and then
  discards the only column that could separate them again. `NF`'s effect is confined to
  the extreme left tail, which is exactly where a learning-curve fit is most sensitive,
  so the cost of getting it wrong is larger than the 0.5% row count suggests.

### D2 — `playcount_cur` counts recorded plays only

* **Chosen.** `playcount_cur` = 1-based index within the cell's recorded plays.
  No play-count table, no estimation of unobserved plays.
* **Rejected alternative A — join `osu_user_beatmap_playcount` and spread the
  unobserved plays.** The pair-level counter gives the real total `A` while the
  records give `n`, so `A − n` plays happened without leaving a row. But the dump
  contains **no information about where those plays sit**: there is no per-play
  position anywhere, and `osu_beatmap_failtimes` is one aggregate distribution per
  *beatmap* summed over *all* players (`type ∈ {fail, exit}` × `p1..p100`;
  means 0.371 for fail, 0.319 for exit, 0.331 pooled). Spreading `A − n` over the
  gaps between records would therefore be a pure assumption, and a second one on
  top of it if the plays were to be weighted by how far they got. Rejected: the
  column would look like a measurement while being a model.
* **Rejected alternative B — weight unobserved plays by their completion fraction.**
  Same objection, one layer worse: it needs a per-play fail position that does not
  exist in the dump.
* **Cost of the choice.** `playcount_cur` is a lower bound and is biased in a way
  that correlates with player behaviour (§8.1). This is accepted deliberately in
  exchange for a column that is exactly what it claims to be.
* **If it is ever revisited**, `A` is available from `osu_user_beatmap_playcount`
  (3,782,234 rows, covering 21,771 of 21,949 4K maps and 998 of 990 users) and the
  failtimes means above are recorded here for that purpose. A useful sanity check
  first: recompute using `A` and compare how much the downstream conclusions move.

### D3 — deduplicate modern against legacy

* **Chosen.** Drop legacy rows whose `score_id` appears in the modern table's
  `legacy_score_id` column.
* **Why.** For the 1k dump the modern table is effectively a superset of the legacy
  table (§5 step 4). Without deduplication, 1,399,615 plays would be counted as
  2,426,455 rows and every cell's `playcount_cur` would run to roughly twice its
  correct maximum — a silent corruption of the only feature in the file. The dedup
  removes 1,026,840 legacy rows, i.e. **73.4% of the legacy half**; the legacy table is
  kept anyway because it is the only source for the 10k dump and because it reaches
  further back in time.
* **Note for the 10k dump.** There is no modern table, so this is a no-op and the
  legacy table is the only source. The logic must be present and skipped, not
  removed.

### D4 — step 0 does no cell-level filtering

* **Chosen.** Keep everything that is a valid observation.
* **Why.** Filtering is step 1's job (§1). In particular the ACC floor ("junk
  plays"), the `playcount ≤ 2` floor and the `n_scores ≤ 2` floor are all
  *selection rules* that change what question the model answers; they must be
  ablatable against a fixed step-0 population.

### D5 — the metadata table is a separate file, keyed by `beatmap_id`

* **Chosen.** A second CSV, one row per whitelisted beatmap, containing 17 columns
  read straight from `osu_beatmaps`. The play table stays five columns.
* **Rejected alternative A — append the columns to the play table.** The play table
  is 1.40 M rows; duplicating 12 chart columns onto every row would multiply the
  file by ~4× for zero information, and it would break the property that makes the
  play table useful: it is a *minimal* record of what happened, so that a reader can
  audit it in one pass. It would also put step 0 in violation of its own §1
  non-goal ("no metadata").
* **Rejected alternative B — join `osu_beatmapsets` too.** Artist / title / tags /
  `favourite_count` / `submit_date` would double the width for fields nothing in the
  project currently uses, and they add free-text unicode into the CSV. `beatmapset_id`
  is carried, so the join stays one step away when a modelling question needs it.
* **Rejected alternative C — per-mod difficulty from `osu_beatmap_difficulty` /
  `_attribs`.** Step 0 keeps only 1.0× plays, so the only star rating that can ever
  be used is the no-mod one, which `osu_beatmaps.difficultyrating` already is. The
  per-mod tables are ~1.0 GB and 1.2 GB and would turn a 3 MB lookup into a long
  table. `data/legacy/interim/beatmap_difficulty_4k.csv` already holds them if a
  rate-modelled variant is ever built.
* **Rejected alternative D — aggregate `osu_beatmap_failtimes` into the table.**
  Tempting as "chart metadata", but it is a per-beatmap aggregate over all players,
  and §9 D2 already established that it cannot be attributed to a specific play.
  Putting it in the metadata table would invite exactly the per-play interpretation
  D2 rejected.
* **Rejected alternative E — derive features (`star / count_total`, note density,
  rank tiers, star bins).** Derivation is step-1 work; step 0 stores what the dump
  stores so that the derivations stay ablatable.
* **`approved` is emitted raw, not mapped.** The dump's integer status has no
  authoritative code table attached to it, and the observed range on this dump
  (`1 / 3 / 4`) matches osu-web but not the older stable enum. Mapping it in step 0
  would bake a guess into the data. The distribution is recorded in §7 so step 1 can
  decide with the numbers in hand.
* **The whitelist, not the play table, defines the rows.** 1,294 whitelisted
  beatmaps (1k) / 50 (10k) have no plays. Keeping them means the metadata table is a
  property of the dump and stays correct if step 1 narrows the population; it also
  means the only cross-file invariant is one-directional (§6.2 check 4).

---

## 10. Configuration

Nothing is hard-coded. The reference implementation exposes:

| Flag | Default | Meaning |
|---|---|---|
| `--dump-dir` | required | directory holding the score tables |
| `--beatmap-src` | `--dump-dir` | directory holding `osu_beatmaps.sql`; the 10k dump has none, so it points at the 1k dump |
| `--tag` | `1k` | label used for the output filenames (`1k` / `10k`) |
| `--out` | `data/processed/step0_{tag}.csv` | play table; a `.parquet` suffix writes parquet |
| `--meta-out` | `data/processed/beatmap_meta_{tag}.csv` | beatmap metadata CSV |
| `--no-meta` | off | skip the metadata table entirely |
| `--work-dir` | `data/interim/step0_{tag}/` | streaming scratch space |
| `--mod-whitelist` | `CL` | allowed mod token sets **after** canonicalisation — the policy (§9 D1) |
| `--neutral-mods` | `MR,SD,PF` | mods that do not change the chart; canonicalised to "off" before the whitelist test (§9 D1). `NF` is deliberately **not** here — it is in the non-configurable `REJECTED_CHART_NEUTRAL_MODS` |
| `--legacy-mod-bits` | `NF=1,SD=32,PF=16384,MR=1073741824` | acronym→bit map for decoding the legacy mask |
| `--mode` | `3` | `osu_beatmaps.playmode` whitelist (3 = mania) |
| `--keys` | `4` | `osu_beatmaps.diff_size` whitelist (4 = 4K) |
| `--count-dtype` | `int32` | output precision for `playcount_cur` |
| `--float-dtype` | `float32` | output precision for `loss` |
| `--keep-raw` | off | keep the intermediate parquet |

Plays with zero judgements (`total == 0`) are dropped unconditionally.

`--mode` and `--keys` are the same whitelist that feeds the metadata table, so a
run for a different keys mode produces a consistent pair of files — nothing in the
metadata projection is 4K-specific.

`--mod-whitelist` and `--neutral-mods` are the policy; `--legacy-mod-bits` is an
encoding detail of one table. Deriving the bit table from the acronyms would hide a
guess about stable's bit layout inside the code (§9 D1). A neutral mod named in
`--mod-whitelist` is a configuration error and the script refuses to run: it could
never match, because canonicalisation removes it before the test.

The two policy flags are orthogonal, which is the point of §9 D1. The `NF` decision
is deliberately *not* reachable through them: `NF` sits in the non-configurable
`REJECTED_CHART_NEUTRAL_MODS`, so an attempt to whitelist it back is refused at
startup rather than silently honoured:

```bash
# refused by design — NF overlaps the policy:
python scripts/step0_preprocess.py --dump-dir ... --tag 1k \
    --neutral-mods MR,SD,PF --mod-whitelist CL,CL+NF
# -> SystemExit: REJECTED_CHART_NEUTRAL_MODS overlaps the policy: ['NF']
```

Invocations (these are the two commands that produced the §7 numbers):

```bash
# 1k: modern + legacy, CSV play table, plus the beatmap metadata
python scripts/step0_preprocess.py \
    --dump-dir data/raw/extracted/2026_09_01_performance_mania_top_1000 \
    --tag 1k

# 10k: legacy only, no osu_beatmaps.sql and no scores.sql in that dump
python scripts/step0_preprocess.py \
    --dump-dir data/raw/extracted_10k/2026_09_01_performance_mania_top_10000 \
    --beatmap-src data/raw/extracted/2026_09_01_performance_mania_top_1000 \
    --tag 10k --out data/processed/step0_10k.parquet

# play table only (skip the metadata projection)
python scripts/step0_preprocess.py --dump-dir ... --tag 1k --no-meta
```

The 10k run uses `--out ....parquet` because 13.4 M rows as CSV would be ~700 MB
for no benefit; the 1k run keeps CSV so the file stays greppable and diffable.

---

## 11. Implementation

`scripts/step0_preprocess.py` — validated end-to-end on **both** dumps; the numbers
in §7 come from exactly this code. It reads the raw dump SQL directly (no dependency
on `data/legacy/interim/*.csv`), reuses the low-level SQL readers from
`scripts/legacy/parse_dump.py` rather than re-implementing MySQL parsing, and streams both
score tables through a parquet scratch file so the 10k legacy table (16.9 M rows)
never has to fit in memory as Python objects.

`scripts/legacy/step0_preprocess_10k.py` is the superseded predecessor of this script: it
hard-coded the whitelist source to `data/legacy/interim/beatmaps.csv` and only ever wrote
parquet. It is kept only for reference and should not be used — its whitelist source
is a derived CSV rather than the dump, and it has none of the mod policy of §9 D1.

The script is the source of truth. The parts below are the ones worth reading.

**Judgement mapping and the loss formula.** `countgeki → MAX(320)` and
`countkatu → 200` are the two legacy columns whose names do not match their classic
weight — the single easiest thing in step 0 to get wrong:

```python
# classic weight -> key in the modern JSON `statistics` object
CLASSIC = (("perfect", 300), ("great", 300), ("good", 200), ("ok", 100), ("meh", 50))

# legacy row layout: 6 count50  7 count100  8 count300  9 countmiss
#                   10 countgeki (MAX)  11 countkatu (200)  13 enabled_mods  14 date
LEGACY_TOTAL = (6, 7, 8, 9, 10, 11)
LEGACY_NUM = ((10, 300), (8, 300), (11, 200), (7, 100), (6, 50))

def loss_of(acc_num, total):
    acc = (acc_num + 250.0) / (300.0 * (total + 1.0))   # pseudo-250 judgement
    return float(np.log1p(-acc))
```

**Two passes over the sources.** The modern pass has to complete before the legacy
pass can be deduplicated against it. For the 10k dump the modern pass yields nothing,
the dedup set stays empty, and the whole step degenerates to "legacy only" — intended
behaviour, not a special case:

```python
def build_raw(dump, ids, allow, bits, scratch):
    drop = set()
    for row in iter_modern(dump, ids, allow):            # (..., legacy_score_id)
        ...
        if row[5] >= 0:
            drop.add(row[5])                             # -> the dedup set
    ...
    for row in iter_legacy(dump, ids, allow, bits, drop): # skips any score_id in drop
        ...
```

**The mod decode is fail-closed.** The single easiest way to get this wrong is to
decode only the bits you know about, because then every unknown bit is invisible —
`mask=64` (DoubleTime) would decode to an empty token set and be accepted as a plain
`CL` play. Returning `None` for an unnamed bit, and treating `None` as a rejection,
is what makes the filter safe:

```python
def decode_legacy_mods(mask, bits):
    known = 0
    for bit, _ in bits:
        known |= bit
    if mask & ~known:
        return None                      # unnamed bit -> reject, not ignore
    toks = {name for bit, name in bits if mask & bit}
    if "PF" in toks:
        toks.discard("SD")               # stable sets SD together with PF
    return frozenset(toks)

# legacy rows are classic by construction -> 'CL' is implicit
mods = cache[mask]
if mods is None or frozenset({"CL"}) | mods not in allow:
    continue
```

Note the cache holds the **decoded token set**, not the boolean. Caching the boolean
alone leaves `mods` stale across masks — which is the second bug this change had, and
it surfaced as `TypeError: unsupported operand type(s) for |: 'frozenset' and
'NoneType'` on the first interleaving of an allowed and a disallowed mask.

**The per-mod breakdown is printed, not just computed.** It is the only cheap check
that the two encodings agree, and it is what caught the fail-open bug above:

```
[step0] modern kept by mod set (1031150 rows):
          CL                898304  87.117%
          CL+MR             106922  10.369%
          ...
[step0] legacy kept by mod set (372806 rows):
          CL                339363  91.029%
          ...
```

**The sort *is* the definition.** `playcount_cur` comes straight out of `cumcount()`
after the sort, so changing the sort key changes the column:

```python
plays = plays.sort_values(["user_id", "beatmap_id", "ts", "sid"], kind="stable")
plays["playcount_cur"] = (plays.groupby(["user_id", "beatmap_id"], sort=False)
                          .cumcount() + 1)
```

**The metadata table is a projection table, not code.** Every output column is a
`(name, index)` pair into an `osu_beatmaps` row, so adding or dropping a field is a
one-line change and the source index can never drift out of sync with the name:

```python
# output column -> index in an osu_beatmaps row
META_FIELDS = (
    ("beatmap_id", 0), ("beatmapset_id", 1), ("mapper_id", 2), ("version", 5),
    ("star", 19), ("bpm", 27), ("max_combo", 20), ("count_total", 8),
    ("diff_overall", 14), ("diff_drain", 12), ("hit_length", 7),
    ("total_length", 6), ("playcount", 21), ("passcount", 22),
    ("approved", 17), ("last_update", 18), ("checksum", 4),
)
META_INT   = ("beatmap_id", "beatmapset_id", "mapper_id", "max_combo", "count_total",
              "hit_length", "total_length", "playcount", "passcount", "approved")
META_FLOAT = ("star", "bpm", "diff_overall", "diff_drain")
META_STR   = ("version", "last_update", "checksum")
```

**One pass produces both the whitelist and the table.** They are built together so
they cannot disagree about which beatmaps exist — that is what makes the §6.2 check-4
invariant structural rather than hopeful:

```python
def read_beatmaps(dump, mode, keys):
    rows = []
    for line in iter_insert_lines(dump / "osu_beatmaps.sql"):
        for r in parse_careful(line):          # handles commas inside filenames
            if len(r) < 29:                    # cannot project a short row
                continue
            if r[16] != mode or r[13] != keys: # playmode / diff_size
                continue
            rows.append([r[i] for _, i in META_FIELDS])
    ...
    return ids, df.sort_values("beatmap_id", kind="stable").reset_index(drop=True)
```

`parse_careful` (not `rows_simple`) is required here: `filename` and `version`
contain commas — 187 whitelisted maps have a comma in `version` — so a naive
`split(",")` would shift every column after index 3.

**Five diagnostic scripts back the claims in §2, §7, §8.13 and §9 D1.** They are not
part of the pipeline — they read the raw dump and print, they do not write anything:

* `scripts/probe_mod_whitelist.py` — every distinct `enabled_mods` value and every
  distinct modern token set among 4K rows. This is where the `SD|PF = 16416` finding
  came from, and it is the only way to see the *rejected* half of the mod rule.
* `scripts/probe_modern_cl.py` — for every modern 4K row, whether it carries `CL` and
  whether it has a `legacy_score_id`. This is what establishes that `CL` ⟺ the score
  is in the legacy table (100,163 rows without `CL`, all with `legacy_score_id = NULL`)
  and therefore that requiring `CL` is the right comparability rule (§9 D1).
* `scripts/probe_mod_loss.py` — the `loss` distribution per kept mod set, for both
  sources. This is where the §7 canonicalisation table came from, and it re-derives
  the policy flags from the module constants so it cannot drift.
* `scripts/probe_nf_acc.py` — the ACC distribution of NF rows versus non-NF rows, on
  the deduplicated set. This is where the §8.13b table came from. Note the sign trap
  it exists to avoid: `loss` *decreases* with ACC, so `loss ≥ log(1−x)` is the same
  set as `ACC ≤ x`, and a "high ACC" threshold is a *low* `loss` threshold.
* the inline probes recorded in §2.2 — per-key row counts, the `preserve`/`ranked`
  cross-tab, and the `approved` breakdown — are the evidence for the claim that the
  modern table is a play history rather than a leaderboard.

**Self-check.** The script asserts the §6.1 and §6.2 post-conditions and prints
them, so a bad run is visible without a separate verification pass:

```
[step0] neutral mods (canonicalised to 'off'): MR PF SD   [legacy mask 0x40004020]
[step0] mod whitelist (post-canonical): CL
[step0] chart-neutral but REJECTED (not canonicalised): NF
[step0] legacy bit decode: NF=1 SD=32 PF=16384 MR=1073741824
[step0] 3 / 4K beatmaps: 21949 (scanned 235061 rows, 0 too short, from ...)
[step0] meta rows=21949 dup(beatmap_id)=0 star 0.054..10.369 median 3.142
[step0] meta approved counts: 1:19187 3:59 4:2703
[step0] modern: kept 1026950 of 1531094 4K rows (dedup set 1026844)  59s
[step0] legacy: kept 372665 of 1918989 4K rows (dropped 1026840 dup)  86s
[step0] modern kept, as the source spelled it (1026950 rows):
          CL                898304  87.473%
          CL+MR             106922  10.412%
          ...
[step0] modern kept, after canonicalisation:
          CL               1026950 100.000%
[step0] rows=1399615 players=990 beatmaps=20655 cells=866616
[step0] dup(playcount_cur)=0 | contiguous 1..n=True | pc>=1=True | loss<0=True
[step0] timestamp 2013-02-14 11:10:49 .. 2026-08-31 18:50:54
[step0] loss mean -5.6457 sd 2.1435 | playcount_cur mean 2.253 max 125
[step0] meta dup=0 | sorted=True | star/count/bpm>0=True | hit<=total=True |
        play-beatmaps missing from meta=0
```

The last line is the cross-file invariant from §6.2 check 4: the number of beatmaps
in the play table that have no metadata row must be 0. The `kept` lines are the other
invariant worth watching: `modern kept` must be the sum of the modern per-mod counts,
and `legacy kept` must equal `legacy_4k − mod_dropped − deduped`. The canonical block
must be a single row, `CL`, at 100.000% — that is check W5, and it is the cheapest
possible signal that the two-stage policy is being applied to both sources.

**Runtime.** 1k: ≈ 75 s total standalone (118 s on the run above, which overlapped a
10k run and contended for disk). The metadata projection is ≈ 10–15 s of that — one
extra linear pass over the 56 MB `osu_beatmaps.sql`. 10k: **375 s**, dominated by the
2.44 GB / 16.9 M-row legacy table; the 10k dump has no modern table, so the modern
pass is a no-op and the dedup set stays empty.

## 12. Verification checklist

Run after every regeneration of either file.

**Play table** (`data/processed/step0_{tag}.csv` / `.parquet`):

| # | Check | 1k expected | 10k expected |
|---|---|---|---|
| 1 | row count | 1,399,615 | 13,359,355 |
| 2 | distinct `(player_id, beatmap_id)` cells | 866,616 | 7,784,519 |
| 3 | `playcount_cur` is exactly `1..n` inside each cell | always | always |
| 4 | no duplicate `(player_id, beatmap_id, playcount_cur)` | 0 | 0 |
| 5 | `playcount_cur >= 1`, `loss < 0` | always | always |
| 6 | `timestamp` non-decreasing inside each cell, and parseable as `%Y-%m-%d %H:%M:%S` | always | always |
| 7 | `timestamp` range | 2013-02-14 11:10:49 … 2026-08-31 18:50:54 | 2013-02-14 11:03:44 … 2026-08-31 17:00:34 |
| 8 | no `F`-rank rows leaked in | legacy `rank` never `F` | same |
| 9 | no DT/HT/HD/EZ/FL/ScoreV2 rows leaked in | `CL` only, post-canonical | same |
| 10 | duplicate plays across sources | 0 (dedup by `legacy_score_id`) | n/a (no modern table) |
| 11 | `loss` recomputation spot-check | modern row `2057059876` → ACC 0.87674, not the stored 0.868409 | — |
| 12 | cells with exactly one row | 642,350 = 74.12% | 5,641,713 = 72.47% |
| 13 | header is exactly the five columns, in order | `player_id,beatmap_id,timestamp,playcount_cur,loss` | same |
| 14 | **`loss` maximum** | **−0.2669** | **−0.1358** |
| 15 | **`playcount_cur` maximum** | 125 | **527** |
| 16 | `loss` mean / sd | −5.6457 / 2.1435 | −4.7708 / 1.8800 |
| 17 | players / beatmaps | 990 / 20,655 | 9,812 / 21,899 |

Checks 1–7 and 13 are printed by the script at the end of every run (3–6 are
asserted outright); 8–12 and 14–17 need a separate look.

Checks 14 and 15 are the two numbers that moved most across the mod-policy changes,
so they are the cheapest early warning that the policy has changed again. Check 14 in
particular: with `NF` rejected the 1k maximum `loss` is −0.2669 (ACC 0.234) and the
10k maximum is −0.1358 (ACC 0.127); a value near zero means `NF` rows have leaked
back in (§8.13b, §9 D1).

**Mod funnel** (printed every run, §11):

| # | Check | 1k expected |
|---|---|---|
| W1 | modern kept = sum of the modern per-mod counts | 1,026,950 |
| W2 | legacy kept = `legacy_4k − mod_dropped − deduped` | 372,665 = 1,918,989 − 519,484 − 1,026,840 |
| W3 | modern per-mod counts (as spelled) | `CL` 898,304 · `CL+MR` 106,922 · `CL+PF` 9,798 · `CL+SD` 9,432 · `CL+MR+SD` 1,350 · `CL+MR+PF` 1,144 |
| W4 | legacy per-mod counts (post-dedup, as spelled) | `CL` 339,363 · `CL+MR` 32,329 · `CL+PF` 506 · `CL+SD` 305 · `CL+MR+SD` 91 · `CL+MR+PF` 71 |
| W5 | **canonical breakdown is exactly one set, `CL`, at 100.000%, on both sides** | always |
| W6 | no set containing `DT`/`NC`/`HT`/`HD`/`FL`/`EZ`/`HR`/`IN`/`V2` appears in any breakdown | always |

W1–W6 are printed, not asserted. They are the only check that the legacy bitmask
decode and the modern token filter agree, and W2 is the one that would have caught
the fail-open decoder bug (§9 D1). W5 is the check that the canonicalisation is
*total* — if any row survives to the canonical block under a name other than `CL`,
either `--neutral-mods` is missing an entry or the legacy decoder invented a token.
W6 is the check that the rejection half of the rule still rejects.


**Beatmap metadata table** (`data/processed/beatmap_meta_{tag}.csv`):

| # | Check | 1k expected |
|---|---|---|
| M1 | row count == whitelist size | 21,949 |
| M2 | `beatmap_id` unique and strictly increasing | always |
| M3 | `star > 0`, `count_total > 0`, `bpm > 0` | always |
| M4 | `hit_length <= total_length` | always |
| M5 | every `beatmap_id` in the play table has a metadata row | **0 missing** |
| M6 | header is exactly the 17 columns, in order | see §6.2 |
| M7 | `approved` values are a subset of the observed set | `{1, 3, 4}` |
| M8 | `checksum` is 32 hex chars for every row | always |
| M9 | `star` median | 3.142 (whole whitelist) / 3.243 (1k played subset) / 3.144 (10k) |

M1–M5 are printed by the script every run; M5 is the cross-file invariant and is
the one to check first if the two files were generated at different times.

The 17-column header, for copy-paste comparison:

```
beatmap_id,beatmapset_id,mapper_id,version,star,bpm,max_combo,count_total,diff_overall,diff_drain,hit_length,total_length,playcount,passcount,approved,last_update,checksum
```

---

## 13. Open questions

1. **Should `playcount_cur` be normalised?** Raw `playcount_cur` conflates "how far
   into this cell" with "how big is this cell". A step-1 candidate is to keep both
   `playcount_cur` and the cell size `n` (derivable as `max(playcount_cur)` per cell)
   and let the model see both.
2. **72–74% of cells are single-record.** 74.12% for 1k, 72.47% for 10k. Decide in
   step 1 whether they are usable at all — with `playcount_cur ≡ 1` they contribute
   no order information.
3. **Does the mod set deserve a column after all?** Superseded by §13.10, which
   makes the same argument with much stronger evidence.
4. **Revisit D2?** If the practice-depth question later turns out to matter more than
   the measurement-error question, `osu_user_beatmap_playcount` is one join away and
   the failtimes means are recorded in D2. Worth doing only with a pre-registered
   comparison against the current column.
5. **What is `timestamp` for downstream?** It is in the file so that spans, gaps and
   calendar drift stay derivable without going back to the dump. Step 1 should decide
   which of those become features (`span = last − first` per cell, gap since the
   previous play, calendar year for player-year entities) and whether the raw
   timestamp itself should ever reach the model — an unprocessed absolute date is a
   leakage-prone feature, a derived span is not.
6. **Is `star` a legitimate covariate given it is a snapshot?** §8.8: `star` is the
   2026 value attached to plays from 2013. Before using it in the difficulty
   regression, check how many whitelisted maps were updated after the plays on them
   (`last_update > max(timestamp)` per beatmap). If that share is small, the
   snapshot approximation is safe; if it is large, the regression needs a
   time-varying difficulty term or a restriction to plays after `last_update`.
7. **Should the metadata table carry a "changed since first play" flag?** A cheap
   derived boolean (`last_update` later than the map's earliest play) would make the
   §6.1/§13.6 concern checkable per row. It is a derivation, so by §9 D5 it is step-1
   work — but it is the derivation most likely to be needed first.
8. **Does the played-vs-whitelist gap matter?** 1,294 (1k) / 50 (10k) whitelisted
   maps have no plays from the pool. If a modelling question is about *chart*
   difficulty rather than about observed performance, those maps are usable rows; if
   it is about learning curves, they are not. Keep them and let step 1 choose.
9. **Is `count_total` (note count) enough, or is a note-count-per-second needed?**
   `count_total` and `hit_length` are both in the file, so density is one division
   away — but the division belongs to step 1 (§9 D5), and it should be checked
   against the 2-note stub rows first.
10. **Should the play table carry a `neutral_mods` flag as a sixth column? — open,
    and now much narrower.**

    Under §9 D1 the chart-neutral mods are split: `MR`/`SD`/`PF` are folded into the
    `CL` bucket, and `NF` is rejected outright. So a row's mod set is deliberately not
    part of its identity, and the one mod whose presence *did* matter — `NF` — is no
    longer in the file at all. The remaining question is therefore only whether step 1
    needs to **ablate the `SD`/`PF` rows** (and, weakly, the `MR` rows) from the folded
    `CL` bucket.

    If yes, a column is the cheap way:

    ```
    player_id, beatmap_id, timestamp, playcount_cur, loss, neutral_mods
    ```

    with `neutral_mods ∈ {"", SD, PF, MR, ...}` or a small bitmask over the three
    folded acronyms. It is already computed in the inner loop — `canon`/`tokens` exist
    at the point the row is emitted — so the implementation is a one-line change to
    `RAW_COLS` plus the yields, and a re-run.

    The arguments for:

    * it makes the §7 ablation ("does the `CL+PF` ceiling-pinning move any downstream
      estimate?") a `WHERE` clause instead of a 2.4 GB dump re-scan;
    * it costs ~2 bytes/row after dictionary encoding, and the alternative
      (`chart_id`-style bit packing) was already rejected for being unreadable.

    The argument against is the stated spec: the play table is five columns, and
    §9 D5 rejects metadata on the play row. But this is not *metadata* — it is a
    property of the observation, exactly like `playcount_cur`. The counter-argument
    that used to be decisive ("the whitelist has two values that are believed
    equivalent") is now stronger rather than weaker: the policy *asserts* the folded
    mods are equivalent, and the cleanest way to test an assertion is to keep the
    evidence needed to falsify it.

    Note that this is **not** a request to change the policy. Canonicalising and
    recording which mods were dropped are separate decisions; the recommendation here
    is to keep the policy and add the column. It is also fine to decide "no" — the §7
    numbers already bound the effect of the folded mods, so this is a convenience
    question, not a correctness one. Re-admitting `NF` is a *policy* change and goes
    through §9 D1, not through this column (a column cannot record rows that were
    never emitted).
