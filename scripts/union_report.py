#!/usr/bin/env python
"""Union dataset report: scale, source overlap, source coverage, labels, content.

Reads the built union (step0 play table + step1 pairbest table) together with the
source step0/step1 tables and the step0 merge report, and answers four questions:

  * did the union keep the sources?  every source step1 pair is looked up by
    (player_id_base, beatmap_id) in the union; misses are diagnosed against the
    union best ACC (the only rule that can drop a pair when plays are added);
  * what is genuinely new?  union pairs that no source table contains, and pairs
    that only become valid because two datasets each contributed one play;
  * what does the union look like as training data?  split sizes, charts,
    players, label window, chart support, content coverage, descriptive star mix;
  * do the step1 invariants hold?  the step1 report checks are surfaced here.

Usage:
  python scripts/union_report.py \
      --step0 data/processed/step0_full.parquet \
      --pairbest data/processed/step1_full_pairbest.parquet \
      --step0-report data/processed/step0_full_union.json \
      --step1-report data/processed/step1_full_pairbest.json \
      --source-step0 data/processed/step0_1k.csv=1k \
      --source-step0 data/processed/step0_10k.parquet=10k \
      --source-step0 data/processed/step0_randommerge.parquet=randommerge \
      --source-pairbest data/processed/step1_1k_pairbest.parquet=1k \
      --source-pairbest data/processed/step1_10k_pairbest.parquet=10k \
      --source-pairbest data/processed/step1_randommerge_pairbest.parquet=randommerge \
      --meta data/processed/beatmap_meta_full.csv \
      --content data/processed/chart_content_cm3p_local.parquet \
      --json data/processed/step1_full_union_qa.json \
      --out docs/full_union_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from randommerge_report import pair_table  # noqa: E402

LOW_ACC = 0.90
HIGH_ACC = 0.998


def parse_inputs(items: list[str]) -> list[tuple[str, Path]]:
    out = []
    for item in items:
        if "=" in item:
            path, tag = item.split("=", 1)
        else:
            path, tag = item, Path(item).stem
        out.append((tag, Path(path)))
    return out


def load_plays(path: Path) -> pd.DataFrame:
    cols = ["player_id", "beatmap_id", "timestamp", "playcount_cur", "loss"]
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=cols)
    return pd.read_csv(path, usecols=cols)


def load_pairbest(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def acc_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pair-level best loss / n_plays for a step0 play table."""
    t = pair_table(df)
    t["best_acc"] = 1 - np.exp(t["best_loss"].astype(np.float64))
    return t


def step1_keys(df: pd.DataFrame) -> pd.DataFrame:
    base = df["player_id_base"] if "player_id_base" in df.columns else df["player_id"]
    seg = df["player_seg"] if "player_seg" in df.columns else pd.Series(0, index=df.index)
    return pd.DataFrame({"player_id_base": base.to_numpy(),
                         "player_seg": seg.to_numpy(),
                         "beatmap_id": df["beatmap_id"].to_numpy(),
                         "split": df["split"].to_numpy() if "split" in df.columns else ""})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step0", required=True, type=Path)
    ap.add_argument("--pairbest", required=True, type=Path)
    ap.add_argument("--step0-report", type=Path, default=None)
    ap.add_argument("--step1-report", type=Path, default=None)
    ap.add_argument("--source-step0", action="append", default=[], metavar="PATH=TAG")
    ap.add_argument("--source-pairbest", action="append", default=[], metavar="PATH=TAG")
    ap.add_argument("--meta", type=Path, default=None)
    ap.add_argument("--content", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--title", default="数据集并集：数据量报告")
    args = ap.parse_args()

    src_plays = parse_inputs(args.source_step0)
    src_pairs = parse_inputs(args.source_pairbest)
    s0 = load_plays(args.step0)
    pb = load_pairbest(args.pairbest)
    merge_report = json.loads(args.step0_report.read_text()) if args.step0_report else None
    step1_report = json.loads(args.step1_report.read_text()) if args.step1_report else None

    lines: list[str] = []
    add = lines.append
    add("# " + args.title)
    add("")
    add("> 自动生成。并集在 step0 的 play 行上做；step1 规则与 1k / 10k / randommerge 相同。")
    add("")

    report: dict = {"step0": str(args.step0), "pairbest": str(args.pairbest)}

    # ---------------------------------------------------------------- 1. scale
    split_counts = pb["split"].value_counts().to_dict() if "split" in pb.columns else {}
    base_col = "player_id_base" if "player_id_base" in pb.columns else "player_id"
    per_player_pairs = pb.groupby(base_col).size()
    scale = {
        "plays": int(len(s0)), "players": int(s0.player_id.nunique()),
        "charts": int(s0.beatmap_id.nunique()),
        "pairs": int(s0.groupby(["player_id", "beatmap_id"], observed=True).ngroups),
        "step1_rows": int(len(pb)), "step1_players": int(pb[base_col].nunique()),
        "step1_player_segments": int(pb["player_id"].nunique()),
        "step1_charts": int(pb.beatmap_id.nunique()),
        "split": {k: int(v) for k, v in split_counts.items()},
        "pairs_per_player_p10_p50_p90": [int(x) for x in
                                         np.percentile(per_player_pairs, [10, 50, 90])],
        "t_min": str(s0.timestamp.min()), "t_max": str(s0.timestamp.max()),
    }
    report["scale"] = scale
    add("## 1. 规模")
    add("")
    add("| 项 | 值 |")
    add("|---|---:|")
    add(f"| play 行（合并后） | {scale['plays']:,} |")
    add(f"| 玩家（合并后） | {scale['players']:,} |")
    add(f"| 谱面 | {scale['charts']:,} |")
    add(f"| 对（player, beatmap） | {scale['pairs']:,} |")
    add(f"| step1 输出 | {scale['step1_rows']:,} 对 / {scale['step1_player_segments']:,} 玩家段 / "
        f"{scale['step1_players']:,} 基础玩家 |")
    add(f"| train / test | {split_counts.get('train', 0):,} / {split_counts.get('test', 0):,} |")
    add(f"| step1 谱面 | {scale['step1_charts']:,} |")
    add(f"| 每玩家对数 p10/p50/p90 | {scale['pairs_per_player_p10_p50_p90'][0]}/"
        f"{scale['pairs_per_player_p10_p50_p90'][1]}/"
        f"{scale['pairs_per_player_p10_p50_p90'][2]} |")
    add(f"| 时间跨度 | {scale['t_min']} → {scale['t_max']} |")
    add("")

    # ------------------------------------------------------- 2. sources & overlap
    add("## 2. 来源与重叠")
    add("")
    per_source = merge_report["per_source"] if merge_report else {}
    add("| 来源 | play | 玩家 | 谱面 | 对 |")
    add("|---|---:|---:|---:|---:|")
    for tag, path in src_plays:
        if tag in per_source:
            v = per_source[tag]
            add(f"| {tag} | {v['plays']:,} | {v['players']:,} | {v['charts']:,} | {v['pairs']:,} |")
        else:
            df = load_plays(path)
            add(f"| {tag} | {len(df):,} | {df.player_id.nunique():,} | "
                f"{df.beatmap_id.nunique():,} | "
                f"{df.groupby(['player_id', 'beatmap_id'], observed=True).ngroups:,} |")
    add("")
    if merge_report:
        m = merge_report["merged"]
        d = merge_report["duplicates"]
        add("| 项 | 值 |")
        add("|---|---:|")
        add(f"| 输入 play 行合计 | {m['input_rows']:,} |")
        add(f"| 参与重复 key 的 play 行 | {d['rows_on_duplicate_keys']:,}"
            f"（{d['duplicate_keys']:,} 个 key；删除 {m['duplicate_rows_dropped']:,} 行） |")
        add(f"| 其中 loss 冲突 | {d['loss_conflicts']:,} |")
        add(f"| 合并后 play | {m['plays']:,} |")
        add(f"| 合并后对 | {m['pairs']:,} |")
        add(f"| n_plays>=2 的对 | {m['pairs_with_ge2_plays']:,} |")
        add("")
        add("两两重复 play（同一场的重复观测）与共享对：")
        add("")
        add("| 来源对 | 重复 play | loss 冲突 | 共享对 | 占较小来源 play |")
        add("|---|---:|---:|---:|---:|")
        for key, v in merge_report["pairwise"].items():
            add(f"| {key} | {v['duplicate_plays']:,} | {v['loss_conflicts']:,} | "
                f"{v['pairs_in_both']:,} | {v['play_overlap_frac_of_smaller']:.2%} |")
        add("")
        res = merge_report.get("resurrected_pairs") or {}
        if res:
            add("单次对复活（某来源里 n_plays=1、并集后 >=2 的对）：")
            add("")
            add("| 来源 | 单次对 | 并集后 >=2 | 所有来源都只有 1 条（真正复活） |")
            add("|---|---:|---:|---:|")
            for tag, v in res.items():
                add(f"| {tag} | {v['single_play_pairs']:,} | {v['single_but_ge2_in_union']:,} | "
                    f"{v['single_in_every_source_ge2_in_union']:,} |")
            add("")

    # ------------------------------------------------- 3. source step1 coverage
    add("## 3. 来源 step1 表在并集里的覆盖")
    add("")
    add("按 (基础玩家, 谱面) 匹配：365 天空窗重分段只改段号，不减少观测。来源对消失只有"
        "两种原因——合并后 n_plays<2（源里的第二条是同一秒的重复行）或 best ACC 被并集"
        "推过 0.998。split 标签由并集自己的 step1 重新划分，不复用来源的 train/test。")
    add("")
    full_keys_df = step1_keys(pb)
    base_chart = full_keys_df[["player_id_base", "beatmap_id"]].drop_duplicates()
    full_key_set = set(map(tuple, base_chart.to_numpy()))
    full_ident = set(map(tuple, full_keys_df[["player_id_base", "player_seg", "beatmap_id"]]
                         .drop_duplicates().to_numpy()))
    union_source_ident: set = set()
    union_source_basechart: set = set()
    coverage = {}
    add("| 来源 | step1 对 | 谱面 | 覆盖 (base, chart) | 丢失 | 合并后 n_plays<2 | best ACC>0.998 | 其他 | 保留原段身份 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tag, path in src_pairs:
        sp = load_pairbest(path)
        k = step1_keys(sp)
        bc = k[["player_id_base", "beatmap_id"]].drop_duplicates()
        bc_rows = list(map(tuple, bc.to_numpy()))
        ident = set(map(tuple, k[["player_id_base", "player_seg", "beatmap_id"]]
                        .drop_duplicates().to_numpy()))
        union_source_ident |= ident
        union_source_basechart |= set(bc_rows)
        missing = [kk for kk in bc_rows if kk not in full_key_set]
        n_dedupe = n_high = n_other = 0
        if missing:
            # classify each miss: (a) fewer than 2 distinct union plays, i.e. the source
            # n_plays was a duplicate row for the same second; (b) best ACC pushed over
            # the cap by a better play from another dump; (c) anything else.
            miss = pd.DataFrame(missing, columns=["player_id", "beatmap_id"])
            sub = s0[s0.player_id.isin(set(miss.player_id))]
            agg = (sub.groupby(["player_id", "beatmap_id"], observed=True)["loss"]
                   .agg(n="size", best="min")
                   .reindex(pd.MultiIndex.from_frame(miss)))
            n_plays = agg["n"].to_numpy(dtype=np.float64, na_value=np.nan)
            acc = 1 - np.exp(agg["best"].to_numpy(dtype=np.float64, na_value=np.nan))
            n_dedupe = int(np.nansum(n_plays < 2))
            rest = ~(n_plays < 2)
            n_high = int(np.nansum(rest & (acc > HIGH_ACC)))
            n_other = int(len(missing) - n_dedupe - n_high)
        n_ident_kept = sum(1 for x in ident if x in full_ident)
        cov = 1.0 - len(missing) / max(len(bc_rows), 1)
        coverage[tag] = {
            "rows": int(len(sp)), "pairs": int(len(bc_rows)),
            "charts": int(sp.beatmap_id.nunique()),
            "coverage": float(cov), "missing": len(missing),
            "missing_dedupe_n_plays_lt_2": n_dedupe,
            "missing_chart_best_acc_gt_0.998": n_high,
            "missing_other": n_other,
            "identity_kept": int(n_ident_kept), "identity_pairs": int(len(ident)),
        }
        add(f"| {tag} | {len(sp):,} | {sp.beatmap_id.nunique():,} | {cov:.4%} | "
            f"{len(missing):,} | {n_dedupe:,} | {n_high:,} | {n_other:,} | "
            f"{n_ident_kept:,}/{len(ident):,} |")
        print(f"[coverage] {tag}: {cov:.4%}, {len(missing)} missing "
              f"(dedupe n<2: {n_dedupe}, ACC>{HIGH_ACC}: {n_high}, other: {n_other}), "
              f"identity kept {n_ident_kept}/{len(ident)}", flush=True)
    add("")
    report["source_coverage"] = coverage

    new_basechart = full_key_set - union_source_basechart
    new_ident = full_ident - union_source_ident
    report["new_pairs"] = {
        "base_chart_not_in_any_source": int(len(new_basechart)),
        "identity_not_in_any_source": int(len(new_ident)),
        "resurrected_total": int(sum(v["single_in_every_source_ge2_in_union"]
                                     for v in (merge_report or {}).get("resurrected_pairs", {}).values())),
    }

    # ------------------------------------------------------------- 4. labels
    acc = 1 - np.exp(pb["loss"].astype(np.float64))
    add("## 4. 标签（step1 行上每对的最好成绩）")
    add("")
    add("| 项 | 值 |")
    add("|---|---:|")
    add(f"| best ACC 中位数 | {float(acc.median()):.4f} |")
    add(f"| ACC 窗口 | ({LOW_ACC}, {HIGH_ACC}]，loss "
        f"[{float(pb['loss'].max()):.4f}, {float(pb['loss'].min()):.4f}) |")
    add(f"| n_plays 中位数 / 均值 | {float(pb['n_plays'].median()):.0f} / "
        f"{float(pb['n_plays'].mean()):.2f} |")
    add("")
    report["labels"] = {
        "best_acc_median": float(acc.median()),
        "n_plays_median": float(pb["n_plays"].median()),
        "n_plays_mean": float(pb["n_plays"].mean()),
        "loss_min": float(pb["loss"].min()), "loss_max": float(pb["loss"].max()),
    }

    # ------------------------------------------------------ 5. chart support
    players_per_chart = pb.groupby("beatmap_id").size()
    add("## 5. 每图支持度（step1 行）")
    add("")
    add("| 每图玩家数 | 谱面数 | 占比 |")
    add("|---|---:|---:|")
    bins = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, 10 ** 9)]
    support = {}
    for lo, hi in bins:
        n = int(((players_per_chart >= lo) & (players_per_chart <= hi)).sum())
        label = str(lo) if lo == hi else (f"{lo}+" if hi > 10 ** 8 else f"{lo}-{hi}")
        support[label] = n
        add(f"| {label} | {n:,} | {n / len(players_per_chart):.2%} |")
    add("")
    report["chart_support"] = support

    # ------------------------------------------------ 6. content & star mix
    add("## 6. 内容覆盖与 star 分布（描述性）")
    add("")
    extra: dict = {}
    if args.content and args.content.exists():
        content_ids = set(pd.read_parquet(args.content, columns=["beatmap_id"])["beatmap_id"])
        chart_hit = pb.beatmap_id.isin(content_ids)
        charts_with = int(pb.loc[chart_hit, "beatmap_id"].nunique())
        extra["content"] = {
            "charts": int(pb.beatmap_id.nunique()),
            "charts_with_content": charts_with,
            "rows_with_content": int(chart_hit.sum()),
            "row_coverage": float(chart_hit.mean()),
            "chart_coverage": charts_with / pb.beatmap_id.nunique(),
        }
        add(f"* CM3P legacy 内容表：{charts_with:,}/{pb.beatmap_id.nunique():,} 图"
            f"（{extra['content']['chart_coverage']:.2%}）、"
            f"{int(chart_hit.sum()):,}/{len(pb):,} 行"
            f"（{extra['content']['row_coverage']:.2%}）。")
    if args.meta and args.meta.exists():
        meta = pd.read_csv(args.meta, usecols=["beatmap_id", "star"])
        star = pb[["beatmap_id"]].merge(meta, on="beatmap_id", how="left")["star"]
        extra["star"] = {
            "median": float(star.median()),
            "missing_rows": int(star.isna().sum()),
            "le3_frac": float((star <= 3).mean()),
            "le4_frac": float((star <= 4).mean()),
        }
        add(f"* step1 行 star 中位数 {extra['star']['median']:.2f}；"
            f"<=3 星行 {extra['star']['le3_frac']:.2%}、<=4 星行 {extra['star']['le4_frac']:.2%}"
            f"（描述性统计，不作分层/加权/评测口径）。")
    add("")
    report["extra"] = extra

    # ---------------------------------------------------- 7. step1 invariants
    if step1_report:
        rules = step1_report.get("rules", {})
        funnel = step1_report.get("funnel", {})
        checks = step1_report.get("checks", {})
        add("## 7. step1 漏斗与校验")
        add("")
        add("清洗规则：玩家 >= " + str(rules.get("min_plays_per_player", "?")) +
            " 条、空窗 " + str(rules.get("gap_days", "?")) + " 天、每对最高成绩、"
            "n_plays >= " + str(rules.get("min_plays_per_pair", "?")) +
            "、ACC (" + str(rules.get("acc_floor", "?")) + ", " +
            str(rules.get("max_best_acc", "?")) + "]。")
        add("")
        add("| 项 | 值 |")
        add("|---|---:|")
        for key in ("step0_rows", "step0_pairs", "dropped_low_play_rows",
                    "pairs_below_min_plays", "dropped_low_acc_pairs",
                    "dropped_high_acc_pairs", "kept_pairs", "train_rows", "test_rows",
                    "extra_segments"):
            if key in funnel:
                add(f"| {key} | {funnel[key]:,} |")
        add("")
        bad = {k: v for k, v in checks.items() if not v}
        add(f"校验：{len(checks) - len(bad)}/{len(checks)} 通过"
            + (f"；未通过：{bad}" if bad else "。"))
        add("")
        report["step1_checks"] = checks
        report["step1_funnel"] = funnel
        if bad:
            print(f"[qa] step1 checks failed: {bad}", file=sys.stderr)
            return 1

    # --------------------------------------------------------- 8. new pairs
    add("## 8. 并集新增的对")
    add("")
    add(f"* 按 (基础玩家, 谱面)：**{len(new_basechart):,}** 个 step1 对不在任何来源 step1 表里"
        f"（单次对真正复活：{report['new_pairs']['resurrected_total']:,}）。")
    add(f"* 按 (基础玩家, 段号, 谱面)：{len(new_ident):,} 个身份不在任何来源里。"
        f"空窗重分段会改段号，这类差异不是新观测。")
    add("")
    # ------------------------------------------------------------ 9. caveats
    add("## 9. 口径边界")
    add("")
    add("* 绝对 NLL / RMSE 不能和 1k、10k、randommerge 的单集数字互比：总体、ceiling 与")
    add("  test 集都不同；要比较模型，必须在同一套 full 划分上重跑。")
    add("* 来源 step1 的 split 不复用：pair-random 的随机数按有序对集合展开，并集是对新的")
    add("  对集合重新划分（seed 仍为 20260928），所以来源的 train/test 标签在这里没有意义。")
    add("* 合并在 play 级：同一 (玩家, 谱面) 的 n_plays、playcount_cur 与 365 天空窗都会按")
    add("  并集时间线重算；跨 dump 的同一场 play 只保留一条。")
    add("* randommerge 在 README 里的 453,406 对 / 20,388 图是**内容过滤后**的数字；")
    add("  本表 step1 未做内容过滤，CM3P 训练前要按内容表取子集。")
    add("* star 分桶只是描述性统计（见 docs/low_difficulty_data.md 的勘误），不作分层/加权口径。")
    add("")

    text = "\n".join(lines)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print("wrote", args.out)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print("wrote", args.json)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
