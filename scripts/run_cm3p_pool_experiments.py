#!/usr/bin/env python
"""T3 driver: A0-A6 pooling -> P2 chart-disjoint rank models -> official NLL.

All variants share the same canon T1 window store; A0 is the T3 control, not a
legacy-table parity target.  The primary comparison is each variant vs A0.

What it does
  1. build data/processed/chart_pool_features_1k/a0.parquet (window mean),
     fit the A0 chart-disjoint linear model and dump its fold heads;
  2. build a1..a5 features from the window store using those heads;
  3. fit the same P2 protocol (5 chart folds, seed 0, lambda 1e-3) per variant;
  4. optionally train the A6 TCN signal;
  5. merge every signal with the P2 baselines, run step2_rank_eval.py
     (--split test --ceiling-fit all --common-maps --strict-pairs) and
     paired_delta_nll.py against A0;
  6. write results/cm3p_pool_1k/summary.json + SUMMARY.md with the Go/no-go.

Usage
  python scripts/run_cm3p_pool_experiments.py [--variants a0,a1,a2,a3,a4,a5]
  python scripts/run_cm3p_pool_experiments.py --quick        # smoke test only
  python scripts/run_cm3p_pool_experiments.py --skip-eval    # only build signals

Everything is idempotent: an existing <name>.parquet/json is reused unless
--force.  The token-cost columns come from results/cm3p_rank_1k/token_budget_all.json
when it exists (scripts/cm3p_token_budget.py).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STEP1 = ROOT / "data/processed/step1_1k_pairbest.parquet"
STEP2 = ROOT / "scripts/step2_cm3p_rank.py"
EVAL = ROOT / "scripts/step2_rank_eval.py"
PAIRED = ROOT / "scripts/paired_delta_nll.py"
STRAT = ROOT / "scripts/stratified_delta_nll.py"
POOL = ROOT / "scripts/cm3p_pool.py"
POOL_DIR = ROOT / "data/processed/chart_pool_features_1k"
OUT = ROOT / "results/cm3p_pool_1k"
BASE_SIGNALS = ROOT / "results/cm3p_rank_1k/signals_p2.parquet"
TOKEN_BUDGET = ROOT / "results/cm3p_rank_1k/token_budget_all.json"
VARIANTS = ["a0", "a1", "a2", "a3", "a4", "a5", "a6"]
BASELINES = ["cm3p_linear_cd_l1e3", "msd_overall", "sunny_star", "daniel_star"]
CACHE = Path("/tmp/cm3p_pool_cache")
WIN_INDEX = ROOT / ".cache/cm3p/index.parquet"
WIN_DIR = ROOT / ".cache/cm3p/win"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], what: str) -> None:
    log("RUN " + " ".join(str(c) for c in cmd))
    p = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), text=True,
                       capture_output=True)
    if p.stdout:
        print(p.stdout, end="", flush=True)
    if p.returncode != 0:
        if p.stderr:
            print(p.stderr, end="", flush=True)
        raise SystemExit(f"{what} failed (exit {p.returncode})")
    if p.stderr and "Warning" not in p.stderr and "OMP" not in p.stderr:
        print(p.stderr, end="", flush=True)


def _have(path: Path, force: bool) -> bool:
    if path.exists() and not force:
        log(f"skip {path.name} (exists)")
        return True
    return False


def _name(v: str) -> str:
    return f"pool_{v}"


def build_features(variants: list[str], heads: Path, force: bool) -> dict:
    meta = {}
    todo = [v for v in variants
            if v != "a6" and not _have(POOL_DIR / f"{v}.parquet", force)]
    if not todo:
        return meta
    cmd = [sys.executable, POOL, "--variants", ",".join(todo),
           "--out-dir", POOL_DIR, "--index", WIN_INDEX, "--win-dir", WIN_DIR]
    if heads.exists():
        cmd += ["--heads", heads]
    elif any(v in ("a2", "a3", "a4") for v in todo):
        raise SystemExit(f"{heads} missing: a2/a3/a4 need the A0 head bundle")
    run(cmd, "cm3p_pool features")
    m = POOL_DIR / "meta.json"
    if m.exists():
        meta = json.loads(m.read_text())
    return meta


def fit_variant(v: str, force: bool) -> dict:
    name = _name(v)
    sig = OUT / f"{name}.parquet"
    js = OUT / f"{name}.json"
    if _have(js, force):
        return json.loads(js.read_text())
    cmd = [sys.executable, STEP2, "--model", "linear", "--chart-only",
           "--chart-folds", "5", "--chart-seed", "0", "--val-frac", "0",
           "--lam", "1e-3", "--name", name,
           "--content-path", POOL_DIR / f"{v}.parquet",
           "--content-col", "features",
           "--out-signals", sig, "--out-json", js]
    if v == "a0":
        cmd += ["--dump-heads", OUT / "a0_heads.npz"]
    run(cmd, f"fit {name}")
    return json.loads(js.read_text())


def train_a6(force: bool) -> None:
    name = _name("a6")
    sig = OUT / f"{name}.parquet"
    if _have(sig, force):
        return
    run([sys.executable, POOL, "--variants", "a6",
         "--heads", OUT / "a0_heads.npz", "--step1", STEP1,
         "--win-dir", WIN_DIR, "--out-signals", sig,
         "--epochs", "80", "--threads", "6"], "a6 TCN")


def merge_signals(variants: list[str]) -> pd.DataFrame:
    frames = []
    for v in variants:
        p = OUT / f"{_name(v)}.parquet"
        if not p.exists():
            raise SystemExit(f"missing signal {p}")
        d = pd.read_parquet(p)
        if v == "a6" and "a6" in d.columns:
            d = d.rename(columns={"a6": _name("a6")})
        frames.append(d.drop_duplicates("beatmap_id").set_index("beatmap_id"))
    base = pd.read_parquet(BASE_SIGNALS).drop_duplicates("beatmap_id"
                                                         ).set_index("beatmap_id")
    for c in BASELINES:
        if c in base.columns:
            frames.append(base[[c]])
    out = pd.concat(frames, axis=1).reset_index()
    out.to_parquet(OUT / "signals_pool.parquet", index=False)
    log(f"signals_pool.parquet: {out.shape[0]:,} charts x {out.shape[1]} cols")
    return out


def evaluate(variants: list[str], force: bool) -> dict:
    sigs = [_name(v) for v in variants] + [b for b in BASELINES]
    ev = OUT / "rank_eval_pool.json"
    if not _have(ev, force):
        run([sys.executable, EVAL, "--in", STEP1,
             "--signal-table", OUT / "signals_pool.parquet",
             "--signals", ",".join(sigs), "--split", "test",
             "--ceiling-fit", "all", "--common-maps", "--strict-pairs",
             "--folds", "2", "--bootstrap", "400",
             "--cache-dir", CACHE, "--out", ev], "rank_eval")
    pairs = [f"{_name(v)}:{_name('a0')}" for v in variants if v != "a0"]
    pairs += [f"{_name(v)}:cm3p_linear_cd_l1e3" for v in variants]
    pd_path = OUT / "paired_delta_pool.json"
    if pairs and not _have(pd_path, force):
        run([sys.executable, PAIRED, "--in", STEP1,
             "--signal-table", OUT / "signals_pool.parquet",
             "--signals", ",".join(sigs), "--pairs", ",".join(pairs),
             "--split", "test", "--bootstrap", "400", "--out", pd_path],
            "paired delta")
    return {"eval": json.loads(ev.read_text()) if ev.exists() else {},
            "paired": json.loads(pd_path.read_text()) if pd_path.exists() else {}}


def stratify_if_needed(args) -> dict | None:
    """Plan section 9: only when canon is *worse* than legacy on P2."""
    if not args.stratify:
        return None
    pd_path = OUT / "paired_delta_pool.json"
    if not pd_path.exists():
        return None
    pair = json.loads(pd_path.read_text()).get("pairs", {})
    d = pair.get("pool_a0:cm3p_linear_cd_l1e3", {}).get("delta")
    if d is None or d <= 0:
        log(f"stratify skipped: canon-legacy delta = {d} (not > 0)")
        return None
    out = OUT / "stratified_pool_a0_vs_legacy.json"
    run([sys.executable, STRAT, "--in", STEP1,
         "--signal-table", OUT / "signals_pool.parquet",
         "--signals", "pool_a0,cm3p_linear_cd_l1e3",
         "--pairs", "pool_a0:cm3p_linear_cd_l1e3",
         "--strata-table", args.strata_table, "--stratum-col", args.stratum_col,
         "--split", "test", "--bootstrap", "400", "--out", out], "stratified delta")
    return json.loads(out.read_text()) if out.exists() else None


def summary(variants: list[str], ev: dict, feature_meta: dict,
            fits: dict) -> None:
    rows = []
    sig = ev.get("eval", {}).get("signals", {})
    pair = ev.get("paired", {}).get("pairs", {})
    for v in variants:
        n = _name(v)
        r = sig.get(n, {})
        p = pair.get(f"{n}:{_name('a0')}", {})
        rows.append({
            "variant": v, "name": n,
            "width": feature_meta.get("widths", {}).get(v),
            "test_nll_strict": r.get("nll_per_instance"),
            "captured": r.get("captured"),
            "lw": (r.get("per_player") or {}).get("lw"),
            "delta_vs_a0": p.get("delta"),
            "ci_lo": (p.get("ci") or [None, None])[0],
            "ci_hi": (p.get("ci") or [None, None])[1],
            "p_lt0": p.get("p_delta_lt_0"),
            "test_nll_vs_legacy": pair.get(f"{n}:cm3p_linear_cd_l1e3", {}).get("delta"),
            "seconds_fit": fits.get(v, {}).get("seconds"),
            "params": fits.get(v, {}).get("content_dim"),
        })
    # A0 is the T3 control (same canon window store); the legacy table is only
    # a historical reference, never an A0 parity target.
    control = sig.get(_name("a0"), {})
    legacy = sig.get("cm3p_linear_cd_l1e3", {})
    out = {"baseline": {"name": _name("a0"),
                        "test_nll_strict": control.get("nll_per_instance"),
                        "captured": control.get("captured")},
           "legacy_reference": {"name": "cm3p_linear_cd_l1e3",
                                "test_nll_strict": legacy.get("nll_per_instance"),
                                "captured": legacy.get("captured")},
           "variants": rows,
           "feature_meta": feature_meta,
           "stratified": ev.get("stratified"),
           "token_budget": (json.loads(TOKEN_BUDGET.read_text())
                            if TOKEN_BUDGET.exists() else None)}
    (OUT / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    lines = ["# CM3P window pooling (T3)", "",
             f"T3 control (canon A0, same T1): NLL "
             f"{control.get('nll_per_instance')} captured {control.get('captured')}", "",
             f"legacy reference (old table, historical only): NLL "
             f"{legacy.get('nll_per_instance')} captured {legacy.get('captured')}", "",
             "| variant | width | strict NLL | captured | Δ vs A0 | 95% CI | P(Δ<0) |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['variant']} | {r['width']} | {r['test_nll_strict']} | "
                     f"{r['captured']} | {r['delta_vs_a0']} | "
                     f"[{r['ci_lo']}, {r['ci_hi']}] | {r['p_lt0']} |")
    best = min((r for r in rows if r["test_nll_strict"] is not None),
               key=lambda r: r["test_nll_strict"], default=None)
    if best:
        lines += ["", f"best overall: {best['variant']} NLL {best['test_nll_strict']}"]
    cand = [r for r in rows if r["variant"] in ("a2", "a3", "a4")
            and r["test_nll_strict"] is not None]
    if cand:
        b = min(cand, key=lambda r: r["test_nll_strict"])
        lines.append(f"best A2/A3/A4: {b['variant']} NLL {b['test_nll_strict']}")
        lo, hi = b.get("ci_lo"), b.get("ci_hi")
        if lo is not None and hi is not None and hi < 0:
            lines.append("Go/no-go: pooling is the main effect "
                         "(paired Δ vs A0, 95% CI entirely < 0).")
        elif lo is not None and hi is not None and lo <= 0 <= hi:
            lines.append("Go/no-go: no significant pooling gain vs A0 "
                         "(paired 95% CI includes 0).")
        else:
            lines.append("Go/no-go: inconclusive (paired Δ vs A0 CI not usable).")
    if out.get("stratified"):
        st = out["stratified"]
        lines += ["", "## canon vs legacy, within count_total deciles",
                  "| stratum | charts | pairs | Δ NLL | 95% CI | P(Δ<0) |",
                  "|---|---|---|---|---|---|"]
        for r in st.get("strata", []):
            if r.get("skipped"):
                lines.append(f"| {r['stratum']} | {r['n_charts']} | 0 | skipped | | |")
            else:
                lines.append(f"| {r['stratum']} | {r['n_charts']} | {r['n_pairs']} | "
                             f"{r['delta']:+.5f} | [{r['ci'][0]:+.5f}, {r['ci'][1]:+.5f}] | "
                             f"{r['p_delta_lt_0']:.3f} |")
    if out.get("token_budget"):
        tb = out["token_budget"]
        lines += ["", "## token budget",
                  f"canon {tb['canon']['tokens_per_note']['mean']:.3f} tokens/note "
                  f"vs legacy {tb['legacy']['tokens_per_note']['mean']:.3f}; "
                  f"saved {tb['savings_percent']:.1f}%",
                  f"pad_efficiency (length-sorted) {tb['pad_efficiency']}",
                  f"mania columns emitted: {tb['column_tokens']:,} tokens"]
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    log("wrote summary.json / SUMMARY.md")
    print("\n".join(lines), flush=True)


def main() -> int:
    global WIN_INDEX, WIN_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="a0,a1,a2,a3,a4,a5")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-a6", action="store_true")
    ap.add_argument("--stratify", action="store_true",
                    help="if canon (pool_a0) is worse than legacy, run the "
                         "count_total-decile Delta check (plan section 9)")
    ap.add_argument("--strata-table", default="data/processed/rank_signals_1k.parquet")
    ap.add_argument("--stratum-col", default="count_total")
    ap.add_argument("--window-index", default=str(WIN_INDEX))
    ap.add_argument("--window-dir", default=str(WIN_DIR))
    ap.add_argument("--quick", action="store_true",
                    help="only check prerequisites and print the plan")
    args = ap.parse_args()
    variants = [v.strip().lower() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variants {bad}")
    WIN_INDEX = Path(args.window_index)
    WIN_DIR = Path(args.window_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    for p in (STEP1, BASE_SIGNALS):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    idx = WIN_INDEX
    if not idx.exists():
        raise SystemExit(f"missing {idx}: run step0c --windows on first")
    if args.quick:
        print(f"window index: {idx}\nfeatures: {POOL_DIR}\nresults: {OUT}\n"
              f"variants: {variants}\nbaseline: {BASE_SIGNALS}")
        return 0

    # A0 first: it produces the heads that a2/a3/a4 need.
    if "a0" not in variants:
        raise SystemExit("a0 must be in --variants: it produces the head bundle")
    meta0 = build_features(["a0"], OUT / "a0_heads.npz", args.force)
    fit_variant("a0", args.force)
    heads = OUT / "a0_heads.npz"
    if not heads.exists():
        raise SystemExit(f"{heads} missing: the A0 fit did not dump its heads")
    rest = [v for v in variants if v != "a0" and v != "a6"]
    feature_meta = build_features(rest, heads, args.force)
    feature_meta.setdefault("widths", {})
    feature_meta["widths"] = {**meta0.get("widths", {}), **feature_meta["widths"]}
    fits = {"a0": json.loads((OUT / f"{_name('a0')}.json").read_text())}
    for v in rest:
        fits[v] = fit_variant(v, args.force)
    if "a6" in variants and not args.skip_a6:
        train_a6(args.force)
    merge_signals([v for v in variants if (OUT / f"{_name(v)}.parquet").exists()])
    if not args.skip_eval:
        ev = evaluate([v for v in variants if (OUT / f"{_name(v)}.parquet").exists()],
                      args.force)
        ev["stratified"] = stratify_if_needed(args)
        summary([v for v in variants if (OUT / f"{_name(v)}.parquet").exists()],
                ev, feature_meta, fits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
