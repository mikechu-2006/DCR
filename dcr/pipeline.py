"""流水线：解析 step 顺序 → 接好输入输出 → 执行 → 登记产物与指标。

一次运行的完整语义：

1. 按 BUILTIN_ORDER（step0 → step1 → step2）再 extra_steps 的顺序展开要跑的 step；
2. 每个 step 的输入按优先级解析：配置里的 inputs → 上游 step 本次的产物 → cache_dir；
3. 生成完整命令行（写进 manifest 与 run.sh），子进程执行，输出同时进实验日志；
4. 产物打指纹（大小/sha256），机器可读报告（step1 漏斗、step2 指标）读出来汇总；
5. 任何一步失败：记下错误、写 manifest、返回非零退出码。
"""

from __future__ import annotations

import glob as globmod
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import steps as step_registry
from .config import BUILTIN_ORDER, STEP_DEFAULT_VERSION, cache_dir_of, dump_yaml
from .errors import DcrError
from .experiment import Experiment, new_experiment_dir
from .paths import DEFAULT_EXPERIMENT_ROOT, REPO_ROOT, display, rel_link, resolve_path
from .run import format_cmd, run_command
from .steps import PlanCtx, StepDef, flags


@dataclass
class Options:
    dry_run: bool = False
    only: tuple[str, ...] = ()
    skip: tuple[str, ...] = ()
    exp_dir: Path | None = None
    root: Path | None = None
    resume: bool = False
    force: bool = False
    hash: bool | None = None
    echo: bool = True
    keep_going: bool = False


@dataclass
class StepItem:
    name: str
    version: str
    spec: dict
    sdef: StepDef | None = None
    extra: bool = False

    @property
    def title(self) -> str:
        if self.sdef is not None:
            return self.sdef.title or Path(self.sdef.script).name
        return str(self.spec.get("description") or Path(str(self.spec.get("script", "cmd"))).name)


# --------------------------------------------------------------------- step 展开
def build_items(cfg: Mapping, opts: Options) -> list[StepItem]:
    items: list[StepItem] = []
    unknown = sorted(set(cfg.get("steps", {})) - set(BUILTIN_ORDER))
    if unknown:
        raise DcrError(f"配置里有未知的 step: {', '.join(unknown)}；"
                       f"内置 step 只有 {', '.join(BUILTIN_ORDER)}"
                       f"（其他脚本请用 extra_steps）")
    for name in BUILTIN_ORDER:
        spec = cfg.get("steps", {}).get(name)
        if spec is None:
            continue
        version = str(spec.get("version"))
        sdef = step_registry.get(name, version)
        if sdef is None:
            known = ", ".join(step_registry.available_versions(name)) or "(无)"
            raise DcrError(f"未知的 step 版本 steps.{name}.version = {version!r}；{name} 可用版本：{known}")
        items.append(StepItem(name, version, dict(spec), sdef))

    extras = []
    for spec in cfg.get("extra_steps", []):
        extras.append(StepItem(str(spec["name"]), str(spec.get("version", "custom")),
                               dict(spec), extra=True))
    extras = _order_extras(extras)
    items.extend(extras)

    only, skip = set(opts.only), set(opts.skip)
    if only:
        unknown = only - {i.name for i in items}
        if unknown:
            raise DcrError(f"--only 里有配置中不存在的 step: {', '.join(sorted(unknown))}")
        items = [i for i in items if i.name in only]
    if skip:
        items = [i for i in items if i.name not in skip]
    return items


def _order_extras(extras: list[StepItem]) -> list[StepItem]:
    """extra_steps 支持 after: [name]：只在这些 step 之后跑（稳定拓扑排序）。"""
    by_name = {i.name: i for i in extras}
    ordered: list[StepItem] = []
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(item: StepItem) -> None:
        if item.name in done:
            return
        if item.name in visiting:
            raise DcrError(f"extra_steps 的 after 存在环：{item.name}")
        visiting.add(item.name)
        for dep in item.spec.get("after") or []:
            if dep in by_name:
                visit(by_name[dep])
        visiting.discard(item.name)
        done.add(item.name)
        ordered.append(item)

    for item in extras:
        visit(item)
    return ordered


# --------------------------------------------------------------------- 输入解析
def _effective_version(cfg: Mapping | None, step: str) -> str:
    """配置里写的版本；没写时用全局默认（step1 现在是 pairbest）。"""
    spec = ((cfg or {}).get("steps") or {}).get(step) or {}
    return str(spec.get("version") or STEP_DEFAULT_VERSION.get(step, ""))


def _cache_candidates(step: str, output: str, tag: str,
                      cfg: Mapping | None = None) -> tuple[str, ...]:
    """cache_dir 里的习惯文件名；共享 .osu 缓存是绝对路径，原样返回。

    step1 的产物名取决于版本：默认 pairbest 用 step1_{tag}_pairbest.*，
    旧的 play 级 clean 仍用 step1_{tag}.*；这里跟着配置里的 version 走，
    否则 step2-only 的配置会静默读到另一套总体。
    """
    if (step, output) == ("step0", "osu_cache"):
        # 600 MB 的 .osu 语料不放 cache_dir（它跨实验复用），见 cfg.osu_cache_dir
        return (str(resolve_path((cfg or {}).get("osu_cache_dir") or "data/osu_cache")),)
    if (step, output) == ("step1", "plays"):
        if _effective_version(cfg, "step1") == "pairbest":
            return (f"step1_{tag}_pairbest.parquet", f"step1_{tag}_pairbest.csv")
        return (f"step1_{tag}.parquet", f"step1_{tag}.csv")
    table = {
        ("step0", "plays"): ("step0_{tag}.parquet", "step0_{tag}.csv"),
        ("step0", "meta"): ("beatmap_meta_{tag}.csv",),
        ("step0", "content"): ("chart_content_cm3p.parquet",),
        ("step0", "whitelist"): ("whitelist_{tag}.csv",),
    }
    return tuple(name.format(tag=tag) for name in table.get((step, output), ()))


def resolve_inputs(
    item: StepItem,
    cfg: Mapping,
    exp: Experiment,
    produced: Mapping[str, Mapping[str, list[Path]]],
    link: bool = True,
) -> dict[str, tuple[Path | None, str]]:
    """输入名 → (路径, 来源)。来源 ∈ config / step:<name> / cache / none / missing。

    优先级：配置里写死的 inputs > 本次上游 step 的产物 > cache_dir 里的同名产物。
    命中 cache 时可选地在实验目录 artifacts/ 下建相对符号链接，让实验目录自描述。
    """
    tag = str(cfg.get("tag", "1k"))
    user_inputs = item.spec.get("inputs") or {}
    declared: Mapping[str, tuple[str, str] | None] = item.sdef.inputs if item.sdef else {}
    optional = set(item.sdef.optional) if item.sdef else set()
    cache_root = cache_dir_of(cfg)
    link_cache = bool(cfg.get("io", {}).get("link_cache", True))
    resolved: dict[str, tuple[Path | None, str]] = {}

    names = list(declared)
    for extra_name in user_inputs:
        if extra_name not in names:
            names.append(extra_name)

    for name in names:
        if name in user_inputs:
            path = exp.path(str(user_inputs[name]))
            if not path.exists():
                raise DcrError(f"steps.{item.name}.inputs.{name} 指向的文件不存在: {display(path)}")
            resolved[name] = (path, "config")
            continue
        source = declared.get(name)
        if source is not None:
            src_step, src_out = source
            paths = produced.get(src_step, {}).get(src_out)
            if paths:
                resolved[name] = (paths[0], f"step:{src_step}")
                continue
            found = None
            for candidate in _cache_candidates(src_step, src_out, tag, cfg):
                path = (Path(candidate) if Path(candidate).is_absolute()
                        else cache_root / candidate)
                if path.exists():
                    found = path
                    if link and link_cache:
                        try:
                            rel_link(path, exp.artifacts / candidate)
                        except OSError:
                            pass
                    break
            resolved[name] = (found, "cache") if found else (
                (None, "none") if name in optional else (None, "missing"))
            continue
        resolved[name] = (None, "none")
    return resolved


def _missing_input_error(item: StepItem, name: str, cfg: Mapping) -> DcrError:
    cache_root = display(cache_dir_of(cfg))
    hints = [f"steps.{item.name}.inputs.{name}: <路径>  （直接指定）"]
    if item.sdef and item.sdef.inputs.get(name):
        src_step, src_out = item.sdef.inputs[name]  # type: ignore[misc]
        hints.append(f"在配置里加上 steps.{src_step}（本次跑出 {src_out}）")
        cands = ", ".join(_cache_candidates(src_step, src_out,
                                             str(cfg.get("tag", "1k")), cfg))
        hints.append(f"把 {cands} 放进 cache_dir（{cache_root}）")
    return DcrError(f"steps.{item.name} 的输入 {name} 无法解析。可选做法：\n  - "
                    + "\n  - ".join(hints))


# --------------------------------------------------------------------- 产物匹配
def _match_outputs(raw: Mapping[str, Any]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for name, value in raw.items():
        pattern = str(value)
        if globmod.has_magic(pattern):
            matches = sorted(Path(p) for p in globmod.glob(pattern))
            # 没有匹配时保留模式本身：dry-run 能显示「将要生成什么」，
            # 真实运行里它会被标记成产物缺失
            out[name] = matches or [Path(pattern)]
        else:
            out[name] = [Path(pattern)]
    return out


def _read_reports(item: StepItem, outputs: Mapping[str, list[Path]]) -> dict | None:
    if not item.sdef or not item.sdef.report:
        return None
    reports: dict[str, Any] = {}
    for name in item.sdef.report:
        for path in outputs.get(name, []):
            if path.suffix == ".json" and path.is_file():
                try:
                    reports[name] = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    reports[name] = {"error": f"报告读取失败: {exc}"}
                break
    return reports or None


# ------------------------------------------------------------------------- 执行
def run(cfg: Mapping, config_path: Path | None, opts: Options) -> int:
    items = build_items(cfg, opts)
    if not items:
        raise DcrError(
            "没有要执行的 step：配置里的 steps / extra_steps 为空（或都被 --only/--skip 过滤掉了）。\n"
            "  可直接用现成配置：configs/exp_std_1k.yaml（只跑 step2，复用缓存）"
            "、configs/exp_full_1k.yaml（从 dump 全流程）")

    exp_dir = Path(opts.exp_dir).resolve() if opts.exp_dir else new_experiment_dir(
        opts.root or DEFAULT_EXPERIMENT_ROOT, str(cfg.get("name", "")))
    existed = exp_dir.exists()
    exp = Experiment(exp_dir, cfg, config_path, echo=opts.echo, resume=opts.resume)
    do_hash = cfg.get("io", {}).get("hash", True) if opts.hash is None else opts.hash
    hash_max = int(cfg.get("io", {}).get("hash_max_mb", 512))

    exp.log(f"[实验] {exp.dir}")
    if config_path:
        exp.log(f"[配置] {display(config_path)}")
    exp.log(f"[步骤] " + " → ".join(f"{i.name}/{i.version}" for i in items))
    if opts.dry_run:
        exp.log("[模式] --dry-run：只打印计划，不创建目录、不执行")

    if not opts.dry_run:
        exp.ensure()
        exp.write_config_snapshot(dump_yaml(cfg))
        if existed and not opts.resume:
            exp.log(f"[注意] 目录已存在，同名产物会被覆盖（--resume 可跳过已完成步骤）")

    produced: dict[str, dict[str, list[Path]]] = {}
    entries: list[dict] = []
    failed: str | None = None

    for index, item in enumerate(items, start=1):
        header = f"[{index}/{len(items)}] {item.name}/{item.version}  {item.title}"
        resolved = resolve_inputs(item, cfg, exp, produced, link=not opts.dry_run)
        for name, (path, source) in resolved.items():
            if path is None:
                raise _missing_input_error(item, name, cfg)

        placeholder = {name: (str(path) if path else None) for name, (path, _) in resolved.items()}
        params = exp.expand(item.spec.get("params") or {}, placeholder)

        ctx = PlanCtx(name=item.name, version=item.version, cfg=cfg, exp=exp,
                      params=params, inputs={k: v[0] for k, v in resolved.items()}, spec=item.spec)
        if item.extra:
            argv, raw_outputs = _plan_extra(ctx)
        else:
            assert item.sdef is not None
            argv, raw_outputs = item.sdef.plan(ctx)
        outputs = _match_outputs(raw_outputs)

        python = str(cfg.get("python") or sys.executable)
        if item.extra and item.spec.get("cmd"):
            cmd = [exp.expand(str(x), placeholder) for x in item.spec["cmd"]]
        else:
            script = item.sdef.script if item.sdef else str(item.spec["script"])
            cmd = [python, str(resolve_path(script)), *argv]

        exp.section(header)
        for name, (path, source) in resolved.items():
            exp.log(f"  输入 {name:8s} [{source}] {display(path) if path else '-'}")
        exp.log(f"  命令 {format_cmd(cmd)}")

        entry: dict[str, Any] = {
            "name": item.name,
            "version": item.version,
            "title": item.title,
            "status": "planned" if opts.dry_run else "pending",
            "cmd": cmd,
            "cmd_line": format_cmd(cmd),
            "params": params,
            "inputs": {name: {"path": str(p) if p else None, "source": src}
                       for name, (p, src) in resolved.items()},
            "outputs": {},
            "log": None,
        }

        if opts.dry_run:
            exp.log(f"  产出 " + ", ".join(f"{k}={display(v[0]) if v else '(未生成)'}"
                                           for k, v in outputs.items()))
            for name, paths in outputs.items():
                entry["outputs"][name] = [{"path": str(p), "rel": exp.relative(p)} for p in paths]
            entries.append(entry)
            produced[item.name] = outputs      # 让后续 step 的 dry-run 显示真实上游
            continue

        outputs_exist = all(paths and all(p.exists() for p in paths) for paths in outputs.values())
        if opts.resume and outputs_exist and not opts.force:
            entry["status"] = "reused"
            for name, paths in outputs.items():
                entry["outputs"][name] = [exp.fingerprint(p, do_hash=do_hash, max_mb=hash_max)
                                          for p in paths]
            exp.log("  [跳过] --resume：产物已存在  " + exp.describe_outputs(
                [p for paths in outputs.values() for p in paths if p.exists()]))
            entries.append(entry)
            exp.record_step(entry)
            produced[item.name] = outputs
            continue

        log_path = exp.logs / f"{index - 1:02d}_{item.name}_{item.version}.log"
        entry["log"] = exp.relative(log_path)
        started = datetime.now().isoformat(timespec="seconds")
        code, seconds = run_command(cmd, log_path=log_path, cwd=REPO_ROOT,
                                    env=cfg.get("env") or {}, echo=opts.echo)
        entry["started"], entry["seconds"] = started, round(seconds, 2)
        entry["exit_code"] = code

        # 通配产物（如 step2_1k_*.npz）在运行前匹配不到，成功后重新匹配一次
        final_outputs = _match_outputs(raw_outputs) if code == 0 else outputs
        for name, paths in final_outputs.items():
            entry["outputs"][name] = [exp.fingerprint(p, do_hash=do_hash, max_mb=hash_max)
                                      for p in paths]
        produced[item.name] = final_outputs

        if code != 0:
            entry["status"] = "failed"
            entry["error"] = f"退出码 {code}，见 {entry['log']}"
            entries.append(entry)
            exp.record_step(entry)
            failed = f"{item.name}/{item.version} 退出码 {code}"
            exp.log(f"  [失败] {failed}")
            if not opts.keep_going:
                break
        else:
            missing = [name for name, paths in final_outputs.items()
                       if not paths or not all(p.exists() for p in paths)]
            entry["status"] = "ok" if not missing else "ok(产物缺失)"
            if missing:
                entry["error"] = f"声明的产物没有生成: {', '.join(missing)}"
            report = _read_reports(item, final_outputs)
            if report:
                entry["report"] = report
            entries.append(entry)
            exp.record_step(entry)
            exp.log(f"  [完成] {seconds:.1f}s  " + exp.describe_outputs(
                [p for paths in outputs.values() for p in paths]))

    # 汇总取合并后的 manifest：--resume 时被跳过的 step 仍保留上一次的指标
    summary = _build_summary(exp.manifest["steps"] if not opts.dry_run else entries)
    if not opts.dry_run:
        exp.set_summary(summary)
        exp.write_run_sh(env=cfg.get("env") or {})
        exp.finish("failed" if failed else "ok", error=failed)

    _print_summary(exp, entries, summary, dry_run=opts.dry_run)
    if failed:
        exp.log(f"[失败] {failed}")
        return 1
    if not opts.dry_run:
        exp.log(f"[完成] manifest: {exp.relative(exp.dir / 'manifest.json')}")
    return 0


def _plan_extra(ctx: PlanCtx) -> tuple[list[str], dict[str, Any]]:
    """任意脚本：params 直接变成命令行参数，outputs 收进实验目录。"""
    spec = ctx.spec
    outputs_spec = spec.get("outputs") or []
    if isinstance(outputs_spec, Mapping):
        raw_outputs = {str(k): ctx.exp.expand(str(v)) for k, v in outputs_spec.items()}
    else:
        raw_outputs = {f"out{i}": ctx.exp.expand(str(v)) for i, v in enumerate(outputs_spec)}
    if not raw_outputs:
        raw_outputs = {"results": ctx.exp.results}
    if spec.get("cmd"):
        return [], raw_outputs
    argv = list(flags(ctx.params, skip=frozenset(), where=f"extra_steps.{ctx.name}.params"))
    for arg in spec.get("args") or []:
        argv.append(ctx.exp.expand(str(arg)))
    return argv, raw_outputs


def _build_summary(entries: Iterable[Mapping]) -> dict:
    summary: dict[str, Any] = {}
    for entry in entries:
        report = entry.get("report")
        if not report:
            continue
        name = str(entry["name"])
        if name == "step2" and "summary" in report and isinstance(report["summary"], Mapping):
            runs = report["summary"].get("runs", {})
            summary[name] = {
                variant: {"results": run.get("results", {}),
                          "sizes": run.get("sizes"),
                          "seconds": run.get("seconds")}
                for variant, run in runs.items()
            }
        elif name == "step1" and "report" in report:
            rep = report["report"]
            summary[name] = {"funnel": rep.get("funnel"), "checks": rep.get("checks"),
                             "loss": rep.get("loss"), "split": rep.get("split"),
                             "rules": rep.get("rules")}
        else:
            summary[name] = report
    return summary


def _print_summary(exp: Experiment, entries: list[dict], summary: Mapping, dry_run: bool = False) -> None:
    exp.section("实验汇总" if not dry_run else "计划汇总（--dry-run）")
    exp.log(f"实验目录  {exp.relative(exp.dir)}")
    exp.log("步骤")
    for entry in entries:
        status = entry.get("status", "?")
        secs = f"{entry.get('seconds', 0):8.1f}s" if entry.get("seconds") else " " * 9
        outs = [p["rel"] for paths in entry.get("outputs", {}).values() for p in paths]
        shown = ", ".join(outs[:3]) + (" …" if len(outs) > 3 else "")
        exp.log(f"  {entry['name']:8s} {entry['version']:14s} {status:14s} {secs}  {shown}")

    step2 = summary.get("step2") if isinstance(summary, Mapping) else None
    if step2:
        exp.log("")
        exp.log("模型指标（全部算在 split=test 的行上；split=train 的行只用于拟合）")
        exp.log(f"  {'变体':28s} {'模型':14s} {'RMSE':>8s} {'Spearman':>9s} {'n_scored':>10s}")
        for variant, run in step2.items():
            for model, metrics in (run.get("results") or {}).items():
                exp.log(f"  {variant[:28]:28s} {model:14s} "
                        f"{metrics.get('rmse', float('nan')):8.4f} "
                        f"{metrics.get('spearman', float('nan')):9.4f} "
                        f"{metrics.get('n_scored', 0):10,d}")
    step1 = summary.get("step1") if isinstance(summary, Mapping) else None
    if step1 and step1.get("funnel"):
        f = step1["funnel"]
        split = step1.get("split") or {}
        bad = [k for k, v in (step1.get("checks") or {}).items() if not v]
        exp.log("")
        if split:
            exp.log(f"划分方式  {split.get('method')}  train_frac={split.get('train_frac')}  "
                    f"seed={split.get('seed')}  单位={split.get('unit')}"
                    f"（{split.get('n_units', 0):,}，train {split.get('n_train_units', 0):,} / "
                    f"test {split.get('n_test_units', 0):,}）")
        rules = step1.get("rules") or {}
        if f.get("pairs_after_epoch_split") is not None:
            # step1/pairbest：入口是 play，出口是一对一行；不要套用 clean 的漏斗行。
            exp.log(f"漏斗      step0 {f.get('step0_rows', 0):,} 行 / "
                    f"{f.get('step0_pairs', 0):,} 对"
                    f" → 玩家<{rules.get('min_plays_per_player', 0)} 条 -"
                    f"{f.get('dropped_low_play_rows', 0):,}"
                    f" → 空窗拆出 +{f.get('extra_segments', 0):,} 个玩家段"
                    f" → 单次对 -{f.get('pairs_below_min_plays', 0):,}"
                    f" → ACC<={rules.get('acc_floor', 0):g} -{f.get('dropped_low_acc_pairs', 0):,}"
                    f" → ACC>{rules.get('max_best_acc', 0):g} -{f.get('dropped_high_acc_pairs', 0):,}"
                    f" → 保留 {f.get('kept_pairs', 0):,} 对 "
                    f"(step0 行的 {f.get('retained_frac', 0):.2%})"
                    f" → train {f.get('train_rows', 0):,} / test {f.get('test_rows', 0):,}"
                    + (f"  校验 {len(bad)} 项未通过: {bad}" if bad else "  校验全部通过"))
        else:
            exp.log(f"行数      step0 {f.get('step0_rows', 0):,}"
                    f" → 删玩家(<{rules.get('min_plays_per_player', 0)}) "
                    f"{f.get('dropped_low_play_rows', 0):,}"
                    f" → test {f.get('test_rows', 0):,} / train 候选 {f.get('train_candidates', 0):,}"
                    f" → 删 ACC {f.get('dropped_acc', 0):,}"
                    f" → 删热身 {f.get('dropped_earliest', 0):,}"
                    f" → train {f.get('train_rows', 0):,}（占全部 {f.get('retained_frac', 0):.1%}）"
                    + (f"  校验 {len(bad)} 项未通过: {bad}" if bad else "  校验全部通过"))
