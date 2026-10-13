"""step 注册表：把 YAML 里的 step/version/params 翻译成一条完整命令行。

一个 step 可以有多个版本（procedure.md：每个 step 可能存在多种版本，或允许
使用多种不同参数）。这里集中声明每个版本的：

  * script     —— 实现在 scripts/ 下的哪个程序；
  * inputs     —— 逻辑输入名 → (上游 step, 上游产物名)；
  * cache      —— 该产物在 cache_dir 里的习惯文件名（不跑上游时自动取用）；
  * report     —— 运行时把哪个产物当机器可读报告读出来汇总；
  * reserved   —— 由编排层接管的参数（路径/标签），用户再写会明确报错；
  * plan       —— 生成 argv 与产物路径。

参数命名规则：YAML 里写 snake_case（如 acc_floor），自动变成 --acc-floor；
值为 null/false 表示不传；列表默认按逗号拼成一个参数（list_style=repeat 时
重复该 flag）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .errors import DcrError
from .experiment import Experiment
from .paths import resolve_path


# --------------------------------------------------------------------------- 上下文
@dataclass
class PlanCtx:
    """生成一条 step 命令所需的全部信息。"""

    name: str
    version: str
    cfg: Mapping
    exp: Experiment
    params: dict
    inputs: dict[str, Path | None] = field(default_factory=dict)
    spec: dict = field(default_factory=dict)

    @property
    def tag(self) -> str:
        return str(self.cfg.get("tag", "1k"))

    def p(self, key: str, default: Any = None) -> Any:
        value = self.params.get(key, default)
        return default if value is None else value

    def path(self, key: str) -> Path:
        value = self.params.get(key)
        if value is None or value == "":
            raise DcrError(f"steps.{self.name}.params.{key} 缺失")
        return resolve_path(value)

    def out_format(self) -> str:
        return str(self.cfg.get("io", {}).get("out_format", "parquet"))


@dataclass(frozen=True)
class StepDef:
    name: str
    version: str
    script: str
    plan: Callable[[PlanCtx], tuple[list[str], dict[str, Any]]]
    title: str = ""
    inputs: Mapping[str, tuple[str, str] | None] = field(default_factory=dict)
    optional: frozenset[str] = frozenset()
    cache: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    report: frozenset[str] = frozenset()
    reserved: frozenset[str] = frozenset()
    list_style: Mapping[str, str] = field(default_factory=dict)


# ------------------------------------------------------------------------- 参数翻译
def flags(params: Mapping[str, Any], skip: frozenset[str] | set[str],
           list_style: Mapping[str, str] = {}, where: str = "") -> list[str]:
    argv: list[str] = []
    for key, value in params.items():
        if key in skip or value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, (list, tuple)):
            items = [str(v) for v in value]
            if not items:
                continue
            if list_style.get(key) == "repeat":
                for item in items:
                    argv += [flag, item]
            else:
                argv += [flag, ",".join(items)]
        elif isinstance(value, Mapping):
            raise DcrError(
                f"{where or 'params'}: 参数 {key} 是映射，无法转成命令行；"
                f"请用字符串/数字/布尔/列表")
        elif value is True:
            argv.append(flag)
        else:
            argv += [flag, str(value)]
    return argv


def _check_reserved(ctx: PlanCtx, reserved: frozenset[str]) -> None:
    bad = sorted(set(ctx.params) & reserved)
    if bad:
        raise DcrError(
            f"steps.{ctx.name}.params 里的 {', '.join(bad)} 由编排层接管，不能在 params 里设置。\n"
            f"  输入路径请写 steps.{ctx.name}.inputs，或把产物放进 cache_dir 自动取用；"
            f"输出路径由实验目录决定（artifacts/ 与 results/）")


# ----------------------------------------------------------------- step0 / preprocess
def _plan_step0_preprocess(ctx: PlanCtx) -> tuple[list[str], dict[str, Any]]:
    reserved = frozenset({"tag", "out", "meta_out", "work_dir", "no_meta", "whitelist"})
    _check_reserved(ctx, reserved)
    fmt = ctx.out_format()
    suffix = "parquet" if fmt == "parquet" else "csv"
    plays = ctx.exp.artifacts / f"step0_{ctx.tag}.{suffix}"
    meta = ctx.exp.artifacts / f"beatmap_meta_{ctx.tag}.csv"
    scratch = ctx.exp.artifacts / "_scratch" / f"step0_{ctx.tag}"
    argv = ["--dump-dir", str(ctx.path("dump_dir")),
            "--tag", ctx.tag,
            "--out", str(plays),
            "--meta-out", str(meta),
            "--work-dir", str(scratch)]
    # 白名单表（step0/whitelist 的产物，或 cache_dir 里的 whitelist_{tag}.csv）：
    # 给了就剔除 cacheable=False 的谱面；没给就按 dump 的全量 4K 白名单跑
    whitelist = ctx.inputs.get("whitelist")
    if whitelist is not None:
        argv += ["--whitelist", str(whitelist)]
    argv += flags(ctx.params, skip=reserved | {"dump_dir"}, where=f"steps.{ctx.name}.params")
    return argv, {"plays": plays, "meta": meta}


# -------------------------------------------------------------- step0 / chart_content
def _plan_step0_chart_content(ctx: PlanCtx) -> tuple[list[str], dict[str, Any]]:
    reserved = frozenset({"tag", "out", "meta_dir"})
    _check_reserved(ctx, reserved)
    out = ctx.exp.artifacts / "chart_content_cm3p.parquet"
    meta = ctx.inputs.get("meta")
    argv = ["--tag", ctx.tag, "--out", str(out), "--meta-dir", str(meta.parent if meta else ctx.exp.artifacts)]
    argv += flags(ctx.params, skip=reserved, where=f"steps.{ctx.name}.params")
    return argv, {"content": out}


# ------------------------------------------------------------------ step0 / fetch_osu
def _plan_step0_fetch_osu(ctx: PlanCtx) -> tuple[list[str], dict[str, Any]]:
    reserved = frozenset({"state"})
    _check_reserved(ctx, reserved)
    # 共享缓存（cfg.osu_cache_dir，默认 data/osu_cache），不是每个实验一份：
    # 600 MB 的 .osu 语料要跨实验复用，重下没有意义
    default_cache = resolve_path(ctx.cfg.get("osu_cache_dir") or "data/osu_cache")
    out_dir = resolve_path(ctx.p("out_dir", default_cache))
    state = out_dir / "_fetch_state.json"
    argv = ["--out-dir", str(out_dir), "--state", str(state)]
    argv += flags(ctx.params, skip=reserved | {"out_dir"}, where=f"steps.{ctx.name}.params")
    return argv, {"osu_cache": out_dir}


# --------------------------------------------------------------- step0 / whitelist
def _plan_step0_whitelist(ctx: PlanCtx) -> tuple[list[str], dict[str, Any]]:
    reserved = frozenset({"tag", "meta", "cache", "out", "manifest_out", "dropped_out",
                          "json", "state"})
    _check_reserved(ctx, reserved)
    meta = ctx.inputs.get("meta")
    if meta is None:
        raise DcrError(f"steps.{ctx.name} 需要输入 meta（step0 的 beatmap_meta 表）；"
                       f"先跑 step0/preprocess，或把 beatmap_meta_{ctx.tag}.csv 放进 cache_dir")
    out = ctx.exp.artifacts / f"whitelist_{ctx.tag}.csv"
    report = ctx.exp.results / f"whitelist_{ctx.tag}_report.json"
    cache = resolve_path(ctx.cfg.get("osu_cache_dir") or "data/osu_cache")
    argv = ["--tag", ctx.tag, "--meta", str(meta), "--cache", str(cache),
            "--out", str(out), "--json", str(report)]
    argv += flags(ctx.params, skip=reserved, where=f"steps.{ctx.name}.params")
    return argv, {"whitelist": out, "report": report}


# ---------------------------------------------------------------- step0 / fetch_tags
def _plan_step0_fetch_tags(ctx: PlanCtx) -> tuple[list[str], dict[str, Any]]:
    reserved = frozenset({"tag", "out_dir", "meta", "work_dir"})
    _check_reserved(ctx, reserved)
    out_dir = ctx.exp.artifacts
    meta = ctx.inputs.get("meta")
    # 页面缓存放在 cache_dir 下：换实验目录重跑时能断点续爬（--refresh 可强制重抓）
    default_work = resolve_path(
        Path(str(ctx.cfg.get("cache_dir", "data/processed"))) / f"step0f_{ctx.tag}")
    work = resolve_path(ctx.p("work_dir", default_work))
    argv = ["--tag", ctx.tag, "--out-dir", str(out_dir), "--work-dir", str(work)]
    if meta is not None:
        argv += ["--meta", str(meta)]
    argv += flags(ctx.params, skip=reserved, where=f"steps.{ctx.name}.params")
    return argv, {"tags": out_dir / f"chart_user_tags_{ctx.tag}.parquet",
                  "vocab": out_dir / f"chart_user_tag_vocab_{ctx.tag}.csv"}


# --------------------------------------------------------------- step0 / validate_mix
def _plan_step0_validate_mix(ctx: PlanCtx) -> tuple[list[str], dict[str, Any]]:
    reserved = frozenset({"computed", "json"})
    _check_reserved(ctx, reserved)
    computed = ctx.inputs.get("computed")
    if computed is None:
        raise DcrError(f"steps.{ctx.name} 需要输入 computed（通常是 step0/chart_content 的产物），"
                       f"请在 inputs 里指定或在 cache_dir 里放好内容向量表")
    report = ctx.exp.results / "cm3p_mix_validation.json"
    argv = ["--computed", str(computed), "--json", str(report)]
    argv += flags(ctx.params, skip=reserved, where=f"steps.{ctx.name}.params")
    return argv, {"report": report}


# ------------------------------------------------------------------- step1 / clean
def _plan_step1_clean(ctx: PlanCtx) -> tuple[list[str], dict[str, Any]]:
    reserved = frozenset({"tag", "in", "inp", "out", "no_csv", "no_parquet"})
    _check_reserved(ctx, reserved)
    src = ctx.inputs.get("plays")
    if src is None:
        raise DcrError(f"steps.{ctx.name} 需要输入 plays（step0 的 play 表）")
    formats = str(ctx.p("formats", "both")).lower()
    if formats not in ("both", "csv", "parquet"):
        raise DcrError(f"steps.{ctx.name}.params.formats 只能是 both/csv/parquet，得到 {formats!r}")
    base = ctx.exp.artifacts / f"step1_{ctx.tag}"
    argv = ["--tag", ctx.tag, "--in", str(src), "--out", str(base.with_suffix(".csv"))]
    if formats == "csv":
        argv.append("--no-parquet")   # 只留 CSV（step1 默认两种格式都写）
    if formats == "parquet":
        argv.append("--no-csv")
    outputs: dict[str, Any] = {"report": base.with_suffix(".json")}
    if formats in ("both", "parquet"):
        outputs["plays"] = base.with_suffix(".parquet")
    else:
        outputs["plays"] = base.with_suffix(".csv")
    if formats == "both":
        outputs["plays_csv"] = base.with_suffix(".csv")
    argv += flags(ctx.params, skip=reserved | {"formats"}, where=f"steps.{ctx.name}.params")
    return argv, outputs


# ------------------------------------------------------------------ step2 / predict
def _plan_step2_predict(ctx: PlanCtx) -> tuple[list[str], dict[str, Any]]:
    reserved = frozenset({"tag", "train_path", "proc_dir", "out_dir", "content_path"})
    _check_reserved(ctx, reserved)   # out_prefix / target 由下面的 ctx.p() 读取
    train = ctx.inputs.get("train")
    if train is None:
        raise DcrError(f"steps.{ctx.name} 需要输入 train（step1 的输出表，带 split 列）")
    content = ctx.inputs.get("content")
    prefix = str(ctx.p("out_prefix", "step2"))
    target = str(ctx.p("target", "loss"))
    argv = ["--tag", ctx.tag,
            "--train-path", str(train),
            "--proc-dir", str(ctx.exp.artifacts),
            "--out-dir", str(ctx.exp.results),
            "--out-prefix", prefix]
    if content is not None:
        argv += ["--content-path", str(content)]
    argv += flags(ctx.params, skip=reserved | {"out_prefix", "target"},
                   where=f"steps.{ctx.name}.params")
    outputs = {
        "summary": ctx.exp.results / f"{prefix}_{ctx.tag}_summary_{target}.json",
        "npz": ctx.exp.results / f"{prefix}_{ctx.tag}_*.npz",
    }
    return argv, outputs


# ------------------------------------------------------------------------ 注册表
REGISTRY: dict[tuple[str, str], StepDef] = {}


def register(step_def: StepDef) -> StepDef:
    REGISTRY[(step_def.name, step_def.version)] = step_def
    return step_def


register(StepDef(
    name="step0", version="preprocess", script="scripts/step0_preprocess.py",
    title="预处理：raw dump → play 表 + 谱面元数据",
    plan=_plan_step0_preprocess,
    # 可选输入：cache_dir 里有 whitelist_{tag}.csv 时自动取用，剔除不可缓存谱面
    inputs={"whitelist": ("step0", "whitelist")},
    optional=frozenset({"whitelist"}),
    cache={"plays": ("step0_{tag}.parquet", "step0_{tag}.csv"),
           "meta": ("beatmap_meta_{tag}.csv",)},
))

register(StepDef(
    name="step0", version="chart_content", script="scripts/step0c_chart_content.py",
    title="谱面内容向量：beatmap_id → CM3P embedding",
    plan=_plan_step0_chart_content,
    inputs={"meta": ("step0", "meta")},
    optional=frozenset({"meta"}),
    cache={"content": ("chart_content_cm3p.parquet",)},
))

register(StepDef(
    name="step0", version="fetch_osu", script="scripts/step0d_fetch_osu.py",
    title="抓取 .osu 文件（CM3P 自算向量的前置步骤）",
    plan=_plan_step0_fetch_osu,
    cache={"osu_cache": ("osu_cache",)},
))

register(StepDef(
    name="step0", version="whitelist", script="scripts/step0g_whitelist.py",
    title="白名单表 + .osu 缓存状态（不可缓存的谱面在这里标出，step0 据此剔除）",
    plan=_plan_step0_whitelist,
    inputs={"meta": ("step0", "meta")},
    cache={"whitelist": ("whitelist_{tag}.csv",)},
    report=frozenset({"report"}),
))

register(StepDef(
    name="step0", version="fetch_tags", script="scripts/step0f_fetch_tags.py",
    title="谱面标签：爬 osu! 谱面页的玩家投票 tag",
    plan=_plan_step0_fetch_tags,
    inputs={"meta": ("step0", "meta")},
    optional=frozenset({"meta"}),
))

register(StepDef(
    name="step0", version="validate_mix", script="scripts/step0e_validate_mix.py",
    title="校验不同来源的内容向量能否混用",
    plan=_plan_step0_validate_mix,
    inputs={"computed": ("step0", "content")},
    report=frozenset({"report"}),
))

register(StepDef(
    name="step1", version="clean", script="scripts/step1_clean.py",
    title="划分 + 清洗：train/test 分离，然后 ACC 门槛 + 每玩家热身截断",
    plan=_plan_step1_clean,
    inputs={"plays": ("step0", "plays")},
    cache={"plays": ("step1_{tag}.parquet", "step1_{tag}.csv")},
    report=frozenset({"report"}),
))

register(StepDef(
    name="step2", version="predict", script="scripts/step2_predict.py",
    title="训练 + 推理：低秩回归与指标",
    plan=_plan_step2_predict,
    inputs={"train": ("step1", "plays"), "content": ("step0", "content")},
    optional=frozenset({"content"}),
    cache={"train": ("step1_{tag}.parquet", "step1_{tag}.csv"),
           "test": ("step0_{tag}.parquet", "step0_{tag}.csv"),
           "content": ("chart_content_cm3p.parquet",)},
    report=frozenset({"summary"}),
))


def available_versions(step: str) -> list[str]:
    return sorted(v for (n, v) in REGISTRY if n == step)


def get(step: str, version: str) -> StepDef | None:
    return REGISTRY.get((step, version))


def known_steps() -> list[tuple[str, str, str]]:
    return sorted((d.name, d.version, d.title or d.script) for d in REGISTRY.values())