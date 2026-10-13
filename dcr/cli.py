"""main.py 的命令行界面。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, steps as step_registry
from .config import dump_yaml, load_config, resolve_config
from .errors import DcrError
from .paths import REPO_ROOT, resolve_path
from .pipeline import Options, run


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="main.py",
        description="DCR 实验入口：一个 YAML = 一次实验 = 一个 experiments_{时间戳}/ 目录。",
        epilog="示例:  python main.py -c configs/exp_std_1k.yaml\n"
               "       python main.py -c configs/exp_std_1k.yaml --set steps.step2.params.epochs=5 --dry-run\n"
               "       python main.py --list-steps",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", type=Path, required=False, default=None,
                    help="实验 YAML（必填，除非 --list-steps/-V），如 configs/exp_std_1k.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="临时覆盖配置，可重复；KEY 用点号路径，如 steps.step2.params.dim=32")
    ap.add_argument("-t", "--tag", default=None, help="覆盖 tag（等价 --set tag=...）")
    ap.add_argument("-n", "--name", default=None, help="覆盖实验名（目录后缀）")
    ap.add_argument("--exp-dir", type=Path, default=None,
                    help="指定实验目录（默认 experiments_{时间戳}[_{name}]）")
    ap.add_argument("--exp-root", type=Path, default=None, help="实验目录的父目录（默认仓库根）")
    ap.add_argument("--only", default="", help="只跑这些 step（逗号分隔）")
    ap.add_argument("--skip", default="", help="跳过这些 step（逗号分隔）")
    ap.add_argument("--resume", action="store_true",
                    help="产物已存在就跳过该 step（配合 --exp-dir 续跑）")
    ap.add_argument("--force", action="store_true", help="即使 --resume 也重跑")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划与命令，不执行、不建目录")
    ap.add_argument("--keep-going", action="store_true", help="某步失败后继续后面的 step")
    ap.add_argument("--no-hash", action="store_true", help="不计算产物 sha256")
    ap.add_argument("--print-config", action="store_true", help="打印解析后的完整配置后退出")
    ap.add_argument("-q", "--quiet", action="store_true", help="不转发子进程输出到终端（仍然写日志）")
    ap.add_argument("--list-steps", action="store_true", help="列出内置 step 版本后退出")
    ap.add_argument("-V", "--version", action="version", version=f"DCR {__version__}")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_steps:
        print("内置 step 版本：")
        for name, version, title in step_registry.known_steps():
            print(f"  {name:8s} {version:14s} {title}")
        print("\n其余任意脚本可用 extra_steps 挂进同一套编排（见 configs/default.yaml）。")
        return 0

    if args.config is None:
        print("错误: 请用 -c/--config 指定实验配置，例如 configs/exp_std_1k.yaml\n"
              "      python main.py --list-steps 可看内置 step 版本", file=sys.stderr)
        return 2
    config_path = args.config
    overrides = list(args.overrides)
    if args.tag:
        overrides.append(f"tag={args.tag}")
    if args.name:
        overrides.append(f"name={args.name}")

    try:
        cfg = resolve_config(load_config(config_path), overrides)
        if args.print_config:
            print(dump_yaml(cfg), end="")
            return 0
        opts = Options(
            dry_run=args.dry_run,
            only=tuple(x.strip() for x in args.only.split(",") if x.strip()),
            skip=tuple(x.strip() for x in args.skip.split(",") if x.strip()),
            exp_dir=args.exp_dir,
            root=Path(args.exp_root).resolve() if args.exp_root else REPO_ROOT,
            resume=args.resume,
            force=args.force,
            hash=False if args.no_hash else None,
            echo=not args.quiet,
            keep_going=args.keep_going,
        )
        code = run(cfg, resolve_path(config_path), opts)   # 相对路径按仓库根解析
    except DcrError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已中断（Ctrl-C）", file=sys.stderr)
        return 130
    return code


def entrypoint() -> None:
    raise SystemExit(main())
