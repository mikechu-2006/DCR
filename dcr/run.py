"""子进程执行：把 step 程序的输出同时打到终端与实验日志。"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

from .paths import REPO_ROOT


def format_cmd(cmd: list[str]) -> str:
    return shlex.join(str(c) for c in cmd)


def run_command(
    cmd: list[str],
    log_path: Path | None = None,
    cwd: Path | None = None,
    env: dict | None = None,
    echo: bool = True,
) -> tuple[int, float]:
    """跑一条命令，逐行转发输出；返回 (returncode, seconds)。

    - stdout 与 stderr 合并后按行流出，长任务（step0 解析 dump）也能实时看到进度；
    - 同时写进 log_path（实验目录 logs/ 下），失败后可直接翻日志。
    """
    started = time.time()
    fh = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = log_path.open("w", encoding="utf-8")
        fh.write(f"# {format_cmd(cmd)}\n# cwd={cwd or REPO_ROOT}\n\n")
        fh.flush()

    full_env = dict(os.environ)
    if env:
        full_env.update({str(k): str(v) for k, v in env.items()})
    full_env.setdefault("PYTHONUNBUFFERED", "1")

    proc = subprocess.Popen(
        [str(c) for c in cmd],
        cwd=str(cwd or REPO_ROOT),
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if echo:
                print(line, end="", flush=True)
            if fh is not None:
                fh.write(line)
                fh.flush()
        code = proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        try:
            code = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = -2
        raise
    finally:
        if fh is not None:
            fh.write(f"\n# exit={proc.returncode}  {time.time() - started:.1f}s\n")
            fh.close()
    return code, time.time() - started
