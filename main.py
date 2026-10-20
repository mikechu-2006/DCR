#!/usr/bin/env python
"""DCR 统一入口 —— 一个 YAML 跑完一次实验。

    python main.py -c configs/exp_std_1k.yaml          # 按配置跑
    python main.py -c configs/exp_std_1k.yaml --dry-run  # 只看要跑什么
    python main.py --list-steps                        # 看内置 step 版本

细节见 docs/procedure.md：main.py 按 YAML 调用各 step 程序，指定实验目录，
把实验参数、中间文件与结果集中放在 experiments/<时间戳>/ 下。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dcr.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
