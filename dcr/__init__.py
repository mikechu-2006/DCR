"""DCR 实验编排包（experiment orchestration）。

把「一次实验要跑多个脚本 + 一长串参数」收敛成一条命令：

    python main.py -c configs/<exp>.yaml

一个 YAML 描述每个 step 的版本与参数；运行产生的参数快照、日志、中间文件与
结果全部落在 experiments/<时间戳>[_{name}]/ 里，manifest.json 记录每一步的
完整命令行、耗时、产物指纹与指标。

step 的实现仍然在 scripts/ 下（一个 step 可以有多个版本），本包只负责按依赖
顺序调用它们、把路径接起来、把产物收进实验目录。见 docs/procedure.md。
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
