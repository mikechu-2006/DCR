"""编排层异常。"""

from __future__ import annotations


class DcrError(RuntimeError):
    """配置非法、输入缺失、步骤失败等可预期的错误（只打印一句话，不带调用栈）。"""
