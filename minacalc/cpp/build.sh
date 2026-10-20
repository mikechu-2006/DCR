#!/usr/bin/env bash
# 编译逐区间难度导出器（需要 Etterna MinaCalc 源码树）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/external/minacalc-standalone"
g++ -std=c++17 -O2 -I "$SRC" \
    "$ROOT/minacalc/cpp/export_itv.cpp" \
    "$SRC/MinaCalc/MinaCalc.cpp" \
    -o "$SRC/build/export_itv"
echo "built: $SRC/build/export_itv"
