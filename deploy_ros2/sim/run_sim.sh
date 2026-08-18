#!/usr/bin/env bash
# 运行 Isaac Sim headless 仿真脚本（h1_sim.py）。
# 关键：unset PYTHONPATH（避免 .bashrc 里系统 ROS 污染）；不 source 系统 ROS。
# 用法: ./run_sim.sh <script.py> [args...]
# 环境变量: ISAAC_PYTHON 指定 isaac 环境的 python（默认探测 ~/miniconda3）
set -euo pipefail

SCRIPT="$1"; shift

# 定位 isaac 环境 python
if [ -n "${ISAAC_PYTHON:-}" ]; then
  PYTHON_BIN="$ISAAC_PYTHON"
elif [ -x ~/miniconda3/envs/isaac/bin/python ]; then
  PYTHON_BIN=~/miniconda3/envs/isaac/bin/python
else
  echo "ERROR: 找不到 isaac 环境 python，请设置 ISAAC_PYTHON" >&2
  exit 1
fi

export OMNI_KIT_ACCEPT_EULA=Y
unset PYTHONPATH
unset AMENT_PREFIX_PATH || true
unset ROS_PREFIX_PATH || true
unset COLCON_PREFIX_PATH || true
unset RMW_IMPLEMENTATION || true

# 剔除系统 ROS2 库路径，避免与 isaacsim 内置 Jazzy rclpy 的 C 扩展冲突
export LD_LIBRARY_PATH="$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '^/opt/ros' | paste -sd:)"
# 追加内置 Jazzy 库路径
SITE_PKG=$("$PYTHON_BIN" -c 'import site; print(site.getsitepackages()[0])')
BRIDGE_LIB="${ISAAC_BRIDGE_LIB:-$SITE_PKG/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib}"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$BRIDGE_LIB"

exec "$PYTHON_BIN" -u "$SCRIPT" "$@"
