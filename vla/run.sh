#!/bin/bash
# 重跑 vla_demo 直到相机正常，成功即备份。
# 安全规则：绝不预先删除任何备份；只有新成果验证成功后，才覆盖旧的 good 备份。
# Windows 专用（依赖 tasklist/taskkill）；PYTHON_BIN 指定 isaac 环境 python。
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON_BIN:-python}"
DEMO="$ROOT/vla/vla_demo.py"
FRAMES="$ROOT/assets/demo/frames_vla"
MP4="$ROOT/assets/demo/vla_demo.mp4"
GOOD_FRAMES="$ROOT/assets/demo/frames_vla_good"
GOOD_MP4="$ROOT/assets/demo/vla_demo_good.mp4"
LOG="$ROOT/vla_demo.log"

for i in 1 2 3 4; do
  echo "===== 尝试 $i/4 ====="
  # 只清本次运行残留帧（绝不动 good 备份）
  rm -f "$FRAMES"/frame_*.png "$MP4"
  # 清残留 python 进程（防多实例互抢 GPU 导致黑帧）
  tasklist 2>/dev/null | grep -i python | awk '{print $2}' | while read pid; do taskkill //PID $pid //F 2>/dev/null; done
  sleep 2
  CUDA_VISIBLE_DEVICES=0 "$PY" "$DEMO" > "$LOG" 2>&1
  code=$?

  # 判断是否成功：帧数足够(>=150) 且 中间帧亮度正常（黑帧则判定失败）
  RESULT=$("$PY" -c "
import numpy as np
from PIL import Image
import glob
f = sorted(glob.glob(r'$FRAMES/frame_*.png'))
if len(f) < 150:
    print('SHORT')
elif np.array(Image.open(f[len(f)//2])).astype(float).mean() > 5.0:
    print('OK')
else:
    print('DARK')
" 2>/dev/null)

  echo "退出码=$code 帧数=$RESULT"
  if [ "$RESULT" = "OK" ]; then
    echo "成功！备份成果..."
    rm -rf "$GOOD_FRAMES"
    cp -r "$FRAMES" "$GOOD_FRAMES"
    cp "$MP4" "$GOOD_MP4"
    echo "备份完成: $GOOD_FRAMES/ + $GOOD_MP4"
    exit 0
  fi
  echo "相机黑帧或失败，重试..."
done
echo "4 次都失败（本次帧保留在 assets/demo/frames_vla/ 供检查）"
exit 1
