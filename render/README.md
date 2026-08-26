# 渲染

离线回放行走日志 → 2K 视频。纯回放，不跑物理、不连 ROS2。

```bash
python render/render.py --log assets/demo/walk_log.npz --out assets/demo/walk_vla.mp4 --fps 30
```

## 前置

Windows `isaac` 环境（GPU）。输入 `walk_log.npz` 由 WSL2 仿真端生成
（`deploy_ros2/sim/`）。
