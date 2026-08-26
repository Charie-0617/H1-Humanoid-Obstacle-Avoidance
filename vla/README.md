# VLA 闭环（Windows 仿真）

YOLO 真视觉 → VLM 决策 → RL 执行 → 路径归位。

## 脚本

- `vla_demo.py` — 完整闭环 demo，输出 `assets/demo/vla_demo.mp4`
- `record.py` — 跑闭环并记录轨迹 `assets/demo/traj.npz`
- `replay.py` — 重放轨迹 → 2K 视频 `assets/demo/demo_vla.mp4`（不加载 VLM）
- `run.sh` — 跑 vla_demo，黑帧自动重试（Windows 专用）
- `yolo/` — YOLO 数据集采集 / 标注 / 训练

## 前置

- VLM 首次运行自动下载（4.2GB），需先 `huggingface-cli login`
- YOLO 权重 `assets/models/yolo/best.pt` 由 `git lfs pull` 提供

## 运行闭环

```bash
bash vla/run.sh        # 黑帧自动重试
# 或
python vla/vla_demo.py
```

## YOLO 训练链

```bash
python vla/yolo/collect.py     # 仿真采集障碍帧
python vla/yolo/annotate.py    # 自动标注 → YOLO 格式
python vla/yolo/train.py       # 微调 YOLOv8n
```
