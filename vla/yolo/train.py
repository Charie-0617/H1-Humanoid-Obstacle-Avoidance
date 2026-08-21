"""从 yolov8n.pt 迁移学习，微调识别橙色障碍。
"""

import os
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]  # 仓库根目录
DATA = str(ROOT / "assets" / "dataset" / "data.yaml")


def main():
    # data.yaml 用相对路径 path: .，需在数据集目录下运行
    os.chdir(os.path.dirname(DATA))
    model = YOLO("yolov8n.pt")  # 自动下载 COCO 预训练权重
    print("[train] 开始微调...", flush=True)
    results = model.train(
        data=DATA,
        epochs=100,
        imgsz=320,
        freeze=10,
        patience=20,
        batch=8,
        workers=0,  # Windows 避免 spawn 多进程问题
        optimizer="AdamW",
        lr0=0.001,
        mosaic=0.5,
        device=0,
        project=str(ROOT / "runs"),
        name="obstacle",
        verbose=True,
    )
    # 产物在 runs/obstacle/weights/best.pt，需拷贝到 assets/models/yolo/best.pt 供闭环使用
    print("[train] 训练完成 → runs/obstacle/weights/best.pt", flush=True)


if __name__ == "__main__":
    main()
