"""障碍颜色已知，用 HSV 阈值自动标框 → YOLO 格式，划分 train/val，写 data.yaml。
"""

import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]  # 仓库根目录
BASE = str(ROOT / "assets" / "dataset")
RAW = os.path.join(BASE, "raw")
IMG_TRAIN = os.path.join(BASE, "images", "train")
IMG_VAL = os.path.join(BASE, "images", "val")
LBL_TRAIN = os.path.join(BASE, "labels", "train")
LBL_VAL = os.path.join(BASE, "labels", "val")
VIS_DIR = os.path.join(BASE, "vis")


def find_orange_bbox(img_bgr):
    """返回橙色障碍 bbox (x1,y1,x2,y2) 或 None。"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # 橙色阈值（障碍色 RGB(0.9,0.35,0.05) → HSV H≈13°）
    lower = np.array([6, 80, 60])
    upper = np.array([28, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    # 形态学清理
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50:  # 过滤噪声
            continue
        if area > best_area:
            best_area = area
            best = cv2.boundingRect(c)  # x, y, w, h
    if best is None:
        return None
    x, y, w, h = best
    return (x, y, x + w, y + h)


def main():
    os.makedirs(IMG_TRAIN, exist_ok=True)
    os.makedirs(IMG_VAL, exist_ok=True)
    os.makedirs(LBL_TRAIN, exist_ok=True)
    os.makedirs(LBL_VAL, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    frames = sorted(f for f in os.listdir(RAW) if f.endswith(".png"))
    print(f"[annotate] 帧数: {len(frames)}", flush=True)

    random.seed(42)  # 固定种子，train/val 划分可复现
    random.shuffle(frames)
    n_val = max(1, int(len(frames) * 0.2))  # 20% 作验证集
    val_frames = set(frames[:n_val])
    train_frames = [f for f in frames if f not in val_frames]

    stats = {"train": 0, "val": 0, "pos": 0, "neg": 0}
    for f in frames:
        split = "val" if f in val_frames else "train"
        src = os.path.join(RAW, f)
        img = cv2.imread(src)  # BGR
        if img is None:
            continue
        h, w = img.shape[:2]
        bbox = find_orange_bbox(img)
        # 复制图像
        dst_img = os.path.join(IMG_TRAIN if split == "train" else IMG_VAL, f)
        shutil.copy(src, dst_img)
        # 写 label
        lbl_path = os.path.join(LBL_TRAIN if split == "train" else LBL_VAL, f.replace(".png", ".txt"))
        stats[split] += 1
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            # 裁剪到 [0,1] 防越界
            cx, cy, bw, bh = [max(0.0, min(1.0, v)) for v in (cx, cy, bw, bh)]
            with open(lbl_path, "w") as fp:
                fp.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            stats["pos"] += 1
        else:
            open(lbl_path, "w").close()  # 空 label = 负样本
            stats["neg"] += 1

    print(f"[annotate] train={stats['train']} val={stats['val']} | 含障碍={stats['pos']} 背景={stats['neg']}", flush=True)

    # data.yaml（相对路径，训练时在数据集目录下运行即可）
    with open(os.path.join(BASE, "data.yaml"), "w") as fp:
        fp.write("path: .\n")
        fp.write("train: images/train\n")
        fp.write("val: images/val\n")
        fp.write("names:\n  0: obstacle\n")

    # 可视化验证（抽 8 张画框，含正负样本）
    vis_files = [f for f in frames if f not in val_frames][:8]
    for f in vis_files:
        img = cv2.imread(os.path.join(RAW, f))
        h, w = img.shape[:2]
        bbox = find_orange_bbox(img)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(VIS_DIR, f.replace(".png", "_vis.png")), img)
    print(f"[annotate] 可视化已存 {VIS_DIR}/", flush=True)
    print(f"[annotate] data.yaml: {os.path.join(BASE, 'data.yaml')}", flush=True)


if __name__ == "__main__":
    main()
