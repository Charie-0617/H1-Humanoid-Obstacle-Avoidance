"""选最优 checkpoint 导出部署产物（policy.onnx / policy.pt / samples.json）。

用法：ISAACLAB_ROOT=<IsaacLab 源码> python export.py --log <训练日志> --run_dir <日志目录>
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录
EXPORTED = str(ROOT / "assets" / "models")  # 导出产物目录
ISAACLAB = os.environ.get("ISAACLAB_ROOT", "")  # IsaacLab 源码目录，由环境变量指定
if not ISAACLAB:
    raise RuntimeError("请设置 ISAACLAB_ROOT 指向 IsaacLab 源码目录")
PY = sys.executable  # 复用当前 Python 解释器

import checkpoint as ckpt  # 同目录的 checkpoint 解析工具


def find_best_ckpt(log_path, run_dir):
    pairs = ckpt.parse_rewards(log_path)
    ckpt_iters = set()
    for ck in glob.glob(os.path.join(run_dir, "model_*.pt")):
        m = re.search(r"model_(\d+)\.pt", os.path.basename(ck))
        if m:
            ckpt_iters.add(int(m.group(1)))
    candidates = [p for p in pairs if p[0] in ckpt_iters]
    if not candidates:
        raise RuntimeError("日志里的奖励与已保存的 checkpoint 无法匹配")
    best_iter, best_reward = max(candidates, key=lambda x: x[1])
    return os.path.join(run_dir, f"model_{best_iter}.pt"), best_iter, best_reward


def run(cmd, desc):
    print(f"[export] {desc}")
    print(f"  > {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=False)
    if r.returncode != 0:
        raise RuntimeError(f"Step failed ({desc}), returncode={r.returncode}")
    print("  OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="训练 stdout 日志")
    parser.add_argument("--run_dir", required=True, help="含 checkpoint 的训练日志目录")
    args = parser.parse_args()

    os.makedirs(EXPORTED, exist_ok=True)

    # 1. 最优 checkpoint
    best_ckpt, best_iter, best_reward = find_best_ckpt(args.log, args.run_dir)
    print(f"\n=== Best checkpoint: iter {best_iter}, reward {best_reward:.3f} ===")
    print(f"  {best_ckpt}")
    shutil.copy2(best_ckpt, os.path.join(EXPORTED, "best_model.pt"))
    print("  copied -> assets/models/best_model.pt")

    # 2. 用 play.py 导出 ONNX + JIT（产物落在 run_dir/exported/ 下）
    export_dir = os.path.join(args.run_dir, "exported")
    os.chdir(ISAACLAB)
    run(
        f'"{PY}" scripts/reinforcement_learning/rsl_rl/play.py '
        f"--task=Isaac-Velocity-Flat-H1-v0 --num_envs=4 --headless "
        f'--checkpoint="{best_ckpt}"',
        "exporting ONNX + JIT via play.py",
    )
    for name in ("policy.onnx", "policy.pt"):
        src = os.path.join(export_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(EXPORTED, name))
            print(f"  copied -> assets/models/{name}")
        else:
            print(f"  WARN: {src} not found")

    # 3. 用 samples.py 录制样本。临时把它的 CHECKPOINT/OUT_JSON 常量
    #    替换为本次运行的目标，跑完还原。
    rs = str(ROOT / "training" / "samples.py")
    with open(rs, encoding="utf-8") as f:
        src_text = f.read()
    patched = re.sub(r'^CHECKPOINT = r".*"$', f'CHECKPOINT = r"{best_ckpt}"', src_text, flags=re.M)
    patched = re.sub(
        r'^OUT_JSON = r".*"$',
        f'OUT_JSON = r"{os.path.join(EXPORTED, "samples.json")}"',
        patched,
        flags=re.M,
    )
    with open(rs, "w", encoding="utf-8") as f:
        f.write(patched)
    try:
        run(f'"{PY}" "{rs}"', "recording 20-step samples.json")
    finally:
        with open(rs, "w", encoding="utf-8") as f:
            f.write(src_text)

    print("\n=== EXPORT COMPLETE ===")
    for name in ("policy.onnx", "policy.pt", "samples.json", "best_model.pt"):
        p = os.path.join(EXPORTED, name)
        print(f"  {name}: {os.path.getsize(p)} bytes" if os.path.exists(p) else f"  {name}: MISSING")


if __name__ == "__main__":
    main()
