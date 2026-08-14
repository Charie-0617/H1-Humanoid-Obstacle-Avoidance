"""从训练日志里挑奖励最高的 checkpoint，--copy 拷到 assets/models/best_model.pt。

用法：python checkpoint.py --log <训练日志> --run_dir <日志目录> [--copy]
"""

import argparse
import glob
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录
DEFAULT_OUT = str(ROOT / "assets" / "models" / "best_model.pt")


def parse_rewards(log_path):
    """从 rsl_rl 的 stdout 日志解析 iter -> mean_reward。"""
    iter_re = re.compile(r"Learning iteration (\d+)/\d+")
    rew_re = re.compile(r"Mean reward: ([\d.\-]+)")
    iters, rewards = [], []
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = iter_re.search(line)
            if m:
                iters.append(int(m.group(1)))
                continue
            m = rew_re.search(line)
            if m:
                rewards.append(float(m.group(1)))
    # reward 紧跟 iteration 行，zip 配对即可
    pairs = list(zip(iters, rewards))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="训练 stdout 日志")
    parser.add_argument("--run_dir", required=True, help="含 checkpoint 的训练日志目录")
    parser.add_argument("--out", default=DEFAULT_OUT, help="最优 checkpoint 输出路径")
    parser.add_argument("--copy", action="store_true", help="把最优 checkpoint 拷贝到 out")
    args = parser.parse_args()

    pairs = parse_rewards(args.log)
    if not pairs:
        print("ERROR: no iter/reward pairs parsed from log")
        return

    # 只保留有 checkpoint 保存的迭代
    ckpt_iters = set()
    for ck in glob.glob(os.path.join(args.run_dir, "model_*.pt")):
        m = re.search(r"model_(\d+)\.pt", os.path.basename(ck))
        if m:
            ckpt_iters.add(int(m.group(1)))

    candidates = [p for p in pairs if p[0] in ckpt_iters]
    if not candidates:
        print("ERROR: no parsed rewards match saved checkpoints")
        print("  parsed iters: ", min(p[0] for p in pairs), "-", max(p[0] for p in pairs))
        print("  ckpt iters:   ", sorted(ckpt_iters))
        return

    best_iter, best_reward = max(candidates, key=lambda x: x[1])
    best_ckpt = os.path.join(args.run_dir, f"model_{best_iter}.pt")
    print("=== Best checkpoint ===")
    print(f"  iteration : {best_iter}")
    print(f"  mean reward: {best_reward:.3f}")
    print(f"  checkpoint: {best_ckpt}")

    if args.copy:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        shutil.copy2(best_ckpt, args.out)
        print(f"  copied to : {args.out}")


if __name__ == "__main__":
    main()
