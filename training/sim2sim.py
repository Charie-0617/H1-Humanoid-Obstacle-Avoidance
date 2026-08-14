"""验证 ONNX 和 PyTorch 策略在相同观测下输出一致（最大误差 < 1e-4）。

用法：python sim2sim.py [--export_dir assets/models]
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export_dir",
        default=str(ROOT / "assets" / "models"),
        help="含 policy.onnx 与 policy.pt 的目录",
    )
    args = parser.parse_args()
    export_dir = args.export_dir

    ort_sess = ort.InferenceSession(
        f"{export_dir}/policy.onnx", providers=["CPUExecutionProvider"]
    )
    jit_policy = torch.jit.load(f"{export_dir}/policy.pt")
    jit_policy.eval()

    obs_dim, act_dim, n_samples = 69, 19, 50
    max_err = 0.0
    mean_err = 0.0
    for _ in range(n_samples):
        # 合理范围内的随机观测
        obs = np.random.uniform(-5.0, 5.0, size=(1, obs_dim)).astype(np.float32)
        onnx_actions = ort_sess.run(None, {"obs": obs})[0]  # [1, 19]
        with torch.no_grad():
            jit_actions = jit_policy(torch.from_numpy(obs)).numpy()

        err = np.abs(onnx_actions - jit_actions)
        max_err = max(max_err, float(err.max()))
        mean_err = max(mean_err, float(err.mean()))

    print(f"ONNX 输入 obs[{obs_dim}] -> 输出 actions[{act_dim}]")
    print(f"样本数: {n_samples}")
    print(f"最大绝对误差: {max_err:.6e}")
    print(f"最大平均误差: {mean_err:.6e}")
    print("RESULT: PASS (误差 < 1e-4)" if max_err < 1e-4 else "RESULT: FAIL (误差 >= 1e-4)")


if __name__ == "__main__":
    main()
