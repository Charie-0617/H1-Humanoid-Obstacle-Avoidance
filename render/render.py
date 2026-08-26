#!/usr/bin/env python
"""回放 walk_log.npz，逐帧摆位 H1 离屏渲染成 mp4。纯回放，不跑物理。

用法：python render/render.py --log assets/demo/walk_log.npz --out out.mp4 --fps 30
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

# 用 AppLauncher 启动（SimulationApp 直启无法启用离屏相机渲染管线）
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录

parser = argparse.ArgumentParser(description="Replay H1 walk log to video")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--log", type=str, default=str(ROOT / "assets" / "demo" / "walk_log.npz"))
parser.add_argument("--out", type=str, default=str(ROOT / "assets" / "demo" / "walk_vla.mp4"))
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--frames_dir", type=str, default="frames")
parser.add_argument("--max_frames", type=int, default=0, help="Render only first N frames (0=all)")
args = parser.parse_args()

app_launcher = AppLauncher(
    headless=True,
    enable_cameras=True,
    device="cuda:0",
    # Windows 双 GPU 需强制用 NVIDIA（否则可能选到 AMD 核显，几何渲染会坏）；
    # textureMipCountBudget 限制显存，避免 8G 卡编译 shader 时 OOM
    kit_args="--/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false "
             "--/rtx/resourcemanager/textureMipCountBudget=256",
)
app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab_assets import H1_CFG

# 场景：地面 + H1 + 跟踪相机
@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    num_envs: int = 1
    env_spacing: float = 4.0
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(color=(0.82, 0.80, 0.76)),  # 暖米色，避免机器人被蓝灰地面染色
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.88, 0.86, 0.82)),  # 暖白光环境，均匀照亮机器人
    )
    sun = AssetBaseCfg(
        prim_path="/World/Sun",
        spawn=sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 0.93, 0.82)),
    )
    # 参照标记柱：沿行走路径右侧等距排开，让画面有参照
    marker0 = AssetBaseCfg(
        prim_path="/World/Marker0",
        spawn=sim_utils.CuboidCfg(size=(0.15, 0.15, 1.2),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.05))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.375, 1.545, 0.6)),
    )
    marker1 = AssetBaseCfg(
        prim_path="/World/Marker1",
        spawn=sim_utils.CuboidCfg(size=(0.15, 0.15, 1.2),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 0.9))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.375, 2.205, 0.6)),
    )
    marker2 = AssetBaseCfg(
        prim_path="/World/Marker2",
        spawn=sim_utils.CuboidCfg(size=(0.15, 0.15, 1.2),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.05))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.125, 2.865, 0.6)),
    )
    marker3 = AssetBaseCfg(
        prim_path="/World/Marker3",
        spawn=sim_utils.CuboidCfg(size=(0.15, 0.15, 1.2),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 0.9))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.875, 3.525, 0.6)),
    )
    marker4 = AssetBaseCfg(
        prim_path="/World/Marker4",
        spawn=sim_utils.CuboidCfg(size=(0.15, 0.15, 1.2),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.05))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-2.625, 4.185, 0.6)),
    )
    robot: ArticulationCfg = H1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    camera = CameraCfg(
        prim_path="/World/Camera",
        update_period=0.0,
        height=1440,
        width=2560,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )

# 用 FXAA 而非 TAA：TAA 是时序抗锯齿，逐帧离线渲染会黑帧
render_cfg = sim_utils.RenderCfg(
    rendering_mode="balanced",
    antialiasing_mode="FXAA",
)
sim = SimulationContext(
    sim_utils.SimulationCfg(dt=0.005, device="cuda:0", render_interval=4, render=render_cfg)
)
scene = InteractiveScene(ReplaySceneCfg())
sim.reset()
scene.reset()

# 关掉默认视口，避免干扰离屏相机
import carb
carb.settings.get_settings().set("/app/window/hideUi", True)
carb.settings.get_settings().set("/app/viewport/enabled", False)
# headless 无默认曝光，手动给稳定值（filmIso 80，实测亮度稳定）
carb.settings.get_settings().set("/rtx/post/tonemap/filmIso", 80.0)
carb.settings.get_settings().set("/rtx/post/tonemap/whitepoint", 2500.0)
carb.settings.get_settings().set("/rtx/post/tonemap/enabled", True)
# 饱和度 +20%，让外壳颜色和地面层次清晰
carb.settings.get_settings().set("/rtx/post/colorgrad/enabled", True)
carb.settings.get_settings().set("/rtx/post/colorcorr/saturation", 1.2)


# 加载日志
data = np.load(args.log, allow_pickle=True)
t = data["t"]
joint_pos = data["joint_pos"]           # (N,19)
root_pos = data["root_pos"]             # (N,3)
root_quat_wxyz = data["root_quat_wxyz"] # (N,4)
robot = scene["robot"]
default_jp = robot.data.default_joint_pos[0].clone()
print(f"[render] 加载 {len(t)} 帧, 时长 {t[-1]:.1f}s", flush=True)

os.makedirs(args.frames_dir, exist_ok=True)

# 相机跟踪：eye 在机器人后方上方，看机器人
def look_at_quat(eye, target):
    """从 eye 看向 target 的四元数（wxyz），相机默认 -Z 前向。"""
    fwd = target - eye
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, fwd); right /= (np.linalg.norm(right) + 1e-9)
    up2 = np.cross(fwd, right)
    # 旋转矩阵 -> 四元数
    m = np.stack([right, up2, -fwd], axis=1)
    qw = np.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    qx = (m[2, 1] - m[1, 2]) / (4.0 * qw + 1e-9)
    qy = (m[0, 2] - m[2, 0]) / (4.0 * qw + 1e-9)
    qz = (m[1, 0] - m[0, 1]) / (4.0 * qw + 1e-9)
    return np.array([qw, qx, qy, qz])

# 预热前摆好第 0 帧姿态，让曝光在预热时就收敛，避免前几帧偏暗
root_state = robot.data.default_root_state.clone()
root_state[0, 0:3] = torch.tensor(root_pos[0], dtype=torch.float32)
root_state[0, 3:7] = torch.tensor(root_quat_wxyz[0], dtype=torch.float32)
robot.write_root_pose_to_sim(root_state[:, :7])
robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
robot.write_joint_state_to_sim(
    torch.tensor(joint_pos[0], dtype=torch.float32).unsqueeze(0), torch.zeros(1, robot.num_joints))

# 预热：编译 shader + 曝光收敛
print("[render] 预热渲染管线...", flush=True)
for _ in range(30):
    sim.render()
    scene.update(sim.get_physics_dt())
# 再渲一帧确认有内容
sim.render()
scene.update(sim.get_physics_dt())

from PIL import Image

def save_img(raw, path):
    """按 dtype 保存：uint8 直接存，[0,1] 乘 255，HDR 归一化。"""
    raw = np.asarray(raw).astype(np.float64)
    if raw.max() > 1.0 + 1e-6:  # 已经是 [0,255] 或 HDR
        if raw.max() > 255.0:  # HDR 线性归一化到可见范围
            raw = raw / (raw.max() + 1e-9) * 255.0
    else:  # [0,1]
        raw = raw * 255.0
    Image.fromarray(np.clip(raw, 0, 255).astype(np.uint8)).save(path)

n_frames = len(t) if args.max_frames <= 0 else min(args.max_frames, len(t))
for i in range(n_frames):
    # 摆位 H1
    root_state = robot.data.default_root_state.clone()
    root_state[0, 0:3] = torch.tensor(root_pos[i], dtype=torch.float32)
    root_state[0, 3:7] = torch.tensor(root_quat_wxyz[i], dtype=torch.float32)
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
    robot.write_joint_state_to_sim(
        torch.tensor(joint_pos[i], dtype=torch.float32).unsqueeze(0), torch.zeros(1, robot.num_joints))

    # 相机跟踪：官方 look-at API，高度降到 1.8m 让脚部入画
    eye = root_pos[i] + np.array([-3.0, -3.0, 1.8])   # 后方斜上方
    target = root_pos[i] + np.array([0.0, 0.0, 0.2])
    scene["camera"].set_world_poses_from_view(
        torch.tensor([eye], dtype=torch.float32, device="cuda:0"),
        torch.tensor([target], dtype=torch.float32, device="cuda:0"),
    )

    # 渲染（不步进物理）
    sim.render()
    scene.update(sim.get_physics_dt())

    img = scene["camera"].data.output["rgb"][0, ..., :3].cpu().numpy()
    save_img(img, os.path.join(args.frames_dir, f"frame_{i:05d}.png"))

    if i % 50 == 0:
        print(f"[render] 帧 {i}/{len(t)} (t={t[i]:.1f}s)", flush=True)

print("[render] 渲染完成，用 ffmpeg 合成 mp4...", flush=True)
os.system(
    f'ffmpeg -y -framerate {args.fps} -i "{os.path.join(args.frames_dir, "frame_%05d.png")}" '
    f'-c:v libx264 -pix_fmt yuv420p "{args.out}"'
)
print(f"[render] 完成: {args.out}", flush=True)
app.close()
