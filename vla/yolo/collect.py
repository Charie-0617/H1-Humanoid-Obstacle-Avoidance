"""让 H1 走绕障流程，从第一人称相机采帧做 YOLO 训练数据（含负样本）。

不加载 VLM，用仿真检测触发转弯。
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

sys.argv.append("--enable_cameras")
app_launcher = AppLauncher(
    headless=True,
    enable_cameras=True,
    device="cuda:0",
    kit_args="--/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false "
             "--/rtx/resourcemanager/textureMipCountBudget=256",
)
app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

ROOT = Path(__file__).resolve().parents[2]  # 仓库根目录

OBSTACLE_POS = (5.0, 0.0, 0.6)
POLICY_PATH = str(ROOT / "assets" / "models" / "best_model.pt")
RAW_DIR = str(ROOT / "assets" / "dataset" / "raw")
N_STEPS = 700
DECISION_EVERY = 75
SAVE_EVERY = 3


def quat_to_yaw(q):
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def heading_error(cur_yaw):
    """相对目标航向（+X）的偏差，归一化到 [-pi, pi]。"""
    err = -cur_yaw
    return (err + np.pi) % (2.0 * np.pi) - np.pi


def main():
    print("[collect] 加载策略...", flush=True)
    from rsl_rl.modules.actor_critic import ActorCritic
    from tensordict import TensorDict

    ckpt = torch.load(POLICY_PATH, map_location="cuda:0", weights_only=False)
    obs_td = TensorDict({"policy": torch.zeros(1, 69), "critic": torch.zeros(1, 69)}, batch_size=[1])
    policy = ActorCritic(
        obs=obs_td, obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=19, actor_hidden_dims=[128, 128, 128], critic_hidden_dims=[128, 128, 128],
        activation="elu", init_noise_std=1.0,
    )
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()
    policy.to("cuda:0")

    env_cfg = H1FlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.render = sim_utils.RenderCfg(rendering_mode="balanced", antialiasing_mode="FXAA")
    env_cfg.scene.sky_light.spawn.intensity = 2500.0
    env_cfg.scene.sun = AssetBaseCfg(
        prim_path="/World/Sun", spawn=sim_utils.DistantLightCfg(intensity=800.0, color=(1.0, 0.93, 0.82)),
    )
    # 禁用 reset 位姿随机化，初始朝向固定朝 +X（否则每轮起点不同，采集画面混乱）
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.scene.obstacle = AssetBaseCfg(
        prim_path="/World/Obstacle",
        spawn=sim_utils.CylinderCfg(
            radius=0.3, height=1.2,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.35, 0.05)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=OBSTACLE_POS),
    )
    env_cfg.scene.cam_vlm = CameraCfg(
        prim_path="/World/CameraVlm", update_period=0.0, height=240, width=320, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)),
    )
    env = ManagerBasedRLEnv(env_cfg)
    env = RslRlVecEnvWrapper(env)

    artic = env.unwrapped.scene.articulations["robot"]
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    cam_vlm = env.unwrapped.scene["cam_vlm"]

    import carb
    carb.settings.get_settings().set("/rtx/post/tonemap/filmIso", 80.0)
    carb.settings.get_settings().set("/rtx/post/tonemap/whitepoint", 2500.0)

    os.makedirs(RAW_DIR, exist_ok=True)

    # 固定初始位姿
    env.reset()
    root_state = artic.data.default_root_state.clone()
    root_state[0, :3] = torch.tensor([0.0, 0.0, 1.05], device="cuda:0")
    root_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda:0")
    artic.write_root_pose_to_sim(root_state[:, :7])
    env.unwrapped.sim.step()
    env.unwrapped.scene.update(env.unwrapped.sim.get_physics_dt())

    # 预热渲染（固定位姿，看障碍中心 1.2m）
    cam_vlm.set_world_poses_from_view(
        torch.tensor([[0.0, 0.0, 1.7]], dtype=torch.float32, device="cuda:0"),
        torch.tensor([[4.0, 0.0, 1.2]], dtype=torch.float32, device="cuda:0"),
    )
    for _ in range(40):
        env.step(torch.zeros(1, 19, device="cuda:0"))
    sim = env.unwrapped.sim
    scene = env.unwrapped.scene
    for _ in range(10):
        sim.render()
        scene.update(sim.get_physics_dt())

    obs_pos = np.array(OBSTACLE_POS)
    mode = "idle"          # idle直行 / turning绕障 / recover回正
    cmd = [0.6, 0.0, 0.0]  # 速度命令 (vx, vy, yaw)
    saved = 0
    print("[collect] 开始采集...", flush=True)

    obs_td = env.get_observations()
    with torch.no_grad():
        for step in range(N_STEPS):
            rpos = artic.data.root_pos_w[0].cpu().numpy()
            cur_yaw = quat_to_yaw(artic.data.root_quat_w[0].cpu().numpy())
            fwd = np.array([np.cos(cur_yaw), np.sin(cur_yaw)])

            # VLM 相机：第一人称前视，眼睛在躯干上方 0.65m（约 1.7m）
            eye_vlm = (float(rpos[0]), float(rpos[1]), float(rpos[2] + 0.65))
            tgt_vlm = (float(rpos[0] + fwd[0] * 5.0), float(rpos[1] + fwd[1] * 5.0), float(rpos[2] + 0.15))
            cam_vlm.set_world_poses_from_view(
                torch.tensor([eye_vlm], dtype=torch.float32, device="cuda:0"),
                torch.tensor([tgt_vlm], dtype=torch.float32, device="cuda:0"),
            )

            # 决策：仿真检测触发转弯（采集阶段不加载 VLM，画面多样性即目标）
            dist = np.hypot(obs_pos[0] - rpos[0], obs_pos[1] - rpos[1])
            if step % DECISION_EVERY == 0:
                if dist < 3.5 and mode == "idle":  # 距障碍 <3.5m 开始绕
                    mode = "turning"
                    cmd = [0.4, 0.0, 0.5]  # 左转绕障
                elif mode == "turning" and dist < 3.5:
                    pass  # 障碍仍在近处，继续绕
                elif mode == "turning":
                    mode = "recover"  # 障碍已在身后，回正
                if mode == "recover":
                    err = heading_error(cur_yaw)  # 相对 +X 航向偏差，转回直行
                    if abs(err) < 0.09:
                        cmd = [0.6, 0.0, 0.0]
                    else:
                        cmd = [0.6, 0.0, float(np.clip(err, -0.4, 0.4))]
                cmd_term.command[:, :3] = torch.tensor([cmd], device="cuda:0")

            obs_td = env.get_observations()
            actions = policy.act_inference(obs_td)
            obs_td, reward, terminated, _ = env.step(actions)

            # 存 cam_vlm 帧（先显式渲染确保相机出图，黑帧跳过）
            if step % SAVE_EVERY == 0:
                sim.render()
                scene.update(sim.get_physics_dt())
                img = cam_vlm.data.output["rgb"][0, ..., :3].cpu().numpy()
                if img.max() <= 1.0:
                    img = img * 255
                if img.mean() > 5.0:
                    from PIL import Image
                    Image.fromarray(img.astype(np.uint8)).save(os.path.join(RAW_DIR, f"frame_{step:05d}.png"))
                    saved += 1
                else:
                    print(f"[collect] step={step} 黑帧跳过", flush=True)

    print(f"[collect] 完成，存 {saved} 帧 → {RAW_DIR}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
