"""和 vla_demo 一样跑闭环，但不存帧，把轨迹记下来给 replay.py 重放出片。

输出：assets/demo/traj.npz
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# import 顺序固定：torch → VLM → AppLauncher → isaaclab
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

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

ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录

# VLM 模型 ID：首次运行自动从 HuggingFace 下载（门控模型需先 huggingface-cli login）
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
# 路径配置（相对仓库根目录）
POLICY_PATH = str(ROOT / "assets" / "models" / "best_model.pt")
YOLO_WEIGHTS = str(ROOT / "assets" / "models" / "yolo" / "best.pt")
OBSTACLE_POS = (5.0, 0.0, 0.6)
TARGET_HEADING = np.array([1.0, 0.0])
LANGUAGE_CMD = "绕过前方的障碍物，继续前进"
OUT_NPZ = str(ROOT / "assets" / "demo" / "traj.npz")
DECISION_EVERY = 75
N_STEPS = 850
CAM_VLM_H, CAM_VLM_W = 240, 320

TURN_CMD = {"左转": (0.4, -0.5), "右转": (0.4, 0.5)}  # 实证：yaw 正=右转，yaw 负=左转
STRAIGHT_CMD = (0.6, 0.0)
YAW_MAX = 0.6  # 回正/归位最大角速度（更快归位）
YAW_DEADZONE = 0.09

PROMPT_TURN = (
    "你是人形机器人 H1 的导航决策器。\n"
    "传感器检测结果：正前方中央有障碍物挡住去路。\n"
    "你的任务：{instruction}\n"
    "请选择绕开方向，从以下选一个：左转 / 右转。只输出一个动作词。"
)


def detect_obstacle_yolo(yolo_model, img_vlm):
    from PIL import Image as PILImage
    r = yolo_model(PILImage.fromarray(img_vlm.astype(np.uint8)), conf=0.25, verbose=False)[0]
    if len(r.boxes) == 0:
        return "没有检测到明显障碍物"
    boxes = r.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    x1, y1, x2, y2 = boxes[areas.argmax()]
    W = img_vlm.shape[1]
    H = img_vlm.shape[0]
    cx = (x1 + x2) / 2.0
    area_ratio = ((x2 - x1) * (y2 - y1)) / (W * H)
    if abs(cx - W / 2.0) < W * 0.3:
        if area_ratio > 0.05:
            return "障碍物在正前方中央"
        return "障碍物在正前方远处"
    elif cx < W / 2.0:
        return "障碍物在画面左侧"
    else:
        return "障碍物在画面右侧"


def quat_to_yaw(q):
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def heading_error(cur_yaw, target=TARGET_HEADING):
    tgt_yaw = np.arctan2(target[1], target[0])
    err = tgt_yaw - cur_yaw
    return (err + np.pi) % (2.0 * np.pi) - np.pi


def recover_cmd(rpos, cur_yaw, obs_x):
    """回到原路线：过障前只回正朝向保持偏移，过障后横向归位到 y=0 直线。"""
    if rpos[0] < obs_x - 0.5:
        # 尚未绕过障碍：只回正朝向（保持横向偏移，避免归位过早撞障）
        err = heading_error(cur_yaw)
        yaw = float(np.clip(err, -YAW_MAX, YAW_MAX))
        done = abs(err) < YAW_DEADZONE
        return [STRAIGHT_CMD[0], 0.0, yaw], done
    else:
        # 已绕过障碍最宽处：朝原路线上前方 2m 的点走（修正横向偏移，回 y=0 直线）
        des_yaw = np.arctan2(0.0 - rpos[1], 2.0)
        err = (des_yaw - cur_yaw + np.pi) % (2.0 * np.pi) - np.pi
        yaw = float(np.clip(err, -YAW_MAX, YAW_MAX))
        done = abs(err) < YAW_DEADZONE and abs(rpos[1]) < 0.1
        return [STRAIGHT_CMD[0], 0.0, yaw], done


def vlm_decision(model, processor, pil_img):
    prompt = PROMPT_TURN.format(instruction=LANGUAGE_CMD)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_img], return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    result = processor.decode(out[0], skip_special_tokens=True)
    action = result[result.rfind("assistant") + 9:].strip().replace("\n", "").replace("。", "")
    for key in TURN_CMD:
        if key in action:
            return key, action
    return "左转", action


def main():
    print("[record] 加载策略...", flush=True)
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

    print("[record] 加载 YOLO...", flush=True)
    from ultralytics import YOLO
    yolo_model = YOLO(YOLO_WEIGHTS)

    env_cfg = H1FlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.render = sim_utils.RenderCfg(rendering_mode="balanced", antialiasing_mode="FXAA")
    env_cfg.scene.sky_light.spawn.intensity = 2500.0
    env_cfg.scene.sun = AssetBaseCfg(
        prim_path="/World/Sun", spawn=sim_utils.DistantLightCfg(intensity=800.0, color=(1.0, 0.93, 0.82)),
    )
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.scene.obstacle = AssetBaseCfg(
        prim_path="/World/Obstacle",
        spawn=sim_utils.CylinderCfg(
            radius=0.3, height=1.2,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.35, 0.05)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=OBSTACLE_POS),
    )
    env_cfg.scene.cam_rec = CameraCfg(
        prim_path="/World/Camera", update_period=0.0, height=720, width=1280, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)),
    )
    env_cfg.scene.cam_vlm = CameraCfg(
        prim_path="/World/CameraVlm", update_period=0.0, height=CAM_VLM_H, width=CAM_VLM_W, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)),
    )
    env = ManagerBasedRLEnv(env_cfg)
    env = RslRlVecEnvWrapper(env)

    artic = env.unwrapped.scene.articulations["robot"]
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    cam_rec = env.unwrapped.scene["cam_rec"]
    cam_vlm = env.unwrapped.scene["cam_vlm"]

    import carb
    carb.settings.get_settings().set("/rtx/post/tonemap/filmIso", 80.0)
    carb.settings.get_settings().set("/rtx/post/tonemap/whitepoint", 2500.0)

    # 固定初始位姿
    env.reset()
    root_state = artic.data.default_root_state.clone()
    root_state[0, :3] = torch.tensor([0.0, 0.0, 1.05], device="cuda:0")
    root_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda:0")
    artic.write_root_pose_to_sim(root_state[:, :7])
    env.unwrapped.sim.step()
    env.unwrapped.scene.update(env.unwrapped.sim.get_physics_dt())

    # 预热
    cam_rec.set_world_poses_from_view(
        torch.tensor([[0.0, -3.5, 2.05]], dtype=torch.float32, device="cuda:0"),
        torch.tensor([[4.0, 0.0, 1.25]], dtype=torch.float32, device="cuda:0"),
    )
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

    print("[record] 加载 VLM (4bit)...", flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, load_in_4bit=True, device_map="cuda:0"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    print(f"[record] VLM 加载完成 | 总显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)

    obs_pos = np.array(OBSTACLE_POS)
    mode = "idle"
    turn_steps = 0
    cmd = [0.6, 0.0, 0.0]

    # 记录缓冲
    traj = {"root_pos": [], "root_quat": [], "joint_pos": [],
            "cam_rec_eye": [], "cam_rec_tgt": [], "cam_vlm_eye": [], "cam_vlm_tgt": []}

    print("[record] 开始闭环（记录轨迹）...", flush=True)
    start_time = time.time()
    obs_td = env.get_observations()
    with torch.no_grad():
        for step in range(N_STEPS):
            rpos = artic.data.root_pos_w[0].cpu().numpy()
            cur_yaw = quat_to_yaw(artic.data.root_quat_w[0].cpu().numpy())
            fwd = np.array([np.cos(cur_yaw), np.sin(cur_yaw)])

            eye = (float(rpos[0] - fwd[0] * 3.5), float(rpos[1] - fwd[1] * 3.5), float(rpos[2] + 1.0))
            tgt = (float(rpos[0] + fwd[0] * 4.0), float(rpos[1] + fwd[1] * 4.0), float(rpos[2] + 0.2))
            cam_rec.set_world_poses_from_view(
                torch.tensor([eye], dtype=torch.float32, device="cuda:0"),
                torch.tensor([tgt], dtype=torch.float32, device="cuda:0"),
            )
            eye_vlm = (float(rpos[0]), float(rpos[1]), float(rpos[2] + 0.65))
            tgt_vlm = (float(rpos[0] + fwd[0] * 5.0), float(rpos[1] + fwd[1] * 5.0), float(rpos[2] + 0.15))
            cam_vlm.set_world_poses_from_view(
                torch.tensor([eye_vlm], dtype=torch.float32, device="cuda:0"),
                torch.tensor([tgt_vlm], dtype=torch.float32, device="cuda:0"),
            )

            img_vlm = cam_vlm.data.output["rgb"][0, ..., :3].cpu().numpy()
            if img_vlm.max() <= 1.0:
                img_vlm = img_vlm * 255

            # 决策（YOLO 视觉检测）
            if step % DECISION_EVERY == 0:
                hint = detect_obstacle_yolo(yolo_model, img_vlm)
                if hint == "障碍物在正前方中央":
                    mode = "turning"
                    turn_steps = 0
                    pil_img = Image.fromarray(img_vlm.astype(np.uint8)).convert("RGB")
                    action, raw = vlm_decision(model, processor, pil_img)
                    cmd = [TURN_CMD[action][0], 0.0, TURN_CMD[action][1]]
                    print(f"[record] step={step} YOLO: {hint} | VLM选: {action}", flush=True)
                else:
                    if mode == "turning":
                        turn_steps += 1
                        if turn_steps < 2:
                            # 障碍已不在中央，但保持转弯 1 个决策周期确保偏移足够再回正
                            pass  # cmd 保持转弯值
                        else:
                            mode = "recover"
                    if mode == "recover":
                        cmd, done = recover_cmd(rpos, cur_yaw, OBSTACLE_POS[0])
                        if done:
                            mode = "idle"
                    print(f"[record] step={step} YOLO: {hint} | 模式={mode}", flush=True)

            elif mode == "recover":
                # 非决策步：每步朝目标刷新（平滑归位到原路线）
                cmd, _ = recover_cmd(rpos, cur_yaw, OBSTACLE_POS[0])

            # 关键：每步都写命令（决策步更新 cmd，非决策步保持并持续生效，否则策略收到 yaw=0 不转）
            cmd_term.command[:, :3] = torch.tensor([cmd], device="cuda:0")

            obs_td = env.get_observations()
            actions = policy.act_inference(obs_td)
            obs_td, reward, terminated, _ = env.step(actions)

            # 每 2 步记录（与重放帧率匹配）
            if step % 2 == 0:
                traj["root_pos"].append(artic.data.root_pos_w[0].cpu().numpy())
                traj["root_quat"].append(artic.data.root_quat_w[0].cpu().numpy())
                traj["joint_pos"].append(artic.data.joint_pos[0].cpu().numpy())
                traj["cam_rec_eye"].append(np.array(eye, dtype=np.float32))
                traj["cam_rec_tgt"].append(np.array(tgt, dtype=np.float32))
                traj["cam_vlm_eye"].append(np.array(eye_vlm, dtype=np.float32))
                traj["cam_vlm_tgt"].append(np.array(tgt_vlm, dtype=np.float32))
            if terminated:
                print(f"[record] step={step} 摔倒/终止", flush=True)
                break

    elapsed = time.time() - start_time
    n = len(traj["root_pos"])
    print(f"[record] 闭环完成: {step+1} 步, 记录 {n} 帧, 实耗 {elapsed:.0f}s", flush=True)

    out = {k: np.array(v, dtype=np.float32) for k, v in traj.items()}
    np.savez(OUT_NPZ, **out)
    print(f"[record] 轨迹已存: {OUT_NPZ}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
