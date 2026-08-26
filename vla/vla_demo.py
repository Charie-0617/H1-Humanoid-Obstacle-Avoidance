"""H1 分层避障 Demo：YOLO 检测障碍 → Qwen2-VL 选绕向 → RL 策略绕行 → 回正直行，存帧合成视频。
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# import 顺序固定：torch → VLM → AppLauncher → isaaclab，
# torch/VLM 必须先于 Isaac Sim 导入，否则运行时报错
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

from isaaclab.app import AppLauncher

# Isaac Sim 5.0 headless 相机黑帧 bug (#3250) workaround：硬编码 + sys.argv
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
OBSTACLE_POS = (5.0, 0.0, 0.6)  # 固定：正前方 5m 橙色圆柱（放远，障碍在画面停留更久）
TARGET_HEADING = np.array([1.0, 0.0])  # 目标航向：+X
LANGUAGE_CMD = "绕过前方的障碍物，继续前进"
OUT_FRAMES_DIR = str(ROOT / "assets" / "demo" / "frames_vla")
OUT_VIDEO = str(ROOT / "assets" / "demo" / "vla_demo.mp4")
YOLO_WEIGHTS = str(ROOT / "assets" / "models" / "yolo" / "best.pt")
DECISION_EVERY = 75  # 每 75 策略步(≈1.5s)做一次完整决策
N_STEPS = 700  # 总步数(≈14s)
CAM_REC_H, CAM_REC_W = 1080, 1920  # 录制相机：1080p 第三人称（观众视角）
CAM_VLM_H, CAM_VLM_W = 240, 320    # VLM 感知相机：低清前视（机器人之眼）

# 转弯指令 (vx, yaw)。实测 yaw 正=右转，负=左转
TURN_CMD = {"左转": (0.4, -0.5), "右转": (0.4, 0.5)}
STRAIGHT_CMD = (0.6, 0.0)  # 直走
YAW_MAX = 0.4  # 回正最大角速度
YAW_DEADZONE = 0.09  # 偏差 <5° 视为已回正，直走

# 障碍在正前方中央时的约束 prompt：只给左/右选项，强制 VLM 选绕开方向
PROMPT_TURN = (
    "你是人形机器人 H1 的导航决策器。\n"
    "传感器检测结果：正前方中央有障碍物挡住去路。\n"
    "你的任务：{instruction}\n"
    "请选择绕开方向，从以下选一个：左转 / 右转。只输出一个动作词。"
)


# YOLO 真视觉检测：从相机画面判断障碍相对机器人的方向
def detect_obstacle_yolo(yolo_model, img_vlm):
    """YOLO 视觉检测障碍 → 判断位置（真视觉感知，替代仿真状态）。

    返回：
      - "障碍物在正前方中央"：障碍在画面中央且已接近(bbox>5%画面)，需转弯
      - "障碍物在正前方远处"：障碍在画面中央但还远，继续直走
      - "障碍物在画面左侧/右侧"：障碍偏离，回正/直走
      - "没有检测到明显障碍物"：未检出，直走
    """
    from PIL import Image as PILImage
    r = yolo_model(PILImage.fromarray(img_vlm.astype(np.uint8)), conf=0.25, verbose=False)[0]
    if len(r.boxes) == 0:
        return "没有检测到明显障碍物"
    boxes = r.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    x1, y1, x2, y2 = boxes[areas.argmax()]  # 取最大检测框
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
    """四元数 (w,x,y,z) → 偏航角 yaw（弧度）。"""
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def heading_error(cur_yaw, target=TARGET_HEADING):
    """当前 yaw 相对目标航向的偏差，归一化到 [-pi, pi]。正=需左转。"""
    tgt_yaw = np.arctan2(target[1], target[0])
    err = tgt_yaw - cur_yaw
    return (err + np.pi) % (2.0 * np.pi) - np.pi


# VLM 决策
def overlay_pip(frame, pip, det_box=None):
    """录制帧右上角叠加 VLM 第一人称画中画 + YOLO 检测框 + 白框 + 标签。"""
    from PIL import Image, ImageDraw, ImageFont
    fimg = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
    pimg = Image.fromarray(pip.astype(np.uint8)).convert("RGB")
    # 在画中画上画 YOLO 检测框（原始 320x240 坐标系，缩放前画）
    if det_box is not None:
        d0 = ImageDraw.Draw(pimg)
        x1, y1, x2, y2 = [float(v) for v in det_box]
        d0.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        try:
            font0 = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 14)
        except Exception:
            font0 = ImageFont.load_default()
        d0.text((x1 + 2, max(0, y1 - 16)), "YOLO", fill=(0, 255, 0),
                stroke_width=2, stroke_fill=(0, 0, 0), font=font0)
    # 画中画尺寸：宽 420，按比例缩放
    pw = 420
    ph = int(pw * pimg.height / pimg.width)
    pimg = pimg.resize((pw, ph), Image.BILINEAR)
    x, y = fimg.width - pw - 24, 24
    fimg.paste(pimg, (x, y))
    d = ImageDraw.Draw(fimg)
    d.rectangle([x - 4, y - 4, x + pw + 3, y + ph + 3], outline=(255, 255, 255), width=3)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
    except Exception:
        font = ImageFont.load_default()
    d.text((x, y + ph + 10), "VLM 视角（第一人称）", fill=(255, 255, 255),
           stroke_width=2, stroke_fill=(0, 0, 0), font=font)
    return np.array(fimg)


def vlm_decision(model, processor, pil_img):
    """VLM 决策：障碍在正前方中央时，从 左转/右转 里选绕开方向。"""
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
    return "左转", action  # 解析失败回退左转（障碍在正前方必须绕）


def main():
    # 加载策略
    print("[vla] 加载策略...", flush=True)
    from rsl_rl.modules.actor_critic import ActorCritic
    from tensordict import TensorDict

    ckpt = torch.load(POLICY_PATH, map_location="cuda:0", weights_only=False)
    obs_td = TensorDict({"policy": torch.zeros(1, 69), "critic": torch.zeros(1, 69)}, batch_size=[1])
    policy = ActorCritic(
        obs=obs_td,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=19,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
        init_noise_std=1.0,
    )
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()
    policy.to("cuda:0")
    print(f"[vla] 策略加载完成 | 显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)

    # 加载 YOLO（真视觉感知，替代仿真检测）
    print("[vla] 加载 YOLO...", flush=True)
    from ultralytics import YOLO
    yolo_model = YOLO(YOLO_WEIGHTS)
    print(f"[vla] YOLO 加载完成 | 显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)

    # 环境：H1 + 圆柱障碍 + 相机
    env_cfg = H1FlatEnvCfg()
    env_cfg.scene.num_envs = 1
    # 渲染配置：balanced + FXAA（实测成功出图，TAA 逐帧渲染会黑帧）
    env_cfg.sim.render = sim_utils.RenderCfg(rendering_mode="balanced", antialiasing_mode="FXAA")
    # 调亮场景：DomeLight 主光 + 弱太阳（避免白色大理石地面被太阳直射反光过曝）
    # 实测：750→亮度47；3000+sun2500→亮度110但地面高光带过曝(y≈345-453)。
    env_cfg.scene.sky_light.spawn.intensity = 2500.0
    env_cfg.scene.sun = AssetBaseCfg(
        prim_path="/World/Sun",
        spawn=sim_utils.DistantLightCfg(intensity=800.0, color=(1.0, 0.93, 0.82)),
    )
    # 关键修复：禁用 reset 的位姿随机化，初始朝向确定（朝 +X）
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.scene.obstacle = AssetBaseCfg(
        prim_path="/World/Obstacle",
        spawn=sim_utils.CylinderCfg(
            radius=0.3,
            height=1.2,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.35, 0.05)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=OBSTACLE_POS),
    )
    # 录制相机：第三人称，1080p（观众视角，用于视频）
    env_cfg.scene.cam_rec = CameraCfg(
        prim_path="/World/Camera",
        update_period=0.0,
        height=CAM_REC_H,
        width=CAM_REC_W,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    # VLM 感知相机：第一人称前视，低清（机器人之眼，VLM 输入）
    env_cfg.scene.cam_vlm = CameraCfg(
        prim_path="/World/CameraVlm",
        update_period=0.0,
        height=CAM_VLM_H,
        width=CAM_VLM_W,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    env = ManagerBasedRLEnv(env_cfg)
    env = RslRlVecEnvWrapper(env)

    artic = env.unwrapped.scene.articulations["robot"]
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    cam_rec = env.unwrapped.scene["cam_rec"]
    cam_vlm = env.unwrapped.scene["cam_vlm"]

    # 曝光设置（对齐渲染时的稳定配置）
    import carb
    carb.settings.get_settings().set("/rtx/post/tonemap/filmIso", 80.0)
    carb.settings.get_settings().set("/rtx/post/tonemap/whitepoint", 2500.0)

    os.makedirs(OUT_FRAMES_DIR, exist_ok=True)

    # 固定初始位姿（reset 随机化已禁用，再写一次兜底）
    env.reset()
    root_state = artic.data.default_root_state.clone()
    root_state[0, :3] = torch.tensor([0.0, 0.0, 1.05], device="cuda:0")
    root_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda:0")  # 朝向 +X
    artic.write_root_pose_to_sim(root_state[:, :7])
    env.unwrapped.sim.step()
    env.unwrapped.scene.update(env.unwrapped.sim.get_physics_dt())
    start_yaw = quat_to_yaw(artic.data.root_quat_w[0].cpu().numpy())
    print(f"[vla] 初始位姿: pos={artic.data.root_pos_w[0, :3].cpu().numpy()} yaw={np.degrees(start_yaw):.1f}°", flush=True)

    # 注意：此处不做相机探针——第一次渲染前相机数据必为 0，会误报黑帧。
    # 相机检查统一放到预热后（预热会让相机真正渲染出图）。

    # 加载 VLM（位姿固定后再加载，避免抢占 GPU 拖慢启动）
    print("[vla] 加载 VLM (4bit)...", flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, load_in_4bit=True, device_map="cuda:0"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    print(f"[vla] VLM 加载完成 | 总显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)

    # 预热：策略步进 + 显式渲染确保相机出图
    print("[vla] 预热渲染...", flush=True)
    # 先给两台相机初始位姿（避免默认无效位姿导致黑帧）
    cam_rec.set_world_poses_from_view(
        torch.tensor([[0.0, -2.5, 2.85]], dtype=torch.float32, device="cuda:0"),
        torch.tensor([[4.0, 0.0, 1.35]], dtype=torch.float32, device="cuda:0"),
    )
    cam_vlm.set_world_poses_from_view(
        torch.tensor([[0.0, 0.0, 1.7]], dtype=torch.float32, device="cuda:0"),
        torch.tensor([[4.0, 0.0, 1.2]], dtype=torch.float32, device="cuda:0"),
    )
    for _ in range(40):
        env.step(torch.zeros(1, 19, device="cuda:0"))
    # 显式渲染数帧，确保相机渲染完成
    sim = env.unwrapped.sim
    scene = env.unwrapped.scene
    for _ in range(10):
        sim.render()
        scene.update(sim.get_physics_dt())
    probe = cam_rec.data.output["rgb"][0, ..., :3].cpu().numpy()
    print(f"[vla] 预热后录制相机亮度: {probe.mean():.1f}", flush=True)
    if probe.mean() < 5.0:
        # 仍黑：可能首帧延迟，再渲染几次
        for _ in range(8):
            sim.render()
            scene.update(sim.get_physics_dt())
        probe = cam_rec.data.output["rgb"][0, ..., :3].cpu().numpy()
        print(f"[vla] 二次渲染后亮度: {probe.mean():.1f}", flush=True)
    if probe.mean() < 5.0:
        print("[vla] 相机黑帧，退出（外层将重试）", flush=True)
        os._exit(3)

    # 初始指令：直走
    cmd = [STRAIGHT_CMD[0], 0.0, STRAIGHT_CMD[1]]
    cmd_term.command[:, :3] = torch.tensor([cmd], device="cuda:0")

    # 主循环
    print(f"[vla] 开始 demo: 指令='{LANGUAGE_CMD}' | 障碍物在 {OBSTACLE_POS}", flush=True)
    start_time = time.time()
    recorded = []
    decisions = []
    mode = "idle"  # idle(直行) / turning(绕障) / recover(回正)
    turn_steps = 0
    turn_dir = "左转"
    obs_pos = np.array(OBSTACLE_POS)

    obs_td = env.get_observations()
    frame_idx = 0
    with torch.no_grad():
        for step in range(N_STEPS):
            # 当前朝向（固定初始朝向 +X 后，四元数 yaw 即实际朝向）
            rpos = artic.data.root_pos_w[0].cpu().numpy()
            cur_yaw = quat_to_yaw(artic.data.root_quat_w[0].cpu().numpy())
            fwd = np.array([np.cos(cur_yaw), np.sin(cur_yaw)])

            # 录制相机：第三人称正后方（观众视角，拉远到 3.5m 让机器人全身入画）
            eye = (float(rpos[0] - fwd[0] * 3.5), float(rpos[1] - fwd[1] * 3.5), float(rpos[2] + 1.0))
            tgt = (float(rpos[0] + fwd[0] * 4.0), float(rpos[1] + fwd[1] * 4.0), float(rpos[2] + 0.2))
            cam_rec.set_world_poses_from_view(
                torch.tensor([eye], dtype=torch.float32, device="cuda:0"),
                torch.tensor([tgt], dtype=torch.float32, device="cuda:0"),
            )
            # VLM 感知相机：第一人称前视（眼睛高度 rpos+0.65≈1.7m，看障碍中心 1.2m）
            eye_vlm = (float(rpos[0]), float(rpos[1]), float(rpos[2] + 0.65))
            tgt_vlm = (float(rpos[0] + fwd[0] * 5.0), float(rpos[1] + fwd[1] * 5.0), float(rpos[2] + 0.15))
            cam_vlm.set_world_poses_from_view(
                torch.tensor([eye_vlm], dtype=torch.float32, device="cuda:0"),
                torch.tensor([tgt_vlm], dtype=torch.float32, device="cuda:0"),
            )

            # 相机数据在上一次 env.step 后已更新，直接读
            img = cam_rec.data.output["rgb"][0, ..., :3].cpu().numpy()
            if img.max() <= 1.0:
                img = img * 255
            img_vlm = cam_vlm.data.output["rgb"][0, ..., :3].cpu().numpy()
            if img_vlm.max() <= 1.0:
                img_vlm = img_vlm * 255

            # 决策（YOLO 视觉检测）
            if step % DECISION_EVERY == 0:
                hint = detect_obstacle_yolo(yolo_model, img_vlm)
                if hint == "障碍物在正前方中央":
                    # 障碍挡路且已接近：VLM 选绕向，进入转弯
                    mode = "turning"
                    pil_img = Image.fromarray(img_vlm.astype(np.uint8)).convert("RGB")
                    action, raw = vlm_decision(model, processor, pil_img)
                    turn_dir = action
                    cmd = [TURN_CMD[action][0], 0.0, TURN_CMD[action][1]]
                    decisions.append((step, hint, f"VLM:{action}", cmd))
                    print(f"[vla] step={step} YOLO: {hint} | VLM选: {action} | 指令: {cmd}", flush=True)
                else:
                    # 障碍不在正前方或还远：转弯保持足额偏移后再回正
                    if mode == "turning":
                        turn_steps += 1
                        if turn_steps < 2:
                            pass  # 保持转弯，确保偏移足够
                        else:
                            mode = "recover"
                    if mode == "recover":
                        err = heading_error(cur_yaw)
                        if abs(err) < YAW_DEADZONE:
                            cmd = [STRAIGHT_CMD[0], 0.0, STRAIGHT_CMD[1]]
                        else:
                            cmd = [STRAIGHT_CMD[0], 0.0, float(np.clip(err, -YAW_MAX, YAW_MAX))]
                    decisions.append((step, hint, f"回正 yaw={np.degrees(err):.0f}°", cmd))
                    print(f"[vla] step={step} YOLO: {hint} | 模式={mode} | 指令: {cmd}", flush=True)
                cmd_term.command[:, :3] = torch.tensor([cmd], device="cuda:0")
            else:
                # 非决策步：回正模式下每步刷新 yaw，平滑收敛
                if mode == "recover":
                    err = heading_error(cur_yaw)
                    if abs(err) < YAW_DEADZONE:
                        cmd = [STRAIGHT_CMD[0], 0.0, STRAIGHT_CMD[1]]
                        mode = "idle"
                    else:
                        cmd = [STRAIGHT_CMD[0], 0.0, float(np.clip(err, -YAW_MAX, YAW_MAX))]
                cmd_term.command[:, :3] = torch.tensor([cmd], device="cuda:0")

            # 策略推理 + 步进（env.step 会更新相机）
            obs_td = env.get_observations()
            actions = policy.act_inference(obs_td)
            obs_td, reward, terminated, _ = env.step(actions)

            # 记录帧（每 2 步存一帧 = 25fps；叠加 VLM 画中画 + YOLO 检测框）
            if step % 2 == 0:
                det_box = None
                r = yolo_model(Image.fromarray(img_vlm.astype(np.uint8)), conf=0.25, verbose=False)[0]
                if len(r.boxes):
                    boxes = r.boxes.xyxy.cpu().numpy()
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    det_box = boxes[areas.argmax()]
                frame_out = overlay_pip(img, img_vlm, det_box)
                Image.fromarray(frame_out.astype(np.uint8)).save(
                    os.path.join(OUT_FRAMES_DIR, f"frame_{frame_idx:05d}.png")
                )
                frame_idx += 1
                recorded.append(step)
            if terminated:
                print(f"[vla] step={step} 摔倒/终止", flush=True)
                break

    elapsed = time.time() - start_time
    print(f"[vla] demo 完成: {step+1} 步, 实耗时 {elapsed:.0f}s, 存帧 {len(recorded)}", flush=True)
    print(f"[vla] 决策记录 ({len(decisions)} 次):", flush=True)
    for d in decisions:
        print(f"  step={d[0]} 检测:{d[1]} → {d[2]} 指令:{d[3]}", flush=True)

    # 合成视频
    print("[vla] 合成视频...", flush=True)
    os.system(
        f'ffmpeg -y -framerate 25 -i "{OUT_FRAMES_DIR}/frame_%05d.png" '
        f'-c:v libx264 -pix_fmt yuv420p "{OUT_VIDEO}"'
    )
    print(f"[vla] 完成: {OUT_VIDEO}", flush=True)

    # 干净退出：跳过 Isaac/Omniverse 的框架卸载阶段。
    # 该阶段与 torch/VLM(bitsandbytes) 同进程会 segfault（Py_FinalizeEx + releaseFrameworkAndTerminate），
    # 但 demo 已完成、视频已合成，成果不受影响。os._exit 直接终止进程，退出码 0。
    os._exit(0)


if __name__ == "__main__":
    main()
