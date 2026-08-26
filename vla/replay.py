"""重放 record.py 的轨迹，渲染 2K 视频（2560x1440 + 画中画 + YOLO 绿框）。

输出：assets/demo/demo_vla.mp4
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

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

# 路径配置（相对仓库根目录）
YOLO_WEIGHTS = str(ROOT / "assets" / "models" / "yolo" / "best.pt")
OBSTACLE_POS = (5.0, 0.0, 0.6)
TRAJ_PATH = str(ROOT / "assets" / "demo" / "traj.npz")
OUT_FRAMES_DIR = str(ROOT / "assets" / "demo" / "frames")
OUT_VIDEO = str(ROOT / "assets" / "demo" / "demo_vla.mp4")
CAM_REC_H, CAM_REC_W = 1440, 2560  # 2K
CAM_VLM_H, CAM_VLM_W = 240, 320


def overlay_pip(frame, pip, det_box=None):
    """录制帧右上角叠加 VLM 第一人称画中画 + YOLO 检测框 + 白框 + 标签。"""
    from PIL import Image, ImageDraw, ImageFont
    fimg = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
    pimg = Image.fromarray(pip.astype(np.uint8)).convert("RGB")
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
    # 画中画尺寸：宽 560（2K 画面里占比与 1080p 的 420 相近），按比例缩放
    pw = 560
    ph = int(pw * pimg.height / pimg.width)
    pimg = pimg.resize((pw, ph), Image.BILINEAR)
    x, y = fimg.width - pw - 36, 36
    fimg.paste(pimg, (x, y))
    d = ImageDraw.Draw(fimg)
    d.rectangle([x - 4, y - 4, x + pw + 3, y + ph + 3], outline=(255, 255, 255), width=4)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 26)
    except Exception:
        font = ImageFont.load_default()
    d.text((x, y + ph + 12), "VLM 视角（第一人称）", fill=(255, 255, 255),
           stroke_width=3, stroke_fill=(0, 0, 0), font=font)
    return np.array(fimg)


def main():
    print("[replay] 加载轨迹...", flush=True)
    traj = np.load(TRAJ_PATH)
    root_pos = traj["root_pos"]      # [N,3]
    root_quat = traj["root_quat"]    # [N,4]
    joint_pos = traj["joint_pos"]    # [N,19]
    rec_eye = traj["cam_rec_eye"]    # [N,3]
    rec_tgt = traj["cam_rec_tgt"]
    vlm_eye = traj["cam_vlm_eye"]
    vlm_tgt = traj["cam_vlm_tgt"]
    N = root_pos.shape[0]
    print(f"[replay] 轨迹 {N} 帧", flush=True)

    print("[replay] 加载 YOLO...", flush=True)
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
        prim_path="/World/Camera", update_period=0.0, height=CAM_REC_H, width=CAM_REC_W, data_types=["rgb"],
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
    cam_rec = env.unwrapped.scene["cam_rec"]
    cam_vlm = env.unwrapped.scene["cam_vlm"]

    import carb
    carb.settings.get_settings().set("/rtx/post/tonemap/filmIso", 80.0)
    carb.settings.get_settings().set("/rtx/post/tonemap/whitepoint", 2500.0)

    os.makedirs(OUT_FRAMES_DIR, exist_ok=True)

    env.reset()
    sim = env.unwrapped.sim
    scene = env.unwrapped.scene

    # 预热渲染
    cam_rec.set_world_poses_from_view(
        torch.tensor([rec_eye[0]], dtype=torch.float32, device="cuda:0"),
        torch.tensor([rec_tgt[0]], dtype=torch.float32, device="cuda:0"),
    )
    cam_vlm.set_world_poses_from_view(
        torch.tensor([vlm_eye[0]], dtype=torch.float32, device="cuda:0"),
        torch.tensor([vlm_tgt[0]], dtype=torch.float32, device="cuda:0"),
    )
    for _ in range(30):
        sim.render()
        scene.update(sim.get_physics_dt())

    print("[replay] 开始 2K 重放渲染...", flush=True)
    saved = 0
    for i in range(N):
        # 摆机器人姿态（root pose + 关节）
        pose = np.concatenate([root_pos[i], root_quat[i]]).astype(np.float32)
        artic.write_root_pose_to_sim(torch.tensor([pose], device="cuda:0"))
        jp = joint_pos[i].astype(np.float32)
        artic.write_joint_state_to_sim(
            torch.tensor([jp], device="cuda:0"),
            torch.zeros(1, 19, device="cuda:0"),
        )
        # 摆相机
        cam_rec.set_world_poses_from_view(
            torch.tensor([rec_eye[i]], dtype=torch.float32, device="cuda:0"),
            torch.tensor([rec_tgt[i]], dtype=torch.float32, device="cuda:0"),
        )
        cam_vlm.set_world_poses_from_view(
            torch.tensor([vlm_eye[i]], dtype=torch.float32, device="cuda:0"),
            torch.tensor([vlm_tgt[i]], dtype=torch.float32, device="cuda:0"),
        )
        # 渲染
        sim.render()
        scene.update(sim.get_physics_dt())

        # 读帧
        img = cam_rec.data.output["rgb"][0, ..., :3].cpu().numpy()
        if img.max() <= 1.0:
            img = img * 255
        img_vlm = cam_vlm.data.output["rgb"][0, ..., :3].cpu().numpy()
        if img_vlm.max() <= 1.0:
            img_vlm = img_vlm * 255

        # YOLO 画框（画中画小窗）
        det_box = None
        r = yolo_model(Image.fromarray(img_vlm.astype(np.uint8)), conf=0.25, verbose=False)[0]
        if len(r.boxes):
            boxes = r.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            det_box = boxes[areas.argmax()]

        frame_out = overlay_pip(img, img_vlm, det_box)
        Image.fromarray(frame_out.astype(np.uint8)).save(
            os.path.join(OUT_FRAMES_DIR, f"frame_{i:05d}.png")
        )
        saved += 1
        if i % 50 == 0:
            print(f"[replay] 帧 {i}/{N} 亮度={img.mean():.0f}", flush=True)

    print(f"[replay] 重放完成: {saved} 帧 → {OUT_FRAMES_DIR}", flush=True)
    print("[replay] 合成 2K 视频...", flush=True)
    os.system(
        f'ffmpeg -y -framerate 25 -i "{OUT_FRAMES_DIR}/frame_%05d.png" '
        f'-c:v libx264 -pix_fmt yuv420p -crf 18 "{OUT_VIDEO}"'
    )
    print(f"[replay] 完成: {OUT_VIDEO}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
