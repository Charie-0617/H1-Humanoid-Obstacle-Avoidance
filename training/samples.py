"""在 H1 环境里跑推理，记录 20 步样本到 samples.json（关节/基座/观测/动作）。
"""

import json
import os
from pathlib import Path

import torch  # 必须在 SimulationApp 之前导入

from isaacsim import SimulationApp

ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录
ISAACLAB_ROOT = os.environ.get("ISAACLAB_ROOT", "")
if not ISAACLAB_ROOT:
    raise RuntimeError("请设置 ISAACLAB_ROOT 指向 IsaacLab 源码目录")

app = SimulationApp(
    {
        "headless": True,
        "device": "cuda:0",
        "experience": os.path.join(ISAACLAB_ROOT, "apps/isaaclab.python.headless.kit"),
    }
)

import isaaclab_tasks  # noqa: F401 —— 注册任务环境，必须导入
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# 训练 checkpoint：由 export.py 运行时动态替换为绝对路径；
# 独立使用时指向已选好的模型。OUT_JSON 同理。
CHECKPOINT = r"assets/models/best_model.pt"
OUT_JSON = r"assets/models/samples.json"
N_STEPS = 20


def _resolve(p):
    """相对路径以仓库根目录为基准，绝对路径（export.py 注入的）原样使用。"""
    return p if os.path.isabs(p) else str(ROOT / p)


# 构建环境
env_cfg = H1FlatEnvCfg()
env_cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(env_cfg)
env = RslRlVecEnvWrapper(env)

# 从 checkpoint 构建策略
ckpt = torch.load(_resolve(CHECKPOINT), map_location="cuda:0")
from rsl_rl.modules.actor_critic import ActorCritic
from tensordict import TensorDict

num_actions = 19
# 用占位 TensorDict 构建 ActorCritic，结构要和 ckpt 里的模型一致
obs_td = TensorDict(
    {"policy": torch.zeros(1, 69), "critic": torch.zeros(1, 69)}, batch_size=[1]
)
obs_groups = {"policy": ["policy"], "critic": ["policy"]}
policy = ActorCritic(
    obs=obs_td,
    obs_groups=obs_groups,
    num_actions=num_actions,
    actor_hidden_dims=[128, 128, 128],
    critic_hidden_dims=[128, 128, 128],
    activation="elu",
    init_noise_std=1.0,
)
policy.load_state_dict(ckpt["model_state_dict"])
policy.eval()
policy.to("cuda:0")

obs_td = env.get_observations()  # TensorDict 观测
samples = []

artic = env.unwrapped.scene.articulations["robot"]

with torch.no_grad():
    for step in range(N_STEPS):
        # 步进前读状态，保证 obs 和原始状态同一时刻
        joint_pos = artic.data.joint_pos.cpu().numpy().flatten().tolist()
        joint_vel = artic.data.joint_vel.cpu().numpy().flatten().tolist()
        base_quat = artic.data.root_quat_w.cpu().numpy().flatten().tolist()  # wxyz
        base_ang_vel = artic.data.root_ang_vel_w.cpu().numpy().flatten().tolist()
        cmd_vel = env.unwrapped.command_manager.get_command("base_velocity").cpu().numpy().flatten().tolist()

        # 记录喂给策略的 obs（和状态同一时刻）
        raw_obs = obs_td.get("policy").cpu().numpy().flatten().tolist()  # [69]

        actions = policy.act_inference(obs_td)  # [1, 19]
        raw_action = actions.cpu().numpy().flatten().tolist()  # [19]

        obs_td, reward, terminated, _ = env.step(actions)

        sample = {
            "step": step,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "base_quat_wxyz": base_quat,
            "base_ang_vel": base_ang_vel,
            "cmd_vel": cmd_vel,
            "obs": raw_obs,
            "action": raw_action,
        }
        samples.append(sample)

        print(f"step {step}: done={terminated} reward={reward.item():.3f}")

with open(_resolve(OUT_JSON), "w") as f:
    json.dump(samples, f, indent=2)
print(f"Saved {len(samples)} samples to {OUT_JSON}")

app.close()
