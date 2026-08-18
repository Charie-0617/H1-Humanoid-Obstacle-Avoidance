#!/usr/bin/env python
"""H1 仿真器：Isaac Sim headless 200Hz 跑 H1Flat 环境，经 ROS2 和部署节点对接。

发布 /joint_states /imu /base_vel /cmd_vel，订阅 /joint_cmd。
链路：cmd_vel → 部署节点(ONNX) → joint_cmd → 本仿真器 → H1 行走

用法：./run_sim.sh h1_sim.py --record walk_log.npz --duration 15 --cmd 0.45 --stand 3
"""
import argparse
import time

from isaacsim import SimulationApp

config = {
    "headless": True,
    "physics_dt": 0.005,
    "extra_args": [
        "--/app/renderer/enabled=false",
        "--/persistent/isaac/asset_root/cloud=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1",
        "--enable", "isaacsim.ros2.bridge",
    ],
}
app = SimulationApp(config)

import numpy as np
import torch

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg_PLAY

parser = argparse.ArgumentParser()
parser.add_argument("--record", type=str, default=None, help="输出 walk_log.npz 路径")
parser.add_argument("--duration", type=float, default=15.0)
parser.add_argument("--cmd", type=float, default=0.45)
parser.add_argument("--stand", type=float, default=3.0)
parser.add_argument("--ramp", type=float, default=1.5)
args = parser.parse_args()

# 原生 H1Flat 环境
env_cfg = H1FlatEnvCfg_PLAY()
env_cfg.scene.num_envs = 1
env_cfg.sim.device = "cpu"
env = gym.make("Isaac-Velocity-Flat-H1-v0", cfg=env_cfg)
robot = env.unwrapped.scene["robot"]

n_joints = robot.num_joints
joint_names = list(robot.joint_names)
name_to_idx = {name: i for i, name in enumerate(joint_names)}
default_jp = robot.data.default_joint_pos[0].clone()
print(f"[h1_sim] H1 就绪: {n_joints} 关节", flush=True)

env.reset()
step_dt = env_cfg.decimation * env_cfg.sim.dt  # 0.02s (50Hz)

# ROS2
import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, JointState

rclpy.init()
node = rclpy.create_node("h1_sim_node")
pub_js = node.create_publisher(JointState, "joint_states", 10)
pub_imu = node.create_publisher(Imu, "imu", 10)
pub_bv = node.create_publisher(Twist, "base_vel", 10)
pub_cmd = node.create_publisher(Twist, "cmd_vel", 10)

# 订阅 /joint_cmd：绝对位置目标 -> 环境原始动作 (target - default) / 0.5
raw_action = torch.zeros(1, n_joints, dtype=torch.float32)

def on_joint_cmd(msg):
    """部署节点发来绝对关节位置目标，反推成环境的原始动作。

    部署节点按 target = default + 0.5*action 发布，环境内部又按同样的
    公式执行，所以这里做逆运算把位置目标还原成 raw_action。
    """
    a = raw_action[0]
    for i, nm in enumerate(msg.name):
        idx = name_to_idx.get(nm)
        if idx is not None and i < len(msg.position):
            a[idx] = float((msg.position[i] - default_jp[idx]) / 0.5)

print("[h1_sim] ROS2 节点就绪: 发 /joint_states /imu /base_vel /cmd_vel, 收 /joint_cmd", flush=True)

# 部署节点预热：先发状态+cmd=0，等收到第一条 joint_cmd 再开始步进
joint_cmd_received = {"v": False}

def _on_joint_cmd(msg):
    on_joint_cmd(msg)
    joint_cmd_received["v"] = True

node.create_subscription(JointState, "joint_cmd", _on_joint_cmd, 10)

def _publish_state():
    """把机器人当前状态发布出去：joint_states / imu / base_vel。"""
    js = JointState()
    js.name = joint_names
    js.position = [float(v) for v in robot.data.joint_pos[0]]
    js.velocity = [float(v) for v in robot.data.joint_vel[0]]
    pub_js.publish(js)

    imu = Imu()
    # Isaac 四元数是 wxyz，ROS 的 Imu 是 xyzw，这里重排
    q = robot.data.root_quat_w[0]
    imu.orientation.x = float(q[1]); imu.orientation.y = float(q[2])
    imu.orientation.z = float(q[3]); imu.orientation.w = float(q[0])
    # root_ang_vel_b 是机体系角速度，直接给 IMU
    av = robot.data.root_ang_vel_b[0]
    imu.angular_velocity.x = float(av[0]); imu.angular_velocity.y = float(av[1]); imu.angular_velocity.z = float(av[2])
    pub_imu.publish(imu)

    bv = Twist()
    lv = robot.data.root_lin_vel_b[0]
    bv.linear.x = float(lv[0]); bv.linear.y = float(lv[1]); bv.linear.z = float(lv[2])
    pub_bv.publish(bv)

def _publish_cmd(cmd_x):
    cmd = Twist()
    cmd.linear.x = cmd_x
    pub_cmd.publish(cmd)

# 统一用一个订阅（on_joint_cmd 更新 raw_action，_on_joint_cmd 标记收到）
warmup_start = time.time()
while not joint_cmd_received["v"]:
    _publish_state()
    _publish_cmd(0.0)
    rclpy.spin_once(node, timeout_sec=0.05)
    if time.time() - warmup_start > 15.0:
        print("[h1_sim] 警告：15 秒内未收到部署节点 joint_cmd，继续", flush=True)
        break
print(f"[h1_sim] 部署节点已就绪（预热 {time.time()-warmup_start:.1f}s）", flush=True)

# 主循环（请求-响应同步，消除观测滞后）
log_t, log_jp, log_rp, log_rq = [], [], [], []
n_steps = int(args.duration / step_dt)

for i in range(n_steps):
    t_sim = i * step_dt

    # cmd_vel：0 站立 -> 爬坡 -> 目标速度
    if t_sim < args.stand:
        cmd_x = 0.0
    else:
        cmd_x = min(args.cmd, args.cmd * (t_sim - args.stand) / args.ramp)

    # 1) 发状态
    _publish_state()
    # 2) 给部署节点一点时间缓存最新状态（消除 cmd 与状态到达的竞态）
    rclpy.spin_once(node, timeout_sec=0.005)
    # 3) 发 cmd_vel 触发推理
    _publish_cmd(cmd_x)
    # 4) 等 joint_cmd 回来（最多 60ms）
    joint_cmd_received["v"] = False
    t0 = time.time()
    while not joint_cmd_received["v"]:
        rclpy.spin_once(node, timeout_sec=0.005)
        if time.time() - t0 > 0.06:
            break
    # 5) 应用动作（环境内部按默认+0.5*action 执行 PD）
    env.step(raw_action)

    # 日志
    if args.record:
        log_t.append(t_sim)
        log_jp.append(robot.data.joint_pos[0].tolist())
        rp = robot.data.root_pos_w[0]; rq_ = robot.data.root_quat_w[0]
        log_rp.append([float(rp[0]), float(rp[1]), float(rp[2])])
        log_rq.append([float(rq_[0]), float(rq_[1]), float(rq_[2]), float(rq_[3])])

    # 进度
    if i % 25 == 0:
        rp = robot.data.root_pos_w[0]
        print(f"[h1_sim] t={t_sim:.1f}s 基座=({rp[0]:.2f},{rp[1]:.2f},{rp[2]:.2f}) cmd_x={cmd_x:.2f}", flush=True)

# 收尾
if args.record and log_t:
    np.savez_compressed(
        args.record,
        t=np.array(log_t),
        joint_pos=np.array(log_jp),
        root_pos=np.array(log_rp),
        root_quat_wxyz=np.array(log_rq),
        joint_names=np.array(joint_names, dtype=object),
    )
    print(f"[h1_sim] 日志已保存: {args.record}（{len(log_t)} 帧）", flush=True)

print("[h1_sim] 完成", flush=True)
rclpy.shutdown()
app.close()
