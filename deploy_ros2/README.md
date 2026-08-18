# H1 策略部署（C++ / ROS2 / ONNX）

把训练好的 H1 行走策略部署到 ROS2：订阅 `joint_states` / `imu` / `cmd_vel`，构造 69 维观测，ONNX 推理出 19 维关节位置目标，发布到 `joint_cmd`。

## 模型

- `policy.onnx`：输入 `obs[1,69]` → 输出 `actions[1,19]`
- 训练关闭观测归一化，ONNX 直接吃原始观测

关节顺序、观测布局、动作公式（`target = default_pos + 0.5 × action`）都在 `h1_joints.json`。

## 编译 & 运行

```bash
cd ~/isaac-humanoid/deploy_ros2
source /opt/ros/jazzy/setup.bash
colcon build --packages-select h1_policy_deploy
source install/setup.bash

# onnxruntime 库路径 + 模型路径
export LD_LIBRARY_PATH=~/isaac-humanoid/deploy_ros2/onnxruntime-linux-x64-1.29.0/lib:$LD_LIBRARY_PATH
export H1_POLICY_ONNX=assets/models/policy.onnx
ros2 run h1_policy_deploy policy_node

# 另开终端跑仿真，形成闭环
cd ~/isaac-humanoid/deploy_ros2/sim && ./run_sim.sh h1_sim.py
```
