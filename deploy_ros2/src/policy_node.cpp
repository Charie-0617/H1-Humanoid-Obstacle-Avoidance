/**
 * H1 PPO 策略部署节点（C++ + ROS2 + ONNX Runtime）。
 *
 * 订阅 joint_states + IMU + cmd_vel → 拼 69 维观测 → ONNX 推理 → 发布 joint_cmd。
 * 观测布局和关节顺序见 h1_joints.json（部署契约）。
 */
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int64_t OBS_DIM = 69;   // H1 观测维度
constexpr int64_t ACT_DIM = 19;   // H1 动作维度
constexpr float kActionScale = 0.5f;

// 19 个驱动关节名，顺序与 ONNX 输出、观测排列一致
const std::vector<std::string> kJointNames = {
    "left_hip_yaw",       "right_hip_yaw",      "torso",
    "left_hip_roll",      "right_hip_roll",
    "left_shoulder_pitch","right_shoulder_pitch",
    "left_hip_pitch",     "right_hip_pitch",
    "left_shoulder_roll", "right_shoulder_roll",
    "left_knee",          "right_knee",
    "left_shoulder_yaw",  "right_shoulder_yaw",
    "left_ankle",         "right_ankle",
    "left_elbow",         "right_elbow",
};

// 默认位形，与 kJointNames 对齐
const std::vector<float> kDefaultPos = {
    0.0f,  0.0f,  0.0f,
    0.0f,  0.0f,
    0.28f, 0.28f,
    -0.28f,-0.28f,
    0.0f,  0.0f,
    0.79f, 0.79f,
    0.0f,  0.0f,
    -0.52f,-0.52f,
    0.52f, 0.52f,
};

// 四元数旋转 q(x,y,z,w)，body -> world
std::array<float, 3> quat_rotate(const std::array<float, 4>& q,
                                 const std::array<float, 3>& v) {
  const float x = q[0], y = q[1], z = q[2], w = q[3];
  const float vx = v[0], vy = v[1], vz = v[2];
  // t = 2 * (q_vec x v)
  const float tx = 2.0f * (y * vz - z * vy);
  const float ty = 2.0f * (z * vx - x * vz);
  const float tz = 2.0f * (x * vy - y * vx);
  return {
      vx + w * tx + y * tz - z * ty,
      vy + w * ty + z * tx - x * tz,
      vz + w * tz + x * ty - y * tx,
  };
}

// world -> body
std::array<float, 3> quat_rotate_inverse(const std::array<float, 4>& q,
                                         const std::array<float, 3>& v) {
  return quat_rotate({-q[0], -q[1], -q[2], q[3]}, v);
}

// 定位 ONNX 模型：优先环境变量 H1_POLICY_ONNX，其次包内相对路径
std::string resolve_model_path() {
  const char* env = std::getenv("H1_POLICY_ONNX");
  if (env != nullptr) return std::string(env);
  const std::vector<std::string> candidates = {
      "models/policy.onnx",
      "install/h1_policy_deploy/lib/h1_policy_deploy/models/policy.onnx",
  };
  for (const auto& p : candidates) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (f != nullptr) { std::fclose(f); return p; }
  }
  return "models/policy.onnx";  // 兜底，节点启动时再报错
}

}  // namespace

class PolicyNode : public rclcpp::Node {
 public:
  PolicyNode() : Node("h1_policy_node") {
    std::string model_path = resolve_model_path();
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "h1_policy");
    session_ = std::make_unique<Ort::Session>(
        *env_, model_path.c_str(), Ort::SessionOptions{});

    Ort::AllocatorWithDefaultOptions alloc;
    input_name_ = session_->GetInputNameAllocated(0, alloc).get();
    output_name_ = session_->GetOutputNameAllocated(0, alloc).get();
    RCLCPP_INFO(get_logger(), "ONNX 加载完成: %s  输入 '%s' (%ld 维) -> 输出 '%s' (%ld 维)",
                model_path.c_str(), input_name_.c_str(), OBS_DIM,
                output_name_.c_str(), ACT_DIM);

    joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "joint_states", 10,
        [this](const sensor_msgs::msg::JointState::SharedPtr msg) { on_joint(msg); });
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "imu", 10,
        [this](const sensor_msgs::msg::Imu::SharedPtr msg) { on_imu(msg); });
    cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        "cmd_vel", 10,
        [this](const geometry_msgs::msg::Twist::SharedPtr msg) { on_cmd(msg); });
    base_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        "base_vel", 10,
        [this](const geometry_msgs::msg::Twist::SharedPtr msg) { on_base_vel(msg); });
    pub_ = create_publisher<sensor_msgs::msg::JointState>("joint_cmd", 10);

    joint_pos_.assign(ACT_DIM, 0.0);
    joint_vel_.assign(ACT_DIM, 0.0);
    last_action_.assign(ACT_DIM, 0.0f);
    imu_quat_ = {1.0f, 0.0f, 0.0f, 0.0f};  // x,y,z,w
    imu_angvel_ = {0.0f, 0.0f, 0.0f};

    RCLCPP_INFO(get_logger(), "H1 策略部署节点已就绪（69 维观测 -> 19 维动作）");
  }

 private:
  void on_joint(const sensor_msgs::msg::JointState::SharedPtr msg) {
    // 按名字匹配到 kJointNames 顺序
    int matched = 0;
    for (size_t i = 0; i < kJointNames.size(); ++i) {
      for (size_t j = 0; j < msg->name.size(); ++j) {
        if (msg->name[j] == kJointNames[i]) {
          joint_pos_[i] = (j < msg->position.size()) ? msg->position[j] : 0.0;
          joint_vel_[i] = (j < msg->velocity.size()) ? msg->velocity[j] : 0.0;
          ++matched;
          break;
        }
      }
    }
    if (matched != static_cast<int>(ACT_DIM)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "关节名匹配 %d/%ld（期望 %ld）", matched,
                           kJointNames.size(), ACT_DIM);
    }
  }

  void on_base_vel(const geometry_msgs::msg::Twist::SharedPtr msg) {
    base_vel_ = {static_cast<float>(msg->linear.x),
                 static_cast<float>(msg->linear.y),
                 static_cast<float>(msg->linear.z)};
    has_base_vel_ = true;
  }

  void on_imu(const sensor_msgs::msg::Imu::SharedPtr msg) {
    imu_quat_ = {static_cast<float>(msg->orientation.x),
                 static_cast<float>(msg->orientation.y),
                 static_cast<float>(msg->orientation.z),
                 static_cast<float>(msg->orientation.w)};
    imu_angvel_ = {static_cast<float>(msg->angular_velocity.x),
                   static_cast<float>(msg->angular_velocity.y),
                   static_cast<float>(msg->angular_velocity.z)};
  }

  void on_cmd(const geometry_msgs::msg::Twist::SharedPtr msg) {
    std::vector<float> obs = build_obs(msg);
    // H1_DEBUG_OBS 时打印观测，用于 Sim2Sim 对比
    if (std::getenv("H1_DEBUG_OBS") != nullptr) {
      std::printf("OBS_DEBUG ");
      for (float v : obs) std::printf("%.6f ", v);
      std::printf("\n");
      std::fflush(stdout);
    }
    std::vector<float> action = infer(obs);

    // 关节位置目标 = 默认位形 + 0.5 * action
    sensor_msgs::msg::JointState js;
    js.header.stamp = now();
    js.name = kJointNames;
    js.position.resize(ACT_DIM);
    for (int i = 0; i < ACT_DIM; ++i) {
      js.position[i] = static_cast<double>(kDefaultPos[i] + kActionScale * action[i]);
    }
    pub_->publish(js);

    // 更新上一步动作
    last_action_ = action;

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                         "cmd_vel=(%.2f, %.2f) -> 发布 19 维关节目标",
                         msg->linear.x, msg->linear.y);
  }

  // 构造 69 维观测，严格按头部布局
  std::vector<float> build_obs(const geometry_msgs::msg::Twist::SharedPtr& cmd) {
    std::vector<float> obs(OBS_DIM, 0.0f);

    // [0:3] base_lin_vel：有真实基座速度用真实值，否则回退 cmd_vel
    obs[0] = has_base_vel_ ? base_vel_[0] : static_cast<float>(cmd->linear.x);
    obs[1] = has_base_vel_ ? base_vel_[1] : static_cast<float>(cmd->linear.y);
    obs[2] = has_base_vel_ ? base_vel_[2] : static_cast<float>(cmd->linear.z);

    // [3:6] base_ang_vel：IMU 角速度
    obs[3] = imu_angvel_[0];
    obs[4] = imu_angvel_[1];
    obs[5] = imu_angvel_[2];

    // [6:9] projected_gravity：机体系下重力方向
    auto g = quat_rotate_inverse(imu_quat_, {0.0f, 0.0f, -1.0f});
    obs[6] = g[0];
    obs[7] = g[1];
    obs[8] = g[2];

    // [9:12] velocity_commands
    obs[9] = static_cast<float>(cmd->linear.x);
    obs[10] = static_cast<float>(cmd->linear.y);
    obs[11] = static_cast<float>(cmd->angular.z);

    // [12:31] joint_pos_rel（实际角 - 默认位形）
    for (int i = 0; i < ACT_DIM; ++i)
      obs[12 + i] = static_cast<float>(joint_pos_[i]) - kDefaultPos[i];

    // [31:50] joint_vel
    for (int i = 0; i < ACT_DIM; ++i)
      obs[31 + i] = static_cast<float>(joint_vel_[i]);

    // [50:69] last_action
    for (int i = 0; i < ACT_DIM; ++i)
      obs[50 + i] = last_action_[i];

    return obs;
  }

  // ONNX 推理：输入 [1,69] 观测，输出 [1,19] 动作
  std::vector<float> infer(const std::vector<float>& obs) {
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = {1, OBS_DIM};
    auto input_tensor = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(obs.data()), obs.size(), shape.data(), shape.size());

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                 input_names, &input_tensor, 1,
                                 output_names, 1);

    const float* act = outputs[0].GetTensorData<float>();
    return std::vector<float>(act, act + ACT_DIM);
  }

  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;

  // 缓存的传感器状态
  std::vector<double> joint_pos_;
  std::vector<double> joint_vel_;
  std::vector<float> last_action_;
  std::array<float, 4> imu_quat_;   // x,y,z,w
  std::array<float, 3> imu_angvel_;
  std::array<float, 3> base_vel_ = {0.0f, 0.0f, 0.0f};  // 基座线速度（body 系）
  bool has_base_vel_ = false;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr base_vel_sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PolicyNode>());
  rclcpp::shutdown();
  return 0;
}
