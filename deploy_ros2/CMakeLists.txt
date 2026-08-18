cmake_minimum_required(VERSION 3.8)
project(h1_policy_deploy)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# ONNX Runtime（预编译包，含 include/ 和 lib/）
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-1.29.0")
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

add_executable(policy_node src/policy_node.cpp)
ament_target_dependencies(policy_node rclcpp geometry_msgs sensor_msgs)
target_link_libraries(policy_node onnxruntime)

install(TARGETS policy_node DESTINATION lib/${PROJECT_NAME})

ament_package()
