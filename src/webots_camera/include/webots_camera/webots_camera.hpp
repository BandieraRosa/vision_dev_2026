#pragma once

#include <cv_bridge/cv_bridge.h>

#include <geometry_msgs/msg/quaternion.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "webots_ros2_driver/PluginInterface.hpp"
#include "webots_ros2_driver/WebotsNode.hpp"

// Webots Headers
#include <opencv2/core.hpp>
#include <webots/Camera.hpp>
#include <webots/Node.hpp>
#include <webots/Robot.hpp>
#include <webots/Supervisor.hpp>

namespace webots_camera
{
class WebotsCamera : public webots_ros2_driver::PluginInterface
{
 public:
  void init(webots_ros2_driver::WebotsNode* node,
            std::unordered_map<std::string, std::string>& parameters) override;
  void step() override;
  ~WebotsCamera();

 private:
  // 模拟 Hik 相机发布 CameraInfo (基于 Webots 真实 FOV 计算)
  sensor_msgs::msg::CameraInfo GetCameraInfo(int width, int height, double fov,
                                             rclcpp::Time now);

  // 云台姿态解算 (保留原有逻辑)
  geometry_msgs::msg::Quaternion CalculateGimbalRotation();

  webots_ros2_driver::WebotsNode* node_;
  webots::Supervisor* robot_;

  // Webots Devices
  webots::Camera* webots_camera_;
  webots::Node* webots_camera_node_;  // 用于获取姿态

  // Parameters mimicking HikCamera
  std::string camera_name_;
  std::string frame_id_;
  double target_fps_;

  // Time control
  int publish_period_ms_;
  int last_publish_time_ms_ = 0;

  // ROS Publishers
  std::shared_ptr<image_transport::ImageTransport> it_;

  // 与 HikCameraNode 保持一致: 使用 image_transport::CameraPublisher
  image_transport::CameraPublisher camera_pub_;

  // 额外的云台姿态发布
  rclcpp::Publisher<geometry_msgs::msg::Quaternion>::SharedPtr rotation_pub_;
};
}  // namespace webots_camera
