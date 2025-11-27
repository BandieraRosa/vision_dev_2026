#include "rm_simulator_driver.hpp"

namespace rm_simulator_driver
{
SimulatorDriverNode::SimulatorDriverNode(const rclcpp::NodeOptions& options)
    : Node("simulator_driver", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting SimulatorDriverNode!");

  target_sub_ = this->create_subscription<auto_aim_interfaces::msg::Send>(
      "/tracker/send", rclcpp::SensorDataQoS(),
      std::bind(&SimulatorDriverNode::TargetCallback, this, std::placeholders::_1));

  target_eulr_pub_ =
      this->create_publisher<geometry_msgs::msg::Vector3Stamped>("/target_eulr", 10);

  fire_notify_pub_ = this->create_publisher<std_msgs::msg::Bool>("/fire_notify", 10);
}

void SimulatorDriverNode::TargetCallback(
    const auto_aim_interfaces::msg::Send::SharedPtr msg)
{
  // 1. 发送云台目标角度
  auto gimbal_cmd = geometry_msgs::msg::Vector3Stamped();
  gimbal_cmd.header.stamp = this->now();
  gimbal_cmd.header.frame_id = "odom";  // 或你的目标坐标系
  gimbal_cmd.vector.x = msg->yaw;       // Yaw
  gimbal_cmd.vector.y = msg->pitch;     // Pitch
  gimbal_cmd.vector.z = 0;              // Roll (通常为0)
  target_eulr_pub_->publish(gimbal_cmd);

  // 2. 发送开火指令
  auto shoot_cmd = std_msgs::msg::Bool();
  shoot_cmd.data = msg->is_fire;
  fire_notify_pub_->publish(shoot_cmd);
}
}  // namespace rm_simulator_driver
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_simulator_driver::SimulatorDriverNode)
