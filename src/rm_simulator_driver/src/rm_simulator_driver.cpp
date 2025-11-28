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
  auto target_eulr = geometry_msgs::msg::Vector3Stamped();
  target_eulr.header.stamp = this->now();
  target_eulr.header.frame_id = "odom";  // 目标坐标系
  target_eulr.vector.x = msg->yaw;       // Yaw
  target_eulr.vector.y = msg->pitch;     // Pitch
  target_eulr.vector.z = 0;              // Roll (通常为0)
  target_eulr_pub_->publish(target_eulr);

  // 2. 发送开火指令
  auto fire_notify = std_msgs::msg::Bool();
  fire_notify.data = msg->is_fire;
  fire_notify_pub_->publish(fire_notify);
}
}  // namespace rm_simulator_driver
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_simulator_driver::SimulatorDriverNode)
