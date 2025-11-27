#ifndef SIMULATOR_DRIVER_HPP_
#define SIMULATOR_DRIVER_HPP_

#include "auto_aim_interfaces/msg/send.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
namespace rm_simulator_driver
{
class SimulatorDriverNode : public rclcpp::Node
{
 public:
  explicit SimulatorDriverNode(const rclcpp::NodeOptions& options);

 private:
  void TargetCallback(const auto_aim_interfaces::msg::Send::SharedPtr msg);

  // 订阅 tracker_node
  rclcpp::Subscription<auto_aim_interfaces::msg::Send>::SharedPtr target_sub_;

  // 发布到仿真器
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr target_eulr_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr fire_notify_pub_;
};
}  // namespace rm_simulator_driver
#endif  // GIMBAL_CONTROLLER_HPP_
