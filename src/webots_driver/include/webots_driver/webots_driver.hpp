#ifndef WEBOTS_DRIVER_HPP
#define WEBOTS_DRIVER_HPP

#include <geometry_msgs/msg/vector3.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <webots/LED.hpp>
#include <webots/Motor.hpp>
#include <webots/Supervisor.hpp>

#include "webots_ros2_driver/PluginInterface.hpp"
#include "webots_ros2_driver/WebotsNode.hpp"

namespace webots_driver
{
class WebotsRobotDriver : public webots_ros2_driver::PluginInterface
{
 public:
  void init(webots_ros2_driver::WebotsNode* node,
            std::unordered_map<std::string, std::string>& parameters) override;
  void step() override;
  
  ~WebotsRobotDriver();

 private:
  webots_ros2_driver::WebotsNode* node_;

  webots::Supervisor* robot_;

  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr bullet_speed_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  double bullet_speed_val_;

  webots::Motor* pitch_motor_;
  webots::Motor* yaw_motor_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr target_eulr_sub_;

  void GimbalCallback(geometry_msgs::msg::Vector3::SharedPtr msg);

  webots::LED* fire_led_;
  rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr fire_notify_sub_;

  void FireNotifyCallback(std_msgs::msg::UInt8::SharedPtr msg);
};
}  // namespace webots_driver

#endif  // WEBOTS_DRIVER_HPP
