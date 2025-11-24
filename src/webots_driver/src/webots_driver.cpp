#include "webots_driver/webots_driver.hpp"

#include <algorithm>

#include "pluginlib/class_list_macros.hpp"

namespace webots_driver
{
void WebotsRobotDriver::init(webots_ros2_driver::WebotsNode* node,
                             std::unordered_map<std::string, std::string>& parameters)
{
  node_ = node;

  robot_ = new webots::Supervisor();

  bullet_speed_val_ = 30.0;
  if (parameters.find("bullet_speed") != parameters.end())
  {
    try
    {
      bullet_speed_val_ = std::stod(parameters["bullet_speed"]);
    }
    catch (...)
    {
      RCLCPP_WARN(node_->get_logger(),
                  "Invalid bullet_speed parameter, using default 30.0");
    }
  }

  bullet_speed_pub_ = node_->create_publisher<std_msgs::msg::Float32>("bullet_speed", 10);

  timer_ = node_->create_wall_timer(std::chrono::milliseconds(100),
                                    [this]()
                                    {
                                      std_msgs::msg::Float32 msg;
                                      msg.data =
                                          static_cast<float>(this->bullet_speed_val_);
                                      this->bullet_speed_pub_->publish(msg);
                                    });

  pitch_motor_ = robot_->getMotor("target_motor_pitch");
  yaw_motor_ = robot_->getMotor("target_motor_yaw");

  if (pitch_motor_)
  {
    pitch_motor_->setVelocity(10.0);
    pitch_motor_->setPosition(0.0);
  }
  else
  {
    RCLCPP_ERROR(node_->get_logger(), "Motor 'target_motor_pitch' not found!");
  }

  if (yaw_motor_)
  {
    yaw_motor_->setVelocity(10.0);
    yaw_motor_->setPosition(0.0);
  }
  else
  {
    RCLCPP_ERROR(node_->get_logger(), "Motor 'target_motor_yaw' not found!");
  }

  target_eulr_sub_ = node_->create_subscription<geometry_msgs::msg::Vector3>(
      "target_eulr", 10,
      std::bind(&WebotsRobotDriver::GimbalCallback, this, std::placeholders::_1));

  fire_led_ = robot_->getLED("fire_led");
  if (!fire_led_)
  {
    RCLCPP_WARN(node_->get_logger(), "LED 'fire_led' not found!");
  }

  fire_notify_sub_ = node_->create_subscription<std_msgs::msg::UInt8>(
      "fire_notify", 10,
      std::bind(&WebotsRobotDriver::FireNotifyCallback, this, std::placeholders::_1));

  RCLCPP_INFO(node_->get_logger(), "WebotsDriver initialized successfully.");
}

WebotsRobotDriver::~WebotsRobotDriver()
{
  if (robot_)
  {
    delete robot_;
    robot_ = nullptr;
  }
}

void WebotsRobotDriver::step() {}

void WebotsRobotDriver::GimbalCallback(const geometry_msgs::msg::Vector3::SharedPtr& msg)
{
  if (pitch_motor_)
  {
    double pitch = std::clamp(msg->x, -0.65, 0.37);
    pitch_motor_->setPosition(pitch);
  }
  if (yaw_motor_)
  {
    yaw_motor_->setPosition(msg->y);
  }
}

void WebotsRobotDriver::FireNotifyCallback(const std_msgs::msg::UInt8::SharedPtr& msg)
{
  if (fire_led_)
  {
    fire_led_->set(msg->data > 0 ? 1 : 0);
  }
}

}  // namespace webots_driver

PLUGINLIB_EXPORT_CLASS(webots_driver::WebotsRobotDriver,
                       webots_ros2_driver::PluginInterface)
