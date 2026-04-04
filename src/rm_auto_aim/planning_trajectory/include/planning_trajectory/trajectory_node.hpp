#ifndef ARMOR_PROCESSOR__PROCESSOR_NODE_HPP_
#define ARMOR_PROCESSOR__PROCESSOR_NODE_HPP_

// ROS
#include <message_filters/subscriber.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>

#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/int32.hpp>

#include "auto_aim_interfaces/msg/send.hpp"
#include "auto_aim_interfaces/msg/target.hpp"
#include "auto_aim_interfaces/msg/tracker_info.hpp"
#include "auto_aim_interfaces/msg/velocity.hpp"
#include "planning_trajectory/trajectory.hpp"
#include "planning_trajectory/trajectory_solver.hpp"
#include "auto_aim_interfaces/msg/trajectory_info.hpp"

namespace rm_auto_aim
{
using velocity_tf2_filter = tf2_ros::MessageFilter<auto_aim_interfaces::msg::Velocity>;
class PlanningTrajectoryNode : public rclcpp::Node
{
 public:
  explicit PlanningTrajectoryNode(const rclcpp::NodeOptions& options);

 private:
  void Init();

  void VelocityCallback(const auto_aim_interfaces::msg::Velocity::SharedPtr velocity_msg);

  void TargetCallback(const auto_aim_interfaces::msg::Target::SharedPtr target_msg);

  void timer_callback();

  std::pair<double, double> GetGimbalYawAndPitch();

  std::unique_ptr<TrajectorySolver> gaf_solver_;

  rclcpp::Subscription<auto_aim_interfaces::msg::Velocity>::SharedPtr velocity_sub_;

  rclcpp::Subscription<auto_aim_interfaces::msg::Target>::SharedPtr target_sub_;

  rclcpp::TimerBase::SharedPtr timer_;
  std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;

  // Publisher
  rclcpp::Publisher<auto_aim_interfaces::msg::Send>::SharedPtr send_pub_;
  rclcpp::Publisher<auto_aim_interfaces::msg::TrajectoryInfo>::SharedPtr info_pub_;

  // Lob shot
  std::string table_filename_normal_;
  std::string table_filename_lob_;
  Table::TableConfig table_config_;
  Table::TableConfig table_config_lob_;

  // control
  double k_yaw_;
  double k_pitch_;
  bool is_hero_;

  bool tracking_{false};
  // SolveTrajectory::ArmorVelocity target_velocity_;
  // SolveTrajectory::ArmorPostion target_position_;
  TrajectorySolver::Target target_;
  double send_frequency_;
  int send_count_{0};
  double dt_;
  double q_yaw_;
  double q_pitch_;
  double q_vy_;
  double q_ay_;
  double r_yaw_;
  double r_pitch_;

  std::unique_ptr<Trajectory> trajectory_;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_PROCESSOR__PROCESSOR_NODE_HPP_