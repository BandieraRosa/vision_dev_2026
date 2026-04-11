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
#include "tracker.hpp"

namespace rm_auto_aim
{
using armors_tf2_filter = tf2_ros::MessageFilter<auto_aim_interfaces::msg::Armors>;
class ArmorTrackerNode : public rclcpp::Node
{
 public:
  explicit ArmorTrackerNode(const rclcpp::NodeOptions& options);

 private:
  void InitParameters();

  void ArmorsCallback(const auto_aim_interfaces::msg::Armors::SharedPtr armors_ptr);

  // Maximum allowable armor distance in the XOY plane
  double max_armor_distance_;

  // The time when the last message was received
  rclcpp::Time last_time_;
  double dt_;

  // Armor tracker
  double s2_q_x_, s2_q_y_, s2_q_z_, s2_q_yaw_, s2_q_r_;
  double s2_q_x_armor_, s2_q_y_armor_, s2_q_z_armor_, s2_q_yaw_armor_, s2_q_r_armor_;
  double s2qxyz_outpost_, s2qyaw_outpost_, s2qr_outpost_;
  double r_xyz_factor_, r_yaw_;

  double r_xyz_base_;
  double r_xyz_dist_gain_;
  double r_xyz_oblique_gain_;

  double r_z_scale_;

  double r_yaw_base_;
  double r_yaw_dist_gain_;
  double r_yaw_oblique_gain_;
  double r_yaw_outpost_scale_;

  double lost_time_thres_;
  double change_time_thres_;
  std::unique_ptr<Tracker> tracker_;

  // Subscriber with tf2 message_filter
  std::string target_frame_;
  std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;
  message_filters::Subscriber<auto_aim_interfaces::msg::Armors> armors_sub_;
  std::shared_ptr<armors_tf2_filter> armors_filter_;

  // Tracker info publisher
  rclcpp::Publisher<auto_aim_interfaces::msg::TrackerInfo>::SharedPtr info_pub_;

  // Publisher
  rclcpp::Publisher<auto_aim_interfaces::msg::Target>::SharedPtr target_pub_;

  // Lob shot
  bool is_hero_{false};
  std::string last_frame_id_ = "camera_optical_frame";
};

}  // namespace rm_auto_aim

#endif  // ARMOR_PROCESSOR__PROCESSOR_NODE_HPP_
