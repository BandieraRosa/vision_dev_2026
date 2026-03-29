
#ifndef ARMOR_PROCESSOR__TRACKER_HPP_
#define ARMOR_PROCESSOR__TRACKER_HPP_

// ROS
#include <angles/angles.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "armor_tracker/extended_kalman_filter.hpp"
#include "auto_aim_interfaces/msg/armors.hpp"

namespace rm_auto_aim
{

using VoidBoolFunc = std::function<void(bool)>;

// 装甲板数量 正常的4块 前哨站三块
enum class ArmorsNum : uint8_t
{
  NORMAL_4 = 4,
  OUTPOST_3 = 3
};

class Tracker  // 整车观测
{
 public:
  Tracker(double max_match_distance, double max_match_yaw_diff);

  using Armors = auto_aim_interfaces::msg::Armors;
  using Armor = auto_aim_interfaces::msg::Armor;

  void Init(const Armors::SharedPtr& armors_msg);

  bool MatchArmor(const Armors::SharedPtr& armors_msg, Eigen::VectorXd& ekf_prediction,
                  int& same_id_armors_count, double& min_position_diff, double& yaw_diff);
  void ClampTargetRadius();
  void UpdateTrackerState(bool& matched);
  void Update(const Armors::SharedPtr& armors_msg);

  ExtendedKalmanFilter ekf;

  int tracking_thres;
  int lost_thres;

  enum class State : uint8_t
  {             // 四个状态
    LOST,       // 丢失
    DETECTING,  // 观测中
    TRACKING,   // 跟踪中
    TEMP_LOST,  // 临时丢失
  } tracker_state;

  // 装甲板情况
  std::string tracked_id;          // 装甲板号
  Armor tracked_armor;             // 被跟踪的装甲板
  ArmorsNum tracked_armors_num;    // 被跟踪装甲版数
  std::string tracked_armor_type;  // 被跟踪装甲板类型
  Armor last_tracked_armor{};      // 上一次被跟踪的装甲板
  bool first_tracked = true;
  bool is_outpost = false;
  VoidBoolFunc switch_q_;

  double info_position_diff;
  double info_yaw_diff;

  static double outpost_dz;
  static double outpost_r;
  static int outpost_idx;
  static double outpost_cast_threshold;

  Eigen::VectorXd measurement;  // 测量

  Eigen::VectorXd target_state;  // 目标状态

  Eigen::Vector3d predicted_position{};

  //? 储存另一片装甲板信息
  double dz, another_r;

 private:
  void InitEkf(const Armor& a);

  void UpdateArmorsNum(const Armor& a);

  void ResetState(double& yaw, const geometry_msgs::msg::Point& position);
  void UpdateJumpedState(const geometry_msgs::msg::Point& position, double yaw);
  void HandleArmorJump(const Armor& current_armor);
  void SoftBreakEKF(const double y_pri, const double y_mea);
  void VelocityConstrain(double vx_max, double vy_max,
                                    double vz_max, double vyaw_max,
                                    double yaw_coupling);

  double OrientationToYaw(const geometry_msgs::msg::Quaternion& q);

  Eigen::Vector3d GetArmorPositionFromState(const Eigen::VectorXd& x);

  void SwitchEKFParams();

  double max_match_distance_;
  double max_match_yaw_diff_;

  int detect_count_;
  int lost_count_;
  int y_diff_count_ = 0;

  double last_yaw_;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_PROCESSOR__TRACKER_HPP_
