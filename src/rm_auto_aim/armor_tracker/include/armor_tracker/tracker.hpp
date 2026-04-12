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

  // ---- 核心流程 ----
  void Init(const Armors::SharedPtr& armors_msg);
  void Update(const Armors::SharedPtr& armors_msg);

  // ---- 子步骤（从 Update 中拆分） ----

  /// @brief 在多装甲板场景下判断是否切换跟踪目标
  void DoYouWantToChangeTarget(const Armors::SharedPtr& armors_msg);

  /// @brief 匹配当前帧中与 tracked_id 相同的装甲板，返回是否匹配成功
  ///        匹配成功时 tracked_armor 被更新为最佳匹配
  bool MatchArmor(const Armors::SharedPtr& armors_msg,
                  const Eigen::VectorXd& ekf_prediction, int& same_id_armors_count,
                  double& min_position_diff, double& yaw_diff);

  /// @brief 匹配成功后，执行 EKF update 并做速度约束
  void UpdateEKF(double measured_yaw, const geometry_msgs::msg::Point& p);

  /// @brief 防止半径发散
  void ClampTargetRadius();

  /// @brief 跟踪状态机转移
  void UpdateTrackerState(bool matched);
  bool NeedManeuverBoost() const { return maneuver_boost_count_ > 0; }

  void ArmManeuverBoost(int count = 4)
  {
    if (count > maneuver_boost_count_)
    {
      maneuver_boost_count_ = count;
    }
  }

  void TickManeuverBoost()
  {
    if (maneuver_boost_count_ > 0)
    {
      --maneuver_boost_count_;
    }
  }

  ExtendedKalmanFilter ekf;

  int tracking_thres;
  int lost_thres;
  std::string last_closest_id;
  int change_thres;

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

  double radius_min = 0.12;           // 半径下限
  double radius_max = 0.4;            // 半径上限
  double default_init_radius = 0.26;  // 非前哨站初始半径
  double outpost_vyaw_abs = 0.8;      // 前哨站固定角速度绝对值

  Eigen::VectorXd measurement;   // 测量
  Eigen::VectorXd target_state;  // 目标状态
  Eigen::Vector3d predicted_position{};

  //? 储存另一片装甲板信息
  double dz, another_r;

 private:
  static double NormalizeAngle(double a) { return std::remainder(a, 2.0 * M_PI); }
  static double AngleDiff(double a, double b) { return NormalizeAngle(a - b); }
  void InitEkf(const Armor& a);
  void UpdateArmorsNum(const Armor& a);
  void ResetState(double& yaw, const geometry_msgs::msg::Point& position);
  void UpdateJumpedState(const geometry_msgs::msg::Point& position, double yaw);
  void HandleArmorJump(const Armor& current_armor);
  void SoftBreakEKF(const Eigen::Vector2d& innovation_xy);
  void VelocityConstrain(double vx_max, double vy_max, double vz_max, double vyaw_max,
                         double yaw_coupling);

  double OrientationToYaw(const geometry_msgs::msg::Quaternion& q);
  Eigen::Vector3d GetArmorPositionFromState(const Eigen::VectorXd& x);
  void SwitchEKFParams();
  void CompensatePredictionLag(const Eigen::VectorXd& innovation);
  double max_match_distance_;
  double max_match_yaw_diff_;
  int maneuver_boost_count_ = 0;

  int detect_count_ = 0;
  int lost_count_ = 0;
  int change_count_ = 0;
  int lag_diff_count_ = 0;

  double last_yaw_ = 0.0;  // FIX: 显式初始化，避免首次 OrientationToYaw 结果随机
};

}  // namespace rm_auto_aim

#endif  // ARMOR_PROCESSOR__TRACKER_HPP_
