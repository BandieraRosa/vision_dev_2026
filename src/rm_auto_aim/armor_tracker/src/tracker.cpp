#include "armor_tracker/tracker.hpp"

namespace rm_auto_aim
{
double Tracker::outpost_dz = 0.1;
double Tracker::outpost_r = 0.2765;
int Tracker::outpost_idx = 0;
double Tracker::outpost_cast_threshold = 0.0;

Tracker::Tracker(double max_match_distance, double max_match_yaw_diff)
    : tracker_state(State::LOST),
      tracked_id(std::string("")),
      measurement(Eigen::VectorXd::Zero(4)),
      target_state(Eigen::VectorXd::Zero(9)),
      max_match_distance_(max_match_distance),
      max_match_yaw_diff_(max_match_yaw_diff)
{
  outpost_idx = 0;
}

void Tracker::Init(const Armors::SharedPtr& armors_msg)
{
  if (armors_msg->armors.empty())
  {
    return;
  }

  double min_distance = DBL_MAX;
  tracked_armor = armors_msg->armors[0];
  for (const auto& armor : armors_msg->armors)
  {
    if (armor.distance_to_image_center < min_distance)
    {
      min_distance = armor.distance_to_image_center;
      tracked_armor = armor;
    }
  }

  InitEkf(tracked_armor);
  RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "Init EKF!");

  tracked_id = tracked_armor.number;
  tracked_armor_type = tracked_armor.type;
  tracker_state = State::DETECTING;
  UpdateArmorsNum(tracked_armor);
}

void Tracker::Update(const Armors::SharedPtr& armors_msg)
{
  Eigen::VectorXd ekf_prediction = ekf.predict();
  RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF predict");

  bool matched = false;
  target_state = ekf_prediction;

  if (armors_msg->armors.empty())
  {
    UpdateTrackerState(matched);
    return;
  }
  int same_id_armors_count = 0;
  predicted_position = GetArmorPositionFromState(ekf_prediction);
  double min_position_diff = DBL_MAX;
  double yaw_diff = DBL_MAX;

  matched = MatchArmor(armors_msg, ekf_prediction, same_id_armors_count,
                       min_position_diff, yaw_diff);
  if (matched)
  {
    double measured_yaw = OrientationToYaw(tracked_armor.pose.orientation);
    UpdateEKF(measured_yaw, tracked_armor.pose.position);
  }

  info_position_diff = min_position_diff;
  info_yaw_diff = yaw_diff;

  if (!matched)
  {
    double health_rate = ekf.GetHealthRate();
    double effective_yaw_thresh =
        (tracked_armor.number != "outpost") ? max_match_yaw_diff_
                                            : max_match_yaw_diff_ + 0.7;

    if (same_id_armors_count == 1 && yaw_diff > effective_yaw_thresh)
    {
      RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "armor_yaw_diff: %f", yaw_diff);
      HandleArmorJump(tracked_armor);
    }
    else if (0.5 < health_rate && health_rate < 0.8)
    {
      SoftBreakEKF(ekf_prediction(2), tracked_armor.pose.position.y);
    }
    else if (health_rate < 0.5)
    {
      RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "EKF health rate: %f",
                  health_rate);
    }
  }

  // 防止半径扩散
  ClampTargetRadius();

  // 跟踪状态机制处理
  UpdateTrackerState(matched);

  if (armors_msg->armors.size() > 1)
  {
    DoYouWantToChangeTarget(armors_msg);
  }
}

void Tracker::DoYouWantToChangeTarget(const Armors::SharedPtr& armors_msg)
{
  if (armors_msg->armors.empty())
  {
    return;
  }

  double min_distance = DBL_MAX;
  auto closest_armor = armors_msg->armors[0];
  for (const auto& armor : armors_msg->armors)
  {
    if (armor.distance_to_image_center < min_distance)
    {
      min_distance = armor.distance_to_image_center;
      closest_armor = armor;
    }
  }

  if (closest_armor.number != tracked_id && closest_armor.number == last_closest_id)
  {
    if (change_count_ < change_thres)
    {
      change_count_++;
    }
    else
    {
      RCLCPP_WARN(rclcpp::get_logger("armor_tracker"),
                  "Confirmed target change to armor ID: %s",
                  closest_armor.number.c_str());
      tracked_id = closest_armor.number;
      tracked_armor = closest_armor;
      InitEkf(tracked_armor);
      UpdateArmorsNum(tracked_armor);
      tracker_state = State::DETECTING;
      change_count_ = 0;
    }
  }
  else
  {
    change_count_ = 0;
  }

  last_closest_id = closest_armor.number;
}

bool Tracker::MatchArmor(const Armors::SharedPtr& armors_msg,
                         const Eigen::VectorXd& ekf_prediction,
                         int& same_id_armors_count, double& min_position_diff,
                         double& yaw_diff)
{
  int correct_match_count = 0;
  double best_yaw_diff = DBL_MAX;

  for (const auto& armor : armors_msg->armors)
  {
    if (armor.number == tracked_id)
    {
      same_id_armors_count++;

      auto p = armor.pose.position;
      Eigen::Vector3d position_vec(p.x, p.y, p.z);
      double position_diff = (predicted_position - position_vec).norm();

      double local_yaw_diff =
          std::abs(OrientationToYaw(armor.pose.orientation) - ekf_prediction(6));

      double effective_yaw_thresh =
          (tracked_armor.number != "outpost") ? max_match_yaw_diff_
                                              : max_match_yaw_diff_ + 0.7;

      if (position_diff < max_match_distance_ && local_yaw_diff < effective_yaw_thresh)
      {
        if (position_diff < min_position_diff)
        {
          min_position_diff = position_diff;
          best_yaw_diff = local_yaw_diff;
          tracked_armor = armor;
        }
        correct_match_count++;
      }
      else
      {
        // 即使不匹配，也记录最小 yaw_diff 供外部跳变判断使用
        if (local_yaw_diff < best_yaw_diff)
        {
          best_yaw_diff = local_yaw_diff;
        }
      }
    }
  }

  yaw_diff = best_yaw_diff;

  return correct_match_count > 0;
}

void Tracker::UpdateEKF(double measured_yaw, const geometry_msgs::msg::Point& p)
{
  geometry_msgs::msg::Point adjusted_p = p;
  if (tracked_armors_num == ArmorsNum::OUTPOST_3)
  {
    adjusted_p.z = p.z + (1 - outpost_idx) * outpost_dz;
  }

  measurement = Eigen::Vector4d(adjusted_p.x, adjusted_p.y, adjusted_p.z, measured_yaw);
  target_state = ekf.update(measurement);

  if (tracked_id == "outpost")
  {
    target_state(1) = 0;
    target_state(3) = 0;
    target_state(5) = 0;
    // FIX: 正负对称的固定角速度
    target_state(7) =
        (target_state(7) > 0 ? 1.0 : -1.0) * outpost_vyaw_abs;
  }
  else
  {
    VelocityConstrain(10, 10, 10, 10, 1);
  }

  RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update");
}

void Tracker::ClampTargetRadius()
{
  target_state(8) = (tracked_id == "outpost")
                        ? outpost_r
                        : std::clamp(target_state(8), radius_min, radius_max);
  ekf.setState(target_state);
}

void Tracker::UpdateTrackerState(bool matched)
{
  if (tracker_state == State::DETECTING)
  {
    if (matched)
    {
      detect_count_++;
      if (detect_count_ > tracking_thres)
      {
        detect_count_ = 0;
        tracker_state = State::TRACKING;
      }
    }
    else
    {
      detect_count_ = 0;
      tracker_state = State::LOST;
    }
  }
  else if (tracker_state == State::TRACKING)
  {
    if (!matched)
    {
      tracker_state = State::TEMP_LOST;
      lost_count_++;
    }
  }
  else if (tracker_state == State::TEMP_LOST)
  {
    if (!matched)
    {
      lost_count_++;
      if (lost_count_ > lost_thres)
      {
        lost_count_ = 0;
        tracker_state = State::LOST;
      }
    }
    else
    {
      tracker_state = State::TRACKING;
      lost_count_ = 0;
    }
  }
}

void Tracker::SwitchEKFParams() { switch_q_(is_outpost); }

void Tracker::InitEkf(const Armor& armor)
{
  first_tracked = true;
  is_outpost = (armor.number == "outpost");
  SwitchEKFParams();

  double xa = armor.pose.position.x;
  double ya = armor.pose.position.y;
  double za = armor.pose.position.z;

  {
    tf2::Quaternion tf_q;
    tf2::fromMsg(armor.pose.orientation, tf_q);
    double roll = NAN, pitch = NAN, yaw = NAN;
    tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
    last_yaw_ = yaw;
  }
  double yaw = OrientationToYaw(armor.pose.orientation);

  double r = is_outpost ? outpost_r : default_init_radius;
  double xc = xa + r * cos(yaw);
  double yc = ya + r * sin(yaw);
  dz = 0;
  another_r = r;

  target_state = Eigen::VectorXd::Zero(9);
  target_state << xc, 0, yc, 0, za, 0, yaw, 0, r;

  ekf.setState(target_state);
}

void Tracker::HandleArmorJump(const Armor& current_armor)
{
  if (first_tracked)
  {
    first_tracked = false;
    last_tracked_armor = current_armor;
  }

  auto position = current_armor.pose.position;
  auto orientation = current_armor.pose.orientation;
  double yaw = OrientationToYaw(orientation);

  UpdateArmorsNum(current_armor);
  UpdateJumpedState(position, yaw);

  RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "Armor jump!");

  Eigen::Vector3d current_p(position.x, position.y, position.z);
  Eigen::Vector3d infer_p = GetArmorPositionFromState(target_state);
  if ((current_p - infer_p).norm() > max_match_distance_)
  {
    ResetState(yaw, position);
  }

  last_tracked_armor = current_armor;
  ekf.setState(target_state);
}

void Tracker::SoftBreakEKF(const double y_pri, const double y_mea)
{
  int sign = (target_state(3) > 0) - (target_state(3) < 0);
  if ((y_pri - y_mea) * sign > 0.1)
  {
    y_diff_count_++;
  }
  else
  {
    y_diff_count_ = 0;
  }
  if (y_diff_count_ > 3)
  {
    RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "Soft break EKF!");
    target_state(3) *= -0.5;
    ekf.setState(target_state);
    y_diff_count_ = 0;
  }
}

void Tracker::VelocityConstrain(double vx_max, double vy_max, double vz_max,
                                double vyaw_max, double yaw_coupling = 1.0)
{
  double& vx = target_state(1);
  double& vy = target_state(3);
  double& vz = target_state(5);
  double& vyaw = target_state(7);

  const double k = std::clamp(yaw_coupling, 0.0, 1.0);

  vyaw = std::clamp(vyaw, -vyaw_max, vyaw_max);

  const double yaw_ratio = std::abs(vyaw) / vyaw_max;
  const double trans_budget = std::sqrt(std::max(0.0, 1.0 - k * yaw_ratio * yaw_ratio));

  const double trans_ratio =
      std::sqrt((vx * vx) / (vx_max * vx_max) + (vy * vy) / (vy_max * vy_max) +
                (vz * vz) / (vz_max * vz_max));

  if (trans_ratio > trans_budget && trans_ratio > 0)
  {
    const double scale = trans_budget / trans_ratio;
    vx *= scale;
    vy *= scale;
    vz *= scale;
  }
}

void Tracker::UpdateArmorsNum(const Armor& armor)
{
  if (armor.number == "outpost")
  {
    tracked_armors_num = ArmorsNum::OUTPOST_3;
  }
  else
  {
    tracked_armors_num = ArmorsNum::NORMAL_4;
  }
}

void Tracker::ResetState(double& yaw, const geometry_msgs::msg::Point& p)
{
  double r = target_state(8);
  target_state(0) = p.x + r * cos(yaw);
  target_state(1) = 0;
  target_state(2) = p.y + r * sin(yaw);
  target_state(3) = 0;
  target_state(4) = p.z;
  target_state(5) = 0;
  if (tracked_armors_num == ArmorsNum::OUTPOST_3)
  {
    outpost_idx = 0;
  }
  RCLCPP_ERROR(rclcpp::get_logger("armor_tracker"), "Reset State!");
}

void Tracker::UpdateJumpedState(const geometry_msgs::msg::Point& position, double yaw)
{
  target_state(6) = yaw;

  if (tracked_armors_num == ArmorsNum::NORMAL_4)
  {
    dz = target_state(4) - position.z;
    std::swap(target_state(8), another_r);
    target_state(4) = position.z;
  }
  else if (tracked_armors_num == ArmorsNum::OUTPOST_3)
  {
    double z_diff = last_tracked_armor.pose.position.z - position.z;
    int sign = target_state(7) > 0 ? 1 : -1;
    if (std::fabs(z_diff) > outpost_cast_threshold)
    {
      outpost_idx = (sign == 1) ? 0 : 2;
    }
    else
    {
      outpost_idx = (outpost_idx + sign + 3) % 3;
    }
    RCLCPP_INFO(rclcpp::get_logger("armor_tracker"),
                "Outpost Jump: z_diff=%.3f, current_idx=%d", z_diff, outpost_idx);
  }
}

double Tracker::OrientationToYaw(const geometry_msgs::msg::Quaternion& q)
{
  tf2::Quaternion tf_q;
  tf2::fromMsg(q, tf_q);
  double roll = NAN, pitch = NAN, yaw = NAN;
  tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
  yaw = last_yaw_ + angles::shortest_angular_distance(last_yaw_, yaw);
  last_yaw_ = yaw;
  return yaw;
}

Eigen::Vector3d Tracker::GetArmorPositionFromState(const Eigen::VectorXd& x)
{
  double xc = x(0), yc = x(2), za = x(4);
  double yaw = x(6), r = x(8);
  double xa = xc - r * cos(yaw);
  double ya = yc - r * sin(yaw);
  return Eigen::Vector3d(xa, ya, za);
}

}  // namespace rm_auto_aim

