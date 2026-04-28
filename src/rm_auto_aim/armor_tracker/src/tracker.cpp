#include "armor_tracker/tracker.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <numbers>
#include <rclcpp/logger.hpp>

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
  if (jump_cooldown_ > 0)
  {
    --jump_cooldown_;  // ← 每帧递减
  }
  Eigen::VectorXd ekf_prediction = ekf.Predict();
  TickManeuverBoost();
  RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF predict");

  bool matched = false;
  bool is_jump = false;
  target_state = ekf_prediction;

  const auto& armors = armors_msg->armors;

  std::vector<Armor> target_id_armors;
  target_id_armors.reserve(armors.size());

  std::copy_if(armors.begin(), armors.end(), std::back_inserter(target_id_armors),
               [id = tracked_id](const Armor& armor) { return armor.number == id; });

  bool allow_jump_candidate = false;

  if (target_id_armors.size() != armors.size())
  {
    RCLCPP_WARN(rclcpp::get_logger("armor_tracker"),
                "Target id armors size != armors size");
    DoYouWantToChangeTarget(armors_msg);
  }
  // target_id_armors = FilterSameIdArmorsYaw(target_id_armors, allow_jump_candidate);
  if (target_id_armors.empty())
  {
    if (allow_jump_candidate)
    {
      ArmManeuverBoost();
    }
    ClampTargetRadius();
    UpdateTrackerState(matched);
    return;
  }

  predicted_position = GetArmorPositionFromState(ekf_prediction);
  double position_diff = DBL_MAX;
  double yaw_diff = DBL_MAX;

  // 尝试匹配 tracked_id 的装甲板
  matched = MatchArmor(target_id_armors, position_diff, yaw_diff, is_jump);
  info_position_diff = position_diff;
  info_yaw_diff = yaw_diff;

  if (matched)
  {
    double measured_yaw = OrientationToYaw(tracked_armor.pose.orientation);
    UpdateEKF(measured_yaw, tracked_armor.pose.position);
  }
  else if (is_jump)
  {
    if (jump_cooldown_ <= 0)
    {
      RCLCPP_ERROR(rclcpp::get_logger("armor_tracker"), "Armor Jump!");
      HandleArmorJump(tracked_armor);
      jump_cooldown_ = JUMP_COOLDOWN_FRAMES;
    }
  }
  else
  {
    SoftBreakEKF(tracked_armor.pose.position);
    double health_rate = ekf.GetHealthRate();
    if (health_rate < 0.2)
    {
      RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "EKF health rate: %f",
                  health_rate);
      double tracked_armor_yaw = OrientationToYaw(tracked_armor.pose.orientation);
      ResetState(tracked_armor_yaw, tracked_armor.pose.position);
    }
  }

  ClampTargetRadius();
  UpdateTrackerState(matched);
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

// std::vector<Tracker::Armor> Tracker::FilterSameIdArmorsYaw(
//     const std::vector<Armor>& target_id_armors, bool& allow_jump_candidate)
// {
//   allow_jump_candidate = false;
//   if (target_id_armors.empty() || target_state.size() < 9)
//   {
//     return target_id_armors;
//   }

//   const double sign = target_state(7) >= 0.0 ? 1.0 : -1.0;
//   const double a2a_yaw_diff =
//       2.0 * std::numbers::pi_v<double> / static_cast<double>(tracked_armors_num);
//   const double current_phase = target_state(6);
//   const double jump_phase = target_state(6) - sign * a2a_yaw_diff;

//   const bool fast_rotation = std::fabs(target_state(7)) > 8.0 || NeedManeuverBoost();
//   const double normal_gate = std::max(max_match_yaw_diff_ * 1.25, a2a_yaw_diff * 0.32);
//   const double fast_gate = std::max(normal_gate, a2a_yaw_diff * 0.72);
//   const double fallback_gate = std::max(fast_gate, a2a_yaw_diff * 0.90);

//   std::vector<Armor> filtered;
//   filtered.reserve(target_id_armors.size());

//   double best_score = DBL_MAX;
//   Armor best_adjusted{};
//   bool has_best = false;
//   bool best_is_jump = false;

//   for (const auto& armor : target_id_armors)
//   {
//     const double raw_yaw = RawOrientationToYaw(armor.pose.orientation);
//     const auto current_yaw_opt = NormalizeArmorYawCandidate(raw_yaw, current_phase);
//     const auto jump_yaw_opt = NormalizeArmorYawCandidate(raw_yaw, jump_phase);
//     if (!current_yaw_opt || !jump_yaw_opt)
//     {
//       continue;
//     }

//     const double current_yaw = *current_yaw_opt;
//     const double jump_yaw = *jump_yaw_opt;
//     const double current_diff = std::fabs(AngleDiff(current_yaw, current_phase));
//     const double jump_diff = std::fabs(AngleDiff(jump_yaw, jump_phase));

//     const bool prefer_jump = jump_diff < current_diff;
//     const double chosen_yaw = prefer_jump ? jump_yaw : current_yaw;
//     const double chosen_diff = prefer_jump ? jump_diff : current_diff;
//     const double gate = fast_rotation ? fast_gate : normal_gate;

//     Armor adjusted = armor;
//     adjusted.pose.orientation = YawToOrientationLike(armor.pose.orientation,
//     chosen_yaw);

//     if (chosen_diff < best_score)
//     {
//       best_score = chosen_diff;
//       best_adjusted = adjusted;
//       has_best = true;
//       best_is_jump = prefer_jump;
//     }

//     if (current_diff <= normal_gate)
//     {
//       adjusted.pose.orientation = YawToOrientationLike(armor.pose.orientation,
//       current_yaw); filtered.push_back(adjusted);
//     }
//     else if (jump_diff <= normal_gate || (fast_rotation && jump_diff <= gate))
//     {
//       adjusted.pose.orientation = YawToOrientationLike(armor.pose.orientation,
//       jump_yaw); filtered.push_back(adjusted); allow_jump_candidate = true;
//     }
//     else if (fast_rotation && chosen_diff <= gate)
//     {
//       filtered.push_back(adjusted);
//       allow_jump_candidate = allow_jump_candidate || prefer_jump;
//     }
//     else
//     {
//       RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"),
//                    "Drop abnormal same-id armor yaw: current_diff=%.3f jump_diff=%.3f",
//                    current_diff, jump_diff);
//     }
//   }

//   if (filtered.empty() && has_best && best_score <= fallback_gate)
//   {
//     filtered.push_back(best_adjusted);
//     allow_jump_candidate = best_is_jump;
//     ArmManeuverBoost();
//   }

//   return filtered;
// }

bool Tracker::MatchArmor(const std::vector<Armor>& target_id_armors,
                         double& position_diff, double& yaw_diff, bool& is_jump)
{
  double sign = target_state(7) >= 0.0 ? 1.0 : -1.0;
  double a2a_yaw_diff =
      2 * std::numbers::pi_v<double> / static_cast<double>(tracked_armors_num);
  double next_yaw = target_state(6) - sign * a2a_yaw_diff;

  if (target_id_armors.size() == 1)
  {
    const auto& armor = target_id_armors[0];
    double match_yaw_diff =
        abs(AngleDiff(OrientationToYaw(armor.pose.orientation), target_state(6)));
    double jump_yaw_diff =
        abs(AngleDiff(OrientationToYaw(armor.pose.orientation), next_yaw));

    auto p = armor.pose.position;
    Eigen::Vector3d position_vec(p.x, p.y, p.z);
    double match_position_diff = (predicted_position - position_vec).norm();

    if (match_position_diff < max_match_distance_)
    {
      if (match_yaw_diff < max_match_yaw_diff_)
      {
        tracked_armor = armor;
        is_jump = false;
        return true;
      }
      else if (jump_yaw_diff < max_match_yaw_diff_)
      {
        tracked_armor = armor;
        is_jump = true;
        return false;
      }
    }
    else
    {
      tracked_armor = armor;
      is_jump = false;
      return false;
    }

    position_diff = match_position_diff;
    yaw_diff = match_yaw_diff;
  }

  else if (target_id_armors.size() == 2)
  {
    const auto& armor0 = target_id_armors[0];
    const auto& armor1 = target_id_armors[1];
    double yaw_diff021 = abs(AngleDiff(OrientationToYaw(armor0.pose.orientation),
                                       OrientationToYaw(armor1.pose.orientation)));

    if (yaw_diff021 < a2a_yaw_diff * 0.5 || yaw_diff021 > a2a_yaw_diff * 1.5)
    {
      tracked_armor = armor0;
      is_jump = false;
      return false;
    }

    double match_yaw_diff0 =
        abs(AngleDiff(OrientationToYaw(armor0.pose.orientation), target_state(6)));
    double match_yaw_diff1 =
        abs(AngleDiff(OrientationToYaw(armor1.pose.orientation), target_state(6)));

    if (match_yaw_diff0 < max_match_yaw_diff_)
    {
      tracked_armor = armor0;
      is_jump = false;
      return true;
    }
    else if (match_yaw_diff1 < max_match_yaw_diff_)
    {
      tracked_armor = armor1;
      is_jump = false;
      return true;
    }
    else
    {
      tracked_armor = armor0;
      is_jump = false;
      return false;
    }
  }
  tracked_armor = target_id_armors[0];
  is_jump = false;
  return false;
}

void Tracker::UpdateEKF(double measured_yaw, const geometry_msgs::msg::Point& armor_pose)
{
  update_count_++;
  geometry_msgs::msg::Point adjusted_p = armor_pose;
  if (tracked_armors_num == ArmorsNum::OUTPOST_3)
  {
    adjusted_p.z = adjusted_p.z + (1 - outpost_idx) * outpost_dz;
    auto measurement_z = outpost_idx == 1 ? adjusted_p.z : target_state(4);
    measurement =
        Eigen::Vector4d(adjusted_p.x, adjusted_p.y, measurement_z, measured_yaw);
  }
  else
  {
    measurement = Eigen::Vector4d(adjusted_p.x, adjusted_p.y, adjusted_p.z, measured_yaw);
  }

  target_state = ekf.Update(measurement);

  if (tracked_id == "outpost")
  {
    target_state(1) = 0;
    target_state(3) = 0;
    target_state(5) = 0;

    if (update_count_ <= 400)
    {
      last_v_yaw_ += target_state(7) / 400.0;
      last_z_ += target_state(4) / 400.0;
    }
    if (update_count_ > 400)
    {
      if (std::fabs(last_v_yaw_) > 1.5)
      {
        target_state(7) = std::copysign(0.8 * std::numbers::pi, last_v_yaw_);
      }
      target_state(4) = std::clamp(target_state(4), last_z_ - 0.001, last_z_ + 0.001);
      last_v_yaw_ = target_state(7);
      last_z_ = target_state(4);
    }
  }
  else if (std::fabs(target_state(7)) > 1.5)
  {
    if (update_count_ <= 40)
    {
      last_r_ += target_state(8) / 40.0;
    }
    if (update_count_ > 40)
    {
      target_state(8) = std::clamp(target_state(8), last_r_ - 0.0001, last_r_ + 0.0001);
      last_r_ = target_state(8);
    }
    // VelocityConstrain(5, 5, 5, 16, 1);
  }
  else
  {
    update_count_ = 0;
    target_state(8) = std::clamp(target_state(8), last_r_ - 0.001, last_r_ + 0.001);
    last_r_ = target_state(8);
  }

  RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update");
}

void Tracker::ClampTargetRadius()
{
  target_state(8) = (tracked_id == "outpost")
                        ? outpost_r
                        : std::clamp(target_state(8), radius_min, radius_max);
  ekf.SetState(target_state);
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
  jump_cooldown_ = 0;  // ← 新目标，清除冷却
  maneuver_boost_count_ = 0;
  update_count_ = 0;

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

  ekf.SetState(target_state);
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

  // UpdateArmorsNum(current_armor);
  UpdateJumpedState(position, yaw);

  last_tracked_armor = current_armor;
  ekf.SetState(target_state);
}

void Tracker::SoftBreakEKF(const geometry_msgs::msg::Point& armor_position)
{
  const double yaw = target_state(6);
  const double r = target_state(8);

  Eigen::Vector2d predicted_center(target_state(0), target_state(2));
  Eigen::Vector2d measured_armor(armor_position.x, armor_position.y);

  // 用当前 yaw/r 把装甲板观测反推为中心观测
  Eigen::Vector2d measured_center(measured_armor.x() + r * std::cos(yaw),
                                  measured_armor.y() + r * std::sin(yaw));

  // 预测 - 观测
  Eigen::Vector2d model_error = predicted_center - measured_center;
  const double error_norm = model_error.norm();

  if (error_norm < 0.08)
  {
    lag_diff_count_ = 0;
    return;
  }

  Eigen::Vector2d v_xy(target_state(1), target_state(3));
  const double v_norm = v_xy.norm();

  if (v_norm < 0.10)
  {
    lag_diff_count_ = 0;
    return;
  }

  Eigen::Vector2d v_dir = v_xy / v_norm;

  // 模型误差在速度方向上的投影
  // > 0 表示预测点在当前速度方向上冲过头
  const double forward_error = model_error.dot(v_dir);

  // 横向误差，用二维叉积绝对值表示
  const double lateral_error =
      std::abs(v_dir.x() * model_error.y() - v_dir.y() * model_error.x());

  const bool model_outward_drift =
      forward_error > 0.04 && forward_error > 0.45 * error_norm;

  const bool possible_turn = lateral_error > 0.06 && forward_error > -0.02;

  if (model_outward_drift)
  {
    lag_diff_count_++;
  }
  else
  {
    lag_diff_count_ = 0;
  }

  // 横向变向迹象：先 boost Q，不一定立刻砍速度
  if (possible_turn)
  {
    ArmManeuverBoost();
  }

  if (lag_diff_count_ >= 2)
  {
    RCLCPP_WARN(rclcpp::get_logger("armor_tracker"),
                "Soft break EKF: forward_error=%.3f lateral_error=%.3f", forward_error,
                lateral_error);

    // 衰减旧的平移速度
    target_state(1) *= 0.35;
    target_state(3) *= 0.35;

    // 接下来几帧增大过程噪声，让观测更容易拉回状态
    ArmManeuverBoost();

    ekf.SetState(target_state);

    lag_diff_count_ = 0;
  }
}

void Tracker::VelocityConstrain(double vx_max, double vy_max, double vz_max,
                                double vyaw_max, double yaw_coupling)
{
  double& vx = target_state(1);
  double& vy = target_state(3);
  double& vz = target_state(5);
  double& vyaw = target_state(7);

  double k = std::clamp(yaw_coupling, 0.0, 1.0);

  vyaw = std::clamp(vyaw, -vyaw_max, vyaw_max);

  double yaw_ratio = std::abs(vyaw) / vyaw_max;
  double trans_budget = std::sqrt(std::max(0.0, 1.0 - k * yaw_ratio * yaw_ratio));

  double trans_ratio =
      std::sqrt((vx * vx) / (vx_max * vx_max) + (vy * vy) / (vy_max * vy_max) +
                (vz * vz) / (vz_max * vz_max));

  if (trans_ratio > trans_budget && trans_ratio > 0)
  {
    double scale = trans_budget / trans_ratio;
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
  target_state(2) = p.y + r * sin(yaw);
  target_state(4) = p.z;
  target_state(6) = yaw;

  // reset 时不要保留旧速度
  target_state(1) = 0.0;
  target_state(3) = 0.0;
  target_state(5) = 0.0;

  if (tracked_id == "outpost")
  {
    target_state(7) = std::clamp(target_state(7), -3.0, 3.0);
  }
  else
  {
    target_state(7) = 0.0;
  }

  Eigen::MatrixXd p_reset = Eigen::MatrixXd::Zero(9, 9);
  p_reset(0, 0) = 0.03;
  p_reset(1, 1) = 2.0;
  p_reset(2, 2) = 0.03;
  p_reset(3, 3) = 2.0;
  p_reset(4, 4) = 0.03;
  p_reset(5, 5) = 1.0;
  p_reset(6, 6) = 0.05;
  p_reset(7, 7) = 4.0;
  p_reset(8, 8) = 0.2;

  ekf.SetState(target_state, p_reset);

  RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "Reset State with velocity and P!");
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
    int sign = target_state(7) > 0 ? 1 : -1;

    double z_diff = last_tracked_armor.pose.position.z - position.z;
    if (sign * z_diff > outpost_cast_threshold)
    {
      outpost_idx = (sign == 1) ? 0 : 2;
    }
    else
    {
      outpost_idx = (outpost_idx + sign + 3) % 3;
    }
  }

  const double r = target_state(8);
}

double Tracker::RawOrientationToYaw(const geometry_msgs::msg::Quaternion& q) const
{
  tf2::Quaternion tf_q;
  tf2::fromMsg(q, tf_q);
  double roll = NAN, pitch = NAN, yaw = NAN;
  tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
  return NormalizeAngle(yaw);
}

std::optional<double> Tracker::NormalizeArmorYawCandidate(double raw_yaw,
                                                          double reference_yaw) const
{
  if (!std::isfinite(raw_yaw) || !std::isfinite(reference_yaw) ||
      static_cast<int>(tracked_armors_num) <= 0)
  {
    return std::nullopt;
  }
  return reference_yaw + angles::shortest_angular_distance(reference_yaw, raw_yaw);
}

geometry_msgs::msg::Quaternion Tracker::YawToOrientationLike(
    const geometry_msgs::msg::Quaternion& src, double yaw) const
{
  tf2::Quaternion tf_q;
  tf2::fromMsg(src, tf_q);
  double roll = NAN, pitch = NAN, old_yaw = NAN;
  tf2::Matrix3x3(tf_q).getRPY(roll, pitch, old_yaw);

  tf2::Quaternion out;
  out.setRPY(roll, pitch, NormalizeAngle(yaw));
  out.normalize();
  return tf2::toMsg(out);
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
