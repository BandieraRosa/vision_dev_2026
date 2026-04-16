#include "armor_tracker/tracker.hpp"

#include <numbers>

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
  if (jump_cooldown_ > 0) --jump_cooldown_;  // ← 每帧递减
  Eigen::VectorXd ekf_prediction = ekf.predict();
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

  if (target_id_armors.empty())
  {
    UpdateTrackerState(matched);
    return;
  }

  if (target_id_armors.size() != armors.size())
  {
    DoYouWantToChangeTarget(armors_msg);
  }

  predicted_position = GetArmorPositionFromState(ekf_prediction);
  double position_diff = DBL_MAX;
  double yaw_diff = DBL_MAX;

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
    RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "armor_yaw_diff: %f", yaw_diff);
    if (jump_cooldown_ <= 0)
    {
      RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "armor_yaw_diff: %f", yaw_diff);
      // ArmManeuverBoost(4);
      HandleArmorJump(tracked_armor);
      jump_cooldown_ = kJumpCooldownFrames;
    }
  }
  else
  {
    double health_rate = ekf.GetHealthRate();
    if (0.5 < health_rate && health_rate < 0.8)
    {
      Eigen::Vector2d innovation_xy(
          tracked_armor.pose.position.x - predicted_position.x(),
          tracked_armor.pose.position.y - predicted_position.y());
      SoftBreakEKF(innovation_xy);
    }

    else if (health_rate < 0.5)
    {
      RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "EKF health rate: %f",
                  health_rate);
      // ResetState(double &yaw, const geometry_msgs::msg::Point &position)
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

bool Tracker::MatchArmor(const std::vector<Armor>& target_id_armors,
                         double& position_diff, double& yaw_diff, bool& is_jump)
{
  const double sign = target_state(7) >= 0.0 ? 1.0 : -1.0;
  double a2a_yaw_diff =
      2 * std::numbers::pi_v<double> / static_cast<double>(tracked_armors_num);
  double next_yaw = target_state(6) - sign * a2a_yaw_diff;

  if (target_id_armors.size() == 1)
  {
    auto armor = target_id_armors[0];
    double match_yaw_diff =
        abs(AngleDiff(OrientationToYaw(armor.pose.orientation), target_state(6)));
    double jump_yaw_diff =
        abs(AngleDiff(OrientationToYaw(armor.pose.orientation), next_yaw));
    auto p = armor.pose.position;
    Eigen::Vector3d position_vec(p.x, p.y, p.z);
    double match_position_diff = (predicted_position - position_vec).norm();
    if (match_yaw_diff < max_match_yaw_diff_)
    {
      if (match_position_diff < max_match_distance_)
      {
        tracked_armor = armor;
        is_jump = false;
        return true;
      }
    }
    else if (jump_yaw_diff < max_match_yaw_diff_)
    {
      tracked_armor = armor;
      is_jump = true;
      return false;
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
    auto armor0 = target_id_armors[0];
    Eigen::Vector3d position_vec0(armor0.pose.position.x, armor0.pose.position.y,
                                  armor0.pose.position.z);
    auto armor1 = target_id_armors[1];
    Eigen::Vector3d position_vec1(armor1.pose.position.x, armor1.pose.position.y,
                                  armor1.pose.position.z);
    double position_diff021 = abs((position_vec0 - position_vec1).norm());
    double yaw_diff021 = abs(AngleDiff(OrientationToYaw(armor0.pose.orientation),
                                       OrientationToYaw(armor1.pose.orientation)));

    if (1.4 * target_state(8) * 0.5 > position_diff ||
        position_diff021 > 1.4 * target_state(8) * 1.5 ||
        a2a_yaw_diff * 0.5 > yaw_diff021 || yaw_diff021 > a2a_yaw_diff * 1.5)
    {
      tracked_armor = armor0;
      is_jump = false;
      return false;
    }

    double match_yaw_diff0 =
        abs(AngleDiff(OrientationToYaw(armor0.pose.orientation), target_state(6)));
    double match_yaw_diff1 =
        abs(AngleDiff(OrientationToYaw(armor1.pose.orientation), target_state(6)));
    double jump_yaw_diff0 =
        abs(AngleDiff(OrientationToYaw(armor0.pose.orientation), next_yaw));
    double jump_yaw_diff1 =
        abs(AngleDiff(OrientationToYaw(armor1.pose.orientation), next_yaw));

    if (match_yaw_diff0 < max_match_yaw_diff_ && jump_yaw_diff1 < max_match_yaw_diff_)
    {
      tracked_armor = armor0;
      is_jump = false;
      return true;
    }
    else if (match_yaw_diff1 < max_match_yaw_diff_ &&
             jump_yaw_diff0 < max_match_yaw_diff_)
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
    //  position_diff = match_position_diff;
    // yaw_diff = match_yaw_diff;
  }
  return false;
}

void Tracker::UpdateEKF(double measured_yaw, const geometry_msgs::msg::Point& p)
{
  geometry_msgs::msg::Point adjusted_p = p;
  if (tracked_armors_num == ArmorsNum::OUTPOST_3)
  {
    adjusted_p.z = adjusted_p.z + (1 - outpost_idx) * outpost_dz;
  }
  measurement = Eigen::Vector4d(adjusted_p.x, adjusted_p.y, adjusted_p.z, measured_yaw);

  // 先基于 x_pri 取 innovation，后面用来判断“是否还在沿旧方向跑”
  const Eigen::VectorXd innovation = ekf.ComputeInnovation(measurement);

  target_state = ekf.update(measurement);

  if (tracked_id == "outpost")
  {
    target_state(1) = 0;
    target_state(3) = 0;
    target_state(5) = 0;
    target_state(7) = 0.8 * ((target_state(7) > 0.5) - (target_state(7) < -0.5));
  }
  else
  {
    // 新增：轻量 lag compensation
    CompensatePredictionLag(innovation);

    VelocityConstrain(5, 5, 5, 16, 1);
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
  jump_cooldown_ = 0;  // ← 新目标，清除冷却
  maneuver_boost_count_ = 0;

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

void Tracker::SoftBreakEKF(const Eigen::Vector2d& innovation_xy)
{
  // innovation_xy = measured_xy - predicted_xy
  const double innovation_norm = innovation_xy.norm();
  if (innovation_norm < 0.03)
  {
    lag_diff_count_ = 0;
    return;
  }

  Eigen::Vector2d v_xy(target_state(1), target_state(3));
  const double v_norm = v_xy.norm();
  if (v_norm < 1e-3)
  {
    lag_diff_count_ = 0;
    return;
  }

  // 残差在当前速度方向上的投影
  // 若 proj < 0，说明观测落在“当前速度方向的反方向”，也就是预测还沿旧方向跑
  const double proj = innovation_xy.dot(v_xy / v_norm);

  // 横向残差占比大，也说明可能在变向
  const double lateral =
      std::sqrt(std::max(0.0, innovation_norm * innovation_norm - proj * proj));

  // 条件 1：沿速度方向明显“拖后”
  // 条件 2：横向残差显著，说明可能在拐弯
  const bool lagging_along_velocity = (proj < -0.03);
  const bool strong_lateral_error = (lateral > 0.05 && lateral > 0.8 * std::abs(proj));

  if (lagging_along_velocity || strong_lateral_error)
  {
    lag_diff_count_++;
  }
  else
  {
    lag_diff_count_ = 0;
  }

  if (lag_diff_count_ > 2)
  {
    RCLCPP_WARN(rclcpp::get_logger("armor_tracker"),
                "Soft break EKF! proj=%.3f lateral=%.3f", proj, lateral);

    // 只刹平面速度，不直接反号，避免抖动
    target_state(1) *= 0.35;
    target_state(3) *= 0.35;

    ekf.setState(target_state);
    lag_diff_count_ = 0;
  }
}

void Tracker::CompensatePredictionLag(const Eigen::VectorXd& innovation)
{
  if (innovation.size() < 4)
  {
    return;
  }

  bool boost_needed = false;

  // innovation 是“装甲板观测 - 装甲板预测”
  // 若 residual 与当前平面速度方向相反，说明滤波器还在沿旧方向跑
  const Eigen::Vector2d v(target_state(1), target_state(3));
  const Eigen::Vector2d e(innovation(0), innovation(1));

  if (v.squaredNorm() > 1e-2)
  {
    const double proj = e.dot(v.normalized());
    if (proj < -0.08)
    {
      target_state(1) *= 0.35;
      target_state(3) *= 0.35;
      boost_needed = true;
    }
  }

  // 转向残差大，说明旧角速度也可能不对
  if (std::abs(innovation(3)) > 0.35)
  {
    target_state(7) *= 0.5;
    boost_needed = true;
  }

  if (boost_needed)
  {
    ArmManeuverBoost(4);
  }

  ekf.setState(target_state);
}

void Tracker::VelocityConstrain(double vx_max, double vy_max, double vz_max,
                                double vyaw_max, double yaw_coupling)
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
  // 位置：用观测重置
  target_state(0) = p.x + r * cos(yaw);
  target_state(2) = p.y + r * sin(yaw);
  target_state(4) = p.z;
  target_state(6) = yaw;

  // if (tracked_armors_num == ArmorsNum::OUTPOST_3)
  // {
  //   outpost_idx = 0;
  // }

  Eigen::MatrixXd P_reset = Eigen::MatrixXd::Zero(9, 9);
  P_reset(0, 0) = 0.05;  // xc
  P_reset(1, 1) = 0.5;   // v_xc   保留了旧值，但不完全信任
  P_reset(2, 2) = 0.05;  // yc
  P_reset(3, 3) = 0.5;   // v_yc
  P_reset(4, 4) = 0.05;  // za
  P_reset(5, 5) = 0.5;   // v_za
  P_reset(6, 6) = 0.1;   // yaw
  P_reset(7, 7) = 2.0;   // v_yaw  保留但给较大不确定性
  P_reset(8, 8) = 1.0;   // r

  ekf.setState(target_state, P_reset);

  RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "Reset State with P!");
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
