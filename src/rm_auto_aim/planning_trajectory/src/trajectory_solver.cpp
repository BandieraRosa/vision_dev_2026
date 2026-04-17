#include "planning_trajectory/trajectory_solver.hpp"

#include <cmath>
#include <memory>

namespace rm_auto_aim
{

TrajectorySolver::TrajectorySolver(const double& k, const double& bias_time,
                                   const double& s_bias, const double& z_bias,
                                   const double& pitch_bias, CalculateMode calculate_mode,
                                   const Table::TableConfig& table_config,
                                   const Table::TableConfig& table_config_lob_)
    : table_(std::make_shared<Table>(table_config)),
      table_lob_(std::make_shared<Table>(table_config_lob_)),
      calculate_mode_(calculate_mode),
      k_(k),
      pitch_bias_(pitch_bias),
      bias_time_(bias_time),
      s_bias_(s_bias),
      z_bias_(z_bias)
{
  if (calculate_mode_ == CalculateMode::TABLE_LOOKUP)
  {
    table_->Init();
    table_lob_->Init();
    current_table_ = table_;
    if (current_table_->IsInit() && table_lob_->IsInit())
    {
      RCLCPP_INFO(logger_, "Trajectory table initialized successfully");
    }
    else if (!table_->IsInit())
    {
      calculate_mode_ = CalculateMode::NORMAL;
      RCLCPP_WARN(logger_, "Using normal calculation mode");
    }
    else
    {
      RCLCPP_WARN(logger_,
                  "LOB table failed to initialize, LOB mode will be unavailable");
    }
  }
}

void TrajectorySolver::Init(
    const auto_aim_interfaces::msg::Velocity::SharedPtr velocity_msg)
{
  if (!std::isnan(velocity_msg->velocity))
  {
    current_v_ = velocity_msg->velocity;
    RCLCPP_DEBUG(logger_, "Velocity updated: %.2f m/s", current_v_);
  }
  else
  {
    RCLCPP_WARN(logger_, "Invalid velocity, using default: 20.0 m/s");
    current_v_ = 12.0f;
  }
}

void TrajectorySolver::ReBuild()
{
  selected_idx_ = LOST;
  choose_next_ = false;
  should_last_shot_ = false;
  turn_s_ = 0.0;
  step_s_ = 0.0;
  selected_time_delay_ = 0.0;
  start_turn_ = time_point::min();
  end_turn_ = time_point::min();
  last_start_turn_ = time_point::min();
  pre_center_ = {};
  pre_position_.fill({});
  last_x_v_ = 0.0;
  last_y_v_ = 0.0;
  last_v_yaw_ = 0.0;
  last_pitch_ = 0.0;
  last_yaw_ = 0.0;
  fly_time_ = 0.0;
}

TrajectorySolver::TarPostion TrajectorySolver::PredictCenter(double time_delay)
{
  TarPostion center;
  if (target_.num == 4)
  {
    center.x = target_.position.x + target_.velocity.x * time_delay;
    center.y = target_.position.y + target_.velocity.y * time_delay;
    center.z = target_.position.z;
    center.yaw = NormalizeAngle(target_.position.yaw + target_.velocity.yaw * time_delay);
  }
  else
  {
    center.x = target_.position.x;
    center.y = target_.position.y;
    center.z = target_.position.z;
    center.yaw = NormalizeAngle(target_.position.yaw + target_.velocity.yaw * time_delay);
  }
  pre_center_ = PredictCenter(time_delay);
  return center;
}

TrajectorySolver::TarPostion TrajectorySolver::PredictArmor(
    int idx, const TrajectorySolver::TarPostion& pre_center)
{
  TarPostion pre_pos;
  const double sign = target_.velocity.yaw >= 0.0 ? 1.0 : -1.0;
  const double delta = sign * idx * 2.0 * std::numbers::pi_v<double> / target_.num;
  const double tmp_yaw = NormalizeAngle(pre_center.yaw - delta);

  if (target_.num == 4)
  {
    const double radius = (idx % 2 == 0) ? target_.radius1 : target_.radius2;
    pre_pos.x = pre_center.x - radius * std::cos(tmp_yaw);
    pre_pos.y = pre_center.y - radius * std::sin(tmp_yaw);
    pre_pos.z = pre_center.z;
    pre_pos.yaw = tmp_yaw;
  }
  else
  {
    const double radius = target_.radius1;
    pre_pos.x = pre_center.x - radius * std::cos(tmp_yaw);
    pre_pos.y = pre_center.y - radius * std::sin(tmp_yaw);

    const int offset = (sign > 0.0) ? idx : ((target_.num - idx) % target_.num);
    const int id = (target_.outpost_idx + offset) % target_.num;

    pre_pos.z = pre_center.z + outpost_dz * (id - 1);
    pre_pos.yaw = tmp_yaw;
  }

  return pre_pos;
}

// 从图片时间到打到的时间：自瞄处理的时间+电控延迟(从视觉发信号到电机动和发弹延迟)+云台转动时间+飞行时间
// msg消息的频率即我们发送开火指令的频率，这可以作为我们的步长时间
void TrajectorySolver::PredictAllArmorPosition(double time_delay)
{
  PredictCenter(time_delay);
  pre_position_.fill({});

  for (int i = 0; i < target_.num; ++i)
  {
    pre_position_[i] = PredictArmor(i, pre_center_);
  }
}

void TrajectorySolver::PredictOneArmorPosition(double time_delay, int idx)
{
  PredictCenter(time_delay);
  pre_position_[idx] = PredictArmor(idx, pre_center_);
}

// 计算简化单向空气阻力模型下的弹道高度，用于正常模式
double TrajectorySolver::MonoDirectionalAirResistanceModel(double s, double angle,
                                                           double v)
{
  double cos_angle = std::cos(angle);
  if (cos_angle <= 0)
  {
    RCLCPP_WARN(logger_, "Invalid angle: cos(angle) <= 0");
    fly_time_ = 0;
    return 0;
  }

  fly_time_ = (std::exp(k_ * s) - 1) / (k_ * v * cos_angle);

  if (fly_time_ < 0)
  {
    RCLCPP_WARN(logger_, "Exceeding maximum range! s: %.2f, v: %.2f", s, v);
    fly_time_ = 0;
    return 0;
  }

  return v * sin(angle) * fly_time_ - GRAVITY * fly_time_ * fly_time_ / 2;
}

// 计算俯仰角(两种模式)
double TrajectorySolver::SolvePitch(double x, double y, double z)
{
  // 计算水平距离
  double distance = std::sqrt(x * x + y * y);
  double target_s = distance + s_bias_;
  double target_z = z + z_bias_;

  double pitch = 0.0f;

  if (calculate_mode_ == CalculateMode::TABLE_LOOKUP && current_table_->IsInit())
  {
    // 查表法
    auto res = current_table_->Check(target_s, target_z);
    if (!std::isnan(res.pitch))
    {
      fly_time_ = res.t;
      pitch = res.pitch;
    }
    else
    {
      fly_time_ = 0;
      pitch = 0.0f;
      RCLCPP_WARN(logger_, "Table lookup nan for s: %.2f, z: %.2f", target_s, target_z);
    }
  }
  else
  {
    // 正常模式下的迭代计算
    double z_temp = target_z;

    for (int i = 0; i < 20; ++i)
    {
      if (std::isnan(z_temp))
      {
        RCLCPP_ERROR(logger_, "z_temp is NaN during iteration");
        return 0.0f;
      }

      pitch = std::atan2(z_temp, target_s);
      double z_actual = MonoDirectionalAirResistanceModel(target_s, pitch, current_v_);
      double dz = 0.3f * (target_z - z_actual);
      z_temp += dz;

      if (fabs(dz) < 1e-5f)
      {
        RCLCPP_DEBUG(logger_, "Pitch convergence after %d iterations", i + 1);
        break;
      }
    }
  }
  pitch += pitch_bias_;
  return pitch;
}

double TrajectorySolver::SolveYaw(double x, double y) const { return std::atan2(y, x); }

double fast_atan(double x, double y)
{
  double x_y = y / x;
  double x_y_2 = x_y * x_y;
  return x_y * (0.99997726f + x_y_2 * (-0.33262347f + x_y_2 * 0.19354346f));
}

// 快速打击符号fast_fire为false时，只打云台和跟踪都就位的装甲板
// bool TrajectorySolver::CanFire(double tar_yaw, double tar_pitch, bool is_fast_fire)
// {
//   if (!HasValidSelection())
//   {
//     return false;
//   }

//   const double distance =
//       std::sqrt(pre_position_[selected_idx_].x * pre_position_[selected_idx_].x +
//                 pre_position_[selected_idx_].y * pre_position_[selected_idx_].y) +
//       s_bias_;

//   const double armor_half_length =
//       (target_.type == "small") ? SMALL_HALF_LENGTH : LARGE_HALF_LENGTH;

//   const double max_yaw_diff = SolveYaw(distance, armor_half_length);

//   const bool stable_tracking = std::fabs(target_.velocity.x - last_x_v_) < 0.4 &&
//                                std::fabs(target_.velocity.y - last_y_v_) < 0.3 &&
//                                std::fabs(target_.velocity.yaw - last_v_yaw_) < 0.3;

//   if (!stable_tracking && !is_fast_fire && !should_last_shot_)
//   {
//     return false;
//   }

//   else
//   {
//     bool yaw_diff_exceeds = fabs(tar_yaw - gimbal_yaw_) > max_yaw_diff &&
//                             fabs(tar_pitch - gimbal_pitch_) > 0.02;
//     if (choose_next_)
//     {
//       if (yaw_diff_exceeds)
//       {
//         // RCLCPP_WARN(logger_, "云台和跟踪都未就位");
//         return false;
//       }
//       // RCLCPP_WARN(logger_, "云台就位而跟踪未就位");
//       return is_fast_fire;
//     }
//     else
//     {
//       if (yaw_diff_exceeds)
//       {
//         // RCLCPP_WARN(logger_, "跟踪就位而云台未就位");
//         return is_fast_fire;
//       }
//       // RCLCPP_DEBUG(logger_, "云台和跟踪都就位");
//       return true;
//     }
//   }
// }
std::pair<double, double> TrajectorySolver::ComputeFireYawWindow(
    const TarPostion& armor) const
{
  const double half_length_ = (target_.type == "small") ? SMALL_HALF_LENGTH : LARGE_HALF_LENGTH;

  const double sy = std::sin(armor.yaw);
  const double cy = std::cos(armor.yaw);

  const double ax = armor.x - half_length_ * sy;
  const double ay = armor.y + half_length_ * cy;
  const double bx = armor.x + half_length_ * sy;
  const double by = armor.y - half_length_ * cy;

  const double angle_c = SolveYaw(armor.x, armor.y);
  const double angle_a = SolveYaw(ax, ay);
  const double angle_b = SolveYaw(bx, by);

  const double lo = AngleDiff(angle_b, angle_c);
  const double hi = AngleDiff(angle_a, angle_c);
  return {std::min(lo, hi), std::max(lo, hi)};
}

bool TrajectorySolver::CanFire(double tar_yaw, double tar_pitch, bool is_fast_fire)
{
  if (!HasValidSelection()) return false;

  const auto& p = pre_position_[selected_idx_];
  const auto [yaw_lo, yaw_hi] = ComputeFireYawWindow(p);

  const double control_delta = AngleDiff(tar_yaw, gimbal_yaw_);
  const bool yaw_ok = (control_delta >= yaw_lo) && (control_delta <= yaw_hi);
  const bool pitch_ok = std::fabs(tar_pitch - gimbal_pitch_) <= 0.02;

  const bool stable_tracking = std::fabs(target_.velocity.x - last_x_v_) < 0.4 &&
                               std::fabs(target_.velocity.y - last_y_v_) < 0.3 &&
                               std::fabs(target_.velocity.yaw - last_v_yaw_) < 0.3;

  if (!stable_tracking && !is_fast_fire && !should_last_shot_)
  {
    return false;
  }

  const bool angle_diff_exceeds = !yaw_ok || !pitch_ok;

  if (choose_next_)
  {
    if (angle_diff_exceeds) return false;
    return is_fast_fire;
  }
  if (angle_diff_exceeds) return is_fast_fire;
  return true;
}

void TrajectorySolver::GlobalSelectArmor(double time_delay)
{
  double best_cost = std::numeric_limits<double>::infinity();
  int best_idx = 0;
  double best_time = time_delay;

  PredictAllArmorPosition(time_delay);

  for (int i = 0; i < target_.num; ++i)
  {
    const double base_yaw = SolveYaw(pre_position_[i].x, pre_position_[i].y);
    const double turn_time = 0.05 * std::fabs(AngleDiff(base_yaw, gimbal_yaw_));

    const TarPostion center = PredictCenter(time_delay + turn_time);
    const TarPostion armor = PredictArmor(i, center);

    const double aim_yaw = SolveYaw(armor.x, armor.y);
    const double cost = std::fabs(AngleDiff(aim_yaw, gimbal_yaw_));

    if (cost < best_cost)
    {
      best_cost = cost;
      best_idx = i;
      best_time = time_delay + turn_time;
    }
  }

  selected_idx_ = best_idx;
  selected_time_delay_ = best_time;
  PredictOneArmorPosition(best_time, best_idx);
}

// void TrajectorySolver::LocalSelectArmor(double time_delay)
// {
//   if (std::fabs(target_.velocity.yaw) < 0.3)
//   {
//     selected_idx_ = 0;
//     selected_time_delay_ = time_delay;
//     PredictOneArmorPosition(time_delay, selected_idx_);
//     return;
//   }

//   const TarPostion center0 = PredictCenter(time_delay);
//   const TarPostion armor0 = PredictArmor(0, center0);
//   const double center_yaw_0 = SolveYaw(center0.x, center0.y);
//   const double armor_yaw_err_0 =
//       std::fabs(AngleDiff(SolveYaw(armor0.x, armor0.y), center_yaw_0));
//   const double s_0 = armor0.x * armor0.x + armor0.y * armor0.y;

//   const double t1 = time_delay + turn_s_;
//   const TarPostion center1 = PredictCenter(t1);
//   const TarPostion armor1 = PredictArmor(1, center1);
//   const double center_yaw_1 = SolveYaw(center1.x, center1.y);
//   const double armor_yaw_err_1 =
//       std::fabs(AngleDiff(SolveYaw(armor1.x, armor1.y), center_yaw_1));
//   const double s_1 = armor1.x * armor1.x + armor1.y * armor1.y;

//   choose_next_ = (armor_yaw_err_1 <= 1.5 * armor_yaw_err_0) && (s_1 <= s_0);

//   selected_idx_ = choose_next_ ? 1 : 0;
//   selected_time_delay_ = choose_next_ ? t1 : time_delay;
//   PredictOneArmorPosition(selected_time_delay_, selected_idx_);
// }

void TrajectorySolver::LocalSelectArmor(double time_delay)
{
  if (std::fabs(target_.velocity.yaw) < 0.3)
  {
    selected_idx_ = 0;
    selected_time_delay_ = time_delay;
    choose_next_ = false;
    PredictOneArmorPosition(time_delay, selected_idx_);
    return;
  }

  const TarPostion center_curr = PredictCenter(time_delay);
  const TarPostion armor_curr0 = PredictArmor(0, center_curr);
  const TarPostion armor_curr1 = PredictArmor(1, center_curr);
  const double center_yaw_curr = SolveYaw(pre_center_.x, pre_center_.y);
  const double armor_yaw_curr0 = SolveYaw(armor_curr0.x, armor_curr0.y);
  const double armor_yaw_curr1 = SolveYaw(armor_curr1.x, armor_curr1.y);

  const double yaw_turn_delta = std::fabs(AngleDiff(armor_yaw_curr1, armor_yaw_curr0));
  const double target_yaw_speed =
      target_.velocity.yaw * (target_.radius1 + target_.radius2) / 2 - target_.velocity.y;
  const double turn_time =
      yaw_turn_delta / (std::fabs(gimbal_yaw_speed_) + std::fabs(target_yaw_speed));

  const TarPostion center_next = PredictCenter(time_delay + turn_time);
  const TarPostion armor_next_0 = PredictArmor(0, center_next);
  const TarPostion armor_next_1 = PredictArmor(1, center_next);
  const double center_yaw_next = SolveYaw(center_next.x, center_next.y);
  const double armor_yaw_next0 = SolveYaw(armor_next_0.x, armor_next_0.y);
  const double armor_yaw_next1 = SolveYaw(armor_next_1.x, armor_next_1.y);

  const double s_0 = armor_next_0.x * armor_next_0.x + armor_next_0.y * armor_next_0.y;
  const double s_1 = armor_next_1.x * armor_next_1.x + armor_next_1.y * armor_next_1.y;

  choose_next_ = (armor_yaw_next1 <= armor_yaw_next0) && (s_1 <= s_0);

  selected_idx_ = choose_next_ ? 1 : 0;
  selected_time_delay_ = choose_next_ ? time_delay + turn_time : time_delay;
  PredictOneArmorPosition(selected_time_delay_, selected_idx_);
}

// // 在一个合理的时间窗口内用二分法/牛顿法求 f(t)=0
// double TrajectorySolver::FindSwitchTime(double t_lo, double t_hi, double tol = 1e-4)
// {
//   auto f = [&](double t) -> double {
//     const auto c0 = PredictCenter(t);
//     const auto a0 = PredictArmor(0, c0);
//     double cy0 = SolveYaw(c0.x, c0.y);
//     double err0 = std::fabs(AngleDiff(SolveYaw(a0.x, a0.y), cy0));

//     const auto c1 = PredictCenter(t + turn_s_);
//     const auto a1 = PredictArmor(1, c1);
//     double cy1 = SolveYaw(c1.x, c1.y);
//     double err1 = std::fabs(AngleDiff(SolveYaw(a1.x, a1.y), cy1));

//     return err1 - err0;  // < 0 时 choose_next_ 为 true
//   };

//   // 二分法
//   double lo = t_lo, hi = t_hi;
//   while (hi - lo > tol) {
//     double mid = (lo + hi) / 2.0;
//     if (f(mid) > 0.0)
//       lo = mid;
//     else
//       hi = mid;
//   }
//   return (lo + hi) / 2.0;
// }

void TrajectorySolver::PreSelectArmor(double time_delay)
{
  const int current_idx = selected_idx_;
  const int next_idx = (current_idx + 1) % target_.num;

  const double pre_time_delay = time_delay + 2.0 * bias_time_;

  const TarPostion center0 = PredictCenter(pre_time_delay);
  const TarPostion armor0 = PredictArmor(current_idx, center0);
  const double center_yaw_0 = SolveYaw(center0.x, center0.y);
  const double armor_yaw_err_0 =
      std::fabs(AngleDiff(SolveYaw(armor0.x, armor0.y), center_yaw_0));
  const double s_0 = armor0.x * armor0.x + armor0.y * armor0.y;

  const TarPostion center1 = PredictCenter(pre_time_delay + turn_s_);
  const TarPostion armor1 = PredictArmor(next_idx, center1);
  const double center_yaw_1 = SolveYaw(center1.x, center1.y);
  const double armor_yaw_err_1 =
      std::fabs(AngleDiff(SolveYaw(armor1.x, armor1.y), center_yaw_1));
  const double s_1 = armor1.x * armor1.x + armor1.y * armor1.y;

  const bool pre_turn = (armor_yaw_err_1 <= armor_yaw_err_0) && (s_1 <= s_0);

  should_last_shot_ = !(!choose_next_ && pre_turn);
}

void TrajectorySolver::AutoSelectArmor(double time_delay, bool is_pre_select)
{
  if (selected_idx_ == LOST)
  {
    GlobalSelectArmor(time_delay);
  }
  else
  {
    LocalSelectArmor(time_delay);
  }

  if (is_pre_select)
  {
    PreSelectArmor(time_delay);
  }
  else
  {
    should_last_shot_ = true;
  }
}

void TrajectorySolver::UpdateFireLogicMode()
{
  if (choose_next_ && !last_choose_next_)
  {
    start_turn_ = std::chrono::steady_clock::now();
  }
  else if (!choose_next_ && last_choose_next_)
  {
    end_turn_ = std::chrono::steady_clock::now();
  }

  const bool has_complete_cycle =
      (end_turn_ != time_point::min() && start_turn_ != time_point::min() &&
       last_start_turn_ != time_point::min());

  if (has_complete_cycle)
  {
    turn_s_ = std::chrono::duration<double>(end_turn_ - start_turn_).count();
    step_s_ = std::chrono::duration<double>(start_turn_ - last_start_turn_).count();

    if (step_s_ > 1e-6)
    {
      const double ratio = turn_s_ / step_s_;

      if (fire_logic_mode_ == FireLogicMode::COMMON)
      {
        if (ratio >= 0.99)
        {
          fire_logic_mode_ = FireLogicMode::SPIN_TEMP;
        }
      }
      else if (fire_logic_mode_ == FireLogicMode::SPIN_TEMP)
      {
        if (ratio < 0.99 - 0.05)
        {
          fire_logic_mode_ = FireLogicMode::COMMON;
        }
        else if (ratio >= 0.99)
        {
          fire_logic_mode_ = FireLogicMode::SPIN;
        }
      }
      else if (fire_logic_mode_ == FireLogicMode::SPIN)
      {
        if (ratio < 0.99 - 0.05)
        {
          fire_logic_mode_ = FireLogicMode::COMMON;
        }
      }
    }

    last_start_turn_ = start_turn_;
    start_turn_ = time_point::min();
    end_turn_ = time_point::min();
  }
  else if (choose_next_ && !last_choose_next_)
  {
    last_start_turn_ = start_turn_;
  }

  if (HasValidSelection())
  {
    last_selected_idx_for_turn_ = selected_idx_;
  }
}

void TrajectorySolver::UpdateSolveState(double& pitch, double& yaw, bool& is_fire,
                                        double& aim_x, double& aim_y, double& aim_z,
                                        int& idx)
{
  idx = selected_idx_;

  if (!HasValidSelection())
  {
    aim_x = 0.0;
    aim_y = 0.0;
    aim_z = 0.0;
    pitch = last_pitch_;
    yaw = last_yaw_;
    is_fire = false;
    idx = LOST;
    return;
  }

  aim_x = pre_position_[selected_idx_].x;
  aim_y = pre_position_[selected_idx_].y;
  aim_z = pre_position_[selected_idx_].z;

  pitch = SolvePitch(aim_x, aim_y, aim_z);

  if (fire_logic_mode_ == FireLogicMode::SPIN)
  {
    yaw = SolveYaw(pre_center_.x, pre_center_.y);

    const double aim_yaw = SolveYaw(aim_x, aim_y);
    is_fire = std::fabs(AngleDiff(aim_yaw, yaw)) > 0.01 && CanFire(gimbal_yaw_, pitch, false);
    if (is_fire)
    {
      yaw = aim_yaw;
    }
  }
  else
  {
    yaw = SolveYaw(aim_x, aim_y);
    is_fire = CanFire(yaw, pitch, false);
  }

  last_pitch_ = pitch;
  last_yaw_ = yaw;
  last_x_v_ = target_.velocity.x;
  last_y_v_ = target_.velocity.y;
  last_v_yaw_ = target_.velocity.yaw;
  last_choose_next_ = choose_next_;
}

void TrajectorySolver::AutoSolveTrajectory(double& pitch, double& yaw, bool& is_fire,
                                           double& aim_x, double& aim_y, double& aim_z,
                                           int& idx, const Target& target,
                                           double gimbal_yaw, double gimbal_pitch,
                                           const double send_time, double gimbal_yaw_speed)
{
  target_ = target;
  gimbal_yaw_ = gimbal_yaw;
  gimbal_pitch_ = gimbal_pitch;
  gimbal_yaw_speed_ = gimbal_yaw_speed;

  fire_logic_mode_ = FireLogicMode::COMMON;

  double time_delay = fly_time_ + bias_time_ + send_time;
  AutoSelectArmor(time_delay);
  UpdateSolveState(pitch, yaw, is_fire, aim_x, aim_y, aim_z, idx);

  // 若希望每块装甲板只打一发
  // if(no_fire == true)
  // {
  //   is_fire = false;
  // }
  // if(is_fire == true)
  // {
  //   no_fire = true;
  // }
  // if(choose_next_ == false && last_choose_next_ == true)
  // {
  //   no_fire = false;
  // }
}

}  // namespace rm_auto_aim
// 没有LOST，预瞄考虑装甲板的位置变化，使用这一时刻与下一时刻的yaw变换计算