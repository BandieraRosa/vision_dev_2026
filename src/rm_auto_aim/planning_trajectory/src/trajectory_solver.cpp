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
  selected_idx_ = SpecialArmor::LOST;
  last_x_v_ = 0.0f;
  last_y_v_ = 0.0f;
  last_v_yaw_ = 0.0f;
}

TrajectorySolver::TarPostion TrajectorySolver::PredictCenter(double time_delay)
{
  TarPostion center;
  if (target_.num == 4)
  {
    center.x = target_.position.x + target_.velocity.x * time_delay;
    center.y = target_.position.y + target_.velocity.y * time_delay;
    center.z = target_.position.z;
    center.yaw = target_.position.yaw + target_.velocity.yaw * time_delay;
  }
  else
  {
    center.x = target_.position.x;
    center.y = target_.position.y;
    center.z = target_.position.z;
    center.yaw = target_.position.yaw + target_.velocity.yaw * time_delay;
  }
  return center;
}

TrajectorySolver::TarPostion TrajectorySolver::PredictArmor(
    double time_delay, int idx, TrajectorySolver::TarPostion& pre_center)
{
  TarPostion pre_pos;

  if (target_.num == 4)
  {
    double sign = target_.velocity.yaw > 0 ? 1.0f : -1.0f;

    double radius = idx % 2 ? target_.radius2 : target_.radius1;

    double tmp_yaw = pre_center.yaw - sign * idx * 2.0f * M_PI / target_.num;

    pre_pos.x = pre_center.x - radius * std::cos(tmp_yaw);
    pre_pos.y = pre_center.y - radius * std::sin(tmp_yaw);
    pre_pos.z = pre_center.z;
    pre_pos.yaw = std::fmod(tmp_yaw + M_PI, 2.0f * M_PI) - M_PI;
  }
  else  // 3个装甲板,是前哨站
  {
    double radius = target_.radius1;
    double sign = target_.velocity.yaw > 0 ? 1.0f : -1.0f;
    double tmp_yaw = pre_center.yaw - sign * idx * 2.0f * M_PI / target_.num;

    pre_pos.x = pre_center.x - radius * std::cos(tmp_yaw);
    pre_pos.y = pre_center.y - radius * std::sin(tmp_yaw);

    int id =
        target_.outpost_idx + (sign == 1.0f ? idx : (target_.num - idx)) % target_.num;
    pre_pos.z = pre_center.z + outpost_dz * (id - 1);

    pre_pos.yaw = std::fmod(tmp_yaw + M_PI, 2.0f * M_PI) - M_PI;
  }

  return pre_pos;
}

// 从图片时间到打到的时间：自瞄处理的时间+电控延迟(从视觉发信号到电机动和发弹延迟)+云台转动时间+飞行时间
// msg消息的频率即我们发送开火指令的频率，这可以作为我们的步长时间
void TrajectorySolver::PredictAllArmorPosition(double time_delay)
{
  TarPostion pre_center = PredictCenter(time_delay);
  for (int i = 0; i < target_.num; i++)
  {
    pre_position_[i] = PredictArmor(time_delay, i, pre_center);
  }
}

void TrajectorySolver::PredictOneArmorPosition(double time_delay, int idx)
{
  TarPostion pre_center = PredictCenter(time_delay);
  pre_position_[idx] = PredictArmor(time_delay, idx, pre_center);
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

double TrajectorySolver::SolveYaw(double x, double y) { return std::atan2(y, x); }

double fast_atan(double x, double y)
{
  double x_y = y / x;
  double x_y_2 = x_y * x_y;
  return x_y * (0.99997726f + x_y_2 * (-0.33262347f + x_y_2 * 0.19354346f));
}

// 快速打击符号fast_fire为false时，只打云台和跟踪都就位的装甲板
bool TrajectorySolver::CanFire(double tar_yaw, bool is_fast_fire = false)
{
  double distance =
      std::sqrt(pre_position_[selected_idx_].x * pre_position_[selected_idx_].x +
                pre_position_[selected_idx_].y * pre_position_[selected_idx_].y) +
      s_bias_;
  double armor_half_length =
      target_.type == "small" ? SMALL_HALF_LENGTH : LARGE_HALF_LENGTH;
  double max_yaw_diff = SolveYaw(distance, armor_half_length);

  if (!(fabs(target_.velocity.x - last_x_v_) < 0.4f &&
        fabs(target_.velocity.y - last_y_v_) < 0.3f &&
        fabs(target_.velocity.yaw - last_v_yaw_) < 0.3f) &&
      !is_fast_fire && !should_last_shot_)
  {
    return false;
  }
  else
  {
    bool yaw_diff_exceeds = fabs(tar_yaw - gimbal_yaw_) > max_yaw_diff;
    if (is_turn_)
    {
      if (yaw_diff_exceeds)
      {
        // RCLCPP_WARN(logger_, "云台和跟踪都未就位");
        return false;
      }
      // RCLCPP_WARN(logger_, "云台就位而跟踪未就位");
      return is_fast_fire;
    }
    else
    {
      if (yaw_diff_exceeds)
      {
        // RCLCPP_WARN(logger_, "跟踪就位而云台未就位");
        return is_fast_fire;
      }
      // RCLCPP_DEBUG(logger_, "云台和跟踪都就位");
      return true;
    }
  }
}

void TrajectorySolver::GlobalSelectArmor(double time_delay)
{
  double min_aim_yaw = M_PI;
  int selected_idx;
  PredictAllArmorPosition(time_delay);
  for (int i = 0; i < target_.num; i++)
  {
    double toyaw = fabs(SolveYaw(pre_position_[i].x, pre_position_[i].y) - gimbal_yaw_);
    double turn_time = 0.05f * toyaw;
    PredictOneArmorPosition(time_delay + turn_time, i);
    double aim_yaw = SolveYaw(pre_position_[i].x, pre_position_[i].y);
    if (aim_yaw < min_aim_yaw)
    {
      min_aim_yaw = aim_yaw;
      selected_idx = i;
    }
  }
  RCLCPP_ERROR(logger_, "Global Select idx: %d", selected_idx);
  selected_idx_ = selected_idx;
}

void TrajectorySolver::LocalSelectArmor(double time_delay)
{
  if (std::fabs(target_.velocity.yaw) < 0.3f)
  {
    selected_idx_ = 0;
    PredictOneArmorPosition(time_delay, 0);
    return;
  }

  PredictOneArmorPosition(time_delay, 0);
  double center_yaw_0 = SolveYaw(pre_position_[0].x, pre_position_[0].y);
  double s_0 =
      pre_position_[0].x * pre_position_[0].x + pre_position_[0].y * pre_position_[0].y;

  PredictOneArmorPosition(time_delay + turn_s_, 1);
  double center_yaw_1 = SolveYaw(pre_position_[1].x, pre_position_[1].y);
  double s_1 =
      pre_position_[1].x * pre_position_[1].x + pre_position_[1].y * pre_position_[1].y;
  selected_idx_ =
      fabs(SolveYaw(pre_position_[1].x, pre_position_[1].y) - center_yaw_1) <=
                  1.2 * fabs(SolveYaw(pre_position_[0].x, pre_position_[0].y) -
                             center_yaw_0) &&
              s_1 <= s_0
          ? 1
          : 0;
}

void TrajectorySolver::PreSelectArmor(double time_delay)
{
  double pre_time_delay = time_delay + 2* bias_time_;
  PredictOneArmorPosition(pre_time_delay, 0);
  double center_yaw_0 = SolveYaw(pre_position_[0].x, pre_position_[0].y);
  double s_0 =
      pre_position_[0].x * pre_position_[0].x + pre_position_[0].y * pre_position_[0].y;

  PredictOneArmorPosition(pre_time_delay + turn_s_, 1);
  double center_yaw_1 = SolveYaw(pre_position_[1].x, pre_position_[1].y);
  double s_1 =
      pre_position_[1].x * pre_position_[1].x + pre_position_[1].y * pre_position_[1].y;

  bool pre_turn =
      fabs(SolveYaw(pre_position_[1].x, pre_position_[1].y) - center_yaw_1) <=
                  fabs(SolveYaw(pre_position_[0].x, pre_position_[0].y) - center_yaw_0) &&
              s_1 <= s_0
          ? 1
          : 0;
  if (!is_turn_ && pre_turn)
  {
    should_last_shot_ = false;
  }
  else
  {
    should_last_shot_ = true;
  }
}

void TrajectorySolver::AutoSelectArmor(double time_delay, bool is_pre_select = false)
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
  bool last_is_turn = is_turn_;

  if (selected_idx_ == 0)
  {
    if (is_turn_)
    {
      end_turn_ = std::chrono::high_resolution_clock::now();
    }
    is_turn_ = false;
  }
  else if (selected_idx_ == 1)
  {
    if (!is_turn_)
    {
      start_turn_ = std::chrono::high_resolution_clock::now();
    }
    is_turn_ = true;
  }

  bool has_complete_cycle =
      (end_turn_ != std::chrono::high_resolution_clock::time_point::min() &&
       start_turn_ != std::chrono::high_resolution_clock::time_point::min() &&
       last_start_turn_ != std::chrono::high_resolution_clock::time_point::min());

  if (has_complete_cycle)
  {
    turn_s_ = std::chrono::duration<double, std::milli>(end_turn_ - start_turn_).count();
    step_s_ =
        std::chrono::duration<double, std::milli>(start_turn_ - last_start_turn_).count();
    if (step_s_ > 0.0)
    {
      double ratio = turn_s_ / step_s_;

      // 随便给的阈值
      static constexpr double SPIN_THRESHOLD = 0.99;
      static constexpr double HYSTERESIS = 0.05;

      if (fire_logic_mode_ == FireLogicMode::COMMON)
      {
        if (ratio >= SPIN_THRESHOLD)
        {
          fire_logic_mode_ = FireLogicMode::SPIN_TEMP;
          RCLCPP_INFO(logger_, "进入SPIN_TEMP模式, ratio: %.3f", ratio);
        }
      }
      else if (fire_logic_mode_ == FireLogicMode::SPIN_TEMP)
      {
        if (ratio < SPIN_THRESHOLD - HYSTERESIS)
        {
          fire_logic_mode_ = FireLogicMode::COMMON;
          RCLCPP_INFO(logger_, "返回COMMON模式, ratio: %.3f", ratio);
        }
        else if (ratio >= SPIN_THRESHOLD)
        {
          fire_logic_mode_ = FireLogicMode::SPIN;
          RCLCPP_INFO(logger_, "稳定进入SPIN模式, ratio: %.3f", ratio);
        }
      }

      else if (fire_logic_mode_ == FireLogicMode::SPIN)
      {
        if (ratio < SPIN_THRESHOLD - HYSTERESIS)
        {
          fire_logic_mode_ = FireLogicMode::COMMON;
          RCLCPP_INFO(logger_, "退出SPIN模式, ratio: %.3f", ratio);
        }
      }
    }

    last_start_turn_ = start_turn_;
    end_turn_ = std::chrono::high_resolution_clock::time_point::min();
    start_turn_ = std::chrono::high_resolution_clock::time_point::min();
  }
  else if (is_turn_ && !last_is_turn)
  {
    last_start_turn_ = start_turn_;
  }
}

void TrajectorySolver::UpdateSolveState(double& pitch, double& yaw, bool& is_fire,
                                        double& aim_x, double& aim_y, double& aim_z,
                                        int& idx)
{
  idx = selected_idx_;

  // 理论上没有LOST
  if (selected_idx_ == LOST)
  {
    aim_x = pre_position_[0].x;
    aim_y = pre_position_[0].y;
    aim_z = pre_position_[0].z;
    pitch = SolvePitch(aim_x, aim_y, aim_z);
    yaw = SolveYaw(aim_x, aim_y);
    is_fire = CanFire(yaw, false);
  }

  // 英雄打击前哨站和步兵打击高速旋转
  else if (fire_logic_mode_ == FireLogicMode::SPIN)
  {
    aim_x = pre_position_[selected_idx_].x;
    aim_y = pre_position_[selected_idx_].y;
    aim_z = pre_position_[selected_idx_].z;
    pitch = SolvePitch(aim_x, aim_y, aim_z);
    yaw = SolveYaw(pre_center_.x, pre_center_.y);

    double aim_yaw = SolveYaw(aim_x, aim_y);
    is_fire = std::fabs(aim_yaw - yaw) > 0.02f && !is_turn_;
    if (is_fire)
    {
      yaw = aim_yaw;
    }
  }

  else  // COMMON和SPIN_TEMP模式
  {
    aim_x = pre_position_[selected_idx_].x;
    aim_y = pre_position_[selected_idx_].y;
    aim_z = pre_position_[selected_idx_].z;
    pitch = SolvePitch(aim_x, aim_y, aim_z);
    yaw = SolveYaw(aim_x, aim_y);
    is_fire = CanFire(yaw, false);
  }

  if (selected_idx_ != LOST || is_fire)
  {
    last_yaw_ = yaw;
  }
  last_x_v_ = target_.velocity.x;
  last_y_v_ = target_.velocity.y;
  last_v_yaw_ = target_.velocity.yaw;
}

void TrajectorySolver::AutoSolveTrajectory(double& pitch, double& yaw, bool& is_fire,
                                           double& aim_x, double& aim_y, double& aim_z,
                                           int& idx, const Target& target,
                                           double gimbal_yaw, const double send_time)
{
  target_ = target;
  gimbal_yaw_ = gimbal_yaw;
  auto start = std::chrono::high_resolution_clock::now();

  fire_logic_mode_ = FireLogicMode::COMMON;

  double time_delay = fly_time_+bias_time_+send_time;
  AutoSelectArmor(time_delay);
  UpdateSolveState(pitch, yaw, is_fire, aim_x, aim_y, aim_z, idx);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  RCLCPP_DEBUG(logger_, "Trajectory solve time: %ld us", duration.count());
}

}  // namespace rm_auto_aim
// 没有LOST，预瞄考虑装甲板的位置变化，使用这一时刻与下一时刻的yaw变换计算