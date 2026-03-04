#include "armor_tracker/SolveTrajectory.hpp"

#include <cmath>

namespace rm_auto_aim
{

SolveTrajectory::SolveTrajectory(const double& k, const double& bias_time,
                                 const double& s_bias, const double& z_bias,
                                 const double& pitch_bias, CalculateMode calculate_mode,
                                 const TrajectoryTable::TableConfig& table_config)
    : table_(std::make_unique<TrajectoryTable>(table_config)),
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
    if (table_->IsInit())
    {
      RCLCPP_INFO(logger_, "Trajectory table initialized successfully");
    }
    else
    {
      calculate_mode_ = CalculateMode::NORMAL;
      RCLCPP_WARN(logger_, "Using normal calculation mode");
    }
  }
}

void SolveTrajectory::Init(
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

void SolveTrajectory::ReBuild()
{
  selected_idx_ = SpecialArmor::LOST;
  last_x_v_ = 0.0f;
  last_y_v_ = 0.0f;
  last_v_yaw_ = 0.0f;
}

SolveTrajectory::TargetPostion SolveTrajectory::PredictCenter(
    const auto_aim_interfaces::msg::Target::SharedPtr& msg, double time_delay)
{
  TargetPostion center;
  if (msg->armors_num == 4)
  {
    center.x = msg->position.x + msg->velocity.x * time_delay;
    center.y = msg->position.y + msg->velocity.y * time_delay;
    center.z = msg->position.z;
    center.yaw = msg->yaw + msg->v_yaw * time_delay;
  }
  else
  {
    center.x = msg->position.x;
    center.y = msg->position.y;
    center.z = msg->position.z;
    center.yaw = msg->yaw + msg->v_yaw * time_delay;
  }
  return center;
}

SolveTrajectory::TargetPostion SolveTrajectory::PredictArmor(
    const auto_aim_interfaces::msg::Target::SharedPtr& msg, double time_delay, int idx,
    SolveTrajectory::TargetPostion& pre_center)
{
  TargetPostion pre_pos;

  if (msg->armors_num == 4)
  {
    double sign = msg->v_yaw > 0 ? 1.0f : -1.0f;

    double radius = idx % 2 ? msg->radius_2 : msg->radius_1;

    double tmp_yaw = pre_center.yaw - sign * idx * 2.0f * M_PI / msg->armors_num;

    pre_pos.x = pre_center.x - radius * std::cos(tmp_yaw);
    pre_pos.y = pre_center.y - radius * std::sin(tmp_yaw);
    pre_pos.z = pre_center.z;
    pre_pos.yaw = std::fmod(tmp_yaw + M_PI, 2.0f * M_PI) - M_PI;
  }
  else  // 3个装甲板,是前哨站
  {
    double radius = msg->radius_1;
    double tmp_yaw = pre_center.yaw - idx * 2.0f * M_PI / msg->armors_num;

    pre_pos.x = pre_center.x - radius * std::cos(tmp_yaw);
    pre_pos.y = pre_center.y - radius * std::sin(tmp_yaw);

    int id = (idx + Tracker::outpost_idx) % msg->armors_num;
    pre_pos.z = pre_center.z + Tracker::outpost_dz * (id - 1);

    pre_pos.yaw = std::fmod(tmp_yaw + M_PI, 2.0f * M_PI) - M_PI;
  }

  return pre_pos;
}

// 从图片时间到打到的时间：自瞄处理的时间+电控延迟(从视觉发信号到电机动和发弹延迟)+云台转动时间+飞行时间
// msg消息的频率即我们发送开火指令的频率，这可以作为我们的步长时间
void SolveTrajectory::PredictAllArmorPosition(
    const auto_aim_interfaces::msg::Target::SharedPtr& msg, double time_delay)
{
  TargetPostion pre_center = PredictCenter(msg, time_delay);
  for (int i = 0; i < msg->armors_num; i++)
  {
    pre_position_[i] = PredictArmor(msg, time_delay, i, pre_center);
  }
}

SolveTrajectory::TargetPostion SolveTrajectory::PredictOneArmorPosition(
    const auto_aim_interfaces::msg::Target::SharedPtr& msg, double time_delay, int idx,
    bool flag)
{
  TargetPostion pre_center = PredictCenter(msg, time_delay);
  if (flag)
  {
    pre_center_ = pre_center;
  }
  return PredictArmor(msg, time_delay, idx, pre_center);
}

// 计算简化单向空气阻力模型下的弹道高度，用于正常模式
double SolveTrajectory::MonoDirectionalAirResistanceModel(double s, double angle,
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
double SolveTrajectory::SolvePitch(double x, double y, double z)
{
  // 计算水平距离
  double distance = std::sqrt(x * x + y * y);
  double target_s = distance + s_bias_;
  double target_z = z + z_bias_;

  double pitch = 0.0f;

  if (calculate_mode_ == CalculateMode::TABLE_LOOKUP && table_->IsInit())
  {
    // 查表法
    auto res = table_->Check(target_s, target_z);
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

double SolveTrajectory::SolveYaw(double x, double y) { return std::atan2(y, x); }

double fast_atan(double x, double y)
{
  double x_y = y / x;
  double x_y_2 = x_y * x_y;
  return x_y * (0.99997726f + x_y_2 * (-0.33262347f + x_y_2 * 0.19354346f));
}

// 快速打击符号fast_fire为false时，只打云台和跟踪都就位的装甲板
bool SolveTrajectory::CanFire(double tar_yaw,
                              const auto_aim_interfaces::msg::Target::SharedPtr& msg,
                              bool is_fast_fire = false)
{
  double distance =
      std::sqrt(pre_position_[selected_idx_].x * pre_position_[selected_idx_].x +
                pre_position_[selected_idx_].y * pre_position_[selected_idx_].y) +
      s_bias_;
  double armor_half_length = msg->type == "small" ? SMALL_HALF_LENGTH : LARGE_HALF_LENGTH;
  double max_yaw_diff = SolveYaw(distance, armor_half_length);

  if (!(fabs(msg->velocity.x - last_x_v_) < 0.4f &&
        fabs(msg->velocity.y - last_y_v_) < 0.3f &&
        fabs(msg->v_yaw - last_v_yaw_) < 0.3f) &&
      !is_fast_fire && !should_last_shot_)
  {
    return false;
  }
  else
  {
    bool yaw_diff_exceeds = fabs(tar_yaw - msg->gimbal_yaw) > max_yaw_diff;
    if (is_turn_)
    {
      if (yaw_diff_exceeds)
      {
        RCLCPP_WARN(logger_, "云台和跟踪都未就位");
        return false;
      }
      RCLCPP_WARN(logger_, "云台就位而跟踪未就位");
      return is_fast_fire;
    }
    else
    {
      if (yaw_diff_exceeds)
      {
        RCLCPP_WARN(logger_, "跟踪就位而云台未就位");
        return is_fast_fire;
      }
      RCLCPP_DEBUG(logger_, "云台和跟踪都就位");
      return true;
    }
  }
}

void SolveTrajectory::GlobalSelectArmor(
    const auto_aim_interfaces::msg::Target::SharedPtr& msg)
{
  float min_aim_yaw = M_PI;
  int selected_idx;
  PredictAllArmorPosition(msg, bias_time_ + fly_time_);
  for (int i = 0; i < msg->armors_num; i++)
  {
    float toyaw =
        fabs(SolveYaw(pre_position_[i].x, pre_position_[i].y) - msg->gimbal_yaw);
    float turn_time = 0.05f * toyaw;
    SolveTrajectory::TargetPostion pre_position =
        PredictOneArmorPosition(msg, turn_time + bias_time_ + fly_time_, i, true);
    float aim_yaw = SolveYaw(pre_position.x, pre_position.y);
    if (aim_yaw < min_aim_yaw)
    {
      min_aim_yaw = aim_yaw;
      selected_idx = i;
    }
  }
  RCLCPP_ERROR(logger_, "Global Select idx: %d", selected_idx);
  selected_idx_ = selected_idx;
}

void SolveTrajectory::LocalSelectArmor(
    const auto_aim_interfaces::msg::Target::SharedPtr& msg)
{
  SolveTrajectory::TargetPostion pre_position_0 =
      PredictOneArmorPosition(msg, bias_time_ + fly_time_, 0, true);
  double center_yaw_0 = SolveYaw(pre_position_0.x, pre_position_0.y);
  double s_0 =
      pre_position_0.x * pre_position_0.x + pre_position_0.y * pre_position_0.y;

  SolveTrajectory::TargetPostion pre_position_1 =
      PredictOneArmorPosition(msg, turn_s_ + bias_time_ + fly_time_, 1, true);
  double center_yaw_1 = SolveYaw(pre_position_1.x, pre_position_1.y);
  double s_1 =
      pre_position_1.x * pre_position_1.x + pre_position_1.y * pre_position_1.y;

  selected_idx_ =
      fabs(SolveYaw(pre_position_1.x, pre_position_1.y) - center_yaw_1) <=
                  fabs(SolveYaw(pre_position_0.x, pre_position_0.y) - center_yaw_0) &&
              s_1 <= s_0
          ? 1
          : 0;
}

void SolveTrajectory::PreSelectArmor(
    const auto_aim_interfaces::msg::Target::SharedPtr& msg)
{
  double time_delay = bias_time_ + fly_time_;
  SolveTrajectory::TargetPostion pre_position_0 =
      PredictOneArmorPosition(msg, time_delay, 0, true);
  double center_yaw_0 = SolveYaw(pre_position_0.x, pre_position_0.y);
  double s_0 =
      pre_position_0.x * pre_position_0.x + pre_position_0.y * pre_position_0.y;

  SolveTrajectory::TargetPostion pre_position_1 =
      PredictOneArmorPosition(msg, time_delay + turn_s_, 1, true);
  double center_yaw_1 = SolveYaw(pre_position_1.x, pre_position_1.y);
  double s_1 =
      pre_position_1.x * pre_position_1.x + pre_position_1.y * pre_position_1.y;

  bool pre_turn =
      fabs(SolveYaw(pre_position_1.x, pre_position_1.y) - center_yaw_1) <=
                  fabs(SolveYaw(pre_position_0.x, pre_position_0.y) - center_yaw_0) &&
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

void SolveTrajectory::AutoSelectArmor(
    const auto_aim_interfaces::msg::Target::SharedPtr& msg, bool is_pre_select = false)
{
  if (selected_idx_ == LOST)
  {
    GlobalSelectArmor(msg);
  }
  else
  {
    LocalSelectArmor(msg);
  }

  if (is_pre_select)
  {
    PreSelectArmor(msg);
  }
  else
  {
    should_last_shot_ = true;
  }
}

void SolveTrajectory::UpdateFireLogicMode()
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

void SolveTrajectory::UpdateSolveState(
    double& pitch, double& yaw, bool& is_fire, double& aim_x, double& aim_y,
    double& aim_z, int& idx, const auto_aim_interfaces::msg::Target::SharedPtr& msg)
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
    is_fire = CanFire(yaw, msg);
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
    is_fire = fabs(aim_yaw - yaw) < 0.02f && is_turn_;
    // RCLCPP_ERROR(logger_, "aim_yaw: %f, yaw: %f, diff: %f", aim_yaw, yaw,
    //              fabs(aim_yaw - yaw));
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
    is_fire = CanFire(yaw, msg);
  }

  if (selected_idx_ != LOST || is_fire)
  {
    last_yaw_ = yaw;
  }
  last_x_v_ = msg->velocity.x;
  last_y_v_ = msg->velocity.y;
  last_v_yaw_ = msg->v_yaw;
}

void SolveTrajectory::AutoSolveTrajectory(
    double& pitch, double& yaw, bool& is_fire, double& aim_x, double& aim_y,
    double& aim_z, int& idx, const auto_aim_interfaces::msg::Target::SharedPtr msg)
{
  auto start = std::chrono::high_resolution_clock::now();

  if (!msg)
  {
    RCLCPP_ERROR(logger_, "Invalid target message");
    return;
  }

  // fire_logic_mode_ = FireLogicMode::SPIN;
  AutoSelectArmor(msg);
  UpdateSolveState(pitch, yaw, is_fire, aim_x, aim_y, aim_z, idx, msg);
  if (fabs(msg->v_yaw) > 0.5f)
  {
    UpdateFireLogicMode();
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  RCLCPP_DEBUG(logger_, "Trajectory solve time: %ld us", duration.count());
}

bool SolveTrajectory::ReloadTable(const std::string& new_filename)
{
  if (new_filename.empty())
  {
    RCLCPP_WARN(logger_, "ReloadTable called with empty filename, skipping");
    return false;
  }

  RCLCPP_INFO(logger_, "Reloading trajectory table: %s", new_filename.c_str());

  if (!table_->ReloadTable(new_filename))
  {
    RCLCPP_ERROR(logger_, "Failed to reload table: %s, falling back to NORMAL mode",
                 new_filename.c_str());
    calculate_mode_ = CalculateMode::NORMAL;
    return false;
  }

  calculate_mode_ = CalculateMode::TABLE_LOOKUP;
  RCLCPP_INFO(logger_, "Trajectory table reloaded successfully");
  return true;
}

}  // namespace rm_auto_aim
// 没有LOST，预瞄考虑装甲板的位置变化，使用这一时刻与下一时刻的yaw变换计算