#include "armor_tracker/SolveTrajectory.hpp"

namespace rm_auto_aim
{

SolveTrajectory::SolveTrajectory(const float& k, const float& bias_time,
                                 const float& s_bias, const float& z_bias,
                                 const float& pitch_bias, CalculateMode calculate_mode,
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
  last_yaw_ = 0.0f;
}

// 从图片时间到打到的时间：自瞄处理的时间+电控延迟(从视觉发信号到电机动和发弹延迟)+云台转动时间+飞行时间
// msg消息的频率即我们发送开火指令的频率，这可以作为我们的步长时间
void SolveTrajectory::PredictAllArmorPosition(
    const auto_aim_interfaces::msg::Target::SharedPtr& msg, float time_delay)
{
  pre_x_center_ = msg->position.x + msg->velocity.x * time_delay;
  pre_y_center_ = msg->position.y + msg->velocity.y * time_delay;
  pre_z_center_ = msg->position.z;
  pre_yaw_ = msg->yaw + msg->v_yaw * time_delay;

  if (msg->v_yaw > 0)
  {
    for (int i = 0; i < msg->armors_num; i++)
    {
      float radius = i % 2 ? msg->radius_2 : msg->radius_1;
      // float radius = msg->radius_1;

      float tmp_yaw = pre_yaw_ - i * 2.0f * M_PI / msg->armors_num;

      pre_position_[i].x = pre_x_center_ - radius * std::cos(tmp_yaw);
      pre_position_[i].y = pre_y_center_ - radius * std::sin(tmp_yaw);
      if (msg->armors_num == 4)
      {
        pre_position_[i].z = msg->position.z;
      }
      else if (msg->armors_num == 3)
      {
        if (Tracker::outpost_idx == 0)
        {
          if (i == 0)
          {
            pre_position_[i].z = msg->position.z - Tracker::outpost_dz;
          }
          else if (i == 1)
          {
            pre_position_[i].z = msg->position.z;
          }
          else
          {
            pre_position_[i].z = msg->position.z + Tracker::outpost_dz;
          }
        }
        else if (Tracker::outpost_idx == 1)
        {
          if (i == 0)
          {
            pre_position_[i].z = msg->position.z;
          }
          else if (i == 1)
          {
            pre_position_[i].z = msg->position.z + Tracker::outpost_dz;
          }
          else
          {
            pre_position_[i].z = msg->position.z - Tracker::outpost_dz;
          }
        }
        else if (Tracker::outpost_idx == 2)
        {
          if (i == 0)
          {
            pre_position_[i].z = msg->position.z + Tracker::outpost_dz;
          }
          else if (i == 1)
          {
            pre_position_[i].z = msg->position.z - Tracker::outpost_dz;
          }
          else
          {
            pre_position_[i].z = msg->position.z;
          }
        }
      }
      pre_position_[i].yaw = std::fmod(tmp_yaw + M_PI, 2.0f * M_PI) - M_PI;
    }
  }
  else
  {
    for (int i = 0; i < msg->armors_num; i++)
    {
      float radius = i % 2 ? msg->radius_2 : msg->radius_1;
      // float radius = msg->radius_1;

      float tmp_yaw = pre_yaw_ + i * 2.0f * M_PI / msg->armors_num;

      pre_position_[i].x = pre_x_center_ - radius * std::cos(tmp_yaw);
      pre_position_[i].y = pre_y_center_ - radius * std::sin(tmp_yaw);
      pre_position_[i].z = msg->position.z;
      pre_position_[i].yaw = std::fmod(tmp_yaw + M_PI, 2.0f * M_PI) - M_PI;
    }
  }
}

void SolveTrajectory::PredictOneArmorPosition(
    const auto_aim_interfaces::msg::Target::SharedPtr& msg, float time_delay, int idx)
{
}

// 计算简化单向空气阻力模型下的弹道高度，用于正常模式
float SolveTrajectory::MonoDirectionalAirResistanceModel(float s, float angle, float v)
{
  float cos_angle = std::cos(angle);
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
float SolveTrajectory::SolvePitch(float x, float y, float z)
{
  // RCLCPP_WARN(logger_, "x: %.2f, y: %.2f, z: %.2f", x, y, z);
  // 计算水平距离
  float distance = std::sqrt(x * x + y * y);
  // RCLCPP_DEBUG(logger_, "Distance: %.2f， Target x: %.2f, Target y: %.2f",
  //              distance, x, y);
  float target_s = distance + s_bias_;
  float target_z = z + z_bias_;

  float pitch = 0.0f;

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
    float z_temp = target_z;

    for (int i = 0; i < 20; ++i)
    {
      if (std::isnan(z_temp))
      {
        RCLCPP_ERROR(logger_, "z_temp is NaN during iteration");
        return 0.0f;
      }

      pitch = std::atan2(z_temp, target_s);
      float z_actual = MonoDirectionalAirResistanceModel(target_s, pitch, current_v_);
      float dz = 0.3f * (target_z - z_actual);
      z_temp += dz;

      if (fabsf(dz) < 1e-5f)
      {
        RCLCPP_DEBUG(logger_, "Pitch convergence after %d iterations", i + 1);
        break;
      }
    }
  }
  pitch += pitch_bias_;
  return pitch;
}

float SolveTrajectory::SolveYaw(float x, float y) { return std::atan2(y, x); }

float fast_atan(float x, float y)
{
  float x_y = y / x;
  float x_y_2 = x_y * x_y;
  return x_y * (0.99997726f + x_y_2 * (-0.33262347f + x_y_2 * 0.19354346f));
}

// 判断是否满足开火条件,保守打击，只打真正在跟踪的装甲板
bool SolveTrajectory::CanFire(float aim_yaw,
                              const auto_aim_interfaces::msg::Target::SharedPtr& msg)
{
  if (is_turn_)
  {
    return false;
  }
  else
  {
    return true;
  }

  float cam_yaw = msg->camera_yaw + 0.05 * (aim_yaw - msg->camera_yaw);
  RCLCPP_WARN(logger_, "aim_yaw:%f,cam_yaw:%f", aim_yaw, cam_yaw);
  return fabs(msg->velocity.x - last_x_v_) < 0.4f &&
         fabs(msg->velocity.y - last_y_v_) < 0.3f &&
         fabs(msg->v_yaw - last_v_yaw_) < 0.3f &&
         fabs(aim_yaw - cam_yaw) * pre_y_center_ < 0.03;
}

// 选择最优装甲板,使得同样时间里aiming时间占比最长，且尽量连续,尽量以中心展开
void SolveTrajectory::GlobalSelectArmor(
    const auto_aim_interfaces::msg::Target::SharedPtr& msg)
{
  float max_aim_yaw = M_PI;
  int selected_idx;
  for (int i = 0; i < msg->armors_num; i++)
  {
    float toyaw =
        fabs(SolveYaw(pre_position_[i].x, pre_position_[i].y) - msg->camera_yaw);
    float turn_time = 0.05f * toyaw;
    PredictAllArmorPosition(msg, turn_time + bias_time_ + fly_time_);
    float aim_yaw = SolveYaw(pre_position_[i].x, pre_position_[i].y);
    if (aim_yaw < max_aim_yaw)
    {
      max_aim_yaw = aim_yaw;
      selected_idx = i;
    }
  }
  RCLCPP_ERROR(logger_, "GGGGGG");
  selected_idx_ = selected_idx;
}

void SolveTrajectory::LocalSelectArmor(
    const auto_aim_interfaces::msg::Target::SharedPtr& msg)
{
  float center_yaw = SolveYaw(pre_x_center_, pre_y_center_);
  float s_0 =
      pre_position_[0].x * pre_position_[0].x + pre_position_[0].y * pre_position_[0].y;
  float s_1 =
      pre_position_[1].x * pre_position_[1].x + pre_position_[1].y * pre_position_[1].y;
  is_turn_ = fabs(SolveYaw(pre_position_[1].x, pre_position_[1].y) - center_yaw) <=
          fabs(SolveYaw(pre_position_[0].x, pre_position_[0].y) - center_yaw) &&
      s_1 <= s_0;
  selected_idx_ = is_turn_ ? 1 : 0;
}

// 不择板，判断此时发弹能否打击到目标
void SolveTrajectory::FireLogicIsTop(
    float& pitch, float& yaw, bool& is_fire, float& aim_x, float& aim_y, float& aim_z,
    int& idx, const auto_aim_interfaces::msg::Target::SharedPtr& msg)
{
  fire_logic_mode_ = FireLogicMode::SPIN;
  float time_delay = bias_time_ + fly_time_;
  PredictAllArmorPosition(msg, time_delay);
  if (last_selected_idx_ == LOST)
  {
    int selected_idx = GlobalSelectArmor(msg);
    UpdateSolveState(selected_idx, pitch, yaw, is_fire, aim_x, aim_y, aim_z, idx, msg);
  }
  else
  {
    int selected_idx = LocalSelectArmor(msg);
    UpdateSolveState(selected_idx, pitch, yaw, is_fire, aim_x, aim_y, aim_z, idx, msg);
  }
}

// 择板，判断此时发弹是否有合适的目标
inline void SolveTrajectory::FireLogicDefault(
    float& pitch, float& yaw, bool& is_fire, float& aim_x, float& aim_y, float& aim_z,
    int& idx, const auto_aim_interfaces::msg::Target::SharedPtr& msg)
{
  float time_delay = bias_time_ + fly_time_;
  PredictAllArmorPosition(msg, time_delay);

  if (selected_idx_ == LOST)
  {
    GlobalSelectArmor(msg);
  }
  else
  {
    LocalSelectArmor(msg);
  }
}

void SolveTrajectory::UpdateFireLogicMode()
{
  if (selected_idx_ == 0)
  {
    if (is_turn_)
    {
      end_turn_ = std::chrono::high_resolution_clock::now();
    }
    is_turn_ = false;
  }
  if (selected_idx_ == 1)
  {
    if (!is_turn_)
    {
      start_turn_ = std::chrono::high_resolution_clock::now();
    }
    is_turn_ = true;
  }

  auto turn_duration = end_turn_ - start_turn_;
  auto one_step_duration = end_turn_ - last_end_turn_;

  if (turn_duration / one_step_duration < 0.99)
  {
    fire_logic_mode_ = FireLogicMode::COMMON;
  }
  else
  {
    fire_logic_mode_ = FireLogicMode::SPIN;
  }

  last_end_turn_ = end_turn_;
}

void SolveTrajectory::UpdateSolveState(
    int selected_idx, float& pitch, float& yaw, bool& is_fire, float& aim_x, float& aim_y,
    float& aim_z, int& idx, const auto_aim_interfaces::msg::Target::SharedPtr& msg)
{
  idx = selected_idx;
  if (fire_logic_mode_ == FireLogicMode::SPIN)
  {
    aim_x = pre_position_[0].x;
    aim_y = pre_position_[0].y;
    aim_z = pre_position_[0].z;
    pitch = SolvePitch(aim_x, aim_y, aim_z);
    yaw = SolveYaw(pre_x_center_, pre_y_center_);
    is_fire = fabs(SolveYaw(aim_x, aim_y) - yaw) < 0.01f;
    if (is_fire)
    {
      yaw = SolveYaw(aim_x, aim_y);
    }
  }
  // 理论上不会有selected_idx == LOST
  else if (selected_idx == LOST)
  {
    aim_x = pre_position_[0].x;
    aim_y = pre_position_[0].y;
    aim_z = pre_position_[0].z;
    pitch = SolvePitch(aim_x, aim_y, aim_z);
    yaw = SolveYaw(aim_x, aim_y);
    is_fire = CanFire(yaw, msg);
  }
  else
  {
    aim_x = pre_position_[selected_idx].x;
    aim_y = pre_position_[selected_idx].y;
    aim_z = pre_position_[selected_idx].z;
    pitch = SolvePitch(aim_x, aim_y, aim_z);
    yaw = SolveYaw(aim_x, aim_y);
    is_fire = CanFire(yaw, msg);
  }

  if (selected_idx != LOST || is_fire)
  {
    last_yaw_ = yaw;
    last_selected_idx_ = selected_idx;
  }
  last_x_v_ = msg->velocity.x;
  last_y_v_ = msg->velocity.y;
  last_v_yaw_ = msg->v_yaw;
}

void SolveTrajectory::AutoSolveTrajectory(
    float& pitch, float& yaw, bool& is_fire, float& aim_x, float& aim_y, float& aim_z,
    int& idx, const auto_aim_interfaces::msg::Target::SharedPtr msg)
{
  auto start = std::chrono::high_resolution_clock::now();

  if (!msg)
  {
    RCLCPP_ERROR(logger_, "Invalid target message");
    return;
  }
  FireLogicDefault(pitch, yaw, is_fire, aim_x, aim_y, aim_z, idx, msg);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  RCLCPP_DEBUG(logger_, "Trajectory solve time: %ld us", duration.count());
}
}  // namespace rm_auto_aim
// 没有LOST，预瞄考虑装甲板的位置变化，使用这一时刻与下一时刻的yaw变换计算