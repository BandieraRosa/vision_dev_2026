#include "armor_tracker/SolveTrajectory.hpp"

#include <cmath>

namespace rm_auto_aim {

SolveTrajectory::SolveTrajectory(
    const float &k, const int &bias_time, const float &s_bias,
    const float &z_bias, const float &pitch_bias, CalculateMode calculate_mode,
    const TrajectoryTable::TableConfig &table_config)
    : table_(std::make_unique<TrajectoryTable>(table_config)),
      calculate_mode_(calculate_mode), k_(k), pitch_bias_(pitch_bias),
      bias_time_(bias_time), s_bias_(s_bias), z_bias_(z_bias) {
  if (calculate_mode_ == CalculateMode::TABLE_LOOKUP) {
    table_->Init();
    if (table_->IsInit()) {
      RCLCPP_INFO(logger_, "Trajectory table initialized successfully");
    } else {
      calculate_mode_ = CalculateMode::NORMAL;
      RCLCPP_WARN(logger_, "Using normal calculation mode");
    }
  }
}

void SolveTrajectory::Init(
    const auto_aim_interfaces::msg::Velocity::SharedPtr velocity_msg) {
  if (!std::isnan(velocity_msg->velocity)) {
    current_v_ = velocity_msg->velocity;
    RCLCPP_DEBUG(logger_, "Velocity updated: %.2f m/s", current_v_);
  } else {
    RCLCPP_WARN(logger_, "Invalid velocity, using default: 12.0 m/s");
    current_v_ = 12.0f;
  }
}

void SolveTrajectory::ReBuild() { last_selected_idx_ = SpecialArmor::LOST; }

// 整车建模，计算各装甲板位置
void SolveTrajectory::CalculateArmorPosition(
    const auto_aim_interfaces::msg::Target::SharedPtr &msg) {
  for (int i = 0; i < msg->armors_num; i++) {
    float radius = static_cast<float>(i % 2 ? msg->radius_2 : msg->radius_1);

    float tmp_yaw = static_cast<float>(msg->yaw + static_cast<float>(i) * 2.0f *
                                                      M_PI / msg->armors_num);

    tar_position_[i].x =
        static_cast<float>(msg->position.x - radius * std::cos(tmp_yaw));
    tar_position_[i].y =
        static_cast<float>(msg->position.y - radius * std::sin(tmp_yaw));
    tar_position_[i].z = static_cast<float>(msg->position.z);
    tar_position_[i].yaw = static_cast<float>(tmp_yaw);
  }
}

// 从图片时间到打到的时间：自瞄处理的时间+电控延迟(从视觉发信号到电机动和发弹延迟)+云台转动时间+飞行时间
// msg消息的频率即我们发送开火指令的频率，这可以作为我们的步长时间
void SolveTrajectory::PredictArmorPosition(
    const auto_aim_interfaces::msg::Target::SharedPtr &msg, float time_delay) {
  pre_x_center_ = msg->position.x + msg->velocity.x * time_delay;
  pre_y_center_ = msg->position.y + msg->velocity.y * time_delay;
  pre_z_center_ = msg->position.z;
  pre_yaw_ = msg->yaw + msg->v_yaw * time_delay;

  for (int i = 0; i < msg->armors_num; i++) {
    float radius = i % 2 ? msg->radius_2 : msg->radius_1;

    float tmp_yaw = pre_yaw_ + i * 2.0f * M_PI / msg->armors_num;

    pre_position_[i].x = pre_x_center_ - radius * cos(tmp_yaw);
    pre_position_[i].y = pre_y_center_ - radius * sin(tmp_yaw);
    pre_position_[i].z = msg->position.z;
    pre_position_[i].yaw = std::fmod(tmp_yaw + M_PI, 2.0f * M_PI) - M_PI;
  }
}

// 计算简化单向空气阻力模型下的弹道高度，用于正常模式
float SolveTrajectory::MonoDirectionalAirResistanceModel(float s, float angle,
                                                         float v) {
  float cos_angle = cos(angle);
  if (cos_angle <= 0) {
    RCLCPP_WARN(logger_, "Invalid angle: cos(angle) <= 0");
    fly_time_ = 0;
    return 0;
  }

  fly_time_ = (exp(k_ * s) - 1) / (k_ * v * cos_angle);

  if (fly_time_ < 0) {
    RCLCPP_WARN(logger_, "Exceeding maximum range! s: %.2f, v: %.2f", s, v);
    fly_time_ = 0;
    return 0;
  }

  return v * sin(angle) * fly_time_ - GRAVITY * fly_time_ * fly_time_ / 2;
}

// 计算俯仰角(两种模式)
float SolveTrajectory::SolvePitch(float x, float y, float z) {
  // 计算水平距离
  float distance = sqrt(x * x + y * y);
  float target_s = distance + s_bias_;
  float target_z = z + z_bias_;

  float pitch = 0.0f;

  if (calculate_mode_ == CalculateMode::TABLE_LOOKUP && table_->IsInit()) {
    // 查表法
    auto res = table_->Check(target_s, target_z);
    fly_time_ = res.t;
    pitch = static_cast<float>(res.pitch) + pitch_bias_;
    RCLCPP_DEBUG(logger_, "Table lookup - s: %.2f, z: %.2f, pitch: %.4f",
                 target_s, target_z, pitch);
  } else {
    // 正常模式下的迭代计算
    float z_temp = target_z;

    for (int i = 0; i < 20; ++i) {
      if (std::isnan(z_temp)) {
        RCLCPP_ERROR(logger_, "z_temp is NaN during iteration");
        return 0.0f;
      }

      pitch = std::atan2(z_temp, target_s);
      float z_actual =
          MonoDirectionalAirResistanceModel(target_s, pitch, current_v_);
      float dz = 0.3f * (target_z - z_actual);
      z_temp += dz;

      if (fabsf(dz) < 1e-5f) {
        RCLCPP_DEBUG(logger_, "Pitch convergence after %d iterations", i + 1);
        break;
      }
    }
    pitch += pitch_bias_;
  }
  return pitch;
}

float SolveTrajectory::SolveYaw(float x, float y) { return std::atan2(y, x); }

float fast_atan(float x, float y) {
  float x_y = y / x;
  float x_y_2 = x_y * x_y;
  return x_y * (0.99997726f + x_y_2 * (-0.33262347f + x_y_2 * 0.19354346f));
}

// 判断是否满足开火条件,保守打击，只打真正在跟踪的装甲板
bool SolveTrajectory::CanFire(
    float aim_yaw, float max_yaw_diff,
    const auto_aim_interfaces::msg::Target::SharedPtr &msg) {
  RCLCPP_DEBUG(logger_,
               "x_v_diff=%.3f, y_v_diff=%.3f,yaw_diff=%.3f,cam_to_x=%.3f",
               msg->velocity.x - last_x_v_, msg->velocity.y - last_y_v_,
               aim_yaw - msg->camera_yaw, msg->armor_x);
  return fabs(msg->velocity.x - last_x_v_) < 0.1f &&
         fabs(msg->velocity.y - last_y_v_) < 0.1f &&
         std::abs(aim_yaw - msg->camera_yaw) < max_yaw_diff &&
         std::abs(msg->armor_x) < 0.07f;
}

// 选择最优装甲板,使得同样时间里aiming时间占比最长，且尽量连续,尽量以中心展开
int SolveTrajectory::SelectArmor(
    const auto_aim_interfaces::msg::Target::SharedPtr &msg) {
  int selected_idx = -1;
  // 当无可开火装甲板时，选择到下一装甲板出现的位置预瞄
  float min_yaw = std::numeric_limits<float>::max();
  for (int i = 0; i < msg->armors_num; i++) {
    float aim_yaw = fabsf(pre_position_[i].yaw -
                          SolveYaw(pre_position_[i].x, pre_position_[i].y));
    if (aim_yaw < min_yaw) {
      min_yaw = aim_yaw;
      selected_idx = i;
    }
  }
  return selected_idx;
}

// 不择板，判断此时发弹能否打击到目标
void SolveTrajectory::FireLogicIsTop(
    float &pitch, float &yaw, bool &is_fire, float &aim_x, float &aim_y,
    float &aim_z, const auto_aim_interfaces::msg::Target::SharedPtr &msg) {
  float time_delay = bias_time_ + fly_time_;
  PredictArmorPosition(msg, time_delay);
  aim_x = pre_x_center_;
  aim_y = pre_y_center_;
  aim_z = pre_z_center_;
  if (last_selected_idx_ == LOST) {
    float toyaw = SolveYaw(pre_x_center_, pre_y_center_) - msg->camera_yaw;
    PredictArmorPosition(msg, time_delay +
                                  toyaw / (0.58f + msg->v_yaw)); ////////////
  }
  UpdateSolveState(CENTER, pitch, yaw, is_fire, aim_x, aim_y, aim_z, msg);
}

// 择板，判断此时发弹是否有合适的目标
void SolveTrajectory::FireLogicDefault(
    float &pitch, float &yaw, bool &is_fire, float &aim_x, float &aim_y,
    float &aim_z, const auto_aim_interfaces::msg::Target::SharedPtr &msg) {
  float time_delay = bias_time_ + fly_time_;
  PredictArmorPosition(msg, time_delay);

  if (last_selected_idx_ == LOST) {
    int idx = SelectArmor(msg);
    float toyaw =
        SolveYaw(pre_position_[idx].x, pre_position_[idx].y) - msg->camera_yaw;
    float turn_time = fabsf(toyaw) / 0.58f;
    PredictArmorPosition(msg, time_delay + turn_time); ////////////
    int selected_idx = SelectArmor(msg);
    UpdateSolveState(selected_idx, pitch, yaw, is_fire, aim_x, aim_y, aim_z,
                     msg);
    RCLCPP_DEBUG(logger_, "fly_time=%.3fs, turn_time= %.3fs, selected_idx=%d",
                 fly_time_, turn_time, selected_idx);

  } else {
    int selected_idx = SelectArmor(msg);
    if (selected_idx == last_selected_idx_) {
      RCLCPP_DEBUG(logger_, "selected_idx=%d", selected_idx);
      UpdateSolveState(selected_idx, pitch, yaw, is_fire, aim_x, aim_y, aim_z,
                       msg);
    } else if (selected_idx != last_selected_idx_) {
      yaw = SolveYaw(pre_position_[selected_idx].x,
                     pre_position_[selected_idx].y);
      last_yaw_ = SolveYaw(pre_position_[last_selected_idx_].x,
                           pre_position_[last_selected_idx_].y);
      RCLCPP_DEBUG(
          logger_,
          "selected_idx=%d,last_selected_idx_=%d, yaw - last_yaw_=%.3f",
          selected_idx, last_selected_idx_, yaw - last_yaw_);
      if (fabsf(yaw - last_yaw_) < 0.08) {
        fire_logic_mode_ = FireLogicMode::SPIN;
        selected_idx = CENTER;
        UpdateSolveState(selected_idx, pitch, yaw, is_fire, aim_x, aim_y, aim_z,
                         msg);
      }
    }
  }
}
void SolveTrajectory::UpdateSolveState(
    int selected_idx, float &pitch, float &yaw, bool &is_fire, float &aim_x,
    float &aim_y, float &aim_z,
    const auto_aim_interfaces::msg::Target::SharedPtr &msg) {
  if (selected_idx == CENTER) {
    aim_x = pre_x_center_;
    aim_y = pre_y_center_;
    aim_z = pre_z_center_;
  } else {
    aim_x = pre_position_[selected_idx].x;
    aim_y = pre_position_[selected_idx].y;
    aim_z = pre_position_[selected_idx].z;
  }
  last_x_v_ = msg->velocity.x;
  last_y_v_ = msg->velocity.y;
  pitch = SolvePitch(aim_x, aim_y, aim_z);
  yaw = SolveYaw(aim_x, aim_y);
  is_fire = CanFire(yaw, 0.2f, msg);
  if (selected_idx != LOST || is_fire) {
    last_yaw_ = yaw;
    last_selected_idx_ = selected_idx;
  }
}

void SolveTrajectory::AutoSolveTrajectory(
    float &pitch, float &yaw, bool &is_fire, float &aim_x, float &aim_y,
    float &aim_z, const auto_aim_interfaces::msg::Target::SharedPtr msg) {
  auto start = std::chrono::high_resolution_clock::now();

  if (!msg) {
    RCLCPP_ERROR(logger_, "Invalid target message");
    return;
  }

  FireLogicDefault(pitch, yaw, is_fire, aim_x, aim_y, aim_z, msg);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  RCLCPP_DEBUG(logger_, "Trajectory solve time: %ld us", duration.count());
}
} // namespace rm_auto_aim
// 没有LOST，预瞄考虑装甲板的位置变化，使用这一时刻与下一时刻的yaw变换计算