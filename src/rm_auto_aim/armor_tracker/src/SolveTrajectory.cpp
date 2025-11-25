// TODO 完整弹道模型
// TODO 适配英雄机器人弹道解算

// STD
#include "armor_tracker/SolveTrajectory.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

namespace rm_auto_aim
{

//=============================================================================
// 构造函数
//=============================================================================
SolveTrajectory::SolveTrajectory(const float& k, const int& bias_time,
                                 const float& s_bias, const float& z_bias)
    : k(k),
      bias_time(bias_time),
      s_bias(s_bias),
      z_bias(z_bias),
      trajectory_table_(nullptr)
{
}

SolveTrajectory::SolveTrajectory(const float& k, const int& bias_time,
                                 const float& s_bias, const float& z_bias,
                                 const TrajectoryTable::TableConfig& table_config)
    : k(k), bias_time(bias_time), s_bias(s_bias), z_bias(z_bias)
{
  trajectory_table_ = std::make_unique<TrajectoryTable>(table_config);
}

//=============================================================================
// 初始化
//=============================================================================
void SolveTrajectory::Init(
    const auto_aim_interfaces::msg::Velocity::SharedPtr velocity_msg)
{
  if (!std::isnan(velocity_msg->velocity))
  {
    current_v = velocity_msg->velocity;
  }
  else
  {
    current_v = 12;  // 默认值
  }
}

bool SolveTrajectory::InitTrajectoryTable()
{
  if (trajectory_table_)
  {
    return trajectory_table_->Init();
  }
  return false;
}

//=============================================================================
// 单方向空气阻力弹道模型
//=============================================================================
/*
@brief 简单物理模型
@param s:m 距离
@param v:m/s 速度
@param angle:rad 角度
@return z:m 高度
*/
float SolveTrajectory::MonoDirectionalAirResistanceModel(float s, float v, float angle)
{
  // t为给定v与angle时的飞行时间
  fly_time = static_cast<float>((std::exp(k * s) - 1) / (k * v * std::cos(angle)));
  if (fly_time < 0)
  {
    // 由于严重超出最大射程，计算过程中浮点数溢出，导致t变成负数
    printf("[WARN]: Exceeding the maximum range!\n");
    // 重置t，防止下次调用会出现nan
    fly_time = 0;
    return 0;
  }
  // z为给定v与angle时的高度
  float z = static_cast<float>(v * std::sin(angle) * fly_time -
                               GRAVITY * fly_time * fly_time / 2);

  return z;
}

//=============================================================================
// 完整弹道模型 (TODO)
//=============================================================================
float SolveTrajectory::CompleteAirResistanceModel(float s, float v, float angle)
{
  // TODO: Implement complete air resistance model
  (void)s;
  (void)v;
  (void)angle;
  return 0.0f;
}

//=============================================================================
// Pitch轴解算 - 主入口 (集成查表逻辑)
//=============================================================================
/*
@brief pitch轴解算，优先使用查表，查表失败则回退到迭代法
@param s:m 水平距离
@param z:m 高度
@param v:m/s 弹速
@return angle_pitch:rad
*/
float SolveTrajectory::PitchTrajectoryCompensation(float s, float z, float v)
{
  // 优先尝试查表
  if (IsUsingTable())
  {
    float pitch = PitchTrajectoryCompensationWithTable(s, z, v);
    if (!std::isnan(pitch))
    {
      return pitch;
    }
    // 查表失败，回退到迭代法
    std::cout << "[SolveTrajectory] 查表失败，迭代法" << '\n';
  }

  // 使用迭代法
  return PitchTrajectoryCompensationIterative(s, z, v);
}

//=============================================================================
// Pitch轴解算 - 查表方法
//=============================================================================
float SolveTrajectory::PitchTrajectoryCompensationWithTable(float s, float z, float v)
{
  if (!trajectory_table_ || !trajectory_table_->IsInit())
  {
    return NAN;
  }

  // 查表：s为水平距离(x)，z为高度(y)
  // 注意：查找表是针对特定弹速生成的，如果当前弹速与表的弹速差异较大，
  // 可能需要进行插值或回退到迭代法
  TrajectoryTable::Cell cell = trajectory_table_->Check(
      s, z, table_x_bias_, table_y_bias_, table_pitch_bias_, table_t_bias_);

  if (std::isnan(cell.pitch))
  {
    return NAN;
  }

  // 更新飞行时间
  fly_time = cell.t;

  return cell.pitch;
}

//=============================================================================
// Pitch轴解算 - 迭代方法 (原始实现)
//=============================================================================
/*
@brief pitch轴解算 一般全解算次数在20-27之间
@param s:m 距离
@param z:m 高度
@param v:m/s 弹速
@return angle_pitch:rad
*/
float SolveTrajectory::PitchTrajectoryCompensationIterative(float s, float z, float v)
{
  // 初始
  float z_temp = z;
  float angle_pitch = 0.0f;

  // 迭代求解 pitch
  for (int i = 0; i < 22; i++)
  {
    angle_pitch = std::atan2(z_temp, s);
    // 单方向空气阻力模型
    float z_actual = MonoDirectionalAirResistanceModel(s, v, angle_pitch);
    float dz = 0.3f * (z - z_actual);
    z_temp += dz;

    if (std::fabs(dz) < 0.00001f)
    {
      break;
    }
  }

  return angle_pitch;
}

//=============================================================================
// 判断是否开火
//=============================================================================
/*
@brief 一种线性预测
@param tmp_yaw 装甲板yaw
@param v_yaw yaw速度
@param timeDelay 时间延迟
@return true/false
*/
bool SolveTrajectory::ShouldFire(float tmp_yaw, float v_yaw, float timeDelay)
{
  // 击打旋转一圈之后的
  return std::fabs((tmp_yaw + v_yaw * timeDelay) - 2 * PI) < 0.001;
}

//=============================================================================
// 解算四块装甲板位置
//=============================================================================
/*
@brief 根据当前观测到的装甲板信息，计算出来所有装甲板位置
@param msg 目标消息
@param use_1 使用radius_1的标志
@param use_average_radius 是否使用平均半径
*/
void SolveTrajectory::CalculateArmorPosition(
    const auto_aim_interfaces::msg::Target::SharedPtr& msg, bool use_1,
    bool use_average_radius)
{
  tmp_yaws.clear();

  min_yaw_in_cycle = std::numeric_limits<float>::max();
  max_yaw_in_cycle = std::numeric_limits<float>::lowest();

  // 对每块装甲板
  for (int i = 0; i < msg->armors_num; i++)
  {
    // 计算 tmp_yaw，目标yaw换算
    float tmp_yaw = static_cast<float>(tar_yaw + i * 2.0 * PI / msg->armors_num);
    tmp_yaws.push_back(tmp_yaw);
    min_yaw_in_cycle = std::min(min_yaw_in_cycle, tmp_yaw);
    max_yaw_in_cycle = std::max(max_yaw_in_cycle, tmp_yaw);

    // 半径
    float r{};
    if (use_average_radius)
    {
      r = static_cast<float>(msg->radius_1 + msg->radius_2) / 2;
    }
    else
    {
      r = use_1 ? static_cast<float>(msg->radius_1) : static_cast<float>(msg->radius_2);
    }

    // 三角函数计算装甲板位置
    tar_position[i].x = static_cast<float>(msg->position.x - r * std::cos(tmp_yaw));
    tar_position[i].y = static_cast<float>(msg->position.y - r * std::sin(tmp_yaw));
    tar_position[i].z = static_cast<float>(msg->position.z);
    tar_position[i].yaw = tmp_yaw;
    use_1 = !use_1;
  }
}

//=============================================================================
// 解算Pitch && Yaw
//=============================================================================
/*
@brief 计算pitch和yaw角度
@param idx 最适合开火的装甲板索引
@param msg 目标消息
@param timeDelay 延迟时间
@param s_bias 枪口前推偏置
@param z_bias z偏置
@param current_v 弹速
@param use_target_center_for_yaw 是否使用目标中心计算yaw
@param aim_x/aim_y/aim_z 打击落点 (输出)
@return pair<pitch, yaw>
*/
std::pair<float, float> SolveTrajectory::CalculatePitchAndYaw(
    int idx, const auto_aim_interfaces::msg::Target::SharedPtr& msg, float timeDelay,
    float s_bias, float z_bias, float current_v, bool use_target_center_for_yaw,
    float& aim_x, float& aim_y, float& aim_z)
{
  // 对打击目标xyz进行线性预测
  aim_x = static_cast<float>(tar_position[idx].x + msg->velocity.x * timeDelay);
  aim_y = static_cast<float>(tar_position[idx].y + msg->velocity.y * timeDelay);
  aim_z = static_cast<float>(tar_position[idx].z);

  // 切换识别装甲板还是robot中心
  double yaw_x = use_target_center_for_yaw ? msg->position.x : aim_x;
  double yaw_y = use_target_center_for_yaw ? msg->position.y : aim_y;

  // 计算水平距离
  float horizontal_distance = std::sqrt(aim_x * aim_x + aim_y * aim_y) - s_bias;

  // pitch轴解算 (自动选择查表或迭代)
  float pitch =
      PitchTrajectoryCompensation(horizontal_distance, aim_z + z_bias, current_v);

  // yaw轴解算
  float yaw = static_cast<float>(std::atan2(yaw_y, yaw_x));

  return std::make_pair(pitch, yaw);
}

//=============================================================================
// 选择装甲板
//=============================================================================
int SolveTrajectory::SelectArmor(const auto_aim_interfaces::msg::Target::SharedPtr& msg,
                                 bool select_by_min_yaw)
{
  int selected_armor_idx = 0;

  select_by_min_yaw = false;

  if (select_by_min_yaw)
  {
    // 选择枪管到目标装甲板yaw最小的装甲板
    double min_yaw_diff = std::fabs(msg->yaw - tar_position[0].yaw);
    for (int i = 1; i < msg->armors_num; i++)
    {
      double temp_yaw_diff = std::fabs(msg->yaw - tar_position[i].yaw);
      if (temp_yaw_diff < min_yaw_diff)
      {
        min_yaw_diff = temp_yaw_diff;
        selected_armor_idx = i;
      }
    }
  }
  else
  {
    // 选择离机器人最近的装甲板
    float min_distance = std::numeric_limits<float>::max();
    for (int i = 0; i < msg->armors_num; i++)
    {
      float distance = std::sqrt(tar_position[i].x * tar_position[i].x +
                                 tar_position[i].y * tar_position[i].y +
                                 tar_position[i].z * tar_position[i].z);
      if (distance < min_distance)
      {
        min_distance = distance;
        selected_armor_idx = i;
      }
    }
  }

  return selected_armor_idx;
}

//=============================================================================
// 最优开火逻辑 (陀螺模式)
//=============================================================================
void SolveTrajectory::FireLogicIsTop(
    float& pitch, float& yaw, float& aim_x, float& aim_y, float& aim_z,
    const auto_aim_interfaces::msg::Target::SharedPtr& msg)
{
  tar_yaw = static_cast<float>(msg->yaw);
  // 线性预测
  float time_delay = static_cast<float>(bias_time / 1000.0 + fly_time);

  // 装甲板id顺序，以四块装甲板为例，逆时针编号
  //       2
  //    3     1
  //       0
  int idx = 0;
  bool is_fire = false;

  if (msg->armors_num == ARMOR_NUM_OUTPOST)
  {
    CalculateArmorPosition(msg, false, true);
    for (size_t i = 0; i < tmp_yaws.size(); i++)
    {
      float tmp_yaw = tmp_yaws[i];
      if (ShouldFire(tmp_yaw, static_cast<float>(msg->v_yaw), time_delay))
      {
        is_fire = true;
        idx = static_cast<int>(i);
        if (fireCallback)
        {
          fireCallback(is_fire);
        }
        break;
      }
    }
  }
  else
  {
    // 普通装甲板
    CalculateArmorPosition(msg, false, false);
    for (size_t i = 0; i < tmp_yaws.size(); i++)
    {
      float tmp_yaw = tmp_yaws[i];
      if (ShouldFire(tmp_yaw, static_cast<float>(msg->v_yaw), time_delay))
      {
        is_fire = true;
        idx = static_cast<int>(i);
        if (fireCallback)
        {
          fireCallback(is_fire);
        }
        break;
      }
    }
  }

  std::cout << "idx: " << idx << '\n';

  // 解算pitch和yaw
  auto pitch_and_yaw =
      CalculatePitchAndYaw(idx, msg, time_delay, s_bias, z_bias,
                           static_cast<float>(current_v), false, aim_x, aim_y, aim_z);
  pitch = pitch_and_yaw.first;
  yaw = pitch_and_yaw.second;
}

//=============================================================================
// 默认开火逻辑
//=============================================================================
void SolveTrajectory::FireLogicDefault(
    float& pitch, float& yaw, float& aim_x, float& aim_y, float& aim_z,
    const auto_aim_interfaces::msg::Target::SharedPtr& msg)
{
  // 线性预测
  float time_delay = static_cast<float>(bias_time / 1000.0 + fly_time);
  tar_yaw += static_cast<float>(msg->v_yaw) * time_delay;

  int idx = 0;

  CalculateArmorPosition(msg, false, false);
  for (size_t i = 0; i < tmp_yaws.size(); i++)
  {
    idx = SelectArmor(msg, false);
    break;
  }
  std::cout << "idx: " << idx << '\n';

  auto pitch_and_yaw =
      CalculatePitchAndYaw(idx, msg, time_delay, s_bias, z_bias,
                           static_cast<float>(current_v), false, aim_x, aim_y, aim_z);
  pitch = pitch_and_yaw.first;
  yaw = pitch_and_yaw.second;
}

//=============================================================================
// 自动弹道解算 - 主入口
//=============================================================================
/*
@brief 根据最优决策得出被击打装甲板，自动解算弹道
@param pitch:rad 传出pitch
@param yaw:rad 传出yaw
@param aim_x/aim_y/aim_z 打击目标坐标 (传出)
@param msg 目标消息
*/
void SolveTrajectory::AutoSolveTrajectory(
    float& pitch, float& yaw, float& aim_x, float& aim_y, float& aim_z,
    const auto_aim_interfaces::msg::Target::SharedPtr msg)
{
  // 优先开火逻辑
  FireLogicIsTop(pitch, yaw, aim_x, aim_y, aim_z, msg);
}

// 从坐标轴正向看向原点，逆时针方向为正

}  // namespace rm_auto_aim
