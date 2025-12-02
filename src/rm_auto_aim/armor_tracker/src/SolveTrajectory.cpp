// TODO 完整弹道模型
// TODO 适配英雄机器人弹道解算

// STD
#include "armor_tracker/SolveTrajectory.hpp"

namespace rm_auto_aim
{

//=============================================================================
// 构造函数
//=============================================================================
SolveTrajectory::SolveTrajectory(const float& k, const int& bias_time,
                                 const float& s_bias, const float& z_bias,
                                 CalculateMode calculate_mode,
                                 const TrajectoryTable::TableConfig& table_config)
    : k_(k),
      table_(std::make_unique<TrajectoryTable>(table_config)),
      calculate_mode_(calculate_mode),
      bias_time_(bias_time),
      s_bias_(s_bias),
      z_bias_(z_bias)
{
  if (calculate_mode_ == CalculateMode::TABLE_LOOKUP)
  {
    table_->Init();
  }
}

//=============================================================================
// 初始化
//=============================================================================
void SolveTrajectory::Init(
    const auto_aim_interfaces::msg::Velocity::SharedPtr velocity_msg)
{
  if (!std::isnan(velocity_msg->velocity))
  {
    current_v_ = velocity_msg->velocity;
  }
  else
  {
    current_v_ = 12.0f;  // 默认值
  }
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
  // 飞行时间 t = (e^{k s} - 1) / (k v cos(angle))
  fly_time_ = (std::exp(k_ * s) - 1.0) / (k_ * v * std::cos(angle));

  if (std::isnan(fly_time_))
  {
    std::cerr << "[SolveTrajectory] Fly time is nan! s: " << s << " v: " << v
              << " angle: " << angle << '\n';
    fly_time_ = 0.0;
    return 0.0f;
  }

  if (fly_time_ < 0.0)
  {
    std::cerr << "[SolveTrajectory] Fly time is negative! s: " << s << " v: " << v
              << " angle: " << angle << '\n';
    fly_time_ = 0.0;
    return 0.0f;
  }

  // z = v sin(angle) t - 0.5 g t^2
  float z = static_cast<float>(v * std::sin(angle) * fly_time_ -
                               GRAVITY * fly_time_ * fly_time_ / 2.0);
  return z;
}

//=============================================================================
// 完整弹道模型 (TODO)
//=============================================================================
float SolveTrajectory::CompleteAirResistanceModel(float /*s*/, float /*v*/,
                                                  float /*angle*/)
{
  // TODO: Implement complete air resistance model
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
  if (calculate_mode_ == CalculateMode::NORMAL || !table_->IsInit())
  {
    float z_temp = z;
    float angle_pitch = 0.0f;
    // 经验：20~27 次迭代通常足够收敛
    for (int i = 0; i < 22; ++i)
    {
      if (std::isnan(z_temp))
      {
        std::cerr << "[SolveTrajectory] z_temp is nan! s: " << s << " v: " << v
                  << " angle_pitch: " << angle_pitch << '\n';
        return 0.0f;
      }

      angle_pitch = std::atan2(z_temp, s);

      const float Z_ACTUAL = MonoDirectionalAirResistanceModel(s, v, angle_pitch);
      const float DZ = 0.3f * (z - Z_ACTUAL);
      z_temp += DZ;

      if (std::fabs(DZ) < 1e-5f)
      {
        break;
      }
    }

    return angle_pitch;
  }
  else if (calculate_mode_ == CalculateMode::TABLE_LOOKUP && table_->IsInit())
  {
    auto res = table_->Check(s, z);
    fly_time_ = res.t;
    return static_cast<float>(res.pitch);
  }

  return 0.0f;
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
  tmp_yaws_.clear();

  min_yaw_in_cycle_ = std::numeric_limits<float>::max();
  max_yaw_in_cycle_ = std::numeric_limits<float>::lowest();

  // 对每块装甲板
  for (int i = 0; i < msg->armors_num; i++)
  {
    // 计算 tmp_yaw，目标yaw换算
    float tmp_yaw = static_cast<float>(tar_yaw_ + i * 2.0 * PI / msg->armors_num);
    tmp_yaws_.push_back(tmp_yaw);
    min_yaw_in_cycle_ = std::min(min_yaw_in_cycle_, tmp_yaw);
    max_yaw_in_cycle_ = std::max(max_yaw_in_cycle_, tmp_yaw);

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
    tar_position_[i].x = static_cast<float>(msg->position.x - r * std::cos(tmp_yaw));
    tar_position_[i].y = static_cast<float>(msg->position.y - r * std::sin(tmp_yaw));
    tar_position_[i].z = static_cast<float>(msg->position.z);
    tar_position_[i].yaw = tmp_yaw;
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
  aim_x = static_cast<float>(tar_position_[idx].x + msg->velocity.x * timeDelay);
  aim_y = static_cast<float>(tar_position_[idx].y + msg->velocity.y * timeDelay);
  aim_z = static_cast<float>(tar_position_[idx].z);

  // 切换识别装甲板还是robot中心
  double yaw_x = use_target_center_for_yaw ? msg->position.x : aim_x;
  double yaw_y = use_target_center_for_yaw ? msg->position.y : aim_y;

  // 计算水平距离
  float horizontal_distance = std::sqrt(aim_x * aim_x + aim_y * aim_y) - s_bias;
  float z_goal = aim_z + z_bias;
  // pitch轴解算 (自动选择查表或迭代)
  float pitch = PitchTrajectoryCompensation(horizontal_distance, z_goal, current_v);

  // yaw轴解算
  float yaw = static_cast<float>(std::atan2(yaw_y, yaw_x));

  return {pitch, yaw};
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
    double min_yaw_diff = std::fabs(msg->yaw - tar_position_[0].yaw);
    for (int i = 1; i < msg->armors_num; i++)
    {
      double temp_yaw_diff = std::fabs(msg->yaw - tar_position_[i].yaw);
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
      float dx = tar_position_[i].x;
      float dy = tar_position_[i].y;
      float dz = tar_position_[i].z;
      float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
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
  tar_yaw_ = static_cast<float>(msg->yaw);
  // 线性预测
  float time_delay = static_cast<float>(bias_time_ / 1000.0 + fly_time_);

  // 装甲板id顺序，以四块装甲板为例，逆时针编号
  //       2
  //    3     1
  //       0
  int idx = 0;
  bool is_fire = false;

  if (msg->armors_num == ARMOR_NUM_OUTPOST)
  {
    CalculateArmorPosition(msg, false, true);
    for (size_t i = 0; i < tmp_yaws_.size(); i++)
    {
      float tmp_yaw = tmp_yaws_[i];
      if (ShouldFire(tmp_yaw, static_cast<float>(msg->v_yaw), time_delay))
      {
        is_fire = true;
        idx = static_cast<int>(i);
        if (fire_callback_)
        {
          fire_callback_(is_fire);
        }
        break;
      }
    }
  }
  else
  {
    // 普通装甲板
    CalculateArmorPosition(msg, false, false);
    for (size_t i = 0; i < tmp_yaws_.size(); i++)
    {
      float tmp_yaw = tmp_yaws_[i];
      if (ShouldFire(tmp_yaw, static_cast<float>(msg->v_yaw), time_delay))
      {
        is_fire = true;
        idx = static_cast<int>(i);
        if (fire_callback_)
        {
          fire_callback_(is_fire);
        }
        break;
      }
    }
  }

  // std::cout << "idx: " << idx << '\n';

  // 解算pitch和yaw
  auto [p, y] =
      CalculatePitchAndYaw(idx, msg, time_delay, s_bias_, z_bias_,
                           static_cast<float>(current_v_), false, aim_x, aim_y, aim_z);
  pitch = p;
  yaw = y;
}

//=============================================================================
// 默认开火逻辑
//=============================================================================
void SolveTrajectory::FireLogicDefault(
    float& pitch, float& yaw, float& aim_x, float& aim_y, float& aim_z,
    const auto_aim_interfaces::msg::Target::SharedPtr& msg)
{
  // 线性预测
  float time_delay = static_cast<float>(bias_time_ / 1000.0 + fly_time_);
  tar_yaw_ += static_cast<float>(msg->v_yaw) * time_delay;

  int idx = 0;

  CalculateArmorPosition(msg, false, false);
  for (size_t i = 0; i < tmp_yaws_.size(); i++)
  {
    idx = SelectArmor(msg, false);
    break;
  }
  // std::cout << "idx: " << idx << '\n';

  auto [p, y] =
      CalculatePitchAndYaw(idx, msg, time_delay, s_bias_, z_bias_,
                           static_cast<float>(current_v_), false, aim_x, aim_y, aim_z);
  pitch = p;
  yaw = y;
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
