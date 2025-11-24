#ifndef __SOLVETRAJECTORY_H__
#define __SOLVETRAJECTORY_H__

#ifndef PI
#define PI 3.1415926535f
#endif
#define GRAVITY 9.78
typedef unsigned char uint8_t;
#include <functional>

#include "auto_aim_interfaces/msg/target.hpp"
#include "auto_aim_interfaces/msg/velocity.hpp"

namespace rm_auto_aim
{
class SolveTrajectory
{
 public:
  enum ARMOR_ID : uint8_t
  {
    ARMOR_OUTPOST = 0,
    ARMOR_HERO = 1,
    ARMOR_ENGINEER = 2,
    ARMOR_INFANTRY3 = 3,
    ARMOR_INFANTRY4 = 4,
    ARMOR_INFANTRY5 = 5,
    ARMOR_GUARD = 6,
    ARMOR_BASE = 7
  };

  enum ARMOR_NUM : uint8_t
  {
    ARMOR_NUM_OUTPOST = 3,
    ARMOR_NUM_NORMAL = 4
  };

  enum BULLET_TYPE : uint8_t
  {
    BULLET_17 = 0,
    BULLET_42 = 1
  };

  // 用于存储目标装甲板的信息
  struct TargetPos
  {
    float x;    // 装甲板在世界坐标系下的x
    float y;    // 装甲板在世界坐标系下的y
    float z;    // 装甲板在世界坐标系下的z
    float yaw;  // 装甲板坐标系相对于世界坐标系的yaw角
  };

  SolveTrajectory(const float& k, const int& bias_time, const float& s_bias,
                  const float& z_bias);

  float k;  // 弹道系数

  // 自身参数
  // enum BULLET_TYPE bullet_type;  //自身机器人类型 0-步兵 1-英雄
  double current_v;  // 当前弹速
  double fly_time;

  // 目标参数
  int bias_time;  // 偏置时间
  float s_bias;   // 枪口前推的距离
  float z_bias;   // yaw轴电机到枪口水平面的垂直距离

  float tar_yaw;  // 目标yaw

  // std::vector<tar_pos> tar_position;

  struct TargetPos tar_position[4];

  std::vector<float> tmp_yaws;

  float min_yaw_in_cycle;
  float max_yaw_in_cycle;

  void Init(const auto_aim_interfaces::msg::Velocity::SharedPtr velocity_msg);

  // 单方向空气阻力模型
  float MonoDirectionalAirResistanceModel(float s, float v, float angle);

  // pitch弹道补偿
  float PitchTrajectoryCompensation(float s, float y, float v);

  bool ShouldFire(float tmp_yaw, float v_yaw, float timeDelay);

  using FireCallback = std::function<void(bool)>;

  void SetFireCallback(FireCallback callback) { fireCallback = callback; }

  void CalculateArmorPosition(const auto_aim_interfaces::msg::Target::SharedPtr& msg,
                              bool use_1, bool use_average_radius);

  std::pair<float, float> CalculatePitchAndYaw(
      int idx, const auto_aim_interfaces::msg::Target::SharedPtr& msg, float timeDelay,
      float s_bias, float z_bias, float current_v, bool use_target_center_for_yaw,
      float& aim_x, float& aim_y, float& aim_z);

  int SelectArmor(const auto_aim_interfaces::msg::Target::SharedPtr& msg,
                  bool select_by_min_yaw);

  void FireLogicIsTop(float& pitch, float& yaw, float& aim_x, float& aim_y, float& aim_z,
                      const auto_aim_interfaces::msg::Target::SharedPtr& msg);

  void FireLogicDefault(float& pitch, float& yaw, float& aim_x, float& aim_y,
                        float& aim_z,
                        const auto_aim_interfaces::msg::Target::SharedPtr& msg);

  // 根据最优决策得出被击打装甲板 自动解算弹道
  void AutoSolveTrajectory(float& pitch, float& yaw, float& aim_x, float& aim_y,
                           float& aim_z,
                           const auto_aim_interfaces::msg::Target::SharedPtr msg);

 private:
  FireCallback fireCallback;

  // 完全空气阻力模型
  float CompleteAirResistanceModel(float s, float v, float angle);
};

}  // namespace rm_auto_aim

#endif /*__SOLVETRAJECTORY_H__*/
