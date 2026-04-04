#pragma once

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>

#include "auto_aim_interfaces/msg/target.hpp"
#include "auto_aim_interfaces/msg/velocity.hpp"
#include "planning_trajectory/table.hpp"

namespace rm_auto_aim
{
using time_point = std::chrono::high_resolution_clock::time_point;

class TrajectorySolver
{
 public:
  static constexpr double GRAVITY = 9.78f;
  static constexpr double SMALL_HALF_LENGTH = 135 / 2.0 / 1000.0;
  static constexpr double LARGE_HALF_LENGTH = 230 / 2.0 / 1000.0;
  static constexpr double outpost_dz = 0.1;

  enum CalculateMode : uint8_t
  {
    NORMAL = 0,
    TABLE_LOOKUP = 1
  };

  enum class FireLogicMode
  {
    OUTPOST,
    SPIN,
    SPIN_TEMP,
    COMMON,
    BUFF
  };

  enum SpecialArmor : int8_t
  {
    LOST = -2,
    CENTER = -1
  };

  // 用于存储目标装甲板的信息
  struct TarPostion
  {
    double x;    // 装甲板在世界坐标系下的x
    double y;    // 装甲板在世界坐标系下的y
    double z;    // 装甲板在世界坐标系下的z
    double yaw;  // 装甲板坐标系相对于世界坐标系的yaw角
  };

  struct TarVelocity
  {
    double x;
    double y;
    double z;
    double yaw;
  };

  struct Target
  {
    TarPostion position;
    TarVelocity velocity;
    int num;
    std::string type;
    double radius1;
    double radius2;
    int outpost_idx;
  };

  struct control
  {
    double yaw;
    double pitch;
    double vel_yaw;
    double acc_yaw;
    bool is_fire;
  };

  // 构造函数
  TrajectorySolver(const double& k, const double& bias_time, const double& s_bias,
                   const double& z_bias, const double& pitch_bias,
                   CalculateMode calculate_mode, const Table::TableConfig& table_config,
                   const Table::TableConfig& table_config_lob_);

  // 初始化弹速
  void Init(const auto_aim_interfaces::msg::Velocity::SharedPtr velocity_msg);

  void ReBuild();

  // 单方向空气阻力模型
  double MonoDirectionalAirResistanceModel(double s, double v, double angle);
  TarPostion PredictCenter(double time_delay);
  TarPostion PredictArmor(double time_delay, int idx,
                          TrajectorySolver::TarPostion& pre_center);
  void PredictAllArmorPosition(double time_delay);
  void PredictOneArmorPosition(double time_delay, int idx);

  double SolvePitch(double x, double y, double z);
  double SolveYaw(double x, double y);
  bool CanFire(double yaw, bool flag);
  void GlobalSelectArmor(double time_delay);
  void LocalSelectArmor(double time_delay);
  void PreSelectArmor(double time_delay);
  void AutoSelectArmor(double time_delay, bool is_pre_select);

  void UpdateFireLogicMode();
  void UpdateSolveState(double& pitch, double& yaw, bool& is_fire, double& aim_x,
                        double& aim_y, double& aim_z, int& idx);

  // 根据最优决策得出被击打装甲板 自动解算弹道
  void AutoSolveTrajectory(double& pitch, double& yaw, bool& is_fire, double& aim_x,
                           double& aim_y, double& aim_z, int& idx, const Target& target,
                           double gimbal_yaw, const double send_time);

  bool SwitchTable()
  {
    if (!table_lob_)
    {
      RCLCPP_WARN(logger_, "LOB table not initialized, cannot switch");
      return false;
    }
    current_table_ = (current_table_ == table_) ? table_lob_ : table_;
    if (current_table_->IsInit())
    {
      RCLCPP_INFO(logger_, "Switched trajectory table to: %s",
                  current_table_->IsInit() ? "LOB" : "Normal");
      return true;
    }
    else
    {
      RCLCPP_WARN(logger_,
                  "Failed to switch trajectory table, current table is not initialized.");
      return false;
    }
  }

 private:
  // Logger
  rclcpp::Logger logger_{rclcpp::get_logger("armor_tracker")};

  // 自身参数
  double current_v_;  // 当前弹速
  double fly_time_;   // 飞行时间
  time_point start_turn_;
  time_point end_turn_;
  time_point last_start_turn_;
  double turn_s_{0.0f};
  double step_s_{0.0f};

  TarPostion pre_center_;
  TarPostion pre_position_[4];
  Target target_;
  double gimbal_yaw_;

  // 弹道查找表
  std::shared_ptr<Table> table_;
  std::shared_ptr<Table> table_lob_;
  std::shared_ptr<Table> current_table_;
  CalculateMode calculate_mode_ = CalculateMode::TABLE_LOOKUP;  ///< 弹道计算模式
  FireLogicMode fire_logic_mode_{FireLogicMode::COMMON};

  // 目标参数
  double k_;  // 弹道系数
  double pitch_bias_;
  double bias_time_;  // 偏置时间
  double s_bias_;     // 枪口前推的距离
  double z_bias_;     // yaw轴电机到枪口水平面的垂直距离

  double last_pitch_;
  double last_yaw_;
  int selected_idx_{SpecialArmor::LOST};
  double last_x_v_{0.0f};
  double last_y_v_{0.0f};
  double last_v_yaw_{0.0f};

  bool is_turn_ = false;
  bool should_last_shot_ = false;
};

}  // namespace rm_auto_aim
