#pragma once

#ifndef PI
#define PI 3.1415926535f
#endif
#define GRAVITY 9.78

#include <cmath>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "auto_aim_interfaces/msg/target.hpp"
#include "auto_aim_interfaces/msg/velocity.hpp"

namespace rm_auto_aim
{

//=============================================================================
// 弹道查找表类
//=============================================================================
class TrajectoryTable
{
 public:
  struct TableConfig
  {
    double max_x, min_x, max_y, min_y, resolution;
    size_t x_dim, y_dim;
    std::string filename;

    TableConfig(double max_x, double min_x, double max_y, double min_y, double resolution,
                std::string filename)
        : max_x(max_x),
          min_x(min_x),
          max_y(max_y),
          min_y(min_y),
          resolution(resolution),
          x_dim(static_cast<size_t>((max_x - min_x) / resolution) + 1),
          y_dim(static_cast<size_t>((max_y - min_y) / resolution) + 1),
          filename(std::move(filename))
    {
    }
  };

  struct Cell
  {
    float pitch;
    float t;
    float v;
  };

  explicit TrajectoryTable(const TableConfig& config)
      : MAX_X(config.max_x),
        MIN_X(config.min_x),
        MAX_Y(config.max_y),
        MIN_Y(config.min_y),
        RESOLUTION(config.resolution),
        X_DIM(config.x_dim),
        Y_DIM(config.y_dim),
        filename_(config.filename)
  {
  }

  ~TrajectoryTable() = default;

  // 查表获取弹道参数
  Cell Check(float x, float y, float x_bias = 0, float y_bias = 0,
             float pitch_bias = 0.02f, float t_bias = 0) const
  {
    if (!init_)
    {
      return {NAN, NAN, NAN};
    }

    // 边界检查
    float adjusted_x = x + x_bias;
    float adjusted_y = y + y_bias;

    if (adjusted_x < MIN_X || adjusted_x > MAX_X || adjusted_y < MIN_Y ||
        adjusted_y > MAX_Y)
    {
      return {NAN, NAN, NAN};
    }

    size_t xc = static_cast<size_t>(std::round((adjusted_x - MIN_X) / RESOLUTION));
    size_t yc = static_cast<size_t>(std::round((adjusted_y - MIN_Y) / RESOLUTION));
    xc = std::min(xc, X_DIM - 1);
    yc = std::min(yc, Y_DIM - 1);

    Cell ge = table_[xc * Y_DIM + yc];

    return {ge.pitch + pitch_bias, ge.t + t_bias, ge.v};
  }

  // 初始化：从二进制文件加载表
  bool Init()
  {
    table_.resize(X_DIM * Y_DIM);

    std::ifstream file_in(filename_, std::ios::in | std::ios::binary);

    if (!file_in)
    {
      std::cerr << "[TrajectoryTable] 错误: 无法打开文件 " << filename_
                << "，使用默认弹道解算" << '\n';
      init_ = false;
      return false;
    }

    const std::size_t BYTES_TO_READ = X_DIM * Y_DIM * sizeof(Cell);

    file_in.read(reinterpret_cast<char*>(table_.data()),
                 static_cast<std::streamsize>(BYTES_TO_READ));

    if (!file_in || file_in.gcount() != static_cast<std::streamsize>(BYTES_TO_READ))
    {
      std::cerr
          << "[TrajectoryTable] 错误: 读取数据失败或文件大小不匹配，使用默认弹道解算"
          << '\n';
      init_ = false;
      return false;
    }

    file_in.close();
    init_ = true;
    std::cout << "[TrajectoryTable] 弹道查找表加载成功: " << filename_ << '\n';
    return true;
  }

  bool IsInit() const { return init_; }

  // Getter
  double GetMinX() const { return MIN_X; }
  double GetMaxX() const { return MAX_X; }
  double GetMinY() const { return MIN_Y; }
  double GetMaxY() const { return MAX_Y; }

 private:
  const double MAX_X;
  const double MIN_X;
  const double MAX_Y;
  const double MIN_Y;
  const double RESOLUTION;

  const size_t X_DIM;
  const size_t Y_DIM;

  bool init_ = false;
  std::string filename_;
  std::vector<Cell> table_;
};

//=============================================================================
// 弹道解算主类
//=============================================================================
class SolveTrajectory
{
 public:
  enum TargetArmorId : uint8_t
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

  enum TargetArmorNum : uint8_t
  {
    ARMOR_NUM_OUTPOST = 3,
    ARMOR_NUM_NORMAL = 4
  };

  enum BulletType : uint8_t
  {
    BULLET_17 = 0,
    BULLET_42 = 1
  };

  // 用于存储目标装甲板的信息
  struct TargetPostion
  {
    float x;    // 装甲板在世界坐标系下的x
    float y;    // 装甲板在世界坐标系下的y
    float z;    // 装甲板在世界坐标系下的z
    float yaw;  // 装甲板坐标系相对于世界坐标系的yaw角
  };

  // 构造函数
  SolveTrajectory(const float& k, const int& bias_time, const float& s_bias,
                  const float& z_bias);

  // 带查找表配置的构造函数
  SolveTrajectory(const float& k, const int& bias_time, const float& s_bias,
                  const float& z_bias, const TrajectoryTable::TableConfig& table_config);

  float k;  // 弹道系数

  // 自身参数
  double current_v;  // 当前弹速
  double fly_time;   // 飞行时间

  // 目标参数
  int bias_time;  // 偏置时间
  float s_bias;   // 枪口前推的距离
  float z_bias;   // yaw轴电机到枪口水平面的垂直距离

  float tar_yaw;  // 目标yaw

  struct TargetPostion tar_position[4];

  std::vector<float> tmp_yaws;

  float min_yaw_in_cycle;
  float max_yaw_in_cycle;

  // 初始化弹速
  void Init(const auto_aim_interfaces::msg::Velocity::SharedPtr velocity_msg);

  // 初始化弹道查找表
  bool InitTrajectoryTable();

  // 设置是否使用查表
  void SetUseTable(bool use_table) { use_table_ = use_table; }
  bool IsUsingTable() const
  {
    return use_table_ && trajectory_table_ && trajectory_table_->IsInit();
  }

  // 单方向空气阻力模型
  float MonoDirectionalAirResistanceModel(float s, float v, float angle);

  // pitch弹道补偿 (集成查表逻辑)
  float PitchTrajectoryCompensation(float s, float z, float v);

  // 使用查表的pitch解算
  float PitchTrajectoryCompensationWithTable(float s, float z, float v);

  // 使用迭代的pitch解算 (原始方法)
  float PitchTrajectoryCompensationIterative(float s, float z, float v);

  bool ShouldFire(float tmp_yaw, float v_yaw, float timeDelay);

  using FireCallback = std::function<void(bool)>;

  void SetFireCallback(FireCallback callback) { fireCallback = std::move(callback); }

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

  // 弹道查找表
  std::unique_ptr<TrajectoryTable> trajectory_table_;
  bool use_table_ = true;  // 是否优先使用查表

  // 查表偏置参数
  float table_x_bias_ = 0.0f;
  float table_y_bias_ = 0.0f;
  float table_pitch_bias_ = 0.02f;
  float table_t_bias_ = 0.0f;
};

}  // namespace rm_auto_aim

#pragma once
