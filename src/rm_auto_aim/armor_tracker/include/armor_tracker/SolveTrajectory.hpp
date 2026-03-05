#pragma once

#include <cmath>
#include <fstream>

#include "auto_aim_interfaces/msg/target.hpp"
#include "auto_aim_interfaces/msg/velocity.hpp"
#include "tracker.hpp"

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
    double pitch;
    double t;
    double v;
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
  Cell Check(double x, double y) const
  {
    if (!init_)
    {
      return {NAN, NAN, NAN};
    }

    // 边界检查
    double adjusted_x = x;
    double adjusted_y = y;

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

    return {ge.pitch, ge.t, ge.v};
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
      std::cerr << "[TrajectoryTable] 错误: "
                   "读取数据失败或文件大小不匹配，使用默认弹道解算"
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

  bool ReloadTable(const std::string& new_filename)
  {
    filename_ = new_filename;
    init_ = false;
    return Init();
  }

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
using time_point = std::chrono::high_resolution_clock::time_point;

class SolveTrajectory
{
 public:
  static constexpr double GRAVITY = 9.78f;
  static constexpr double SMALL_HALF_LENGTH = 135 / 2.0 / 1000.0;
  static constexpr double LARGE_HALF_LENGTH = 225 / 2.0 / 1000.0;

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
  struct TargetPostion
  {
    double x;    // 装甲板在世界坐标系下的x
    double y;    // 装甲板在世界坐标系下的y
    double z;    // 装甲板在世界坐标系下的z
    double yaw;  // 装甲板坐标系相对于世界坐标系的yaw角
  };

  // 构造函数
  SolveTrajectory(const double& k, const double& bias_time, const double& s_bias,
                  const double& z_bias, const double& pitch_bias,
                  CalculateMode calculate_mode,
                  const TrajectoryTable::TableConfig& table_config);

  // 初始化弹速
  void Init(const auto_aim_interfaces::msg::Velocity::SharedPtr velocity_msg);

  void ReBuild();

  // 单方向空气阻力模型
  double MonoDirectionalAirResistanceModel(double s, double v, double angle);
  TargetPostion PredictCenter(const auto_aim_interfaces::msg::Target::SharedPtr& msg,
                              double time_delay);
  TargetPostion PredictArmor(const auto_aim_interfaces::msg::Target::SharedPtr& msg,
                             double time_delay, int idx,
                             SolveTrajectory::TargetPostion& pre_center);
  void PredictAllArmorPosition(const auto_aim_interfaces::msg::Target::SharedPtr& msg,
                               double time_delay);
  void PredictOneArmorPosition(
      const auto_aim_interfaces::msg::Target::SharedPtr& msg, double time_delay, int idx);

  double SolvePitch(double x, double y, double z);
  double SolveYaw(double x, double y);
  bool CanFire(double yaw, const auto_aim_interfaces::msg::Target::SharedPtr& msg,
               bool flag);

  void GlobalSelectArmor(const auto_aim_interfaces::msg::Target::SharedPtr& msg);
  void LocalSelectArmor(const auto_aim_interfaces::msg::Target::SharedPtr& msg);
  void PreSelectArmor(const auto_aim_interfaces::msg::Target::SharedPtr& msg);
  void AutoSelectArmor(const auto_aim_interfaces::msg::Target::SharedPtr& msg,
                       bool is_pre_select);

  void UpdateFireLogicMode();
  void UpdateSolveState(double& pitch, double& yaw, bool& is_fire, double& aim_x,
                        double& aim_y, double& aim_z, int& idx,
                        const auto_aim_interfaces::msg::Target::SharedPtr& msg);

  // 根据最优决策得出被击打装甲板 自动解算弹道
  void AutoSolveTrajectory(double& pitch, double& yaw, bool& is_fire, double& aim_x,
                           double& aim_y, double& aim_z, int& idx,
                           const auto_aim_interfaces::msg::Target::SharedPtr msg);

  bool ReloadTable(const std::string& new_filename);

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

  TargetPostion pre_center_;
  TargetPostion pre_position_[4];

  // 弹道查找表
  std::unique_ptr<TrajectoryTable> table_;
  CalculateMode calculate_mode_ = CalculateMode::NORMAL;  ///< 弹道计算模式
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
