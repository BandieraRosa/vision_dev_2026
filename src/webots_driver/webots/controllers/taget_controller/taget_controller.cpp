// File: target_controller.cpp
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <webots/LED.hpp>
#include <webots/Motor.hpp>
#include <webots/Robot.hpp>

using namespace webots;

// ========================== 配置区域（全部参数在这里改） ==========================
namespace Config
{
// 全局开关
static inline bool BIG_ARMOR = false;  // true: led1/2 亮, false: led3/4 亮

// 设备名（与 world 中 device 名一致）
inline const std::array<const char*, 6> MOTOR_NAMES = {
    "target_motor_z",     "target_motor_x",   "target_motor_y",
    "target_motor_pitch", "target_motor_yaw", "target_motor_roll",
};

// LED 名称（如无需改名可不动）
inline const char* LED1 = "led1";
inline const char* LED2 = "led2";
inline const char* LED3 = "led3";
inline const char* LED4 = "led4";

// 随机平滑直线电机驱动器：统一默认参数
struct DriverDefaults
{
  double speedScale = 0.10;     // 统一放慢倍数：1.0/0.5/0.33/0.1 ...
  double baseVmaxRatio = 0.70;  // 速度上限 = baseVmaxRatio * motor->getMaxVelocity()
  double baseAmax = 5.00;       // 最大加速度 m/s^2
  double baseRetarget = 0.60;   // 换速周期 s（越小越频繁）
  double defaultMin = -0.25;    // 未提供 min/max 时的默认行程
  double defaultMax = 0.25;
  double edgeBufferPct = 0.02;      // 端点缓冲比例
  double edgeBufferAbsMin = 0.003;  // 端点缓冲绝对下限 m
  unsigned baseSeed = 12345;        // 随机种子基数（每个电机会在此基础上+index）
  double positionForce = 1000.0;    // 位置模式使用的力上限
};
inline const DriverDefaults DRIVER_DEFAULTS{};

// 针对特定电机的默认行程覆盖（仅在设备缺省 min/max 情况下生效）
// 索引对应 MOTOR_NAMES：3=pitch, 4=yaw, 5=roll
struct RangeOverride
{
  bool enable = true;
  double minPos = -0.75;
  double maxPos = 0.75;
};
inline const RangeOverride ORIENT_RANGE_OVERRIDE{true, -0.75, 0.75};
}  // namespace Config
// ==============================================================================

static inline double clampd(double x, double lo, double hi)
{
  return x < lo ? lo : (x > hi ? hi : x);
}

// 随机平滑直线电机驱动器（不依赖 PositionSensor）
class RandomLinearDriver
{
 public:
  struct Params
  {
    double speedScale = 1.0;     // 统一放慢倍数
    double baseVmaxRatio = 0.7;  // 速度上限 = baseVmaxRatio * motor->getMaxVelocity()
    double baseAmax = 5.0;       // 最大加速度 m/s^2
    double baseRetarget = 0.6;   // 换速周期 s（越小越频繁）
    double defaultMin = -0.25;   // 未提供 min/max 时的默认行程
    double defaultMax = 0.25;
    double edgeBufferPct = 0.02;      // 端点缓冲比例
    double edgeBufferAbsMin = 0.003;  // 端点缓冲绝对下限 m
    unsigned seed = 12345;            // 随机种子
    double positionForce = 1000.0;    // 位置模式使用的力上限
  };

  RandomLinearDriver(Motor* motor, const Params& p)
      : m_motor_(motor), m_params_(p), m_rng_(p.seed), m_uni_(-1.0, 1.0)
  {
    if (!m_motor_)
    {
      throw std::runtime_error("Motor is null");
    }

    // 行程（若无边界则用默认）
    m_min_ = m_motor_->getMinPosition();
    m_max_ = m_motor_->getMaxPosition();
    const bool bounded =
        std::isfinite(m_min_) && std::isfinite(m_max_) && m_max_ > m_min_;
    if (!bounded)
    {
      m_min_ = m_params_.defaultMin;
      m_max_ = m_params_.defaultMax;
      std::cerr << "[WARN] Motor has no min/max, using default [" << m_min_ << ", "
                << m_max_ << "]\n";
    }
    m_span_ = m_max_ - m_min_;
    m_buffer_ = std::max(m_span_ * m_params_.edgeBufferPct, m_params_.edgeBufferAbsMin);

    // 速度/加速度/换速周期
    double vmax_hw = m_motor_->getMaxVelocity();
    if (!std::isfinite(vmax_hw) || vmax_hw <= 0.0)
    {
      vmax_hw = 1.0;
    }

    const double S = clampd(m_params_.speedScale, 0.001, 10.0);
    m_V_MAX_ = vmax_hw * m_params_.baseVmaxRatio * S;
    m_A_MAX_ = m_params_.baseAmax * S;
    m_retargetPeriod_ = std::max(0.05, m_params_.baseRetarget / S);  // 下限避免过于频繁

    // 初始状态：从中心起步
    m_x_ = 0.5 * (m_min_ + m_max_);
    m_motor_->setForce(m_params_.positionForce);
    m_motor_->setVelocity(vmax_hw);  // 位置模式下给足内部允许速度
    m_motor_->setPosition(m_x_);

    // 初始随机目标速度
    m_v_ = 0.0;
    m_v_goal_ = m_V_MAX_ * m_uni_(m_rng_);
  }

  // 每步调用：dt（秒）
  void Step(double dt)
  {
    // 1) 到期随机换目标速度
    m_tSinceRetarget_ += dt;
    if (m_tSinceRetarget_ >= m_retargetPeriod_)
    {
      m_tSinceRetarget_ = 0.0;
      m_v_goal_ = m_V_MAX_ * m_uni_(m_rng_);  // [-V_MAX, +V_MAX]
      if (m_x_ > m_max_ - m_buffer_)
      {
        m_v_goal_ = -std::fabs(m_v_goal_);  // 靠右端向内
      }
      if (m_x_ < m_min_ + m_buffer_)
      {
        m_v_goal_ = std::fabs(m_v_goal_);  // 靠左端向内
      }
    }

    // 2) 限加速度逼近 v_goal
    const double DV = clampd(m_v_goal_ - m_v_, -m_A_MAX_ * dt, m_A_MAX_ * dt);
    m_v_ += DV;

    // 3) 积分位置并做端点反射
    double x_new = m_x_ + m_v_ * dt;
    if (x_new > m_max_)
    {
      x_new = m_max_ - (x_new - m_max_);
      m_v_ = -std::fabs(m_v_);
      m_v_goal_ = -std::fabs(m_v_goal_);
    }
    else if (x_new < m_min_)
    {
      x_new = m_min_ + (m_min_ - x_new);
      m_v_ = std::fabs(m_v_);
      m_v_goal_ = std::fabs(m_v_goal_);
    }
    m_x_ = clampd(x_new, m_min_, m_max_);

    // 4) 下发位置
    m_motor_->setPosition(m_x_);
  }

 private:
  Motor* m_motor_ = nullptr;
  Params m_params_;

  // 行程
  double m_min_ = 0.0, m_max_ = 0.0, m_span_ = 0.0, m_buffer_ = 0.0;

  // 内部状态
  double m_x_ = 0.0;       // 位置
  double m_v_ = 0.0;       // 速度
  double m_v_goal_ = 0.0;  // 目标速度
  double m_V_MAX_ = 0.0;   // 速度上限
  double m_A_MAX_ = 0.0;   // 最大加速度
  double m_retargetPeriod_ = 0.6;
  double m_tSinceRetarget_ = 0.0;

  // 随机
  std::mt19937 m_rng_;
  std::uniform_real_distribution<double> m_uni_;
};

// ========================== main ==========================
int main(int, char**)
{
  using namespace Config;

  Robot robot;
  const int TIMESTEP = static_cast<int>(robot.getBasicTimeStep());
  const double DT = TIMESTEP / 1000.0;

  LED* led1 = robot.getLED(LED1);
  LED* led2 = robot.getLED(LED2);
  LED* led3 = robot.getLED(LED3);
  LED* led4 = robot.getLED(LED4);

  // 基于全局默认构造每个电机参数：不同 seed
  std::vector<std::unique_ptr<RandomLinearDriver>> drivers;
  drivers.reserve(MOTOR_NAMES.size());

  for (size_t i = 0; i < MOTOR_NAMES.size(); ++i)
  {
    if (Motor* m = robot.getMotor(MOTOR_NAMES[i]))
    {
      RandomLinearDriver::Params p{};
      p.speedScale = DRIVER_DEFAULTS.speedScale;
      p.baseVmaxRatio = DRIVER_DEFAULTS.baseVmaxRatio;
      p.baseAmax = DRIVER_DEFAULTS.baseAmax;
      p.baseRetarget = DRIVER_DEFAULTS.baseRetarget;
      p.defaultMin = DRIVER_DEFAULTS.defaultMin;
      p.defaultMax = DRIVER_DEFAULTS.defaultMax;
      p.edgeBufferPct = DRIVER_DEFAULTS.edgeBufferPct;
      p.edgeBufferAbsMin = DRIVER_DEFAULTS.edgeBufferAbsMin;
      p.positionForce = DRIVER_DEFAULTS.positionForce;
      p.seed = DRIVER_DEFAULTS.baseSeed + static_cast<unsigned>(i + 1);

      // 对 pitch/yaw/roll（索引 3/4/5）放宽默认行程（仅在硬件未给 min/max 时才用到）
      if (ORIENT_RANGE_OVERRIDE.enable && (i >= 3 && i <= 5))
      {
        p.defaultMin = ORIENT_RANGE_OVERRIDE.minPos;
        p.defaultMax = ORIENT_RANGE_OVERRIDE.maxPos;
      }

      try
      {
        drivers.emplace_back(std::make_unique<RandomLinearDriver>(m, p));
      }
      catch (const std::exception& e)
      {
        std::cerr << "[ERROR] " << e.what() << " for " << MOTOR_NAMES[i] << "\n";
      }
    }
    else
    {
      std::cerr << "[ERROR] Motor not found: " << MOTOR_NAMES[i] << "\n";
    }
  }

  if (drivers.empty())
  {
    std::cerr << "[FATAL] No valid motors.\n";
    return 1;
  }

  // 主循环
  while (robot.step(TIMESTEP) != -1)
  {
    for (auto& d : drivers) d->Step(DT);

    if (BIG_ARMOR)
    {
      if (led1) led1->set(255);
      if (led2) led2->set(255);
      if (led3) led3->set(0);
      if (led4) led4->set(0);
    }
    else
    {
      if (led1) led1->set(0);
      if (led2) led2->set(0);
      if (led3) led3->set(255);
      if (led4) led4->set(255);
    }
  }
  return 0;
}
