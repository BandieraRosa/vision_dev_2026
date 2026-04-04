#ifndef TRAJECTORY_HPP
#define TRAJECTORY_HPP

#include <cmath>
#include <algorithm>

#include "planning_trajectory/trajectory_solver.hpp"
#include "armor_tracker/extended_kalman_filter.hpp"

namespace rm_auto_aim
{

class ConstrainedPlanner
{
public:
  ConstrainedPlanner(double wn, double v_max, double a_max, double dt)
      : wn_(wn), v_max_(v_max), a_max_(a_max), dt_(dt)
  {
  }

  void setState(double theta0, double v0);

  double update(double target_pos, double target_vel, double target_acc);


private:

  // ---- 参数 ----
  double wn_;      // 自然频率 (rad/s)
  double v_max_;   // 最大角速度 (°/s)
  double a_max_;   // 最大角加速度 (°/s²)
  double dt_;      // 更新周期 (s)

  // ---- 状态 ----
  double theta_s_ = 0.0;   // 规划器输出角度
  double v_s_ = 0.0;       // 规划器输出速度
  double a_cmd_ = 0.0;     // 规划器输出加速度
};



class Trajectory
{
 public:
  Trajectory();
  ~Trajectory();
  ExtendedKalmanFilter ekf_;
  ConstrainedPlanner planner_;
  void UpdatePlanTrajectory(TrajectorySolver::control& cmd, const int init_phase);
};

}  // namespace rm_auto_aim

#endif  // TRAJECTORY_HPP
  