#include "planning_trajectory/trajectory.hpp"

namespace rm_auto_aim
{

void ConstrainedPlanner::setState(double theta0, double v0 = 0.0)
{
  theta_s_ = theta0;
  v_s_ = v0;
  a_cmd_ = 0.0;
}

double ConstrainedPlanner::update(double target_pos, double target_vel = 0.0,
                                  double target_acc = 0.0)
{
  double e = target_pos - theta_s_;  // 位置误差
  double ve = v_s_ - target_vel;     // 速度误差

  double d_brake = (ve * ve) / (2.0 * a_max_);  // 制动距离

  // 正在接近目标 && 速度超出目标速度
  bool approaching = (e > 0.0 && v_s_ > target_vel) || (e < 0.0 && v_s_ < target_vel);
  bool too_close = std::abs(e) <= d_brake;

  
  // 设置速度死区，防止不必要的抖动
  double vel_thresh = std::abs(target_vel) * 0.001 + 1.0;
  // 速度方向错了，全力制动归零
  bool wrong_direction =
      (e > 0.0 && v_s_ < -vel_thresh) || (e < 0.0 && v_s_ > vel_thresh);

  double a_desired = 0.0;

  if (approaching && too_close)
  {
    // a = (v_target² - v_s²) / (2·distance)
    if (std::abs(e) > 1e-6)
    {
      a_desired = (target_vel * target_vel - v_s_ * v_s_) / (2.0 * e);
    }
    else
    {
      a_desired = -std::copysign(a_max_, ve);
    }
  }
  else if (wrong_direction)
  {
    // 速度方向错了，全力制动归零
    a_desired = -std::copysign(a_max_, v_s_);
  }
  else
  {
    // a = ωn²·e - 2ωn·(v_s - v_target) + a_target
    a_desired = wn_ * wn_ * e - 2.0 * wn_ * ve + target_acc;
  }

  a_desired = std::clamp(a_desired, -a_max_, a_max_);

  double v_old = v_s_;
  v_s_ += a_desired * dt_;

  v_s_ = std::clamp(v_s_, -v_max_, v_max_);

  a_cmd_ = (v_s_ - v_old) / dt_;

  theta_s_ += v_s_ * dt_;

  return theta_s_;
}

void Trajectory::UpdatePlanTrajectory(TrajectorySolver::control& cmd,
                                      const int init_phase)
{
  if (init_phase == 0)
  {
    Eigen::VectorXd x0(4);
    x0 << cmd.yaw, 0, 0, cmd.pitch;
    ekf_.setState(x0);
  }
  else if (init_phase == 1)
  {
    Eigen::VectorXd z(2);
    z << cmd.yaw, cmd.pitch;
    Eigen::VectorXd x_post = ekf_.update(z);
    planner_.setState(x_post(0), x_post(1));
  }
  else
  {
    Eigen::VectorXd z(2);
    z << cmd.yaw, cmd.pitch;
    Eigen::VectorXd x(4);
    x = ekf_.update(z);
    cmd.yaw = planner_.update(x[0], x[1], x[2]);
    cmd.vel_yaw = x[1];
    cmd.acc_yaw = x[2];
    cmd.pitch = x[3];
  }
}
}  // namespace rm_auto_aim
