#include "armor_tracker/extended_kalman_filter.hpp"

namespace rm_auto_aim
{
/*
f:过程函数
h:观测函数
j_f:过程函数的雅可比矩阵
j_h:测量函数的雅可比矩阵
u_q:过程噪声协方差矩阵
u_r:测量噪声协方差矩阵
P0:初始状态协方差矩阵
*/
ExtendedKalmanFilter::ExtendedKalmanFilter(const VecVecFunc& f, const VecVecFunc& h,
                                           const VecMatFunc& j_f, const VecMatFunc& j_h,
                                           const VoidMatFunc& u_q, const VecMatFunc& u_r,
                                           const Eigen::MatrixXd& P0)
    : f(f),
      h(h),
      jacobian_f(j_f),
      jacobian_h(j_h),
      update_Q(u_q),
      update_R(u_r),
      P_post(P0),
      n(P0.rows()),
      I(Eigen::MatrixXd::Identity(n, n)),
      x_pri(n),
      x_post(n)
{
}

void ExtendedKalmanFilter::setState(const Eigen::VectorXd& x0) { x_post = x0; }

Eigen::MatrixXd ExtendedKalmanFilter::predict()
{
  F = jacobian_f(x_post), Q = update_Q();

  x_pri = f(x_post);
  P_pri = F * P_post * F.transpose() + Q;

  // handle the case when there will be no measurement before the next predict
  x_post = x_pri;
  P_post = P_pri;

  return x_pri;
}

Eigen::MatrixXd ExtendedKalmanFilter::update(const Eigen::VectorXd& z)
{
  H = jacobian_h(x_pri), R = update_R(x_pri);

  K = P_pri * H.transpose() *
      (H * P_pri * H.transpose() + R).inverse();  // inverse计算逆矩阵
  x_post = x_pri + K * (z - h(x_pri));
  P_post = (I - K * H) * P_pri;

  Eigen::VectorXd innovation = z - h(x_pri);
  Eigen::MatrixXd s = H * P_pri * H.transpose() + R;
  double nis = innovation.transpose() * s.inverse() * innovation;
  nis_window_.push_back(nis);
  if (nis_window_.size() > 100)
  {
    nis_window_.pop_front();
  }

  return x_post;
}

Eigen::VectorXd ExtendedKalmanFilter::getState()
{
  return x_post;
}

double ExtendedKalmanFilter::GetHealthRate()
{
  if (nis_window_.size() < 20)
  {
    return 1.0;
  }
  int health = static_cast<int>(std::count_if(nis_window_.begin(), nis_window_.end(),
                                            [](double v)
                                            { return v < 9.49; }));  // chi2(4, 0.05)
  return static_cast<double>(health) / nis_window_.size();
}

}  // namespace rm_auto_aim