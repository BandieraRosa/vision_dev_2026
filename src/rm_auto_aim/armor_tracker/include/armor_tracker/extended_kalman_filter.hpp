#ifndef ARMOR_PROCESSOR__KALMAN_FILTER_HPP_
#define ARMOR_PROCESSOR__KALMAN_FILTER_HPP_

#include <Eigen/Dense>
#include <functional>

namespace rm_auto_aim
{

class EKF
{
 public:
  // CV模型：9维状态向量（去除加速度项）
  enum STATE : std::uint8_t
  {
    X_CENTER = 0,
    V_X_CENTER = 1,
    Y_CENTER = 2,
    V_Y_CENTER = 3,
    Z_ARMOR = 4,
    V_Z_ARMOR = 5,
    YAW = 6,
    V_YAW = 7,
    ROBOT_R = 8,
  };

  using VecVecFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;
  using VecMatFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>;
  using VoidMatFunc = std::function<Eigen::MatrixXd()>;

  EKF() = default;

  explicit EKF(const VecVecFunc& f, const VecVecFunc& h, const VecMatFunc& j_f,
               const VecMatFunc& j_h, const VoidMatFunc& u_q, const VecMatFunc& u_r,
               const Eigen::MatrixXd& p_0);

  // 设置初始状态
  void SetState(const Eigen::VectorXd& x0);
  // 获取当前状态
  Eigen::VectorXd GetState() const;
  // 设置状态和协方差矩阵
  void SetStateWithUncertainty(const Eigen::VectorXd& x0, const Eigen::VectorXd& diagP);

  // 获取当前协方差矩阵
  const Eigen::MatrixXd GetCovariance() const;
  Eigen::MatrixXd GetCovariance();
  // 设置协方差矩阵
  void SetCovariance(const Eigen::MatrixXd& p);
  // 打印协方差矩阵
  void PrintCovariance() const;

  // 获取过程函数
  VecVecFunc GetObservation() const;
  // 获取状态转移函数
  VecVecFunc GetStateTransition() const;

  // 将先验状态转换为后验状态
  void PriToPost();

  // 预测
  Eigen::MatrixXd Predict();

  // 更新
  Eigen::MatrixXd Update(const Eigen::VectorXd& z);

 private:
  // 过程函数
  VecVecFunc f_;
  // 观测函数
  VecVecFunc h_;
  // 过程函数的雅可比矩阵
  VecMatFunc jacobian_f_;
  Eigen::MatrixXd m_f_;
  // 观测函数的雅可比矩阵
  VecMatFunc jacobian_h_;
  Eigen::MatrixXd m_h_;
  // 过程噪声协方差矩阵
  VoidMatFunc update_q_;
  Eigen::MatrixXd m_q_;
  // 测量噪声协方差矩阵
  VecMatFunc update_r_;
  Eigen::MatrixXd m_r_;

  // 先验状态协方差矩阵
  Eigen::MatrixXd p_pri_;
  // 后验状态协方差矩阵
  Eigen::MatrixXd p_post_;

  // 卡尔曼增益
  Eigen::MatrixXd k_;

  // 状态维度
  int dimensions_;

  // 单位矩阵
  Eigen::MatrixXd i_;

  // 先验状态
  Eigen::VectorXd x_pri_;
  // 后验状态
  Eigen::VectorXd x_post_;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_PROCESSOR__KALMAN_FILTER_HPP_
