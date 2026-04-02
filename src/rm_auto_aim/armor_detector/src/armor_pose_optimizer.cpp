#include "armor_detector/armor_pose_optimizer.hpp"

namespace rm_auto_aim
{

// ============================================================================
//  构造与设置
// ============================================================================

ArmorPoseOptimizer::ArmorPoseOptimizer(const Params& params)
    : params_(params),
      ground_pitch_rad_(params.standard_pitch_deg * M_PI / 180.0),
      outpost_pitch_rad_(params.outpost_pitch_deg * M_PI / 180.0)
{
}

void ArmorPoseOptimizer::SetCameraIntrinsics(const cv::Mat& camera_matrix,
                                             const cv::Mat& dist_coeffs)
{
  cv_camera_matrix_ = camera_matrix.clone();
  cv_dist_coeffs_ = dist_coeffs.clone();

  // 提取内参：fx, fy, cx, cy
  // camera_matrix 是 3x3 矩阵：
  //   [fx  0  cx]
  //   [0  fy  cy]
  //   [0   0   1]
  fx_ = camera_matrix.at<double>(0, 0);
  fy_ = camera_matrix.at<double>(1, 1);
  cx_ = camera_matrix.at<double>(0, 2);
  cy_ = camera_matrix.at<double>(1, 2);

  intrinsics_set_ = true;
}

void ArmorPoseOptimizer::SetCameraToGimbalRotation(const Eigen::Matrix3d& R_gimbal_cam)
{
  r_gimbal_cam_ = R_gimbal_cam;
  r_cam_gimbal_ = R_gimbal_cam.transpose();  // 正交矩阵的逆就是转置
  transform_set_ = true;
}

// ============================================================================
//  主接口
// ============================================================================

bool ArmorPoseOptimizer::Optimize(const Armor& armor, cv::Mat& rvec, cv::Mat& tvec)
{
  if (!intrinsics_set_ || !transform_set_)
  {
    return false;
  }

  // ---- Step 1: 检查先验约束 ----
  Eigen::Matrix3d r_cam = RvecToRotationMatrix(rvec);
  double yaw_init = NAN, pitch_prior = NAN;
  if (!CheckConstraint(r_cam, yaw_init, pitch_prior))
  {
    // pitch/roll 超过阈值，不适用 yaw 优化
    return false;
  }

  // ---- Step 2: 准备图像点和物体点 ----

  // 图像点：与 PnPSolver::SolvePnP 中的顺序一致
  std::array<cv::Point2f, 4> image_points = {armor.left_light.bottom,
                                             armor.left_light.top, armor.right_light.top,
                                             armor.right_light.bottom};

  // 去畸变
  auto img_points_ud = UndistortPoints(image_points);

  // 物体点：与 PnPSolver 中定义的顺序一致
  // 模型坐标系：x 前、y 左、z 上
  // 装甲板在 yz 平面上，x = 0
  std::array<Eigen::Vector3d, 4> obj_points;
  if (armor.type == ArmorType::SMALL)
  {
    // SMALL_ARMOR_POINTS 的定义顺序（从 pnp_solver.hpp）：
    //   [0]=(0, +half_y, -half_z)  左下
    //   [1]=(0, +half_y, +half_z)  左上
    //   [2]=(0, -half_y, +half_z)  右上
    //   [3]=(0, -half_y, -half_z)  右下
    constexpr double HY = SMALL_HALF_Y;
    constexpr double HZ = SMALL_HALF_Z;
    obj_points = {Eigen::Vector3d(0, HY, -HZ), Eigen::Vector3d(0, HY, HZ),
                  Eigen::Vector3d(0, -HY, HZ), Eigen::Vector3d(0, -HY, -HZ)};
  }
  else
  {
    constexpr double HY = LARGE_HALF_Y;
    constexpr double HZ = LARGE_HALF_Z;
    obj_points = {Eigen::Vector3d(0, HY, -HZ), Eigen::Vector3d(0, HY, HZ),
                  Eigen::Vector3d(0, -HY, HZ), Eigen::Vector3d(0, -HY, -HZ)};
  }

  // ---- Step 3: yaw 优化 ----
  double yaw = yaw_init;
  Eigen::Vector3d t_cam(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

  bool opt_success = false;

  switch (params_.optimize_method)
  {
    case Params::RANGE_ONLY:
    {
      // 两阶段搜索法
      opt_success = RunRangeSolve(yaw, pitch_prior, t_cam, obj_points, img_points_ud);
      break;
    }

    case Params::RANGE_THEN_LM:
    {
      // 融合法：粗搜索定位全局最优 yaw 邻域 → LM 联合精优化 yaw + tvec
      //
      // 粗搜索保证 yaw 落入正确的极值盆地，
      // LM 在此基础上联合优化 yaw 和 tvec，获得亚像素精度。
      opt_success = RunRangeSolve(yaw, pitch_prior, t_cam, obj_points, img_points_ud);
      if (opt_success)
      {
        // 以范围搜索结果为初始点运行 LM 精优化
        // 即使 LM 未能进一步改善，范围搜索的结果仍然有效
        // （RunLM 返回 false 时 yaw/t_cam 保持 range search 的值不变）
        RunLM(yaw, pitch_prior, t_cam, obj_points, img_points_ud);
      }
      break;
    }

    case Params::LM_ONLY:
    default:
    {
      // 纯 Levenberg-Marquardt 迭代法
      opt_success = RunLM(yaw, pitch_prior, t_cam, obj_points, img_points_ud);
      break;
    }
  }

  if (!opt_success)
  {
    return false;
  }

  // ---- Step 4: 写回结果 ----
  // 优化后旋转：R_cam = R_cam_gimbal * Rz(yaw) * Ry(pitch_prior)
  Eigen::Matrix3d r_cam_optimized = r_cam_gimbal_ * Rz(yaw) * Ry(pitch_prior);
  rvec = RotationMatrixToRvec(r_cam_optimized);

  tvec.at<double>(0) = t_cam(0);
  tvec.at<double>(1) = t_cam(1);
  tvec.at<double>(2) = t_cam(2);

  return true;
}

// ============================================================================
//  先验约束检查
// ============================================================================

bool ArmorPoseOptimizer::CheckConstraint(const Eigen::Matrix3d& R_cam, double& yaw_init,
                                         double& pitch_prior)
{
  // 转换到惯性系
  Eigen::Matrix3d r_gimbal = r_gimbal_cam_ * R_cam;

  // ZYX 欧拉角分解：R = Rz(yaw) * Ry(pitch) * Rx(roll)
  Eigen::Vector3d euler = DecomposeZYX(r_gimbal);  // [yaw, pitch, roll]

  double measured_pitch = euler(1);  // 弧度
  double measured_roll = euler(2);

  // roll 必须接近 0°
  double roll_deg = std::abs(measured_roll) * 180.0 / M_PI;
  if (roll_deg > params_.max_roll_deviation)
  {
    return false;
  }

  // pitch 必须接近某个已知先验值
  // 计算与两个先验的偏差，选取更近的那个
  double diff_ground = std::abs(measured_pitch - ground_pitch_rad_) * 180.0 / M_PI;
  double diff_outpost = std::abs(measured_pitch - outpost_pitch_rad_) * 180.0 / M_PI;

  if (diff_ground <= diff_outpost)
  {
    if (diff_ground > params_.max_pitch_deviation)
    {
      return false;
    }
    pitch_prior = ground_pitch_rad_;
  }
  else
  {
    if (diff_outpost > params_.max_pitch_deviation)
    {
      return false;
    }
    pitch_prior = outpost_pitch_rad_;
  }

  yaw_init = euler(0);
  return true;
}

// ============================================================================
//  去畸变
// ============================================================================

std::array<Eigen::Vector2d, 4> ArmorPoseOptimizer::UndistortPoints(
    const std::array<cv::Point2f, 4>& points)
{
  // 将 4 个点放入 vector 以便调用 cv::undistortPoints
  std::vector<cv::Point2f> distorted(points.begin(), points.end());
  std::vector<cv::Point2f> undistorted;

  // 传入 P = camera_matrix，使输出为去畸变后的像素坐标（而非归一化坐标）
  cv::undistortPoints(distorted, undistorted, cv_camera_matrix_, cv_dist_coeffs_,
                      cv::noArray(), cv_camera_matrix_);

  std::array<Eigen::Vector2d, 4> result;
  for (int i = 0; i < 4; ++i)
  {
    result[i] = Eigen::Vector2d(undistorted[i].x, undistorted[i].y);
  }
  return result;
}

// ============================================================================
//  残差与雅可比计算
// ============================================================================
//
//  旋转模型（惯性系下）：
//    R_gimbal = Rz(yaw) * Ry(pitch_prior)
//
//  相机系下：
//    R_cam = R_cam_gimbal * Rz(yaw) * Ry(pitch_prior)
//
//  3D 点变换：
//    P_cam = R_cam * P_obj + t_cam
//
//  对 yaw 的偏导数：
//    dP_cam/dyaw = R_cam_gimbal * dRz/dyaw * Ry(pitch_prior) * P_obj
//
//  对 t_cam 的偏导数：
//    dP_cam/dt = I_{3×3}
//
// ============================================================================

bool ArmorPoseOptimizer::ComputeResidualAndJacobian(
    double yaw, double pitch_prior, const Eigen::Vector3d& t_cam,
    const std::array<Eigen::Vector3d, 4>& obj_points,
    const std::array<Eigen::Vector2d, 4>& img_points_ud,
    Eigen::Matrix<double, 8, 1>& residual, Eigen::Matrix<double, 8, 4>& jacobian)
{
  // 预计算：这些矩阵在一次调用中对所有点通用
  Eigen::Matrix3d ry_prior = Ry(pitch_prior);
  Eigen::Matrix3d r_cam = r_cam_gimbal_ * Rz(yaw) * ry_prior;
  Eigen::Matrix3d d_r_cam_dyaw = r_cam_gimbal_ * DRzDyaw(yaw) * ry_prior;

  for (auto i = 0; i < 4; ++i)
  {
    // 变换到相机坐标系
    Eigen::Vector3d p_cam = r_cam * obj_points[i] + t_cam;
    double x = p_cam(0), y = p_cam(1), z = p_cam(2);

    // 点在相机后方或过近，投影无意义，会导致除零 / 数值爆炸
    if (z <= 1e-6)
    {
      return false;
    }

    double z_inv = 1.0 / z;
    double z_inv2 = z_inv * z_inv;

    // 针孔投影
    double u_proj = fx_ * x * z_inv + cx_;
    double v_proj = fy_ * y * z_inv + cy_;

    // 残差 = 观测 - 预测
    residual(2 * i + 0) = img_points_ud[i](0) - u_proj;
    residual(2 * i + 1) = img_points_ud[i](1) - v_proj;

    // ---- 雅可比矩阵 ----

    // 投影函数对 P_cam 的雅可比 (2x3)
    //   J_proj = [ fx/Z     0    -fx*X/Z² ]
    //            [   0    fy/Z   -fy*Y/Z² ]
    Eigen::Matrix<double, 2, 3> j_proj;
    j_proj << fx_ * z_inv, 0.0, -fx_ * x * z_inv2, 0.0, fy_ * z_inv, -fy_ * y * z_inv2;

    // P_cam 对 yaw 的偏导 (3×1)
    Eigen::Vector3d d_p_dyaw = d_r_cam_dyaw * obj_points[i];

    // 残差对参数 [yaw, tx, ty, tz] 的雅可比 (2×4)
    // r = obs - proj，所以 dr/dθ = -d(proj)/dθ = -J_proj * dP/dθ
    Eigen::Matrix<double, 2, 4> j_i;
    j_i.col(0) = -j_proj * d_p_dyaw;
    j_i.block<2, 3>(0, 1) = -j_proj;  // dP/dt = I

    int64_t index = static_cast<int64_t>(2) * i;
    jacobian.block<2, 4>(index, 0) = j_i;
  }

  return true;
}

// ============================================================================
//  Levenberg-Marquardt 迭代
// ============================================================================

bool ArmorPoseOptimizer::RunLM(double& yaw, double pitch_prior, Eigen::Vector3d& t_cam,
                               const std::array<Eigen::Vector3d, 4>& obj_points,
                               const std::array<Eigen::Vector2d, 4>& img_points_ud)
{
  double lambda = params_.initial_lambda;

  Eigen::Matrix<double, 8, 1> residual;
  Eigen::Matrix<double, 8, 4> jacobian;

  // 初始状态如果就有点在相机后方，直接放弃优化
  if (!ComputeResidualAndJacobian(yaw, pitch_prior, t_cam, obj_points, img_points_ud,
                                  residual, jacobian))
  {
    return false;
  }

  double cost = residual.squaredNorm();
  bool ever_succeeded = false;  // 至少一次成功

  for (int iter = 0; iter < params_.max_iterations; ++iter)
  {
    // 正规方程
    Eigen::Matrix4d h = jacobian.transpose() * jacobian;

    // g = J^T * r （注意：残差 = obs - proj，标准形式下梯度为 -J^T*r，
    //              正规方程求解的是 H*delta = -g = J^T * r）
    //              但我们的残差定义使得 delta = -(H + lambda*diag(H))^(-1) * J^T * r
    //              等价于 (H + lambda*diag(H)) * delta = -J^T * r
    Eigen::Vector4d g = jacobian.transpose() * residual;

    // 加阻尼：H_damped = H + lambda * diag(H)
    Eigen::Matrix4d h_damped = h;
    for (int k = 0; k < 4; ++k)
    {
      h_damped(k, k) += lambda * std::max(h(k, k), 1e-10);
    }

    // 求解 H_damped * delta = -g
    // 由于 g = J^T * residual，而 residual = obs - proj，
    // 更新方向应使 residual 减小，即 delta 的方向应使 proj 更接近 obs。
    // 正确的更新公式为：delta = -(H_damped)^{-1} * J^T * r
    //                          = (H_damped)^{-1} * (-g)
    // 这里 -g 对应让代价函数下降的方向。
    Eigen::Vector4d delta = h_damped.ldlt().solve(-g);

    // ---- 尝试更新 ----
    double yaw_new = yaw + delta(0);
    Eigen::Vector3d t_cam_new = t_cam + delta.tail<3>();

    // 计算新的残差和代价
    Eigen::Matrix<double, 8, 1> residual_new;
    Eigen::Matrix<double, 8, 4> jacobian_new;

    // 检查新参数是否导致 z <= 0（点落到相机后方），如果是则视为步骤失败
    if (!ComputeResidualAndJacobian(yaw_new, pitch_prior, t_cam_new, obj_points,
                                    img_points_ud, residual_new, jacobian_new))
    {
      lambda *= params_.lambda_scale_up;
      if (lambda > 1e10)
      {
        return ever_succeeded;
      }
      continue;
    }

    double cost_new = residual_new.squaredNorm();

    if (cost_new < cost)
    {
      // 步骤成功：接受更新，减小阻尼
      yaw = yaw_new;
      t_cam = t_cam_new;
      residual = residual_new;
      jacobian = jacobian_new;
      ever_succeeded = true;

      // 检查收敛条件
      double relative_cost_change = (cost - cost_new) / (cost + 1e-15);
      cost = cost_new;
      lambda /= params_.lambda_scale_down;
      // 防止 lambda 下溢
      lambda = std::max(lambda, 1e-15);

      if (delta.norm() < params_.convergence_eps ||
          relative_cost_change < params_.cost_convergence_eps)
      {
        return true;  // 收敛
      }
    }
    else
    {
      // 步骤失败：增大阻尼，不更新参数
      lambda *= params_.lambda_scale_up;

      // 阻尼过大说明已在极值点附近或问题有困难
      if (lambda > 1e10)
      {
        // 若从未成功过，说明初始点附近就无法改善，返回失败
        return ever_succeeded;
      }
    }
  }

  // 达到最大迭代次数
  // 若有过成功步骤，参数已被改善，视为成功；否则视为失败
  return ever_succeeded;
}

bool ArmorPoseOptimizer::RunRangeSolve(
    double& yaw, double pitch_prior, Eigen::Vector3d& t_cam,
    const std::array<Eigen::Vector3d, 4>& obj_points,
    const std::array<Eigen::Vector2d, 4>& img_points_ud)
{
  const double HALF_RANGE_RAD = params_.range_search_half_range_deg * M_PI / 180.0;
  const double COARSE_STEP_RAD = params_.range_search_coarse_step_deg * M_PI / 180.0;
  const double FINE_RANGE_RAD = params_.range_search_fine_range_deg * M_PI / 180.0;
  const double FINE_STEP_RAD = params_.range_search_fine_step_deg * M_PI / 180.0;

  // 以当前初始 yaw 为中心进行搜索
  const double CENTER_YAW = yaw;
  double best_yaw = yaw;
  double min_error =
      ComputeReprojectionError(yaw, pitch_prior, t_cam, obj_points, img_points_ud);

  // 初始状态如果所有点都在相机后方，直接放弃
  if (min_error >= 1e9)
  {
    return false;
  }

  // ---- Stage 1: 粗搜索 ----
  // 以初始 yaw 为中心，在 ±half_range 范围内以 coarse_step 步长搜索
  for (double candidate = CENTER_YAW - HALF_RANGE_RAD;
       candidate <= CENTER_YAW + HALF_RANGE_RAD; candidate += COARSE_STEP_RAD)
  {
    double error = ComputeReprojectionError(candidate, pitch_prior, t_cam, obj_points,
                                            img_points_ud);
    if (error < min_error)
    {
      min_error = error;
      best_yaw = candidate;
    }
  }

  // ---- Stage 2: 精搜索 ----
  // 在粗搜索最优解附近 ±fine_range 范围内以 fine_step 步长细化
  const double FINE_START = best_yaw - FINE_RANGE_RAD;
  const double FINE_END = best_yaw + FINE_RANGE_RAD;
  for (double candidate = FINE_START; candidate <= FINE_END; candidate += FINE_STEP_RAD)
  {
    double error = ComputeReprojectionError(candidate, pitch_prior, t_cam, obj_points,
                                            img_points_ud);
    if (error < min_error)
    {
      min_error = error;
      best_yaw = candidate;
    }
  }

  yaw = best_yaw;
  // 范围搜索法仅优化 yaw，t_cam 保持 PnP 原始值不变
  return true;
}

// ============================================================================
//  重投影误差计算
// ============================================================================

double ArmorPoseOptimizer::ComputeReprojectionError(
    double yaw, double pitch_prior, const Eigen::Vector3d& t_cam,
    const std::array<Eigen::Vector3d, 4>& obj_points,
    const std::array<Eigen::Vector2d, 4>& img_points_ud)
{
  // 旋转模型与 ComputeResidualAndJacobian 一致：
  //   R_cam = R_cam_gimbal * Rz(yaw) * Ry(pitch_prior)
  Eigen::Matrix3d r_cam = r_cam_gimbal_ * Rz(yaw) * Ry(pitch_prior);

  double total_error = 0.0;
  for (int i = 0; i < 4; ++i)
  {
    Eigen::Vector3d p_cam = r_cam * obj_points[i] + t_cam;
    double z = p_cam(2);

    // 点在相机后方，返回极大值表示该 yaw 无效
    if (z <= 1e-6)
    {
      return 1e9;
    }

    double u_proj = fx_ * p_cam(0) / z + cx_;
    double v_proj = fy_ * p_cam(1) / z + cy_;

    double du = img_points_ud[i](0) - u_proj;
    double dv = img_points_ud[i](1) - v_proj;
    total_error += du * du + dv * dv;
  }
  return total_error;
}

// ============================================================================
//  工具函数
// ============================================================================

Eigen::Matrix3d ArmorPoseOptimizer::RvecToRotationMatrix(const cv::Mat& rvec)
{
  cv::Mat r_cv;
  cv::Rodrigues(rvec, r_cv);
  Eigen::Matrix3d r;
  cv::cv2eigen(r_cv, r);
  return r;
}

cv::Mat ArmorPoseOptimizer::RotationMatrixToRvec(const Eigen::Matrix3d& R)
{
  cv::Mat r_cv;
  cv::eigen2cv(R, r_cv);
  cv::Mat rvec;
  cv::Rodrigues(r_cv, rvec);
  return rvec;
}

Eigen::Vector3d ArmorPoseOptimizer::DecomposeZYX(const Eigen::Matrix3d& R)
{
  // R = Rz(yaw) * Ry(pitch) * Rx(roll)
  //
  //   R(2,0) = -sin(pitch)
  //   R(2,1) =  cos(pitch)*sin(roll)
  //   R(2,2) =  cos(pitch)*cos(roll)
  //   R(1,0) =  sin(yaw)*cos(pitch)
  //   R(0,0) =  cos(yaw)*cos(pitch)

  double pitch = std::asin(-std::clamp(R(2, 0), -1.0, 1.0));
  double cos_pitch = std::cos(pitch);

  double yaw = NAN, roll = NAN;
  if (std::abs(cos_pitch) > 1e-6)
  {
    // 正常情况
    yaw = std::atan2(R(1, 0), R(0, 0));
    roll = std::atan2(R(2, 1), R(2, 2));
  }
  else
  {
    // 万向节锁（pitch ≈ ±90°），设 roll = 0
    roll = 0.0;
    yaw = std::atan2(-R(0, 1), R(1, 1));
  }

  return Eigen::Vector3d(yaw, pitch, roll);
}

Eigen::Matrix3d ArmorPoseOptimizer::Rz(double yaw)
{
  double c = std::cos(yaw), s = std::sin(yaw);
  Eigen::Matrix3d r;
  r << c, -s, 0, s, c, 0, 0, 0, 1;
  return r;
}

Eigen::Matrix3d ArmorPoseOptimizer::DRzDyaw(double yaw)
{
  double c = std::cos(yaw), s = std::sin(yaw);
  Eigen::Matrix3d d_r;
  d_r << -s, -c, 0, c, -s, 0, 0, 0, 0;
  return d_r;
}

Eigen::Matrix3d ArmorPoseOptimizer::Ry(double pitch)
{
  double c = std::cos(pitch), s = std::sin(pitch);
  Eigen::Matrix3d r;
  r << c, 0, s, 0, 1, 0, -s, 0, c;
  return r;
}

}  // namespace rm_auto_aim