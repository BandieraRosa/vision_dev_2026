#ifndef ARMOR_DETECTOR__ARMOR_POSE_OPTIMIZER_HPP_
#define ARMOR_DETECTOR__ARMOR_POSE_OPTIMIZER_HPP_

#include <Eigen/Dense>
#include <array>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include "armor_detector/armor.hpp"

namespace rm_auto_aim
{

class ArmorPoseOptimizer
{
 public:
  struct Params
  {
    // 地面兵种装甲板安装 pitch
    double standard_pitch_deg = 15.0;
    // 前哨站装甲板安装 pitch
    double outpost_pitch_deg = -15.0;

    // 先验约束容差（单位：度）
    // pitch 偏差超过此值时认为先验不可用（如上坡、飞坡）
    double max_pitch_deviation = 10.0;
    // roll 偏差超过此值时认为先验不可用
    double max_roll_deviation = 10.0;

    // ---- 优化方法选择 ----
    enum class OptimizeMethod : uint8_t
    {
      LM = 0,
      RANGE = 1,
      RANGE_LM = 2,  // 在范围搜索基础上以最优解为起点再跑 LM
    };
    OptimizeMethod optimize_method = OptimizeMethod::RANGE_LM;

    // LM 优化参数
    int max_iterations = 20;             // 最大迭代次数
    double convergence_eps = 1e-6;       // 收敛阈值：参数更新量的范数
    double cost_convergence_eps = 1e-8;  // 收敛阈值：代价函数相对变化量
    double initial_lambda = 1e-3;        // LM 初始阻尼因子
    double lambda_scale_up = 10.0;       // 步骤失败时阻尼放大倍数
    double lambda_scale_down = 10.0;     // 步骤成功时阻尼缩小倍数

    // 范围搜索参数
    double range_search_half_range_deg =
        70.0;  // 粗搜索半范围（°），以初始 yaw 为中心 ±70°
    double range_search_coarse_step_deg = 1.0;  // 粗搜索步长（°）
    double range_search_fine_range_deg =
        2.0;                                  // 精搜索半范围（°），在粗搜索最优解附近 ±2°
    double range_search_fine_step_deg = 0.1;  // 精搜索步长（°）
  };

  explicit ArmorPoseOptimizer(const Params& params);

  /// @brief 设置相机内参（从 sensor_msgs::msg::CameraInfo 中提取）
  /// @param camera_matrix 3x3 相机内参矩阵 (K)
  /// @param dist_coeffs 畸变系数向量
  void SetCameraIntrinsics(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs);

  /// @brief 设置从相机光学坐标系(camera_optical_frame)到惯性系(gimbal_odom)的旋转
  ///
  /// 此变换可通过 tf2 查询获得：
  ///   auto tf = tf_buffer.lookupTransform("gimbal_odom", camera_frame_id, ...);
  ///   然后从 tf.transform.rotation 中提取旋转矩阵。
  ///
  /// @param R_gimbal_cam 从相机系到惯性系的旋转矩阵
  void SetCameraToGimbalRotation(const Eigen::Matrix3d& R_gimbal_cam);

  /// @brief 对初始 6-DOF PnP 结果进行 yaw 约束优化
  ///
  /// 工作流程：
  ///   1. 将初始旋转转到惯性系，分解 ZYX 欧拉角
  ///   2. 检查 pitch 是否接近已知先验（±15°），roll 是否接近 0°
  ///   3. 若满足，锁定 pitch=先验值, roll=0，以初始 yaw 和 tvec 为起点运行 LM
  ///   4. 输出优化后的 rvec, tvec（仍在相机系下）
  ///
  /// @param armor   检测到的装甲板
  /// @param rvec    [in/out] 旋转向量（相机系，Rodrigues）
  /// @param tvec    [in/out] 平移向量（相机系）
  /// @return true 表示优化成功；false 表示先验不满足或优化失败
  bool Optimize(const Armor& armor, cv::Mat& rvec, cv::Mat& tvec);

 private:
  // ---- 旋转矩阵工具 ----

  /// 将 cv::Mat 旋转向量（Rodrigues）转为 3x3 旋转矩阵
  static Eigen::Matrix3d RvecToRotationMatrix(const cv::Mat& rvec);

  /// 将 3x3 旋转矩阵转为 cv::Mat 旋转向量（Rodrigues）
  static cv::Mat RotationMatrixToRvec(const Eigen::Matrix3d& R);

  /// ZYX 欧拉角分解：R = Rz(yaw) * Ry(pitch) * Rx(roll)
  /// @return [yaw, pitch, roll]（弧度）
  static Eigen::Vector3d DecomposeZYX(const Eigen::Matrix3d& R);

  /// 构造绕 z 轴的旋转矩阵
  static Eigen::Matrix3d Rz(double yaw);

  /// Rz 对 yaw 的解析导数
  static Eigen::Matrix3d DRzDyaw(double yaw);

  /// 绕 y 轴旋转矩阵（装甲板安装 pitch）
  static Eigen::Matrix3d Ry(double pitch);

  /// 根据装甲板类型构造 4 个物体顶点坐标
  /// 模型坐标系：x 前、y 左、z 上，装甲板在 yz 平面上（x = 0）
  /// 顺序：左下、左上、右上、右下（与 PnPSolver 一致）
  static std::array<Eigen::Vector3d, 4> ExtractObjectPoints(ArmorType type);

  /// 构造相机系下的旋转矩阵：R_cam = R_cam_gimbal * Rz(yaw) * Ry(pitch_prior)
  Eigen::Matrix3d BuildCameraRotation(double yaw, double pitch_prior) const;

  /// 将 4 个物体点经刚体变换后做针孔投影，输出像素坐标
  /// @return false 若存在 z <= 1e-6 的退化点（点在相机后方）
  bool ProjectPoints(const Eigen::Matrix3d& R_cam, const Eigen::Vector3d& t_cam,
                     const std::array<Eigen::Vector3d, 4>& obj_points,
                     std::array<Eigen::Vector2d, 4>& projected) const;

  /// 计算给定 (yaw, pitch_prior, t_cam) 下的重投影误差平方和
  /// @return 重投影误差平方和；若有点在相机后方则返回 1e9
  double ComputeReprojectionError(double yaw, double pitch_prior,
                                  const Eigen::Vector3d& t_cam,
                                  const std::array<Eigen::Vector3d, 4>& obj_points,
                                  const std::array<Eigen::Vector2d, 4>& img_points_ud);

  // ---- 优化核心 ----

  /// 检查先验约束是否满足，并提取初始 yaw 和匹配到的 pitch 先验
  /// @param R_cam       相机系下的旋转矩阵（初始 PnP）
  /// @param yaw_init    [out] 惯性系下的初始 yaw
  /// @param pitch_prior [out] 匹配到的装甲板安装 pitch 先验（弧度）
  /// @return true 表示先验约束满足
  bool CheckConstraint(const Eigen::Matrix3d& R_cam, double& yaw_init,
                       double& pitch_prior);

  /// 去畸变图像点，返回去畸变后的像素坐标
  std::array<Eigen::Vector2d, 4> UndistortPoints(
      const std::array<cv::Point2f, 4>& points);

  /// 计算残差与雅可比
  /// @return false 表示存在 z <= 0 的退化情况（点在相机后方），结果不可用
  bool ComputeResidualAndJacobian(double yaw, double pitch_prior,
                                  const Eigen::Vector3d& t_cam,
                                  const std::array<Eigen::Vector3d, 4>& obj_points,
                                  const std::array<Eigen::Vector2d, 4>& img_points_ud,
                                  Eigen::Matrix<double, 8, 1>& residual,
                                  Eigen::Matrix<double, 8, 4>& jacobian);

  /// LM 迭代求解
  bool RunLM(double& yaw, double pitch_prior, Eigen::Vector3d& t_cam,
             const std::array<Eigen::Vector3d, 4>& obj_points,
             const std::array<Eigen::Vector2d, 4>& img_points_ud);

  bool RunRangeSolve(double& yaw, double pitch_prior, Eigen::Vector3d& t_cam,
                     const std::array<Eigen::Vector3d, 4>& obj_points,
                     const std::array<Eigen::Vector2d, 4>& img_points_ud,
                     double* best_error_out = nullptr);

  /// 固定旋转矩阵，解析求解最优平移向量
  ///
  /// 数学原理：对于固定 R = R_cam_gimbal * Rz(yaw) * Ry(pitch_prior)，
  /// 令 q_i = R * P_obj_i，投影方程展开后关于 t = (tx, ty, tz) 为线性系统：
  ///   [fx   0   -(u'_i - cx)] [tx]   [(u'_i - cx)*qz_i - fx*qx_i]
  ///   [0    fy  -(v'_i - cy)] [ty] = [(v'_i - cy)*qz_i - fy*qy_i]
  ///                            [tz]
  /// 4 个点产生 8 个方程，通过 3×3 正规方程 A^T A t = A^T b 求解。
  ///
  /// @param rotated_points 预计算的 R * P_obj_i
  /// @return false 若线性系统退化（不应在正常装甲板几何下发生）
  bool SolveTranslationForFixedRotation(
      const std::array<Eigen::Vector3d, 4>& rotated_points,
      const std::array<Eigen::Vector2d, 4>& img_points_ud, Eigen::Vector3d& t_out) const;

  /// 评估候选 yaw：解析求解最优平移后计算重投影误差
  /// @return 重投影误差平方和；若退化（线性系统奇异 / z <= 0）则返回 1e9
  double EvaluateCandidateYaw(double yaw, double pitch_prior,
                              const std::array<Eigen::Vector3d, 4>& obj_points,
                              const std::array<Eigen::Vector2d, 4>& img_points_ud,
                              Eigen::Vector3d& t_out);

  // Unit: mm
  static constexpr float SMALL_ARMOR_WIDTH = 135;
  static constexpr float SMALL_ARMOR_HEIGHT = 55;
  static constexpr float LARGE_ARMOR_WIDTH = 230;
  static constexpr float LARGE_ARMOR_HEIGHT = 55;

  // Four vertices of armor in 3d
  // Unit: m
  static constexpr double SMALL_HALF_Y = SMALL_ARMOR_WIDTH / 2.0 / 1000.0;
  static constexpr double SMALL_HALF_Z = SMALL_ARMOR_HEIGHT / 2.0 / 1000.0;
  static constexpr double LARGE_HALF_Y = LARGE_ARMOR_WIDTH / 2.0 / 1000.0;
  static constexpr double LARGE_HALF_Z = LARGE_ARMOR_HEIGHT / 2.0 / 1000.0;

  // ---- 成员变量 ----
  Params params_;

  // 装甲板安装 pitch 先验值（弧度制，构造时转换）
  double ground_pitch_rad_;
  double outpost_pitch_rad_;

  // 相机内参
  cv::Mat cv_camera_matrix_;
  cv::Mat cv_dist_coeffs_;
  double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0;
  bool intrinsics_set_ = false;

  // 坐标系变换
  Eigen::Matrix3d r_gimbal_cam_;  // 相机系 → 惯性系 的旋转
  Eigen::Matrix3d r_cam_gimbal_;  // 惯性系 → 相机系 的旋转（= R_gimbal_cam_^T）
  bool transform_set_ = false;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__ARMOR_POSE_OPTIMIZER_HPP_