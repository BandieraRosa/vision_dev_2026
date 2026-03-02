#include "armor_detector/pnp_solver.hpp"

#include <opencv2/calib3d.hpp>
#include <vector>

namespace rm_auto_aim
{
PnPSolver::PnPSolver(const std::array<double, 9>& camera_matrix,
                     const std::vector<double>& dist_coeffs)
    : camera_matrix_(cv::Mat(3, 3, CV_64F)), dist_coeffs_(cv::Mat(1, 5, CV_64F))
{
  std::memcpy(camera_matrix_.data, camera_matrix.data(), 9 * sizeof(double));
  size_t num_dist_coeffs = std::min(dist_coeffs.size(), static_cast<size_t>(5));
  std::memcpy(dist_coeffs_.data, dist_coeffs.data(), num_dist_coeffs * sizeof(double));
}

bool PnPSolver::SolvePnP(const Armor& armor, cv::Mat& rvec, cv::Mat& tvec)
{
  std::array<cv::Point2f, 4> image_armor_points = {
      armor.left_light.bottom, armor.left_light.top, armor.right_light.top,
      armor.right_light.bottom};

  // Solve pnp
  auto object_points =
      armor.type == ArmorType::SMALL ? SMALL_ARMOR_POINTS : LARGE_ARMOR_POINTS;

  std::vector<cv::Mat> rvecs, tvecs;
  std::vector<double> re_projection_error;
  int solutions = cv::solvePnPGeneric(object_points, image_armor_points, camera_matrix_,
                                      dist_coeffs_, rvecs, tvecs, false,
                                      cv::SOLVEPNP_IPPE,  // 使用 IPPE 算法获取多个解
                                      cv::noArray(), cv::noArray(), re_projection_error);

  if (solutions == 0)
  {
    return false;
  }

  double z_data[3]{0, 0, 10};
  cv::Mat z_vector(cv::Size(1, 3), CV_64FC1, z_data);

  cv::Mat r_0, r_1;
  cv::Rodrigues(rvecs.front(), r_0);
  cv::Rodrigues(rvecs.back(), r_1);

  cv::Mat z_camera_0 = r_0 * z_vector + tvecs.front();
  cv::Mat z_camera_1 = r_1 * z_vector + tvecs.back();

  cv::Mat r, t;
  if (z_camera_0.at<double>(2, 0) > 0)
  {
    rvec = rvecs.front();
    tvec = tvecs.front();
  }
  else
  {
    rvec = rvecs.back();
    tvec = tvecs.back();
  }
  return true;
}

float PnPSolver::CalculateDistanceToCenter(
    const cv::Point2f& image_point)  // 计算给定图像点到图像中心的距离
{
  float cx = static_cast<float>(camera_matrix_.at<double>(0, 2));
  float cy = static_cast<float>(camera_matrix_.at<double>(1, 2));
  return static_cast<float>(cv::norm(image_point - cv::Point2f(cx, cy)));
}

}  // namespace rm_auto_aim
