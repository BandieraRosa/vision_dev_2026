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

  // Unit: m
  constexpr double SMALL_HALF_Y = SMALL_ARMOR_WIDTH / 2.0 / 1000.0;
  constexpr double SMALL_HALF_Z = SMALL_ARMOR_HEIGHT / 2.0 / 1000.0;
  constexpr double LARGE_HALF_Y = LARGE_ARMOR_WIDTH / 2.0 / 1000.0;
  constexpr double LARGE_HALF_Z = LARGE_ARMOR_HEIGHT / 2.0 / 1000.0;

  // Start from bottom left in clockwise order
  // Model coordinate: x forward, y left, z up
  small_armor_points_.emplace_back(cv::Point3f(0, SMALL_HALF_Y, -SMALL_HALF_Z));
  small_armor_points_.emplace_back(cv::Point3f(0, SMALL_HALF_Y, SMALL_HALF_Z));
  small_armor_points_.emplace_back(cv::Point3f(0, -SMALL_HALF_Y, SMALL_HALF_Z));
  small_armor_points_.emplace_back(cv::Point3f(0, -SMALL_HALF_Y, -SMALL_HALF_Z));

  large_armor_points_.emplace_back(cv::Point3f(0, LARGE_HALF_Y, -LARGE_HALF_Z));
  large_armor_points_.emplace_back(cv::Point3f(0, LARGE_HALF_Y, LARGE_HALF_Z));
  large_armor_points_.emplace_back(cv::Point3f(0, -LARGE_HALF_Y, LARGE_HALF_Z));
  large_armor_points_.emplace_back(cv::Point3f(0, -LARGE_HALF_Y, -LARGE_HALF_Z));
}

bool PnPSolver::SolvePnP(const Armor& armor, cv::Mat& rvec, cv::Mat& tvec)
{
  std::vector<cv::Point2f> image_armor_points;

  // Fill in image points
  image_armor_points.emplace_back(armor.left_light.bottom);
  image_armor_points.emplace_back(armor.left_light.top);
  image_armor_points.emplace_back(armor.right_light.top);
  image_armor_points.emplace_back(armor.right_light.bottom);

  // Solve pnp
  auto object_points =
      armor.type == ArmorType::SMALL ? small_armor_points_ : large_armor_points_;
  return cv::solvePnP(object_points, image_armor_points, camera_matrix_, dist_coeffs_,
                      rvec, tvec, false, cv::SOLVEPNP_IPPE);
}

float PnPSolver::CalculateDistanceToCenter(
    const cv::Point2f& image_point)  // 计算给定图像点到图像中心的距离
{
  float cx = static_cast<float>(camera_matrix_.at<double>(0, 2));
  float cy = static_cast<float>(camera_matrix_.at<double>(1, 2));
  return static_cast<float>(cv::norm(image_point - cv::Point2f(cx, cy)));
}

}  // namespace rm_auto_aim
