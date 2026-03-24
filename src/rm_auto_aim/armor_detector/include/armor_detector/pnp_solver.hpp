#ifndef ARMOR_DETECTOR__PNP_SOLVER_HPP_
#define ARMOR_DETECTOR__PNP_SOLVER_HPP_

#include <geometry_msgs/msg/point.hpp>
#include <opencv2/core.hpp>

// STD
#include <array>
#include <vector>

#include "armor_detector/armor.hpp"

namespace rm_auto_aim
{
class PnPSolver
{
 public:
  PnPSolver(const std::array<double, 9>& camera_matrix,
            const std::vector<double>& distortion_coefficients);

  // Get 3d position
  bool SolvePnP(const Armor& armor, cv::Mat& rvec, cv::Mat& tvec);

  // Calculate the distance between armor center and image center
  float CalculateDistanceToCenter(const cv::Point2f& image_point);

  void SetCameraInfo(const std::array<double, 9>& camera_matrix,
                     const std::vector<double>& distortion_coefficients);

 private:
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;

  // Unit: mm
  static constexpr float SMALL_ARMOR_WIDTH = 135;
  static constexpr float SMALL_ARMOR_HEIGHT = 55;
  static constexpr float LARGE_ARMOR_WIDTH = 225;
  static constexpr float LARGE_ARMOR_HEIGHT = 55;

  // Four vertices of armor in 3d
  // Unit: m
  static constexpr double SMALL_HALF_Y = SMALL_ARMOR_WIDTH / 2.0 / 1000.0;
  static constexpr double SMALL_HALF_Z = SMALL_ARMOR_HEIGHT / 2.0 / 1000.0;
  static constexpr double LARGE_HALF_Y = LARGE_ARMOR_WIDTH / 2.0 / 1000.0;
  static constexpr double LARGE_HALF_Z = LARGE_ARMOR_HEIGHT / 2.0 / 1000.0;

  // Start from bottom left in clockwise order
  // Model coordinate: x forward, y left, z up
  static const std::array<cv::Point3f, 4> SMALL_ARMOR_POINTS;
  static const std::array<cv::Point3f, 4> LARGE_ARMOR_POINTS;
};

inline const std::array<cv::Point3f, 4> PnPSolver::SMALL_ARMOR_POINTS = {{
    {0, SMALL_HALF_Y, -SMALL_HALF_Z},
    {0, SMALL_HALF_Y, SMALL_HALF_Z},
    {0, -SMALL_HALF_Y, SMALL_HALF_Z},
    {0, -SMALL_HALF_Y, -SMALL_HALF_Z},
}};

inline const std::array<cv::Point3f, 4> PnPSolver::LARGE_ARMOR_POINTS = {{
    {0, LARGE_HALF_Y, -LARGE_HALF_Z},
    {0, LARGE_HALF_Y, LARGE_HALF_Z},
    {0, -LARGE_HALF_Y, LARGE_HALF_Z},
    {0, -LARGE_HALF_Y, -LARGE_HALF_Z},
}};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__PNP_SOLVER_HPP_
