#ifndef LIGHT_CORNER__CORRECTOR_HPP_
#define LIGHT_CORNER__CORRECTOR_HPP_

#include <opencv2/core.hpp>

#include "armor.hpp"

namespace rm_auto_aim
{

struct SymmetryAxis
{
  cv::Point2f centroid;
  cv::Point2f direction;
  double mean_val;  // Mean brightness
};

// This class is used to improve the precision of the corner points of the light bar.
// First, the PCA algorithm is used to find the symmetry axis of the light bar,
// and then along the symmetry axis to find the corner points of the light bar based on
// the gradient of brightness.
class LightCornerCorrector
{
 public:
  explicit LightCornerCorrector() noexcept {}

  // Correct the corners of the armor's lights
  void CorrectCorners(Armor &armor, const cv::Mat &gray_img);

 private:
  // Find the symmetry axis of the light
  SymmetryAxis FindSymmetryAxis(const cv::Mat &gray_img, const Light &light);

  // Find the corner of the light
  cv::Point2f FindCorner(const cv::Mat &gray_img, const Light &light,
                         const SymmetryAxis &axis, const std::string &order);
};

};  // namespace rm_auto_aim

#endif  // LIGHT_CORNER__CORRECTOR_HPP_