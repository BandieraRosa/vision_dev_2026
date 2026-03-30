#ifndef ARMOR_DETECTOR__LIGHT_CORNER_CORRECTOR_HPP_
#define ARMOR_DETECTOR__LIGHT_CORNER_CORRECTOR_HPP_

#include <opencv2/core.hpp>

#include "armor.hpp"

namespace rm_auto_aim
{

struct SymmetryAxis
{
  cv::Point2f centroid;
  cv::Point2f direction;
  double mean_val;  // 平均亮度
};

// 此类用于提高灯条角点的精度。
// 首先使用 PCA（主成分分析）找到灯条的对称轴，
// 然后沿对称轴根据亮度梯度查找灯条的角点。
class LightCornerCorrector
{
 public:
  explicit LightCornerCorrector() noexcept {}

  // 修正装甲灯条的角点
  void CorrectCorners(Armor &armor, const cv::Mat &gray_img);

 private:
  // 查找灯条的对称轴
  SymmetryAxis FindSymmetryAxis(const cv::Mat &gray_img, const Light &light);

  // 沿对称轴查找灯条的角点
  cv::Point2f FindCorner(const cv::Mat &gray_img, const Light &light,
                         const SymmetryAxis &axis, const std::string &order);
};

};  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__LIGHT_CORNER_CORRECTOR_HPP_