#ifndef ARMOR_DETECTOR__DETECTOR_BASE_HPP_
#define ARMOR_DETECTOR__DETECTOR_BASE_HPP_

#include <opencv2/core/mat.hpp>

#include "armor_detector/detection_result.hpp"

namespace rm_auto_aim
{

class DetectorBase
{
 public:
  virtual ~DetectorBase() = default;
  virtual DetectionResult Detect(const cv::Mat& input) = 0;
  virtual void DrawResults(cv::Mat& img) = 0;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__DETECTOR_BASE_HPP_
