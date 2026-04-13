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
  std::vector<std::pair<std::string, uint64_t>> debug_latencies_;  // 各环节延迟
  const std::vector<std::pair<std::string, uint64_t>>& GetDebugLatencies() const
  {
    return debug_latencies_;
  }
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__DETECTOR_BASE_HPP_
