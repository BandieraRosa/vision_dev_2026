#ifndef ARMOR_DETECTOR__DETECTOR_BASE_HPP_
#define ARMOR_DETECTOR__DETECTOR_BASE_HPP_

// OpenCV
#include <opencv2/core/mat.hpp>

// STD
#include <vector>

namespace rm_auto_aim
{
struct Armor;
struct DebugData;

class DetectorBase
{
 public:
  virtual ~DetectorBase() = default;
  virtual std::vector<Armor> Detect(const cv::Mat& input);
  virtual void DrawResults(cv::Mat& img);

  // debug
  bool debug_enabled = false;
  virtual const DebugData& GetDebugData();
  virtual const cv::Mat& GetBinaryImage();
  virtual const cv::Mat& GetNumbersImage();
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__DETECTOR_BASE_HPP_