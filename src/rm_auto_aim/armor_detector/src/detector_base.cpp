#include "armor_detector/detector_base.hpp"

#include "armor_detector/detector.hpp"

namespace rm_auto_aim
{

std::vector<Armor> DetectorBase::Detect(const cv::Mat&) { return {}; }

void DetectorBase::DrawResults(cv::Mat&) {}

const DebugData& DetectorBase::GetDebugData()
{
  static DebugData data;
  return data;
}

const cv::Mat& DetectorBase::GetBinaryImage()
{
  static cv::Mat img;
  return img;
}

const cv::Mat& DetectorBase::GetNumbersImage()
{
  static cv::Mat img;
  return img;
}

}  // namespace rm_auto_aim
