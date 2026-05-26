#ifndef ARMOR_DETECTOR__MODEL_ADAPTER_HPP_
#define ARMOR_DETECTOR__MODEL_ADAPTER_HPP_

#include <opencv2/core.hpp>
#include <vector>

#include "armor_detector/armor.hpp"
#include "armor_detector/inference_backend.hpp"

namespace rm_auto_aim
{

class ModelAdapter
{
 public:
  virtual ~ModelAdapter() = default;

  virtual std::vector<Armor> Decode(const YoloInferenceOutput& output) = 0;
  virtual void DrawResults(cv::Mat& img, const std::vector<Armor>& armors) const = 0;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__MODEL_ADAPTER_HPP_
