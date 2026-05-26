#ifndef ARMOR_DETECTOR__INFERENCE_BACKEND_HPP_
#define ARMOR_DETECTOR__INFERENCE_BACKEND_HPP_

#include <cstdint>
#include <opencv2/core.hpp>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "armor_detector/yolo_common.hpp"

namespace rm_auto_aim
{

enum class YoloOutputLayout : uint8_t
{
  RAW,
  END_TO_END
};

struct YoloEndToEndOutput
{
  int count = 0;
  int kpt_channels = 0;
  std::span<const float> scores;
  std::span<const int> classes;
  std::span<const float> keypoints;
};

struct YoloInferenceOutput
{
  YoloOutputLayout layout = YoloOutputLayout::RAW;
  double scale = 1.0;
  cv::Mat raw_output;
  YoloEndToEndOutput end_to_end;
  std::vector<std::pair<std::string, std::uint64_t>> latencies;
};

class InferenceBackend
{
 public:
  virtual ~InferenceBackend() = default;

  virtual YoloBackendKind BackendKind() const noexcept = 0;
  virtual YoloInferenceOutput Infer(const cv::Mat& rgb_img) = 0;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__INFERENCE_BACKEND_HPP_
