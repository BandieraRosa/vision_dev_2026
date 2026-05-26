#ifndef ARMOR_DETECTOR__YOLO11_ADAPTER_HPP_
#define ARMOR_DETECTOR__YOLO11_ADAPTER_HPP_

#include <opencv2/core.hpp>
#include <vector>

#include "armor_detector/model_adapter.hpp"
#include "armor_detector/yolo_common.hpp"

namespace rm_auto_aim
{

class Yolo11Adapter final : public ModelAdapter
{
 public:
  Yolo11Adapter(const YoloParams& params, YoloBackendKind backend_kind);

  std::vector<Armor> Decode(const YoloInferenceOutput& output) override;
  void DrawResults(cv::Mat& img, const std::vector<Armor>& armors) const override;

 private:
  std::vector<Armor> ParseRaw(double scale, const cv::Mat& output) const;
  std::vector<Armor> ParseEndToEnd(double scale, const YoloEndToEndOutput& output) const;

  YoloParams params_;
  YoloBackendKind backend_kind_;
  YoloClassMap class_map_;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__YOLO11_ADAPTER_HPP_
