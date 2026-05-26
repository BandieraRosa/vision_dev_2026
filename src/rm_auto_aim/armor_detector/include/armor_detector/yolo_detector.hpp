#ifndef ARMOR_DETECTOR__YOLO_DETECTOR_HPP_
#define ARMOR_DETECTOR__YOLO_DETECTOR_HPP_

#if ARMOR_DETECTOR_HAS_OPENVINO || ARMOR_DETECTOR_HAS_TENSORRT

#include <memory>
#include <opencv2/core.hpp>
#include <vector>

#include "armor_detector/armor.hpp"
#include "armor_detector/detector_base.hpp"
#include "armor_detector/yolo_common.hpp"

namespace rclcpp
{
class Node;
}

namespace rm_auto_aim
{

class InferenceBackend;
class ModelAdapter;

class YoloDetector : public DetectorBase
{
 public:
  using YoloParams = ::rm_auto_aim::YoloParams;

  static std::unique_ptr<YoloDetector> Create(rclcpp::Node& node);

  explicit YoloDetector(const YoloParams& params);
  ~YoloDetector() override;

  YoloDetector(const YoloDetector&) = delete;
  YoloDetector& operator=(const YoloDetector&) = delete;

  DetectionResult Detect(const cv::Mat& rgb_img) override;
  void DrawResults(cv::Mat& img) override;

 private:
  YoloParams params_;
  std::unique_ptr<InferenceBackend> backend_;
  std::unique_ptr<ModelAdapter> model_adapter_;
  std::vector<Armor> last_armors_;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR_HAS_OPENVINO || ARMOR_DETECTOR_HAS_TENSORRT

#endif  // ARMOR_DETECTOR__YOLO_DETECTOR_HPP_
