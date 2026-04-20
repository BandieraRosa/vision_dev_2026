#ifndef ARMOR_DETECTOR__YOLO_DETECTOR_HPP_
#define ARMOR_DETECTOR__YOLO_DETECTOR_HPP_

#if ARMOR_DETECTOR_HAS_OPENVINO

#include <memory>
#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

#include "armor_detector/armor.hpp"
#include "armor_detector/detector_base.hpp"

namespace rclcpp
{
class Node;
}

namespace rm_auto_aim
{

class YoloDetector : public DetectorBase
{
 public:
  struct YoloParams
  {
    std::string model_path;
    std::string device = "CPU";
    int input_size = 640;
    float score_threshold = 0.7f;
    float min_confidence = 0.8f;
    float nms_threshold = 0.3f;
    std::vector<std::string> ignore_classes;
    int detect_color;
    int num_keypoints = 4;
    float large_armor_ratio_threshold = 3.2f;
  };

  static std::unique_ptr<YoloDetector> Create(rclcpp::Node& node);

  explicit YoloDetector(const YoloParams& params);

  DetectionResult Detect(const cv::Mat& rgb_img) override;

  void DrawResults(cv::Mat& img) override;

 private:
  std::vector<Armor> Parse(double scale, cv::Mat& output);
  void SortKeypoints(std::vector<cv::Point2f>& keypoints);
  ArmorType DetermineArmorType(const Light& light_1, const Light& light_2);

  YoloParams params_;
  int class_num_;

  ov::Core core_;
  ov::CompiledModel compiled_model_;
  ov::InferRequest infer_request_;

  std::vector<Armor> last_armors_;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR_HAS_OPENVINO

#endif  // ARMOR_DETECTOR__YOLO_DETECTOR_HPP_
