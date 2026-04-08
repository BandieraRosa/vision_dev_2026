#ifndef ARMOR_DETECTOR__YOLO_DETECTOR_HPP_
#define ARMOR_DETECTOR__YOLO_DETECTOR_HPP_

#if ARMOR_DETECTOR_HAS_OPENVINO

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

#include "armor_detector/armor.hpp"
#include "armor_detector/detector_base.hpp"
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
    float large_armor_ratio_threshold = 3.2f;  // 灯条中心距离 / 平均灯条长度
  };

  explicit YoloDetector(const YoloParams& params);

  // 检测入口
  std::vector<Armor> Detect(const cv::Mat& rgb_img);

  // 在图像上绘制检测结果
  void DrawResults(cv::Mat& img);

 private:
  // 解析模型输出张量
  std::vector<Armor> Parse(double scale, cv::Mat& output);

  void SortKeypoints(std::vector<cv::Point2f>& keypoints);

  // 判断装甲板类型
  ArmorType DetermineArmorType(const Light& light_1, const Light& light_2);

  YoloParams params_;
  int class_num_;

  ov::Core core_;
  ov::CompiledModel compiled_model_;

  std::vector<Armor> last_armors_;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR_HAS_OPENVINO

#endif  // ARMOR_DETECTOR__YOLO_DETECTOR_HPP_