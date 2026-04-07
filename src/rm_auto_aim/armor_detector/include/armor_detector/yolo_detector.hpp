#ifndef ARMOR_DETECTOR__YOLO_DETECTOR_HPP_
#define ARMOR_DETECTOR__YOLO_DETECTOR_HPP_

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

#include "armor_detector/armor.hpp"

namespace rm_auto_aim
{

class YoloDetector
{
 public:
  struct YoloParams
  {
    std::string model_path;
    std::string device = "CPU";
    float score_threshold = 0.7f;
    float min_confidence = 0.8f;
    float nms_threshold = 0.3f;
  };

  explicit YoloDetector(const YoloParams& config);

  // 检测入口
  std::vector<Armor> Detect(const cv::Mat& rgb_img, int detect_color,
                            const std::vector<std::string>& ignore_classes);

  // 在图像上绘制检测结果
  void DrawResults(cv::Mat& img);

 private:
  static constexpr int INPUT_SIZE = 640;
  static constexpr int NUM_KEYPOINTS = 4;
  // 灯条中心距离 / 平均灯条长度
  static constexpr float LARGE_ARMOR_RATIO_THRESHOLD = 3.2f;

  // 解析模型输出张量
  std::vector<Armor> Parse(double scale, cv::Mat& output, int detect_color,
                           const std::vector<std::string>& ignore_classes);

  static void SortKeypoints(std::vector<cv::Point2f>& keypoints);

  static int DetectColor(const cv::Mat& rgb_img,
                         const std::vector<cv::Point2f>& keypoints);

  // 判断装甲板类型
  static ArmorType DetermineArmorType(const Light& light_1, const Light& light_2);

  YoloParams config_;
  int class_num_;

  ov::Core core_;
  ov::CompiledModel compiled_model_;

  std::vector<Armor> last_armors_;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__YOLO_DETECTOR_HPP_