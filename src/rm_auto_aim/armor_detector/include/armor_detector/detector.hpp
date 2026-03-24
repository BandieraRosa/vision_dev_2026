#ifndef ARMOR_DETECTOR__DETECTOR_HPP_
#define ARMOR_DETECTOR__DETECTOR_HPP_

// OpenCV
#include <opencv2/core.hpp>

// STD
#include <cmath>
#include <vector>

#include "armor_detector/armor.hpp"
#include "armor_detector/number_classifier.hpp"
#include "auto_aim_interfaces/msg/debug_armors.hpp"
#include "auto_aim_interfaces/msg/debug_lights.hpp"

namespace rm_auto_aim
{
class Detector
{
 public:
  struct LightParams
  {
    // width / height
    double min_ratio;
    double max_ratio;
    // vertical angle与垂直方向的最大差角
    double max_angle;
  };

  struct ArmorParams
  {
    double min_light_ratio;
    // light pairs distance
    double min_small_center_distance;
    double max_small_center_distance;
    double min_large_center_distance;
    double max_large_center_distance;
    // horizontal angle
    double max_angle;
  };

  Detector(const int& binary_lower_thres, const int& binary_upper_thres, const int& color,
           const LightParams& l, const ArmorParams& a);

  std::vector<Armor> Detect(const cv::Mat& input);

  cv::Mat PreprocessImage(const cv::Mat& input);
  std::vector<Light> FindLights(const cv::Mat& rbg_img, const cv::Mat& binary_img);
  std::vector<Armor> MatchLights(const std::vector<Light>& lights);

  // For debug usage
  cv::Mat GetAllNumbersImage();
  void DrawResults(cv::Mat& img);

  // int binary_thres;
  int binary_lower_thres_;
  int binary_upper_thres_;
  int detect_color;
  LightParams l;
  ArmorParams a;

  std::unique_ptr<NumberClassifier> classifier;

  // Debug msgs
  cv::Mat binary_img;
  auto_aim_interfaces::msg::DebugLights debug_lights;
  auto_aim_interfaces::msg::DebugArmors debug_armors;

 private:
  bool IsLight(const Light& possible_light);
  bool ContainLight(const Light& light_1, const Light& light_2,
                    const std::vector<Light>& lights);
  ArmorType IsArmor(const Light& light_1, const Light& light_2);

  std::vector<Light> lights_;
  std::vector<Armor> armors_;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__DETECTOR_HPP_
