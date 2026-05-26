#ifndef ARMOR_DETECTOR__YOLO_COMMON_HPP_
#define ARMOR_DETECTOR__YOLO_COMMON_HPP_

#include <cstdint>
#include <opencv2/core.hpp>
#include <string>
#include <string_view>
#include <vector>

#include "armor_detector/armor.hpp"

namespace rm_auto_aim
{

enum class YoloBackendKind : std::uint8_t
{
  OPENVINO,
  TENSORRT
};

struct YoloParams
{
  std::string model_path;
  std::string metadata_path;
#if ARMOR_DETECTOR_HAS_OPENVINO
  std::string backend = "openvino";
#elif ARMOR_DETECTOR_HAS_TENSORRT
  std::string backend = "tensorrt";
#else
  std::string backend;
#endif
  std::string device = "CPU";
  int input_size = 640;
  float score_threshold = 0.7f;
  float min_confidence = 0.8f;
  float nms_threshold = 0.3f;
  std::vector<std::string> ignore_classes;
  int detect_color = RED;
  int num_keypoints = 4;
  float large_armor_ratio_threshold = 3.2f;
  bool end_to_end = false;
};

struct YoloClassMap
{
  std::vector<std::string> raw_labels;
  std::vector<std::string> mapped_labels;
  std::vector<int> colors;
  std::vector<std::uint8_t> ignored;

  int ClassCount() const noexcept { return static_cast<int>(raw_labels.size()); }
};

std::vector<std::string> default_yolo11_labels();
std::string map_yolo_label(std::string_view raw_label);
YoloClassMap build_yolo_class_map(const std::vector<std::string>& raw_labels,
                                  const std::vector<std::string>& ignore_classes);
bool validate_yolo_metadata_file(const std::string& metadata_path,
                                 std::string* error_message);
std::vector<std::string> load_yolo_labels_from_metadata(const std::string& metadata_path,
                                                        std::string* error_message);
std::string resolve_package_model_path(const std::string& model_name);
std::string resolve_optional_package_model_path(const std::string& path);
void sort_yolo_armor_keypoints(std::vector<cv::Point2f>& keypoints);
ArmorType determine_yolo_armor_type(const Light& light_1, const Light& light_2,
                                    float large_armor_ratio_threshold);
void draw_yolo_armors(cv::Mat& img, const std::vector<Armor>& armors);

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__YOLO_COMMON_HPP_
