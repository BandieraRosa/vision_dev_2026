#ifndef ARMOR_DETECTOR__DETECTION_RESULT_HPP_
#define ARMOR_DETECTOR__DETECTION_RESULT_HPP_

#include <opencv2/core/mat.hpp>
#include <optional>
#include <vector>

#include "armor_detector/armor.hpp"
#include "auto_aim_interfaces/msg/debug_armors.hpp"
#include "auto_aim_interfaces/msg/debug_lights.hpp"

namespace rm_auto_aim
{

struct DebugData
{
  auto_aim_interfaces::msg::DebugLights lights;
  auto_aim_interfaces::msg::DebugArmors armors;
};

struct DetectionResult
{
  std::vector<Armor> armors;
  std::optional<cv::Mat> binary_image;
  std::optional<DebugData> debug_data;
  std::optional<cv::Mat> numbers_image;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__DETECTION_RESULT_HPP_
