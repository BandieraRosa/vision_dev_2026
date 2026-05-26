#include "armor_detector/yolo_common.hpp"

#include <algorithm>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <array>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <string>
#include <system_error>

namespace rm_auto_aim
{
namespace
{

constexpr std::uintmax_t MAX_METADATA_FILE_BYTES =
    static_cast<const std::uintmax_t>(1024U * 1024U);

constexpr std::array<std::string_view, 38> YOLO11_LABELS = {
    "Bsentry",       "Rsentry",       "Esentry",      "Bone",         "Rone",
    "Eone",          "Btwo",          "Rtwo",         "Etwo",         "Bthree",
    "Rthree",        "Ethree",        "Bfour",        "Rfour",        "Efour",
    "Bfive",         "Rfive",         "Efive",        "Boutpost",     "Routpost",
    "Eoutpost",      "Bbase",         "Rbase",        "Ebase",        "Pbase",
    "Bbasesmall",    "Rbasesmall",    "Ebasesmall",   "Pbasesmall",   "Bbalancethree",
    "Rbalancethree", "Ebalancethree", "Bbalancefour", "Rbalancefour", "Ebalancefour",
    "Bbalancefive",  "Rbalancefive",  "Ebalancefive"};

std::string trim(std::string_view text)
{
  auto begin = text.begin();
  auto end = text.end();
  while (begin != end && std::isspace(static_cast<unsigned char>(*begin)))
  {
    ++begin;
  }
  while (begin != end && std::isspace(static_cast<unsigned char>(*(end - 1))))
  {
    --end;
  }
  return std::string(begin, end);
}

std::string clean_label_token(std::string token)
{
  token = trim(token);
  while (!token.empty() && (token.back() == ',' || token.back() == ']'))
  {
    token.pop_back();
    token = trim(token);
  }
  while (!token.empty() && token.front() == '[')
  {
    token.erase(token.begin());
    token = trim(token);
  }
  if (token.size() >= 2 && ((token.front() == '"' && token.back() == '"') ||
                            (token.front() == '\'' && token.back() == '\'')))
  {
    token = token.substr(1, token.size() - 2);
  }
  return trim(token);
}

bool contains_traversal(const std::filesystem::path& path)
{
  for (const auto& part : path)
  {
    if (part == "..")
    {
      return true;
    }
  }
  return false;
}

bool is_integer_key(std::string_view key)
{
  if (key.empty())
  {
    return false;
  }
  return std::all_of(key.begin(), key.end(), [](char ch)
                     { return std::isdigit(static_cast<unsigned char>(ch)); });
}

void append_inline_list(std::string_view value, std::vector<std::string>& labels)
{
  std::stringstream stream{std::string(value)};
  std::string token;
  while (std::getline(stream, token, ','))
  {
    token = clean_label_token(token);
    if (!token.empty())
    {
      labels.emplace_back(std::move(token));
    }
  }
}

}  // namespace

std::vector<std::string> default_yolo11_labels()
{
  std::vector<std::string> labels;
  labels.reserve(YOLO11_LABELS.size());
  for (auto label : YOLO11_LABELS)
  {
    labels.emplace_back(label);
  }
  return labels;
}

std::string map_yolo_label(std::string_view raw_label)
{
  if (raw_label.empty())
  {
    return "negative";
  }

  if (raw_label[0] == 'B' || raw_label[0] == 'R' || raw_label[0] == 'E' ||
      raw_label[0] == 'P')
  {
    raw_label.remove_prefix(1);
  }

  if (raw_label == "sentry")
  {
    return "guard";
  }
  if (raw_label == "one")
  {
    return "1";
  }
  if (raw_label == "two")
  {
    return "2";
  }
  if (raw_label == "three" || raw_label == "balancethree")
  {
    return "3";
  }
  if (raw_label == "four" || raw_label == "balancefour")
  {
    return "4";
  }
  if (raw_label == "five" || raw_label == "balancefive")
  {
    return "5";
  }
  if (raw_label == "outpost")
  {
    return "outpost";
  }
  if (raw_label == "base" || raw_label == "basesmall")
  {
    return "base";
  }

  return "negative";
}

YoloClassMap build_yolo_class_map(const std::vector<std::string>& raw_labels,
                                  const std::vector<std::string>& ignore_classes)
{
  YoloClassMap class_map;
  class_map.raw_labels = raw_labels;
  class_map.mapped_labels.resize(raw_labels.size());
  class_map.colors.resize(raw_labels.size());
  class_map.ignored.resize(raw_labels.size());

  for (std::size_t i = 0; i < raw_labels.size(); ++i)
  {
    const auto& raw = raw_labels[i];
    class_map.mapped_labels[i] = map_yolo_label(raw);

    char first = raw.empty() ? '\0' : raw[0];
    class_map.colors[i] = (first == 'R') ? RED : (first == 'B') ? BLUE : -1;
    class_map.ignored[i] = (std::find(ignore_classes.begin(), ignore_classes.end(),
                                      class_map.mapped_labels[i]) != ignore_classes.end())
                               ? std::uint8_t{1}
                               : std::uint8_t{0};
  }

  return class_map;
}

bool validate_yolo_metadata_file(const std::string& metadata_path,
                                 std::string* error_message)
{
  std::error_code ec;
  if (!std::filesystem::is_regular_file(metadata_path, ec))
  {
    if (error_message != nullptr)
    {
      *error_message = ec ? ec.message() : "metadata path is not a regular file";
    }
    return false;
  }

  auto size = std::filesystem::file_size(metadata_path, ec);
  if (ec)
  {
    if (error_message != nullptr)
    {
      *error_message = ec.message();
    }
    return false;
  }
  if (size > MAX_METADATA_FILE_BYTES)
  {
    if (error_message != nullptr)
    {
      *error_message = "metadata file is larger than 1 MiB";
    }
    return false;
  }
  return true;
}

std::vector<std::string> load_yolo_labels_from_metadata(const std::string& metadata_path,
                                                        std::string* error_message)
{
  if (!validate_yolo_metadata_file(metadata_path, error_message))
  {
    return {};
  }

  std::ifstream file(metadata_path);
  if (!file)
  {
    if (error_message != nullptr)
    {
      *error_message = "failed to open metadata file";
    }
    return {};
  }

  std::vector<std::string> labels;
  std::string line;
  bool saw_named_label_section = false;
  while (std::getline(file, line))
  {
    auto comment_pos = line.find('#');
    if (comment_pos != std::string::npos)
    {
      line.erase(comment_pos);
    }
    line = trim(line);
    if (line.empty())
    {
      continue;
    }

    if (line.rfind('-', 0) == 0)
    {
      auto label = clean_label_token(line.substr(1));
      if (!label.empty())
      {
        labels.emplace_back(std::move(label));
      }
      continue;
    }

    auto colon_pos = line.find(':');
    if (colon_pos != std::string::npos)
    {
      auto key = trim(std::string_view(line).substr(0, colon_pos));
      auto value = trim(std::string_view(line).substr(colon_pos + 1));
      if (key == "labels" || key == "classes" || key == "names")
      {
        saw_named_label_section = true;
        if (!value.empty())
        {
          append_inline_list(value, labels);
        }
        continue;
      }
      if (is_integer_key(key))
      {
        auto label = clean_label_token(value);
        if (!label.empty())
        {
          labels.emplace_back(std::move(label));
        }
      }
      continue;
    }

    if (!saw_named_label_section)
    {
      auto label = clean_label_token(line);
      if (!label.empty())
      {
        labels.emplace_back(std::move(label));
      }
    }
  }

  if (labels.empty() && error_message != nullptr)
  {
    *error_message = "metadata did not contain labels/classes/names entries";
  }
  return labels;
}

std::string resolve_package_model_path(const std::string& model_name)
{
  if (model_name.empty())
  {
    return {};
  }

  std::filesystem::path path(model_name);
  if (path.is_absolute())
  {
    throw std::runtime_error("Absolute YOLO model paths are not allowed: " + model_name);
  }
  if (contains_traversal(path))
  {
    throw std::runtime_error("YOLO model path must not contain '..': " + model_name);
  }

  auto pkg_path = ament_index_cpp::get_package_share_directory("armor_detector");
  auto model_dir = (std::filesystem::path(pkg_path) / "model").lexically_normal();
  auto resolved = (model_dir / path).lexically_normal();
  auto relative = resolved.lexically_relative(model_dir);
  if (relative.empty() || contains_traversal(relative))
  {
    throw std::runtime_error("YOLO model path escapes package model directory: " +
                             model_name);
  }
  return resolved.string();
}

std::string resolve_optional_package_model_path(const std::string& path)
{
  if (path.empty())
  {
    return {};
  }
  return resolve_package_model_path(path);
}

void sort_yolo_armor_keypoints(std::vector<cv::Point2f>& keypoints)
{
  if (keypoints.size() != 4)
  {
    return;
  }

  std::sort(keypoints.begin(), keypoints.end(),
            [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });

  std::vector<cv::Point2f> top_points = {keypoints[0], keypoints[1]};
  std::vector<cv::Point2f> bottom_points = {keypoints[2], keypoints[3]};

  std::sort(top_points.begin(), top_points.end(),
            [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
  std::sort(bottom_points.begin(), bottom_points.end(),
            [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });

  keypoints[0] = top_points[0];
  keypoints[1] = bottom_points[0];
  keypoints[2] = top_points[1];
  keypoints[3] = bottom_points[1];
}

ArmorType determine_yolo_armor_type(const Light& light_1, const Light& light_2,
                                    float large_armor_ratio_threshold)
{
  auto avg_length = (light_1.length + light_2.length) / 2.0;
  auto center_distance = cv::norm(light_2.center - light_1.center);

  if (avg_length < 1e-6)
  {
    return ArmorType::INVALID;
  }

  double ratio = center_distance / avg_length;
  return ratio > large_armor_ratio_threshold ? ArmorType::LARGE : ArmorType::SMALL;
}

void draw_yolo_armors(cv::Mat& img, const std::vector<Armor>& armors)
{
  for (const auto& armor : armors)
  {
    cv::circle(img, armor.left_light.top, 3, cv::Scalar(255, 255, 255), 1);
    cv::circle(img, armor.left_light.bottom, 3, cv::Scalar(255, 255, 255), 1);
    cv::circle(img, armor.right_light.top, 3, cv::Scalar(255, 255, 255), 1);
    cv::circle(img, armor.right_light.bottom, 3, cv::Scalar(255, 255, 255), 1);

    auto line_color =
        (armor.left_light.color == RED) ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255);
    cv::line(img, armor.left_light.top, armor.left_light.bottom, line_color, 1);
    cv::line(img, armor.right_light.top, armor.right_light.bottom, line_color, 1);
  }

  for (const auto& armor : armors)
  {
    cv::line(img, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0),
             2);
    cv::line(img, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0),
             2);
  }

  for (const auto& armor : armors)
  {
    cv::putText(img, armor.classfication_result, armor.left_light.top,
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
  }
}

}  // namespace rm_auto_aim
