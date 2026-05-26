#include "armor_detector/yolo_detector.hpp"

#if ARMOR_DETECTOR_HAS_OPENVINO || ARMOR_DETECTOR_HAS_TENSORRT

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <rclcpp/rclcpp.hpp>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "armor_detector/inference_backend.hpp"
#include "armor_detector/model_adapter.hpp"
#if ARMOR_DETECTOR_HAS_OPENVINO
#include "armor_detector/openvino_backend.hpp"
#endif
#if ARMOR_DETECTOR_HAS_TENSORRT
#include "armor_detector/tensorrt_backend.hpp"
#endif
#include "armor_detector/yolo11_adapter.hpp"

namespace rm_auto_aim
{
namespace
{

template <typename T>
T get_parameter(rclcpp::Node& node, const std::string& name, const T& default_value)
{
  if (!node.has_parameter(name))
  {
    return node.declare_parameter<T>(name, default_value);
  }
  return node.get_parameter(name).get_value<T>();
}

void validate_metadata_path(rclcpp::Node& node, const std::string& metadata_path)
{
  if (metadata_path.empty())
  {
    return;
  }

  std::error_code ec;
  if (!std::filesystem::is_regular_file(metadata_path, ec))
  {
    auto error_message = ec ? ec.message() : "path is not a regular file";
    RCLCPP_ERROR(node.get_logger(), "Invalid YOLO metadata file '%s': %s",
                 metadata_path.c_str(), error_message.c_str());
    throw std::runtime_error("Invalid YOLO metadata file: " + metadata_path + " (" +
                             error_message + ")");
  }

  auto size = std::filesystem::file_size(metadata_path, ec);
  if (ec || size > static_cast<uintmax_t>(1024U * 1024U))
  {
    auto error_message = ec ? ec.message() : "metadata file is larger than 1 MiB";
    RCLCPP_ERROR(node.get_logger(), "Invalid YOLO metadata file '%s': %s",
                 metadata_path.c_str(), error_message.c_str());
    throw std::runtime_error("Invalid YOLO metadata file: " + metadata_path + " (" +
                             error_message + ")");
  }
}

std::string default_backend_name()
{
#if ARMOR_DETECTOR_HAS_OPENVINO
  return "openvino";
#elif ARMOR_DETECTOR_HAS_TENSORRT
  return "tensorrt";
#else
  return {};
#endif
}

std::string normalize_backend_name(std::string backend)
{
  std::transform(backend.begin(), backend.end(), backend.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return backend;
}

std::unique_ptr<InferenceBackend> make_backend(const YoloParams& params)
{
  if (params.backend == "openvino")
  {
#if ARMOR_DETECTOR_HAS_OPENVINO
    return std::make_unique<OpenVinoBackend>(params);
#else
    throw std::runtime_error(
        "yolo.backend is 'openvino', but armor_detector was built without OpenVINO "
        "support");
#endif
  }

  if (params.backend == "tensorrt")
  {
#if ARMOR_DETECTOR_HAS_TENSORRT
    return std::make_unique<TensorRtBackend>(params);
#else
    throw std::runtime_error(
        "yolo.backend is 'tensorrt', but armor_detector was built without TensorRT "
        "support");
#endif
  }

  throw std::runtime_error("Unsupported yolo.backend value: '" + params.backend +
                           "'. Expected 'openvino' or 'tensorrt'.");
}

}  // namespace

std::unique_ptr<YoloDetector> YoloDetector::Create(rclcpp::Node& node)
{
  auto model_name = get_parameter<std::string>(node, "yolo.model_path", "");
  if (model_name.empty())
  {
    RCLCPP_ERROR(
        node.get_logger(),
        "Parameter 'yolo.model_path' must not be empty when detector_type is 'yolo'.");
    throw std::runtime_error("Parameter 'yolo.model_path' must not be empty");
  }

  auto model_path = resolve_package_model_path(model_name);
  if (!std::filesystem::is_regular_file(model_path))
  {
    RCLCPP_ERROR(node.get_logger(), "YOLO model file not found or not regular: %s",
                 model_path.c_str());
    throw std::runtime_error("YOLO model file not found or not regular: " + model_path);
  }

  auto metadata_name = get_parameter<std::string>(node, "yolo.metadata_path", "");
  auto metadata_path = resolve_optional_package_model_path(metadata_name);
  validate_metadata_path(node, metadata_path);

  auto backend = normalize_backend_name(
      get_parameter<std::string>(node, "yolo.backend", default_backend_name()));

  YoloParams yolo_params = {
      .model_path = model_path,
      .metadata_path = metadata_path,
      .backend = backend,
      .device = get_parameter<std::string>(node, "yolo.device", "CPU"),
      .input_size = get_parameter<int>(node, "yolo.input_size", 640),
      .score_threshold =
          static_cast<float>(get_parameter<double>(node, "yolo.score_threshold", 0.7)),
      .min_confidence =
          static_cast<float>(get_parameter<double>(node, "yolo.min_confidence", 0.8)),
      .nms_threshold =
          static_cast<float>(get_parameter<double>(node, "yolo.nms_threshold", 0.3)),
      .ignore_classes =
          get_parameter<std::vector<std::string>>(node, "ignore_classes", {"negative"}),
      .detect_color = get_parameter<int>(node, "detect_color", RED),
      .num_keypoints = get_parameter<int>(node, "yolo.num_keypoints", 4),
      .large_armor_ratio_threshold = static_cast<float>(
          get_parameter<double>(node, "yolo.large_armor_ratio_threshold", 3.2)),
      .end_to_end = get_parameter<bool>(node, "yolo.end_to_end", false)};

  return std::make_unique<YoloDetector>(yolo_params);
}

YoloDetector::YoloDetector(const YoloParams& params) : params_(params)
{
  backend_ = make_backend(params_);
  model_adapter_ = std::make_unique<Yolo11Adapter>(params_, backend_->BackendKind());
}

YoloDetector::~YoloDetector() = default;

DetectionResult YoloDetector::Detect(const cv::Mat& rgb_img)
{
  auto total_start_time = std::chrono::steady_clock::now();
  DetectionResult result;

  auto inference_output = backend_->Infer(rgb_img);
  debug_latencies_ = inference_output.latencies;

  auto parse_start_time = std::chrono::steady_clock::now();
  last_armors_ = model_adapter_->Decode(inference_output);
  result.armors = last_armors_;
  auto parse_end_time = std::chrono::steady_clock::now();

  auto parse_latency = std::chrono::duration_cast<std::chrono::microseconds>(
                           parse_end_time - parse_start_time)
                           .count();
  debug_latencies_.emplace_back("Parse Output",
                                static_cast<std::uint64_t>(parse_latency));

  if (backend_->BackendKind() == YoloBackendKind::TENSORRT)
  {
    auto total_end_time = std::chrono::steady_clock::now();
    auto total_latency = std::chrono::duration_cast<std::chrono::microseconds>(
                             total_end_time - total_start_time)
                             .count();
    debug_latencies_.emplace_back("Total", static_cast<std::uint64_t>(total_latency));
  }

  return result;
}

void YoloDetector::DrawResults(cv::Mat& img)
{
  model_adapter_->DrawResults(img, last_armors_);
}

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR_HAS_OPENVINO || ARMOR_DETECTOR_HAS_TENSORRT
