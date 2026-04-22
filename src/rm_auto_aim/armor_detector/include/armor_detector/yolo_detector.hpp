#ifndef ARMOR_DETECTOR__YOLO_DETECTOR_HPP_
#define ARMOR_DETECTOR__YOLO_DETECTOR_HPP_

#if defined(ARMOR_DETECTOR_HAS_OPENVINO) || defined(ARMOR_DETECTOR_HAS_TENSORRT)

#include <cstddef>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "armor_detector/armor.hpp"
#include "armor_detector/detector_base.hpp"

#if ARMOR_DETECTOR_HAS_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime.h>

#elif ARMOR_DETECTOR_HAS_OPENVINO
#include <openvino/openvino.hpp>
#endif  // ARMOR_DETECTOR_HAS_OPENVINO / ARMOR_DETECTOR_HAS_TENSORRT

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

  ~YoloDetector() override;
  YoloDetector(const YoloDetector&) = delete;
  YoloDetector& operator=(const YoloDetector&) = delete;

  DetectionResult Detect(const cv::Mat& rgb_img) override;

  void DrawResults(cv::Mat& img) override;

 private:
  std::vector<Armor> Parse(double scale, cv::Mat& output);
  void SortKeypoints(std::vector<cv::Point2f>& keypoints);
  ArmorType DetermineArmorType(const Light& light_1, const Light& light_2);

  YoloParams params_;
  int class_num_;

#if ARMOR_DETECTOR_HAS_TENSORRT
  std::unique_ptr<nvinfer1::IRuntime> trt_runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> trt_engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> trt_context_;

  std::string trt_input_name_;
  std::string trt_output_name_;

  nvinfer1::Dims trt_input_dims_{};
  nvinfer1::Dims trt_output_dims_{};

  void* trt_d_input_ = nullptr;
  void* trt_d_output_ = nullptr;
  cudaStream_t trt_stream_ = nullptr;

  std::size_t trt_input_bytes_ = 0;
  std::size_t trt_output_bytes_ = 0;

  std::vector<float> trt_host_input_;
  std::vector<float> trt_host_output_;

#elif ARMOR_DETECTOR_HAS_OPENVINO
  ov::Core core_;
  ov::CompiledModel compiled_model_;
  ov::InferRequest infer_request_;
#endif  // ARMOR_DETECTOR_HAS_OPENVINO / ARMOR_DETECTOR_HAS_TENSORRT

  std::vector<Armor> last_armors_;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR_HAS_OPENVINO || ARMOR_DETECTOR_HAS_TENSORRT

#endif  // ARMOR_DETECTOR__YOLO_DETECTOR_HPP_
