#ifndef ARMOR_DETECTOR__TENSORRT_BACKEND_HPP_
#define ARMOR_DETECTOR__TENSORRT_BACKEND_HPP_

#if ARMOR_DETECTOR_HAS_TENSORRT

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "armor_detector/gpu_preprocessor.hpp"
#include "armor_detector/inference_backend.hpp"
#include "armor_detector/yolo_common.hpp"

namespace rm_auto_aim
{

class TensorRtBackend final : public InferenceBackend
{
 public:
  explicit TensorRtBackend(const YoloParams& params);
  ~TensorRtBackend() override;

  TensorRtBackend(const TensorRtBackend&) = delete;
  TensorRtBackend& operator=(const TensorRtBackend&) = delete;

  YoloBackendKind BackendKind() const noexcept override
  {
    return YoloBackendKind::TENSORRT;
  }

  YoloInferenceOutput Infer(const cv::Mat& rgb_img) override;

 private:
  void InitRaw();
  void InitEndToEnd();
  YoloInferenceOutput InferRaw(const cv::Mat& rgb_img);
  YoloInferenceOutput InferEndToEnd(const cv::Mat& rgb_img);

  YoloParams params_;

  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  std::string raw_input_name_;
  std::string raw_output_name_;
  nvinfer1::Dims raw_input_dims_{};
  nvinfer1::Dims raw_output_dims_{};
  void* raw_d_input_ = nullptr;
  void* raw_d_output_ = nullptr;
  cudaStream_t raw_stream_ = nullptr;
  std::size_t raw_input_bytes_ = 0;
  std::size_t raw_output_bytes_ = 0;
  std::vector<float> raw_host_output_;

  std::string e2e_input_name_;
  std::string e2e_num_name_;
  std::string e2e_boxes_name_;
  std::string e2e_scores_name_;
  std::string e2e_classes_name_;
  std::string e2e_kpts_name_;

  void* d_input_ = nullptr;
  int* d_num_ = nullptr;
  float* d_boxes_ = nullptr;
  float* d_scores_ = nullptr;
  int* d_classes_ = nullptr;
  float* d_kpts_ = nullptr;

  int* h_num_ = nullptr;
  float* h_boxes_ = nullptr;
  float* h_scores_ = nullptr;
  int* h_classes_ = nullptr;
  float* h_kpts_ = nullptr;

  cudaStream_t stream_ = nullptr;
  int keep_topk_ = 0;
  int kpt_channels_ = 0;
  std::size_t input_bytes_ = 0;

  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  bool graph_ready_ = false;

  std::unique_ptr<GpuPreprocessor> preprocessor_;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR_HAS_TENSORRT
#endif  // ARMOR_DETECTOR__TENSORRT_BACKEND_HPP_
