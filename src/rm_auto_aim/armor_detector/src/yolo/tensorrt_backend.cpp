#include "armor_detector/tensorrt_backend.hpp"

#if ARMOR_DETECTOR_HAS_TENSORRT

#include <chrono>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <rclcpp/rclcpp.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rm_auto_aim
{
namespace
{

class TRTLogger : public nvinfer1::ILogger
{
 public:
  void log(Severity severity, const char* msg) noexcept override
  {
    if (severity <= Severity::kWARNING)
    {
      RCLCPP_WARN(rclcpp::get_logger("armor_detector"), "[TensorRT] %s", msg);
    }
  }
};

TRTLogger& get_trt_logger()
{
  static TRTLogger logger;
  return logger;
}

void ensure_trt_plugins_initialized()
{
  static bool initialized = false;
  if (initialized)
  {
    return;
  }

  if (!initLibNvInferPlugins(&get_trt_logger(), ""))
  {
    throw std::runtime_error("initLibNvInferPlugins failed");
  }

  auto* registry = getPluginRegistry();
  auto* creator = registry->getPluginCreator("EfficientNMS_TRT", "1", "");
  if (!creator)
  {
    throw std::runtime_error(
        "EfficientNMS_TRT plugin creator not found after initLibNvInferPlugins");
  }

  initialized = true;
  RCLCPP_INFO(rclcpp::get_logger("armor_detector"),
              "TensorRT plugins initialized, EfficientNMS_TRT found");
}

bool has_dynamic_dim(const nvinfer1::Dims& dims)
{
  for (int i = 0; i < dims.nbDims; ++i)
  {
    if (dims.d[i] == -1)
    {
      return true;
    }
  }
  return false;
}

int64_t tensor_volume(const nvinfer1::Dims& dims)
{
  int64_t volume = 1;
  for (int i = 0; i < dims.nbDims; ++i)
  {
    if (dims.d[i] < 0)
    {
      return -1;
    }
    volume *= dims.d[i];
  }
  return volume;
}

std::string dims_to_string(const nvinfer1::Dims& dims)
{
  std::ostringstream oss;
  oss << "[";
  for (int i = 0; i < dims.nbDims; ++i)
  {
    oss << dims.d[i];
    if (i + 1 < dims.nbDims)
    {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

std::vector<char> read_binary_file(const std::string& path)
{
  std::ifstream file(path, std::ios::binary);
  if (!file)
  {
    throw std::runtime_error("Failed to open engine file: " + path);
  }

  file.seekg(0, std::ios::end);
  std::size_t size = static_cast<std::size_t>(file.tellg());
  file.seekg(0, std::ios::beg);

  std::vector<char> data(size);
  file.read(data.data(), static_cast<std::streamsize>(size));
  return data;
}

#define ARMOR_DETECTOR_CHECK_CUDA(call)                                                  \
  do                                                                                     \
  {                                                                                      \
    cudaError_t err__ = (call);                                                          \
    if (err__ != cudaSuccess)                                                            \
    {                                                                                    \
      throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err__) + \
                               " at " __FILE__ ":" + std::to_string(__LINE__));          \
    }                                                                                    \
  } while (0)

template <typename T>
void cuda_alloc_device(T** p, std::size_t n)
{
  ARMOR_DETECTOR_CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(p), n * sizeof(T)));
}

template <typename T>
void cuda_alloc_pinned(T** p, std::size_t n)
{
  ARMOR_DETECTOR_CHECK_CUDA(
      cudaHostAlloc(reinterpret_cast<void**>(p), n * sizeof(T), cudaHostAllocDefault));
}

}  // namespace

TensorRtBackend::TensorRtBackend(const YoloParams& params) : params_(params)
{
  if (params_.end_to_end)
  {
    InitEndToEnd();
  }
  else
  {
    InitRaw();
  }
}

void TensorRtBackend::InitRaw()
{
  auto engine_data = read_binary_file(params_.model_path);

  runtime_.reset(nvinfer1::createInferRuntime(get_trt_logger()));
  if (!runtime_)
  {
    throw std::runtime_error("Failed to create TensorRT runtime");
  }

  engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  if (!engine_)
  {
    throw std::runtime_error("Failed to deserialize TensorRT engine: " +
                             params_.model_path);
  }

  context_.reset(engine_->createExecutionContext());
  if (!context_)
  {
    throw std::runtime_error("Failed to create TensorRT execution context");
  }

  for (int i = 0; i < engine_->getNbIOTensors(); ++i)
  {
    const char* name = engine_->getIOTensorName(i);
    auto mode = engine_->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT)
    {
      raw_input_name_ = name;
    }
    else if (mode == nvinfer1::TensorIOMode::kOUTPUT)
    {
      raw_output_name_ = name;
    }
  }
  if (raw_input_name_.empty() || raw_output_name_.empty())
  {
    throw std::runtime_error("Cannot find input/output tensor names in engine");
  }

  nvinfer1::Dims input_dims_raw = engine_->getTensorShape(raw_input_name_.c_str());
  if (has_dynamic_dim(input_dims_raw))
  {
    nvinfer1::Dims4 real_input_dims{1, 3, params_.input_size, params_.input_size};
    if (!context_->setInputShape(raw_input_name_.c_str(), real_input_dims))
    {
      throw std::runtime_error("Failed to set TensorRT input shape");
    }
  }

  raw_input_dims_ = context_->getTensorShape(raw_input_name_.c_str());
  raw_output_dims_ = context_->getTensorShape(raw_output_name_.c_str());

  RCLCPP_INFO(rclcpp::get_logger("armor_detector"), "TensorRT input = %s, output = %s",
              dims_to_string(raw_input_dims_).c_str(),
              dims_to_string(raw_output_dims_).c_str());

  if (has_dynamic_dim(raw_input_dims_) || has_dynamic_dim(raw_output_dims_))
  {
    throw std::runtime_error("Dynamic dims unresolved after setInputShape");
  }

  if (engine_->getTensorDataType(raw_input_name_.c_str()) != nvinfer1::DataType::kFLOAT ||
      engine_->getTensorDataType(raw_output_name_.c_str()) != nvinfer1::DataType::kFLOAT)
  {
    throw std::runtime_error("TensorRT engine must have FP32 IO for this detector");
  }

  if (raw_output_dims_.nbDims != 3)
  {
    throw std::runtime_error("Unexpected output rank for YOLO, expected 3 but got " +
                             std::to_string(raw_output_dims_.nbDims));
  }

  int64_t input_elems = tensor_volume(raw_input_dims_);
  int64_t output_elems = tensor_volume(raw_output_dims_);
  if (input_elems <= 0 || output_elems <= 0)
  {
    throw std::runtime_error("Invalid tensor volume");
  }

  raw_input_bytes_ = static_cast<std::size_t>(input_elems) * sizeof(float);
  raw_output_bytes_ = static_cast<std::size_t>(output_elems) * sizeof(float);
  raw_host_output_.resize(static_cast<std::size_t>(output_elems));

  ARMOR_DETECTOR_CHECK_CUDA(cudaMalloc(&raw_d_input_, raw_input_bytes_));
  ARMOR_DETECTOR_CHECK_CUDA(cudaMalloc(&raw_d_output_, raw_output_bytes_));
  ARMOR_DETECTOR_CHECK_CUDA(cudaStreamCreate(&raw_stream_));

  if (!context_->setTensorAddress(raw_input_name_.c_str(), raw_d_input_))
  {
    throw std::runtime_error("Failed to bind TensorRT input tensor address");
  }
  if (!context_->setTensorAddress(raw_output_name_.c_str(), raw_d_output_))
  {
    throw std::runtime_error("Failed to bind TensorRT output tensor address");
  }

  GpuPreprocessor::Config pp_cfg;
  pp_cfg.dst_size = params_.input_size;
  pp_cfg.swap_rb = false;
  preprocessor_ = std::make_unique<GpuPreprocessor>(pp_cfg);
}

void TensorRtBackend::InitEndToEnd()
{
  cudaError_t err = cudaSetDeviceFlags(cudaDeviceScheduleSpin);
  unsigned int flags = 0;
  cudaGetDeviceFlags(&flags);
  RCLCPP_INFO(rclcpp::get_logger("armor_detector"),
              "setDeviceFlags=%s, current flags=0x%x, spin=%d", cudaGetErrorString(err),
              flags, (flags & cudaDeviceScheduleMask) == cudaDeviceScheduleSpin);

  ensure_trt_plugins_initialized();
  auto engine_data = read_binary_file(params_.model_path);

  runtime_.reset(nvinfer1::createInferRuntime(get_trt_logger()));
  if (!runtime_)
  {
    throw std::runtime_error("createInferRuntime failed");
  }

  engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  if (!engine_)
  {
    throw std::runtime_error("deserializeCudaEngine failed: " + params_.model_path);
  }

  context_.reset(engine_->createExecutionContext());
  if (!context_)
  {
    throw std::runtime_error("createExecutionContext failed");
  }

  for (int i = 0; i < engine_->getNbIOTensors(); ++i)
  {
    const char* name = engine_->getIOTensorName(i);
    auto mode = engine_->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT)
    {
      e2e_input_name_ = name;
    }
    else
    {
      std::string tensor_name = name;
      if (tensor_name == "num_dets")
      {
        e2e_num_name_ = tensor_name;
      }
      else if (tensor_name == "det_boxes")
      {
        e2e_boxes_name_ = tensor_name;
      }
      else if (tensor_name == "det_scores")
      {
        e2e_scores_name_ = tensor_name;
      }
      else if (tensor_name == "det_classes")
      {
        e2e_classes_name_ = tensor_name;
      }
      else if (tensor_name == "det_kpts")
      {
        e2e_kpts_name_ = tensor_name;
      }
    }
  }

  if (e2e_input_name_.empty() || e2e_num_name_.empty() || e2e_boxes_name_.empty() ||
      e2e_scores_name_.empty() || e2e_classes_name_.empty() || e2e_kpts_name_.empty())
  {
    throw std::runtime_error(
        "Engine IO names mismatch. Expected one input and outputs: "
        "num_dets / det_boxes / det_scores / det_classes / det_kpts");
  }

  auto raw_in = engine_->getTensorShape(e2e_input_name_.c_str());
  if (has_dynamic_dim(raw_in))
  {
    nvinfer1::Dims4 shp{1, 3, params_.input_size, params_.input_size};
    if (!context_->setInputShape(e2e_input_name_.c_str(), shp))
    {
      throw std::runtime_error("setInputShape failed");
    }
  }

  auto in_dims = context_->getTensorShape(e2e_input_name_.c_str());
  auto boxes_dims = context_->getTensorShape(e2e_boxes_name_.c_str());
  auto kpts_dims = context_->getTensorShape(e2e_kpts_name_.c_str());
  if (has_dynamic_dim(in_dims) || has_dynamic_dim(boxes_dims) ||
      has_dynamic_dim(kpts_dims))
  {
    throw std::runtime_error("Unresolved dynamic dims after setInputShape");
  }

  keep_topk_ = static_cast<int>(boxes_dims.d[1]);
  kpt_channels_ = static_cast<int>(kpts_dims.d[2]);

  RCLCPP_INFO(rclcpp::get_logger("armor_detector"),
              "Engine: input=%s, keep_topk=%d, kpt_channels=%d",
              dims_to_string(in_dims).c_str(), keep_topk_, kpt_channels_);

  std::size_t input_elems = static_cast<std::size_t>(tensor_volume(in_dims));
  input_bytes_ = input_elems * sizeof(float);

  cuda_alloc_device(reinterpret_cast<float**>(&d_input_), input_elems);
  cuda_alloc_device(&d_num_, static_cast<std::size_t>(1));
  cuda_alloc_device(&d_boxes_, static_cast<std::size_t>(keep_topk_) * 4);
  cuda_alloc_device(&d_scores_, static_cast<std::size_t>(keep_topk_));
  cuda_alloc_device(&d_classes_, static_cast<std::size_t>(keep_topk_));
  cuda_alloc_device(&d_kpts_, static_cast<std::size_t>(keep_topk_) * kpt_channels_);

  cuda_alloc_pinned(&h_num_, static_cast<std::size_t>(1));
  cuda_alloc_pinned(&h_boxes_, static_cast<std::size_t>(keep_topk_) * 4);
  cuda_alloc_pinned(&h_scores_, static_cast<std::size_t>(keep_topk_));
  cuda_alloc_pinned(&h_classes_, static_cast<std::size_t>(keep_topk_));
  cuda_alloc_pinned(&h_kpts_, static_cast<std::size_t>(keep_topk_) * kpt_channels_);

  ARMOR_DETECTOR_CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

  auto bind = [&](const std::string& name, void* ptr)
  {
    if (!context_->setTensorAddress(name.c_str(), ptr))
    {
      throw std::runtime_error("setTensorAddress failed: " + name);
    }
  };
  bind(e2e_input_name_, d_input_);
  bind(e2e_num_name_, d_num_);
  bind(e2e_boxes_name_, d_boxes_);
  bind(e2e_scores_name_, d_scores_);
  bind(e2e_classes_name_, d_classes_);
  bind(e2e_kpts_name_, d_kpts_);

  GpuPreprocessor::Config pp_cfg;
  pp_cfg.dst_size = params_.input_size;
  pp_cfg.swap_rb = false;
  preprocessor_ = std::make_unique<GpuPreprocessor>(pp_cfg);
}

TensorRtBackend::~TensorRtBackend()
{
  if (graph_exec_)
  {
    cudaGraphExecDestroy(graph_exec_);
  }
  if (graph_)
  {
    cudaGraphDestroy(graph_);
  }

  context_.reset();
  engine_.reset();
  runtime_.reset();

  if (stream_)
  {
    cudaStreamDestroy(stream_);
  }
  if (d_input_)
  {
    cudaFree(d_input_);
  }
  if (d_num_)
  {
    cudaFree(d_num_);
  }
  if (d_boxes_)
  {
    cudaFree(d_boxes_);
  }
  if (d_scores_)
  {
    cudaFree(d_scores_);
  }
  if (d_classes_)
  {
    cudaFree(d_classes_);
  }
  if (d_kpts_)
  {
    cudaFree(d_kpts_);
  }
  if (h_num_)
  {
    cudaFreeHost(h_num_);
  }
  if (h_boxes_)
  {
    cudaFreeHost(h_boxes_);
  }
  if (h_scores_)
  {
    cudaFreeHost(h_scores_);
  }
  if (h_classes_)
  {
    cudaFreeHost(h_classes_);
  }
  if (h_kpts_)
  {
    cudaFreeHost(h_kpts_);
  }

  if (raw_stream_ != nullptr)
  {
    cudaStreamDestroy(raw_stream_);
  }
  if (raw_d_input_ != nullptr)
  {
    cudaFree(raw_d_input_);
  }
  if (raw_d_output_ != nullptr)
  {
    cudaFree(raw_d_output_);
  }
}

YoloInferenceOutput TensorRtBackend::Infer(const cv::Mat& rgb_img)
{
  if (params_.end_to_end)
  {
    return InferEndToEnd(rgb_img);
  }
  return InferRaw(rgb_img);
}

YoloInferenceOutput TensorRtBackend::InferRaw(const cv::Mat& rgb_img)
{
  YoloInferenceOutput output;
  output.layout = YoloOutputLayout::RAW;
  if (rgb_img.empty())
  {
    return output;
  }

  auto infer_start_time = std::chrono::steady_clock::now();

  output.scale =
      preprocessor_->Run(rgb_img, static_cast<float*>(raw_d_input_), raw_stream_);

  if (!context_->enqueueV3(raw_stream_))
  {
    throw std::runtime_error("TensorRT enqueueV3 failed");
  }

  ARMOR_DETECTOR_CHECK_CUDA(cudaMemcpyAsync(raw_host_output_.data(), raw_d_output_,
                                            raw_output_bytes_, cudaMemcpyDeviceToHost,
                                            raw_stream_));
  ARMOR_DETECTOR_CHECK_CUDA(cudaStreamSynchronize(raw_stream_));

  auto infer_end_time = std::chrono::steady_clock::now();
  auto infer_latency = std::chrono::duration_cast<std::chrono::microseconds>(
                           infer_end_time - infer_start_time)
                           .count();
  output.latencies.emplace_back("Inference", static_cast<std::uint64_t>(infer_latency));

  output.raw_output =
      cv::Mat(static_cast<int>(raw_output_dims_.d[1]),
              static_cast<int>(raw_output_dims_.d[2]), CV_32F, raw_host_output_.data());
  return output;
}

YoloInferenceOutput TensorRtBackend::InferEndToEnd(const cv::Mat& rgb_img)
{
  YoloInferenceOutput output;
  output.layout = YoloOutputLayout::END_TO_END;
  if (rgb_img.empty())
  {
    return output;
  }

  auto t_pre_start = std::chrono::steady_clock::now();

  preprocessor_->EnsureInitialized(rgb_img.rows, rgb_img.cols);
  preprocessor_->StageHost(rgb_img);
  output.scale = preprocessor_->GetScale();

  auto t_infer_start = std::chrono::steady_clock::now();

  if (!graph_ready_)
  {
    preprocessor_->Launch(static_cast<float*>(d_input_), stream_);
    if (!context_->enqueueV3(stream_))
    {
      throw std::runtime_error("enqueueV3 failed (warmup)");
    }
    ARMOR_DETECTOR_CHECK_CUDA(
        cudaMemcpyAsync(h_num_, d_num_, sizeof(int), cudaMemcpyDeviceToHost, stream_));
    ARMOR_DETECTOR_CHECK_CUDA(cudaMemcpyAsync(
        h_boxes_, d_boxes_, static_cast<std::size_t>(keep_topk_) * 4 * sizeof(float),
        cudaMemcpyDeviceToHost, stream_));
    ARMOR_DETECTOR_CHECK_CUDA(cudaMemcpyAsync(
        h_scores_, d_scores_, static_cast<std::size_t>(keep_topk_) * sizeof(float),
        cudaMemcpyDeviceToHost, stream_));
    ARMOR_DETECTOR_CHECK_CUDA(cudaMemcpyAsync(
        h_classes_, d_classes_, static_cast<std::size_t>(keep_topk_) * sizeof(int),
        cudaMemcpyDeviceToHost, stream_));
    ARMOR_DETECTOR_CHECK_CUDA(cudaMemcpyAsync(
        h_kpts_, d_kpts_,
        static_cast<std::size_t>(keep_topk_) * kpt_channels_ * sizeof(float),
        cudaMemcpyDeviceToHost, stream_));
    ARMOR_DETECTOR_CHECK_CUDA(cudaStreamSynchronize(stream_));
    preprocessor_->StageHost(rgb_img);

    ARMOR_DETECTOR_CHECK_CUDA(
        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));

    preprocessor_->Launch(static_cast<float*>(d_input_), stream_);
    context_->enqueueV3(stream_);

    cudaMemcpyAsync(h_num_, d_num_, sizeof(int), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(h_boxes_, d_boxes_,
                    static_cast<std::size_t>(keep_topk_) * 4 * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(h_scores_, d_scores_,
                    static_cast<std::size_t>(keep_topk_) * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(h_classes_, d_classes_,
                    static_cast<std::size_t>(keep_topk_) * sizeof(int),
                    cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(h_kpts_, d_kpts_,
                    static_cast<std::size_t>(keep_topk_) * kpt_channels_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);

    ARMOR_DETECTOR_CHECK_CUDA(cudaStreamEndCapture(stream_, &graph_));
    ARMOR_DETECTOR_CHECK_CUDA(
        cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));
    graph_ready_ = true;

    RCLCPP_INFO(rclcpp::get_logger("armor_detector"),
                "CUDA Graph captured and instantiated");
  }
  else
  {
    cudaGraphLaunch(graph_exec_, stream_);
    cudaStreamSynchronize(stream_);
  }

  auto t_infer_end = std::chrono::steady_clock::now();

  output.end_to_end.count = h_num_[0];
  output.end_to_end.kpt_channels = kpt_channels_;
  output.end_to_end.scores =
      std::span<const float>(h_scores_, static_cast<std::size_t>(keep_topk_));
  output.end_to_end.classes =
      std::span<const int>(h_classes_, static_cast<std::size_t>(keep_topk_));
  output.end_to_end.keypoints =
      std::span<const float>(h_kpts_, static_cast<std::size_t>(keep_topk_) *
                                          static_cast<std::size_t>(kpt_channels_));

  using us = std::chrono::microseconds;
  output.latencies.emplace_back(
      "Preprocess",
      static_cast<std::uint64_t>(
          std::chrono::duration_cast<us>(t_infer_start - t_pre_start).count()));
  output.latencies.emplace_back(
      "Inference",
      static_cast<std::uint64_t>(
          std::chrono::duration_cast<us>(t_infer_end - t_infer_start).count()));

  return output;
}

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR_HAS_TENSORRT
