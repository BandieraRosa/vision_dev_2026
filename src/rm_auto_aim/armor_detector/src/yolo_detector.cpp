#include "armor_detector/yolo_detector.hpp"

#include <algorithm>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "armor_detector/detector_base.hpp"

namespace rm_auto_aim
{
namespace
{

static const std::array<std::string, 38> YOLO11_MODEL_LABELS = {
    "Bsentry",       "Rsentry",       "Esentry",      "Bone",         "Rone",
    "Eone",          "Btwo",          "Rtwo",         "Etwo",         "Bthree",
    "Rthree",        "Ethree",        "Bfour",        "Rfour",        "Efour",
    "Bfive",         "Rfive",         "Efive",        "Boutpost",     "Routpost",
    "Eoutpost",      "Bbase",         "Rbase",        "Ebase",        "Pbase",
    "Bbasesmall",    "Rbasesmall",    "Ebasesmall",   "Pbasesmall",   "Bbalancethree",
    "Rbalancethree", "Ebalancethree", "Bbalancefour", "Rbalancefour", "Ebalancefour",
    "Bbalancefive",  "Rbalancefive",  "Ebalancefive"};

template <typename T>
T get_parameter(rclcpp::Node& node, const std::string& name, const T& default_value)
{
  if (!node.has_parameter(name))
  {
    return node.declare_parameter<T>(name, default_value);
  }
  return node.get_parameter(name).get_value<T>();
}

std::string map_label(std::string_view raw_label)
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
  if (raw_label == "three")
  {
    return "3";
  }
  if (raw_label == "balancethree")
  {
    return "3";
  }
  if (raw_label == "four")
  {
    return "4";
  }
  if (raw_label == "balancefour")
  {
    return "4";
  }
  if (raw_label == "five")
  {
    return "5";
  }
  if (raw_label == "balancefive")
  {
    return "5";
  }
  if (raw_label == "outpost")
  {
    return "outpost";
  }
  if (raw_label == "base")
  {
    return "base";
  }
  if (raw_label == "basesmall")
  {
    return "base";
  }

  return "negative";
}

#if ARMOR_DETECTOR_HAS_TENSORRT

class TRTLogger : public nvinfer1::ILogger
{
 public:
  void log(Severity severity, const char* msg) noexcept override
  {
    // 只打印 WARNING 及以上，避免 INFO 刷屏
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

inline void ensure_trt_plugins_initialized()
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

inline bool has_dynamic_dim(const nvinfer1::Dims& dims)
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

inline int64_t tensor_volume(const nvinfer1::Dims& dims)
{
  int64_t v = 1;
  for (int i = 0; i < dims.nbDims; ++i)
  {
    if (dims.d[i] < 0)
    {
      return -1;
    }
    v *= dims.d[i];
  }
  return v;
}

inline std::string dims_to_string(const nvinfer1::Dims& dims)
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

inline std::vector<char> read_binary_file(const std::string& path)
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

#endif  // ARMOR_DETECTOR_HAS_TENSORRT

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

  auto pkg_path = ament_index_cpp::get_package_share_directory("armor_detector");
  auto model_path = pkg_path + "/model/" + model_name;
  if (!std::filesystem::exists(model_path))
  {
    RCLCPP_ERROR(node.get_logger(), "YOLO model file not found: %s", model_path.c_str());
    throw std::runtime_error("YOLO model file not found: " + model_path);
  }

  YoloParams yolo_params = {
      .model_path = model_path,
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

#if ARMOR_DETECTOR_HAS_TENSORRT

namespace
{
template <typename T>
inline void cuda_alloc_device(T** p, std::size_t n)
{
  ARMOR_DETECTOR_CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(p), n * sizeof(T)));
}
template <typename T>
inline void cuda_alloc_pinned(T** p, std::size_t n)
{
  ARMOR_DETECTOR_CHECK_CUDA(
      cudaHostAlloc(reinterpret_cast<void**>(p), n * sizeof(T), cudaHostAllocDefault));
}
}  // namespace

YoloDetector::YoloDetector(const YoloParams& params)
    : params_(params), class_num_(static_cast<int>(YOLO11_MODEL_LABELS.size()))
{
  if (params_.end_to_end)
  {
    InitTrtEnd2End();
  }
  else
  {
    InitTrtRaw();
  }
}

void YoloDetector::InitTrtRaw()
{
  auto engine_data = read_binary_file(params_.model_path);

  trt_runtime_.reset(nvinfer1::createInferRuntime(get_trt_logger()));
  if (!trt_runtime_)
  {
    throw std::runtime_error("Failed to create TensorRT runtime");
  }

  trt_engine_.reset(
      trt_runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  if (!trt_engine_)
  {
    throw std::runtime_error("Failed to deserialize TensorRT engine: " +
                             params_.model_path);
  }

  trt_context_.reset(trt_engine_->createExecutionContext());
  if (!trt_context_)
  {
    throw std::runtime_error("Failed to create TensorRT execution context");
  }

  for (int i = 0; i < trt_engine_->getNbIOTensors(); ++i)
  {
    const char* name = trt_engine_->getIOTensorName(i);
    auto mode = trt_engine_->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT)
    {
      trt_input_name_ = name;
    }
    else if (mode == nvinfer1::TensorIOMode::kOUTPUT)
    {
      trt_output_name_ = name;
    }
  }
  if (trt_input_name_.empty() || trt_output_name_.empty())
  {
    throw std::runtime_error("Cannot find input/output tensor names in engine");
  }

  nvinfer1::Dims input_dims_raw = trt_engine_->getTensorShape(trt_input_name_.c_str());
  if (has_dynamic_dim(input_dims_raw))
  {
    nvinfer1::Dims4 real_input_dims{1, 3, params_.input_size, params_.input_size};
    if (!trt_context_->setInputShape(trt_input_name_.c_str(), real_input_dims))
    {
      throw std::runtime_error("Failed to set TensorRT input shape");
    }
  }

  trt_input_dims_ = trt_context_->getTensorShape(trt_input_name_.c_str());
  trt_output_dims_ = trt_context_->getTensorShape(trt_output_name_.c_str());

  RCLCPP_INFO(rclcpp::get_logger("armor_detector"), "TensorRT input = %s, output = %s",
              dims_to_string(trt_input_dims_).c_str(),
              dims_to_string(trt_output_dims_).c_str());

  if (has_dynamic_dim(trt_input_dims_) || has_dynamic_dim(trt_output_dims_))
  {
    throw std::runtime_error("Dynamic dims unresolved after setInputShape");
  }

  if (trt_engine_->getTensorDataType(trt_input_name_.c_str()) !=
          nvinfer1::DataType::kFLOAT ||
      trt_engine_->getTensorDataType(trt_output_name_.c_str()) !=
          nvinfer1::DataType::kFLOAT)
  {
    throw std::runtime_error("TensorRT engine must have FP32 IO for this detector");
  }

  if (trt_output_dims_.nbDims != 3)
  {
    throw std::runtime_error("Unexpected output rank for YOLO, expected 3 but got " +
                             std::to_string(trt_output_dims_.nbDims));
  }

  int64_t input_elems = tensor_volume(trt_input_dims_);
  int64_t output_elems = tensor_volume(trt_output_dims_);
  if (input_elems <= 0 || output_elems <= 0)
  {
    throw std::runtime_error("Invalid tensor volume");
  }

  trt_input_bytes_ = static_cast<std::size_t>(input_elems) * sizeof(float);
  trt_output_bytes_ = static_cast<std::size_t>(output_elems) * sizeof(float);

  // trt_host_input_ 不再需要 (预处理已全部在 GPU)
  trt_host_output_.resize(static_cast<std::size_t>(output_elems));

  ARMOR_DETECTOR_CHECK_CUDA(cudaMalloc(&trt_d_input_, trt_input_bytes_));
  ARMOR_DETECTOR_CHECK_CUDA(cudaMalloc(&trt_d_output_, trt_output_bytes_));
  ARMOR_DETECTOR_CHECK_CUDA(cudaStreamCreate(&trt_stream_));

  if (!trt_context_->setTensorAddress(trt_input_name_.c_str(), trt_d_input_))
  {
    throw std::runtime_error("Failed to bind TensorRT input tensor address");
  }
  if (!trt_context_->setTensorAddress(trt_output_name_.c_str(), trt_d_output_))
  {
    throw std::runtime_error("Failed to bind TensorRT output tensor address");
  }

  // --- GPU 预处理模块 ---
  GpuPreprocessor::Config pp_cfg;
  pp_cfg.dst_size = params_.input_size;
  pp_cfg.swap_rb = false;  // 输入 rgb_img 已是模型期望顺序
  preprocessor_ = std::make_unique<GpuPreprocessor>(pp_cfg);
}

void YoloDetector::InitTrtEnd2End()
{
  cudaError_t err = cudaSetDeviceFlags(cudaDeviceScheduleSpin);
  unsigned int flags = 0;
  cudaGetDeviceFlags(&flags);
  RCLCPP_INFO(rclcpp::get_logger("armor_detector"),
              "setDeviceFlags=%s, current flags=0x%x, spin=%d", cudaGetErrorString(err),
              flags, (flags & cudaDeviceScheduleMask) == cudaDeviceScheduleSpin);
  ensure_trt_plugins_initialized();
  auto engine_data = read_binary_file(params_.model_path);

  trt_runtime_.reset(nvinfer1::createInferRuntime(get_trt_logger()));
  if (!trt_runtime_)
  {
    throw std::runtime_error("createInferRuntime failed");
  }

  trt_engine_.reset(
      trt_runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  if (!trt_engine_)
  {
    throw std::runtime_error("deserializeCudaEngine failed: " + params_.model_path);
  }

  trt_context_.reset(trt_engine_->createExecutionContext());
  if (!trt_context_)
  {
    throw std::runtime_error("createExecutionContext failed");
  }

  for (int i = 0; i < trt_engine_->getNbIOTensors(); ++i)
  {
    const char* name = trt_engine_->getIOTensorName(i);
    auto mode = trt_engine_->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT)
    {
      in_name_ = name;
    }
    else
    {
      std::string n = name;
      if (n == "num_dets")
      {
        out_num_name_ = n;
      }
      else if (n == "det_boxes")
      {
        out_boxes_name_ = n;
      }
      else if (n == "det_scores")
      {
        out_scores_name_ = n;
      }
      else if (n == "det_classes")
      {
        out_classes_name_ = n;
      }
      else if (n == "det_kpts")
      {
        out_kpts_name_ = n;
      }
    }
  }
  if (in_name_.empty() || out_num_name_.empty() || out_boxes_name_.empty() ||
      out_scores_name_.empty() || out_classes_name_.empty() || out_kpts_name_.empty())
  {
    throw std::runtime_error(
        "Engine IO names mismatch. Expected one input and outputs: "
        "num_dets / det_boxes / det_scores / det_classes / det_kpts");
  }

  auto raw_in = trt_engine_->getTensorShape(in_name_.c_str());
  if (has_dynamic_dim(raw_in))
  {
    nvinfer1::Dims4 shp{1, 3, params_.input_size, params_.input_size};
    if (!trt_context_->setInputShape(in_name_.c_str(), shp))
    {
      throw std::runtime_error("setInputShape failed");
    }
  }

  auto in_dims = trt_context_->getTensorShape(in_name_.c_str());
  auto boxes_dims = trt_context_->getTensorShape(out_boxes_name_.c_str());
  auto kpts_dims = trt_context_->getTensorShape(out_kpts_name_.c_str());
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

  // h_input_ 不再需要 (预处理已全部在 GPU), 输出 pinned 保留
  cuda_alloc_pinned(&h_num_, static_cast<std::size_t>(1));
  cuda_alloc_pinned(&h_boxes_, static_cast<std::size_t>(keep_topk_) * 4);
  cuda_alloc_pinned(&h_scores_, static_cast<std::size_t>(keep_topk_));
  cuda_alloc_pinned(&h_classes_, static_cast<std::size_t>(keep_topk_));
  cuda_alloc_pinned(&h_kpts_, static_cast<std::size_t>(keep_topk_) * kpt_channels_);

  ARMOR_DETECTOR_CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

  auto bind = [&](const std::string& n, void* p)
  {
    if (!trt_context_->setTensorAddress(n.c_str(), p))
    {
      throw std::runtime_error("setTensorAddress failed: " + n);
    }
  };
  bind(in_name_, d_input_);
  bind(out_num_name_, d_num_);
  bind(out_boxes_name_, d_boxes_);
  bind(out_scores_name_, d_scores_);
  bind(out_classes_name_, d_classes_);
  bind(out_kpts_name_, d_kpts_);

  // --- GPU 预处理模块 ---
  GpuPreprocessor::Config pp_cfg;
  pp_cfg.dst_size = params_.input_size;
  pp_cfg.swap_rb = false;
  preprocessor_ = std::make_unique<GpuPreprocessor>(pp_cfg);

  // cudaEventCreateWithFlags(&ev_start_, cudaEventDefault);
  // cudaEventCreateWithFlags(&ev_end_, cudaEventDefault);
}

YoloDetector::~YoloDetector()
{
  if (graph_exec_)
  {
    cudaGraphExecDestroy(graph_exec_);
  }
  if (graph_)
  {
    cudaGraphDestroy(graph_);
  }

  trt_context_.reset();
  trt_engine_.reset();
  trt_runtime_.reset();

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

  if (trt_stream_ != nullptr)
  {
    cudaStreamDestroy(trt_stream_);
    trt_stream_ = nullptr;
  }
  if (trt_d_input_ != nullptr)
  {
    cudaFree(trt_d_input_);
    trt_d_input_ = nullptr;
  }
  if (trt_d_output_ != nullptr)
  {
    cudaFree(trt_d_output_);
    trt_d_output_ = nullptr;
  }
}

DetectionResult YoloDetector::Detect(const cv::Mat& rgb_img)
{
  auto total_start_time = std::chrono::steady_clock::now();
  DetectionResult result;
  if (params_.end_to_end)
  {
    result = DetectTrtEnd2End(rgb_img);
  }
  else
  {
    result = DetectTrtRaw(rgb_img);
  }
  auto total_end_time = std::chrono::steady_clock::now();
  auto total_latency = std::chrono::duration_cast<std::chrono::microseconds>(
                           total_end_time - total_start_time)
                           .count();
  debug_latencies_.emplace_back("Total", static_cast<uint64_t>(total_latency));
  return result;
}

DetectionResult YoloDetector::DetectTrtRaw(const cv::Mat& rgb_img)
{
  debug_latencies_.clear();
  DetectionResult result;
  if (rgb_img.empty())
  {
    return result;
  }

  auto infer_start_time = std::chrono::steady_clock::now();

  const double SCALE =
      preprocessor_->Run(rgb_img, static_cast<float*>(trt_d_input_), trt_stream_);

  if (!trt_context_->enqueueV3(trt_stream_))
  {
    throw std::runtime_error("TensorRT enqueueV3 failed");
  }

  ARMOR_DETECTOR_CHECK_CUDA(cudaMemcpyAsync(trt_host_output_.data(), trt_d_output_,
                                            trt_output_bytes_, cudaMemcpyDeviceToHost,
                                            trt_stream_));
  ARMOR_DETECTOR_CHECK_CUDA(cudaStreamSynchronize(trt_stream_));

  auto infer_end_time = std::chrono::steady_clock::now();
  auto infer_latency = std::chrono::duration_cast<std::chrono::microseconds>(
                           infer_end_time - infer_start_time)
                           .count();
  debug_latencies_.emplace_back("Inference", static_cast<uint64_t>(infer_latency));

  auto parse_start_time = std::chrono::steady_clock::now();
  cv::Mat output(static_cast<int>(trt_output_dims_.d[1]),
                 static_cast<int>(trt_output_dims_.d[2]), CV_32F,
                 trt_host_output_.data());
  last_armors_ = Parse(SCALE, output);
  result.armors = last_armors_;
  auto parse_end_time = std::chrono::steady_clock::now();
  auto parse_latency = std::chrono::duration_cast<std::chrono::microseconds>(
                           parse_end_time - parse_start_time)
                           .count();
  debug_latencies_.emplace_back("Parse Output", static_cast<uint64_t>(parse_latency));

  return result;
}

DetectionResult YoloDetector::DetectTrtEnd2End(const cv::Mat& rgb_img)
{
  // auto t0 = std::chrono::steady_clock::now();
  debug_latencies_.clear();
  DetectionResult result;
  if (rgb_img.empty())
  {
    return result;
  }

  auto t_pre_start = std::chrono::steady_clock::now();

  preprocessor_->EnsureInitialized(rgb_img.rows, rgb_img.cols);
  preprocessor_->StageHost(rgb_img);
  // auto t1 = std::chrono::steady_clock::now();
  const double SCALE = preprocessor_->GetScale();

  auto t_infer_start = std::chrono::steady_clock::now();

  // auto t2 = std::chrono::steady_clock::now();
  // auto t3 = std::chrono::steady_clock::now();
  if (!graph_ready_)
  {
    preprocessor_->Launch(static_cast<float*>(d_input_), stream_);
    if (!trt_context_->enqueueV3(stream_))
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
    // t2 = std::chrono::steady_clock::now();
    ARMOR_DETECTOR_CHECK_CUDA(cudaStreamSynchronize(stream_));
    // t3 = std::chrono::steady_clock::now();
    preprocessor_->StageHost(rgb_img);

    // capture
    ARMOR_DETECTOR_CHECK_CUDA(
        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));

    preprocessor_->Launch(static_cast<float*>(d_input_), stream_);
    trt_context_->enqueueV3(stream_);

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
    // cudaEventRecord(ev_start_, stream_);
    cudaGraphLaunch(graph_exec_, stream_);
    // cudaEventRecord(ev_end_, stream_);
    // t2 = std::chrono::steady_clock::now();
    cudaStreamSynchronize(stream_);
    // t3 = std::chrono::steady_clock::now();
    // float gpu_ms = 0.f;
    // cudaEventElapsedTime(&gpu_ms, ev_start_, ev_end_);
    // RCLCPP_WARN(rclcpp::get_logger("armor_detector"), "GPU time (CUDA Graph): %.2f ms",
    //             gpu_ms);
  }

  auto t_infer_end = std::chrono::steady_clock::now();

  auto t_parse_start = std::chrono::steady_clock::now();
  last_armors_ = ParseEnd2End(SCALE);
  // auto t4 = std::chrono::steady_clock::now();
  result.armors = last_armors_;
  auto t_parse_end = std::chrono::steady_clock::now();

  using us = std::chrono::microseconds;
  debug_latencies_.emplace_back(
      "Preprocess",
      static_cast<uint64_t>(
          std::chrono::duration_cast<us>(t_infer_start - t_pre_start).count()));
  debug_latencies_.emplace_back(
      "Inference",
      static_cast<uint64_t>(
          std::chrono::duration_cast<us>(t_infer_end - t_infer_start).count()));
  debug_latencies_.emplace_back(
      "Parse Output",
      static_cast<uint64_t>(
          std::chrono::duration_cast<us>(t_parse_end - t_parse_start).count()));
  // RCLCPP_INFO(rclcpp::get_logger("armor_detector"),
  //             "t1-t0: %lu us, t2-t1: %lu us, t3-t2: %lu us, t4-t3: %lu us",
  //             static_cast<uint64_t>(std::chrono::duration_cast<us>(t1 - t0).count()),
  //             static_cast<uint64_t>(std::chrono::duration_cast<us>(t2 - t1).count()),
  //             static_cast<uint64_t>(std::chrono::duration_cast<us>(t3 - t2).count()),
  //             static_cast<uint64_t>(std::chrono::duration_cast<us>(t4 - t3).count()));
  return result;
}

std::vector<Armor> YoloDetector::ParseEnd2End(double scale)
{
  std::vector<Armor> armors;
  int n = h_num_[0];
  if (n <= 0)
  {
    return armors;
  }
  if (n > keep_topk_)
  {
    n = keep_topk_;
  }
  armors.reserve(static_cast<std::size_t>(n));

  float inv_scale = static_cast<float>(1.0 / scale);
  int kp_dim = (kpt_channels_ >= params_.num_keypoints * 3) ? 3 : 2;

  for (int i = 0; i < n; ++i)
  {
    float conf = h_scores_[i];
    if (conf < params_.min_confidence)
    {
      continue;
    }

    int cls = h_classes_[i];
    if (cls < 0 || cls >= class_num_)
    {
      continue;
    }

    const auto& raw_label = YOLO11_MODEL_LABELS[cls];

    // 颜色过滤 (前置, 尽早剪枝)
    int color = (raw_label[0] == 'R')   ? RED
                : (raw_label[0] == 'B') ? BLUE
                                        : params_.detect_color;
    if (color != params_.detect_color)
    {
      continue;
    }

    // ignore_classes 过滤
    std::string label = map_label(raw_label);
    if (std::find(params_.ignore_classes.begin(), params_.ignore_classes.end(), label) !=
        params_.ignore_classes.end())
    {
      continue;
    }

    // 取 4 个关键点 (输入 640 尺度 -> 原图尺度)
    const float* kp = h_kpts_ + static_cast<std::size_t>(i) * kpt_channels_;
    std::vector<cv::Point2f> kps;
    kps.reserve(static_cast<std::size_t>(params_.num_keypoints));
    for (int k = 0; k < params_.num_keypoints; ++k)
    {
      kps.emplace_back(kp[static_cast<ptrdiff_t>(k * kp_dim)] * inv_scale,
                       kp[k * kp_dim + 1] * inv_scale);
    }
    SortKeypoints(kps);

    Light ll(kps[0], kps[1], color);
    Light rl(kps[2], kps[3], color);

    Armor armor(ll, rl);
    armor.type = DetermineArmorType(ll, rl);
    armor.number = label;
    armor.confidence = conf;

    std::ostringstream oss;
    oss << label << ": " << std::fixed << std::setprecision(2) << conf;
    armor.classfication_result = oss.str();

    armors.emplace_back(std::move(armor));
  }

  return armors;
}

#elif ARMOR_DETECTOR_HAS_OPENVINO

YoloDetector::YoloDetector(const YoloParams& params)
    : params_(params), class_num_(static_cast<int>(YOLO11_MODEL_LABELS.size()))
{
  auto model = core_.read_model(params_.model_path);

  // 预处理
  ov::preprocess::PrePostProcessor ppp(model);
  auto& input = ppp.input();

  input.tensor()
      .set_element_type(ov::element::u8)
      .set_shape({1, params_.input_size, params_.input_size, 3})
      .set_layout("NHWC")
      .set_color_format(ov::preprocess::ColorFormat::RGB);

  input.model().set_layout("NCHW");

  input.preprocess().convert_element_type(ov::element::f32).scale(255.0);

  model = ppp.build();
  compiled_model_ =
      core_.compile_model(model, params_.device,
                          ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
  infer_request_ = compiled_model_.create_infer_request();
}

YoloDetector::~YoloDetector() = default;

DetectionResult YoloDetector::Detect(const cv::Mat& rgb_img)
{
  debug_latencies_.clear();
  DetectionResult result;
  auto infer_start_time = std::chrono::steady_clock::now();
  if (rgb_img.empty())
  {
    return result;
  }

  auto x_scale = static_cast<double>(params_.input_size) / rgb_img.rows;
  auto y_scale = static_cast<double>(params_.input_size) / rgb_img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(rgb_img.rows * scale);
  auto w = static_cast<int>(rgb_img.cols * scale);

  cv::Mat input_mat(params_.input_size, params_.input_size, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Rect roi(0, 0, w, h);
  cv::resize(rgb_img, input_mat(roi), {w, h});

  ov::Tensor input_tensor(ov::element::u8,
                          {1, static_cast<uint64_t>(params_.input_size),
                           static_cast<uint64_t>(params_.input_size), 3},
                          input_mat.data);
  infer_request_.set_input_tensor(input_tensor);
  infer_request_.infer();

  auto infer_end_time = std::chrono::steady_clock::now();
  auto infer_latency = std::chrono::duration_cast<std::chrono::microseconds>(
                           infer_end_time - infer_start_time)
                           .count();
  debug_latencies_.emplace_back("Inference", static_cast<uint64_t>(infer_latency));

  auto parse_start_time = std::chrono::steady_clock::now();

  if (params_.end_to_end)
  {
    auto num_tensor = infer_request_.get_tensor("num_dets");
    auto scores_tensor = infer_request_.get_tensor("det_scores");
    auto classes_tensor = infer_request_.get_tensor("det_classes");
    auto kpts_tensor = infer_request_.get_tensor("det_kpts");

    const int* num = static_cast<const int*>(num_tensor.data());
    const float* scores = static_cast<const float*>(scores_tensor.data());
    const int* classes = static_cast<const int*>(classes_tensor.data());
    const float* kpts = static_cast<const float*>(kpts_tensor.data());

    const auto& scores_shape = scores_tensor.get_shape();
    const auto& kpts_shape = kpts_tensor.get_shape();

    int n = num[0];
    int keep_topk = scores_shape.size() >= 2 ? static_cast<int>(scores_shape[1]) : 0;
    int kpt_channels = kpts_shape.size() >= 3 ? static_cast<int>(kpts_shape[2]) : 0;

    if (n < 0)
    {
      n = 0;
    }
    if (n > keep_topk)
    {
      n = keep_topk;
    }

    last_armors_ = ParseOpenVinoEnd2End(scale, n, scores, classes, kpts, kpt_channels);
  }
  else
  {
    // 输出
    auto output_tensor = infer_request_.get_output_tensor();
    const auto& output_shape = output_tensor.get_shape();
    cv::Mat output(static_cast<int>(output_shape[1]), static_cast<int>(output_shape[2]),
                   CV_32F, output_tensor.data());

    // 后处理
    last_armors_ = Parse(scale, output);
  }

  result.armors = last_armors_;
  auto parse_end_time = std::chrono::steady_clock::now();
  auto parse_latency = std::chrono::duration_cast<std::chrono::microseconds>(
                           parse_end_time - parse_start_time)
                           .count();
  debug_latencies_.emplace_back("Parse Output", static_cast<uint64_t>(parse_latency));
  return result;
}

std::vector<Armor> YoloDetector::ParseOpenVinoEnd2End(double scale, int n,
                                                      const float* scores,
                                                      const int* classes,
                                                      const float* kpts, int kpt_channels)
{
  std::vector<Armor> armors;
  if (n <= 0 || scores == nullptr || classes == nullptr || kpts == nullptr ||
      kpt_channels < params_.num_keypoints * 2)
  {
    return armors;
  }

  armors.reserve(static_cast<std::size_t>(n));

  float inv_scale = static_cast<float>(1.0 / scale);
  int kp_dim = (kpt_channels >= params_.num_keypoints * 3) ? 3 : 2;

  for (int i = 0; i < n; ++i)
  {
    float conf = scores[i];

    // build_openvino_end2end.py 会把 padding/无效行的 score/class 标成 -1。
    if (conf < params_.min_confidence)
    {
      continue;
    }

    int cls = classes[i];
    if (cls < 0 || cls >= class_num_)
    {
      continue;
    }

    const auto& raw_label = YOLO11_MODEL_LABELS[cls];

    int color = (raw_label[0] == 'R')   ? RED
                : (raw_label[0] == 'B') ? BLUE
                                        : params_.detect_color;
    if (color != params_.detect_color)
    {
      continue;
    }

    std::string label = map_label(raw_label);
    if (std::find(params_.ignore_classes.begin(), params_.ignore_classes.end(), label) !=
        params_.ignore_classes.end())
    {
      continue;
    }

    const float* kp = kpts + static_cast<std::size_t>(i) * kpt_channels;

    std::vector<cv::Point2f> keypoints;
    keypoints.reserve(static_cast<std::size_t>(params_.num_keypoints));
    for (int k = 0; k < params_.num_keypoints; ++k)
    {
      keypoints.emplace_back(kp[k * kp_dim] * inv_scale, kp[k * kp_dim + 1] * inv_scale);
    }

    SortKeypoints(keypoints);

    Light ll(keypoints[0], keypoints[1], color);
    Light rl(keypoints[2], keypoints[3], color);

    Armor armor(ll, rl);
    armor.type = DetermineArmorType(ll, rl);
    armor.number = label;
    armor.confidence = conf;

    std::ostringstream oss;
    oss << label << ": " << std::fixed << std::setprecision(2) << conf;
    armor.classfication_result = oss.str();

    armors.emplace_back(std::move(armor));
  }

  return armors;
}

#endif  // ARMOR_DETECTOR_HAS_TENSORRT / ARMOR_DETECTOR_HAS_OPENVINO

// 输出张量布局: [x, y, w, h, cls_0, ..., cls_N, kp0_x, kp0_y, ...]
std::vector<Armor> YoloDetector::Parse(double scale, cv::Mat& output)
{
  // 转置：[features, num_detections] → [num_detections, features]
  cv::transpose(output, output);

  int kp_start = 4 + class_num_;
  int total_cols = output.cols;
  int kp_cols = total_cols - kp_start;
  int kp_stride = (kp_cols >= params_.num_keypoints * 3) ? 3 : 2;

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> all_keypoints;

  for (int r = 0; r < output.rows; r++)
  {
    // 取类别分数
    auto scores = output.row(r).colRange(4, 4 + class_num_);

    double score = NAN;
    cv::Point max_point;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &max_point);

    if (score < params_.score_threshold)
    {
      continue;
    }

    // 边界框
    auto x = output.row(r).at<float>(0);
    auto y = output.row(r).at<float>(1);
    auto w = output.row(r).at<float>(2);
    auto h = output.row(r).at<float>(3);
    auto left = static_cast<int>((x - 0.5 * w) / scale);
    auto top = static_cast<int>((y - 0.5 * h) / scale);
    auto width = static_cast<int>(w / scale);
    auto height = static_cast<int>(h / scale);

    // 关键点
    std::vector<cv::Point2f> keypoints;
    keypoints.reserve(params_.num_keypoints);
    for (int i = 0; i < params_.num_keypoints; i++)
    {
      float kx =
          output.row(r).at<float>(kp_start + i * kp_stride) / static_cast<float>(scale);
      float ky = output.row(r).at<float>(kp_start + i * kp_stride + 1) /
                 static_cast<float>(scale);
      keypoints.emplace_back(kx, ky);
    }

    class_ids.emplace_back(max_point.x);
    confidences.emplace_back(static_cast<float>(score));
    boxes.emplace_back(left, top, width, height);
    all_keypoints.emplace_back(std::move(keypoints));
  }

  // NMS
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, params_.score_threshold, params_.nms_threshold,
                    indices);

  std::vector<Armor> armors;
  armors.reserve(indices.size());

  for (const auto& i : indices)
  {
    int class_id = class_ids[i];
    if (class_id < 0 || class_id >= class_num_)
    {
      continue;
    }

    const auto& raw_label = YOLO11_MODEL_LABELS[class_id];
    std::string label = map_label(raw_label);

    // 跳过忽略类别
    if (std::find(params_.ignore_classes.begin(), params_.ignore_classes.end(), label) !=
        params_.ignore_classes.end())
    {
      continue;
    }

    // 置信度过滤
    if (confidences[i] < params_.min_confidence)
    {
      continue;
    }

    // 关键点
    auto& kps = all_keypoints[i];
    SortKeypoints(kps);

    int color = (raw_label[0] == 'R')
                    ? RED
                    : ((raw_label[0] == 'B') ? BLUE : params_.detect_color);
    if (color != params_.detect_color)
    {
      continue;
    }

    Light left_light(kps[0], kps[1], color);
    Light right_light(kps[2], kps[3], color);

    // 构造 Armor
    Armor armor(left_light, right_light);

    // 由关键点几何比例判断大小类型
    armor.type = DetermineArmorType(left_light, right_light);

    armor.number = label;
    armor.confidence = confidences[i];

    std::ostringstream oss;
    oss << label << ": " << std::fixed << std::setprecision(2) << confidences[i];
    armor.classfication_result = oss.str();

    armors.emplace_back(armor);
  }

  return armors;
}

void YoloDetector::SortKeypoints(std::vector<cv::Point2f>& keypoints)
{
  if (keypoints.size() != 4)
  {
    return;
  }

  // 按 y 升序，分出上方两点和下方两点
  std::sort(keypoints.begin(), keypoints.end(),
            [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });

  std::vector<cv::Point2f> top_points = {keypoints[0], keypoints[1]};
  std::vector<cv::Point2f> bottom_points = {keypoints[2], keypoints[3]};

  // 上方两点按 x 升序
  std::sort(top_points.begin(), top_points.end(),
            [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });

  // 下方两点按 x 升序
  std::sort(bottom_points.begin(), bottom_points.end(),
            [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });

  keypoints[0] = top_points[0];
  keypoints[1] = bottom_points[0];
  keypoints[2] = top_points[1];
  keypoints[3] = bottom_points[1];
}

ArmorType YoloDetector::DetermineArmorType(const Light& light_1, const Light& light_2)
{
  auto avg_length = (light_1.length + light_2.length) / 2.0;
  auto center_distance = cv::norm(light_2.center - light_1.center);

  if (avg_length < 1e-6)
  {
    return ArmorType::INVALID;
  }

  double ratio = center_distance / avg_length;
  return ratio > params_.large_armor_ratio_threshold ? ArmorType::LARGE
                                                     : ArmorType::SMALL;
}

void YoloDetector::DrawResults(cv::Mat& img)
{
  // Draw Lights
  for (const auto& armor : last_armors_)
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

  // Draw armors
  for (const auto& armor : last_armors_)
  {
    cv::line(img, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0),
             2);
    cv::line(img, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0),
             2);
  }

  // Show numbers and confidence
  for (const auto& armor : last_armors_)
  {
    cv::putText(img, armor.classfication_result, armor.left_light.top,
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
  }
}

}  // namespace rm_auto_aim
