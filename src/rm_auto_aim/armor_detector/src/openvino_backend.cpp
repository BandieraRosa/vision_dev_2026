#include "armor_detector/openvino_backend.hpp"

#if ARMOR_DETECTOR_HAS_OPENVINO

#include <algorithm>
#include <chrono>
#include <exception>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>

namespace rm_auto_aim
{

OpenVinoBackend::OpenVinoBackend(const YoloParams& params) : params_(params)
{
  auto model = core_.read_model(params_.model_path);

  ov::PartialShape input_shape{1, params_.input_size, params_.input_size, 3};

  ov::preprocess::PrePostProcessor ppp(model);
  auto& input = ppp.input();

  input.tensor()
      .set_element_type(ov::element::u8)
      .set_shape(input_shape)
      .set_layout("NHWC")
      .set_color_format(ov::preprocess::ColorFormat::RGB);

  input.model().set_layout("NCHW");
  input.preprocess().convert_element_type(ov::element::f32).scale(255.0);

  model = ppp.build();

  ov::AnyMap config = {
      ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
      ov::hint::inference_precision(ov::element::f32),
      ov::hint::num_requests(1),
  };
  compiled_model_ = core_.compile_model(model, params_.device, config);
  infer_request_ = compiled_model_.create_infer_request();

  input_tensor_ = infer_request_.get_input_tensor();
  input_mat_ =
      cv::Mat(params_.input_size, params_.input_size, CV_8UC3, input_tensor_.data(),
              static_cast<std::size_t>(params_.input_size) * 3);
  input_mat_.setTo(cv::Scalar(0, 0, 0));

  try
  {
    infer_request_.infer();
  }
  catch (const std::exception& e)
  {
    RCLCPP_WARN(rclcpp::get_logger("armor_detector"),
                "OpenVINO warm-up infer failed (non-fatal): %s", e.what());
  }
}

void OpenVinoBackend::RefreshLetterboxCache(int rows, int cols)
{
  if (rows == last_rows_ && cols == last_cols_)
  {
    return;
  }

  double x_scale = static_cast<double>(params_.input_size) / rows;
  double y_scale = static_cast<double>(params_.input_size) / cols;
  double scale = std::min(x_scale, y_scale);
  int new_h = static_cast<int>(rows * scale);
  int new_w = static_cast<int>(cols * scale);

  input_mat_.setTo(cv::Scalar(0, 0, 0));

  last_rows_ = rows;
  last_cols_ = cols;
  cached_scale_ = scale;
  cached_h_ = new_h;
  cached_w_ = new_w;
}

YoloInferenceOutput OpenVinoBackend::Infer(const cv::Mat& rgb_img)
{
  YoloInferenceOutput output;
  output.layout =
      params_.end_to_end ? YoloOutputLayout::END_TO_END : YoloOutputLayout::RAW;
  if (rgb_img.empty())
  {
    return output;
  }

  auto t_pre_start = std::chrono::steady_clock::now();

  RefreshLetterboxCache(rgb_img.rows, rgb_img.cols);
  cv::Rect roi(0, 0, cached_w_, cached_h_);
  cv::resize(rgb_img, input_mat_(roi), cv::Size(cached_w_, cached_h_));

  auto t_infer_start = std::chrono::steady_clock::now();
  infer_request_.infer();
  auto t_infer_end = std::chrono::steady_clock::now();

  output.scale = cached_scale_;
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

    output.end_to_end.count = n;
    output.end_to_end.kpt_channels = kpt_channels;
    output.end_to_end.scores =
        std::span<const float>(scores, static_cast<std::size_t>(keep_topk));
    output.end_to_end.classes =
        std::span<const int>(classes, static_cast<std::size_t>(keep_topk));
    output.end_to_end.keypoints =
        std::span<const float>(kpts, static_cast<std::size_t>(keep_topk) *
                                         static_cast<std::size_t>(kpt_channels));
  }
  else
  {
    auto output_tensor = infer_request_.get_output_tensor();
    const auto& output_shape = output_tensor.get_shape();
    output.raw_output =
        cv::Mat(static_cast<int>(output_shape[1]), static_cast<int>(output_shape[2]),
                CV_32F, output_tensor.data());
  }

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

#endif  // ARMOR_DETECTOR_HAS_OPENVINO
