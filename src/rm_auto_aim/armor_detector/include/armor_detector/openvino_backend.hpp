#ifndef ARMOR_DETECTOR__OPENVINO_BACKEND_HPP_
#define ARMOR_DETECTOR__OPENVINO_BACKEND_HPP_

#if ARMOR_DETECTOR_HAS_OPENVINO

#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>

#include "armor_detector/inference_backend.hpp"
#include "armor_detector/yolo_common.hpp"

namespace rm_auto_aim
{

class OpenVinoBackend final : public InferenceBackend
{
 public:
  explicit OpenVinoBackend(const YoloParams& params);
  ~OpenVinoBackend() override = default;

  OpenVinoBackend(const OpenVinoBackend&) = delete;
  OpenVinoBackend& operator=(const OpenVinoBackend&) = delete;

  YoloBackendKind BackendKind() const noexcept override
  {
    return YoloBackendKind::OPENVINO;
  }

  YoloInferenceOutput Infer(const cv::Mat& rgb_img) override;

 private:
  void RefreshLetterboxCache(int rows, int cols);

  YoloParams params_;
  ov::Core core_;
  ov::CompiledModel compiled_model_;
  ov::InferRequest infer_request_;
  ov::Tensor input_tensor_;
  cv::Mat input_mat_;
  int last_rows_ = 0;
  int last_cols_ = 0;
  double cached_scale_ = 0.0;
  int cached_w_ = 0;
  int cached_h_ = 0;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR_HAS_OPENVINO
#endif  // ARMOR_DETECTOR__OPENVINO_BACKEND_HPP_
