#include "armor_detector/yolo11_adapter.hpp"

#include <algorithm>
#include <cstdio>
#include <opencv2/dnn.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <utility>

namespace rm_auto_aim
{

Yolo11Adapter::Yolo11Adapter(const YoloParams& params, YoloBackendKind backend_kind)
    : params_(params), backend_kind_(backend_kind)
{
  auto labels = default_yolo11_labels();
  if (!params_.metadata_path.empty())
  {
    std::string error_message;
    auto metadata_labels =
        load_yolo_labels_from_metadata(params_.metadata_path, &error_message);
    if (!metadata_labels.empty())
    {
      labels = std::move(metadata_labels);
      RCLCPP_INFO(rclcpp::get_logger("armor_detector"),
                  "Loaded %zu YOLO labels from metadata: %s", labels.size(),
                  params_.metadata_path.c_str());
    }
    else
    {
      RCLCPP_WARN(rclcpp::get_logger("armor_detector"),
                  "Failed to parse YOLO metadata '%s' (%s); using hardcoded YOLO11 "
                  "labels fallback",
                  params_.metadata_path.c_str(), error_message.c_str());
    }
  }
  else if (backend_kind_ == YoloBackendKind::TENSORRT)
  {
    RCLCPP_WARN(rclcpp::get_logger("armor_detector"),
                "yolo.metadata_path is empty for TensorRT; using hardcoded YOLO11 labels "
                "fallback. Set metadata_path for production/custom engines.");
  }

  class_map_ = build_yolo_class_map(labels, params_.ignore_classes);
}

std::vector<Armor> Yolo11Adapter::Decode(const YoloInferenceOutput& output)
{
  if (output.layout == YoloOutputLayout::END_TO_END)
  {
    return ParseEndToEnd(output.scale, output.end_to_end);
  }
  return ParseRaw(output.scale, output.raw_output);
}

void Yolo11Adapter::DrawResults(cv::Mat& img, const std::vector<Armor>& armors) const
{
  draw_yolo_armors(img, armors);
}

std::vector<Armor> Yolo11Adapter::ParseEndToEnd(double scale,
                                                const YoloEndToEndOutput& output) const
{
  std::vector<Armor> armors;
  int class_num = class_map_.ClassCount();
  if (scale <= 0.0 || output.count <= 0 || output.scores.empty() ||
      output.classes.empty() || output.keypoints.empty() || class_num <= 0 ||
      output.kpt_channels < params_.num_keypoints * 2)
  {
    return armors;
  }

  int n = output.count;
  int max_by_scores =
      static_cast<int>(std::min(output.scores.size(), output.classes.size()));
  if (n > max_by_scores)
  {
    n = max_by_scores;
  }
  int max_by_kpts = static_cast<int>(output.keypoints.size() /
                                     static_cast<std::size_t>(output.kpt_channels));
  if (n > max_by_kpts)
  {
    n = max_by_kpts;
  }
  if (n <= 0)
  {
    return armors;
  }

  armors.reserve(static_cast<std::size_t>(n));
  float inv_scale = static_cast<float>(1.0 / scale);
  int kp_dim = (output.kpt_channels >= params_.num_keypoints * 3) ? 3 : 2;

  for (int i = 0; i < n; ++i)
  {
    float conf = output.scores[static_cast<std::size_t>(i)];
    if (conf < params_.min_confidence)
    {
      continue;
    }

    int cls = output.classes[static_cast<std::size_t>(i)];
    if (cls < 0 || cls >= class_num)
    {
      continue;
    }

    int cls_color = class_map_.colors[static_cast<std::size_t>(cls)];
    int color = (cls_color < 0) ? params_.detect_color : cls_color;
    if (color != params_.detect_color)
    {
      continue;
    }

    if (class_map_.ignored[static_cast<std::size_t>(cls)] != 0)
    {
      continue;
    }

    auto kp_offset = static_cast<std::size_t>(i) * output.kpt_channels;
    std::vector<cv::Point2f> keypoints;
    keypoints.reserve(static_cast<std::size_t>(params_.num_keypoints));
    for (int k = 0; k < params_.num_keypoints; ++k)
    {
      keypoints.emplace_back(
          output.keypoints[kp_offset + static_cast<std::size_t>(k) * kp_dim] * inv_scale,
          output.keypoints[kp_offset + static_cast<std::size_t>(k) * kp_dim + 1] *
              inv_scale);
    }
    sort_yolo_armor_keypoints(keypoints);
    if (keypoints.size() < 4)
    {
      continue;
    }

    Light left_light(keypoints[0], keypoints[1], color);
    Light right_light(keypoints[2], keypoints[3], color);

    Armor armor(left_light, right_light);
    armor.type = determine_yolo_armor_type(left_light, right_light,
                                        params_.large_armor_ratio_threshold);
    armor.number = class_map_.mapped_labels[static_cast<std::size_t>(cls)];
    armor.confidence = conf;

    char buf[64];
    auto _ = std::snprintf(buf, sizeof(buf), "%s: %.2f", armor.number.c_str(),
                           static_cast<double>(conf));
    armor.classfication_result = buf;

    armors.emplace_back(std::move(armor));
  }

  return armors;
}

std::vector<Armor> Yolo11Adapter::ParseRaw(double scale, const cv::Mat& output) const
{
  int class_num = class_map_.ClassCount();
  if (scale <= 0.0 || output.empty() || class_num <= 0 || output.rows < 4 + class_num)
  {
    return {};
  }

  int num_anchors = output.cols;
  int features = output.rows;
  int kp_start = 4 + class_num;
  int kp_cols = features - kp_start;
  if (kp_cols < params_.num_keypoints * 2)
  {
    return {};
  }
  int kp_stride = (kp_cols >= params_.num_keypoints * 3) ? 3 : 2;
  float inv_scale = static_cast<float>(1.0 / scale);

  std::vector<float> max_score(static_cast<std::size_t>(num_anchors), 0.0f);
  std::vector<int> max_class(static_cast<std::size_t>(num_anchors), -1);

  for (int c = 0; c < class_num; ++c)
  {
    const float* score_row = output.ptr<float>(4 + c);
    for (int a = 0; a < num_anchors; ++a)
    {
      float score = score_row[a];
      if (score > max_score[static_cast<std::size_t>(a)])
      {
        max_score[static_cast<std::size_t>(a)] = score;
        max_class[static_cast<std::size_t>(a)] = c;
      }
    }
  }

  const float* row_x = output.ptr<float>(0);
  const float* row_y = output.ptr<float>(1);
  const float* row_w = output.ptr<float>(2);
  const float* row_h = output.ptr<float>(3);

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<int> anchor_indices;

  class_ids.reserve(32);
  confidences.reserve(32);
  boxes.reserve(32);
  anchor_indices.reserve(32);

  for (int a = 0; a < num_anchors; ++a)
  {
    float score = max_score[static_cast<std::size_t>(a)];
    if (score < params_.score_threshold)
    {
      continue;
    }

    int cls = max_class[static_cast<std::size_t>(a)];
    if (cls < 0 || cls >= class_num)
    {
      continue;
    }

    float xc = row_x[a];
    float yc = row_y[a];
    float w = row_w[a];
    float h = row_h[a];

    int left = static_cast<int>((xc - 0.5f * w) / scale);
    int top = static_cast<int>((yc - 0.5f * h) / scale);
    int box_w = static_cast<int>(w / scale);
    int box_h = static_cast<int>(h / scale);

    class_ids.push_back(cls);
    confidences.push_back(score);
    boxes.emplace_back(left, top, box_w, box_h);
    anchor_indices.push_back(a);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, params_.score_threshold, params_.nms_threshold,
                    indices);

  std::vector<Armor> armors;
  armors.reserve(indices.size());

  for (int i : indices)
  {
    int class_id = class_ids[static_cast<std::size_t>(i)];
    if (class_id < 0 || class_id >= class_num)
    {
      continue;
    }

    if (class_map_.ignored[static_cast<std::size_t>(class_id)] != 0)
    {
      continue;
    }

    float conf = confidences[static_cast<std::size_t>(i)];
    if (conf < params_.min_confidence)
    {
      continue;
    }

    int cls_color = class_map_.colors[static_cast<std::size_t>(class_id)];
    int color = (cls_color < 0) ? params_.detect_color : cls_color;
    if (color != params_.detect_color)
    {
      continue;
    }

    int anchor = anchor_indices[static_cast<std::size_t>(i)];
    std::vector<cv::Point2f> keypoints;
    keypoints.reserve(static_cast<std::size_t>(params_.num_keypoints));
    for (int k = 0; k < params_.num_keypoints; ++k)
    {
      const float* kx_row = output.ptr<float>(kp_start + k * kp_stride);
      const float* ky_row = output.ptr<float>(kp_start + k * kp_stride + 1);
      keypoints.emplace_back(kx_row[anchor] * inv_scale, ky_row[anchor] * inv_scale);
    }
    sort_yolo_armor_keypoints(keypoints);
    if (keypoints.size() < 4)
    {
      continue;
    }

    Light left_light(keypoints[0], keypoints[1], color);
    Light right_light(keypoints[2], keypoints[3], color);

    Armor armor(left_light, right_light);
    armor.type = determine_yolo_armor_type(left_light, right_light,
                                        params_.large_armor_ratio_threshold);
    armor.number = class_map_.mapped_labels[static_cast<std::size_t>(class_id)];
    armor.confidence = conf;

    char buf[64];
    auto _ = std::snprintf(buf, sizeof(buf), "%s: %.2f", armor.number.c_str(),
                           static_cast<double>(conf));
    armor.classfication_result = buf;

    armors.emplace_back(std::move(armor));
  }

  return armors;
}

}  // namespace rm_auto_aim
