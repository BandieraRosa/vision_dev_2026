// OpenVINO
#include <openvino/openvino.hpp>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

// STD
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "armor_detector/yolo_detector.hpp"

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

std::string map_label(const std::string& raw_label)
{
  auto strip_prefix = [](const std::string& s) -> std::string
  {
    if (s.empty())
    {
      return s;
    }
    if (s[0] == 'B' || s[0] == 'R' || s[0] == 'E' || s[0] == 'P')
    {
      return s.substr(1);
    }
    return s;
  };

  const std::string NAME = strip_prefix(raw_label);

  if (NAME == "sentry")
  {
    return "guard";
  }
  if (NAME == "one")
  {
    return "1";
  }
  if (NAME == "two")
  {
    return "2";
  }
  if (NAME == "three" || NAME == "balancethree")
  {
    return "3";
  }
  if (NAME == "four" || NAME == "balancefour")
  {
    return "4";
  }
  if (NAME == "five" || NAME == "balancefive")
  {
    return "5";
  }
  if (NAME == "outpost")
  {
    return "outpost";
  }
  if (NAME == "base" || NAME == "basesmall")
  {
    return "base";
  }

  return "negative";
}

}  // namespace

YoloDetector::YoloDetector(const YoloParams& config)
    : config_(config), class_num_(static_cast<int>(YOLO11_MODEL_LABELS.size()))
{
  auto model = core_.read_model(config_.model_path);

  // 预处理
  ov::preprocess::PrePostProcessor ppp(model);
  auto& input = ppp.input();

  input.tensor()
      .set_element_type(ov::element::u8)
      .set_shape({1, INPUT_SIZE, INPUT_SIZE, 3})
      .set_layout("NHWC")
      .set_color_format(ov::preprocess::ColorFormat::RGB);

  input.model().set_layout("NCHW");

  input.preprocess().convert_element_type(ov::element::f32).scale(255.0);

  model = ppp.build();
  compiled_model_ =
      core_.compile_model(model, config_.device,
                          ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
}

std::vector<Armor> YoloDetector::Detect(const cv::Mat& rgb_img, int detect_color,
                                        const std::vector<std::string>& ignore_classes)
{
  if (rgb_img.empty())
  {
    return {};
  }

  auto x_scale = static_cast<double>(INPUT_SIZE) / rgb_img.rows;
  auto y_scale = static_cast<double>(INPUT_SIZE) / rgb_img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(rgb_img.rows * scale);
  auto w = static_cast<int>(rgb_img.cols * scale);

  cv::Mat input_mat(INPUT_SIZE, INPUT_SIZE, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Rect roi(0, 0, w, h);
  cv::resize(rgb_img, input_mat(roi), {w, h});

  // 推理
  ov::Tensor input_tensor(ov::element::u8, {1, INPUT_SIZE, INPUT_SIZE, 3},
                          input_mat.data);
  auto infer_request = compiled_model_.create_infer_request();
  infer_request.set_input_tensor(input_tensor);
  infer_request.infer();

  // 输出
  auto output_tensor = infer_request.get_output_tensor();
  const auto& output_shape = output_tensor.get_shape();
  cv::Mat output(static_cast<int>(output_shape[1]), static_cast<int>(output_shape[2]),
                 CV_32F, output_tensor.data());

  // 后处理
  last_armors_ = Parse(scale, output, detect_color, ignore_classes);
  return last_armors_;
}

// 输出张量布局: [x, y, w, h, cls_0, ..., cls_N, kp0_x, kp0_y, ...]
std::vector<Armor> YoloDetector::Parse(double scale, cv::Mat& output, int detect_color,
                                       const std::vector<std::string>& ignore_classes)
{
  // 转置：[features, num_detections] → [num_detections, features]
  cv::transpose(output, output);

  int kp_start = 4 + class_num_;
  int total_cols = output.cols;
  int kp_cols = total_cols - kp_start;
  int kp_stride = (kp_cols >= NUM_KEYPOINTS * 3) ? 3 : 2;

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

    if (score < config_.score_threshold)
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
    for (int i = 0; i < NUM_KEYPOINTS; i++)
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
  cv::dnn::NMSBoxes(boxes, confidences, config_.score_threshold, config_.nms_threshold,
                    indices);

  std::vector<Armor> armors;

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
    if (std::find(ignore_classes.begin(), ignore_classes.end(), label) !=
        ignore_classes.end())
    {
      continue;
    }

    // 置信度过滤
    if (confidences[i] < config_.min_confidence)
    {
      continue;
    }

    // 关键点
    auto& kps = all_keypoints[i];
    SortKeypoints(kps);

    int color =
        (raw_label[0] == 'R') ? RED : ((raw_label[0] == 'B') ? BLUE : detect_color);
    if (color != detect_color)
    {
      continue;
    }

    Light left_light = Light(kps[0], kps[1], color);
    Light right_light = Light(kps[2], kps[3], color);

    // 构造 Armor
    Armor armor(left_light, right_light);

    // 由关键点几何比例判断大小类型
    armor.type = DetermineArmorType(left_light, right_light);

    armor.number = label;
    armor.confidence = confidences[i];

    std::ostringstream oss;
    oss << raw_label << " -> " << label << ": " << std::fixed << std::setprecision(2)
        << confidences[i];
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
  return ratio > LARGE_ARMOR_RATIO_THRESHOLD ? ArmorType::LARGE : ArmorType::SMALL;
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
