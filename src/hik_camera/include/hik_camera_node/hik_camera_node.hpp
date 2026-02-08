#pragma once

#include <opencv2/imgproc.hpp>

#include "CameraParams.h"
#include "MvCameraControl.h"
#include "camera_info_manager/camera_info_manager.hpp"
#include "image_transport/image_transport.hpp"

namespace HikCamera
{
class HikCameraNode : public rclcpp::Node
{
 public:
  explicit HikCameraNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~HikCameraNode() override;

 private:
  enum class HikStateEnum : uint8_t
  {
    STOPPED,
    RUNNING
  };

  struct Parameters
  {
    double exposure_time;  // us
    double gain;
    bool autocap;
    double frame_rate;
    std::string frame_id;
    std::string camera_name;
    uint8_t rotate = 0;
  };

  struct Protect
  {
    std::mutex mux;
    std::condition_variable is_quit;
    std::thread protect_thread;
  };

  // === 核心逻辑 ===
  bool Read(cv::Mat& image, rclcpp::Time& stamp);
  void CaptureInit();
  void CaptureStop();
  void ProtectRunning();

  void SetFloatValue(const std::string& name, double value);
  void SetEnumValue(const std::string& name, unsigned int value);

  // === 参数 ===
  Parameters params_;
  MV_CC_PIXEL_CONVERT_PARAM convert_param_;
  std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;

  sensor_msgs::msg::Image image_msg_;
  sensor_msgs::msg::CameraInfo camera_info_msg_;
  MV_IMAGE_BASIC_INFO img_info_;

  // === SDK 句柄 ===
  void* handle_{nullptr};

  std::atomic<HikStateEnum> hik_state_{HikStateEnum::STOPPED};
  std::atomic<bool> running_{true};

  std::thread capture_thread_;
  Protect guard_;

  // ROS2 publisher
  image_transport::CameraPublisher camera_pub_;
};
}  // namespace HikCamera