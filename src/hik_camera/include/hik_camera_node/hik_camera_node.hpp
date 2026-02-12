#pragma once

#include <camera_info_manager/camera_info_manager.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/imgproc.hpp>
#include <std_msgs/msg/bool.hpp>

#include "CameraParams.h"
#include "MvCameraControl.h"

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
    uint8_t device_index = 0;
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
  void SwitchCamera(bool to_lob);

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

  // Lob shot camera switch
  std::string camera_info_url_normal_;
  std::string camera_info_url_lob_;
  uint8_t device_index_normal_{0};
  uint8_t device_index_lob_{1};
  bool is_lob_camera_{false};
  std::atomic<bool> in_read_{false};       // Read() 进入 SDK 前置 true，返回后置 false
  std::atomic<bool> is_switching_{false};  // SwitchCamera 期间为 true，防止守护线程误重启
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr lob_shot_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr camera_switch_done_pub_;
};
}  // namespace HikCamera