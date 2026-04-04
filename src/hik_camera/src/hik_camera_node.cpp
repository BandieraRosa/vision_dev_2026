#include "hik_camera_node/hik_camera_node.hpp"

using namespace std::chrono_literals;

namespace HikCamera
{
HikCameraNode::HikCameraNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("hik_camera_node", options)
{
  params_.exposure_time = this->declare_parameter<double>("exposure_time", 1000.0);  // us
  params_.gain = this->declare_parameter<double>("gain", 15.0);
  params_.autocap = this->declare_parameter<bool>("autocap", true);
  params_.frame_rate = this->declare_parameter<double>("frame_rate", 249.0);
  current_frame_id_ = params_.frame_id =
      this->declare_parameter<std::string>("frame_id", "camera_optical_frame");
  current_camera_name_ = params_.camera_name =
      this->declare_parameter<std::string>("camera_name", "gimbal_camera");
  params_.rotate = this->declare_parameter<uint8_t>("rotate", 0);
  current_device_index_ = params_.device_index =
      this->declare_parameter<uint8_t>("device_index", 0);
  const auto& robot_type = this->declare_parameter<std::string>("robot_type", "infantry");
  is_hero_ = (robot_type == "hero");

  RCLCPP_INFO(this->get_logger(), "params has been initialized.");

  // 创建 publisher
  camera_pub_ = image_transport::create_camera_publisher(this, "image_raw",
                                                         rmw_qos_profile_sensor_data);
  RCLCPP_INFO(this->get_logger(), "Camera publisher created.");
  // 初始化相机
  CaptureInit();
  RCLCPP_INFO(this->get_logger(), "Camera initialized.");

  // 创建守护线程，负责自动重启
  guard_.protect_thread = std::thread(&HikCameraNode::ProtectRunning, this);

  MV_CC_GetImageInfo(handle_, &img_info_);
  image_msg_.data.reserve(
      static_cast<size_t>(img_info_.nHeightMax * img_info_.nWidthMax) * 3);
  image_msg_.height = img_info_.nHeightMax;
  image_msg_.width = img_info_.nWidthMax;
  camera_info_manager_ =
      std::make_unique<camera_info_manager::CameraInfoManager>(this, params_.camera_name);
  current_camera_info_url_ = params_.camera_info_url = this->declare_parameter(
      "camera_info_url", "package://hik_camera/config/camera_info.yaml");

  if (camera_info_manager_->validateURL(current_camera_info_url_))
  {
    camera_info_manager_->loadCameraInfo(current_camera_info_url_);
    camera_info_msg_ = camera_info_manager_->getCameraInfo();
  }
  else
  {
    RCLCPP_WARN(this->get_logger(), "Invalid camera info URL: %s",
                current_camera_info_url_.c_str());
  }

  RCLCPP_INFO(this->get_logger(), "Guard thread created.");

  if (is_hero_)
  {
    RCLCPP_WARN(this->get_logger(),
                "Running on robot type: %s, LOB camera support enabled.",
                robot_type.c_str());
    params_.camera_name_lob =
        this->declare_parameter<std::string>("camera_name_lob", "gimbal_camera_lob");
    params_.frame_id_lob =
        this->declare_parameter<std::string>("frame_id_lob", "camera_optical_frame_lob");
    params_.device_index_lob = this->declare_parameter<uint8_t>("device_index_lob", 1);
    params_.camera_info_url_lob = this->declare_parameter(
        "camera_info_url_lob", "package://hik_camera/config/camera_info_lob.yaml");

    camera_switch_done_pub_ = this->create_publisher<std_msgs::msg::Bool>(
        "/camera_switch_done", rclcpp::QoS(1).reliable());

    lob_shot_sub_ = this->create_subscription<std_msgs::msg::Bool>(
        "/lob_shot_switch", rclcpp::QoS(1).reliable(),
        [this](const std_msgs::msg::Bool::SharedPtr msg)
        {
          if (!msg->data)
          {
            return;
          }
          SwitchCamera(!is_lob_camera_);
        });
  }

  // 创建取流线程
  capture_thread_ = std::thread(
      [this]()
      {
        RCLCPP_INFO(this->get_logger(), "Hik SDK capture thread started.");

        while (running_.load())
        {
          if (hik_state_.load() == HikStateEnum::STOPPED)
          {
            std::this_thread::sleep_for(10ms);
            continue;
          }

          cv::Mat image;
          rclcpp::Time stamp;
          bool ok = Read(image, stamp);
          if (!ok || image.empty())
          {
            continue;
          }

          switch (params_.rotate)
          {
            case 1:
              cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
              break;
            case 2:
              cv::rotate(image, image, cv::ROTATE_180);
              break;
            case 3:
              cv::rotate(image, image, cv::ROTATE_90_COUNTERCLOCKWISE);
            default:
              break;
          }
          image_msg_.height = image.rows;
          image_msg_.width = image.cols;
          camera_info_msg_.height = image.rows;
          camera_info_msg_.width = image.cols;
          camera_info_msg_.header.stamp = stamp;
          camera_info_msg_.header.frame_id = current_frame_id_;

          // 将 cv::Mat 转成 sensor_msgs::msg::Image
          image_msg_.header.stamp = stamp;
          image_msg_.header.frame_id = current_frame_id_;
          image_msg_.encoding = "rgb8";
          image_msg_.is_bigendian = false;
          image_msg_.step = static_cast<uint32_t>(image.cols * image.channels());
          image_msg_.data.assign(image.datastart, image.dataend);

          camera_pub_.publish(image_msg_, camera_info_msg_);
        }

        RCLCPP_INFO(this->get_logger(), "Hik SDK capture thread exit.");
      });
}

HikCameraNode::~HikCameraNode()
{
  RCLCPP_INFO(this->get_logger(), "Destroying HikCameraNode...");

  running_.store(false);

  // 通知守护线程退出
  guard_.is_quit.notify_all();

  // 先停采集线程
  if (capture_thread_.joinable())
  {
    capture_thread_.join();
  }

  // 关闭相机
  CaptureStop();

  // 再停守护线程
  if (guard_.protect_thread.joinable())
  {
    guard_.protect_thread.join();
  }

  RCLCPP_INFO(this->get_logger(), "HikCameraNode destroyed.");
}

bool HikCameraNode::Read(cv::Mat& img, rclcpp::Time& timestamp)
{
  in_read_.store(true, std::memory_order_seq_cst);

  if (hik_state_.load(std::memory_order_seq_cst) == HikStateEnum::STOPPED)
  {
    in_read_.store(false, std::memory_order_seq_cst);
    return false;
  }

  MV_FRAME_OUT raw{};
  unsigned int ret{};
  unsigned int n_msec = 100;

  auto start = std::chrono::steady_clock::now();
  ret = MV_CC_GetImageBuffer(handle_, &raw, n_msec); // INFINITE

  if (ret != MV_OK)
  {
    RCLCPP_ERROR(this->get_logger(),
                 "MV_CC_GetImageBuffer failed: 0x%X, switching to Stopped.", ret);
    in_read_.store(false, std::memory_order_seq_cst);
    hik_state_.store(HikStateEnum::STOPPED);
    guard_.is_quit.notify_all();
    return false;
  }

  auto now = std::chrono::steady_clock::now();
  auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start);
  if (duration_ns < std::chrono::nanoseconds(2'000'000))
  {
    MV_CC_FreeImageBuffer(handle_, &raw);
    in_read_.store(false, std::memory_order_seq_cst);
    return false;
  }

  timestamp = this->now();

  cv::Mat raw_img(cv::Size(raw.stFrameInfo.nWidth, raw.stFrameInfo.nHeight), CV_8U,
                  raw.pBufAddr);

  const auto& frame_info = raw.stFrameInfo;
  auto pixel_type = frame_info.enPixelType;

  static const std::unordered_map<MvGvspPixelType, int> TYPE_MAP = {
      {PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGR2BGR},
      {PixelType_Gvsp_BayerRG8, cv::COLOR_BayerRG2BGR},
      {PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGB2BGR},
      {PixelType_Gvsp_BayerBG8, cv::COLOR_BayerBG2BGR}};

  auto it = TYPE_MAP.find(pixel_type);
  if (it == TYPE_MAP.end())
  {
    MV_CC_FreeImageBuffer(handle_, &raw);
    in_read_.store(false, std::memory_order_seq_cst);
    hik_state_.store(HikStateEnum::STOPPED);
    guard_.is_quit.notify_all();
    return false;
  }

  cv::Mat dst_image;
  cv::cvtColor(raw_img, dst_image, it->second);
  img = dst_image;

  ret = MV_CC_FreeImageBuffer(handle_, &raw);
  in_read_.store(false, std::memory_order_seq_cst);

  if (ret != MV_OK)
  {
    RCLCPP_ERROR(this->get_logger(),
                 "MV_CC_FreeImageBuffer failed: 0x%X, switching to Stopped.", ret);
    hik_state_.store(HikStateEnum::STOPPED);
    guard_.is_quit.notify_all();
    return false;
  }

  return true;
}

void HikCameraNode::CaptureInit()
{
  if (!running_.load())
  {
    return;
  }
  unsigned int ret{};
  MV_CC_DEVICE_INFO_LIST device_list{};
  ret = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);
  if (ret != MV_OK)
  {
    RCLCPP_ERROR(this->get_logger(), "MV_CC_EnumDevices failed: 0x%X", ret);
    return;
  }

  if (device_list.nDeviceNum == 0)
  {
    RCLCPP_ERROR(this->get_logger(), "Not found camera!");
    return;
  }

  if (current_device_index_ >= device_list.nDeviceNum)
  {
    RCLCPP_ERROR(this->get_logger(), "Device index %d out of range (found %d cameras)",
                 current_device_index_, device_list.nDeviceNum);
    return;
  }

  ret = MV_CC_CreateHandle(&handle_, device_list.pDeviceInfo[current_device_index_]);
  if (ret != MV_OK)
  {
    RCLCPP_ERROR(this->get_logger(), "MV_CC_CreateHandle failed: 0x%X", ret);
    return;
  }

  ret = MV_CC_OpenDevice(handle_);
  if (ret != MV_OK)
  {
    RCLCPP_ERROR(this->get_logger(), "MV_CC_OpenDevice failed: 0x%X", ret);
    return;
  }

  unsigned int n_image_node_num = 3;
  ret = MV_CC_SetImageNodeNum(handle_, n_image_node_num);
  if (MV_OK != ret)
  {
    // 设置失败
    RCLCPP_ERROR(this->get_logger(), "MV_CC_SetImageNodeNum failed: 0x%X", ret);
    return;
  }

  if (!params_.autocap)
  {
    ret = MV_CC_SetEnumValueByString(handle_, "AcquisitionMode", "Continuous");
    if (MV_OK != ret)
    {
      RCLCPP_ERROR(this->get_logger(), "Set Acquisition Mode to Continuous fail! 0x%X",
                   ret);
      return;
    }

    //    将触发模式设置为开启 (On)
    //    参数 "TriggerMode" 的值: 0 表示 Off, 1 表示 On
    ret = MV_CC_SetEnumValue(handle_, "TriggerMode", 1);
    if (MV_OK != ret)
    {
      RCLCPP_ERROR(this->get_logger(), "Set Trigger Mode to On fail! 0x%X", ret);
      return;
    }

    //    设置触发源为外部硬件触发 (Line0)
    //    可用的值通常有 "Line0", "Line1", "Line2", "Software", "FrequencyConverter" 等
    //    请根据您的物理接线选择正确的一项
    ret = MV_CC_SetEnumValueByString(handle_, "TriggerSource", "Line0");
    if (MV_OK != ret)
    {
      RCLCPP_ERROR(this->get_logger(), "Set Trigger Source to Line0 fail! 0x%X", ret);
      return;
    }

    //    (可选) 设置触发激活方式
    //    例如设置为上升沿触发 "RisingEdge"
    //    其他可选值如 "FallingEdge", "LevelHigh", "LevelLow"
    ret = MV_CC_SetEnumValueByString(handle_, "TriggerActivation", "RisingEdge");
    if (MV_OK != ret)
    {
      RCLCPP_ERROR(this->get_logger(), "Set Trigger Activation to RisingEdge fail! 0x%X",
                   ret);
      return;
    }
  }
  else
  {
    // 将触发模式设置为开启 (On)
    // 参数 "TriggerMode" 的值: 0 表示 Off, 1 表示 On
    ret = MV_CC_SetEnumValue(handle_, "TriggerMode", 0);
    if (MV_OK != ret)
    {
      RCLCPP_ERROR(this->get_logger(), "Set Trigger Mode to Off fail! 0x%X", ret);
      return;
    }
  }
  // 曝光、增益、白平衡等
  SetEnumValue("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS);
  SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
  SetEnumValue("GainAuto", MV_GAIN_MODE_OFF);
  SetEnumValue("PixelFormat", PixelType_Gvsp_BayerRG8);
  SetFloatValue("ExposureTime", params_.exposure_time);
  SetFloatValue("Gain", params_.gain);

  MVCC_ENUMVALUE adc_bit_depth{};
  ret = MV_CC_GetEnumValue(handle_, "ADCBitDepth", &adc_bit_depth);
  if (ret == MV_OK)
  {
    RCLCPP_INFO(this->get_logger(), "Current ADCBitDepth: %u", adc_bit_depth.nCurValue);
    // 设置 ADC 位深为 8 Bits (对应枚举值 2)
    ret = MV_CC_SetEnumValue(handle_, "ADCBitDepth", 2);
    if (MV_OK != ret)
    {
      RCLCPP_ERROR(this->get_logger(), "Set ADC Bit Depth to 8 Bits fail! 0x%X", ret);
      return;
    }
  }
  else
  {
    RCLCPP_WARN(this->get_logger(), "Get ADCBitDepth failed: 0x%X, skip", ret);
  }

  // 帧率
  ret = MV_CC_SetFloatValue(handle_, "AcquisitionFrameRate", 249.0);
  if (ret != MV_OK)
  {
    RCLCPP_ERROR(this->get_logger(), "MV_CC_SetFloatValue(set framerate) failed: 0x%X",
                 ret);
    return;
  }
  ret = MV_CC_StartGrabbing(handle_);
  if (ret != MV_OK)
  {
    RCLCPP_ERROR(this->get_logger(), "MV_CC_StartGrabbing failed: 0x%X", ret);
    return;
  }

  hik_state_.store(HikStateEnum::RUNNING);
  RCLCPP_INFO(this->get_logger(), "Hik camera initialized and started.");
  return;
}

void HikCameraNode::CaptureStop()
{
  hik_state_.store(HikStateEnum::STOPPED);

  if (handle_ == nullptr)
  {
    return;
  }

  unsigned int ret = MV_CC_StopGrabbing(handle_);
  if (ret != MV_OK)
  {
    RCLCPP_ERROR(this->get_logger(), "MV_CC_StopGrabbing failed: 0x%X", ret);
  }

  ret = MV_CC_CloseDevice(handle_);
  if (ret != MV_OK)
  {
    RCLCPP_ERROR(this->get_logger(), "MV_CC_CloseDevice failed: 0x%X", ret);
  }

  ret = MV_CC_DestroyHandle(handle_);
  if (ret != MV_OK)
  {
    RCLCPP_ERROR(this->get_logger(), "MV_CC_DestroyHandle failed: 0x%X", ret);
  }

  handle_ = nullptr;
  RCLCPP_INFO(this->get_logger(), "Hik camera stopped and handle destroyed.");
}

void HikCameraNode::ProtectRunning()
{
  RCLCPP_INFO(this->get_logger(), "Protect thread started.");

  std::unique_lock<std::mutex> lock(this->guard_.mux);
  while (running_.load())
  {
    // 等待条件变量
    this->guard_.is_quit.wait(
        lock,
        [this]
        {
          return (this->hik_state_.load() == HikStateEnum::STOPPED &&
                  !this->is_switching_.load()) ||
                 (!this->running_.load());
        });

    if (!this->running_.load())
    {
      break;
    }

    RCLCPP_INFO(this->get_logger(), "Camera stopped, attempting to restart...");
    this->CaptureStop();
    // 简单延时防止频繁重启
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    this->CaptureInit();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  RCLCPP_INFO(this->get_logger(), "Protect thread exit.");
}

void HikCameraNode::SwitchCamera(bool to_lob)
{
  is_switching_.store(true, std::memory_order_seq_cst);
  RCLCPP_INFO(this->get_logger(), "Switching to %s camera...", to_lob ? "lob" : "normal");

  hik_state_.store(HikStateEnum::STOPPED, std::memory_order_seq_cst);

  // 自旋等待in_read_清零
  while (in_read_.load(std::memory_order_seq_cst))
  {
    std::this_thread::yield();
  }

  CaptureStop();

  current_device_index_ = to_lob ? params_.device_index_lob : params_.device_index;

  current_camera_info_url_ =
      to_lob ? params_.camera_info_url_lob : params_.camera_info_url;
  current_camera_name_ = to_lob ? params_.camera_name_lob : params_.camera_name;
  current_frame_id_ = to_lob ? params_.frame_id_lob : params_.frame_id;

  camera_info_manager_->setCameraName(current_camera_name_);
  if (camera_info_manager_->validateURL(current_camera_info_url_))
  {
    camera_info_manager_->loadCameraInfo(current_camera_info_url_);
    camera_info_msg_ = camera_info_manager_->getCameraInfo();
    RCLCPP_INFO(this->get_logger(), "Loaded camera info: %s",
                current_camera_info_url_.c_str());
  }
  else
  {
    RCLCPP_WARN(this->get_logger(), "Invalid camera info URL for %s: %s",
                to_lob ? "lob" : "normal", current_camera_info_url_.c_str());
  }

  CaptureInit();

  if (hik_state_.load() == HikStateEnum::RUNNING)
  {
    is_lob_camera_ = to_lob;
    std_msgs::msg::Bool done_msg;
    done_msg.data = to_lob;
    camera_switch_done_pub_->publish(done_msg);
    RCLCPP_INFO(this->get_logger(), "Camera switched to %s successfully.",
                to_lob ? "lob" : "normal");
  }
  else
  {
    RCLCPP_ERROR(this->get_logger(),
                 "CaptureInit failed after switching to %s camera, state not updated.",
                 to_lob ? "lob" : "normal");
  }

  is_switching_.store(false, std::memory_order_seq_cst);
}

void HikCameraNode::SetFloatValue(const std::string& name, double value)
{
  if (handle_ == nullptr)
  {
    return;
  }

  unsigned int ret =
      MV_CC_SetFloatValue(handle_, name.c_str(), static_cast<float>(value));
  if (ret != MV_OK)
  {
    RCLCPP_ERROR(this->get_logger(), "MV_CC_SetFloatValue(\"%s\", %f) failed: 0x%X",
                 name.c_str(), value, ret);
  }
}

void HikCameraNode::SetEnumValue(const std::string& name, unsigned int value)
{
  if (handle_ == nullptr)
  {
    return;
  }

  unsigned int ret = MV_CC_SetEnumValue(handle_, name.c_str(), value);
  if (ret != MV_OK)
  {
    RCLCPP_ERROR(this->get_logger(), "MV_CC_SetEnumValue(\"%s\", %u) failed: 0x%X",
                 name.c_str(), value, ret);
  }
}
}  // namespace HikCamera

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(HikCamera::HikCameraNode)
