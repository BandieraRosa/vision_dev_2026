#include "webots_camera/webots_camera.hpp"

#include <cmath>
#include <opencv2/imgproc.hpp>

#include "pluginlib/class_list_macros.hpp"

// --- 数学辅助函数 (保持原有逻辑用于云台解算) ---
struct Quat
{
  double w, x, y, z;
};
struct Euler
{
  double roll, pitch, yaw;
};

static Quat matrix_to_quat(const double* R)
{
  Quat q;
  double trace = R[0] + R[4] + R[8];
  if (trace > 0)
  {
    double s = 0.5 / sqrt(trace + 1.0);
    q.w = 0.25 / s;
    q.x = (R[7] - R[5]) * s;
    q.y = (R[2] - R[6]) * s;
    q.z = (R[3] - R[1]) * s;
  }
  else
  {
    if (R[0] > R[4] && R[0] > R[8])
    {
      double s = 2.0 * sqrt(1.0 + R[0] - R[4] - R[8]);
      q.w = (R[7] - R[5]) / s;
      q.x = 0.25 * s;
      q.y = (R[1] + R[3]) / s;
      q.z = (R[2] + R[6]) / s;
    }
    else if (R[4] > R[8])
    {
      double s = 2.0 * sqrt(1.0 + R[4] - R[0] - R[8]);
      q.w = (R[2] - R[6]) / s;
      q.x = (R[1] + R[3]) / s;
      q.y = 0.25 * s;
      q.z = (R[5] + R[7]) / s;
    }
    else
    {
      double s = 2.0 * sqrt(1.0 + R[8] - R[0] - R[4]);
      q.w = (R[3] - R[1]) / s;
      q.x = (R[2] + R[6]) / s;
      q.y = (R[5] + R[7]) / s;
      q.z = 0.25 * s;
    }
  }
  return q;
}

static Euler quat_to_euler(Quat q)
{
  Euler e;
  double sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
  double cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
  e.roll = std::atan2(sinr_cosp, cosr_cosp);
  double sinp = 2 * (q.w * q.y - q.z * q.x);
  if (std::abs(sinp) >= 1)
  {
    e.pitch = std::copysign(M_PI / 2, sinp);
  }
  else
  {
    e.pitch = std::asin(sinp);
  }
  double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
  double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
  e.yaw = std::atan2(siny_cosp, cosy_cosp);
  return e;
}

static Quat euler_to_quat(Euler e)
{
  Quat q;
  double cy = cos(e.yaw * 0.5);
  double sy = sin(e.yaw * 0.5);
  double cp = cos(e.pitch * 0.5);
  double sp = sin(e.pitch * 0.5);
  double cr = cos(e.roll * 0.5);
  double sr = sin(e.roll * 0.5);
  q.w = cr * cp * cy + sr * sp * sy;
  q.x = sr * cp * cy - cr * sp * sy;
  q.y = cr * sp * cy + sr * cp * sy;
  q.z = cr * cp * sy - sr * sp * cy;
  return q;
}
// -------------------------------------

namespace webots_camera
{

void WebotsCamera::init(webots_ros2_driver::WebotsNode* node,
                        std::unordered_map<std::string, std::string>& parameters)
{
  node_ = node;
  // 使用 new 创建 Supervisor 实例，解决 getInstance 问题
  robot_ = new webots::Supervisor();

  // --- 参数映射 (尽可能对齐 HikCameraNode) ---

  // Webots 设备名 (对应 world 文件中的 DEF 或 name)
  camera_name_ = "camera";
  if (parameters.count("camera_name"))
  {
    camera_name_ = parameters["camera_name"];
  }

  // 对应 HikCamera 的 frame_id 参数
  // Hik 默认是 "narrow_stereo"，这里我们也设为默认，但也允许从 urdf 覆盖
  frame_id_ = "narrow_stereo";
  if (parameters.count("frame_id"))
  {
    frame_id_ = parameters["frame_id"];
  }

  // 对应 HikCamera 的 frame_rate 参数
  target_fps_ = 30.0;
  if (parameters.count("frame_rate"))
  {
    target_fps_ = std::stod(parameters["frame_rate"]);
  }
  else if (parameters.count("fps"))
  {
    target_fps_ = std::stod(parameters["fps"]);  // 兼容旧参数名
  }

  int cam_period = (target_fps_ > 0) ? static_cast<int>(1000.0 / target_fps_) : 33;
  publish_period_ms_ = cam_period;

  // --- 设备初始化 ---
  webots_camera_ = robot_->getCamera(camera_name_);
  if (!webots_camera_)
  {
    RCLCPP_ERROR(node_->get_logger(), "Webots Camera device '%s' not found!",
                 camera_name_.c_str());
  }
  else
  {
    webots_camera_->enable(cam_period);
  }

  webots_camera_node_ = robot_->getFromDef(camera_name_);
  if (!webots_camera_node_)
  {
    RCLCPP_WARN(node_->get_logger(),
                "Could not getFromDef('%s'), gimbal rotation disabled.",
                camera_name_.c_str());
  }

  // --- Publisher 初始化 ---
  it_ = std::make_shared<image_transport::ImageTransport>(node_->shared_from_this());

  // 对齐 HikCamera: 话题名称为 "image_raw"
  // 最终话题通常为: /<node_namespace>/image_raw 和 /<node_namespace>/camera_info
  camera_pub_ = it_->advertiseCamera("image_raw", 1);

  // 额外的姿态发布 (HikCamera 没有这个，但你的系统需要)
  rotation_pub_ = node_->create_publisher<geometry_msgs::msg::Quaternion>("rotation", 10);

  RCLCPP_INFO(node_->get_logger(), "WebotsCamera initialized mimicking HikCamera.");
  RCLCPP_INFO(node_->get_logger(), "  Topic: image_raw, Encoding: rgb8, FrameID: %s",
              frame_id_.c_str());
}

WebotsCamera::~WebotsCamera()
{
  if (webots_camera_)
  {
    webots_camera_->disable();
  }
  if (robot_)
  {
    delete robot_;
    robot_ = nullptr;
  }
}

void WebotsCamera::step()
{
  if (!webots_camera_ || !node_)
  {
    return;
  }

  double now_s = robot_->getTime();
  int now_ms = static_cast<int>(now_s * 1000.0);

  if (now_ms - last_publish_time_ms_ < publish_period_ms_)
  {
    return;
  }
  last_publish_time_ms_ = now_ms;

  // --- 1. 图像处理 (对齐 HikCamera) ---
  const unsigned char* data = webots_camera_->getImage();
  if (data)
  {
    int width = webots_camera_->getWidth();
    int height = webots_camera_->getHeight();

    // Webots 原生数据是 BGRA (8UC4)
    cv::Mat img_bgra(height, width, CV_8UC4, (void*)data);
    cv::Mat img_rgb;

    // HikCamera 输出是 RGB8，所以我们将 BGRA 转为 RGB
    // 注意：Hik 代码中是通过 Bayer -> RGB 转换得到的 RGB8
    cv::cvtColor(img_bgra, img_rgb, cv::COLOR_BGRA2RGB);

    std_msgs::msg::Header header;
    header.stamp = node_->get_clock()->now();
    header.frame_id = frame_id_;

    // 使用 cv_bridge 转换为 sensor_msgs::msg::Image
    // encoding 设为 "rgb8" 以完全匹配 HikCameraNode
    sensor_msgs::msg::Image::SharedPtr img_msg =
        cv_bridge::CvImage(header, "rgb8", img_rgb).toImageMsg();

    // 生成/获取 CameraInfo
    // HikCamera 是从 yaml 加载，Webots 这里我们实时计算理想内参
    sensor_msgs::msg::CameraInfo cam_info =
        GetCameraInfo(width, height, webots_camera_->getFov(), header.stamp);
    cam_info.header = header;

    // 发布
    camera_pub_.publish(*img_msg, cam_info);
  }

  // --- 2. 姿态处理 (保留逻辑) ---
  if (webots_camera_node_ && rotation_pub_)
  {
    auto q_msg = CalculateGimbalRotation();
    rotation_pub_->publish(q_msg);
  }
}

sensor_msgs::msg::CameraInfo WebotsCamera::GetCameraInfo(int width, int height,
                                                         double fov, rclcpp::Time now)
{
  sensor_msgs::msg::CameraInfo info;
  info.header.stamp = now;
  info.header.frame_id = frame_id_;
  info.width = width;
  info.height = height;
  info.distortion_model = "plumb_bob";

  // Webots 仿真相机通常是无畸变的
  info.d = {0.0, 0.0, 0.0, 0.0, 0.0};

  // 计算焦距 f
  double f = width / (2.0 * std::tan(fov / 2.0));
  double cx = width / 2.0;
  double cy = height / 2.0;

  // K 矩阵
  info.k.fill(0.0);
  info.k[0] = f;
  info.k[2] = cx;
  info.k[4] = f;
  info.k[5] = cy;
  info.k[8] = 1.0;

  // P 矩阵
  info.p.fill(0.0);
  info.p[0] = f;
  info.p[2] = cx;
  info.p[5] = f;
  info.p[6] = cy;
  info.p[10] = 1.0;

  // R 矩阵
  info.r.fill(0.0);
  info.r[0] = 1.0;
  info.r[4] = 1.0;
  info.r[8] = 1.0;

  return info;
}

geometry_msgs::msg::Quaternion WebotsCamera::CalculateGimbalRotation()
{
  // 获取 Supervisor 读取的绝对姿态
  const double* r9_cam = webots_camera_node_->getOrientation();

  // 矩阵乘法: R_final = R_cam * COMPENSATION
  // Compensation (0 1 0; 1 0 0; 0 0 -1)
  double r_final[9];
  r_final[0] = r9_cam[1];
  r_final[1] = r9_cam[0];
  r_final[2] = -r9_cam[2];
  r_final[3] = r9_cam[4];
  r_final[4] = r9_cam[3];
  r_final[5] = -r9_cam[5];
  r_final[6] = r9_cam[7];
  r_final[7] = r9_cam[6];
  r_final[8] = -r9_cam[8];

  Quat q = matrix_to_quat(r_final);
  Euler e = quat_to_euler(q);

  // 核心逻辑：Pitch 取反
  e.pitch = -e.pitch;

  Quat q_final = euler_to_quat(e);

  geometry_msgs::msg::Quaternion msg;
  msg.x = q_final.x;
  msg.y = q_final.y;
  msg.z = q_final.z;
  msg.w = q_final.w;
  return msg;
}

}  // namespace webots_camera

PLUGINLIB_EXPORT_CLASS(webots_camera::WebotsCamera, webots_ros2_driver::PluginInterface)
