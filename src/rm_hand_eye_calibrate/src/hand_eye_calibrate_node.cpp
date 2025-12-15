#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2/LinearMath/Matrix3x3.hpp>

class HandEyeCalibrateNode : public rclcpp::Node
{
 public:
  HandEyeCalibrateNode(const rclcpp::NodeOptions& options)
      : Node("hand_eye_calibrator_node", options)
  {
    image_topic_ = declare_parameter<std::string>("image_topic", "/image_raw");
    camera_info_topic_ =
        declare_parameter<std::string>("camera_info_topic", "/camera_info");
    joint_states_topic_ =
        declare_parameter<std::string>("joint_states_topic", "/joint_states");

    yaw_joint_name_ = declare_parameter<std::string>("yaw_joint_name", "yaw_joint");
    pitch_joint_name_ = declare_parameter<std::string>("pitch_joint_name", "pitch_joint");

    board_cols_ = static_cast<int>(declare_parameter<int>("board_cols", 11));
    board_rows_ = static_cast<int>(declare_parameter<int>("board_rows", 8));
    square_size_ = declare_parameter<double>("square_size", 0.02);  // 米

    // 采样时用于过滤过旧检测结果的时间阈值
    max_age_sec_ = declare_parameter<double>("max_age_sec", 0.25);

    // 手眼标定方法：TSAI, PARK, HORAUD, ANDREFF, DANIILIDIS
    method_ = declare_parameter<std::string>("handeye_method", "DANIILIDIS");

    // 若云台 pitch 角正方向与预期相反，则置 true
    invert_pitch_ = declare_parameter<bool>("invert_pitch_sign", false);

    // -------- 可视化调试 --------
    publish_debug_image_ = declare_parameter<bool>("publish_debug_image", true);
    debug_image_topic_ =
        declare_parameter<std::string>("debug_image_topic", "/rm_hand_eye/debug_image");
    debug_image_pub_ = create_publisher<sensor_msgs::msg::Image>(debug_image_topic_,
                                                                 rclcpp::SensorDataQoS());

    // -------- 订阅 --------
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_, rclcpp::SensorDataQoS(),
        std::bind(&HandEyeCalibrateNode::OnCameraInfo, this, std::placeholders::_1));

    // message_filters（图像 + 关节状态近似时间同步）
    image_sub_.subscribe(this, image_topic_, rmw_qos_profile_sensor_data);
    joint_sub_.subscribe(this, joint_states_topic_, rmw_qos_profile_sensor_data);

    using SyncPolicy =
        message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                                                        sensor_msgs::msg::JointState>;
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(10), image_sub_, joint_sub_);
    sync_->registerCallback(std::bind(&HandEyeCalibrateNode::OnSynced, this,
                                      std::placeholders::_1, std::placeholders::_2));

    // -------- 服务 --------
    capture_srv_ = create_service<std_srvs::srv::Trigger>(
        "/rm_hand_eye/capture", std::bind(&HandEyeCalibrateNode::OnCapture, this,
                                          std::placeholders::_1, std::placeholders::_2));

    reset_srv_ = create_service<std_srvs::srv::Trigger>(
        "/rm_hand_eye/reset", std::bind(&HandEyeCalibrateNode::OnReset, this,
                                        std::placeholders::_1, std::placeholders::_2));

    solve_srv_ = create_service<std_srvs::srv::Trigger>(
        "/rm_hand_eye/solve", std::bind(&HandEyeCalibrateNode::OnSolve, this,
                                        std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(get_logger(), "Chessboard inner corners: cols=%d rows=%d square=%.4f m",
                board_cols_, board_rows_, square_size_);
    RCLCPP_INFO(get_logger(), "Debug image: enable=%s topic=%s",
                publish_debug_image_ ? "true" : "false", debug_image_topic_.c_str());
  }

 private:
  struct Detection
  {
    rclcpp::Time stamp;
    cv::Mat R_target2cam;  // 3x3
    cv::Mat t_target2cam;  // 3x1

    cv::Mat R_gripper2base;  // 3x3
    cv::Mat t_gripper2base;  // 3x1（此处为 0）

    double yaw = 0.0;
    double pitch = 0.0;

    double reproj_rmse = 0.0;
  };

  // ---------- 工具函数 ----------
  static cv::Mat Rz(double yaw)
  {
    const double C = std::cos(yaw);
    const double S = std::sin(yaw);
    cv::Mat r = (cv::Mat_<double>(3, 3) << C, -S, 0, S, C, 0, 0, 0, 1);
    return r;
  }

  static cv::Mat Ry(double pitch)
  {
    const double C = std::cos(pitch);
    const double S = std::sin(pitch);
    cv::Mat r = (cv::Mat_<double>(3, 3) << C, 0, S, 0, 1, 0, -S, 0, C);
    return r;
  }

  int ToOpenCvMethod(const std::string& m) const
  {
    std::string u = m;
    for (auto& ch : u)
    {
      ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    if (u == "TSAI")
    {
      return cv::CALIB_HAND_EYE_TSAI;
    }
    if (u == "PARK")
    {
      return cv::CALIB_HAND_EYE_PARK;
    }
    if (u == "HORAUD")
    {
      return cv::CALIB_HAND_EYE_HORAUD;
    }
    if (u == "ANDREFF")
    {
      return cv::CALIB_HAND_EYE_ANDREFF;
    }
    if (u == "DANIILIDIS")
    {
      return cv::CALIB_HAND_EYE_DANIILIDIS;
    }

    RCLCPP_WARN(get_logger(), "Unknown handeye_method='%s', fallback to TSAI", m.c_str());
    return cv::CALIB_HAND_EYE_TSAI;
  }

  std::optional<std::pair<double, double>> ExtractYawPitch(
      const sensor_msgs::msg::JointState& js) const
  {
    int yaw_idx = -1, pitch_idx = -1;
    for (size_t i = 0; i < js.name.size(); ++i)
    {
      if (js.name[i] == yaw_joint_name_)
      {
        yaw_idx = static_cast<int>(i);
      }
      if (js.name[i] == pitch_joint_name_)
      {
        pitch_idx = static_cast<int>(i);
      }
    }
    if (yaw_idx < 0 || pitch_idx < 0)
    {
      return std::nullopt;
    }

    if (yaw_idx >= static_cast<int>(js.position.size()) ||
        pitch_idx >= static_cast<int>(js.position.size()))
    {
      return std::nullopt;
    }
    return std::make_pair(js.position[yaw_idx], js.position[pitch_idx]);
  }

  std::vector<cv::Point3f> MakeObjectPoints() const
  {
    // 棋盘格角点在标定板坐标系下的三维坐标（位于 Z=0 平面）
    std::vector<cv::Point3f> obj;
    obj.reserve(static_cast<int64_t>(board_cols_) * board_rows_);
    for (int r = 0; r < board_rows_; ++r)
    {
      for (int c = 0; c < board_cols_; ++c)
      {
        obj.emplace_back(static_cast<float>(c * square_size_),
                         static_cast<float>(r * square_size_), 0.0f);
      }
    }
    return obj;
  }

  double ReprojectionRmse(const std::vector<cv::Point3f>& obj,
                          const std::vector<cv::Point2f>& img, const cv::Mat& rvec,
                          const cv::Mat& tvec) const
  {
    std::vector<cv::Point2f> proj;
    cv::projectPoints(obj, rvec, tvec, k_, d_, proj);

    double err2 = 0.0;
    for (size_t i = 0; i < img.size(); ++i)
    {
      const double DX = img[i].x - proj[i].x;
      const double DY = img[i].y - proj[i].y;
      err2 += DX * DX + DY * DY;
    }
    return std::sqrt(err2 / static_cast<double>(
                                std::max<int64_t>(1, static_cast<int64_t>(img.size()))));
  }

  static std::string VecToStr(const cv::Mat& t)
  {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << t.at<double>(0, 0) << " " << t.at<double>(1, 0) << " " << t.at<double>(2, 0);
    return ss.str();
  }

  static std::string RpyToStr(double r, double p, double y)
  {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << r << " " << p << " " << y;
    return ss.str();
  }

  static void RotToRpyTf2(const cv::Mat& R, double& roll, double& pitch, double& yaw)
  {
    tf2::Matrix3x3 m(R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
                     R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
                     R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2));
    m.getRPY(roll, pitch, yaw);
  }

  void PublishDebugImage(const std_msgs::msg::Header& header, const cv::Mat& bgr,
                         const std::string& text, bool ok_text) const
  {
    if (!publish_debug_image_)
    {
      return;
    }
    if (!debug_image_pub_)
    {
      return;
    }
    if (bgr.empty())
    {
      return;
    }

    cv::Mat vis = bgr.clone();
    const cv::Scalar COLOR = ok_text ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    cv::putText(vis, text, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2);

    auto out =
        cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, vis).toImageMsg();
    debug_image_pub_->publish(*out);
  }

  // ---------- 回调 ----------
  void OnCameraInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lk(mtx_);

    // K：相机内参矩阵，按行主序展开在 msg->k[0..8]
    k_ = (cv::Mat_<double>(3, 3) << msg->k[0], msg->k[1], msg->k[2], msg->k[3], msg->k[4],
          msg->k[5], msg->k[6], msg->k[7], msg->k[8]);

    // D：畸变参数，长度可能是 4/5/8 等
    d_ = cv::Mat(static_cast<int>(msg->d.size()), 1, CV_64F);
    for (size_t i = 0; i < msg->d.size(); ++i)
    {
      d_.at<double>(static_cast<int>(i), 0) = msg->d[i];
    }

    have_intrinsics_ = true;
  }

  void OnSynced(const sensor_msgs::msg::Image::ConstSharedPtr& img,
                const sensor_msgs::msg::JointState::ConstSharedPtr& js)
  {
    if (!have_intrinsics_)
    {
      return;
    }

    auto yp = ExtractYawPitch(*js);
    if (!yp.has_value())
    {
      // 没拿到关节角也发一帧图，便于你排查 joint_states 是否正确
      try
      {
        cv::Mat frame = cv_bridge::toCvShare(img, "bgr8")->image;
        PublishDebugImage(img->header, frame, "No yaw/pitch in joint_states", false);
      }
      catch (...)
      {
      }
      return;
    }

    const double YAW = yp->first;
    double pitch = yp->second;
    if (invert_pitch_)
    {
      pitch = -pitch;
    }

    cv::Mat frame;
    try
    {
      frame = cv_bridge::toCvShare(img, "bgr8")->image;
    }
    catch (...)
    {
      return;
    }

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    const cv::Size PATTERN_SIZE(board_cols_, board_rows_);
    std::vector<cv::Point2f> corners;

    bool found = cv::findChessboardCorners(
        gray, PATTERN_SIZE, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

    // 为可视化准备一张图
    cv::Mat vis = frame.clone();

    if (!found)
    {
      PublishDebugImage(img->header, vis, "Chessboard NOT found", false);

      std::lock_guard<std::mutex> lk(mtx_);
      last_detection_.reset();
      return;
    }

    // 画角点（先画粗检测结果）
    cv::drawChessboardCorners(vis, PATTERN_SIZE, corners, found);

    // 亚像素角点精修
    cv::cornerSubPix(
        gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));

    // 再画一次精修后的角点（让你看得更明显）
    cv::drawChessboardCorners(vis, PATTERN_SIZE, corners, true);

    // solvePnP：估计标定板相对相机位姿
    const auto OBJ = MakeObjectPoints();
    cv::Mat rvec, tvec;
    bool ok =
        cv::solvePnP(OBJ, corners, k_, d_, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);

    if (!ok)
    {
      PublishDebugImage(img->header, vis, "PnP failed (check intrinsics/board)", false);

      std::lock_guard<std::mutex> lk(mtx_);
      last_detection_.reset();
      return;
    }

    cv::Mat r_target2cam;
    cv::Rodrigues(rvec, r_target2cam);

    // 构造 gripper2base 的旋转：
    // base=gimbal_odom，gripper=pitch_link
    // yaw 绕 +Z，pitch 绕 -Y，因此使用 Ry(-pitch)
    cv::Mat rg2b = Rz(YAW) * Ry(-pitch);
    cv::Mat tg2b = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);

    double rmse = ReprojectionRmse(OBJ, corners, rvec, tvec);

    // 发布调试图：叠加 yaw/pitch/rmse
    {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(3) << "FOUND  yaw=" << YAW
          << "  pitch=" << pitch << "  rmse=" << rmse << " px";
      // rmse 小一般更可靠；这里简单设个阈值用于“绿/红”
      const bool GOOD = (rmse <= 1.0);
      PublishDebugImage(img->header, vis, oss.str(), GOOD);
    }

    Detection det;
    det.stamp = img->header.stamp;
    det.R_target2cam = r_target2cam;
    det.t_target2cam = tvec.clone();
    det.R_gripper2base = rg2b;
    det.t_gripper2base = tg2b;
    det.yaw = YAW;
    det.pitch = pitch;
    det.reproj_rmse = rmse;

    std::lock_guard<std::mutex> lk(mtx_);
    last_detection_ = det;
  }

  // ---------- 服务 ----------
  void OnCapture(const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
                 const std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!last_detection_.has_value())
    {
      res->success = false;
      res->message = "No valid chessboard detection available.";
      return;
    }

    const rclcpp::Time NOW = this->now();
    const double AGE = (NOW - last_detection_->stamp).seconds();
    if (AGE > max_age_sec_)
    {
      res->success = false;
      std::ostringstream ss;
      ss << "Last detection is stale, age=" << AGE << "s > max_age_sec=" << max_age_sec_;
      res->message = ss.str();
      return;
    }

    // 保存一组样本
    r_gripper2base_.push_back(last_detection_->R_gripper2base);
    t_gripper2base_.push_back(last_detection_->t_gripper2base);

    r_target2cam_.push_back(last_detection_->R_target2cam);
    t_target2cam_.push_back(last_detection_->t_target2cam);

    std::ostringstream ss;
    ss << "Captured sample #" << r_target2cam_.size() << " (reproj_rmse=" << std::fixed
       << std::setprecision(3) << last_detection_->reproj_rmse << " px, "
       << "yaw=" << last_detection_->yaw << ", pitch=" << last_detection_->pitch
       << " rad)";
    res->success = true;
    res->message = ss.str();
  }

  void OnReset(const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
               const std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    r_gripper2base_.clear();
    t_gripper2base_.clear();
    r_target2cam_.clear();
    t_target2cam_.clear();
    res->success = true;
    res->message = "Cleared all samples.";
  }

  void OnSolve(const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
               const std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    std::lock_guard<std::mutex> lk(mtx_);

    const size_t N = r_target2cam_.size();
    if (N < 10)
    {
      res->success = false;
      std::ostringstream ss;
      ss << "Not enough samples: " << N << " (recommend >= 20, minimum ~10).";
      res->message = ss.str();
      return;
    }

    cv::Mat r_cam2gripper, t_cam2gripper;
    try
    {
      const int METHOD_CV = ToOpenCvMethod(method_);
      cv::calibrateHandEye(r_gripper2base_, t_gripper2base_, r_target2cam_, t_target2cam_,
                           r_cam2gripper, t_cam2gripper,
                           static_cast<cv::HandEyeCalibrationMethod>(METHOD_CV));
    }
    catch (const std::exception& e)
    {
      res->success = false;
      res->message = std::string("calibrateHandEye exception: ") + e.what();
      return;
    }

    tf2::Matrix3x3 r_calib(r_cam2gripper.at<double>(0, 0), r_cam2gripper.at<double>(0, 1),
                           r_cam2gripper.at<double>(0, 2), r_cam2gripper.at<double>(1, 0),
                           r_cam2gripper.at<double>(1, 1), r_cam2gripper.at<double>(1, 2),
                           r_cam2gripper.at<double>(2, 0), r_cam2gripper.at<double>(2, 1),
                           r_cam2gripper.at<double>(2, 2));
    tf2::Vector3 t_calib(t_cam2gripper.at<double>(0, 0), t_cam2gripper.at<double>(1, 0),
                         t_cam2gripper.at<double>(2, 0));

    tf2::Matrix3x3 r_link2opt;
    r_link2opt.setRPY(-M_PI / 2.0, 0, -M_PI / 2.0);

    tf2::Matrix3x3 r_target = r_calib * r_link2opt.transpose();  // 旋转矩阵逆即转置

    double roll = NAN, pitch = NAN, yaw = NAN;
    r_target.getRPY(roll, pitch, yaw);

    // 按 URDF <origin xyz="..." rpy="..."> 的格式输出
    std::ostringstream ss;
    ss << "\n=== Hand-Eye Result (parent=pitch_link, child=camera_link) ===\n";
    ss << "URDF xyz (m): " << VecToStr(t_cam2gripper) << "\n";
    ss << "URDF rpy (rad): " << RpyToStr(roll, pitch, yaw) << "\n";
    ss << "Method: " << method_ << "  Samples: " << N << "\n";

    res->success = true;
    res->message = ss.str();

    RCLCPP_INFO(get_logger(), "%s", res->message.c_str());
  }

 private:
  // 参数
  std::string image_topic_;
  std::string camera_info_topic_;
  std::string joint_states_topic_;
  std::string yaw_joint_name_;
  std::string pitch_joint_name_;
  int board_cols_;
  int board_rows_;
  double square_size_;
  double max_age_sec_;
  std::string method_;
  bool invert_pitch_;

  // 调试可视化
  bool publish_debug_image_ = true;
  std::string debug_image_topic_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;

  // 内参
  bool have_intrinsics_ = false;
  cv::Mat k_, d_;

  // 订阅与同步器
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::JointState> joint_sub_;
  std::shared_ptr<
      message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<
          sensor_msgs::msg::Image, sensor_msgs::msg::JointState>>>
      sync_;

  // 服务
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr capture_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr solve_srv_;

  // 样本数据
  std::vector<cv::Mat> r_gripper2base_;
  std::vector<cv::Mat> t_gripper2base_;
  std::vector<cv::Mat> r_target2cam_;
  std::vector<cv::Mat> t_target2cam_;

  // 最近一次检测缓存
  std::mutex mtx_;
  std::optional<Detection> last_detection_;
};

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(HandEyeCalibrateNode)
