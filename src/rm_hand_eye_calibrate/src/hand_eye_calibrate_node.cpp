#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_srvs/srv/trigger.hpp>

class HandEyeCalibrateNode : public rclcpp::Node
{
 public:
  HandEyeCalibrateNode(const rclcpp::NodeOptions& options)
      : Node("hand_eye_calibrator_node", options)
  {
    // -------- 基础参数 --------
    image_topic_ = declare_parameter<std::string>("image_topic", "/image_raw");
    camera_info_topic_ =
        declare_parameter<std::string>("camera_info_topic", "/camera_info");
    joint_states_topic_ =
        declare_parameter<std::string>("joint_states_topic", "/joint_states");
    yaw_joint_name_ = declare_parameter<std::string>("yaw_joint_name", "yaw_joint");
    pitch_joint_name_ = declare_parameter<std::string>("pitch_joint_name", "pitch_joint");

    // 棋盘格
    board_cols_ = static_cast<int>(declare_parameter<int>("board_cols", 11));
    board_rows_ = static_cast<int>(declare_parameter<int>("board_rows", 8));
    square_size_ = declare_parameter<double>("square_size", 0.02);

    // 算法
    method_ = declare_parameter<std::string>("handeye_method", "TSAI");
    invert_pitch_ = declare_parameter<bool>("invert_pitch_sign", false);

    // -------- 检测阈值 --------
    max_motion_vel_ = declare_parameter<double>("check_max_velocity", 0.05);
    min_blur_score_ = declare_parameter<double>("check_min_blur_score", 100.0);
    min_angle_dist_ = declare_parameter<double>("check_min_angle_dist", 0.087);
    max_age_sec_ = declare_parameter<double>("max_age_sec", 0.5);

    // -------- 调试图像 --------
    publish_debug_image_ = declare_parameter<bool>("publish_debug_image", true);
    debug_image_topic_ =
        declare_parameter<std::string>("debug_image_topic", "/rm_hand_eye/debug_image");

    debug_image_pub_ = create_publisher<sensor_msgs::msg::Image>(debug_image_topic_, 1);

    // -------- 订阅 --------
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_, rclcpp::SensorDataQoS(),
        std::bind(&HandEyeCalibrateNode::OnCameraInfo, this, std::placeholders::_1));

    // 同步订阅
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

    RCLCPP_INFO(get_logger(),
                "HandEye Node Started. Use RQT Service Caller to interact.");
  }

 private:
  struct Detection
  {
    rclcpp::Time stamp;
    cv::Mat R_target2cam;  // 3x3
    cv::Mat t_target2cam;  // 3x1
    cv::Mat R_gripper2base;
    cv::Mat t_gripper2base;
    double yaw = 0.0;
    double pitch = 0.0;
    double reproj_rmse = 0.0;
    double blur_score = 0.0;
  };

  static double LimitRad(double angle)
  {
    while (angle > CV_PI)
    {
      angle -= 2 * CV_PI;
    }
    while (angle < -CV_PI)
    {
      angle += 2 * CV_PI;
    }
    return angle;
  }

  static Eigen::Vector3d CustomEulers(Eigen::Quaterniond q, int axis0, int axis1,
                                      int axis2, bool extrinsic)
  {
    if (!extrinsic)
    {
      std::swap(axis0, axis2);
    }

    auto i = axis0, j = axis1, k = axis2;
    auto is_proper = (i == k);
    if (is_proper)
    {
      k = 3 - i - j;
    }
    auto sign = (i - j) * (j - k) * (k - i) / 2;

    double a = NAN, b = NAN, c = NAN, d = NAN;
    Eigen::Vector4d xyzw = q.coeffs();  // Eigen stored as x,y,z,w
    // 注意：离线代码中 q.coeffs() 返回的是 [x, y, z, w]，但代码逻辑里用了 xyzw[3] 作为 w
    // (实部) 这是正确的，Eigen Quaternion 内部存储顺序确实是 x, y, z, w。

    if (is_proper)
    {
      a = xyzw[3];
      b = xyzw[i];
      c = xyzw[j];
      d = xyzw[k] * sign;
    }
    else
    {
      a = xyzw[3] - xyzw[j];
      b = xyzw[i] + xyzw[k] * sign;
      c = xyzw[j] + xyzw[3];
      d = xyzw[k] * sign - xyzw[i];
    }

    Eigen::Vector3d eulers;
    auto n2 = a * a + b * b + c * c + d * d;
    eulers[1] = std::acos(2 * (a * a + b * b) / n2 - 1);

    auto half_sum = std::atan2(b, a);
    auto half_diff = std::atan2(-d, c);

    auto eps = 1e-7;
    auto safe1 = std::abs(eulers[1]) >= eps;
    auto safe2 = std::abs(eulers[1] - CV_PI) >= eps;
    auto safe = safe1 && safe2;
    if (safe)
    {
      eulers[0] = half_sum + half_diff;
      eulers[2] = half_sum - half_diff;
    }
    else
    {
      if (!extrinsic)
      {
        eulers[0] = 0;
        if (!safe1) eulers[2] = 2 * half_sum;
        if (!safe2) eulers[2] = -2 * half_diff;
      }
      else
      {
        eulers[2] = 0;
        if (!safe1) eulers[0] = 2 * half_sum;
        if (!safe2) eulers[0] = 2 * half_diff;
      }
    }

    for (int i = 0; i < 3; i++) eulers[i] = LimitRad(eulers[i]);

    if (!is_proper)
    {
      eulers[2] *= sign;
      eulers[1] -= CV_PI / 2;
    }

    if (!extrinsic) std::swap(eulers[0], eulers[2]);

    return eulers;
  }

  static Eigen::Vector3d custom_eulers(Eigen::Matrix3d R, int axis0, int axis1, int axis2,
                                       bool extrinsic = true)
  {
    Eigen::Quaterniond q(R);
    return CustomEulers(q, axis0, axis1, axis2, extrinsic);
  }

  static cv::Mat Rz(double yaw)
  {
    double c = cos(yaw), s = sin(yaw);
    return (cv::Mat_<double>(3, 3) << c, -s, 0, s, c, 0, 0, 0, 1);
  }

  static cv::Mat Ry(double pitch)
  {
    double c = cos(pitch), s = sin(pitch);
    return (cv::Mat_<double>(3, 3) << c, 0, s, 0, 1, 0, -s, 0, c);
  }

  static double CalcBlurScore(const cv::Mat& gray)
  {
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    return stddev.val[0] * stddev.val[0];
  }

  bool CheckStatic(const sensor_msgs::msg::JointState& js, std::string& err_msg)
  {
    for (size_t i = 0; i < js.name.size(); ++i)
    {
      if (js.name[i] == yaw_joint_name_ || js.name[i] == pitch_joint_name_)
      {
        if (js.velocity.size() > i && std::abs(js.velocity[i]) > max_motion_vel_)
        {
          std::ostringstream ss;
          ss << "Joint " << js.name[i] << " moving (vel=" << std::fixed
             << std::setprecision(3) << js.velocity[i] << ")";
          err_msg = ss.str();
          return false;
        }
      }
    }
    return true;
  }

  int ToOpenCvMethod(const std::string& m) const
  {
    std::string u = m;
    std::transform(u.begin(), u.end(), u.begin(), ::toupper);
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
    return cv::CALIB_HAND_EYE_TSAI;
  }

  // ---------- 回调 ----------
  void OnCameraInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    k_ = (cv::Mat_<double>(3, 3) << msg->k[0], msg->k[1], msg->k[2], msg->k[3], msg->k[4],
          msg->k[5], msg->k[6], msg->k[7], msg->k[8]);
    d_ = cv::Mat(static_cast<int>(msg->d.size()), 1, CV_64F);
    for (size_t i = 0; i < msg->d.size(); ++i)
    {
      d_.at<double>(static_cast<int>(i), 0) = msg->d[i];
    }
    have_intrinsics_ = true;
  }

  void OnSynced(const sensor_msgs::msg::Image::ConstSharedPtr img,
                const sensor_msgs::msg::JointState::ConstSharedPtr js)
  {
    if (!have_intrinsics_)
    {
      return;
    }

    double yaw = 0.0, pitch = 0.0;
    for (size_t i = 0; i < js->name.size(); ++i)
    {
      if (js->name[i] == yaw_joint_name_)
      {
        yaw = js->position[i];
      }
      if (js->name[i] == pitch_joint_name_)
      {
        pitch = js->position[i];
      }
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
    double blur = CalcBlurScore(gray);

    std::string motion_msg;
    bool is_static = CheckStatic(*js, motion_msg);

    const cv::Size PATTERN_SIZE(board_cols_, board_rows_);
    std::vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners(
        gray, PATTERN_SIZE, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

    cv::Mat vis = frame.clone();
    auto draw_text = [&](const std::string& txt, bool ok, int line)
    {
      cv::Scalar color = ok ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
      cv::putText(vis, txt, cv::Point(20, 30 * line), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                  color, 2);
    };

    if (found)
    {
      cv::cornerSubPix(
          gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
      cv::drawChessboardCorners(vis, PATTERN_SIZE, corners, true);

      std::vector<cv::Point3f> obj_pts;
      for (int r = 0; r < board_rows_; ++r)
      {
        for (int c = 0; c < board_cols_; ++c)
        {
          obj_pts.emplace_back(c * square_size_, r * square_size_, 0.0f);
        }
      }

      cv::Mat rvec, tvec;
      cv::solvePnP(obj_pts, corners, k_, d_, rvec, tvec, false, cv::SOLVEPNP_IPPE);

      std::vector<cv::Point2f> proj;
      cv::projectPoints(obj_pts, rvec, tvec, k_, d_, proj);
      double err_sum = 0;
      for (size_t i = 0; i < corners.size(); ++i)
      {
        err_sum += cv::norm(corners[i] - proj[i]);
      }
      double rmse = std::sqrt(err_sum / static_cast<double>(corners.size()));

      Detection det;
      det.stamp = img->header.stamp;
      cv::Rodrigues(rvec, det.R_target2cam);
      det.t_target2cam = tvec.clone();

      det.R_gripper2base = Rz(yaw) * Ry(-pitch);
      det.t_gripper2base = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);
      det.yaw = yaw;
      det.pitch = pitch;
      det.reproj_rmse = rmse;
      det.blur_score = blur;

      {
        std::lock_guard<std::mutex> lk(mtx_);
        last_detection_ = det;
        current_vis_info_ = {true, blur, is_static, motion_msg, rmse};
      }

      std::ostringstream s1;
      s1 << "Blur: " << std::fixed << std::setprecision(1) << blur;
      draw_text(s1.str(), blur >= min_blur_score_, 1);
      draw_text(is_static ? "STATIC" : "MOVING", is_static, 2);
      std::ostringstream s3;
      s3 << "PnP RMSE: " << std::fixed << std::setprecision(3) << rmse;
      draw_text(s3.str(), rmse < 1.0, 3);
    }
    else
    {
      std::lock_guard<std::mutex> lk(mtx_);
      last_detection_.reset();
      current_vis_info_ = {false, blur, is_static, motion_msg, 0.0};
      draw_text("NO CHESSBOARD", false, 1);
    }

    if (publish_debug_image_ && debug_image_pub_)
    {
      debug_image_pub_->publish(
          *cv_bridge::CvImage(img->header, "bgr8", vis).toImageMsg());
    }
  }

  // ---------- Capture ----------
  void OnCapture(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                 const std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    std::lock_guard<std::mutex> lk(mtx_);

    if (!last_detection_.has_value())
    {
      res->success = false;
      res->message = "No chessboard detected.";
      return;
    }
    const auto& det = last_detection_.value();

    if ((now() - det.stamp).seconds() > max_age_sec_)
    {
      res->success = false;
      res->message = "Detection lagging.";
      return;
    }

    if (!current_vis_info_.is_static)
    {
      res->success = false;
      res->message = "REJECTED: Moving!";
      return;
    }

    if (det.blur_score < min_blur_score_)
    {
      res->success = false;
      res->message = "REJECTED: Blurry!";
      return;
    }

    for (const auto& s : samples_)
    {
      double dist =
          std::sqrt(std::pow(s.yaw - det.yaw, 2) + std::pow(s.pitch - det.pitch, 2));
      if (dist < min_angle_dist_)
      {
        res->success = false;
        res->message = "REJECTED: Too close to existing sample.";
        return;
      }
    }

    samples_.push_back(det);
    res->success = true;
    res->message = "Captured Sample #" + std::to_string(samples_.size()) +
                   " (RMSE: " + std::to_string(det.reproj_rmse) + ")";
  }

  // ---------- Reset ----------
  void OnReset(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
               const std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    samples_.clear();
    res->success = true;
    res->message = "Reset all samples.";
  }

  // ---------- Solve ----------
  void OnSolve(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
               const std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    if (samples_.size() < 5)
    {
      res->success = false;
      res->message = "Need more samples (<5).";
      return;
    }

    std::vector<cv::Mat> r_g2b, t_g2b, r_t2c, t_t2c;
    for (const auto& s : samples_)
    {
      r_g2b.push_back(s.R_gripper2base);
      t_g2b.push_back(s.t_gripper2base);
      r_t2c.push_back(s.R_target2cam);
      t_t2c.push_back(s.t_target2cam);
    }

    cv::Mat r_cam2grip, t_cam2grip;
    try
    {
      int method_id = ToOpenCvMethod(method_);
      cv::calibrateHandEye(r_g2b, t_g2b, r_t2c, t_t2c, r_cam2grip, t_cam2grip,
                           static_cast<cv::HandEyeCalibrationMethod>(method_id));
    }
    catch (cv::Exception& e)
    {
      res->success = false;
      res->message = e.what();
      return;
    }

    std::ostringstream ss;

    tf2::Matrix3x3 r_calib(r_cam2grip.at<double>(0, 0), r_cam2grip.at<double>(0, 1),
                           r_cam2grip.at<double>(0, 2), r_cam2grip.at<double>(1, 0),
                           r_cam2grip.at<double>(1, 1), r_cam2grip.at<double>(1, 2),
                           r_cam2grip.at<double>(2, 0), r_cam2grip.at<double>(2, 1),
                           r_cam2grip.at<double>(2, 2));

    tf2::Matrix3x3 r_link2opt;
    r_link2opt.setRPY(-CV_PI / 2.0, 0, -CV_PI / 2.0);
    tf2::Matrix3x3 r_final = r_calib * r_link2opt.transpose();

    double roll, pitch, yaw;
    r_final.getRPY(roll, pitch, yaw);

    ss << "=== ROS URDF Format ===\n";
    ss << "xyz: \"" << t_cam2grip.at<double>(0) << " " << t_cam2grip.at<double>(1) << " "
       << t_cam2grip.at<double>(2) << "\"\n";
    ss << "rpy: \"" << roll << " " << pitch << " " << yaw << "\"\n\n";

    // --- 2. 离线参考代码的 Ideal Deviation 输出 (Eigen Custom Eulers) ---

    // (A) 转换 R_camera2gimbal 到 Eigen
    Eigen::Matrix3d R_camera2gimbal_eigen;
    cv::cv2eigen(r_cam2grip, R_camera2gimbal_eigen);

    // (B) 定义理想变换 R_gimbal2ideal {{0, -1, 0}, {0, 0, -1}, {1, 0, 0}}
    Eigen::Matrix3d R_gimbal2ideal;
    R_gimbal2ideal << 0, -1, 0, 0, 0, -1, 1, 0, 0;

    // (C) 计算 R_camera2ideal
    Eigen::Matrix3d R_camera2ideal = R_gimbal2ideal * R_camera2gimbal_eigen;

    // (D) 计算偏角 (Axis: 1, 0, 2)
    Eigen::Vector3d ypr = custom_eulers(R_camera2ideal, 1, 0, 2, true);

    ss << "=== Reference Code Format (Ideal Deviation) ===\n";
    // 离线代码输出的是角度，这里我们按要求输出弧度
    ss << "Ideal RPY (rad): \"" << ypr[0] << " " << ypr[1] << " " << ypr[2] << "\"\n";
    ss << "(Note: Ref code uses Y-X-Z eulers for this deviation)";

    res->success = true;
    res->message = ss.str();
    RCLCPP_INFO(get_logger(), "\n%s", ss.str().c_str());
  }

  std::string image_topic_, camera_info_topic_, joint_states_topic_;
  std::string yaw_joint_name_, pitch_joint_name_;
  int board_cols_, board_rows_;
  double square_size_, max_age_sec_;
  std::string method_;
  bool invert_pitch_;
  double max_motion_vel_, min_blur_score_, min_angle_dist_;
  std::mutex mtx_;
  bool have_intrinsics_ = false;
  cv::Mat k_, d_;
  std::optional<Detection> last_detection_;
  struct VisInfo
  {
    bool has_data;
    double blur;
    bool is_static;
    std::string motion_msg;
    double rmse;
  } current_vis_info_;
  std::vector<Detection> samples_;
  bool publish_debug_image_;
  std::string debug_image_topic_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::JointState> joint_sub_;
  std::shared_ptr<
      message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<
          sensor_msgs::msg::Image, sensor_msgs::msg::JointState>>>
      sync_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr capture_srv_, reset_srv_, solve_srv_;
};

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(HandEyeCalibrateNode)
