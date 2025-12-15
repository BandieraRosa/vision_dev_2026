#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf2/LinearMath/Matrix3x3.h>

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
    // 允许的最大关节速度 (rad/s)，超过此速度认为在运动中
    max_motion_vel_ = declare_parameter<double>("check_max_velocity", 0.05);
    // 允许的最小拉普拉斯方差，低于此值认为图像模糊
    min_blur_score_ = declare_parameter<double>("check_min_blur_score", 100.0);
    // 最小采样间隔角度 (rad)，防止采集重复数据 (默认约 5 度)
    min_angle_dist_ = declare_parameter<double>("check_min_angle_dist", 0.087);
    // 数据时效性
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

  // ---------- 坐标系转换 ----------
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

  // 计算拉普拉斯方差 (模糊检测)
  static double CalcBlurScore(const cv::Mat& gray)
  {
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    return stddev.val[0] * stddev.val[0];
  }

  // 检查机器人是否静止
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
    return cv::CALIB_HAND_EYE_DANIILIDIS;
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

    // 1. 解析关节
    double yaw = 0.0, pitch = 0.0;
    // bool found_joints = false; // 未使用
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

    // 2. 图像处理
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

    // 计算模糊度
    double blur = CalcBlurScore(gray);

    // 检查运动状态
    std::string motion_msg;
    bool is_static = CheckStatic(*js, motion_msg);

    // 检测棋盘格
    const cv::Size PATTERN_SIZE(board_cols_, board_rows_);
    std::vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners(
        gray, PATTERN_SIZE, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

    cv::Mat vis = frame.clone();

    // 状态叠加文字
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

      // PnP
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

      // 投影误差
      std::vector<cv::Point2f> proj;
      cv::projectPoints(obj_pts, rvec, tvec, k_, d_, proj);
      double err_sum = 0;
      for (size_t i = 0; i < corners.size(); ++i)
      {
        err_sum += cv::norm(corners[i] - proj[i]);
      }
      double rmse = std::sqrt(err_sum / static_cast<double>(corners.size()));

      // 保存检测结果
      Detection det;
      det.stamp = img->header.stamp;
      cv::Rodrigues(rvec, det.R_target2cam);
      det.t_target2cam = tvec.clone();

      // 构建 Gripper (Pitch Link) 到 Base (Gimbal Odom) 的变换
      double pitch_sign = invert_pitch_ ? -1.0 : 1.0;
      double pitch_val_corrected = pitch * pitch_sign;
      // 注意：此处严格对应 URDF。Ry(-pitch) 对应 pitch 轴 axis="0 -1 0"
      det.R_gripper2base = Rz(yaw) * Ry(-pitch_val_corrected);
      det.t_gripper2base = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);  // 假设轴心重合
      det.yaw = yaw;
      det.pitch = pitch;
      det.reproj_rmse = rmse;
      det.blur_score = blur;

      {
        std::lock_guard<std::mutex> lk(mtx_);
        last_detection_ = det;
        current_vis_info_ = {true, blur, is_static, motion_msg, rmse};
      }

      // 绘制详细信息
      std::ostringstream s1;
      s1 << "Blur Score: " << std::fixed << std::setprecision(1) << blur
         << (blur < min_blur_score_ ? " (BAD)" : " (OK)");
      draw_text(s1.str(), blur >= min_blur_score_, 1);

      std::ostringstream s2;
      s2 << "Motion: " << (is_static ? "STATIC" : "MOVING")
         << (is_static ? "" : (" [" + motion_msg + "]"));
      draw_text(s2.str(), is_static, 2);

      std::ostringstream s3;
      s3 << "PnP RMSE: " << std::fixed << std::setprecision(3) << rmse;
      draw_text(s3.str(), rmse < 1.0, 3);
    }
    else
    {
      std::lock_guard<std::mutex> lk(mtx_);
      last_detection_.reset();
      current_vis_info_ = {false, blur, is_static, motion_msg, 0.0};
      draw_text("Chessboard NOT Found", false, 1);
    }

    // 发布图片
    if (publish_debug_image_ && debug_image_pub_)
    {
      debug_image_pub_->publish(
          *cv_bridge::CvImage(img->header, "bgr8", vis).toImageMsg());
    }
  }

  // ---------- 服务回调：Capture ----------
  void OnCapture(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                 const std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    std::lock_guard<std::mutex> lk(mtx_);

    // 1. 基础存在性检查
    if (!last_detection_.has_value())
    {
      res->success = false;
      res->message = "No chessboard detected currently.";
      return;
    }
    const auto& det = last_detection_.value();

    // 2. 时效性检查
    if ((now() - det.stamp).seconds() > max_age_sec_)
    {
      res->success = false;
      res->message = "Detection too old (lagging). Check system load.";
      return;
    }

    // 3. 静止检测
    if (!current_vis_info_.is_static)
    {
      res->success = false;
      res->message = "REJECTED: Robot is moving! " + current_vis_info_.motion_msg;
      return;
    }

    // 4. 模糊检测
    if (det.blur_score < min_blur_score_)
    {
      res->success = false;
      res->message = "REJECTED: Image too blurry (score " +
                     std::to_string(static_cast<int>(det.blur_score)) + " < " +
                     std::to_string(static_cast<int>(min_blur_score_)) + ")";
      return;
    }

    // 5. 重复/距离检测
    // 简单策略：计算新样本的 (yaw, pitch) 与已有样本的欧氏距离
    for (const auto& s : samples_)
    {
      double dy = s.yaw - det.yaw;
      double dp = s.pitch - det.pitch;
      double dist = std::sqrt(dy * dy + dp * dp);
      if (dist < min_angle_dist_)
      {
        res->success = false;
        res->message =
            "REJECTED: Pose too close to existing sample (dist=" + std::to_string(dist) +
            " rad). Move joint more.";
        return;
      }
    }

    // --- 通过所有检查，保存样本 ---
    samples_.push_back(det);
    res->success = true;
    res->message = "Captured Sample #" + std::to_string(samples_.size()) +
                   " (RMSE: " + std::to_string(det.reproj_rmse) + ")";
  }

  // ---------- 服务回调：Reset ----------
  void OnReset(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
               const std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    samples_.clear();
    res->success = true;
    res->message = "Reset all samples.";
  }

  // ---------- 服务回调：Solve ----------
  void OnSolve(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
               const std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    if (samples_.size() < 5)
    {  // 至少5组
      res->success = false;
      res->message = "Not enough samples (" + std::to_string(samples_.size()) + "<5).";
      return;
    }

    // 准备数据
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

    tf2::Matrix3x3 r_calib(r_cam2grip.at<double>(0, 0), r_cam2grip.at<double>(0, 1),
                           r_cam2grip.at<double>(0, 2), r_cam2grip.at<double>(1, 0),
                           r_cam2grip.at<double>(1, 1), r_cam2grip.at<double>(1, 2),
                           r_cam2grip.at<double>(2, 0), r_cam2grip.at<double>(2, 1),
                           r_cam2grip.at<double>(2, 2));
    tf2::Vector3 t_calib(t_cam2grip.at<double>(0), t_cam2grip.at<double>(1),
                         t_cam2grip.at<double>(2));

    tf2::Matrix3x3 r_link2opt;
    r_link2opt.setRPY(-CV_PI / 2.0, 0, -CV_PI / 2.0);

    tf2::Matrix3x3 r_final = r_calib * r_link2opt.transpose();

    tf2::Vector3 t_final = t_calib;

    double roll = NAN, pitch = NAN, yaw = NAN;
    r_final.getRPY(roll, pitch, yaw);

    std::ostringstream ss;
    ss << "SUCCESS! \n";
    ss << "xyz: '\"" << t_final.x() << " " << t_final.y() << " " << t_final.z() << "\"\n";
    ss << "rpy: '" << roll << " " << pitch << " " << yaw << "'";

    res->success = true;
    res->message = ss.str();
    RCLCPP_INFO(get_logger(), "\n%s", ss.str().c_str());
  }

  // 参数
  std::string image_topic_, camera_info_topic_, joint_states_topic_;
  std::string yaw_joint_name_, pitch_joint_name_;
  int board_cols_, board_rows_;
  double square_size_, max_age_sec_;
  std::string method_;
  bool invert_pitch_;

  // 阈值
  double max_motion_vel_;
  double min_blur_score_;
  double min_angle_dist_;

  // 状态
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
