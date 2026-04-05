#include "armor_tracker/tracker_node.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cmath>

namespace rm_auto_aim
{
ArmorTrackerNode::ArmorTrackerNode(const rclcpp::NodeOptions& options)
    : Node("armor_tracker", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting TrackerNode!");

  InitParameters();

  // EKF
  // xa = x_armor, xc = x_robot_center
  // state: xc, v_xc, yc, v_yc, za, v_za, yaw, v_yaw, r
  // measurement: xa, ya, za, yaw
  // f - Process function 过程函数对状态进行更新
  auto f = [this](const Eigen::VectorXd& x)
  {
    Eigen::VectorXd x_new = x;
    x_new(0) += x(1) * dt_;
    x_new(2) += x(3) * dt_;
    x_new(4) += x(5) * dt_;
    x_new(6) += x(7) * dt_;
    return x_new;
  };
  // J_f - Jacobian of process function
  auto j_f = [this](const Eigen::VectorXd&)
  {
    Eigen::MatrixXd f(9, 9);
    // clang-format off
    f << 1, dt_, 0, 0, 0, 0, 0, 0, 0, 
         0, 1, 0, 0, 0, 0, 0, 0, 0, 
         0, 0, 1, dt_, 0, 0, 0, 0, 0, 
         0, 0, 0, 1, 0, 0, 0, 0, 0, 
         0, 0, 0, 0, 1, dt_, 0, 0, 0, 
         0, 0, 0, 0, 0, 1, 0, 0, 0, 
         0, 0, 0, 0, 0, 0, 1, dt_, 0, 
         0, 0, 0, 0, 0, 0, 0, 1, 0, 
         0, 0, 0, 0, 0, 0, 0, 0, 1;
    // clang-format on
    return f;
  };
  // h - Observation function 观测函数对状态进行测量
  auto h = [](const Eigen::VectorXd& x)
  {
    Eigen::VectorXd z(4);
    double xc = x(0), yc = x(2), yaw = x(6), r = x(8);
    z(0) = xc - r * cos(yaw);  // xa
    z(1) = yc - r * sin(yaw);  // ya
    z(2) = x(4);               // za
    z(3) = x(6);               // yaw
    return z;
  };
  // J_h - Jacobian of observation function
  // 状态量到观测量的一个转换矩阵，将整车c的状态转换为装甲板a的状态，用预测之后的c推出预测之后的a
  auto j_h = [](const Eigen::VectorXd& x)
  {
    Eigen::MatrixXd h(4, 9);
    double yaw = x(6), r = x(8);
    // clang-format off
    //              xc   v_xc yc   v_yc za   v_za yaw         v_yaw r
    h <<  /*xa*/    1,   0,   0,   0,   0,   0,   r*sin(yaw), 0,   -cos(yaw),
          /*ya*/    0,   0,   1,   0,   0,   0,   -r*cos(yaw),0,   -sin(yaw),
          /*za*/    0,   0,   0,   0,   1,   0,   0,          0,   0,
          /*yaw*/   0,   0,   0,   0,   0,   0,   1,          0,   0;
    // clang-format on
    return h;
  };
  // update_Q - process noise covariance matrix 过程噪声协方差矩阵

  s2_q_x_ = s2_q_x_armor_;
  s2_q_y_ = s2_q_y_armor_;
  s2_q_z_ = s2_q_z_armor_;
  s2_q_yaw_ = s2_q_yaw_armor_;
  s2_q_r_ = s2_q_r_armor_;
  auto u_q = [this]()
  {
    Eigen::MatrixXd q(9, 9);
    double t = dt_, x = s2_q_x_, y = s2_q_y_, z = s2_q_z_, yaw = s2_q_yaw_, r = s2_q_r_;
    double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx = pow(t, 2) * x;
    double q_y_y = pow(t, 4) / 4 * y, q_y_vy = pow(t, 3) / 2 * y, q_vy_vy = pow(t, 2) * y;
    double q_z_z = pow(t, 4) / 4 * z, q_z_vz = pow(t, 3) / 2 * z, q_vz_vz = pow(t, 2) * z;
    double q_yaw_yaw = pow(t, 4) / 4 * yaw, q_yaw_vyaw = pow(t, 3) / 2 * yaw,
           q_vyaw_vyaw = pow(t, 2) * yaw;
    double q_r_r = pow(t, 4) / 4 * r;
    // clang-format off
    //    xc      v_xc     yc      v_yc     za      v_za     yaw         v_yaw        r
    q <<  q_x_x,  q_x_vx,  0,      0,       0,      0,       0,          0,           0,
          q_x_vx, q_vx_vx, 0,      0,       0,      0,       0,          0,           0,
          0,      0,       q_y_y,  q_y_vy,  0,      0,       0,          0,           0,
          0,      0,       q_y_vy, q_vy_vy, 0,      0,       0,          0,           0,
          0,      0,       0,      0,       q_z_z,  q_z_vz,  0,          0,           0,
          0,      0,       0,      0,       q_z_vz, q_vz_vz, 0,          0,           0,
          0,      0,       0,      0,       0,      0,       q_yaw_yaw,  q_yaw_vyaw,  0,
          0,      0,       0,      0,       0,      0,       q_yaw_vyaw, q_vyaw_vyaw, 0,
          0,      0,       0,      0,       0,      0,       0,          0,           q_r_r;
    // clang-format on
    return q;
  };
  // update_R - measurement noise covariance matrix 观测噪声协方差矩阵
  auto u_r = [this](const Eigen::VectorXd& x)
  {
    Eigen::DiagonalMatrix<double, 4> r;

    constexpr double d2_min = 0.25;
    constexpr double d2_max = 25.0;

    double min_xyz_var = r_xyz_factor_ * d2_min;
    double max_xyz_var =
        r_xyz_factor_ * d2_max;  // 上限设置，防止距离过大时，测量噪声过大
    // 陀螺仪飘了怎么办，旋转中心与世界系原点相同，不影响距离估计
    double d2 = x[0] * x[0] + x[2] * x[2] + x[4] * x[4];
    d2 = std::clamp(d2, d2_min, d2_max);
    // 线性归一，认为噪声方差与距离平方成正相关，符合实际。距离变大时，噪声方差变化显著，设上限截断效果不一定好，可根据实际情况调整为log归一化
    double r_xyz_var =
        min_xyz_var + (max_xyz_var - min_xyz_var) * (d2 - d2_min) / (d2_max - d2_min);
    double r_x_var = r_xyz_var, r_y_var = r_xyz_var / 2, r_z_var = r_xyz_var / 2;
    double max_yaw_var = r_yaw_ * 10;
    double r_yaw_var = r_yaw_ * 1 / std::fabs(std::cos(x[6]));
    r_yaw_var = std::min(r_yaw_var, max_yaw_var);

    r.diagonal() << r_x_var, r_y_var, r_z_var, r_yaw_var;

    // xyz与yaw在pnp中耦合解算，协方差设为0不算严谨，可尝试设置图像噪声矩阵驱动观测噪声
    // // clang-format off
    // //   xc         yc             za             yaw
    // r << r_xyz_var, 0,             0,             0,
    //      0,         r_xyz_var / 2, 0,             0,
    //      0,         0,             r_xyz_var / 2, 0,
    //      0,         0,             0,             r_yaw_var;
    // // clang-format on

    return r;
  };
  // P - error estimate covariance matrix
  Eigen::DiagonalMatrix<double, 9> p0;
  p0.setIdentity();

  // outpost的EKF参数
  auto switch_q = [this](bool flag)
  {
    s2_q_x_ = flag ? s2qxyz_outpost_ : s2_q_x_armor_;
    s2_q_yaw_ = flag ? s2qyaw_outpost_ : s2_q_yaw_armor_;
    s2_q_r_ = flag ? s2qr_outpost_ : s2_q_r_armor_;
  };

  tracker_->ekf = ExtendedKalmanFilter{f, h, j_f, j_h, u_q, u_r, p0};
  tracker_->switch_q_ = switch_q;
  using std::placeholders::_1;

  // Subscriber with tf2 message_filter
  // tf2 relevant
  tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  // Create the timer interface before call to waitForTransform,
  // to avoid a tf2_ros::CreateTimerInterfaceException exception
  auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
      this->get_node_base_interface(), this->get_node_timers_interface());
  tf2_buffer_->setCreateTimerInterface(timer_interface);
  tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);
  // subscriber and filter
  armors_sub_.subscribe(this, "/detector/armors", rmw_qos_profile_sensor_data);
  target_frame_ = this->declare_parameter("target_frame", "odom");
  armors_filter_ = std::make_shared<armors_tf2_filter>(
      armors_sub_, *tf2_buffer_, target_frame_, 10, this->get_node_logging_interface(),
      this->get_node_clock_interface(), std::chrono::duration<int>(1));

  // Register a callback with tf2_ros::MessageFilter to be called when transforms are
  // available
  armors_filter_->registerCallback(&ArmorTrackerNode::ArmorsCallback, this);



  // Measurement publisher (for debug usage)
  info_pub_ =
      this->create_publisher<auto_aim_interfaces::msg::TrackerInfo>("/tracker/info", 10);

  // Publisher
  target_pub_ = this->create_publisher<auto_aim_interfaces::msg::Target>(
      "/tracker/target", rclcpp::SensorDataQoS());
}

void ArmorTrackerNode::InitParameters()
{
  // 最大可观测装甲板距离
  max_armor_distance_ = this->declare_parameter("max_armor_distance", 10.0);

  auto robot_type = this->declare_parameter<std::string>("robot_type", "default");
  is_hero_ = (robot_type == "hero");

  // Tracker init parameters
  double max_match_distance = this->declare_parameter("tracker.max_match_distance", 0.15);
  double max_match_yaw_diff = this->declare_parameter("tracker.max_match_yaw_diff", 1.0);
  tracker_ = std::make_unique<Tracker>(max_match_distance, max_match_yaw_diff);
  tracker_->tracking_thres =
      static_cast<int>(this->declare_parameter("tracker.tracking_thres", 5));
  Tracker::outpost_cast_threshold = static_cast<double>(
      this->declare_parameter("tracker.outpost.outpost_cast_threshold", 0.18));
  Tracker::outpost_dz =
      static_cast<double>(this->declare_parameter("tracker.outpost.outpost_dz", 0.1));
  Tracker::outpost_r =
      static_cast<double>(this->declare_parameter("tracker.outpost.outpost_r", 0.2765));
  lost_time_thres_ = this->declare_parameter("tracker.lost_time_thres", 0.3);
  change_time_thres_ = this->declare_parameter("tracker.change_time_thres", 0.3);
  // EKF init parameters
  s2_q_x_armor_ = this->declare_parameter("ekf.sigma2_q_x", 0.1);
  s2_q_y_armor_ = this->declare_parameter("ekf.sigma2_q_y", 0.1);
  s2_q_z_armor_ = this->declare_parameter("ekf.sigma2_q_z", 0.1);
  s2_q_yaw_armor_ = this->declare_parameter("ekf.sigma2_q_yaw", 2.0);
  s2_q_r_armor_ = this->declare_parameter("ekf.sigma2_q_r", 80.0);
  s2qxyz_outpost_ = this->declare_parameter("ekf.sigma2_q_xyz_outpost", 0.005);
  s2qyaw_outpost_ = this->declare_parameter("ekf.sigma2_q_yaw_outpost", 2.0);
  s2qr_outpost_ = this->declare_parameter("ekf.sigma2_q_r_outpost", 0.0);

  r_xyz_factor_ = this->declare_parameter("ekf.r_xyz_factor", 0.05);
  r_yaw_ = this->declare_parameter("ekf.r_yaw", 0.02);
}

void ArmorTrackerNode::ArmorsCallback(
    const auto_aim_interfaces::msg::Armors::SharedPtr armors_msg)
{
  if (is_hero_)
  {
    if (armors_msg->header.frame_id != last_frame_id_)
    {
      last_frame_id_ = armors_msg->header.frame_id;
      // todo: switch table
      RCLCPP_INFO(this->get_logger(), "Switched trajectory table due to new frame id: %s",
                  last_frame_id_.c_str());
    }
  }

  // Tranform armor position from image frame to world coordinate
  for (auto& armor : armors_msg->armors)
  {
    geometry_msgs::msg::PoseStamped ps;
    ps.header = armors_msg->header;
    ps.pose = armor.pose;
    try
    {
      armor.pose = tf2_buffer_->transform(ps, target_frame_).pose;
    }
    catch (const tf2::ExtrapolationException& ex)
    {
      RCLCPP_ERROR(get_logger(), "Error while transforming %s", ex.what());
      return;
    }
  }

  // Filter abnormal armors
  armors_msg->armors.erase(
      std::remove_if(
          armors_msg->armors.begin(), armors_msg->armors.end(),
          [this](const auto_aim_interfaces::msg::Armor& armor)
          {
            return std::fabs(armor.pose.position.z) > 2.0 ||
                   Eigen::Vector2d(armor.pose.position.x, armor.pose.position.y).norm() >
                       max_armor_distance_;
          }),
      armors_msg->armors.end());

  // Init message
  auto_aim_interfaces::msg::TrackerInfo info_msg;
  auto_aim_interfaces::msg::Target target_msg;
  auto_aim_interfaces::msg::Send send_msg;
  rclcpp::Time time = armors_msg->header.stamp;
  target_msg.header.stamp = time;
  target_msg.header.frame_id = target_frame_;

  if (tracker_->tracker_state == Tracker::State::LOST)
  {
    tracker_->Init(armors_msg);
    target_msg.tracking = false;
  }
  else
  {
    // 求时间差
    dt_ = (time - last_time_).seconds();
    tracker_->lost_thres = static_cast<int>(lost_time_thres_ / dt_);
    tracker_->change_thres = static_cast<int>(change_time_thres_ / dt_);
    tracker_->Update(armors_msg);

    if (tracker_->tracker_state == Tracker::State::DETECTING)
    {
      target_msg.tracking = false;
    }
    else if (tracker_->tracker_state == Tracker::State::TRACKING ||
             tracker_->tracker_state == Tracker::State::TEMP_LOST)
    {
      target_msg.tracking = true;
      // Fill target message
      const auto& state = tracker_->target_state;
      target_msg.type = tracker_->tracked_armor_type;
      target_msg.armors_num = static_cast<int>(tracker_->tracked_armors_num);
      target_msg.position.x = state(0);
      target_msg.velocity.x = state(1);
      target_msg.position.y = state(2);
      target_msg.velocity.y = state(3);
      target_msg.position.z = state(4);
      target_msg.velocity.z = state(5);
      target_msg.yaw = state(6);
      target_msg.v_yaw = state(7);
      target_msg.radius_1 = state(8);
      target_msg.radius_2 = tracker_->another_r;
      target_msg.dz = tracker_->dz;
      target_msg.outpost_idx = tracker_->outpost_idx;
    }
  }

  last_time_ = time;

  target_pub_->publish(target_msg);

  // Publish Info
  info_msg.position_diff = tracker_->info_position_diff;
  info_msg.yaw_diff = tracker_->info_yaw_diff;
  info_msg.position.x = tracker_->measurement(0);
  info_msg.position.y = tracker_->measurement(1);
  info_msg.position.z = tracker_->measurement(2);
  info_msg.yaw = tracker_->measurement(3);
  info_msg.outpost_idx = Tracker::outpost_idx;
  info_pub_->publish(info_msg);
}
}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its
// library is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorTrackerNode)