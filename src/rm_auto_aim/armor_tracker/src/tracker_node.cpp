#include "armor_tracker/tracker_node.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cmath>

#include "armor_tracker/tracker.hpp"

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
  // f - Process function
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
  // h - Observation function
  auto h = [](const Eigen::VectorXd& x)
  {
    Eigen::VectorXd z(4);
    double xc = x(0), yc = x(2), yaw = x(6), r = x(8);
    z(0) = xc - r * cos(yaw);
    z(1) = yc - r * sin(yaw);
    z(2) = x(4);
    z(3) = x(6);
    return z;
  };
  // J_h - Jacobian of observation function
  auto j_h = [](const Eigen::VectorXd& x)
  {
    Eigen::MatrixXd h(4, 9);
    double yaw = x(6), r = x(8);
    // clang-format off
    h <<  1,   0,   0,   0,   0,   0,   r*sin(yaw), 0,   -cos(yaw),
          0,   0,   1,   0,   0,   0,   -r*cos(yaw),0,   -sin(yaw),
          0,   0,   0,   0,   1,   0,   0,          0,   0,
          0,   0,   0,   0,   0,   0,   1,          0,   0;
    // clang-format on
    return h;
  };
  // update_Q - process noise covariance matrix
  s2_q_x_ = s2_q_x_armor_;
  s2_q_y_ = s2_q_y_armor_;
  s2_q_z_ = s2_q_z_armor_;
  s2_q_yaw_ = s2_q_yaw_armor_;
  s2_q_r_ = s2_q_r_armor_;
  auto u_q = [this]()
  {
    Eigen::MatrixXd q = Eigen::MatrixXd::Zero(9, 9);
    double t = dt_;

    bool boost_on = (tracker_ && tracker_->NeedManeuverBoost());

    double xy_scale = boost_on ? q_boost_xy_ : 1.0;
    double z_scale = boost_on ? q_boost_z_ : 1.0;
    double yaw_scale = boost_on ? q_boost_yaw_ : 1.0;

    auto add_cv_block = [&](int idx_pos, int idx_vel, double sigma2)
    {
      double a = std::pow(t, 4) / 4.0 * sigma2;
      double b = std::pow(t, 3) / 2.0 * sigma2;
      double c = std::pow(t, 2) * sigma2;
      q(idx_pos, idx_pos) = a;
      q(idx_pos, idx_vel) = b;
      q(idx_vel, idx_pos) = b;
      q(idx_vel, idx_vel) = c;
    };

    add_cv_block(0, 1, xy_scale * s2_q_x_);
    add_cv_block(2, 3, xy_scale * s2_q_y_);
    add_cv_block(4, 5, z_scale * s2_q_z_);
    add_cv_block(6, 7, yaw_scale * s2_q_yaw_);

    q(8, 8) = std::max(1e-8, t * s2_q_r_);
    return q;

    // Eigen::MatrixXd q(9, 9);
    // double t = dt_, x = s2_q_x_, y = s2_q_y_, z = s2_q_z_, yaw = s2_q_yaw_, r =
    // s2_q_r_; double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx =
    // pow(t, 2) * x; double q_y_y = pow(t, 4) / 4 * y, q_y_vy = pow(t, 3) / 2 * y,
    // q_vy_vy = pow(t, 2) * y; double q_z_z = pow(t, 4) / 4 * z, q_z_vz = pow(t, 3) / 2 *
    // z, q_vz_vz = pow(t, 2) * z; double q_yaw_yaw = pow(t, 4) / 4 * yaw, q_yaw_vyaw =
    // pow(t, 3) / 2 * yaw,
    //        q_vyaw_vyaw = pow(t, 2) * yaw;
    // double q_r_r = pow(t, 4) / 4 * r;
    // // clang-format off
    // q <<  q_x_x,  q_x_vx,  0,      0,       0,      0,       0,          0, 0,
    //       q_x_vx, q_vx_vx, 0,      0,       0,      0,       0,          0, 0, 0, 0,
    //       q_y_y,  q_y_vy,  0,      0,       0,          0,           0, 0,      0,
    //       q_y_vy, q_vy_vy, 0,      0,       0,          0,           0, 0,      0, 0,
    //       0,       q_z_z,  q_z_vz,  0,          0,           0, 0,      0,       0, 0,
    //       q_z_vz, q_vz_vz, 0,          0,           0, 0,      0,       0,      0, 0,
    //       0,       q_yaw_yaw,  q_yaw_vyaw,  0, 0,      0,       0,      0,       0, 0,
    //       q_yaw_vyaw, q_vyaw_vyaw, 0, 0,      0,       0,      0,       0,      0, 0,
    //       0,           q_r_r;
    // // clang-format on
    // return q;
  };
  // update_R - measurement noise covariance matrix
  auto u_r = [this](const Eigen::VectorXd& x)
  {
    Eigen::DiagonalMatrix<double, 4> r;
    double factor = r_xyz_factor_;
    r.diagonal() << abs(factor * x[0]), abs(factor * x[2]), abs(factor * x[4]), r_yaw_;
    return r;

    // -----------------------------------------------------------------------------------------------------------

    // Eigen::DiagonalMatrix<double, 4> r;

    // constexpr double d2_min = 0.25;
    // constexpr double d2_max = 25.0;

    // double min_xyz_var = r_xyz_factor_ * d2_min;
    // double max_xyz_var = r_xyz_factor_ * d2_max;
    // double d2 = x[0] * x[0] + x[2] * x[2] + x[4] * x[4];
    // d2 = std::clamp(d2, d2_min, d2_max);
    // double r_xyz_var =
    //     min_xyz_var + (max_xyz_var - min_xyz_var) * (d2 - d2_min) / (d2_max - d2_min);
    // double r_x_var = r_xyz_var, r_y_var = r_xyz_var / 2, r_z_var = r_xyz_var / 2;
    // double max_yaw_var = r_yaw_ * 10;
    // double r_yaw_var = r_yaw_ * 1 / std::fabs(std::cos(x[6]));
    // r_yaw_var = std::min(r_yaw_var, max_yaw_var);

    // r.diagonal() << r_x_var, r_y_var, r_z_var, r_yaw_var;
    // return r;
    // -----------------------------------------------------------------------------------------------------------
    // Eigen::DiagonalMatrix<double, 4> r;

    // auto wrap_to_pi = [](double a) { return std::atan2(std::sin(a), std::cos(a)); };

    // const double xc = x(0);
    // const double yc = x(2);
    // const double za = x(4);
    // const double yaw = x(6);
    // const double radius = x(8);

    // const double xa = xc - radius * std::cos(yaw);
    // const double ya = yc - radius * std::sin(yaw);

    // const double dist = std::sqrt(xa * xa + ya * ya + za * za);
    // const double los_yaw = std::atan2(ya, xa);
    // const double oblique = std::min(std::abs(wrap_to_pi(yaw - los_yaw)), 1.57);

    // const double sigma_xy =
    //     r_xyz_base_ + r_xyz_dist_gain_ * dist + r_xyz_oblique_gain_ *
    //     std::log1p(oblique);

    // const double sigma_z = sigma_xy * r_z_scale_;

    // double sigma_yaw =
    //     r_yaw_base_ + r_yaw_dist_gain_ * std::log1p(dist) + r_yaw_oblique_gain_ *
    //     oblique;

    // if (tracker_->tracked_id == "outpost")
    // {
    //   sigma_yaw *= r_yaw_outpost_scale_;
    // }

    // r.diagonal() << sigma_xy * sigma_xy, sigma_xy * sigma_xy, sigma_z * sigma_z,
    //     sigma_yaw * sigma_yaw;
    // return r;
    // -----------------------------------------------------------------------------------------------------------

    // Eigen::MatrixXd r = Eigen::MatrixXd::Zero(4, 4);

    // // 先从状态恢复“预测装甲板位置”（世界系）
    // const double xc = x(0);
    // const double yc = x(2);
    // const double za = x(4);
    // const double armor_yaw = x(6);
    // const double radius = x(8);

    // const Eigen::Vector3d p_world(xc - radius * std::cos(armor_yaw),
    //                               yc - radius * std::sin(armor_yaw), za);

    // // 世界系 -> 相机系
    // // p_cam = R_wc^T * (p_world - t_wc)
    // const Eigen::Vector3d p_cam =
    //     camera_to_world_rot_.transpose() * (p_world - camera_origin_world_);

    // const double dist = std::max(1e-6, p_cam.norm());

    // // ypd 各向异性标准差
    // const double sigma_yaw = r_ypd_yaw_std_;
    // const double sigma_pitch = r_ypd_pitch_std_;
    // const double sigma_distance = r_ypd_distance_std_scale_ * dist * dist;

    // // ypd 对角协方差
    // Eigen::Matrix3d cov_ypd = Eigen::Matrix3d::Zero();
    // cov_ypd(0, 0) = sigma_yaw * sigma_yaw;
    // cov_ypd(1, 1) = sigma_pitch * sigma_pitch;
    // cov_ypd(2, 2) = sigma_distance * sigma_distance;

    // // ypd -> camera xyz
    // const Eigen::Matrix3d j_ypd_to_cam = BuildJacobianYpdToCameraXyz(p_cam);
    // const Eigen::Matrix3d cov_cam = j_ypd_to_cam * cov_ypd * j_ypd_to_cam.transpose();

    // // camera xyz -> world xyz
    // const Eigen::Matrix3d cov_world =
    //     camera_to_world_rot_ * cov_cam * camera_to_world_rot_.transpose();

    // r.block<3, 3>(0, 0) = cov_world;

    // // 第4维仍是装甲板朝向 yaw 的测量噪声
    // r(3, 3) = r_armor_yaw_std_ * r_armor_yaw_std_;

    // return r;
  };
  // P - error estimate covariance matrix
  // Eigen::DiagonalMatrix<double, 9> p0;
  // p0.setIdentity();
  Eigen::MatrixXd p0 = Eigen::MatrixXd::Zero(9, 9);
  p0(0, 0) = 0.05;  // xc
  p0(1, 1) = 1.0;   // v_xc
  p0(2, 2) = 0.05;  // yc
  p0(3, 3) = 1.0;   // v_yc
  p0(4, 4) = 0.05;  // za
  p0(5, 5) = 1.0;   // v_za
  p0(6, 6) = 0.1;   // yaw
  p0(7, 7) = 2.0;   // v_yaw
  p0(8, 8) = 1.0;   // r
  // outpost 的 EKF 参数
  auto switch_q = [this](bool flag)
  {
    s2_q_x_ = flag ? s2qxyz_outpost_ : s2_q_x_armor_;
    s2_q_y_ = flag ? s2qxyz_outpost_ : s2_q_y_armor_;
    s2_q_z_ = flag ? s2qxyz_outpost_ : s2_q_z_armor_;
    s2_q_yaw_ = flag ? s2qyaw_outpost_ : s2_q_yaw_armor_;
    s2_q_r_ = flag ? s2qr_outpost_ : s2_q_r_armor_;
  };

  tracker_->ekf = ExtendedKalmanFilter{f, h, j_f, j_h, u_q, u_r, p0};
  tracker_->switch_q_ = switch_q;

  // Subscriber with tf2 message_filter
  tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
      this->get_node_base_interface(), this->get_node_timers_interface());
  tf2_buffer_->setCreateTimerInterface(timer_interface);
  tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

  armors_sub_.subscribe(this, "/detector/armors", rmw_qos_profile_sensor_data);
  target_frame_ = this->declare_parameter("target_frame", "odom");
  armors_filter_ = std::make_shared<armors_tf2_filter>(
      armors_sub_, *tf2_buffer_, target_frame_, 10, this->get_node_logging_interface(),
      this->get_node_clock_interface(), std::chrono::duration<int>(1));

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
  max_armor_distance_ = this->declare_parameter("max_armor_distance", 10.0);

  auto robot_type = this->declare_parameter<std::string>("robot_type", "default");
  is_hero_ = (robot_type == "hero");

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

  r_xyz_base_ = this->declare_parameter("ekf.r_xyz_base", 0.010);
  r_xyz_dist_gain_ = this->declare_parameter("ekf.r_xyz_dist_gain", 0.003);
  r_xyz_oblique_gain_ = this->declare_parameter("ekf.r_xyz_oblique_gain", 0.010);

  r_z_scale_ = this->declare_parameter("ekf.r_z_scale", 1.30);

  r_yaw_base_ = this->declare_parameter("ekf.r_yaw_base", 0.050);
  r_yaw_dist_gain_ = this->declare_parameter("ekf.r_yaw_dist_gain", 0.030);
  r_yaw_oblique_gain_ = this->declare_parameter("ekf.r_yaw_oblique_gain", 0.150);

  r_yaw_outpost_scale_ = this->declare_parameter("ekf.r_yaw_outpost_scale", 0.7);

  q_boost_xy_ = this->declare_parameter("ekf.q_boost_xy", 4.0);
  q_boost_z_ = this->declare_parameter("ekf.q_boost_z", 1.0);
  q_boost_yaw_ = this->declare_parameter("ekf.q_boost_yaw", 3.0);

  // ypd
  r_ypd_yaw_std_ = this->declare_parameter("ekf.r_ypd_yaw_std", 0.008);
  r_ypd_pitch_std_ = this->declare_parameter("ekf.r_ypd_pitch_std", 0.010);
  r_ypd_distance_std_scale_ =
      this->declare_parameter("ekf.r_ypd_distance_std_scale", 0.010);
  r_armor_yaw_std_ = this->declare_parameter("ekf.r_armor_yaw_std", 0.10);
}

void ArmorTrackerNode::ArmorsCallback(
    const auto_aim_interfaces::msg::Armors::SharedPtr& armors_msg)
{
  // try
  // {
  //   const auto tf = tf2_buffer_->lookupTransform(
  //       target_frame_, armors_msg->header.frame_id, armors_msg->header.stamp);

  //   const auto& t = tf.transform.translation;
  //   const auto& q = tf.transform.rotation;

  //   camera_origin_world_ = Eigen::Vector3d(t.x, t.y, t.z);
  //   camera_to_world_rot_ =
  //       Eigen::Quaterniond(q.w, q.x, q.y, q.z).normalized().toRotationMatrix();
  // }
  // catch (const tf2::TransformException& ex)
  // {
  //   RCLCPP_ERROR(this->get_logger(), "Failed to query camera pose: %s", ex.what());
  //   return;
  // }

  if (is_hero_)
  {
    if (armors_msg->header.frame_id != last_camera_frame_id_)
    {
      last_camera_frame_id_ = armors_msg->header.frame_id;
      RCLCPP_INFO(this->get_logger(), "Switched trajectory table due to new frame id: %s",
                  last_camera_frame_id_.c_str());
    }
  }

  // Transform armor position from image frame to world coordinate
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
            return std::fabs(armor.pose.position.z) > 5 ||
                   Eigen::Vector2d(armor.pose.position.x, armor.pose.position.y).norm() >
                       max_armor_distance_;
          }),
      armors_msg->armors.end());

  // Init message
  auto_aim_interfaces::msg::TrackerInfo info_msg;
  auto_aim_interfaces::msg::Target target_msg;
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
    dt_ = (time - last_time_).seconds();
    dt_ = std::clamp(dt_, 1e-3, 0.1);
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
      signed char num = 0;
      if (tracker_->tracked_armor.number == "outpost")
      {
        num = 10;
      }
      else if (tracker_->tracked_armor.number == "guard")
      {
        num = 7;
      }
      else if (tracker_->tracked_armor.number == "base")
      {
        num = 11;
      }
      else if (!tracker_->tracked_armor.number.empty())
      {
        num = static_cast<signed char>(std::stoi(tracker_->tracked_armor.number));
      }
      target_msg.num = num;
    }
  }
  target_msg.state = static_cast<int8_t>(tracker_->tracker_state);

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

Eigen::Matrix3d ArmorTrackerNode::BuildJacobianYpdToCameraXyz(
    const Eigen::Vector3d& p_cam)
{
  const double X = p_cam.x();
  const double Y = p_cam.y();
  const double Z = p_cam.z();

  const double D = std::max(1e-6, p_cam.norm());
  const double RHO = std::max(1e-6, std::hypot(X, Z));

  const double YAW = std::atan2(X, Z);
  const double PITCH = std::atan2(Y, RHO);

  const double CY = std::cos(YAW);
  const double SY = std::sin(YAW);
  const double CP = std::cos(PITCH);
  const double SP = std::sin(PITCH);

  // x = d * cos(pitch) * sin(yaw)
  // y = d * sin(pitch)
  // z = d * cos(pitch) * cos(yaw)
  Eigen::Matrix3d j;
  j << D * CP * CY, -D * SP * SY, CP * SY, 0.0, D * CP, SP, -D * CP * SY, -D * SP * CY,
      CP * CY;
  return j;
}

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorTrackerNode)
