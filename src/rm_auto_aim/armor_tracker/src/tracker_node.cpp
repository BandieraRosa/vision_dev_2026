#include "armor_tracker/tracker_node.hpp"

#include <cmath>
#include <memory>
#include <rclcpp/logging.hpp>
#include <vector>

namespace rm_auto_aim
{
ArmorTrackerNode::ArmorTrackerNode(const rclcpp::NodeOptions& options)
    : Node("armor_tracker", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting TrackerNode!");

  // Maximum allowable armor distance in the XOY plane
  max_armor_distance_ = this->declare_parameter("max_armor_distance", 10.0);

  // Tracker
  double max_match_distance = this->declare_parameter("tracker.max_match_distance", 0.15);
  double max_match_yaw_diff = this->declare_parameter("tracker.max_match_yaw_diff", 1.0);
  tracker_ = std::make_unique<Tracker>(max_match_distance, max_match_yaw_diff);
  tracker_->tracking_thres =
      static_cast<int>(this->declare_parameter("tracker.tracking_thres", 5));
  lost_time_thres_ = this->declare_parameter("tracker.lost_time_thres", 0.3);

  float k = static_cast<float>(this->declare_parameter("tracker.k", 0.092));
  int bias_time = static_cast<int>(this->declare_parameter("tracker.bias_time", 100));
  float s_bias = static_cast<float>(this->declare_parameter("tracker.s_bias", 0.19133));
  float z_bias = static_cast<float>(this->declare_parameter("tracker.z_bias", 0.21265));
  bool use_table = this->declare_parameter("tracker.calculate_mode", true);

  double max_x = this->declare_parameter("tracker.table.max_x", 13.0);
  double min_x = this->declare_parameter("tracker.table.min_x", 0.0);
  double max_y = this->declare_parameter("tracker.table.max_y", 2.0);
  double min_y = this->declare_parameter("tracker.table.min_y", -1.0);
  double resolution = this->declare_parameter("tracker.table.resolution", 0.01);
  std::string table_filename =
      this->declare_parameter("tracker.table.filename", "table.bin");
  SolveTrajectory::CalculateMode calculate_mode{};
  if (use_table)
  {
    calculate_mode = SolveTrajectory::CalculateMode::TABLE_LOOKUP;
  }
  else
  {
    calculate_mode = SolveTrajectory::CalculateMode::NORMAL;
  }
  TrajectoryTable::TableConfig table_config = {max_x, min_x,      max_y,
                                               min_y, resolution, table_filename};
  solver_ = std::make_unique<SolveTrajectory>(k, bias_time, s_bias, z_bias,
                                              calculate_mode, table_config);

  // ---------------- EKF 设置 (CV模型) ----------------
  // 状态 x = [xc, vxc, yc, vyc, za, vza, yaw, vyaw, r] (9维)
  // 观测 z = [xa, ya, za, yaw] (4维)

  // f - Process function 过程函数对状态进行更新 (CV模型)
  auto f = [this](const Eigen::VectorXd& x)
  {
    Eigen::VectorXd x_new = x;
    double t = dt_;
    // CV模型：位置 += 速度 * 时间
    x_new(EKF::STATE::X_CENTER) += x(EKF::STATE::V_X_CENTER) * t;
    x_new(EKF::STATE::Y_CENTER) += x(EKF::STATE::V_Y_CENTER) * t;
    x_new(EKF::STATE::Z_ARMOR) += x(EKF::STATE::V_Z_ARMOR) * t;
    x_new(EKF::STATE::YAW) += x(EKF::STATE::V_YAW) * t;
    // 速度保持不变（CV模型假设）
    return x_new;
  };

  // J_f - Jacobian of process function (CV模型：9x9矩阵)
  auto j_f = [this](const Eigen::VectorXd&)
  {
    Eigen::MatrixXd f(9, 9);
    f.setIdentity();
    double t = dt_;
    // CV模型雅可比矩阵
    // [xc, vxc, yc, vyc, za, vza, yaw, vyaw, r]
    //   0,   1,  2,   3,  4,   5,   6,    7, 8
    f(EKF::STATE::X_CENTER, EKF::STATE::V_X_CENTER) = t;
    f(EKF::STATE::Y_CENTER, EKF::STATE::V_Y_CENTER) = t;
    f(EKF::STATE::Z_ARMOR, EKF::STATE::V_Z_ARMOR) = t;
    f(EKF::STATE::YAW, EKF::STATE::V_YAW) = t;
    return f;
  };

  // h - Observation function 观测函数对状态进行测量
  auto h = [](const Eigen::VectorXd& x)
  {
    Eigen::VectorXd z(4);
    double xc = x(EKF::STATE::X_CENTER), yc = x(EKF::STATE::Y_CENTER),
           yaw = x(EKF::STATE::YAW), r = x(EKF::STATE::ROBOT_R);
    z(0) = xc - r * std::cos(yaw);  // xa
    z(1) = yc - r * std::sin(yaw);  // ya
    z(2) = x(EKF::STATE::Z_ARMOR);  // za
    z(3) = x(EKF::STATE::YAW);      // yaw
    return z;
  };

  // J_h - Jacobian of observation function (CV模型：4x9矩阵)
  auto j_h = [](const Eigen::VectorXd& x)
  {
    Eigen::MatrixXd h(4, 9);
    double yaw = x(EKF::STATE::YAW), r = x(EKF::STATE::ROBOT_R);
    // 状态: [xc, vxc, yc, vyc, za, vza, yaw, vyaw, r]
    // 索引:   0,   1,  2,   3,  4,   5,   6,    7, 8
    h.setZero();
    // d(xa)/d(xc) = 1
    h(0, EKF::STATE::X_CENTER) = 1;
    // d(xa)/d(yaw) = r * sin(yaw)
    h(0, EKF::STATE::YAW) = r * std::sin(yaw);
    // d(xa)/d(r) = -cos(yaw)
    h(0, EKF::STATE::ROBOT_R) = -std::cos(yaw);

    // d(ya)/d(yc) = 1
    h(1, EKF::STATE::Y_CENTER) = 1;
    // d(ya)/d(yaw) = -r * cos(yaw)
    h(1, EKF::STATE::YAW) = -r * std::cos(yaw);
    // d(ya)/d(r) = -sin(yaw)
    h(1, EKF::STATE::ROBOT_R) = -std::sin(yaw);

    // d(za)/d(za) = 1
    h(2, EKF::STATE::Z_ARMOR) = 1;

    // d(yaw)/d(yaw) = 1
    h(3, EKF::STATE::YAW) = 1;

    return h;
  };

  // update_Q - process noise covariance matrix 过程噪声协方差矩阵 (CV模型)
  s2qxyz_ = declare_parameter("ekf.sigma2_q_xyz", 20.0);
  s2qyaw_ = declare_parameter("ekf.sigma2_q_yaw", 100.0);
  s2qr_ = declare_parameter("ekf.sigma2_q_r", 800.0);
  auto u_q = [this]()
  {
    Eigen::MatrixXd q(9, 9);
    q.setZero();
    double t = dt_, x = s2qxyz_, y = s2qyaw_, r = s2qr_;

    double t2 = t * t, t3 = t2 * t, t4 = t3 * t;

    // CV模型的过程噪声矩阵
    // 对于每个位置-速度对，使用离散白噪声加速度模型：
    // Q = [[t^4/4, t^3/2], [t^3/2, t^2]] * sigma^2

    // XYZ 块的元素 (p-位置, v-速度)
    double q_p_p_xyz = t4 / 4.0 * x;
    double q_p_v_xyz = t3 / 2.0 * x;
    double q_v_v_xyz = t2 * x;

    // Yaw 块的元素
    double q_p_p_yaw = t4 / 4.0 * y;
    double q_p_v_yaw = t3 / 2.0 * y;
    double q_v_v_yaw = t2 * y;

    double q_r = t2 * r;

    // [xc, vxc]
    q(EKF::STATE::X_CENTER, EKF::STATE::X_CENTER) = q_p_p_xyz;
    q(EKF::STATE::X_CENTER, EKF::STATE::V_X_CENTER) = q_p_v_xyz;
    q(EKF::STATE::V_X_CENTER, EKF::STATE::X_CENTER) = q_p_v_xyz;
    q(EKF::STATE::V_X_CENTER, EKF::STATE::V_X_CENTER) = q_v_v_xyz;

    // [yc, vyc]
    q(EKF::STATE::Y_CENTER, EKF::STATE::Y_CENTER) = q_p_p_xyz;
    q(EKF::STATE::Y_CENTER, EKF::STATE::V_Y_CENTER) = q_p_v_xyz;
    q(EKF::STATE::V_Y_CENTER, EKF::STATE::Y_CENTER) = q_p_v_xyz;
    q(EKF::STATE::V_Y_CENTER, EKF::STATE::V_Y_CENTER) = q_v_v_xyz;

    // [za, vza]
    q(EKF::STATE::Z_ARMOR, EKF::STATE::Z_ARMOR) = q_p_p_xyz;
    q(EKF::STATE::Z_ARMOR, EKF::STATE::V_Z_ARMOR) = q_p_v_xyz;
    q(EKF::STATE::V_Z_ARMOR, EKF::STATE::Z_ARMOR) = q_p_v_xyz;
    q(EKF::STATE::V_Z_ARMOR, EKF::STATE::V_Z_ARMOR) = q_v_v_xyz;

    // [yaw, vyaw]
    q(EKF::STATE::YAW, EKF::STATE::YAW) = q_p_p_yaw;
    q(EKF::STATE::YAW, EKF::STATE::V_YAW) = q_p_v_yaw;
    q(EKF::STATE::V_YAW, EKF::STATE::YAW) = q_p_v_yaw;
    q(EKF::STATE::V_YAW, EKF::STATE::V_YAW) = q_v_v_yaw;

    // R
    q(EKF::STATE::ROBOT_R, EKF::STATE::ROBOT_R) = q_r;

    return q;
  };

  // update_R - measurement noise covariance matrix 观测噪声协方差矩阵
  r_xyz_factor_ = declare_parameter("ekf.r_xyz_factor", 0.05);
  r_yaw_ = declare_parameter("ekf.r_yaw", 0.02);
  auto u_r = [this](const Eigen::VectorXd& z)
  {
    Eigen::DiagonalMatrix<double, 4> r;
    double dist = z.head<3>().norm();
    double std_dev_xyz = r_xyz_factor_ * dist;  // 标准差
    r.diagonal() << std_dev_xyz * std_dev_xyz, std_dev_xyz * std_dev_xyz,
        std_dev_xyz * std_dev_xyz, r_yaw_;
    return r;
  };

  // P - error estimate covariance matrix (CV模型：9x9)
  Eigen::DiagonalMatrix<double, 9> p0;
  p0.setIdentity();
  tracker_->ekf_ = EKF(f, h, j_f, j_h, u_q, u_r, p0);

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

  velocity_sub_ = this->create_subscription<auto_aim_interfaces::msg::Velocity>(
      "/current_velocity",
      rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data)),
      [this](const auto_aim_interfaces::msg::Velocity::SharedPtr velocity_msg)
      { solver_->Init(velocity_msg); });

  // Measurement publisher (for debug usage)
  info_pub_ =
      this->create_publisher<auto_aim_interfaces::msg::TrackerInfo>("/tracker/info", 10);

  // Publisher
  target_pub_ = this->create_publisher<auto_aim_interfaces::msg::Target>(
      "/tracker/target", rclcpp::SensorDataQoS());

  send_pub_ = this->create_publisher<auto_aim_interfaces::msg::Send>(
      "/tracker/send", rclcpp::SensorDataQoS());

  // Visualization Marker Publisher
  // See http://wiki.ros.org/rviz/DisplayTypes/Marker
  position_marker_.ns = "position";
  position_marker_.type = visualization_msgs::msg::Marker::SPHERE;
  position_marker_.scale.x = position_marker_.scale.y = position_marker_.scale.z = 0.1;
  position_marker_.color.a = 1.0;
  position_marker_.color.g = 1.0;
  linear_v_marker_.type = visualization_msgs::msg::Marker::ARROW;
  linear_v_marker_.ns = "linear_v";
  linear_v_marker_.scale.x = 0.03;
  linear_v_marker_.scale.y = 0.05;
  linear_v_marker_.color.a = 1.0;
  linear_v_marker_.color.r = 1.0;
  linear_v_marker_.color.g = 1.0;
  angular_v_marker_.type = visualization_msgs::msg::Marker::ARROW;
  angular_v_marker_.ns = "angular_v";
  angular_v_marker_.scale.x = 0.03;
  angular_v_marker_.scale.y = 0.05;
  angular_v_marker_.color.a = 1.0;
  angular_v_marker_.color.b = 1.0;
  angular_v_marker_.color.g = 1.0;
  armor_marker_.ns = "armors";
  armor_marker_.type = visualization_msgs::msg::Marker::CUBE;
  armor_marker_.scale.x = 0.03;
  armor_marker_.scale.z = 0.125;
  armor_marker_.color.a = 1.0;
  armor_marker_.color.r = 1.0;
  aiming_point_marker_.header.frame_id = "gimbal_odom";
  aiming_point_marker_.ns = "aiming_point";
  aiming_point_marker_.type = visualization_msgs::msg::Marker::SPHERE;
  aiming_point_marker_.action = visualization_msgs::msg::Marker::ADD;
  aiming_point_marker_.scale.x = aiming_point_marker_.scale.y =
      aiming_point_marker_.scale.z = 0.12;
  aiming_point_marker_.color.r = 1.0;
  aiming_point_marker_.color.g = 1.0;
  aiming_point_marker_.color.b = 1.0;
  aiming_point_marker_.color.a = 1.0;
  aiming_point_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);

  marker_pub_ =
      this->create_publisher<visualization_msgs::msg::MarkerArray>("/tracker/marker", 10);
}

void ArmorTrackerNode::ArmorsCallback(
    const auto_aim_interfaces::msg::Armors::SharedPtr armors_msg)
{
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
            return abs(armor.pose.position.z) > 1.2 ||
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

  // Update tracker
  if (tracker_->tracker_state == Tracker::LOST)
  {
    tracker_->Init(armors_msg);
    // solver_->rebuild();
    target_msg.tracking = false;
  }
  else
  {
    // 求时间差，ema 滤波
    double raw_dt = (time - last_time_).seconds();
    if (raw_dt <= 0.0 || raw_dt > 0.1)
    {
      raw_dt = 0.01;
    }

    static bool dt_inited = false;
    static double dt_ema = 1.0 / 100.0;
    const double ALPHA = 0.9;

    if (!dt_inited)
    {
      dt_ema = raw_dt;
      dt_inited = true;
    }
    else
    {
      dt_ema = ALPHA * dt_ema + (1.0 - ALPHA) * raw_dt;
    }
    dt_ = dt_ema;

    tracker_->lost_thres = std::max(static_cast<int>(lost_time_thres_ / dt_), 1);
    tracker_->Update(armors_msg);

    // Publish Info
    info_msg.position_diff = tracker_->info_position_diff;
    info_msg.yaw_diff = tracker_->info_yaw_diff;
    info_msg.position.x = tracker_->measurement(0);
    info_msg.position.y = tracker_->measurement(1);
    info_msg.position.z = tracker_->measurement(2);
    info_msg.yaw = tracker_->measurement(3);
    info_pub_->publish(info_msg);

    if (tracker_->tracker_state == Tracker::DETECTING)
    {
      target_msg.tracking = false;
    }
    else if (tracker_->tracker_state == Tracker::TRACKING ||
             tracker_->tracker_state == Tracker::TEMP_LOST)
    {
      target_msg.tracking = true;
      // Fill target message (CV模型：9维状态)
      const auto& state = tracker_->target_state;
      target_msg.id = tracker_->tracked_id;
      target_msg.armors_num = static_cast<int>(tracker_->tracked_armors_num);
      target_msg.position.x = state(EKF::STATE::X_CENTER);
      target_msg.position.y = state(EKF::STATE::Y_CENTER);
      target_msg.position.z = state(EKF::STATE::Z_ARMOR);
      target_msg.velocity.x = state(EKF::STATE::V_X_CENTER);
      target_msg.velocity.y = state(EKF::STATE::V_Y_CENTER);
      target_msg.velocity.z = state(EKF::STATE::V_Z_ARMOR);
      target_msg.yaw = state(EKF::STATE::YAW);
      target_msg.v_yaw = state(EKF::STATE::V_YAW);
      target_msg.radius_1 = state(EKF::STATE::ROBOT_R);
      target_msg.radius_2 = tracker_->another_r;
      target_msg.dz = tracker_->dz;

      // 获取当前相机的 yaw 与 tracking 目标的 x
      // float camera_yaw = 0.0f;

      // auto transform_stamped = tf2_buffer_->lookupTransform(
      //     "odom", "camera_optical_frame", tf2::TimePointZero);

      // // 从变换中提取四元数并转换为欧拉角
      // tf2::Quaternion q(
      //     transform_stamped.transform.rotation.x,
      //     transform_stamped.transform.rotation.y,
      //     transform_stamped.transform.rotation.z,
      //     transform_stamped.transform.rotation.w);

      // double camera_roll{}, camera_pitch{}, camera_yaw_tf{};
      // tf2::Matrix3x3(q).getRPY(camera_roll, camera_pitch, camera_yaw_tf);
      // try
      // {
      //   camera_yaw = static_cast<float>(camera_yaw_tf);
      // }
      // catch (const tf2::TransformException& ex)
      // {
      //   RCLCPP_WARN(this->get_logger(), "Could not get camera transform: %s",
      //   ex.what()); camera_yaw = 0.0f;
      // }
      // target_msg.camera_yaw = camera_yaw;
      // target_msg.armor_x = tracker_->tracked_armor.pose.position.x;

      float pitch = 0, yaw = 0, aim_x = 0, aim_y = 0, aim_z = 0;
      auto msg = std::make_shared<auto_aim_interfaces::msg::Target>(target_msg);

      bool is_fire = false;
      solver_->AutoSolveTrajectory(pitch, yaw, aim_x, aim_y, aim_z, msg, is_fire);

      if (abs(aim_x) > 0.01)
      {
        target_msg.aiming_point.x = aim_x;
        target_msg.aiming_point.y = aim_y;
        target_msg.aiming_point.z = aim_z;
      }

      solver_->SetFireCallback([&](bool is_fire) { send_msg.is_fire = is_fire; });
      if (pitch == NAN || yaw == NAN)
      {
        RCLCPP_ERROR(this->get_logger(), "pitch or yaw is NAN!");
      }

      send_msg.is_fire = is_fire;
      send_msg.pitch = pitch;
      send_msg.yaw = yaw;
    }
  }

  last_time_ = time;

  // RCLCPP_INFO(this->get_logger(), "Target Euler: pitch %.2f yaw %.2f", send_msg.pitch,
  //             send_msg.yaw);
  if (send_msg.pitch == NAN || send_msg.yaw == NAN)
  {
    RCLCPP_ERROR(this->get_logger(), "target pitch or yaw is NAN!");
    send_msg.pitch = 0.0;
    send_msg.yaw = 0.0;
  }

  send_pub_->publish(send_msg);
  target_pub_->publish(target_msg);

  PublishMarkers(target_msg);
}

void ArmorTrackerNode::PublishMarkers(const auto_aim_interfaces::msg::Target& target_msg)
{
  position_marker_.header = target_msg.header;
  linear_v_marker_.header = target_msg.header;
  angular_v_marker_.header = target_msg.header;
  armor_marker_.header = target_msg.header;

  visualization_msgs::msg::MarkerArray marker_array;

  if (target_msg.tracking)
  {
    double yaw = target_msg.yaw, r1 = target_msg.radius_1, r2 = target_msg.radius_2;
    double xc = target_msg.position.x, yc = target_msg.position.y,
           za = target_msg.position.z;
    double vx = target_msg.velocity.x, vy = target_msg.velocity.y,
           vz = target_msg.velocity.z;
    double dz = target_msg.dz;

    position_marker_.action = visualization_msgs::msg::Marker::ADD;
    position_marker_.pose.position.x = xc;
    position_marker_.pose.position.y = yc;
    position_marker_.pose.position.z = za + dz / 2;

    linear_v_marker_.action = visualization_msgs::msg::Marker::ADD;
    linear_v_marker_.points.clear();
    linear_v_marker_.points.emplace_back(position_marker_.pose.position);
    geometry_msgs::msg::Point arrow_end = position_marker_.pose.position;
    arrow_end.x += vx;
    arrow_end.y += vy;
    arrow_end.z += vz;
    linear_v_marker_.points.emplace_back(arrow_end);

    angular_v_marker_.action = visualization_msgs::msg::Marker::ADD;
    angular_v_marker_.points.clear();
    angular_v_marker_.points.emplace_back(position_marker_.pose.position);
    arrow_end = position_marker_.pose.position;
    arrow_end.z += target_msg.v_yaw / M_PI;
    angular_v_marker_.points.emplace_back(arrow_end);

    armor_marker_.action = visualization_msgs::msg::Marker::ADD;
    armor_marker_.scale.y = tracker_->tracked_armor.type == "small" ? 0.135 : 0.23;
    bool is_current_pair = true;
    size_t a_n = target_msg.armors_num;
    geometry_msgs::msg::Point p_a;
    double r = 0;
    for (size_t i = 0; i < a_n; i++)
    {
      double tmp_yaw =
          yaw + static_cast<double>(i) * (2 * M_PI / static_cast<double>(a_n));
      // Only 4 armors has 2 radius and height
      if (a_n == 4)
      {
        r = is_current_pair ? r1 : r2;
        p_a.z = za + (is_current_pair ? 0 : dz);
        is_current_pair = !is_current_pair;
      }
      else
      {
        r = r1;
        p_a.z = za;
      }
      p_a.x = xc - r * cos(tmp_yaw);
      p_a.y = yc - r * sin(tmp_yaw);

      armor_marker_.id = static_cast<int>(i);
      armor_marker_.pose.position = p_a;
      tf2::Quaternion q;
      q.setRPY(0, target_msg.id == "outpost" ? -0.26 : 0.26, tmp_yaw);
      armor_marker_.pose.orientation = tf2::toMsg(q);
      marker_array.markers.emplace_back(armor_marker_);
    }

    if (abs(target_msg.aiming_point.x) > 0.01)
    {
      aiming_point_marker_.action = visualization_msgs::msg::Marker::ADD;
      aiming_point_marker_.header = target_msg.header;
      aiming_point_marker_.pose.position = target_msg.aiming_point;
      marker_array.markers.emplace_back(aiming_point_marker_);
    }
  }
  else
  {
    position_marker_.action = visualization_msgs::msg::Marker::DELETE;
    linear_v_marker_.action = visualization_msgs::msg::Marker::DELETE;
    angular_v_marker_.action = visualization_msgs::msg::Marker::DELETE;
    aiming_point_marker_.action = visualization_msgs::msg::Marker::DELETE;
    armor_marker_.action = visualization_msgs::msg::Marker::DELETE;
    marker_array.markers.emplace_back(armor_marker_);
  }

  marker_array.markers.emplace_back(position_marker_);
  marker_array.markers.emplace_back(linear_v_marker_);
  marker_array.markers.emplace_back(angular_v_marker_);
  marker_pub_->publish(marker_array);
}

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its
// library is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorTrackerNode)
