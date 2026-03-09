#include "armor_tracker/tracker_node.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cmath>

#include "armor_tracker/SolveTrajectory.hpp"

namespace rm_auto_aim
{
ArmorTrackerNode::ArmorTrackerNode(const rclcpp::NodeOptions& options)
    : Node("armor_tracker", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting TrackerNode!");

  // Maximum allowable armor distance in the XOY plane
  max_armor_distance_ = this->declare_parameter("max_armor_distance", 10.0);

  auto robot_type = this->declare_parameter<std::string>("robot_type", "default");
  is_hero_ = (robot_type == "hero");

  // Tracker
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

  float k = static_cast<float>(this->declare_parameter("tracker.k", 0.092));
  float bias_time =
      static_cast<float>(this->declare_parameter("tracker.bias_time", 0.01));
  float s_bias = static_cast<float>(this->declare_parameter("tracker.s_bias", 0.19133));
  float z_bias = static_cast<float>(this->declare_parameter("tracker.z_bias", 0.21265));
  float pitch_bias =
      static_cast<float>(this->declare_parameter("tracker.pitch_bias", 0.0));

  bool use_table = this->declare_parameter("tracker.calculate_mode", true);

  double max_x = this->declare_parameter("tracker.table.max_x", 13.0);
  double min_x = this->declare_parameter("tracker.table.min_x", 0.0);
  double max_y = this->declare_parameter("tracker.table.max_y", 2.0);
  double min_y = this->declare_parameter("tracker.table.min_y", -1.0);
  double resolution = this->declare_parameter("tracker.table.resolution", 0.01);
  std::string table_filename =
      this->declare_parameter("tracker.table.filename", "table.bin");
  std::string package_prefix =
      ament_index_cpp::get_package_share_directory("armor_tracker") + "/tools/";
  table_filename_normal_ = package_prefix + table_filename;
  RCLCPP_ERROR(this->get_logger(), "table_filename_normal_: %s",
               table_filename_normal_.c_str());
  if (is_hero_)
  {
    table_filename_lob_ =
        package_prefix + this->declare_parameter("tracker.table.filename_lob", "");
  }
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
                                               min_y, resolution, table_filename_normal_};
  gaf_solver_ = std::make_unique<SolveTrajectory>(
      k, bias_time, s_bias, z_bias, pitch_bias, calculate_mode, table_config);

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
    // clang-format off 临时禁用格式化工具，确保矩阵按指定格式排列
    f << 1, dt_, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, dt_, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, dt_, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, dt_, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1;
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
  s2qxyz_armor_ = declare_parameter("ekf.sigma2_q_xyz", 20.0);
  s2qyaw_armor_ = declare_parameter("ekf.sigma2_q_yaw", 100.0);
  s2qr_armor_ = declare_parameter("ekf.sigma2_q_r", 800.0);
  s2qxyz_outpost_ = declare_parameter("ekf.sigma2_q_xyz_outpost", 0.005);
  s2qyaw_outpost_ = declare_parameter("ekf.sigma2_q_yaw_outpost", 2.0);
  s2qr_outpost_ = declare_parameter("ekf.sigma2_q_r_outpost", 0.0);
  s2qxyz_ = s2qxyz_armor_;
  s2qyaw_ = s2qyaw_armor_;
  s2qr_ = s2qr_armor_;
  auto u_q = [this]()
  {
    Eigen::MatrixXd q(9, 9);
    double t = dt_, x = s2qxyz_, y = s2qyaw_, r = s2qr_;
    double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx = pow(t, 2) * x;
    double q_y_y = pow(t, 4) / 4 * y, q_y_vy = pow(t, 3) / 2 * y, q_vy_vy = pow(t, 2) * y;
    double q_r = pow(t, 4) / 4 * r;
    // clang-format off
    //    xc      v_xc    yc      v_yc    za      v_za    yaw     v_yaw   r
    q <<  q_x_x,  q_x_vx, 0,      0,      0,      0,      0,      0,      0,
          q_x_vx, q_vx_vx,0,      0,      0,      0,      0,      0,      0,
          0,      0,      q_x_x,  q_x_vx, 0,      0,      0,      0,      0,
          0,      0,      q_x_vx, q_vx_vx,0,      0,      0,      0,      0,
          0,      0,      0,      0,      q_x_x,  q_x_vx, 0,      0,      0,
          0,      0,      0,      0,      q_x_vx, q_vx_vx,0,      0,      0,
          0,      0,      0,      0,      0,      0,      q_y_y,  q_y_vy, 0,
          0,      0,      0,      0,      0,      0,      q_y_vy, q_vy_vy,0,
          0,      0,      0,      0,      0,      0,      0,      0,      q_r;
    // clang-format on
    return q;
  };
  // update_R - measurement noise covariance matrix 观测噪声协方差矩阵
  r_xyz_factor_ = declare_parameter("ekf.r_xyz_factor", 0.05);
  r_yaw_ = declare_parameter("ekf.r_yaw", 0.02);
  // todo: dynamic R
  [[maybe_unused]] double center_yaw = std::atan2(
      tracker_->tracked_armor.pose.position.y, tracker_->tracked_armor.pose.position.x);
  // ;double delta_yaw = ;
  auto u_r = [this](const Eigen::VectorXd& z)
  {
    Eigen::DiagonalMatrix<double, 4> r;
    double x = r_xyz_factor_;
    r.diagonal() << abs(x * z[0]), abs(x * z[1]), abs(x * z[2]), r_yaw_;
    return r;
  };
  // P - error estimate covariance matrix
  Eigen::DiagonalMatrix<double, 9> p0;
  p0.setIdentity();

  // outpost的EKF参数
  auto switch_q = [this](bool flag)
  {
    s2qxyz_ = flag ? s2qxyz_outpost_ : s2qxyz_armor_;
    s2qyaw_ = flag ? s2qyaw_outpost_ : s2qyaw_armor_;
    s2qr_ = flag ? s2qr_outpost_ : s2qr_armor_;
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

  // velocity_sub_ = this->create_subscription<auto_aim_interfaces::msg::Velocity>(
  // "/current_velocity",
  // rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data)),
  // std::bind(&ArmorTrackerNode::velocityCallback, this, std::placeholders::_1));

  velocity_sub_ = this->create_subscription<auto_aim_interfaces::msg::Velocity>(
      "/current_velocity",
      rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data)),
      [this](const auto_aim_interfaces::msg::Velocity::SharedPtr velocity_msg)
      { gaf_solver_->Init(velocity_msg); });

  // Measurement publisher (for debug usage)
  info_pub_ =
      this->create_publisher<auto_aim_interfaces::msg::TrackerInfo>("/tracker/info", 10);

  armor_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
      "/tracker/armor_pose", rclcpp::SensorDataQoS());

  // Publisher
  target_pub_ = this->create_publisher<auto_aim_interfaces::msg::Target>(
      "/tracker/target", rclcpp::SensorDataQoS());

  send_pub_ = this->create_publisher<auto_aim_interfaces::msg::Send>(
      "/tracker/send", rclcpp::SensorDataQoS());

  outpost_idx_pub_ = this->create_publisher<std_msgs::msg::Int32>(
      "/tracker/outpost_idx", rclcpp::SensorDataQoS());

  gimbal_yaw_error_pub_ = this->create_publisher<std_msgs::msg::Float32>(
      "/tracker/gimbal_yaw_error", rclcpp::SensorDataQoS());

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
  marker_pub_ =
      this->create_publisher<visualization_msgs::msg::MarkerArray>("/tracker/marker", 10);

  if (is_hero_)
  {
    camera_switch_sub_ = this->create_subscription<std_msgs::msg::Bool>(
        "/camera_switch_done", rclcpp::QoS(1).reliable(),
        [this](const std_msgs::msg::Bool::SharedPtr msg)
        {
          lob_shot_flag_ = msg->data;
          RCLCPP_INFO(this->get_logger(), "Camera switch done, lob_shot_flag: %d",
                      lob_shot_flag_);
          const auto& target_filename =
              lob_shot_flag_ ? table_filename_lob_ : table_filename_normal_;
          if (!gaf_solver_->ReloadTable(target_filename))
          {
            RCLCPP_WARN(this->get_logger(),
                        "Failed to reload trajectory table: %s, "
                        "solver will use fallback mode.",
                        target_filename.c_str());
          }
        });
  }
}

// void ArmorTrackerNode::velocityCallback(const
// auto_aim_interfaces::msg::Velocity::SharedPtr velocity_msg)
// {
//   gaf_solver->init(velocity_msg);
// }

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

  // geometry_msgs::msg::PoseStamped armor_in_gimbal;
  // armor_in_gimbal.header.stamp = time;
  // armor_in_gimbal.header.frame_id = target_frame_;
  // armor_in_gimbal.pose = tracker_->tracked_armor.pose;
  // armor_pose_pub_->publish(armor_in_gimbal);

  // Update tracker
  double gimbal_yaw_error{0.0};
  double bc_yaw{0.0};
  if (tracker_->tracker_state == Tracker::State::LOST)
  {
    tracker_->Init(armors_msg);
    gaf_solver_->ReBuild();
    target_msg.tracking = false;
  }
  else
  {
    // 求时间差
    dt_ = (time - last_time_).seconds();
    tracker_->lost_thres = static_cast<int>(lost_time_thres_ / dt_);
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
      target_msg.id = tracker_->tracked_id;
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

      // 获取当前云台在世界系下的 yaw
      auto transform_stamped =
          tf2_buffer_->lookupTransform("gimbal_odom", "yaw_link", tf2::TimePointZero);

      // 从变换中提取四元数并转换为欧拉角
      tf2::Quaternion q(
          transform_stamped.transform.rotation.x, transform_stamped.transform.rotation.y,
          transform_stamped.transform.rotation.z, transform_stamped.transform.rotation.w);

      double gimbal_roll{}, gimbal_pitch{}, gimbal_yaw{};
      tf2::Matrix3x3(q).getRPY(gimbal_roll, gimbal_pitch, gimbal_yaw);
      target_msg.gimbal_yaw = gimbal_yaw;

      double pitch = 0, yaw = 0, aim_x = 0, aim_y = 0, aim_z = 0;
      int idx{};
      auto msg = std::make_shared<auto_aim_interfaces::msg::Target>(target_msg);

      bool is_fire = false;
      gaf_solver_->AutoSolveTrajectory(pitch, yaw, is_fire, aim_x, aim_y, aim_z, idx,
                                       msg);

      bc_yaw = yaw;
      if (std::fabs(msg->v_yaw) < 6.2f)
      {
        yaw += msg->v_yaw / 3 * 0.002f;
      }
      else
      {
        yaw += msg->v_yaw / std::fabs(msg->v_yaw) * 0.02f;
      }
      gimbal_yaw_error = std::fabs(static_cast<double>(gimbal_yaw - yaw));

      target_msg.aiming_point.x = aim_x;
      target_msg.aiming_point.y = aim_y;
      target_msg.aiming_point.z = aim_z;

      send_msg.is_fire = is_fire;
      send_msg.pitch = pitch;
      send_msg.yaw = yaw;
      send_msg.idx = idx;
    }
  }

  last_time_ = time;

  send_pub_->publish(send_msg);

  target_pub_->publish(target_msg);

  // Publish Info
  info_msg.position_diff = tracker_->info_position_diff;
  info_msg.yaw_diff = tracker_->info_yaw_diff;
  info_msg.bc_yaw = bc_yaw;
  info_msg.gimbal_yaw_error = gimbal_yaw_error;
  info_msg.position.x = tracker_->measurement(0);
  info_msg.position.y = tracker_->measurement(1);
  info_msg.position.z = tracker_->measurement(2);
  info_msg.yaw = tracker_->measurement(3);
  info_msg.outpost_idx = Tracker::outpost_idx;
  info_pub_->publish(info_msg);

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
    // double yaw = target_msg.yaw, r1 = target_msg.radius_1, r2 = target_msg.radius_2;
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
    // bool is_current_pair = true;
    size_t a_n = target_msg.armors_num;
    geometry_msgs::msg::Point p_a;
    // double r = 0;
    SolveTrajectory::TargetPostion center = gaf_solver_->SolveTrajectory::PredictCenter(
        std::make_shared<auto_aim_interfaces::msg::Target>(target_msg), 0);
    for (size_t i = 0; i < a_n; i++)
    {
      // double tmp_yaw =
      //     yaw + static_cast<double>(i) * (2 * M_PI / static_cast<double>(a_n));
      // // Only 4 armors has 2 radius and height
      // if (a_n == 4)
      // {
      //   r = is_current_pair ? r1 : r2;
      //   p_a.z = za + (is_current_pair ? 0 : dz);
      //   is_current_pair = !is_current_pair;
      // }
      // p_a.x = xc - r * cos(tmp_yaw);
      // p_a.y = yc - r * sin(tmp_yaw);

      SolveTrajectory::TargetPostion armor_position =
          gaf_solver_->SolveTrajectory::PredictArmor(
              std::make_shared<auto_aim_interfaces::msg::Target>(target_msg), 0, i,
              center);

      p_a.x = armor_position.x;
      p_a.y = armor_position.y;
      p_a.z = armor_position.z;

      armor_marker_.id = static_cast<int>(i);
      armor_marker_.pose.position = p_a;
      tf2::Quaternion q;
      q.setRPY(0, target_msg.id == "outpost" ? -0.26 : 0.26, armor_position.yaw);
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