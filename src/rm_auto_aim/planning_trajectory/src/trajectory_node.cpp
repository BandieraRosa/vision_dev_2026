#include "planning_trajectory/trajectory_node.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cmath>

#include "planning_trajectory/trajectory.hpp"
#include "planning_trajectory/trajectory_solver.hpp"

namespace rm_auto_aim
{
PlanningTrajectoryNode::PlanningTrajectoryNode(const rclcpp::NodeOptions& options)
    : Node("planning_trajectory", options)
{
  this->Init();
  RCLCPP_INFO(this->get_logger(), "Starting PlanningTrajectoryNode!");
}

void PlanningTrajectoryNode::TargetCallback(
    const auto_aim_interfaces::msg::Target::SharedPtr target_msg)
{
  std::lock_guard<std::mutex> lk(target_mutex_);

  if (target_msg->is_switchtable && !last_switchtable_)
  {
    trajectory_->SwitchTable();
  }
  last_switchtable_ = target_msg->is_switchtable;
  send_time_ = 0;
  tracking_ = target_msg->tracking;

  target_.position.x = target_msg->position.x;
  target_.position.y = target_msg->position.y;
  target_.position.z = target_msg->position.z;
  target_.position.yaw = target_msg->yaw;

  target_.velocity.x = target_msg->velocity.x;
  target_.velocity.y = target_msg->velocity.y;
  target_.velocity.z = target_msg->velocity.z;
  target_.velocity.yaw = target_msg->v_yaw;

  target_.num = target_msg->armors_num;
  target_.type = target_msg->type;
  target_.outpost_idx = target_msg->outpost_idx;

  target_.radius1 = target_msg->radius_1;
  target_.radius2 = target_msg->radius_2;

  target_.number = target_msg->num;

  // 发布 Target 消息

  if (!tracking_)
  {
    trajectory_->Reset();
  }
}

void PlanningTrajectoryNode::PublishStopCommand()
{
  auto_aim_interfaces::msg::Send send_msg;
  send_msg.is_fire = false;
  send_msg.pitch = 0.0;
  send_msg.yaw = 0.0;
  send_msg.vel_yaw = 0.0;
  send_msg.acc_yaw = 0.0;
  send_pub_->publish(send_msg);
}

void PlanningTrajectoryNode::timer_callback()
{
  auto start = std::chrono::high_resolution_clock::now();

  // 先把需要的状态在锁内拷贝出来，尽快释放锁
  bool tracking_local;
  TrajectorySolver::Target target_local;
  {
    std::lock_guard<std::mutex> lk(target_mutex_);
    tracking_local = tracking_;
    target_local = target_;
  }

  if (!tracking_local)
  {
    PublishStopCommand();
    return;
  }

  double gimbal_yaw = 0.0;
  double gimbal_pitch = 0.0;

  try
  {
    const auto gimbal_yaw_pitch = GetGimbalYawAndPitch();
    gimbal_yaw = gimbal_yaw_pitch.first;
    gimbal_pitch = gimbal_yaw_pitch.second;
  }
  catch (const tf2::TransformException& ex)
  {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                         "Get gimbal transform failed: %s", ex.what());
    PublishStopCommand();
    return;
  }

  double aim_x = 0.0, aim_y = 0.0, aim_z = 0.0;
  int idx = TrajectorySolver::LOST;
  TrajectorySolver::control cmd{};

  // 调用 solver 也要加锁，因为 trajectory_ 内部状态会被 Reset() / SwitchTable() 修改

  std::lock_guard<std::mutex> lk(target_mutex_);
  trajectory_->solver().AutoSolveTrajectory(cmd.pitch, cmd.yaw, cmd.is_fire, aim_x, aim_y,
                                            aim_z, idx, target_local, gimbal_yaw,
                                            gimbal_pitch, send_time_, gimbal_yaw_speed_);
  double bc_yaw = cmd.yaw;
  {
    send_time_ += dt_;
    trajectory_->UpdatePlanTrajectory(cmd, gimbal_yaw);
  }

  gimbal_yaw_speed_ = cmd.vel_yaw;
  // publish 放在锁外，避免阻塞 sub 线程
  auto_aim_interfaces::msg::Send send_msg;
  if (send_time_ >= 2 * dt_)
  {
    cmd.is_fire = false;
  }
  send_msg.is_fire = cmd.is_fire;
  send_msg.pitch = cmd.pitch;
  send_msg.yaw = cmd.yaw;
  send_msg.vel_yaw = cmd.vel_yaw;
  send_msg.acc_yaw = cmd.acc_yaw;
  send_msg.num = target_local.number;
  send_pub_->publish(send_msg);

  auto_aim_interfaces::msg::TrajectoryInfo info_msg;
  info_msg.aim_position.x = aim_x;
  info_msg.aim_position.y = aim_y;
  info_msg.aim_position.z = aim_z;
  info_msg.gimbal_yaw = gimbal_yaw;
  info_msg.gimbal_pitch = -gimbal_pitch;
  info_msg.idx = idx;
  info_msg.bc_yaw = bc_yaw;
  info_msg.bc_pitch = cmd.pitch;
  info_pub_->publish(info_msg);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  RCLCPP_DEBUG(this->get_logger(), "Trajectory time: %ld us", duration.count());
}

std::pair<double, double> PlanningTrajectoryNode::GetGimbalYawAndPitch()
{
  std::pair<double, double> gimbal_yaw_pitch{0.0, 0.0};

  const auto transform_stamped_yaw =
      tf2_buffer_->lookupTransform("gimbal_odom", "yaw_link", tf2::TimePointZero);
  const auto transform_stamped_pitch =
      tf2_buffer_->lookupTransform("gimbal_odom", "pitch_link", tf2::TimePointZero);

  tf2::Quaternion q_yaw(transform_stamped_yaw.transform.rotation.x,
                        transform_stamped_yaw.transform.rotation.y,
                        transform_stamped_yaw.transform.rotation.z,
                        transform_stamped_yaw.transform.rotation.w);
  tf2::Quaternion q_pitch(transform_stamped_pitch.transform.rotation.x,
                          transform_stamped_pitch.transform.rotation.y,
                          transform_stamped_pitch.transform.rotation.z,
                          transform_stamped_pitch.transform.rotation.w);

  double roll = 0.0, pitch = 0.0, yaw = 0.0;

  tf2::Matrix3x3(q_yaw).getRPY(roll, pitch, yaw);
  gimbal_yaw_pitch.first = yaw;

  tf2::Matrix3x3(q_pitch).getRPY(roll, pitch, yaw);
  gimbal_yaw_pitch.second = pitch;

  return gimbal_yaw_pitch;
}

void PlanningTrajectoryNode::Init()
{
  // Subscriber with tf2 message_filter
  // tf2 relevant
  tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  // Create the timer interface before call to waitForTransform,
  // to avoid a tf2_ros::CreateTimerInterfaceException exception
  auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
      this->get_node_base_interface(), this->get_node_timers_interface());
  tf2_buffer_->setCreateTimerInterface(timer_interface);
  tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

  // SolveTarget init parameters
  double k = this->declare_parameter("k", 0.092);
  double bias_time = this->declare_parameter("bias_time", 0.01);
  double s_bias = this->declare_parameter("s_bias", 0.0);
  double z_bias = this->declare_parameter("z_bias", 0.0);
  double pitch_bias = this->declare_parameter("pitch_bias", 0.0);
  send_frequency_ = this->declare_parameter("send_frequency", 200.0);
  dt_ = 1.0 / send_frequency_;

  bool use_table = this->declare_parameter("calculate_mode", true);

  double max_x = this->declare_parameter("table.max_x", 13.0);
  double min_x = this->declare_parameter("table.min_x", 0.0);
  double max_y = this->declare_parameter("table.max_y", 2.0);
  double min_y = this->declare_parameter("table.min_y", -1.0);
  double resolution = this->declare_parameter("table.resolution", 0.01);

  double max_x_lob = this->declare_parameter("table.max_x_lob", 22.0);
  double min_x_lob = this->declare_parameter("table.min_x_lob", 0.0);
  double max_y_lob = this->declare_parameter("table.max_y_lob", 3.0);
  double min_y_lob = this->declare_parameter("table.min_y_lob", -1.0);
  double resolution_lob = this->declare_parameter("table.resolution_lob", 0.01);

  k_yaw_ = this->declare_parameter("k_yaw", 0.0);
  k_pitch_ = this->declare_parameter("k_pitch", 0.0);

  std::string package_prefix =
      ament_index_cpp::get_package_share_directory("rm_vision_bringup") + "/config/";
  table_filename_normal_ =
      package_prefix + this->declare_parameter("table.filename", "table.bin");
  ;
  RCLCPP_ERROR(this->get_logger(), "table_filename_normal_: %s",
               table_filename_normal_.c_str());
  auto robot_type = this->declare_parameter<std::string>("robot_type", "default");
  is_hero_ = (robot_type == "hero");

  if (is_hero_)
  {
    table_filename_lob_ =
        package_prefix + this->declare_parameter("table.filename_lob", "");
    RCLCPP_ERROR(this->get_logger(), "table_filename_lob_: %s",
                 table_filename_lob_.c_str());
  }

  TrajectorySolver::CalculateMode calculate_mode =
      use_table ? TrajectorySolver::CalculateMode::TABLE_LOOKUP
                : TrajectorySolver::CalculateMode::NORMAL;

  table_config_ = {max_x, min_x, max_y, min_y, resolution, table_filename_normal_};
  if (is_hero_)
  {
    table_config_lob_ = {max_x_lob, min_x_lob,      max_y_lob,
                         min_y_lob, resolution_lob, table_filename_lob_};
  }
  else
  {
    table_config_lob_ = table_config_;
  }
  // ====== 新增：创建两个互斥型 callback group ======
  timer_cb_group_ =
      this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  sub_cb_group_ =
      this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  // ====== 订阅：放进 sub_cb_group_ ======
  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = sub_cb_group_;

  velocity_sub_ = this->create_subscription<auto_aim_interfaces::msg::Velocity>(
      "/current_velocity",
      rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data)),
      [this](const auto_aim_interfaces::msg::Velocity::SharedPtr velocity_msg)
      { trajectory_->InitVelocity(velocity_msg); }, sub_options);

  target_sub_ = this->create_subscription<auto_aim_interfaces::msg::Target>(
      "/tracker/target", rclcpp::SensorDataQoS(),
      [this](const auto_aim_interfaces::msg::Target::SharedPtr target_msg)
      { TargetCallback(target_msg); }, sub_options);

  // ====== 发布器（无 group 概念，发布是同步调用） ======
  send_pub_ = this->create_publisher<auto_aim_interfaces::msg::Send>(
      "/trajectory/send", rclcpp::SensorDataQoS());
  info_pub_ = this->create_publisher<auto_aim_interfaces::msg::TrajectoryInfo>(
      "/trajectory/info", rclcpp::SensorDataQoS());

  // ====== Timer：放进 timer_cb_group_ ======
  timer_ = this->create_wall_timer(
      std::chrono::duration<double, std::milli>(1000.0 / send_frequency_),
      std::bind(&PlanningTrajectoryNode::timer_callback, this), timer_cb_group_);
  q_yaw_ = this->declare_parameter("ekf.q_yaw", 0.0);
  q_pitch_ = this->declare_parameter("ekf.q_pitch", 0.0);
  q_jerk_ = this->declare_parameter("ekf.q_jerk", 0.0);
  r_yaw_ = this->declare_parameter("ekf.r_yaw", 0.0);
  r_pitch_ = this->declare_parameter("ekf.r_pitch", 0.0);
  auto f = [this](const Eigen::VectorXd& x) -> Eigen::VectorXd
  {
    Eigen::VectorXd x_new(4);
    x_new(0) = x(0) + x(1) * dt_ + 0.5 * x(2) * dt_ * dt_;
    x_new(1) = x(1) + x(2) * dt_;
    x_new(2) = x(2);
    x_new(3) = x(3);
    return x_new;
  };

  auto h = [](const Eigen::VectorXd& x) -> Eigen::VectorXd
  {
    Eigen::VectorXd z(2);
    z(0) = x(0);  // yaw
    z(1) = x(3);  // pitch
    return z;
  };

  // ---- 过程函数雅可比 j_f ----
  auto j_f = [this](const Eigen::VectorXd&) -> Eigen::MatrixXd
  {
    Eigen::MatrixXd F(4, 4);
    // clang-format off
    F << 1, dt_, 0.5 * dt_ * dt_, 0,
         0, 1,   dt_,             0,
         0, 0,   1,               0,
         0, 0,   0,               1;
    // clang-format on
    return F;
  };

  auto j_h = [](const Eigen::VectorXd&) -> Eigen::MatrixXd
  {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 4);
    H(0, 0) = 1.0;  // yaw
    H(1, 3) = 1.0;  // pitch
    return H;
  };
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(4, 4);

  // yaw: 独立噪声
  Q(0, 0) = q_yaw_;
  double T = dt_;  // 采样周期
  double T2 = T * T;
  double T3 = T2 * T;

  Q(1, 1) = q_jerk_ * T3 / 3.0;  // vy-vy
  Q(1, 2) = q_jerk_ * T2 / 2.0;  // vy-ay
  Q(2, 1) = q_jerk_ * T2 / 2.0;  // ay-vy
  Q(2, 2) = q_jerk_ * T;         // ay-ay

  // pitch: 独立噪声
  Q(3, 3) = q_pitch_;

  auto u_q = [Q]() -> Eigen::MatrixXd { return Q; };

  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(2, 2);
  R(0, 0) = r_yaw_;
  R(1, 1) = r_pitch_;
  auto u_r = [R](const Eigen::VectorXd&) -> Eigen::MatrixXd { return R; };

  Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(4, 4);
  P0(0, 0) = 0;
  P0(1, 1) = 1000.0;
  P0(2, 2) = 1000.0;
  P0(3, 3) = 0;

  TrajectorySolver solver(k, bias_time, s_bias, z_bias, pitch_bias, calculate_mode,
                          table_config_, table_config_lob_);
  ExtendedKalmanFilter ekf(f, h, j_f, j_h, u_q, u_r, P0);
  ConstrainedPlanner planner(25, 20, 500, dt_);

  trajectory_ = std::make_unique<Trajectory>(solver, std::move(ekf), std::move(planner));
}
}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its
// library is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::PlanningTrajectoryNode)
