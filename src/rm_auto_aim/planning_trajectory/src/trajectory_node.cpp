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
  if (target_msg->is_switchtable)
  {
    gaf_solver_->SwitchTable();
  }
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

  send_count_ = 0;
}

void PlanningTrajectoryNode::timer_callback()
{
  if (!tracking_)
  {
    return;
  }

  auto_aim_interfaces::msg::Send send_msg;

  double bc_yaw{0.0};
  double bc_pitch{0.0};

  // 获取当前云台在世界系下的 yaw
  auto gimbal_yaw_pitch = GetGimbalYawAndPitch();
  double gimbal_yaw = gimbal_yaw_pitch.first;
  double gimbal_pitch = gimbal_yaw_pitch.second;

  double aim_x = 0, aim_y = 0, aim_z = 0;
  int idx{};
  TrajectorySolver::control cmd;

  // 自然系下pitch向下为正，而AutoSolveTrajectory中pitch向上为正
  gaf_solver_->AutoSolveTrajectory(cmd.pitch, cmd.yaw, cmd.is_fire, aim_x, aim_y, aim_z,
                                   idx, target_, gimbal_yaw, dt_);
  bc_yaw = cmd.yaw;
  bc_pitch = cmd.pitch;

  trajectory_->UpdatePlanTrajectory(cmd, send_count_);
  send_count_++;

  cmd.pitch = -cmd.pitch;

  send_msg.is_fire = cmd.is_fire;
  send_msg.pitch = cmd.pitch;
  send_msg.yaw = cmd.yaw;

  send_pub_->publish(send_msg);

  auto_aim_interfaces::msg::TrajectoryInfo info_msg;
  info_msg.aim_position.x = aim_x;
  info_msg.aim_position.y = aim_y;
  info_msg.aim_position.z = aim_z;
  info_msg.gimbal_yaw = gimbal_yaw;
  info_msg.gimbal_pitch = gimbal_pitch;
  info_msg.idx = idx;
  info_msg.bc_yaw = bc_yaw;
  info_msg.bc_pitch = bc_pitch;
  info_pub_->publish(info_msg);
}

std::pair<double, double> PlanningTrajectoryNode::GetGimbalYawAndPitch()
{
  std::pair<double, double> gimbal_yaw_pitch{0.0, 0.0};
  auto transform_stamped_yaw =
      tf2_buffer_->lookupTransform("gimbal_odom", "yaw_link", tf2::TimePointZero);
  auto transform_stamped_pitch =
      tf2_buffer_->lookupTransform("gimbal_odom", "pitch_link", tf2::TimePointZero);

  // 从变换中提取四元数并转换为欧拉角
  tf2::Quaternion q_yaw(transform_stamped_yaw.transform.rotation.x,
                        transform_stamped_yaw.transform.rotation.y,
                        transform_stamped_yaw.transform.rotation.z,
                        transform_stamped_yaw.transform.rotation.w);
  tf2::Quaternion q_pitch(transform_stamped_pitch.transform.rotation.x,
                          transform_stamped_pitch.transform.rotation.y,
                          transform_stamped_pitch.transform.rotation.z,
                          transform_stamped_pitch.transform.rotation.w);

  double gimbal_roll{}, gimbal_pitch{}, gimbal_yaw{};
  tf2::Matrix3x3(q_yaw).getRPY(gimbal_roll, gimbal_pitch, gimbal_yaw);
  gimbal_yaw_pitch.first = gimbal_yaw;
  tf2::Matrix3x3(q_pitch).getRPY(gimbal_roll, gimbal_pitch, gimbal_yaw);
  gimbal_yaw_pitch.second = gimbal_pitch;
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
  double k = this->declare_parameter("planning_trajectory.k", 0.092);
  double bias_time = this->declare_parameter("planning_trajectory.bias_time", 0.01);
  double s_bias = this->declare_parameter("planning_trajectory.s_bias", 0.0);
  double z_bias = this->declare_parameter("planning_trajectory.z_bias", 0.0);
  double pitch_bias = this->declare_parameter("planning_trajectory.pitch_bias", 0.0);
  send_frequency_ = this->declare_parameter("planning_trajectory.send_frequency", 200.0);
  dt_ = 1.0 / send_frequency_;

  bool use_table = this->declare_parameter("planning_trajectory.calculate_mode", true);

  double max_x = this->declare_parameter("planning_trajectory.table.max_x", 13.0);
  double min_x = this->declare_parameter("planning_trajectory.table.min_x", 0.0);
  double max_y = this->declare_parameter("planning_trajectory.table.max_y", 2.0);
  double min_y = this->declare_parameter("planning_trajectory.table.min_y", -1.0);
  double resolution =
      this->declare_parameter("planning_trajectory.table.resolution", 0.01);

  double max_x_lob = this->declare_parameter("planning_trajectory.table.max_x_lob", 22.0);
  double min_x_lob = this->declare_parameter("planning_trajectory.table.min_x_lob", 0.0);
  double max_y_lob = this->declare_parameter("planning_trajectory.table.max_y_lob", 3.0);
  double min_y_lob = this->declare_parameter("planning_trajectory.table.min_y_lob", -1.0);
  double resolution_lob =
      this->declare_parameter("planning_trajectory.table.resolution_lob", 0.01);

  k_yaw_ = this->declare_parameter("planning_trajectory.k_yaw", 0.0);
  k_pitch_ = this->declare_parameter("planning_trajectory.k_pitch", 0.0);

  std::string package_prefix =
      ament_index_cpp::get_package_share_directory("rm_vision_bringup") + "/config/";
  table_filename_normal_ =
      package_prefix +
      this->declare_parameter("planning_trajectory.table.filename", "table.bin");
  ;
  RCLCPP_ERROR(this->get_logger(), "table_filename_normal_: %s",
               table_filename_normal_.c_str());
  auto robot_type = this->declare_parameter<std::string>("robot_type", "default");
  is_hero_ = (robot_type == "hero");

  if (is_hero_)
  {
    table_filename_lob_ =
        package_prefix +
        this->declare_parameter("planning_trajectory.table.filename_lob", "");
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
  gaf_solver_ = std::make_unique<TrajectorySolver>(k, bias_time, s_bias, z_bias,
                                                   pitch_bias, calculate_mode,
                                                   table_config_, table_config_lob_);
  velocity_sub_ = this->create_subscription<auto_aim_interfaces::msg::Velocity>(
      "/current_velocity",
      rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data)),
      [this](const auto_aim_interfaces::msg::Velocity::SharedPtr velocity_msg)
      { gaf_solver_->Init(velocity_msg); });
  target_sub_ = this->create_subscription<auto_aim_interfaces::msg::Target>(
      "/tracker/target", rclcpp::SensorDataQoS(),
      [this](const auto_aim_interfaces::msg::Target::SharedPtr target_msg)
      { TargetCallback(target_msg); });

  send_pub_ = this->create_publisher<auto_aim_interfaces::msg::Send>(
      "/trajectory/send", rclcpp::SensorDataQoS());
  info_pub_ = this->create_publisher<auto_aim_interfaces::msg::TrajectoryInfo>(
      "/trajectory/info", rclcpp::SensorDataQoS());

  timer_ = this->create_wall_timer(
      std::chrono::duration<double, std::milli>(1000.0 / send_frequency_),
      std::bind(&PlanningTrajectoryNode::timer_callback, this));

  q_yaw_ = this->declare_parameter("planning_trajectory.ekf.q_yaw", 0.0);
  q_pitch_ = this->declare_parameter("planning_trajectory.ekf.q_pitch", 0.0);
  q_vy_ = this->declare_parameter("planning_trajectory.ekf.q_vy", 0.0);
  q_ay_ = this->declare_parameter("planning_trajectory.ekf.q_ay", 0.0);
  r_yaw_ = this->declare_parameter("planning_trajectory.ekf.r_yaw", 0.0);
  r_pitch_ = this->declare_parameter("planning_trajectory.ekf.r_pitch", 0.0);
  auto f = [this](const Eigen::VectorXd& x) -> Eigen::VectorXd
  {
    Eigen::VectorXd x_new(4);
    x_new(0) = x(0) + x(1) * dt_ + 0.5 * dt_ * dt_;
    x_new(1) = x(1) + x(2) * dt_;
    x_new(2) = x(2);
    x_new(3) = x(3);
    return x_new;
  };

  auto h = [](const Eigen::VectorXd& x) -> Eigen::VectorXd
  {
    Eigen::VectorXd z(2);
    z(0) = x(0);
    z(1) = x(2);
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
    Eigen::MatrixXd H(2, 4);
    // clang-format off
    H << 1, 0, 
         0, 0, 
         0, 0,
         0, 1;
    // clang-format on
    return H;
  };

  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(4, 4);
  Q(0, 0) = q_yaw_;
  Q(1, 1) = q_vy_;
  Q(2, 2) = q_ay_;
  Q(3, 3) = q_pitch_;
  auto u_q = [Q]() -> Eigen::MatrixXd { return Q; };

  Eigen::MatrixXd R(2, 2);
  R(0, 0) = r_yaw_;
  R(1, 1) = r_pitch_;
  auto u_r = [R](const Eigen::VectorXd&) -> Eigen::MatrixXd { return R; };

  Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(4, 4);
  P0(0, 0) = 0;
  P0(1, 1) = 1000.0;
  P0(2, 2) = 1000.0;
  P0(3, 3) = 0;

  trajectory_->ekf_ = ExtendedKalmanFilter(f, h, j_f, j_h, u_q, u_r, P0);
  trajectory_->planner_ = ConstrainedPlanner(25, 10, 10, dt_);
}
}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its
// library is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::PlanningTrajectoryNode)
