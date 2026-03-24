#include "rm_serial_driver/rm_serial_driver.hpp"

namespace rm_serial_driver
{
RMSerialDriver::RMSerialDriver(const rclcpp::NodeOptions& options)
    : Node("rm_serial_driver", options)
{
  LibXR::PlatformInit();
  peripherals_ = std::make_unique<LibXR::HardwareContainer>();
  ramfs_ = std::make_unique<LibXR::RamFS>();

  auto vid = this->declare_parameter<std::string>("vid", "16d0");
  auto pid = this->declare_parameter<std::string>("pid", "1492");
  timestamp_offset_ = this->declare_parameter<double>("timestamp_offset", 0);
  auto robot_type = this->declare_parameter<std::string>("robot_type", "default");
  is_hero_ = (robot_type == "hero");
  std::cout << "Serial timestamp_offset: " << timestamp_offset_ << '\n';

  uart_client_ = std::make_unique<LibXR::LinuxUART>(
      vid, pid, 115200, LibXR::LinuxUART::Parity::NO_PARITY, 8, 1);
  terminal_ = std::make_unique<LibXR::Terminal<1024, 64, 16, 128>>(*ramfs_);
  term_thread_ = std::make_unique<LibXR::Thread>();
  term_thread_->Create(terminal_.get(), LibXR::Terminal<1024, 64, 16, 128>::ThreadFun,
                       "terminal", 81900, LibXR::Thread::Priority::MEDIUM);

  static LibXR::HardwareContainer peripherals{
      LibXR::Entry<LibXR::RamFS>({*ramfs_, {"ramfs"}}),
      LibXR::Entry<LibXR::UART>({*uart_client_, {"uart_client"}}),
  };

  // 从下位机接收的话题
  ahrs_quaternion_topic_ =
      LibXR::Topic::FindOrCreate<LibXR::Quaternion<float>>("ahrs_quaternion");

  lob_shot_topic_ = LibXR::Topic::FindOrCreate<uint8_t>("lob_shot");

  LibXR::Topic::Domain referee_domain = LibXR::Topic::Domain("referee");
  bullet_speed_topic_ =
      LibXR::Topic::FindOrCreate<float>("bullet_speed", &referee_domain);

  // 发送到下位机的话题
  LibXR::Topic::Domain tracker_domain = LibXR::Topic::Domain("tracker");
  target_euler_topic_ =
      LibXR::Topic::FindOrCreate<LibXR::EulerAngle<float>>("target_euler");
  fire_notify_topic_ =
      LibXR::Topic::FindOrCreate<uint8_t>("fire_notify", &tracker_domain);

  // 云台关节状态
  joint_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
      "/joint_states", rclcpp::QoS(rclcpp::KeepLast(1)));

  // 弹速
  velocity_pub_ =
      this->create_publisher<auto_aim_interfaces::msg::Velocity>("/current_velocity", 10);

  // 吊射标志
  if (is_hero_)
  {
    lob_shot_pub_ = this->create_publisher<std_msgs::msg::Bool>(
        "/lob_shot_switch", rclcpp::QoS(1).reliable());
  }

  // 打弹（t键打弹，g键停止）
  fire_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", rclcpp::SensorDataQoS(),
      [this](const geometry_msgs::msg::Twist::SharedPtr msg)
      {
        if (msg->linear.z > 0.0)
        {
          fire_notify_ = 1;
        }
        else if (msg->linear.z == 0.0)
        {
          fire_notify_ = 0;
        }
      });

  // 订阅 /tracker/send
  send_sub_ = this->create_subscription<auto_aim_interfaces::msg::Send>(
      "/tracker/send", rclcpp::SensorDataQoS(),
      [this](const auto_aim_interfaces::msg::Send::SharedPtr msg) { SendCallBack(msg); });

  XRobotMain(peripherals);

  // 云台姿态回调
  void (*ahrs_quaternion_cb_fun)(bool, RMSerialDriver* self, LibXR::RawData& data) =
      [](bool, RMSerialDriver* self, LibXR::RawData& data)
  {
    auto quat = reinterpret_cast<LibXR::Quaternion<float>*>(data.addr_);

    LibXR::EulerAngle<float> gimbal = quat->ToEulerAngle();
    if (++self->ahrs_receive_cnt_ % self->ahrs_print_freq_ == 0)
    {
      // RCLCPP_INFO(self->get_logger(),
      //             "Current gimbal Euler angles: roll:%f, pitch:%f, yaw:%f",
      //             gimbal.Roll(), gimbal.Pitch(), gimbal.Yaw());
      self->ahrs_receive_cnt_ = 0;
    }
    // ROS2发布云台关节状态
    sensor_msgs::msg::JointState joint_state;
    joint_state.header.stamp =
        self->now() - rclcpp::Duration::from_seconds(self->timestamp_offset_);
    joint_state.name.push_back("pitch_joint");
    joint_state.name.push_back("yaw_joint");
    joint_state.position.push_back(gimbal.Pitch());
    joint_state.position.push_back(gimbal.Yaw());
    self->joint_state_pub_->publish(joint_state);
  };
  auto ahrs_quaternion_cb = LibXR::Topic::Callback::Create(ahrs_quaternion_cb_fun, this);
  ahrs_quaternion_topic_.RegisterCallback(ahrs_quaternion_cb);

  // 弹速回调
  void (*bullet_speed_cb_fun)(bool, RMSerialDriver* self, LibXR::RawData& data) =
      [](bool, RMSerialDriver* self, LibXR::RawData& data)
  {
    auto bullet_speed = reinterpret_cast<float*>(data.addr_);
    // XR_LOG_INFO("Serial got bullet_speed:%f", *bullet_speed);

    // ROS2发布弹速
    auto_aim_interfaces::msg::Velocity velocity_msg;
    velocity_msg.header.stamp = self->now();
    velocity_msg.velocity = static_cast<double>(*bullet_speed);
    self->velocity_pub_->publish(velocity_msg);
  };
  auto bullet_speed_cb = LibXR::Topic::Callback::Create(bullet_speed_cb_fun, this);
  bullet_speed_topic_.RegisterCallback(bullet_speed_cb);

  // 吊射标志回调
  if (is_hero_)
  {
    void (*lob_shot_cb_fun)(bool, RMSerialDriver* self, LibXR::RawData& data) =
        [](bool, RMSerialDriver* self, LibXR::RawData& data)
    {
      auto val = *reinterpret_cast<uint8_t*>(data.addr_);
      uint8_t prev = self->last_lob_val_;
      self->last_lob_val_ = val;
      if (prev == 0 && val != 0)
      {
        std_msgs::msg::Bool msg;
        msg.data = true;
        self->lob_shot_pub_->publish(msg);
        RCLCPP_INFO(self->get_logger(),
                    "Lob shot edge detected (0->1), published switch.");
      }
    };
    auto lob_shot_cb = LibXR::Topic::Callback::Create(lob_shot_cb_fun, this);
    lob_shot_topic_.RegisterCallback(lob_shot_cb);
  }
}

RMSerialDriver::~RMSerialDriver() {}

// Send消息回调
void RMSerialDriver::SendCallBack(const auto_aim_interfaces::msg::Send::SharedPtr msg)
{
  LibXR::EulerAngle<float> target_euler;
  target_euler.Pitch() = static_cast<float>(msg->pitch);
  target_euler.Yaw() = static_cast<float>(msg->yaw);
  target_euler.Roll() = 0.0f;
  fire_notify_ = msg->is_fire;
  target_euler_topic_.Publish(target_euler);
  fire_notify_topic_.Publish(fire_notify_);
}

}  // namespace rm_serial_driver

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_serial_driver::RMSerialDriver)