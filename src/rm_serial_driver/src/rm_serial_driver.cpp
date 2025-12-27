#include "rm_serial_driver/rm_serial_driver.hpp"

#include <iostream>
#include <rclcpp/logging.hpp>

using namespace std::chrono_literals;

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
  timestamp_offset = this->declare_parameter<double>("timestamp_offset", 0);
  std::cout << "Serial timestamp_offset: " << timestamp_offset << '\n';
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

  // 打弹（t键打弹，g键停止）
  fire_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", rclcpp::SensorDataQoS(),
      [this](const geometry_msgs::msg::Twist::SharedPtr msg)
      {
        if (msg->linear.z > 0.0)
        {
          fire_notify = 1;
        }
        else if (msg->linear.z == 0.0)
        {
          fire_notify = 0;
        }
      });

  // 订阅 /tracker/send
  send_sub_ = this->create_subscription<auto_aim_interfaces::msg::Send>(
      "/tracker/send", rclcpp::SensorDataQoS(),
      std::bind(&RMSerialDriver::SendCallBack, this, std::placeholders::_1));

  XRobotMain(peripherals);

  // 云台姿态回调
  void (*ahrs_quaternion_cb_fun)(bool, RMSerialDriver* self, LibXR::RawData& data) =
      [](bool, RMSerialDriver* self, LibXR::RawData& data)
  {
    auto quat = reinterpret_cast<LibXR::Quaternion<float>*>(data.addr_);

    rm_serial_driver::gimbal_euler gimbal;
    self->ConvertQuaternionToEuler(quat->x(), quat->y(), quat->z(), quat->w(),
                                   gimbal.roll, gimbal.pitch, gimbal.yaw);
    if (++self->ahrs_receive_cnt % self->ahrs_print_freq == 0)
    {
      // RCLCPP_INFO(self->get_logger(),
      //             "Current gimbal Euler angles: roll:%f, pitch:%f, yaw:%f",
      //             gimbal.roll, gimbal.pitch, gimbal.yaw);
      self->ahrs_receive_cnt = 0;
    }
    // ROS2发布云台关节状态
    sensor_msgs::msg::JointState joint_state;
    joint_state.header.stamp =
        self->now() + rclcpp::Duration::from_seconds(self->timestamp_offset);
    joint_state.name.push_back("pitch_joint");
    joint_state.name.push_back("yaw_joint");
    joint_state.position.push_back(gimbal.pitch);
    joint_state.position.push_back(gimbal.yaw);
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

  // while (1)
  // {
  //   LibXR::Thread::Sleep(10);  // 发送延迟，10ms
  // }

  // auto timer_ = this->create_wall_timer(
  //     10ms,
  //     [this]()
  //     {
  //       // LibXR::Thread::Sleep(10);  // 发送延迟，10ms
  //       // LibXR::EulerAngle<float> target_euler;
  //       // target_euler.Pitch() = static_cast<float>(0.6);
  //       // target_euler.Yaw() = static_cast<float>(0.6);
  //       // target_euler.Roll() = 0.0f;
  //       // target_euler_topic_.Publish(target_euler);

  //       std::this_thread::sleep_for(std::chrono::milliseconds(10));
  //     });
}

RMSerialDriver::~RMSerialDriver() {}

// Send消息回调
void RMSerialDriver::SendCallBack(const auto_aim_interfaces::msg::Send::SharedPtr msg)
{
  LibXR::EulerAngle<float> target_euler;
  target_euler.Pitch() = static_cast<float>(msg->pitch);
  target_euler.Yaw() = static_cast<float>(msg->yaw);
  target_euler.Roll() = 0.0f;
  fire_notify = msg->is_fire;
  target_euler_topic_.Publish(target_euler);
  fire_notify_topic_.Publish(fire_notify);
}

/*四元数转欧拉角*/
void RMSerialDriver::ConvertQuaternionToEuler(float qx, float qy, float qz, float qw,
                                              float& roll, float& pitch, float& yaw)
{
  tf2::Quaternion q(static_cast<double>(qx), static_cast<double>(qy),
                    static_cast<double>(qz), static_cast<double>(qw));
  tf2::Matrix3x3 m(q);
  double d_roll = NAN, d_pitch = NAN, d_yaw = NAN;
  m.getRPY(d_roll, d_pitch, d_yaw);
  roll = static_cast<float>(d_roll);
  pitch = static_cast<float>(d_pitch);
  yaw = static_cast<float>(d_yaw);
}

}  // namespace rm_serial_driver

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_serial_driver::RMSerialDriver)
