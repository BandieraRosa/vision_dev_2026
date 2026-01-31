#include "rm_serial_driver/rm_serial_driver.hpp"

#include <cstdint>
#include <rclcpp/logging.hpp>

using namespace std::chrono_literals;

namespace rm_serial_driver
{
TestMotorController::TestMotorController(RMSerialDriver* driver,
                                         std::pair<float, float> pitch_limit,
                                         std::pair<float, float> yaw_limit,
                                         std::size_t pitch_step, std::size_t yaw_step,
                                         int switch_times, Mode pitch_mode, Mode yaw_mode)
    : driver_(driver), pitch_mode_(pitch_mode), yaw_mode_(yaw_mode)
{
  // 参数校验和默认值处理
  if (yaw_step == 0)
  {
    yaw_step = pitch_step;
  }
  if (switch_times <= 0)
  {
    switch_times = 250;
  }

  // 生成测试序列
  auto pitch_table = GenerateSequence(pitch_limit, pitch_step, pitch_mode);
  auto yaw_table = GenerateSequence(yaw_limit, yaw_step, yaw_mode);

  // 创建测试状态（使用智能指针）
  state_ = std::make_unique<TestState>(std::move(pitch_table), std::move(yaw_table),
                                       switch_times);

  // 创建测试线程（但不启动）
  thread_ = std::make_unique<LibXR::Thread>();
}

TestMotorController::~TestMotorController() { Stop(); }

void TestMotorController::Start()
{
  if (running_)
  {
    return;
  }

  running_ = true;

  // 创建并启动线程
  thread_->Create<TestMotorController*>(this, ThreadEntry, "test_motor_publish", 81900,
                                        LibXR::Thread::Priority::MEDIUM);
}

void TestMotorController::Stop()
{
  running_ = false;
  // 线程清理（如果 LibXR::Thread 支持停止机制）
  thread_->Yield();
}

std::vector<float> TestMotorController::GenerateSequence(std::pair<float, float> limit,
                                                         std::size_t step, Mode mode)
{
  float lo = std::min(limit.first, limit.second);
  float hi = std::max(limit.first, limit.second);

  std::size_t n = (step == 0) ? 1 : step;

  std::vector<float> table;
  table.reserve(n);

  if (n == 1 || mode == Mode::FIXED)
  {
    table.push_back((hi + lo) / 2.0f);
    return table;
  }

  for (std::size_t i = 0; i < n; ++i)
  {
    float t = static_cast<float>(i) / static_cast<float>(n - 1);
    table.push_back(lo + (hi - lo) * t);
  }

  // 根据模式调整序列
  if (mode == Mode::DOWN)
  {
    std::reverse(table.begin(), table.end());
  }

  return table;
}

void TestMotorController::AdvanceIndex(SweepIndex& index, std::size_t size, Mode mode)
{
  if (size <= 1)
  {
    return;
  }

  switch (mode)
  {
    case Mode::FIXED:
      // 固定模式：不推进
      return;

    case Mode::UP:
    case Mode::DOWN:
    {
      // 单向循环模式
      index.idx = (index.idx + 1) % size;
      return;
    }

    case Mode::PINGPONG:
    {
      // 往返模式
      if (index.dir > 0)
      {
        // 正向移动
        if (index.idx + 1 >= size)
        {
          // 到达末尾，反向
          index.dir = -1;
          if (index.idx > 0)
          {
            --index.idx;
          }
        }
        else
        {
          ++index.idx;
        }
      }
      else
      {
        // 反向移动
        if (index.idx == 0)
        {
          // 到达起点，正向
          index.dir = +1;
          if (size > 1)
          {
            ++index.idx;
          }
        }
        else
        {
          --index.idx;
        }
      }
      return;
    }
  }
}

void TestMotorController::ThreadFunction()
{
  while (running_)
  {
    LibXR::Thread::Sleep(2);  // 2ms 周期

    // 循环计数并按周期切换目标
    if (++state_->loop_count % state_->switch_period == 0)
    {
      // 推进索引
      AdvanceIndex(state_->pitch_index, state_->pitch_table.size(), pitch_mode_);
      AdvanceIndex(state_->yaw_index, state_->yaw_table.size(), yaw_mode_);

      // 更新当前欧拉角
      state_->UpdateCurrentEuler();

      // 日志输出
      RCLCPP_INFO(driver_->get_logger(), "Publish test target_euler: pitch %f, yaw %f",
                  state_->current_euler.Pitch(), state_->current_euler.Yaw());
    }

    // 高频发布（每次循环都发布）
    driver_->target_euler_topic_.Publish(state_->current_euler);
  }
}

/* ==================== RMSerialDriver 实现 ==================== */

RMSerialDriver::RMSerialDriver(const rclcpp::NodeOptions& options)
    : Node("rm_serial_driver", options)
{
  LibXR::PlatformInit();
  peripherals_ = std::make_unique<LibXR::HardwareContainer>();
  ramfs_ = std::make_unique<LibXR::RamFS>();

  auto vid = this->declare_parameter<std::string>("vid", "16d0");
  auto pid = this->declare_parameter<std::string>("pid", "1492");
  timestamp_offset_ = this->declare_parameter<double>("timestamp_offset", 0);
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
      //   RCLCPP_INFO(self->get_logger(),
      //               "Current gimbal Euler angles: roll:%f, pitch:%f, yaw:%f",
      //               gimbal.roll, gimbal.pitch, gimbal.yaw);
      self->ahrs_receive_cnt = 0;
    }
    // ROS2发布云台关节状态
    sensor_msgs::msg::JointState joint_state;
    joint_state.header.stamp =
        self->now() + rclcpp::Duration::from_seconds(self->timestamp_offset_);
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

#if PUBLISH_TEST
  float pitch_min = this->declare_parameter<float>("test.pitch.min", -0.3f);
  float pitch_max = this->declare_parameter<float>("test.pitch.max", 0.3f);
  int pitch_steps = this->declare_parameter<int>("test.pitch.steps", 10);
  float yaw_min = this->declare_parameter<float>("test.yaw.min", -0.5f);
  float yaw_max = this->declare_parameter<float>("test.yaw.max", 0.5f);
  int yaw_steps = this->declare_parameter<int>("test.yaw.steps", 10);
  int switch_period = this->declare_parameter<int>("test.switch_period", 250);

  test_controller_ = std::make_unique<TestMotorController>(
      this, std::make_pair(pitch_min, pitch_max),  // pitch 范围
      std::make_pair(yaw_min, yaw_max),            // yaw 范围
      static_cast<std::size_t>(pitch_steps),       // pitch 步数
      static_cast<std::size_t>(yaw_steps),         // yaw 步数
      switch_period,                               // 切换周期
      TestMotorController::Mode::PINGPONG,         // pitch 模式
      TestMotorController::Mode::PINGPONG          // yaw 模式
  );

  test_controller_->Start();

  RCLCPP_INFO(this->get_logger(), "Test motor controller started");
#endif
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