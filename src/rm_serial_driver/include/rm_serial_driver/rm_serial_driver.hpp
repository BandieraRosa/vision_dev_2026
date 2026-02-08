#ifndef RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_
#define RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_

#ifndef PUBLISH_TEST
#define PUBLISH_TEST 0
#endif  // PUBLISH_TEST
// ROS2
#include <tf2/LinearMath/Matrix3x3.h>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/node.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
// LibXR
#include "SharedTopic.hpp"
#include "SharedTopicClient.hpp"
#include "linux_uart.hpp"

// ROS2自定义消息包
#include "auto_aim_interfaces/msg/send.hpp"
#include "auto_aim_interfaces/msg/velocity.hpp"

namespace rm_serial_driver
{

/*消息包*/
// 云台欧拉角数据结构体
typedef struct
{
  float pitch;
  float yaw;
  float roll;
} gimbal_euler;

/*LibXR相关*/

// LibXR应用程序入口函数
static void XRobotMain(LibXR::HardwareContainer& hw)
{
  using namespace LibXR;
  static ApplicationManager appmgr;

  static SharedTopic shared_topic(hw, appmgr, "uart_client", 81920, 256,
                                  {{"ahrs_quaternion"}});

  static SharedTopicClient shared_topic_client(hw, appmgr, "uart_client", 81920, 256,
                                               {{"target_euler"}});
  static SharedTopicClient shared_topic_client_1(hw, appmgr, "uart_client", 81920, 256,
                                                 {{"fire_notify", "tracker"}});
}

/* ==================== 测试逻辑架构优化 ==================== */

// 前置声明
class RMSerialDriver;

/**
 * @brief 测试电机控制器 - 封装所有测试相关逻辑
 *
 * 职责：
 * 1. 管理测试参数和状态
 * 2. 生成测试序列
 * 3. 控制测试线程
 * 4. 发布测试数据
 */
class TestMotorController
{
 public:
  // 测试模式枚举
  enum class Mode : uint8_t
  {
    UP = 0,        // 从 min -> max，走到末尾回到开头（循环）
    DOWN = 1,      // 从 max -> min，走到末尾回到开头（循环）
    PINGPONG = 2,  // min <-> max 往返
    FIXED = 3      // 固定在起点（由 mode 决定起点）
  };

  /**
   * @brief 构造函数
   * @param driver 关联的驱动对象（用于访问 topic 和 logger）
   * @param pitch_limit pitch 轴范围
   * @param yaw_limit yaw 轴范围
   * @param pitch_step pitch 轴步数
   * @param yaw_step yaw 轴步数
   * @param switch_times 切换周期（每多少次循环切换一次目标）
   * @param pitch_mode pitch 轴模式
   * @param yaw_mode yaw 轴模式
   */
  TestMotorController(RMSerialDriver* driver, std::pair<float, float> pitch_limit,
                      std::pair<float, float> yaw_limit, std::size_t pitch_step,
                      std::size_t yaw_step, int switch_times, Mode pitch_mode,
                      Mode yaw_mode);

  ~TestMotorController();

  // 启动测试线程
  void Start();

  // 停止测试线程（如需要）
  void Stop();

  // 获取当前测试模式
  Mode GetPitchMode() const { return pitch_mode_; }
  Mode GetYawMode() const { return yaw_mode_; }

 private:
  /**
   * @brief 扫描索引状态
   */
  struct SweepIndex
  {
    std::size_t idx = 0;  // 当前索引
    int dir = +1;         // 方向：+1 正向，-1 反向
  };

  /**
   * @brief 测试状态 - 线程共享数据
   */
  struct TestState
  {
    std::vector<float> pitch_table;          // pitch 测试序列
    std::vector<float> yaw_table;            // yaw 测试序列
    SweepIndex pitch_index;                  // pitch 索引状态
    SweepIndex yaw_index;                    // yaw 索引状态
    int loop_count = 0;                      // 循环计数
    int switch_period = 1;                   // 切换周期
    LibXR::EulerAngle<float> current_euler;  // 当前欧拉角

    TestState(std::vector<float> p_table, std::vector<float> y_table, int period)
        : pitch_table(std::move(p_table)),
          yaw_table(std::move(y_table)),
          switch_period(period)
    {
      UpdateCurrentEuler();
    }

    // 更新当前欧拉角
    void UpdateCurrentEuler()
    {
      current_euler.Pitch() = pitch_table.empty() ? 0.0f : pitch_table[pitch_index.idx];
      current_euler.Yaw() = yaw_table.empty() ? 0.0f : yaw_table[yaw_index.idx];
      current_euler.Roll() = 0.0f;
    }
  };

  /**
   * @brief 生成线性测试序列
   * @param limit 范围限制 (min, max)
   * @param step 步数
   * @param mode 测试模式
   * @return 测试序列
   */
  static std::vector<float> GenerateSequence(std::pair<float, float> limit,
                                             std::size_t step, Mode mode);

  /**
   * @brief 推进索引（根据模式）
   * @param index 索引状态
   * @param size 序列大小
   * @param mode 测试模式
   */
  static void AdvanceIndex(SweepIndex& index, std::size_t size, Mode mode);

  /**
   * @brief 测试线程函数
   */
  void ThreadFunction();

  /**
   * @brief 静态线程入口（适配 LibXR::Thread 接口）
   */
  static void ThreadEntry(TestMotorController* controller)
  {
    controller->ThreadFunction();
  }

  // 成员变量
  RMSerialDriver* driver_;                 // 关联的驱动对象
  Mode pitch_mode_;                        // pitch 测试模式
  Mode yaw_mode_;                          // yaw 测试模式
  std::unique_ptr<TestState> state_;       // 测试状态（智能指针管理）
  std::unique_ptr<LibXR::Thread> thread_;  // 测试线程
  bool running_ = false;                   // 运行标志
};

/* ==================== RMSerialDriver类定义 ==================== */
class RMSerialDriver : public rclcpp::Node
{
 public:
  explicit RMSerialDriver(const rclcpp::NodeOptions& options);
  ~RMSerialDriver() override;

  int ahrs_receive_cnt = 0;
  int ahrs_print_freq = 50;

  // 允许 TestMotorController 访问必要的成员
  friend class TestMotorController;

 private:
  uint8_t fire_notify_ = 1;
  double timestamp_offset_{};

  /* 函数声明 */

  // 四元数转欧拉角函数
  void ConvertQuaternionToEuler(float qx, float qy, float qz, float qw, float& roll,
                                float& pitch, float& yaw);

  // Send消息回调函数
  void SendCallBack(const auto_aim_interfaces::msg::Send::SharedPtr msg);

  /* ROS2发布者 */
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr
      joint_state_pub_;  // 云台关节状态发布者
  rclcpp::Publisher<auto_aim_interfaces::msg::Velocity>::SharedPtr
      velocity_pub_;  // 弹速发布者

  /* ROS2订阅者 */
  rclcpp::Subscription<auto_aim_interfaces::msg::Send>::SharedPtr send_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr fire_sub_;

  /* LibXR Topic (用于发送到下位机) */
  LibXR::Topic ahrs_quaternion_topic_;
  LibXR::Topic bullet_speed_topic_;
  LibXR::Topic target_euler_topic_;
  LibXR::Topic fire_notify_topic_;

  /* LibXR初始化相关成员变量 */
  std::unique_ptr<LibXR::RamFS> ramfs_;
  std::unique_ptr<LibXR::LinuxUART> uart_client_;
  std::unique_ptr<LibXR::Terminal<1024, 64, 16, 128>> terminal_;
  std::unique_ptr<LibXR::Thread> term_thread_;
  std::unique_ptr<LibXR::HardwareContainer> peripherals_;

  /* 测试控制器 - 使用智能指针管理 */
  std::unique_ptr<TestMotorController> test_controller_;
};

}  // namespace rm_serial_driver
#endif  // RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_