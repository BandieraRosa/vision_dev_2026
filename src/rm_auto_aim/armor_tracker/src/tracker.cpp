#include "armor_tracker/tracker.hpp"

namespace rm_auto_aim
{
double Tracker::outpost_dz = 0.1;
double Tracker::outpost_r = 0.2765;
int Tracker::outpost_idx = 0;
double Tracker::outpost_cast_threshold = 0.0;

Tracker::Tracker(double max_match_distance, double max_match_yaw_diff)
    : tracker_state(State::LOST),
      tracked_id(std::string("")),
      measurement(Eigen::VectorXd::Zero(4)),
      target_state(Eigen::VectorXd::Zero(9)),
      max_match_distance_(max_match_distance),
      max_match_yaw_diff_(max_match_yaw_diff)
{
  outpost_idx = 0;
}

void Tracker::Init(const Armors::SharedPtr& armors_msg)
{
  if (armors_msg->armors.empty())
  {
    return;
  }

  double min_distance = DBL_MAX;  // 定义最大初始值
  tracked_armor = armors_msg->armors[0];
  for (const auto& armor : armors_msg->armors)
  {
    if (armor.distance_to_image_center < min_distance)
    {
      min_distance = armor.distance_to_image_center;
      tracked_armor = armor;
    }  // 选择距离屏幕中心最近的装甲板
       // 为被追踪的装甲板，一旦选定跟踪目标后，在LOST之前不会切换目标，即使新目标在中心
  }

  InitEkf(tracked_armor);
  RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "Init EKF!");

  tracked_id = tracked_armor.number;
  tracked_armor_type = tracked_armor.type;
  tracker_state = State::DETECTING;
  UpdateArmorsNum(tracked_armor);
}

void Tracker::Update(const Armors::SharedPtr& armors_msg)
{
  Eigen::VectorXd ekf_prediction = ekf.predict();  // 根据整车c的预测，得出装甲板的位置
  RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF predict");
  bool matched = false;  // 对预测的装甲板和观测的装甲板进行匹配
  target_state = ekf_prediction;  // 整车c的预测向量

  if (!armors_msg->armors.empty())
  {
    int same_id_armors_count = 0;
    predicted_position =
        GetArmorPositionFromState(ekf_prediction);  // 计算,根据卡尔曼得到预测装甲板位置
    double min_position_diff = DBL_MAX;  // 最小位置差值,最大初始值大幅
    double yaw_diff = DBL_MAX;  // 定义yaw差值,预测装甲板和真实装甲板

    MatchArmor(armors_msg, ekf_prediction, same_id_armors_count, min_position_diff,
               yaw_diff);

    // 存储tracker信息
    info_position_diff = min_position_diff;
    info_yaw_diff = yaw_diff;

    // 检查最近装甲的距离和偏航角差是否在阈值范围内
    // 最近装甲板距离与yaw差值比阈值小
    // 找到匹配的装甲板
    matched = MatchArmor(armors_msg, ekf_prediction, same_id_armors_count,
                         min_position_diff, yaw_diff);  // 注意之前的 matched = false
    if (!matched)
    // 未找到匹配的装甲，但仅有一个具有相同 ID 的装甲
    {
      double health_rate = ekf.GetHealthRate();
      if (same_id_armors_count == 1 &&
          yaw_diff > (tracked_armor.number != "outpost" ? max_match_yaw_diff_
                                                        : max_match_yaw_diff_ + 0.7))
      // 偏航角差距大，将此情况视为目标正在旋转并且装甲发生了 **跳变**
      {
        RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "armor_yaw_diff: %f", yaw_diff);
        HandleArmorJump(tracked_armor);  // 跳变处理
      }
      else if (0.5 < health_rate && health_rate < 0.8)
      {
        SoftBreakEKF(ekf_prediction(2), tracked_armor.pose.position.y);
      }
      else if (health_rate < 0.5)
      {
        RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "EKF health rate: %f",
                    health_rate);
        // ResetState(ekf_prediction(6), predicted_position);
      }
    }
  }

  // 防止半径扩散
  ClampTargetRadius();

  // 跟踪状态机制处理
  UpdateTrackerState(matched);
}

bool Tracker::MatchArmor(const Armors::SharedPtr& armors_msg,
                         Eigen::VectorXd& ekf_prediction, int& same_id_armors_count,
                         double& min_position_diff, double& yaw_diff)
{
  // todo 添加当前帧不同id到相机中心的距离判断，保证操作手端的使用，功能实现放到其他函数中

  int correct_match_count = 0;
  for (const auto& armor : armors_msg->armors)
  {  // 遍历当前装甲板
    // 只考虑具有相同 ID 的装甲,忽略其余装甲
    if (armor.number == tracked_id)
    {
      same_id_armors_count++;
      // 误差分析,计算预测位置与当前装甲位置之间的差异
      auto p = armor.pose.position;  // p是真正的观察到的 装甲板的 position
      Eigen::Vector3d position_vec(p.x, p.y, p.z);
      double position_diff = (predicted_position - position_vec).norm();
      double yaw_diff = abs(OrientationToYaw(armor.pose.orientation) - ekf_prediction(6));

      if (position_diff < max_match_distance_ &&
          yaw_diff < (tracked_armor.number != "outpost" ? max_match_yaw_diff_
                                                        : max_match_yaw_diff_ + 0.7))
      {  // 最近装甲板距离与yaw差值比阈值小
        // 找到匹配的装甲板
        // matched = true;  // 注意之前的 matched = false
        if (position_diff < min_position_diff)
        {  // 选择距离最近的装甲板作为匹配装甲板
          min_position_diff = position_diff;
          tracked_armor = armor;
        }
        correct_match_count++;
      }
    }
  }
  if (correct_match_count != 0)
  {
    auto p = tracked_armor.pose.position;
    // 更新 EKF
    double measured_yaw =
        OrientationToYaw(tracked_armor.pose.orientation);  // 测量的yaw值
    if (tracked_armors_num == ArmorsNum::OUTPOST_3)
    {
      p.z = p.z + (1 - outpost_idx) * outpost_dz;
    }
    measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);
    target_state = ekf.update(measurement);

    if (tracked_id == "outpost")
    {
      target_state(1) = 0;
      target_state(3) = 0;
      target_state(5) = 0;
      target_state(7) = 0.8 * (target_state(7) > 0) - (target_state(7) < 0);
    }
    else
    {
      VelocityConstrain(10, 10, 10, 10, 1);
    }

    RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update");
    return true;
  }
  else
  {
    return false;
  }
}

void Tracker::ClampTargetRadius()
{
  target_state(8) =
      (tracked_id == "outpost") ? outpost_r : std::clamp(target_state(8), 0.12, 0.4);

  ekf.setState(target_state);
}

void Tracker::UpdateTrackerState(bool& matched)
{
  if (tracker_state == State::DETECTING)
  {
    if (matched)
    {
      detect_count_++;
      if (detect_count_ > tracking_thres)
      {
        detect_count_ = 0;
        tracker_state = State::TRACKING;
      }
    }
    else
    {
      detect_count_ = 0;
      tracker_state = State::LOST;
    }
  }
  else if (tracker_state == State::TRACKING)
  {
    if (!matched)
    {
      tracker_state = State::TEMP_LOST;
      lost_count_++;
    }
  }
  else if (tracker_state == State::TEMP_LOST)
  {
    if (!matched)
    {
      lost_count_++;
      if (lost_count_ > lost_thres)
      {
        lost_count_ = 0;
        tracker_state = State::LOST;
      }
    }
    else
    {
      tracker_state = State::TRACKING;
      lost_count_ = 0;
    }
  }
}

// 切换EKF参数
void Tracker::SwitchEKFParams() { switch_q_(is_outpost); }

// 初始化ekf
void Tracker::InitEkf(const Armor& armor)
{
  first_tracked = true;
  is_outpost = (armor.number == "outpost");
  SwitchEKFParams();
  double xa = armor.pose.position.x;
  double ya = armor.pose.position.y;
  double za = armor.pose.position.z;
  double yaw = OrientationToYaw(armor.pose.orientation);

  // 设置初始位置在目标后面0.2米
  target_state = Eigen::VectorXd::Zero(9);
  double r = is_outpost ? outpost_r : 0.26;
  double xc = xa + r * cos(yaw);
  double yc = ya + r * sin(yaw);
  dz = 0, another_r = r;
  target_state << xc, 0, yc, 0, za, 0, yaw, 0, r;

  ekf.setState(target_state);
}

void Tracker::HandleArmorJump(const Armor& current_armor)
{
  if (first_tracked)
  {
    first_tracked = false;
    last_tracked_armor = current_armor;
  }

  auto position = current_armor.pose.position;
  auto orientation = current_armor.pose.orientation;

  double yaw = OrientationToYaw(orientation);

  // 更新追踪目标的装甲板数量
  UpdateArmorsNum(current_armor);

  // 更新跳变后的状态
  UpdateJumpedState(position, yaw);

  RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "Armor jump!");

  // 如果位置差大于 max_match_distance_，
  // 将此情况视为 EKF 发散，重置状态。
  Eigen::Vector3d current_p(position.x, position.y, position.z);
  Eigen::Vector3d infer_p = GetArmorPositionFromState(target_state);
  if ((current_p - infer_p).norm() > max_match_distance_)
  {
    ResetState(yaw, position);
  }

  last_tracked_armor = current_armor;
  ekf.setState(target_state);
}

void Tracker::SoftBreakEKF(const double y_pri, const double y_mea)
{
  int sign = (target_state(3) > 0) - (target_state(3) < 0);
  if ((y_pri - y_mea) * sign > 0.1)
  {
    y_diff_count_++;
  }
  else
  {
    y_diff_count_ = 0;
  }
  if (y_diff_count_ > 3)
  {
    RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "Soft break EKF!");
    target_state(3) -= 1.5 * target_state(3);  // v_yc
    ekf.setState(target_state);
    y_diff_count_ = 0;
  }
}

void Tracker::VelocityConstrain(double vx_max, double vy_max, double vz_max,
                                double vyaw_max, double yaw_coupling = 1.0)
{
  double& vx = target_state(1);
  double& vy = target_state(3);
  double& vz = target_state(5);
  double& vyaw = target_state(7);

  const double k = std::clamp(yaw_coupling, 0.0, 1.0);

  // 1) 先硬限制角速度，避免它本身离谱
  vyaw = std::clamp(vyaw, -vyaw_max, vyaw_max);

  // 2) 角速度先占掉一部分预算
  //    k = 0   -> 不耦合
  //    k = 1   -> 完全按椭球预算耦合
  const double yaw_ratio = std::abs(vyaw) / vyaw_max;
  const double trans_budget = std::sqrt(std::max(0.0, 1.0 - k * yaw_ratio * yaw_ratio));

  // 3) 计算当前平移速度占用了多少预算
  const double trans_ratio =
      std::sqrt((vx * vx) / (vx_max * vx_max) + (vy * vy) / (vy_max * vy_max) +
                (vz * vz) / (vz_max * vz_max));

  // 4) 如果超了，就把平移部分按比例压回预算内
  if (trans_ratio > trans_budget && trans_ratio > 0)
  {
    const double scale = trans_budget / trans_ratio;
    vx *= scale;
    vy *= scale;
    vz *= scale;
  }
}

void Tracker::UpdateArmorsNum(const Armor& armor)
{
  if (armor.number == "outpost")
  {
    tracked_armors_num = ArmorsNum::OUTPOST_3;  // 前哨站
  }
  else
  {
    tracked_armors_num = ArmorsNum::NORMAL_4;
  }
}

void Tracker::ResetState(double& yaw, const geometry_msgs::msg::Point& p)
{
  double r = target_state(8);
  target_state(0) = p.x + r * cos(yaw);  // xc
  target_state(1) = 0;                   // vxc
  target_state(2) = p.y + r * sin(yaw);  // yc
  target_state(3) = 0;                   // vyc
  target_state(4) = p.z;                 // za
  target_state(5) = 0;                   // vza
  if (tracked_armors_num == ArmorsNum::OUTPOST_3)
  {
    outpost_idx = 0;
  }
  RCLCPP_ERROR(rclcpp::get_logger("armor_tracker"), "Reset State!");
}

// 处理装甲板 **跳变**
void Tracker::UpdateJumpedState(const geometry_msgs::msg::Point& position, double yaw)
{
  // 更新yaw
  target_state(6) = yaw;

  // 对地面兵种应用不同半径与装甲板高度差
  if (tracked_armors_num == ArmorsNum::NORMAL_4)
  {
    dz = target_state(4) - position.z;
    std::swap(target_state(8), another_r);

    target_state(4) = position.z;
  }
  // 借助跳变更新前哨站索引
  else if (tracked_armors_num == ArmorsNum::OUTPOST_3)
  {
    double z_diff = last_tracked_armor.pose.position.z - position.z;
    // 可能不保证卡尔曼里是哪个装甲板的高度，但总是两个相邻的装甲板相减
    // 基于高——低装甲板高度差强行矫正索引
    // 低 中 高： 0 1 2
    int sign = target_state(7) > 0 ? 1 : -1;
    if (std::fabs(z_diff) > outpost_cast_threshold)
    {
      outpost_idx = (sign == 1) ? 0 : 2;
    }
    else
    {
      outpost_idx = (outpost_idx + sign + 3) % 3;
    }
    RCLCPP_INFO(rclcpp::get_logger("armor_tracker"),
                "Outpost Jump: z_diff=%.3f, current_idx=%d", z_diff, outpost_idx);
  }
}

// 姿态转换成偏航角(yaw)
double Tracker::OrientationToYaw(const geometry_msgs::msg::Quaternion& q)
{
  // Get armor yaw
  tf2::Quaternion tf_q;
  tf2::fromMsg(q, tf_q);
  double roll = NAN, pitch = NAN, yaw = NAN;
  tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
  // Make yaw change continuous (-pi~pi to -inf~inf)
  yaw = last_yaw_ + angles::shortest_angular_distance(last_yaw_, yaw);
  last_yaw_ = yaw;
  return yaw;
}

// 三维计算,根据我们一开始的装甲板解算得到的预测装甲板位置
Eigen::Vector3d Tracker::GetArmorPositionFromState(const Eigen::VectorXd& x)
{
  // 计算当前装甲板的预测位置
  double xc = x(0), yc = x(2), za = x(4);
  double yaw = x(6), r = x(8);
  double xa = xc - r * cos(yaw);
  double ya = yc - r * sin(yaw);
  return Eigen::Vector3d(xa, ya, za);
}

}  // namespace rm_auto_aim