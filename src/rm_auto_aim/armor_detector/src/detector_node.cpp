
#include "armor_detector/detector_node.hpp"

#include <cv_bridge/cv_bridge.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <image_transport/image_transport.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/create_timer_ros.hpp>

namespace rm_auto_aim
{
ArmorDetectorNode::ArmorDetectorNode(const rclcpp::NodeOptions& options)
    : Node("armor_detector", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting DetectorNode!");

  // Detector
  detector_ = InitDetector();

  // Light corner corrector
  corner_corrector_ = InitLightCornerCorrector();

  // Pose optimizer
  pose_optimizer_ = InitPoseOptimizer();
  if (pose_optimizer_)
  {
    RCLCPP_INFO(this->get_logger(), "Pose optimizer enabled.");
    InitTransformListener();
  }

  // Armors Publisher
  armors_pub_ = this->create_publisher<auto_aim_interfaces::msg::Armors>(
      "/detector/armors", rclcpp::SensorDataQoS());

  // Visualization Marker Publisher
  // See http://wiki.ros.org/rviz/DisplayTypes/Marker
  armor_marker_.ns = "armors";
  armor_marker_.action = visualization_msgs::msg::Marker::ADD;
  armor_marker_.type = visualization_msgs::msg::Marker::CUBE;
  armor_marker_.scale.x = 0.05;
  armor_marker_.scale.z = 0.125;
  armor_marker_.color.a = 1.0;
  armor_marker_.color.g = 0.5;
  armor_marker_.color.b = 1.0;
  armor_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);

  text_marker_.ns = "classification";
  text_marker_.action = visualization_msgs::msg::Marker::ADD;
  text_marker_.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  text_marker_.scale.z = 0.1;
  text_marker_.color.a = 1.0;
  text_marker_.color.r = 1.0;
  text_marker_.color.g = 1.0;
  text_marker_.color.b = 1.0;
  text_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);

  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/detector/marker", 10);

  // Debug Publishers
  debug_ = this->declare_parameter("debug", false);
  if (debug_)
  {
    CreateDebugPublishers();
  }

  // Debug param change monitor
  debug_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
  debug_cb_handle_ = debug_param_sub_->add_parameter_callback(
      "debug",
      [this](const rclcpp::Parameter& p)
      {
        debug_ = p.as_bool();
        debug_ ? CreateDebugPublishers() : DestroyDebugPublishers();
      });

  // 创建相机内参订阅者 → 接收一次/camera_info消息 → 提取图像中心、保存内参、初始化 PnP
  // 求解器 → 停止订阅
  auto robot_type = this->declare_parameter<std::string>("robot_type", "default");

  cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      "/camera_info", rclcpp::SensorDataQoS(),
      [this, robot_type](const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info)
      {
        if (current_frame_id_ != "" && camera_info->header.frame_id == current_frame_id_)
        {
          return;
        }
        cam_center_ = cv::Point2f(static_cast<float>(camera_info->k[2]),
                                  static_cast<float>(camera_info->k[5]));
        cam_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
        if (pose_optimizer_)
        {
          pose_optimizer_->SetCameraIntrinsics(
              cv::Mat(3, 3, CV_64F, const_cast<double*>(cam_info_->k.data())),
              cv::Mat(cam_info_->d));
        }
        if (!pnp_solver_)
        {
          pnp_solver_ = InitPnPSolver();
          if (robot_type != "hero")
          {
            cam_info_sub_.reset();  // 只需要接收第一条相机内参消息，之后就可以停掉订阅了
          }
        }
        else
        {
          pnp_solver_->SetCameraInfo(cam_info_->k, cam_info_->d);
        }
        current_frame_id_ = camera_info->header.frame_id;
        RCLCPP_INFO(this->get_logger(), "PnP solver updated (frame_id: %s)",
                    camera_info->header.frame_id.c_str());
      });

  img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw", rclcpp::SensorDataQoS(),
      std::bind(&ArmorDetectorNode::ImageCallback, this, std::placeholders::_1));
}

void ArmorDetectorNode::ImageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr& img_msg)
{
  if (!pnp_solver_)
  {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                         "Waiting for /camera_info, skip current image.");
    return;
  }

  auto armors = DetectArmors(img_msg);

  if (pnp_solver_ != nullptr)
  {
    armors_msg_.header = armor_marker_.header = text_marker_.header = img_msg->header;
    armors_msg_.armors.clear();
    marker_array_.markers.clear();
    armor_marker_.id = 0;
    text_marker_.id = 0;

    auto_aim_interfaces::msg::Armor armor_msg;
    for (const auto& armor : armors)
    {
      cv::Mat rvec, tvec;
      bool success = pnp_solver_->SolvePnP(armor, rvec, tvec);
      if (success)
      {
        if (pose_optimizer_)
        {
          try
          {
            auto odom_to_camera_optical = tf2_buffer_->lookupTransform(
                "gimbal_odom", img_msg->header.frame_id, img_msg->header.stamp,
                rclcpp::Duration::from_seconds(0.01));

            Eigen::Quaterniond q;
            tf2::fromMsg(odom_to_camera_optical.transform.rotation, q);
            Eigen::Matrix3d rmat_gimbal_cam = q.toRotationMatrix();

            auto odom_to_gimbal = tf2_buffer_->lookupTransform(
                "gimbal_odom", "pitch_link", img_msg->header.stamp,
                rclcpp::Duration::from_seconds(0.01));

            tf2::Quaternion qu(
                odom_to_gimbal.transform.rotation.x, odom_to_gimbal.transform.rotation.y,
                odom_to_gimbal.transform.rotation.z, odom_to_gimbal.transform.rotation.w);
            tf2::Matrix3x3 m(qu);
            double roll = NAN, pitch = NAN, yaw = NAN;
            m.getRPY(roll, pitch, yaw);

            pose_optimizer_->SetCameraToGimbalRotation(rmat_gimbal_cam);

            // TF 查询成功后才执行优化；失败时 rvec/tvec 保持原始 PnP 结果不变
            if (!pose_optimizer_->Optimize(armor, rvec, tvec))
            {
              RCLCPP_DEBUG(this->get_logger(),
                           "Pose optimization failed, using original PnP result.");
            }
          }
          catch (const tf2::TransformException& ex)
          {
            RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s", ex.what());
          }
        }
        // Fill basic info
        armor_msg.type = ARMOR_TYPE_STR[static_cast<int>(armor.type)];
        armor_msg.number = armor.number;

        // Fill pose
        armor_msg.pose.position.x = tvec.at<double>(0);
        armor_msg.pose.position.y = tvec.at<double>(1);
        armor_msg.pose.position.z = tvec.at<double>(2);
        // rvec to 3x3 rotation matrix
        cv::Mat rotation_matrix;
        cv::Rodrigues(rvec, rotation_matrix);
        // rotation matrix to quaternion
        tf2::Matrix3x3 tf2_rotation_matrix(
            rotation_matrix.at<double>(0, 0), rotation_matrix.at<double>(0, 1),
            rotation_matrix.at<double>(0, 2), rotation_matrix.at<double>(1, 0),
            rotation_matrix.at<double>(1, 1), rotation_matrix.at<double>(1, 2),
            rotation_matrix.at<double>(2, 0), rotation_matrix.at<double>(2, 1),
            rotation_matrix.at<double>(2, 2));
        tf2::Quaternion tf2_q;
        tf2_rotation_matrix.getRotation(tf2_q);
        armor_msg.pose.orientation = tf2::toMsg(tf2_q);

        // Fill the distance to image center
        armor_msg.distance_to_image_center =
            pnp_solver_->CalculateDistanceToCenter(armor.center);

        // Fill the markers
        armor_marker_.id++;
        armor_marker_.scale.y = armor.type == ArmorType::SMALL ? 0.135 : 0.23;
        armor_marker_.pose = armor_msg.pose;
        text_marker_.id++;
        text_marker_.pose.position = armor_msg.pose.position;
        text_marker_.pose.position.y -= 0.1;
        text_marker_.text = armor.classfication_result;
        armors_msg_.armors.emplace_back(armor_msg);
        marker_array_.markers.emplace_back(armor_marker_);
        marker_array_.markers.emplace_back(text_marker_);
      }
      else
      {
        RCLCPP_WARN(this->get_logger(), "PnP failed!");
      }
    }

    // Publishing detected armors
    armors_pub_->publish(armors_msg_);

    // Publishing marker
    PublishMarkers();
  }
}

std::unique_ptr<Detector> ArmorDetectorNode::InitDetector()
{
  rcl_interfaces::msg::ParameterDescriptor param_desc;
  param_desc.integer_range.resize(1);
  param_desc.integer_range[0].step = 1;
  param_desc.integer_range[0].from_value = 0;
  param_desc.integer_range[0].to_value = 255;
  int binary_lower_thres =
      static_cast<int>(declare_parameter("binary_lower_thres", 160, param_desc));
  int binary_upper_thres =
      static_cast<int>(declare_parameter("binary_upper_thres", 255, param_desc));

  param_desc.description = "0-RED, 1-BLUE";
  param_desc.integer_range[0].from_value = 0;
  param_desc.integer_range[0].to_value = 1;
  auto detect_color = declare_parameter("detect_color", RED, param_desc);

  Detector::LightParams l_params = {
      .min_ratio = declare_parameter("light.min_ratio", 0.1),
      .max_ratio = declare_parameter("light.max_ratio", 0.4),
      .max_angle = declare_parameter("light.max_angle", 40.0)};

  Detector::ArmorParams a_params = {
      .min_light_ratio = declare_parameter("armor.min_light_ratio", 0.7),
      .min_small_center_distance =
          declare_parameter("armor.min_small_center_distance", 0.8),
      .max_small_center_distance =
          declare_parameter("armor.max_small_center_distance", 3.2),
      .min_large_center_distance =
          declare_parameter("armor.min_large_center_distance", 3.2),
      .max_large_center_distance =
          declare_parameter("armor.max_large_center_distance", 5.5),
      .max_angle = declare_parameter("armor.max_angle", 35.0)};

  auto detector = std::make_unique<Detector>(binary_lower_thres, binary_upper_thres,
                                             detect_color, l_params, a_params);

  // Init classifier
  auto pkg_path = ament_index_cpp::get_package_share_directory("armor_detector");
  auto model_path = pkg_path + "/model/mlp.onnx";
  auto label_path = pkg_path + "/model/label.txt";
  double threshold = this->declare_parameter("classifier_threshold", 0.7);
  std::vector<std::string> ignore_classes =
      this->declare_parameter("ignore_classes", std::vector<std::string>{"negative"});
  detector->classifier = std::make_unique<NumberClassifier>(model_path, label_path,
                                                            threshold, ignore_classes);

  return detector;
}

std::unique_ptr<LightCornerCorrector> ArmorDetectorNode::InitLightCornerCorrector()
{
  bool use_corner_corrector =
      static_cast<bool>(this->declare_parameter("use_corner_corrector", false));
  if (use_corner_corrector)
  {
    // Light corner corrector
    return std::make_unique<LightCornerCorrector>();
  }
  return nullptr;
}

std::unique_ptr<PnPSolver> ArmorDetectorNode::InitPnPSolver()
{
  bool use_new_pnp_filter_method =
      this->declare_parameter("pnp_filter.use_new_pnp_filter_method", false);
  PnPSolver::PnpFilterParams pnp_filter_params;
  pnp_filter_params.new_pnp_filter_method = use_new_pnp_filter_method;
  pnp_filter_params.max_normal_dot =
      this->declare_parameter("pnp_filter.max_normal_dot", 0.0);
  pnp_filter_params.reproj_weight =
      this->declare_parameter("pnp_filter.reproj_weight", 1.0);
  pnp_filter_params.normal_weight =
      this->declare_parameter("pnp_filter.normal_weight", 0.5);
  return std::make_unique<PnPSolver>(cam_info_->k, cam_info_->d, pnp_filter_params);
}

std::unique_ptr<ArmorPoseOptimizer> ArmorDetectorNode::InitPoseOptimizer()
{
  bool use_pose_optimize = this->declare_parameter("use_pose_optimizer", false);
  if (!use_pose_optimize)
  {
    return nullptr;
  }
  ArmorPoseOptimizer::Params opt_params;
  opt_params.standard_pitch_deg =
      this->declare_parameter("optimizer.standard_pitch_deg", 15.0);
  opt_params.outpost_pitch_deg =
      this->declare_parameter("optimizer.outpost_pitch_deg", -15.0);
  opt_params.max_pitch_deviation =
      this->declare_parameter("optimizer.max_pitch_deviation", 15.0);
  opt_params.max_roll_deviation =
      this->declare_parameter("optimizer.max_roll_deviation", 15.0);
  opt_params.max_iterations =
      static_cast<int>(this->declare_parameter("optimizer.max_iterations", 20));

  std::string optimize_method_str =
      this->declare_parameter("optimizer.optimize_method", std::string("RANGE_SHORT_LM"));
  if (optimize_method_str == "LM")
  {
    opt_params.optimize_method = ArmorPoseOptimizer::Params::OptimizeMethod::LM;
  }
  else if (optimize_method_str == "RANGE")
  {
    opt_params.optimize_method = ArmorPoseOptimizer::Params::OptimizeMethod::RANGE;
  }
  else if (optimize_method_str == "RANGE_SHORT_LM")
  {
    opt_params.optimize_method = ArmorPoseOptimizer::Params::OptimizeMethod::RANGE_LM;
  }
  opt_params.range_search_half_range_deg =
      this->declare_parameter("optimizer.range_search_half_range_deg", 70.0);
  opt_params.range_search_coarse_step_deg =
      this->declare_parameter("optimizer.range_search_coarse_step_deg", 1.0);
  opt_params.range_search_fine_range_deg =
      this->declare_parameter("optimizer.range_search_fine_range_deg", 2.0);
  opt_params.range_search_fine_step_deg =
      this->declare_parameter("optimizer.range_search_fine_step_deg", 0.1);

  return std::make_unique<ArmorPoseOptimizer>(opt_params);
}

void ArmorDetectorNode::InitTransformListener()
{
  odom_frame_ = this->declare_parameter("target_frame", "gimbal_odom");
  tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
      this->get_node_base_interface(), this->get_node_timers_interface());
  tf2_buffer_->setCreateTimerInterface(timer_interface);
  tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);
}

std::vector<Armor> ArmorDetectorNode::DetectArmors(
    const sensor_msgs::msg::Image::ConstSharedPtr& img_msg)
{
  // Convert ROS img to cv::Mat
  auto img = cv_bridge::toCvShare(img_msg, "rgb8")->image;

  // Update params
  // detector_->binary_thres = static_cast<int>(get_parameter("binary_thres").as_int());
  detector_->binary_upper_thres_ =
      static_cast<int>(get_parameter("binary_upper_thres").as_int());
  detector_->binary_lower_thres_ =
      static_cast<int>(get_parameter("binary_lower_thres").as_int());
  detector_->detect_color = static_cast<int>(get_parameter("detect_color").as_int());
  detector_->classifier->SetThreshold(get_parameter("classifier_threshold").as_double());

  auto armors = detector_->Detect(img);

  // Correct the corners of the detected armors
  if (corner_corrector_ != nullptr)
  {
    for (auto& armor : armors)
    {
      corner_corrector_->CorrectCorners(armor, detector_->binary_img);
    }
  }

  auto final_time = this->now();
  auto latency = (final_time - img_msg->header.stamp).seconds() * 1000;
  RCLCPP_DEBUG_STREAM(this->get_logger(), "Latency: " << latency << "ms");

  // Publish debug info
  if (debug_)
  {
    binary_img_pub_.publish(
        cv_bridge::CvImage(img_msg->header, "mono8", detector_->binary_img).toImageMsg());

    // Sort lights and armors data by x coordinate
    std::sort(detector_->debug_lights.data.begin(), detector_->debug_lights.data.end(),
              [](const auto& l1, const auto& l2) { return l1.center_x < l2.center_x; });
    std::sort(detector_->debug_armors.data.begin(), detector_->debug_armors.data.end(),
              [](const auto& a1, const auto& a2) { return a1.center_x < a2.center_x; });

    lights_data_pub_->publish(detector_->debug_lights);
    armors_data_pub_->publish(detector_->debug_armors);

    if (!armors.empty())
    {
      auto all_num_img = detector_->GetAllNumbersImage();
      number_img_pub_.publish(
          *cv_bridge::CvImage(img_msg->header, "mono8", all_num_img).toImageMsg());
    }

    detector_->DrawResults(img);
    // Draw camera center
    cv::circle(img, cam_center_, 5, cv::Scalar(255, 0, 0), 2);
    // Draw latency
    std::stringstream latency_ss;
    latency_ss << "Latency: " << std::fixed << std::setprecision(2) << latency << "ms";
    auto latency_s = latency_ss.str();
    cv::putText(img, latency_s, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(0, 255, 0), 2);
    result_img_pub_.publish(
        cv_bridge::CvImage(img_msg->header, "rgb8", img).toImageMsg());
  }

  return armors;
}

void ArmorDetectorNode::CreateDebugPublishers()
{
  lights_data_pub_ = this->create_publisher<auto_aim_interfaces::msg::DebugLights>(
      "/detector/debug_lights", 10);
  armors_data_pub_ = this->create_publisher<auto_aim_interfaces::msg::DebugArmors>(
      "/detector/debug_armors", 10);

  binary_img_pub_ = image_transport::create_publisher(this, "/detector/binary_img");
  number_img_pub_ = image_transport::create_publisher(this, "/detector/number_img");
  result_img_pub_ = image_transport::create_publisher(this, "/detector/result_img");
}

void ArmorDetectorNode::DestroyDebugPublishers()
{
  lights_data_pub_.reset();  // 释放调试发布者所占用的资源，并将其置为无效状态
  armors_data_pub_.reset();

  binary_img_pub_.shutdown();  // 关闭调试发布者，使其不再发布任何消息
  number_img_pub_.shutdown();
  result_img_pub_.shutdown();
}

void ArmorDetectorNode::PublishMarkers()
{
  using Marker = visualization_msgs::msg::Marker;
  armor_marker_.action = armors_msg_.armors.empty() ? Marker::DELETE : Marker::ADD;
  marker_array_.markers.emplace_back(armor_marker_);
  marker_pub_->publish(marker_array_);
}

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its
// library is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorDetectorNode)