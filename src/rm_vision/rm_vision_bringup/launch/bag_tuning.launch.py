"""
bag_tuning.launch.py —— 使用 rosbag 离线调参

用途：
  - 不启动相机
  - 不启动串口
  - 不启动 robot_state_publisher
  - 从 bag 中播放:
      /image_raw/compressed
      /camera_info
      /tf
      /tf_static
  - 只启动:
      armor_detector
      armor_tracker
      planning_trajectory
      armor_marker

典型用法：

  ros2 launch rm_vision_bringup bag_tuning.launch.py \
    bag_path:=/path/to/your_bag \
    robot:=<你的机器人配置分支> \
    republish_image:=true \
    rate:=1.0

如果你已经提前提取了 sensor_tf 小包：

  ros2 launch rm_vision_bringup bag_tuning.launch.py \
    bag_path:=/path/to/your_bag \
    robot:=<你的机器人配置分支>
"""

import os
import sys

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

sys.path.append(
    os.path.join(get_package_share_directory("rm_vision_bringup"), "launch")
)


def _safe_taskset_prefix(cpu_list):
    """如果 CPU 列表可用则绑核，否则正常启动。"""
    cpu_list = str(cpu_list).strip()
    if not cpu_list:
        return None

    return (
        'bash -lc \'CPU_LIST="%s"; '
        "if command -v taskset >/dev/null 2>&1 && "
        'taskset -c "$CPU_LIST" true >/dev/null 2>&1; then '
        'exec taskset -c "$CPU_LIST" "$@"; '
        "else "
        'echo "[launch] skip taskset CPU_LIST=$CPU_LIST: unavailable or invalid on this machine" >&2; '
        'exec "$@"; '
        "fi' --"
    ) % cpu_list


def _build_after_checkout(context, *args, **kwargs):
    from common import launch_params, node_params

    robot_type = LaunchConfiguration("robot").perform(context)

    bag_path = LaunchConfiguration("bag_path")
    rate = LaunchConfiguration("rate")

    vision_cpu_list = launch_params.get("vision_cpu_list", "2-6")
    trajectory_cpu_list = launch_params.get("trajectory_cpu_list", "1")

    use_sim_time_param = {"use_sim_time": True}

    # -----------------------------
    # rosbag 播放
    # -----------------------------
    bag_play = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            "-s",
            "mcap",
            bag_path,
            "--clock",
            "--rate",
            rate,
            "--topics",
            "/image_raw/compressed",
            "/camera_info",
            "/tf",
            "/tf_static",
        ],
        output="screen",
    )

    # ------------------------------------------------------------
    # 如果 detector 订阅 sensor_msgs/msg/Image 的 /image_raw，
    # 则需要把 /image_raw/compressed 解码成 /image_raw。
    #
    # 如果你的 detector 本身直接订阅 CompressedImage，
    # 启动时传 republish_image:=false。
    # ------------------------------------------------------------
    image_republish = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration("republish_image")),
        cmd=[
            "ros2",
            "run",
            "image_transport",
            "republish",
            "compressed",
            "in/compressed:=/image_raw/compressed",
            "raw",
            "out:=/image_raw",
        ],
        output="screen",
    )

    # -----------------------------
    # detector
    # -----------------------------
    detector_component = ComposableNode(
        package="armor_detector",
        plugin="rm_auto_aim::ArmorDetectorNode",
        name="armor_detector",
        parameters=[
            node_params,
            {"robot_type": robot_type},
            use_sim_time_param,
        ],
        extra_arguments=[{"use_intra_process_comms": True}],
    )

    # -----------------------------
    # tracker
    # -----------------------------
    tracker_component = ComposableNode(
        package="armor_tracker",
        plugin="rm_auto_aim::ArmorTrackerNode",
        name="armor_tracker",
        parameters=[
            node_params,
            {"robot_type": robot_type},
            use_sim_time_param,
        ],
        extra_arguments=[{"use_intra_process_comms": True}],
    )

    # -----------------------------
    # marker
    # -----------------------------
    marker_component = ComposableNode(
        package="armor_marker",
        plugin="rm_auto_aim::ArmorMarkerNode",
        name="armor_marker",
        parameters=[
            node_params,
            {"robot_type": robot_type},
            use_sim_time_param,
        ],
        extra_arguments=[{"use_intra_process_comms": True}],
    )

    vision_container = ComposableNodeContainer(
        name="vision_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        prefix=_safe_taskset_prefix(vision_cpu_list),
        composable_node_descriptions=[
            detector_component,
            tracker_component,
            marker_component,
        ],
        output="both",
        emulate_tty=True,
        parameters=[
            {"thread_num": os.cpu_count()},
        ],
        ros_arguments=[
            "--ros-args",
            "--log-level",
            "armor_detector:=" + launch_params["detector_log_level"],
            "--log-level",
            "armor_tracker:=" + launch_params["tracker_log_level"],
        ],
        on_exit=Shutdown(),
    )

    # -----------------------------
    # planning_trajectory
    # 保留你 common.py 里的实时线程参数逻辑
    # -----------------------------
    trajectory_rt_defaults = {
        "rt.use_rt_thread": launch_params.get("trajectory_use_rt_thread", True),
        "rt.cpu": int(launch_params.get("trajectory_rt_cpu", 7)),
        "rt.priority": int(launch_params.get("trajectory_rt_priority", 80)),
        "rt.enable_cpu_affinity": launch_params.get(
            "trajectory_enable_cpu_affinity", True
        ),
        "rt.enable_realtime": launch_params.get("trajectory_enable_realtime", True),
        "rt.lock_memory": launch_params.get("trajectory_lock_memory", True),
        "rt.statistics_interval": int(
            launch_params.get("trajectory_rt_statistics_interval", 0)
        ),
    }

    trajectory_component = ComposableNode(
        package="planning_trajectory",
        plugin="rm_auto_aim::PlanningTrajectoryNode",
        name="planning_trajectory",
        parameters=[
            trajectory_rt_defaults,
            node_params,
            {"robot_type": robot_type},
            use_sim_time_param,
        ],
        extra_arguments=[{"use_intra_process_comms": True}],
    )

    trajectory_container = ComposableNodeContainer(
        name="trajectory_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        prefix=_safe_taskset_prefix(trajectory_cpu_list),
        composable_node_descriptions=[
            trajectory_component,
        ],
        output="both",
        emulate_tty=True,
        parameters=[
            {"thread_num": 2},
        ],
        ros_arguments=[
            "--ros-args",
            "--log-level",
            "planning_trajectory:=" + launch_params.get("trajectory_log_level", "INFO"),
        ],
        on_exit=Shutdown(),
    )

    return [
        vision_container,
        trajectory_container,
        image_republish,
        TimerAction(period=2.0, actions=[bag_play]),
    ]


def generate_launch_description():
    ws_root = LaunchConfiguration("ws_root")
    robot = LaunchConfiguration("robot")

    config_repo_rel = "src/rm_vision/rm_vision_bringup/config"

    checkout_robot = ExecuteProcess(
        cmd=[
            "bash",
            "-lc",
            "set -e; "
            'if [ -n "$BRANCH" ]; then '
            '  cd "$WS_ROOT"/' + config_repo_rel + "; "
            "  git rev-parse --is-inside-work-tree >/dev/null 2>&1; "
            '  git checkout "$BRANCH"; '
            "  git status --porcelain; "
            "else "
            '  echo "[launch] robot is empty, skip config git checkout"; '
            "fi",
        ],
        additional_env={
            "WS_ROOT": ws_root,
            "BRANCH": robot,
        },
        output="screen",
    )

    build_nodes = OpaqueFunction(function=_build_after_checkout)

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "ws_root",
                default_value=os.getcwd(),
                description="Workspace root, e.g. /home/rm/vision_dev",
            ),
            DeclareLaunchArgument(
                "robot",
                default_value="",
                description=(
                    "Optional config branch. "
                    "If empty, do not checkout config branch."
                ),
            ),
            DeclareLaunchArgument(
                "bag_path",
                default_value=(
                    "/home/rm/vision_dev/bags/" "vision_20260524_084333_indexed.mcap"
                ),
                description="Input MCAP file or rosbag2 directory.",
            ),
            DeclareLaunchArgument(
                "rate",
                default_value="1.0",
                description="rosbag playback rate.",
            ),
            DeclareLaunchArgument(
                "republish_image",
                default_value="true",
                description=(
                    "true: republish /image_raw/compressed to /image_raw. "
                    "false: detector directly uses /image_raw/compressed."
                ),
            ),
            checkout_robot,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=checkout_robot,
                    on_exit=[build_nodes],
                )
            ),
        ]
    )
