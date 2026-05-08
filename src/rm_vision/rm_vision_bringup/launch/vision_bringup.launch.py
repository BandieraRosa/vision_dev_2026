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
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer


sys.path.append(
    os.path.join(get_package_share_directory("rm_vision_bringup"), "launch")
)


# def _build_after_checkout(context, *args, **kwargs):
#     from common import (
#         launch_params,
#         robot_state_publisher,
#         get_camera_component,
#         get_detector_component,
#         get_tracker_component,
#         get_trajectory_component,
#         get_serial_component,
#         get_marker_component,
#     )

#     robot_type = LaunchConfiguration("robot").perform(context)

#     vision_container = ComposableNodeContainer(
#         name="vision_container",
#         namespace="",
#         package="rclcpp_components",
#         executable="component_container_mt",
#         composable_node_descriptions=[
#             get_camera_component(robot_type),
#             get_detector_component(robot_type),
#             get_tracker_component(robot_type),
#             get_trajectory_component(robot_type),
#             get_serial_component(robot_type),
#             get_marker_component(robot_type),
#         ],
#         output="both",
#         emulate_tty=True,
#         parameters=[
#             {"thread_num": os.cpu_count()},
#         ],
#         ros_arguments=[
#             "--ros-args",
#             "--log-level",
#             "armor_detector:=" + launch_params["detector_log_level"],
#             "--log-level",
#             "armor_tracker:=" + launch_params["tracker_log_level"],
#             "--log-level",
#             "planning_trajectory:=" + launch_params.get("trajectory_log_level", "INFO"),
#             "--log-level",
#             "serial_driver:=" + launch_params["serial_log_level"],
#         ],
#         on_exit=Shutdown(),
#     )

#     # vision_container = ComposableNodeContainer(
#     #     name="vision_container",
#     #     namespace="",
#     #     package="rclcpp_components",
#     #     executable="component_container",
#     #     composable_node_descriptions=[
#     #         get_camera_component(robot_type),
#     #         get_detector_component(robot_type),
#     #         get_tracker_component(robot_type),
#     #         get_marker_component(robot_type),
#     #     ],
#     # )

#     # control_container = ComposableNodeContainer(
#     #     name="control_container",
#     #     namespace="",
#     #     package="rclcpp_components",
#     #     executable="component_container_mt",
#     #     composable_node_descriptions=[
#     #         get_trajectory_component(robot_type),
#     #         get_serial_component(robot_type),
#     #     ],
#     # )

#     return [
#         robot_state_publisher,
#         vision_container,
#         # control_container,
#     ]

# 针对轮腿nuc上的特殊优化，将视觉节点和控制节点分别放到不同的CPU上
def _build_after_checkout(context, *args, **kwargs):
    from common import (
        launch_params,
        robot_state_publisher,
        get_camera_component,
        get_detector_component,
        get_tracker_component,
        get_trajectory_component,
        get_serial_component,
        get_marker_component,
    )

    robot_type = LaunchConfiguration("robot").perform(context)

    # 普通视觉节点：CPU4-6
    vision_container = ComposableNodeContainer(
        name="vision_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        prefix="taskset -c 4-6",
        composable_node_descriptions=[
            get_camera_component(robot_type),
            get_detector_component(robot_type),
            get_tracker_component(robot_type),
            get_serial_component(robot_type),
            get_marker_component(robot_type),
        ],
        output="both",
        emulate_tty=True,
        parameters=[
            {"thread_num": os.cpu_count() - 2},
        ],
        ros_arguments=[
            "--ros-args",
            "--log-level",
            "armor_detector:=" + launch_params["detector_log_level"],
            "--log-level",
            "armor_tracker:=" + launch_params["tracker_log_level"],
            "--log-level",
            "serial_driver:=" + launch_params["serial_log_level"],
        ],
        on_exit=Shutdown(),
    )

    # trajectory 单独容器：CPU7 + SCHED_FIFO 80
    trajectory_container = ComposableNodeContainer(
        name="trajectory_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        prefix="chrt -f 80 taskset -c 7",
        composable_node_descriptions=[
            get_trajectory_component(robot_type),
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
        robot_state_publisher,
        vision_container,
        trajectory_container,
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
            'cd "$WS_ROOT"/' + config_repo_rel + "; "
            "git rev-parse --is-inside-work-tree >/dev/null 2>&1; "
            'git checkout "$BRANCH"; '
            "git status --porcelain",
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
            DeclareLaunchArgument("ws_root", default_value=os.getcwd()),
            DeclareLaunchArgument("robot", default_value=""),
            checkout_robot,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=checkout_robot,
                    on_exit=[build_nodes],
                )
            ),
        ]
    )
