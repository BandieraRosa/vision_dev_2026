import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer

launch_params = yaml.safe_load(
    open(
        os.path.join(
            get_package_share_directory("rm_vision_bringup"),
            "config",
            "launch_params.yaml",
        )
    )
)

robot_description = Command(
    [
        "xacro ",
        os.path.join(
            get_package_share_directory("rm_gimbal_description"),
            "urdf",
            "rm_gimbal.urdf.xacro",
        ),
        " xyz:=",
        launch_params["odom2camera"]["xyz"],
        " rpy:=",
        launch_params["odom2camera"]["rpy"],
        " lob_xyz:=",
        launch_params["odom2camera"]["lob_xyz"],
        " lob_rpy:=",
        launch_params["odom2camera"]["lob_rpy"],
    ]
)

robot_state_publisher = Node(
    package="robot_state_publisher",
    executable="robot_state_publisher",
    parameters=[{"robot_description": robot_description, "publish_frequency": 1000.0}],
)

node_params = os.path.join(
    get_package_share_directory("rm_vision_bringup"), "config", "node_params.yaml"
)

tracker_node = Node(
    package="armor_tracker",
    executable="armor_tracker_node",
    output="both",
    emulate_tty=True,
    parameters=[node_params],
    ros_arguments=[
        "--log-level",
        "armor_tracker:=" + launch_params["tracker_log_level"],
    ],
)


def get_tracker_node(robot_type="default"):
    return Node(
        package="armor_tracker",
        executable="armor_tracker_node",
        output="both",
        emulate_tty=True,
        parameters=[node_params, {"robot_type": robot_type}],
        ros_arguments=[
            "--log-level",
            "armor_tracker:=" + launch_params["tracker_log_level"],
        ],
    )

def get_trajectory_node(robot_type="default"):
    return Node(
        package="planning_trajectory",
        executable="planning_trajectory_node",
        output="both",
        emulate_tty=True,
        parameters=[node_params, {"robot_type": robot_type}],
        ros_arguments=[
            "--log-level",
            "planning_trajectory:=" + launch_params["trajectory_log_level"],
        ],
    )

# def get_trajectory_node(robot_type="default"):
#     """
#     创建 PlanningTrajectoryNode 的 ComposableNodeContainer。
#     trajectory_node 是通过 RCLCPP_COMPONENTS_REGISTER_NODE 注册的组件节点，
#     订阅 /tracker/target 和 /current_velocity，
#     发布 /trajectory/send 和 /trajectory/info。
#     """
#     trajectory_log_level = launch_params.get("trajectory_log_level", "info")

#     return ComposableNodeContainer(
#         name="trajectory_container",
#         namespace="",
#         package="rclcpp_components",
#         executable="component_container",
#         composable_node_descriptions=[
#             ComposableNode(
#                 package="planning_trajectory",
#                 plugin="rm_auto_aim::PlanningTrajectoryNode",
#                 name="planning_trajectory",
#                 parameters=[node_params, {"robot_type": robot_type}],
#                 extra_arguments=[{"use_intra_process_comms": True}],
#             ),
#         ],
#         output="both",
#         emulate_tty=True,
#         ros_arguments=[
#             "--ros-args",
#             "--log-level",
#             "planning_trajectory:=" + trajectory_log_level,
#         ],
#     )

