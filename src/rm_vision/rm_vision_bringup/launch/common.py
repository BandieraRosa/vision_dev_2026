import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.descriptions import ComposableNode

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

# 先保留。若你坚持“绝对全部同进程”，这个也得组件化或移除。
robot_state_publisher = Node(
    package="robot_state_publisher",
    executable="robot_state_publisher",
    parameters=[{"robot_description": robot_description, "publish_frequency": 1000.0}],
)

node_params = os.path.join(
    get_package_share_directory("rm_vision_bringup"), "config", "node_params.yaml"
)


def get_camera_component(robot_type="default"):
    if launch_params["camera"] == "hik":
        return ComposableNode(
            package="hik_camera",
            plugin="HikCamera::HikCameraNode",
            name="camera_node",
            parameters=[node_params, {"robot_type": robot_type}],
            extra_arguments=[{"use_intra_process_comms": True}],
        )
    elif launch_params["camera"] == "mv":
        return ComposableNode(
            package="mindvision_camera",
            plugin="mindvision_camera::MVCameraNode",
            name="camera_node",
            parameters=[node_params, {"robot_type": robot_type}],
            extra_arguments=[{"use_intra_process_comms": True}],
        )
    else:
        raise RuntimeError(f"Unknown camera type: {launch_params['camera']}")


def get_detector_component(robot_type="default"):
    return ComposableNode(
        package="armor_detector",
        plugin="rm_auto_aim::ArmorDetectorNode",
        name="armor_detector",
        parameters=[node_params, {"robot_type": robot_type}],
        extra_arguments=[{"use_intra_process_comms": True}],
    )


def get_tracker_component(robot_type="default"):
    return ComposableNode(
        package="armor_tracker",
        plugin="rm_auto_aim::ArmorTrackerNode",
        name="armor_tracker",
        parameters=[node_params, {"robot_type": robot_type}],
        extra_arguments=[{"use_intra_process_comms": True}],
    )


def get_trajectory_component(robot_type="default"):
    return ComposableNode(
        package="planning_trajectory",
        plugin="rm_auto_aim::PlanningTrajectoryNode",
        name="planning_trajectory",
        parameters=[node_params, {"robot_type": robot_type}],
        extra_arguments=[{"use_intra_process_comms": True}],
    )


def get_serial_component(robot_type="default"):
    return ComposableNode(
        package="rm_serial_driver",
        plugin="rm_serial_driver::RMSerialDriver",
        name="serial_driver",
        parameters=[node_params, {"robot_type": robot_type}],
        extra_arguments=[{"use_intra_process_comms": True}],
    )


def get_marker_component(robot_type="default"):
    return ComposableNode(
        package="armor_marker",
        plugin="rm_auto_aim::ArmorMarkerNode",
        name="armor_marker",
        parameters=[node_params, {"robot_type": robot_type}],
        extra_arguments=[{"use_intra_process_comms": True}],
    )
