import os
import sys
from ament_index_python.packages import get_package_share_directory

sys.path.append(os.path.join(
    get_package_share_directory('rm_vision_bringup'), 'launch'))


def generate_launch_description():
    from common import node_params, launch_params, robot_state_publisher

    from launch_ros.actions import Node, ComposableNodeContainer
    from launch_ros.descriptions import ComposableNode
    from launch_ros.actions import ComposableNodeContainer, Node
    from launch.actions import TimerAction, Shutdown
    from launch import LaunchDescription

    hik_camera_node = ComposableNode(
        package='hik_camera',
        plugin='HikCamera::HikCameraNode',
        name='camera_node',
        parameters=[node_params],
        extra_arguments=[{'use_intra_process_comms': True}],
    )

    hik_camera_container = ComposableNodeContainer(
        name='hik_camera_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[hik_camera_node],
        output='both',
        emulate_tty=True,
        on_exit=Shutdown(),
    )

    serial_driver_node = Node(
        package='rm_serial_driver',
        executable='rm_serial_driver_node',
        name='serial_driver',
        output='both',
        emulate_tty=True,
        parameters=[node_params],
        on_exit=Shutdown(),
        ros_arguments=['--ros-args', '--log-level',
                       'serial_driver:=' + launch_params['serial_log_level']],
    )
    delay_serial_node = TimerAction(
        period=1.5,
        actions=[serial_driver_node],
    )

    hand_eye_node = Node(
        package='rm_hand_eye_calibrate',
        executable='rm_hand_eye_calibrate_node',
        name='hand_eye_calibrate_node',
        output='both',
        emulate_tty=True,
        parameters=[node_params],
        on_exit=Shutdown(),
    )

    delay_hand_eye_node = TimerAction(
        period=2.0,
        actions=[hand_eye_node],
    )

    return LaunchDescription([
        robot_state_publisher,
        hik_camera_container,
        delay_serial_node,
        delay_hand_eye_node,
    ])
