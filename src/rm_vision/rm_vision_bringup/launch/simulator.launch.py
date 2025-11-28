import os
import sys
from ament_index_python.packages import get_package_share_directory
sys.path.append(os.path.join(
    get_package_share_directory('rm_vision_bringup'), 'launch'))


def generate_launch_description():

    from common import launch_params, node_params, tracker_node
    from launch_ros.actions import Node
    from launch.actions import TimerAction, Shutdown
    from launch import LaunchDescription

    detector_node = Node(
        package='armor_detector',
        executable='armor_detector_node',
        emulate_tty=True,
        output='both',
        parameters=[node_params],
        arguments=['--ros-args', '--log-level',
                   'armor_detector:='+launch_params['detector_log_level']],
    )

    simulator_driver_node = Node(
        package='rm_simulator_driver',
        executable='rm_simulator_driver_node',
        name='simulator_driver',
        output='both',
        emulate_tty=True,
        ros_arguments=['--ros-args', '--log-level',
                       ['simulator_driver:=', launch_params['simulator_log_level']]],
    )

    delay_simulator_node = TimerAction(
        period=1.5,
        actions=[simulator_driver_node],
    )

    delay_tracker_node = TimerAction(
        period=2.0,
        actions=[tracker_node],
    )

    return LaunchDescription([
        detector_node,
        delay_simulator_node,
        delay_tracker_node,
    ])
