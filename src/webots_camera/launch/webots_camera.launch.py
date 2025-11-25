import os
import launch
from launch import LaunchDescription
from launch.actions import RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController


def generate_launch_description():
    # --- 配置路径 ---

    # # 1. World 文件所在的包名 (根据你的路径是 webots_driver)
    # world_package_name = 'webots_driver'
    # # 2. 真实的 World 文件名
    # world_file_name = 'auto_aim_test_field.wbt'

    # # 3. URDF 文件所在的包名 (当前包 webots_camera)
    camera_package_name = 'webots_camera'

    # # 获取路径
    # world_pkg_share = get_package_share_directory(world_package_name)
    camera_pkg_share = get_package_share_directory(camera_package_name)

    # # 拼接 World 文件的完整路径
    # # 注意：根据你的源路径 src/webots_driver/webots/worlds/...
    # # 安装后路径通常是 share/webots_driver/webots/worlds/...
    # world_path = os.path.join(
    #     world_pkg_share, 'src', world_package_name, 'webots', 'worlds', world_file_name)

    # # 拼接 URDF 文件的完整路径
    robot_description_path = os.path.join(
        camera_pkg_share, 'urdf', 'webots_camera.urdf')

    # # 检查文件是否存在（可选，方便调试）
    # if not os.path.exists(world_path):
    #     print(f"[ERROR] World file not found at: {world_path}")

    # # --- 启动配置 ---

    # # 启动 Webots
    # webots = WebotsLauncher(
    #     world=world_path,
    #     ros2_supervisor=True
    # )

    # 启动控制器
    my_camera_driver = WebotsController(
        robot_name='self',
        parameters=[
            {'robot_description': robot_description_path},
        ]
    )

    return LaunchDescription([
        my_camera_driver,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                on_exit=[EmitEvent(event=Shutdown())],
            )
        )
    ])
