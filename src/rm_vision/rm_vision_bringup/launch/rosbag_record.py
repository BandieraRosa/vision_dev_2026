import os
import re

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

ROSBAG_RECORD_TOPICS = (
    # Compressed image topics for Foxglove-friendly playback
    "/image_raw/compressed",
    # "/detector/result_img/compressed",
    # Keep this alias for launch/remap setups that publish result_img at root.
    # "/result_img/compressed",
    # Camera metadata and TF
    "/camera_info",
    "/tf",
    "/tf_static",
    # Hardware / serial / control topics
    "/joint_states",
    "/camera_switch_done",
    "/lob_shot_switch",
    "/reset",
    "/current_velocity",
    # Auto aim pipeline topics
    "/detector/armors",
    "/detector/debug_lights",
    "/detector/debug_armors",
    "/tracker/info",
    "/tracker/target",
    "/trajectory/send",
    "/trajectory/info",
    # RViz marker topics
    "/detector/marker",
    "/tracker/marker",
    # Simulator topics. These simply do not match anything on real hardware.
    "/ground_truth/armors",
    "/ground_truth/noisy_armors",
    "/ground_truth/robot_pose",
)

ROSBAG_RECORD_REGEX = (
    "^(" + "|".join(re.escape(topic) for topic in ROSBAG_RECORD_TOPICS) + ")$"
)


def get_rosbag_record_actions(
    default_enabled: str = "true", bag_name_prefix: str = "vision"
) -> list:
    """Return launch actions that start ros2 bag recording for competition logs."""

    qos_profile = os.path.join(
        get_package_share_directory("rm_vision_bringup"),
        "config",
        "rosbag2_record_qos.yaml",
    )

    record_cmd = """
set -e
mkdir -p \"$BAG_OUTPUT_DIR\"
OUTPUT=\"${BAG_OUTPUT_DIR%/}/${BAG_NAME_PREFIX}_$(date +%Y%m%d_%H%M%S)\"

echo \"[rosbag] ROS 2 Humble-compatible recorder\"
echo \"[rosbag] Storage: ${BAG_STORAGE_ID}\"
echo \"[rosbag] Recording compressed-image and non-image topics only.\"
echo \"[rosbag] Regex: ${ROSBAG_RECORD_REGEX}\"
echo \"[rosbag] Output: ${OUTPUT}\"

if [ -n \"$ROSBAG_QOS_PROFILE\" ] && [ -f \"$ROSBAG_QOS_PROFILE\" ]; then
  exec ros2 bag record \\
    -s \"$BAG_STORAGE_ID\" \\
    --qos-profile-overrides-path \"$ROSBAG_QOS_PROFILE\" \\
    --regex \"$ROSBAG_RECORD_REGEX\" \\
    -o \"$OUTPUT\"
else
  echo \"[rosbag] QoS override file not found, recording with rosbag2 defaults: ${ROSBAG_QOS_PROFILE}\" >&2
  exec ros2 bag record \\
    -s \"$BAG_STORAGE_ID\" \\
    --regex \"$ROSBAG_RECORD_REGEX\" \\
    -o \"$OUTPUT\"
fi
"""

    return [
        DeclareLaunchArgument(
            "record_bag",
            default_value=default_enabled,
            description="Start ros2 bag recording when this launch starts.",
        ),
        DeclareLaunchArgument(
            "bag_output_dir",
            default_value=os.path.join(os.getcwd(), "bags"),
            description="Directory where rosbag2 output folders are created.",
        ),
        DeclareLaunchArgument(
            "bag_name_prefix",
            default_value=bag_name_prefix,
            description="Prefix for the generated rosbag2 output folder name.",
        ),
        DeclareLaunchArgument(
            "bag_storage_id",
            default_value="mcap",
            description="rosbag2 storage id. Default is mcap for Foxglove playback; set sqlite3 to use the ROS 2 default backend.",
        ),
        ExecuteProcess(
            condition=IfCondition(LaunchConfiguration("record_bag")),
            cmd=["bash", "-lc", record_cmd],
            additional_env={
                "BAG_OUTPUT_DIR": LaunchConfiguration("bag_output_dir"),
                "BAG_NAME_PREFIX": LaunchConfiguration("bag_name_prefix"),
                "BAG_STORAGE_ID": LaunchConfiguration("bag_storage_id"),
                "ROSBAG_QOS_PROFILE": qos_profile,
                "ROSBAG_RECORD_REGEX": ROSBAG_RECORD_REGEX,
            },
            output="screen",
        ),
    ]
