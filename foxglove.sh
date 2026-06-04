#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")"

ROS_DISTRO="${ROS_DISTRO:-humble}"
source "/opt/ros/$ROS_DISTRO/setup.bash"

WORKSPACE_SETUP="install/setup.bash"
if [[ ! -f "$WORKSPACE_SETUP" ]]; then
    echo "确保当前目录是工作空间根目录，且已编译"
    exit 1
fi
source "$WORKSPACE_SETUP"

exec ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765