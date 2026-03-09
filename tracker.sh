# 设置开机自启
# 在终端中输入 gnome-session-properties
# 在弹出的“启动应用程序首选项”窗口中，点击“添加”，输入名称、命令和注释，然后点击“添加”完成设置。

#!/bin/bash

echo "正在清理旧进程..."
for proc in robot_state_publisher rm_serial_driver_node armor_tracker_node component_container; do
    if pgrep -f "$proc" > /dev/null; then
        echo "杀死 $proc"
        pkill -f "$proc"
        sleep 0.5
        pkill -9 -f "$proc" 2>/dev/null
    fi
done
echo "进程清理完成"
echo ""

delay_time=3
while true
do
    ps -ef | grep "armor_tracker_node" | grep -v "grep" > /dev/null
    if [ $? -ne 0 ]
    then
        echo "tracker未运行，等待 ${delay_time}秒 后启动..."
        sleep $delay_time
         gnome-terminal --title="tracker" -- bash -c "cd /home/nvidia/vision_dev/ && source install/setup.bash && ros2 launch rm_vision_bringup vision_bringup.launch.py"
        if [ $? -eq 0 ]
        then
            echo "tracker started successfully!"
        else
            echo "tracker failed to start!"
        fi
    else
        echo "tracker is running!"
    fi
    sleep $delay_time
done