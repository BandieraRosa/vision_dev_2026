#!/bin/bash

delay_time=6

while true
do
    ps -ef | grep "armor_tracker_node" | grep -v "grep" > /dev/null
    
    if [ $? -ne 0 ]
    then
        echo "[$(date '+%H:%M:%S')] tracker未运行，清理旧进程..."
        
        # 清理冲突进程
        for proc in robot_state_publisher rm_serial_driver_node armor_tracker_node component_container; do
            if pgrep -f "$proc" > /dev/null; then
                echo "  杀死 $proc"
                pkill -f "$proc"
                sleep 0.5
                pkill -9 -f "$proc" 2>/dev/null
            fi
        done
        
        echo "等待 ${delay_time}秒 后启动..."
        sleep $delay_time
        
<<<<<<< HEAD
        gnome-terminal --title="tracker" -- bash -c "cd /home/find/code/vision_/vision_dev/ && source install/setup.bash && ros2 launch rm_vision_bringup vision_bringup.launch.py"
=======
        gnome-terminal --title="tracker" -- bash -c "cd /home/nvidia/vision_dev/ && source install/setup.bash && ros2 launch rm_vision_bringup vision_bringup.launch.py"
>>>>>>> 01660d2 (解决nx libxr问题)
        
        [ $? -eq 0 ] && echo "启动成功" || echo "启动失败"
    else
        echo "[$(date '+%H:%M:%S')] tracker running"
    fi
    
    sleep $delay_time
done