# before all

工作流程：

`master` 为稳定分支，`dev` 为开发中分支。

将dev分支fork下来，进行自己负责的模块的开发/改动。改动完成后首先将上游仓库的新commit都pull下来，如果有冲突则解决冲突，随后向上游仓库提交pr，随后进行review/update，确认没有问题后进行merge。master会在dev阶段性测试无误后被merge。




# 简介

本项目源自于RV并做出了发展，识别器，预测器，弹道结算全放在了上位机，在机器人运动控制做好的情况下，一辆车的配适时间压缩到半小时，视觉从0上手调车只需要半天。

# 调车

### 1.相机标定

打开两个终端，source后分别输入：

```bash
ros2 launch hik_camera hik_camera.launch.py
```

```bash
ros2 run camera_calibration cameracalibrator --size 11x8 --square 0.02 image:=/image_raw camera:=/hik_camera
```

`size` 为棋盘格交点的数量 `width x height`

`square` 为一个黑色棋子的大小，单位为 m

标定完成后修改rm_vision/config/camera_info。

**检验标定结果**

PnP解算的距离与实际距离在7m,5m,3m,1.5m时的误差

### 2.机器人坐标系维护：

根据实际测量修改urdf文件，尽可能符合实际情况。通过可视化展示，看机器人建模是否正确

### 3.对齐时间戳

IMU与相机多传感修正：目标不动，本车动，看position的变化
如果position差值大，则证明时间戳没有对齐，这时应该调整time_offset，去尽量对齐时间戳。
IMU读值快，时间戳对齐采用线性插值法，用最接近图像数据的IMU的两帧时间戳，求和取平均得到的值去和图形数据配对。
时间戳对齐的目标不懂晃动车头，在不丢失目标的情况下动，position.x的波动范围为0.04以内。

### 4.卡尔曼滤波

计算方差：
识别装甲板，观察预测出的x，y，z，Yaw(target中的数据)，
首先静止不动等数据稳定，记录当前值(记为stable)
然后用手缓慢左右摆动枪管，但不要跟丢装甲板(保证卡尔曼滤波是连续的没有断掉)，
然后记录下极大值和极小值的差(记为diff)，然后按以下公式计算出方差

$$
\frac{(\frac {diff} {4})^2}{stable}
$$

### 5.打静止装甲板，修正到目标中心

在确定urdf文件描述正确的情况下，可以硬补。

### 6.不同距离下打旋转装甲板

### 7.自启动脚本 auto.sh

# 使用

### 安装ROS

  [Ubuntu22.04.1安装ROS2入门级教程(ros-humble)_ros humble_Python-AI Xenon的博客-CSDN博客](https://blog.csdn.net/yxn4065/article/details/127352587)

### 创建工作空间

```Shell
mkdir -p ~/AUTO_AIM
```

### 下载源代码

  在 `AUTO_AIM` 目录下

```Shell
git clone https://github.com/QDU-VRobot/RM26_vision_development.git
cd RM26_vision_development
git switch auto_aming_system_dev # 开发分支
git switch auto_aming_system # 稳定分支
```

```Shell
sudo apt install ros-humble-foxglove-bridge
```

### 编译

```Shell
rosdep install --from-paths src --ignore-src -r -y
```

```Shell
colcon build --symlink-install
```

若使用clangd，则：

```Shell
colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

### 运行节点

```Shell
sudo chmod 777 /dev/ttyACM0
```

  运行每个节点，必须新建终端并输入命令，且运行前需要执行 `source install/setup.bash`

```Shell
source install/setup.bash
ros2 launch rm_vision_bringup no_hardware.launch.py
```

```Shell
source install/setup.bash
ros2 launch rm_vision_bringup vision_bringup.launch.py
```

- 单独运行子模块

```Shell
source install/setup.bash
ros2 launch hik_camera hik_camera.launch.py
```

```Shell
source install/setup.bash
ros2 launch rm_serial_driver serial_driver.launch.py
```

# 启动可视化

  打开新的终端

```Shell
source install/setup.bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765
```

# 一些新的修改

## 手眼标定

`rm_hand_eye_calibrate`包，编译好后使用以下命令启动：

```bash
ros2 launch rm_hand_eye_calibrate hand_eye_calibrate.launch.py
```

参数位于 `src/rm_vision/rm_vision_bringup/config/node_params.yaml`

其中：

```yaml
    image_topic: "/image_raw" # 图像话题名称
    camera_info_topic: "/camera_info" # 相机信息话题名称
    joint_states_topic: "/joint_states" # 关节状态话题名称
    yaw_joint_name: "yaw_joint" # Yaw 关节名称
    pitch_joint_name: "pitch_joint" # Pitch 关节名称

    publish_debug_image: true # 是否发布调试图像
    debug_image_topic: "/calibrate_debug_image" # 调试图像话题名称

    board_cols: 11 # 标定板列数
    board_rows: 8 # 标定板行数
    square_size: 0.02 # 方格尺寸（米）

    max_age_sec: 0.25 # 采样时用于过滤过旧检测结果的时间阈值（秒）
    handeye_method: "DANIILIDIS" # 手眼标定方法：TSAI, PARK, HORAUD, ANDREFF, DANIILIDIS
    invert_pitch_sign: false # 若云台 pitch 角正方向与预期相反，则置 true

```

使用通过服务通信触发。

获取样本：

```bash
ros2 service call /rm_hand_eye/capture std_srvs/srv/Trigger "{}"
```

清空样本：

```bash
ros2 service call /rm_hand_eye/reset std_srvs/srv/Trigger "{}"
```

求解：

```bash
ros2 service call /rm_hand_eye/solve std_srvs/srv/Trigger "{}"
```
