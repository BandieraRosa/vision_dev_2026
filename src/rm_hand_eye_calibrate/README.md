# 手眼标定包

编译好后使用以下命令启动：

```bash
ros2 launch rm_hand_eye_calibrate hand_eye_calibrate.launch.py
```

参数位于`src/rm_vision/rm_vision_bringup/config/node_params.yaml`

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

随后终端会出现解算好的x y z/r p y，将其复制到`src/rm_vision/rm_vision_bringup/config/launch_params.yaml`
