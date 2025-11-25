cmake_minimum_required(VERSION 3.8)
project(webots_camera)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(webots_ros2_driver REQUIRED)
find_package(pluginlib REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

# 包含目录
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

# 创建动态库
add_library(${PROJECT_NAME} SHARED
  src/webots_camera.cpp
)

# 链接库 (OpenCV 需要显式链接)
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
)

# 声明依赖关系
ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  webots_ros2_driver
  pluginlib
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
)

# 导出插件描述文件
pluginlib_export_plugin_description_file(webots_ros2_driver plugin_description.xml)

# 安装目标文件
install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# 安装头文件
install(
  DIRECTORY include/
  DESTINATION include
)

# 安装插件描述文件
install(
  FILES plugin_description.xml
  DESTINATION share/${PROJECT_NAME}
)

# 安装 launch 和 urdf (如果有)
install(
  DIRECTORY launch urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
