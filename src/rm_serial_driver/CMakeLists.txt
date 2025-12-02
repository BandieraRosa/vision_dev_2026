cmake_minimum_required(VERSION 3.10)
project(rm_serial_driver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

add_subdirectory(${CMAKE_SOURCE_DIR}/libxr)

ament_auto_add_library(${PROJECT_NAME} SHARED DIRECTORY src)

target_include_directories(
  ${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>
                         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(${PROJECT_NAME} xr)

rclcpp_components_register_node(
  ${PROJECT_NAME} PLUGIN rm_serial_driver::RMSerialDriver EXECUTABLE
  ${PROJECT_NAME}_node)

ament_auto_package(INSTALL_TO_SHARE launch config)
