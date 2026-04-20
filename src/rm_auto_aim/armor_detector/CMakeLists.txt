cmake_minimum_required(VERSION 3.10)
project(armor_detector)

## Use C++14
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
## enforcing cleaner code.
# add_definitions(-Wall -Werror)

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ARMOR_DETECTOR_ENABLE_YOLO "Build YOLO detector with OpenVINO support" OFF)

#######################
## Find dependencies ##
#######################

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
ament_auto_find_build_dependencies()

###########
## Build ##
###########

# 创建列表，内容为 src 目录下的所有 .cpp 文件
file(GLOB_RECURSE SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
list(REMOVE_ITEM SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yolo_detector.cpp
)

ament_auto_add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

if(ARMOR_DETECTOR_ENABLE_YOLO)
  find_package(OpenVINO REQUIRED CONFIG)
  target_sources(${PROJECT_NAME} PRIVATE src/yolo_detector.cpp)
  target_link_libraries(${PROJECT_NAME} openvino::runtime)
  target_compile_definitions(${PROJECT_NAME} PUBLIC ARMOR_DETECTOR_HAS_OPENVINO=1)
  message(STATUS "YOLO detector enabled with OpenVINO")
else()
  target_compile_definitions(${PROJECT_NAME} PUBLIC ARMOR_DETECTOR_HAS_OPENVINO=0)
  message(STATUS "YOLO detector disabled")
endif()

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN rm_auto_aim::ArmorDetectorNode
  EXECUTABLE armor_detector_node
)

#############
## Testing ##
#############
set(BUILD_TESTING ON)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_copyright
    ament_cmake_uncrustify
    ament_cmake_cpplint
  )
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest)

  file(GLOB TEST_SOURCES test/*.cpp)
  foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    ament_add_gtest(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} ${PROJECT_NAME})
  endforeach()
endif()

#############
## Install ##
#############

ament_auto_package(
  INSTALL_TO_SHARE
  model
)
