cmake_minimum_required(VERSION 3.14)
project(libxr LANGUAGES C CXX ASM)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)

set(LIBXR_SHARED_BUILD
    ON
    CACHE BOOL "" FORCE)
set(LIBXR_SYSTEM
    "Linux"
    CACHE STRING "" FORCE)
set(LIBXR_DRIVER
    "Linux"
    CACHE STRING "" FORCE)

add_subdirectory(libxr)

get_target_property(_xr_defs xr INTERFACE_COMPILE_DEFINITIONS)
if(_xr_defs)
  list(REMOVE_ITEM _xr_defs "EIGEN_NO_IO")
  set_target_properties(xr PROPERTIES INTERFACE_COMPILE_DEFINITIONS
                                      "${_xr_defs}")
endif()
target_compile_definitions(xr PRIVATE EIGEN_NO_IO)

set(LIBXR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/libxr)

set_target_properties(xr PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")

target_include_directories(xr PRIVATE ${LIBXR_SRC}/lib/Eigen)

target_link_libraries(xr PUBLIC Eigen3::Eigen)

target_include_directories(
  xr
  PUBLIC $<BUILD_INTERFACE:${LIBXR_SRC}/src>
         $<BUILD_INTERFACE:${LIBXR_SRC}/src/core>
         $<BUILD_INTERFACE:${LIBXR_SRC}/src/system>
         $<BUILD_INTERFACE:${LIBXR_SRC}/src/utils>
         $<BUILD_INTERFACE:${LIBXR_SRC}/src/structure>
         $<BUILD_INTERFACE:${LIBXR_SRC}/src/middleware>
         $<BUILD_INTERFACE:${LIBXR_SRC}/src/driver>
         $<BUILD_INTERFACE:${LIBXR_SRC}/system/Linux>
         $<BUILD_INTERFACE:${LIBXR_SRC}/driver/Linux>
         $<INSTALL_INTERFACE:include/libxr>)

file(
  GLOB_RECURSE _usb_all
  LIST_DIRECTORIES true
  "${LIBXR_SRC}/src/driver/usb/*")
set(_usb_dirs "${LIBXR_SRC}/src/driver/usb")
foreach(_p ${_usb_all})
  if(IS_DIRECTORY "${_p}")
    list(APPEND _usb_dirs "${_p}")
  endif()
endforeach()
list(REMOVE_DUPLICATES _usb_dirs)
foreach(_d ${_usb_dirs})
  target_include_directories(xr PUBLIC $<BUILD_INTERFACE:${_d}>)
endforeach()

set(_header_dirs
    ${LIBXR_SRC}/src
    ${LIBXR_SRC}/src/core
    ${LIBXR_SRC}/src/system
    ${LIBXR_SRC}/src/utils
    ${LIBXR_SRC}/src/structure
    ${LIBXR_SRC}/src/middleware
    ${LIBXR_SRC}/src/driver
    ${LIBXR_SRC}/system/Linux
    ${LIBXR_SRC}/driver/Linux)
foreach(_dir ${_header_dirs})
  install(
    DIRECTORY "${_dir}/"
    DESTINATION include/libxr
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.h")
endforeach()

install(
  DIRECTORY "${LIBXR_SRC}/src/driver/usb/"
  DESTINATION include/libxr
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h")

install(
  TARGETS xr
  EXPORT libxr_export
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

ament_export_targets(libxr_export HAS_LIBRARY_TARGET)
ament_export_include_directories(include/libxr)
ament_export_libraries(xr)
ament_export_dependencies(Eigen3)

ament_package()
