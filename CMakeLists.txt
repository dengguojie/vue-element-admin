cmake_minimum_required(VERSION 3.14)

project(cann C CXX)

# set variables for cann project
set(CANN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CANN_ROOT=${CANN_ROOT}")

# the install path for cann project
if(NOT BUILD_PATH)
  set(BUILD_PATH "${CMAKE_CURRENT_BINARY_DIR}")
endif()
set(INSTALL_PATH "${BUILD_PATH}/install")
message(STATUS "INSTALL_PATH=${INSTALL_PATH}")

# configurations
if(EXISTS "${BUILD_PATH}/config.cmake")
  # the current config
  include("${BUILD_PATH}/config.cmake")
else()
  # the template config
  include("${CANN_ROOT}/cmake/template/config.cmake")
endif()

include(cmake/util.cmake)

# build external prjects
include(ExternalProject)
include(cmake/external/secure_c.cmake)
include(cmake/external/nlohmann_json.cmake)
include(cmake/external/protobuf.cmake)

# find necessary packages
find_package(Python3 COMPONENTS Interpreter)

# compile and link options
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

# sub projects
add_subdirectory(ops/built-in)
add_subdirectory(tbe)
