cmake_minimum_required(VERSION 3.14)

project(cann C CXX)

# set variables for cann project
set(CANN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CANN_ROOT=${CANN_ROOT}")

# the install path for cann project
set(INSTALL_PATH "")
if(DEFINED BUILD_PATH)
  set(INSTALL_PATH "${BUILD_PATH}/install")
else()
  set(INSTALL_PATH "${CMAKE_CURRENT_BINARY_DIR}/install")
endif()
message(STATUS "INSTALL_PATH=${INSTALL_PATH}")

# build external prjects
include(ExternalProject)
include(cmake/external/secure_c.cmake)
include(cmake/external/nlohmann_json.cmake)
include(cmake/external/protobuf.cmake)

# compile and link options
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

# sub projects
add_subdirectory(ops/built-in)
