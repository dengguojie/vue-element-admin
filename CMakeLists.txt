cmake_minimum_required(VERSION 3.14)

project(cann C CXX)

# set variables for cann project
set(CANN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CANN_ROOT=${CANN_ROOT}")

# build external prjects
include(ExternalProject)
include(cmake/external/secure_c.cmake)
include(cmake/external/nlohmann_json.cmake)
include(cmake/external/protobuf.cmake)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

add_subdirectory(ops/built-in)
