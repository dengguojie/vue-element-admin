cmake_minimum_required(VERSION 3.14)

project(cann C CXX)


set(CANN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CANN_ROOT=${CANN_ROOT}")

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")


add_subdirectory(ops/built-in)
