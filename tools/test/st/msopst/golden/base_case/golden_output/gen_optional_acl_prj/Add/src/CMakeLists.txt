# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(acl_op)

# Compile options
add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/run/out")
message(${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(INC_PATH $ENV{DDK_PATH})

if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else ()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif()

set(LIB_PATH $ENV{NPU_HOST_LIB})

if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/acllib/lib64/stub/")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else ()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()

# Header path
include_directories(
    ${INC_PATH}/acllib/include/
    ../inc
)

# add host lib path
link_directories(
    ${LIB_PATH}
)

add_executable(main
        op_runner.cpp
        main.cpp
        common.cpp
        op_execute.cpp
        op_test.cpp
        op_test_desc.cpp
        testcase.cpp)

target_link_libraries(main
        ascendcl
        stdc++)

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# custom command process om conversion
message(STATUS "SOC_VERSION=${SOC_VERSION}")
set(ASCEND_HOME $ENV{ADK_PATH})
