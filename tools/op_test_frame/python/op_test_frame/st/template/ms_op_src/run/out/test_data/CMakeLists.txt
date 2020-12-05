# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_subdirectory(src)
endif()