# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

cmake_minimum_required(VERSION 3.14)
project(OpTestFrame)

option(FOR_LLT "llt compile gcc is 4.9" FALSE)
option(LLT_GCC_PATH "llt gcc path" "")
option(LLT_GXX_PATH "llt g++ path" "")
set(CMAKE_CXX_FLAGS "-std=c++11 -Wl,-z,relro,-z,now,-z,noexecstack -s -fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -fstack-protector-all")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

if("${HI_PYTHON}" STREQUAL "")
    set(HI_PYTHON python3.7)
endif()

set(op_test_frame_root ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(OpTestFrameFiles ALL)

set(OP_TEST_FRAME_WHEEL_PACKAGE_REAL_NAME op_test_frame-0.1-py3-none-any.whl)
add_custom_command(TARGET OpTestFrameFiles
  PRE_BUILD
  COMMAND echo ${op_test_frame_root}
  COMMAND ${op_test_frame_root}/python/build_whl.sh ${HI_PYTHON} ${op_test_frame_root}/python
  COMMAND cp ${op_test_frame_root}/python/dist/* ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND rm -rf ${op_test_frame_root}/python/build
  COMMAND rm -rf ${op_test_frame_root}/python/op_test_frame.egg-info
  COMMAND rm -rf ${op_test_frame_root}/python/dist
  COMMAND rm -rf ${OP_TEST_FRAME_WHEEL_PACKAGE_REAL_NAME}
  COMMAND mv op_test_frame-0.1-*.whl ${OP_TEST_FRAME_WHEEL_PACKAGE_REAL_NAME}
)

if(BUILD_OPEN_PROJECT)
  set(OP_TEST_FRAME ${INSTALL_PATH}/toolkit/tools)
  cann_install(
    TARGET      OpTestFrameFiles
    FILES       ${CMAKE_CURRENT_BINARY_DIR}/${OP_TEST_FRAME_WHEEL_PACKAGE_REAL_NAME}
    DESTINATION ${OP_TEST_FRAME}
  )
else()
  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${OP_TEST_FRAME_WHEEL_PACKAGE_REAL_NAME}
    DESTINATION lib
    OPTIONAL
  )
endif()
