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
cmake_minimum_required(VERSION 3.10)
project(OperatorDemo)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -s")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CAMKE_SKIP_RPATH TRUE)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -pie")
add_compile_options(
    -fstack-protector-all
    -D_FORTIFY_SOURCE=2 -O2
    )

option(BUILD_OPEN_PROJECT "Do not build operator_demo_projects." FALSE)
if(NOT BUILD_OPEN_PROJECT)
  set(OPERATOR_SAMPLE_NAME operator_demo_projects)
  set(OPERATOR_SAMPLE_BUILD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${OPERATOR_SAMPLE_NAME})
  add_custom_target(OperatorDemoProjects ALL DEPENDS ${OPERATOR_SAMPLE_NAME})
  # yellow zone, copy makeself from open source
  set(MAKESELF_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../open_source/makeself)
  file(GLOB ConfigFiles ${MAKESELF_PATH}/COPYING
                      ${MAKESELF_PATH}/makeself.1
                      ${MAKESELF_PATH}/makeself.lsm
                      ${MAKESELF_PATH}/*.sh
                      ${MAKESELF_PATH}/README.md
                      ${MAKESELF_PATH}/VERSION)

  add_custom_command(OUTPUT ${OPERATOR_SAMPLE_NAME}
    DEPENDS ${_proto_cc} ${_proto_h}
    COMMAND rm -rf ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_operator_sample/cmake
    COMMAND rm -rf ${CMAKE_CURRENT_SOURCE_DIR}/tbe_operator_sample/cmake
    COMMAND mkdir -p ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_operator_sample/cmake/util/makeself
    COMMAND mkdir -p ${CMAKE_CURRENT_SOURCE_DIR}/tbe_operator_sample/cmake/util/makeself
    COMMAND cp -rf ${ConfigFiles} ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_operator_sample/cmake/util/makeself
    COMMAND cp -rf ${ConfigFiles} ${CMAKE_CURRENT_SOURCE_DIR}/tbe_operator_sample/cmake/util/makeself
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && chmod +x build.sh && ./build.sh
    COMMAND cp -r ${OPERATOR_SAMPLE_BUILD_PATH} ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${OPERATOR_SAMPLE_BUILD_PATH} ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND rm -rf ${OPERATOR_SAMPLE_BUILD_PATH}
  )

  if(BUILD_OPEN_PROJECT)
    set(OPERATOR_SAMPLE ${INSTALL_PATH}/toolkit/tools)
    cann_install(
      TARGET      OperatorDemoProjects
      FILES       ${CMAKE_CURRENT_BINARY_DIR}/${OPERATOR_SAMPLE_NAME}
      DESTINATION ${OPERATOR_SAMPLE}
    )
  else()
    install(
      FILES       ${CMAKE_CURRENT_BINARY_DIR}/${OPERATOR_SAMPLE_NAME}
      DESTINATION lib/tensor_utils OPTIONAL
    )
  endif()

endif()
