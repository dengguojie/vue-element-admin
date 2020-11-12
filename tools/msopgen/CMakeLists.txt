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
project(Msopgen)

set(MSOPGEN_BUILD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/util)
set(MSOPGEN_WHEEL_OUT ${MSOPGEN_BUILD_PATH}/build/)
set(MSOPGEN_WHEEL_IMAGE op_gen-0.1-py3-none-any.whl)

add_custom_target(MsopgenFiles ALL DEPENDS ${MSOPGEN_WHEEL_IMAGE})
add_custom_command(OUTPUT  ${MSOPGEN_WHEEL_IMAGE}
  COMMAND rm -rf ${MSOPGEN_WHEEL_OUT}
  COMMAND cd ${MSOPGEN_BUILD_PATH} && ./build_whl.sh ${HI_PYTHON}
  COMMAND ${CMAKE_COMMAND} -E copy ${MSOPGEN_WHEEL_OUT}${MSOPGEN_WHEEL_IMAGE} ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND rm -rf ${MSOPGEN_WHEEL_OUT}
)

if(BUILD_OPEN_PROJECT)
  set(MSOPGEN ${INSTALL_PATH}/toolkit/tools)
  cann_install(
    TARGET      MsopgenFiles
    FILES       ${CMAKE_CURRENT_BINARY_DIR}/${MSOPGEN_WHEEL_IMAGE}
    DESTINATION ${MSOPGEN}
  )
else()
  install(
    FILES       ${CMAKE_CURRENT_BINARY_DIR}/${MSOPGEN_WHEEL_IMAGE}
    DESTINATION lib/tensor_utils OPTIONAL
  )
endif()
