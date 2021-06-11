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
project(OperatorSample)

option(BUILD_OPEN_PROJECT "Do not build custom_operator_sample." FALSE)
if(NOT BUILD_OPEN_PROJECT)
  set(METADEF_DIR ${TOP_DIR}/metadef)
  set(_proto_srcs
    "${METADEF_DIR}/proto/onnx/ge_onnx.proto"
)

  protobuf_generate(opp _proto_cc _proto_h ${_proto_srcs})

  set(OPERATOR_SAMPLE_NAME custom_operator_sample)
  set(OPERATOR_SAMPLE_BUILD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${OPERATOR_SAMPLE_NAME})
  #message(${OPERATOR_SAMPLE_BUILD_PATH})
  add_custom_target(OperatorSampleFiles ALL DEPENDS ${OPERATOR_SAMPLE_NAME})
  add_custom_command(OUTPUT ${OPERATOR_SAMPLE_NAME}
    DEPENDS ${_proto_cc} ${_proto_h}
    COMMAND echo ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND cp -r ${_proto_cc} ${CMAKE_CURRENT_SOURCE_DIR}/dependency_files
    COMMAND cp -r ${_proto_h} ${CMAKE_CURRENT_SOURCE_DIR}/dependency_files
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && chmod +x build.sh && ./build.sh
    COMMAND cp -r ${OPERATOR_SAMPLE_BUILD_PATH} ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${OPERATOR_SAMPLE_BUILD_PATH} ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND rm -rf ${OPERATOR_SAMPLE_BUILD_PATH}
)


  if(BUILD_OPEN_PROJECT)
    set(OPERATOR_SAMPLE ${INSTALL_PATH}/toolkit/tools)
    cann_install(
      TARGET      OperatorSampleFiles
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
