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

file(GLOB PROTO_SRCS *.cc *.cpp)
foreach(_op_proto ${PROTO_SRCS})
  get_filename_component(_op_name ${_op_proto} NAME)
  string(REGEX REPLACE "(\.cc$)|(\.cpp$)" "_proto" _target ${_op_name})
  add_library(${_target} SHARED ${_op_proto})

  target_include_directories(${_target} PRIVATE
    ${CANN_ROOT}/inc/external
    ${JSON_INCLUDE}
    ${Protobuf_INCLUDE}
  )

  target_link_libraries(${_target} PRIVATE
    ${graph}
    ${register}
    ${slog}
    ${error_manager}
    ${protobuf}
    c_sec
  )

  set(OPS_PROTO_PATH "${INSTALL_PATH}/opp/op_proto/built-in")
  cann_install(
    TARGET      ${_target}
    FILES       $<TARGET_FILE:${_target}>
    DESTINATION "${OPS_PROTO_PATH}"
  )
endforeach()
