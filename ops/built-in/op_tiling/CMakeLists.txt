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

# the list of source files
set(_op_tiling_files
  # file1
  # file2
)

if(_op_tiling_files)

  add_library(op_tiling SHARED ${_op_tiling_files})

  target_include_directories(op_tiling PRIVATE
    ${CANN_ROOT}/ops/built-in/common/inc
    ${CANN_ROOT}/inc
    ${CANN_ROOT}/inc/external
    ${JSON_INCLUDE}
    ${Protobuf_INCLUDE}
  )

  target_link_libraries(op_tiling PRIVATE
    ${graph}
    ${register}
    ${slog}
    ${error_manager}
    c_sec
    protobuf
  )

  set(OP_TILING_PATH "${INSTALL_PATH}/opp/op_tiling/built-in")
  cann_install(
    TARGET      op_tiling
    FILES       $<TARGET_FILE:op_tiling>
    DESTINATION "${OP_TILING_PATH}"
  )

endif()
