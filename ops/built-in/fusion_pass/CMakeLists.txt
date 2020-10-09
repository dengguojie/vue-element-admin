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

set(_common_fusion_pass_inc
  ${CANN_ROOT}/ops/built-in/common/inc
  ${CANN_ROOT}/inc/
  ${CANN_ROOT}/inc/external
  ${JSON_INCLUDE}
  ${Protobuf_INCLUDE}
)

set(_common_fusion_pass_dep
  ${graph}
  ${register}
  ${slog}
  ${error_manager}
  ${platform}
  c_sec
  protobuf
)

set(_common_srcs
  # file
)

set(_buffer_fusion
  # file
)

set(_aic_srcs
  # file
  ${_common_srcs}
  ${_buffer_fusion}
)

set(_vec_srcs
  ${_common_srcs}
)

set(ops_fusion_pass_path "${install_path}/opp/fusion_pass/built-in")

if(_aic_srcs)
  add_library(ops_fusion_pass_aicore SHARED ${_aic_srcs})
  target_include_directories(ops_fusion_pass_aicore PRIVATE
    ${_common_fusion_pass_inc}
  )
  target_link_libraries(ops_fusion_pass_aicore PRIVATE
    ${_common_fusion_pass_dep}
  )

  cann_install(
    TARGET      ops_fusion_pass_aicore
    FILES       $<target_file:ops_fusion_pass_aicore>
    DESTINATION "${ops_fusion_pass_path}"
  )
endif()

if(_vec_srcs)
  add_library(ops_fusion_pass_vectorcore SHARED ${_vec_srcs})
  target_include_directories(ops_fusion_pass_vectorcore PRIVATE
    ${_common_fusion_pass_inc}
  )
  target_link_libraries(ops_fusion_pass_vectorcore PRIVATE
    ${_common_fusion_pass_dep}
  )

  cann_install(
    TARGET      ops_fusion_pass_vectorcore
    FILES       $<target_file:ops_fusion_pass_vectorcore>
    DESTINATION "${ops_fusion_pass_path}"
  )
endif()
