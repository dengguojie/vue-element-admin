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
  ../op_proto/util/error_util.cc
  ../fusion_pass/common/fp16_t.cc
  concat.cc
  conv2d.cc
  dynamic_atomic_addr_clean.cc
  eletwise.cc
  gather_nd.cc
  gather_v2.cc
  reduce_tiling.cc
  scatter_add.cc
  scatter_nd.cc
  scatter_sub.cc
  scatter_update.cc
  sparse_apply_ftrl_d.cc
  sparse_apply_proximal_adagrad_d.cc
  tiling_dispatch.cc
  unsorted_segment_sum.cc
)

add_library(op_tiling SHARED ${_op_tiling_files})

target_include_directories(op_tiling PRIVATE
  ${CANN_ROOT}/ops/common/inc
  ${CANN_ROOT}/ops/built-in
  ${CANN_ROOT}/inc
  ${CANN_ROOT}/inc/external
  ${CANN_ROOT}/inc/external/graph
  ${CANN_ROOT}/third_party/fwkacllib/inc
  ${CANN_ROOT}/third_party/metadef
  ${JSON_INCLUDE}
  ${Protobuf_INCLUDE}
)

target_link_libraries(op_tiling PRIVATE
  ${graph}
  ${register}
  ${slog}
  ${error_manager}
  ${protobuf}
  c_sec
)

add_dependencies(op_tiling json_headers)

# to solve the compile warnings in fp16_t.cc
target_compile_options(op_tiling PRIVATE
  "-fno-strict-aliasing"
)

set(OP_TILING_PATH "${INSTALL_PATH}/opp/op_tiling/built-in")
cann_install(
  TARGET      op_tiling
  FILES       $<TARGET_FILE:op_tiling>
  DESTINATION "${OP_TILING_PATH}"
)
