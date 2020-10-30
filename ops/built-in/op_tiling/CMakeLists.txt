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
set(_optiling_files
  ../op_proto/util/error_util.cc
  ../fusion_pass/common/fp16_t.cc
  concat.cc
  conv2d_backprop_filter.cc
  conv2d_backprop_input.cc
  conv2d.cc
  conv_tiling.cc
  dynamic_atomic_addr_clean.cc
  eletwise.cc
  fill.cc
  gather.cc
  gather_nd.cc
  gather_v2.cc
  pad_common.cc
  pad_d.cc
  reduce_tiling.cc
  scatter_add.cc
  scatter_nd.cc
  scatter_sub.cc
  scatter_update.cc
  slice.cc
  sparse_apply_ftrl_d.cc
  sparse_apply_proximal_adagrad_d.cc
  split_d.cc
  strided_slice.cc
  strided_slice_grad.cc
  tile_d.cc
  tiling_dispatch.cc
  transpose_d.cc
  unpack.cc
  unsorted_segment_sum.cc
)

add_library(optiling SHARED ${_optiling_files})

target_include_directories(optiling PRIVATE
  ${CANN_ROOT}/ops/common/inc
  ${CANN_ROOT}/ops/built-in
  ${CANN_ROOT}/inc/external
  ${CANN_ROOT}/inc/external/graph
  ${CANN_ROOT}/third_party/fwkacllib/inc
  ${CANN_ROOT}/third_party/metadef
  ${CANN_ROOT}/third_party/metadef/inc
  ${CANN_ROOT}/third_party/metadef/inc/external
  ${CANN_ROOT}/third_party/metadef/inc/external/graph
  ${CANN_ROOT}/third_party/metadef/third_party/graphengine/inc
  ${CANN_ROOT}/third_party/metadef/third_party/graphengine/inc/external
  ${JSON_INCLUDE}
  ${Protobuf_INCLUDE}
)

target_link_libraries(optiling PRIVATE
  ${graph}
  ${register}
  ${slog}
  ${error_manager}
  ${protobuf}
  c_sec
)

add_dependencies(optiling json_headers)

# to solve the compile warnings in fp16_t.cc
target_compile_options(optiling PRIVATE
  "-fno-strict-aliasing"
)

set(OP_TILING_PATH "${INSTALL_PATH}/opp/op_impl/built-in/ai_core/tbe/op_tiling")
cann_install(
  TARGET      optiling
  FILES       $<TARGET_FILE:optiling>
  DESTINATION "${OP_TILING_PATH}"
)
