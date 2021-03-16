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
  bitwise.cc
  broadcast_to.cc
  assign.cc
  avg_pool.cc
  avg_pool_1d.cc
  bias.cc
  bias_add.cc
  bninference_d.cc
  bn_reduce_grad.cc
  concat.cc
  concat_offset.cc
  conv2d_backprop_filter.cc
  conv2d_backprop_input.cc
  conv3d_backprop_input.cc
  conv3d_transpose.cc
  conv2d.cc
  conv2d_transpose.cc
  depthwise_conv2d.cc
  conv3d.cc
  conv3d_backprop_filter.cc
  conv_tiling.cc
  cube_tiling.cc
  deconvolution.cc
  dynamic_atomic_addr_clean.cc
  eletwise.cc
  broadcast.cc
  fill.cc
  gather.cc
  gather_nd.cc
  gather_v2.cc
  lock.cc
  nms_with_mask.cc
  pad_common.cc
  pad_d.cc
  prelu.cc
  reduce_tiling.cc
  resize_nearest_neighbor_v2.cc
  resize_bilinear_v2_grad.cc
  scale.cc
  scatter_add.cc
  scatter_nd.cc
  scatter_sub.cc
  scatter_update.cc
  scatter_max.cc
  scatter_min.cc
  scatter_mul.cc
  scatter_div.cc
  slice.cc
  sparse_apply_ftrl_d.cc
  sparse_apply_proximal_adagrad_d.cc
  split_d.cc
  split_v.cc
  strided_slice.cc
  strided_slice_grad.cc
  select.cc
  tile_d.cc
  tile.cc
  max_pool.cc
  arg_max_v2.cc
  arg_max_with_value.cc
  batch_to_space_nd.cc
  space_to_batch_nd.cc
  drop_out_do_mask.cc
  tiling_dispatch.cc
  top_k.cc
  transpose_d.cc
  transpose.cc
  trans_data.cc
  trans_data_negative_target_tc.cc
  trans_data_positive_target_t_mode100.cc
  trans_data_positive_target_t_mode101.cc
  trans_data_negative_target_h_mode200.cc
  trans_data_positive_source_tc_1010.cc
  trans_data_positive_source_tc_1011.cc
  unpack.cc
  unsorted_segment_sum.cc
  unsorted_segment.cc
  gemm.cc
  bias_add_grad.cc
  reduce_mean.cc
  softmax_tiling.cc
  embedding_dense_grad.cc
  one_hot.cc
)

set(_op_tiling_include
  ${OPS_INCLUDE}
  ${OPS_DIR}/built-in
  ${FWKACLLIB_INCLUDE}
  ${METADEF_DIR}
  ${METADEF_INCLUDE}
  ${METADEF_INCLUDE}/external
  ${METADEF_INCLUDE}/external/graph
  ${GRAPHENGINE_INCLUDE}
  ${GRAPHENGINE_INCLUDE}/external
  ${Protobuf_INCLUDE}
)

set(_op_tiling_link_libs
  -Wl,--no-as-needed
    graph
    register
    alog
    error_manager
    ascend_protobuf
  -Wl,--as-needed
    c_sec
    json
)

if(ENABLE_TEST STREQUAL "")
  add_library(optiling SHARED ${_optiling_files})
  target_include_directories(optiling PRIVATE
    ${_op_tiling_include}
  )
  target_compile_options(optiling PRIVATE
    -O2
    -Dgoogle=ascend_private
    # TODO: Fix warnings
    # -Werror
  )
  target_link_libraries(optiling PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    ${_op_tiling_link_libs}
  )
  # to solve the compile warnings in fp16_t.cc
  target_compile_options(optiling PRIVATE
    "-fno-strict-aliasing"
    "-fno-common"
  )

  if(BUILD_OPEN_PROJECT)
    set(OP_TILING_PATH "${INSTALL_PATH}/opp/op_impl/built-in/ai_core/tbe/op_tiling")
    cann_install(
      TARGET      optiling
      FILES       $<TARGET_FILE:optiling>
      DESTINATION "${OP_TILING_PATH}"
    )
  else()
    install(
      TARGETS optiling OPTIONAL
      EXPORT  optiling-targets
      LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
    )
  endif()
else()
  add_library(optiling_llt STATIC ${_optiling_files})
  target_include_directories(optiling_llt PUBLIC
    ${_op_tiling_include}
  )
  target_compile_options(optiling_llt PUBLIC
    -O0
    -Dgoogle=ascend_private
  )
  target_link_libraries(optiling_llt
    PRIVATE
      $<BUILD_INTERFACE:intf_llt_pub>
    PUBLIC
      ${_op_tiling_link_libs}
  )
endif()
