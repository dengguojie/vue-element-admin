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
  scatter_non_aliasing_add.cc
  binary_cross_entropy.cc
  binary_cross_entropy_grad.cc
  bitwise.cc
  broadcast_to.cc
  assign.cc
  avg_pool.cc
  avg_pool_1d.cc
  avg_pool_grad.cc
  avg_pool3d_grad.cc
  bias.cc
  bias_add.cc
  bn_reduce.cc
  max_pool_grad.cc
  bninference_d.cc
  tile_with_axis.cc
  bn_reduce_grad.cc
  bn_training_update.cc
  concat.cc
  concat_offset.cc
  conv2d_backprop_filter.cc
  conv2d_backprop_input.cc
  conv3d_backprop_input.cc
  conv3d_transpose.cc
  sigmoid_cross_entropy_with_logits_grad_v2.cc
  conv2d.cc
  conv2d_transpose.cc
  depthwise_conv2d.cc
  conv3d.cc
  mse_loss.cc
  mse_loss_grad.cc
  conv3d_backprop_filter.cc
  cube_tiling.cc
  deconvolution.cc
  dynamic_atomic_addr_clean.cc
  eletwise.cc
  broadcast.cc
  fill.cc
  gather.cc
  gather_nd.cc
  gather_v2.cc
  ger.cc
  layer_norm_x_backprop.cc
  lock.cc
  layer_norm_beta_gamma_backprop.cc
  l1_loss_grad.cc
  lp_loss.cc
  nms_with_mask.cc
  pad_common.cc
  pad_d.cc
  pad.cc
  pad_v3.cc
  prelu.cc
  reverse.cc
  reduce_tiling.cc
  resize_nearest_neighbor_v2.cc
  resize_bilinear_v2_grad.cc
  resize_bilinear_v2.cc
  scale.cc
  scatter_add.cc
  scatter_nd.cc
  scatter_sub.cc
  scatter_update.cc
  scatter_max.cc
  scatter_min.cc
  scatter_mul.cc
  scatter_div.cc
  scatter_nd_add.cc
  scatter_nd_sub.cc
  scatter_nd_update.cc
  slice.cc
  softmax_cross_entropy_with_logits.cc
  sparse_apply_ftrl_d.cc
  sparse_apply_proximal_adagrad_d.cc
  split_d.cc
  split_v.cc
  expand.cc
  strided_slice.cc
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
  trans_data_negative_target_tc_201.cc
  trans_data_negative_target_ntc.cc
  trans_data_positive_source_tc_1010.cc
  trans_data_positive_source_tc_1011.cc
  trans_data_positive_source_ntc_100.cc
  unpack.cc
  unsorted_segment_sum.cc
  rnn_gen_mask_v2.cc
  unsorted_segment.cc
  in_top_k.cc
  gemm.cc
  bias_add_grad.cc
  reduce_mean.cc
  softmax_tiling.cc
  flatten.cc
  embedding_dense_grad.cc
  one_hot.cc
  maxpoolwithargmaxv1.cc
  maxpoolgradwithargmaxv1.cc
  bounding_box_decode.cc
  iou.cc
  layer_norm.cc
  bn_update_grad_tiling.cc
  roi_align.cc
  roi_align_grad.cc
  smooth_l1_loss_v2.cc
  nll_loss.cc
  nll_loss_grad.cc
  confusion_softmax_grad.cc
  sort.cc
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
elseif(UT_TEST_ALL OR TILING_UT)
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
