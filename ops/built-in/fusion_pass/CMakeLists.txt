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

set(_proto_srcs
  "${METADEF_DIR}/proto/task.proto"
)

protobuf_generate(opp _proto_cc _proto_h ${_proto_srcs})

set(_common_fusion_pass_inc
  ${OPS_DIR}/built-in/fusion_pass/common
  ${OPS_DIR}/built-in/op_proto/util
  ${OPS_INCLUDE}
  ${FWKACLLIB_INCLUDE}
  ${METADEF_INCLUDE}
  ${METADEF_INCLUDE}/external
  ${METADEF_INCLUDE}/external/graph
  ${METADEF_INCLUDE}/register
  ${GRAPHENGINE_INCLUDE}
  ${GRAPHENGINE_INCLUDE}/external
  ${GRAPHENGINE_INCLUDE}/framework
  ${PROTO_BINARY_DIR}/opp
  ${Protobuf_INCLUDE}
  ${FUSION_ENGINE_INCLUDE}
)

set(_common_fusion_pass_dep
  -Wl,--no-as-needed
    graph
    register
    alog
    error_manager
    platform
    ascend_protobuf
    aicore_utils
  -Wl,--as-needed
    c_sec
    json
)

set(_common_srcs
  ${_proto_h}
  ../op_proto/util/error_util.cc
  common/fp16_t.cc
  common/fusion_addoutput_registry.cc
  common/fusion_const2attr_registry.cc
  common/fusion_precheck_func.cc
  common/fusion_removenode_registry.cc
  common/pattern_fusion_util.cc
  common/lx_fusion_func.cc
  common/quant_host_cpu_op_common.cc
)

set(_aicore_fusion_srcs
  buffer_fusion/ub_fusion/ai_core/aipp_conv/tbe_aipp_common_fusion_pass.cc
  buffer_fusion/ub_fusion/ai_core/aipp_conv/tbe_aipp_conv_relu_maxpooling_fusion_pass.cc
  buffer_fusion/ub_fusion/ai_core/aipp_conv/tbe_aipp_fusion_rule.cc
  buffer_fusion/ub_fusion/ai_core/conv2d_backprop_input/conv2d_bp_input_elemwise_pass.cc
  buffer_fusion/ub_fusion/ai_core/conv2d_backprop_input/conv2d_bp_input_dequant_elemwise_quant_pass.cc
  buffer_fusion/ub_fusion/ai_core/conv2d_backprop_input/tbe_conv2d_backprop_dequant_pass.cc
  buffer_fusion/ub_fusion/ai_core/conv2d_backprop_input/tbe_conv2d_backprop_dequant_quant_pass.cc
  buffer_fusion/ub_fusion/ai_core/conv2d_backprop_input/tbe_conv2d_backprop_elemwise_fusion_pass.cc
  buffer_fusion/ub_fusion/ai_core/conv2d_backprop_input/tbe_conv2d_backprop_requant_pass.cc
  buffer_fusion/ub_fusion/ai_core/conv3d/conv3d_elemwise_pass.cc
  buffer_fusion/ub_fusion/ai_core/conv3d_backprop_input/conv3d_bp_input_elemwise_pass.cc
  buffer_fusion/ub_fusion/ai_core/conv/conv2d_reluv2_pass.cc
  buffer_fusion/ub_fusion/ai_core/conv/conv2d_dequant_add_mul_quant_pass.cc
  buffer_fusion/ub_fusion/ai_core/conv/conv2d_writeselect_stridewrite_pass.cc
  buffer_fusion/ub_fusion/ai_core/conv/tbe_conv_dequant_vadd_relu_pass.cc
  buffer_fusion/ub_fusion/ai_core/conv/tbe_conv_dequant_vadd_relu_quant_pass.cc
  buffer_fusion/ub_fusion/ai_core/conv/tbe_conv_dequantS16_pass.cc
  buffer_fusion/ub_fusion/ai_core/conv/tbe_conv_double_in_fusion_pass.cc
  buffer_fusion/ub_fusion/ai_core/conv/tbe_conv_requant_pass.cc
  buffer_fusion/ub_fusion/ai_core/conv/tbe_conv_sigmoid_mul_quant_pass.cc
  buffer_fusion/ub_fusion/ai_core/conv/tbe_conv_bnreduce_fusion_pass.cc
  buffer_fusion/ub_fusion/ai_core/depthwiseconv/tbe_depthwiseconv_requant_pass.cc
  buffer_fusion/ub_fusion/ai_core/depthwiseconv/tbe_depthwiseconv_dequant_fusion_pass.cc
  buffer_fusion/ub_fusion/ai_core/depthwiseconv/tbe_depthwiseconv_elemwise_fusion_pass.cc
  buffer_fusion/ub_fusion/ai_core/matmul/batch_matmul_confusiontranspose_ub_fusion.cc
  buffer_fusion/ub_fusion/ai_core/matmul/matmul_confusiontranspose_ub_fusion.cc
  buffer_fusion/ub_fusion/ai_core/matmul/matmul_gelugrad_ub_fusion.cc
  buffer_fusion/ub_fusion/ai_core/matmul/matmul_fastgelugrad_ub_fusion.cc
  buffer_fusion/ub_fusion/ai_core/matmul/tbe_fullyconnection_elemwise_fusion_pass.cc
  buffer_fusion/ub_fusion/ai_core/matmul/tbe_matmul_elemwise.cc
  buffer_fusion/ub_fusion/ai_core/matmul/batch_matmul_fusedmuladd_ub_fusion.cc
  buffer_fusion/ub_fusion/ai_core/matmul/matmul_reduce_sum_ub_fusion.cc
  buffer_fusion/ub_fusion/ai_core/matmul/batch_matmul_dropout_do_mask_v3_d_ub_fusion.cc
  buffer_fusion/ub_fusion/ai_core/matmul/matmul_dropout_do_mask_v3_d_ub_fusion.cc
  buffer_fusion/ub_fusion/ai_core/matmul/tbe_matmul_dequant_pass.cc
  buffer_fusion/ub_fusion/ai_core/matmul/tbe_matmul_requant_pass.cc
  buffer_fusion/ub_fusion/ai_core/pooling/tbe_antiquant_maxpooling_fusion_pass.cc
  buffer_fusion/ub_fusion/ai_core/pooling/tbe_pooling_quant_fusion_pass.cc
  buffer_fusion/ub_fusion/ai_core/conv/conv_clipByValue_fusion_pass.cc
  graph_fusion/ai_core/a_a_square_sum_maximum_rsqrt_mul_fusion_pass.cc
  graph_fusion/ai_core/a_conv2d_mul_fusion_pass.cc
  graph_fusion/ai_core/adaptive_max_pool2d_fusion_pass.cc
  graph_fusion/ai_core/add_n_fusion_pass.cc
  graph_fusion/ai_core/apply_addoutput_fusion_pass.cc
  graph_fusion/ai_core/a_reduce_all_fusion_pass.cc
  graph_fusion/ai_core/a_reduce_any_fusion_pass.cc
  graph_fusion/ai_core/a_reduce_max_fusion_pass.cc
  graph_fusion/ai_core/a_reduce_mean_fusion_pass.cc
  graph_fusion/ai_core/a_reduce_min_fusion_pass.cc
  graph_fusion/ai_core/a_reduce_prod_fusion_pass.cc
  graph_fusion/ai_core/a_reduce_sum_fusion_pass.cc
  graph_fusion/ai_core/arg_max_grad_fusion_pass.cc
  graph_fusion/ai_core/arg_max_with_k_fusion_pass.cc
  graph_fusion/ai_core/arg_max_v2_fusion_pass.cc
  graph_fusion/ai_core/avg_pool3d_fusion_pass.cc
  graph_fusion/ai_core/avg_pool3d_grad_fusion_pass.cc
  graph_fusion/ai_core/avg_pool_1d_fusion_pass.cc
  graph_fusion/ai_core/avg_pool_grad_fusion_pass.cc
  graph_fusion/ai_core/avg_pool_pass.cc
  graph_fusion/ai_core/avg_pool_v2_fusion_pass.cc
  graph_fusion/ai_core/avg_pool_v2_grad_fusion_pass.cc
  graph_fusion/ai_core/batch_matmul_fusion_pass.cc
  graph_fusion/ai_core/batch_multi_class_nms_fusion_pass.cc
  graph_fusion/ai_core/batchnorm_bninfer_fusion_pass.cc
  graph_fusion/ai_core/batchnorm_fusion_pass.cc
  graph_fusion/ai_core/batchnormgrad_bninfergrad_fusion_pass.cc
  graph_fusion/ai_core/einsum_fusion_pass.cc
  graph_fusion/ai_core/biasadd_conv_fusion_pass.cc
  graph_fusion/ai_core/combined_non_max_suppression_fusion_pass.cc
  graph_fusion/ai_core/concatd_fusion_pass.cc
  graph_fusion/ai_core/concat_fusion_pass.cc
  graph_fusion/ai_core/concat_v2d_fusion_pass.cc
  graph_fusion/ai_core/concat_v2_fusion_pass.cc
  graph_fusion/ai_core/confusion_matrix_fusion_pass.cc
  graph_fusion/ai_core/confusion_mul_grad_fusion_pass.cc
  graph_fusion/ai_core/const2attr/const2attr_fusion_pass.cc
  graph_fusion/ai_core/confusion_softmax_grad_fusion_pass.cc
  graph_fusion/ai_core/conv2dbackprop_dilation_fusion_pass.cc
  graph_fusion/ai_core/conv_add_fusion_pass.cc
  graph_fusion/ai_core/conv_batchnorm_fusion_pass.cc
  graph_fusion/ai_core/conv_fusion_pass_base.cc
  graph_fusion/ai_core/conv_scale_fusion_pass.cc
  graph_fusion/ai_core/conv_to_fullyconnection_fusion_pass.cc
  graph_fusion/ai_core/copy_fusion_pass.cc
  graph_fusion/ai_core/decode_bbox_v2_insert_transpose_pass.cc
  graph_fusion/ai_core/deconv_weight_trans_fusion_pass.cc
  graph_fusion/ai_core/deformable_conv2d_fusion_pass.cc
  graph_fusion/ai_core/deformable_offsets_grad_fusion_pass.cc
  graph_fusion/ai_core/deformable_offsets_fusion_pass.cc
  graph_fusion/ai_core/derelu_fusion_pass.cc
  graph_fusion/ai_core/diag_fusion_pass.cc
  graph_fusion/ai_core/diag_part_fusion_pass.cc
  graph_fusion/ai_core/dynamic_lstm_fusion_pass.cc
  graph_fusion/ai_core/common_lstm_fusion_pass.cc
  graph_fusion/ai_core/dynamic_gru_v2_fusion_pass.cc
  graph_fusion/ai_core/dynamic_gru_v2_grad_fusion_pass.cc
  graph_fusion/ai_core/dynamic_rnn_grad_fusion_pass.cc
  graph_fusion/ai_core/dynamic_rnn_insert_transpose_pass.cc
  graph_fusion/ai_core/extremum_grad_fusion_pass.cc
  graph_fusion/ai_core/fc_transdata_merge_fusion_pass.cc
  graph_fusion/ai_core/flatten_v2_fusion_pass.cc
  graph_fusion/ai_core/fullyconnection_power_fusion_pass.cc
  graph_fusion/ai_core/fullyconnection_reshape_fusion_pass.cc
  graph_fusion/ai_core/fusedbatchnorm_bert_fusion_pass.cc
  graph_fusion/ai_core/fusedbatchnorm_fusion_pass.cc
  graph_fusion/ai_core/fusedbatchnormgrad_fusion_pass.cc
  graph_fusion/ai_core/fusedbatchnorminfgrad_fusion_pass.cc
  graph_fusion/ai_core/gather_v2_fusion_pass.cc
  graph_fusion/ai_core/gelu_fusion_pass.cc
  graph_fusion/ai_core/gelu_onnx_fusion_pass.cc
  graph_fusion/ai_core/gemm_transpose_fusion_pass.cc
  graph_fusion/ai_core/gru_fusion_pass.cc
  graph_fusion/ai_core/hostbn_fusion_pass.cc
  graph_fusion/ai_core/hostin_fusion_pass.cc
  graph_fusion/ai_core/hard_max_fusion_pass.cc
  graph_fusion/ai_core/imgwarp_fusion_pass.cc
  graph_fusion/ai_core/remap_fusion_pass.cc
  graph_fusion/ai_core/keep_ratio_resize_bilinear_fusion_pass.cc
  graph_fusion/ai_core/lars_v2_fusion_pass.cc
  graph_fusion/ai_core/layernorm_beta_gamma_backprop_fusion_pass.cc
  graph_fusion/ai_core/layernorm_beta_gamma_backprop_v2_fusion_pass.cc
  graph_fusion/ai_core/layer_norm_fusion_pass.cc
  graph_fusion/ai_core/layer_norm_onnx_mul_fusion_pass.cc
  graph_fusion/ai_core/layernormgrad_fusion_pass.cc
  graph_fusion/ai_core/lin_space_fusion_pass.cc
  graph_fusion/ai_core/logsoftmaxgrad_fusion_pass.cc
  graph_fusion/ai_core/lstm_fusion_pass.cc
  graph_fusion/ai_core/map_index_pass.cc
  graph_fusion/ai_core/matmul_biasadd_fusion_pass.cc
  graph_fusion/ai_core/matmul_cast_fusion_pass.cc
  graph_fusion/ai_core/matmulv2_fusion_pass.cc
  graph_fusion/ai_core/matrix_diag_fusion_pass.cc
  graph_fusion/ai_core/matrix_diag_part_fusion_pass.cc
  graph_fusion/ai_core/matrix_set_diag_fusion_pass.cc
  graph_fusion/ai_core/max_pool3d_grad_grad_fusion_pass.cc
  graph_fusion/ai_core/maxpoolwithargmax_fusion_pass.cc
  graph_fusion/ai_core/mean_grad_fusion_pass.cc
  graph_fusion/ai_core/momentum_lossscale_fusion_pass.cc
  graph_fusion/ai_core/mul_add_fusion_pass.cc
  graph_fusion/ai_core/mul_addn_fusion_pass.cc
  graph_fusion/ai_core/mul_addn_l2loss_fusion_pass.cc
  graph_fusion/ai_core/mul_maximum_fusion_pass.cc
  graph_fusion/ai_core/nms_with_mask_fusion_pass.cc
  graph_fusion/ai_core/normalize_fusion_pass.cc
  graph_fusion/ai_core/one_hot_fusion_pass.cc
  graph_fusion/ai_core/pack_fusion_pass.cc
  graph_fusion/ai_core/padd_conv2d_fusion_pass.cc
  graph_fusion/ai_core/padd_depthwise_conv2d_fusion_pass.cc
  graph_fusion/ai_core/padd_maxpool_fusion_pass.cc
  graph_fusion/ai_core/pad_fusion_pass.cc
  graph_fusion/ai_core/pad_v2_fusion_pass.cc
  graph_fusion/ai_core/pad_v3_fusion_pass.cc
  graph_fusion/ai_core/pass_through_fusion_pass.cc
  graph_fusion/ai_core/pass_through_second_fusion_pass.cc
  graph_fusion/ai_core/permute_fusion_pass.cc
  graph_fusion/ai_core/pooling_matrix_pass.cc
  graph_fusion/ai_core/pow_2_square_fusion_pass.cc
  graph_fusion/ai_core/prelu_abs_fusion_pass.cc
  graph_fusion/ai_core/prelu_fusion_pass.cc
  graph_fusion/ai_core/prior_box_fusion_pass.cc
  graph_fusion/ai_core/proposal_fusion_pass.cc
  graph_fusion/ai_core/range_fusion_pass.cc
  graph_fusion/ai_core/reduce_max_d_fusion_pass.cc
  graph_fusion/ai_core/reduce_min_d_fusion_pass.cc
  graph_fusion/ai_core/reduce_sum_fusion_pass.cc
  graph_fusion/ai_core/relu_fusion_pass.cc
  graph_fusion/ai_core/remove_node_fusion_pass.cc
  graph_fusion/ai_core/reshape_transpose_fusion_pass.cc
  graph_fusion/ai_core/resize_nearest_neighbor_grad_d_fusion_pass.cc
  graph_fusion/ai_core/rnn_fusion_pass.cc
  graph_fusion/ai_core/rpn_proposals_fusion_pass.cc
  graph_fusion/ai_core/scatter_nd_fusion_pass.cc
  graph_fusion/ai_core/single_batch_norm_fusion_pass.cc
  graph_fusion/ai_core/softmax_cross_entropy_with_logits.cc
  graph_fusion/ai_core/softmax_cross_entropy_with_logits_grad.cc
  graph_fusion/ai_core/softmax_fusion_pass.cc
  graph_fusion/ai_core/softmax_grad_ext_fusion_pass.cc
  graph_fusion/ai_core/softmax_grad_ext_v2_fusion_pass.cc
  graph_fusion/ai_core/softmax_transpose_fusion_pass.cc
  graph_fusion/ai_core/softplus_tanh_mul_fusion.cc
  graph_fusion/ai_core/space_to_depth_fusion_pass.cc
  graph_fusion/ai_core/sparse_softmax_cross_entropy_with_logits_fusion_pass.cc
  graph_fusion/ai_core/spatial_transformer_d_fusion_pass.cc
  graph_fusion/ai_core/split_conv2d_concat_fusion_pass.cc
  graph_fusion/ai_core/splitd_fusion_pass.cc
  graph_fusion/ai_core/split_fusion_pass.cc
  graph_fusion/ai_core/splitvd_fusion_pass.cc
  graph_fusion/ai_core/splitv_fusion_pass.cc
  graph_fusion/ai_core/spp_fusion_pass.cc
  graph_fusion/ai_core/strided_slice_fusion_pass.cc
  graph_fusion/ai_core/strided_slice_grad_fusion_pass.cc
  graph_fusion/ai_core/tbe_fusion_pass_util.cc
  graph_fusion/ai_core/tbe_ops_pass_util.cc
  graph_fusion/ai_core/tensor_scatter_add_fusion_pass.cc
  graph_fusion/ai_core/tensor_scatter_sub_fusion_pass.cc
  graph_fusion/ai_core/tensor_scatter_update_fusion_pass.cc
  graph_fusion/ai_core/tile_fusion_pass.cc
  graph_fusion/ai_core/top_k_fusion_pass.cc
  graph_fusion/ai_core/transdata_confusion_transpose_fusion_pass.cc
  graph_fusion/ai_core/transdata_transdata_fusion_pass.cc
  graph_fusion/ai_core/transpose_inplaceupdate_fusion_pass.cc
  graph_fusion/ai_core/transpose_reshape_fusion_pass.cc
  graph_fusion/ai_core/unpack_fusion_pass.cc
  graph_fusion/ai_core/unsortedsegmentsumd_fusion_pass.cc
  graph_fusion/ai_core/yolo_fusion_pass.cc
  graph_fusion/ai_core/yolo_v2_detection_output_fusion_pass.cc
  graph_fusion/ai_core/yolo_v3_detection_output_fusion_pass.cc
  graph_fusion/ai_core/yolo_v3_detection_output_v2_fusion_pass.cc
  graph_fusion/ai_core/slice_fusion_pass.cc
  graph_fusion/ai_core/adaptive_avg_pool_fussion_pass.cc
  graph_fusion/ai_core/adaptive_avg_pool_grad_fusion_pass.cc
  graph_fusion/ai_core/resize_fusion_pass.cc
  graph_fusion/ai_core/strided_slice_v2_fusion_pass.cc
  graph_fusion/ai_core/sigmoid_cross_entropy_with_logits_v2_fusion_pass.cc
  graph_fusion/ai_core/conv2dbackprop_filter_mul_fusion_pass.cc
  graph_fusion/ai_core/square_sum_v2_fusion_pass.cc
  graph_fusion/ai_core/conv3d_bp_filter_group_fusion_pass.cc
  graph_fusion/ai_core/non_max_suppression_fusion_pass.cc
  graph_fusion/ai_core/fusedbatchnorm3d_fusion_pass.cc
  graph_fusion/ai_core/fusedbatchnorm3dgrad_fusion_pass.cc
  graph_fusion/ai_core/conv2d_backprop_input_bias_add_fusion_pass.cc
  graph_fusion/ai_core/conv3d_backprop_input_bias_add_fusion_pass.cc
  graph_fusion/ai_core/triu_fusion_pass.cc
  graph_fusion/ai_core/prelu_grad_fusion_pass.cc
  graph_fusion/ai_core/tril_fusion_pass.cc
  graph_fusion/ai_core/affine_grid_fussion_pass.cc
  graph_fusion/ai_core/batch_matmul_v2_reshape_fusion_pass.cc
  graph_fusion/ai_core/sub_sample_fusion_pass.cc
  graph_fusion/ai_core/transposed_update_fusion_pass.cc
  graph_fusion/ai_core/dynamic_rnn_v3_fusion_pass.cc
  graph_fusion/ai_core/z_unsortedsegmentsum_update_fusion_pass.cc
  graph_fusion/ai_core/nll_loss_fusion_pass.cc
  graph_fusion/ai_core/dynamic_rnn_fusion_pass.cc
  graph_fusion/ai_core/resnet50_dbn_dw_fusion_pass.cc
  graph_fusion/ai_core/correlation_fusion_pass.cc
  graph_fusion/ai_core/cast_cast_fusion_pass.cc
  graph_fusion/ai_core/globalavgpool_fusion_pass.cc
  graph_fusion/ai_core/transdata_cast_fusion_pass.cc
  graph_fusion/ai_core/padv3d_avgpool_fusion_pass.cc
  graph_fusion/ai_core/ascend_dequant_quant_antiquant_maxpool_fusion_pass.cc
  graph_fusion/ai_core/softmax_with_drop_out_do_mask_fusion_pass.cc
  graph_fusion/ai_core/single_instance_norm_fusion_pass.cc
  graph_fusion/ai_core/single_instance_norm_grad_fusion_pass.cc
)

set(ops_fusion_pass_path "${INSTALL_PATH}/opp/fusion_pass/built_in")

set(_aic_srcs
  ${_common_srcs}
  ${_aicore_fusion_srcs}
)

set(_llt_aic_srcs
  ${_aic_srcs}
)
if(ENABLE_TEST STREQUAL "")
  add_library(ops_fusion_pass_aicore SHARED ${_aic_srcs})

  target_include_directories(ops_fusion_pass_aicore PRIVATE
    ${_common_fusion_pass_inc}
  )

  target_link_libraries(ops_fusion_pass_aicore PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:$<BUILD_INTERFACE:slog_headers>>
    $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:$<BUILD_INTERFACE:msprof_headers>>
    $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:$<BUILD_INTERFACE:mmpa_headers>>
    ${_common_fusion_pass_dep}
  )

  # to solve the compile warnings in fp16_t.cc
  target_compile_options(ops_fusion_pass_aicore PRIVATE
    "-fno-strict-aliasing"
    "-fno-common"
    -O2
    -Dgoogle=ascend_private
    # TODO: Fix warnings
    # -Werror
  )

  target_compile_definitions(ops_fusion_pass_aicore PRIVATE
    $<$<STREQUAL:${BUILD_OPEN_PROJECT},True>:ONLY_COMPILE_OPEN_SRC>
    )

  if(BUILD_OPEN_PROJECT)
    cann_install(
      TARGET      ops_fusion_pass_aicore
      FILES       $<TARGET_FILE:ops_fusion_pass_aicore>
      DESTINATION "${ops_fusion_pass_path}"
    )
  else()
    install(
      TARGETS ops_fusion_pass_aicore OPTIONAL
      EXPORT  ops_fusion_pass_aicore-targets
      LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
    )
  endif()
  set(_vec_srcs
  ${_common_srcs}
  )

  add_library(ops_fusion_pass_vectorcore SHARED ${_vec_srcs})

  target_include_directories(ops_fusion_pass_vectorcore PRIVATE
    ${_common_fusion_pass_inc}
  )

  target_link_libraries(ops_fusion_pass_vectorcore PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:$<BUILD_INTERFACE:slog_headers>>
    $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:$<BUILD_INTERFACE:msprof_headers>>
    ${_common_fusion_pass_dep}
  )

  # to solve the compile warnings in fp16_t.cc
  target_compile_options(ops_fusion_pass_vectorcore PRIVATE
    "-fno-strict-aliasing"
    -O2
    -Werror
    -Wno-deprecated-declarations
    -Dgoogle=ascend_private
  )

  target_compile_definitions(ops_fusion_pass_vectorcore PRIVATE
    $<$<STREQUAL:${BUILD_OPEN_PROJECT},True>:ONLY_COMPILE_OPEN_SRC>
    )

  if(BUILD_OPEN_PROJECT)
    cann_install(
      TARGET      ops_fusion_pass_vectorcore
      FILES       $<TARGET_FILE:ops_fusion_pass_vectorcore>
      DESTINATION "${ops_fusion_pass_path}/vector_core"
    )
  else()
    install(
      TARGETS ops_fusion_pass_vectorcore OPTIONAL
      EXPORT  ops_fusion_pass_vectorcore-targets
      LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
    )
  endif()
elseif(UT_TEST_ALL OR PASS_UT)
  add_library(ops_fusion_pass_aicore_llt STATIC ${_llt_aic_srcs})
  target_include_directories(ops_fusion_pass_aicore_llt PUBLIC
    ${_common_fusion_pass_inc}
  )
  target_compile_options(ops_fusion_pass_aicore_llt PUBLIC
    -O0
    -Dgoogle=ascend_private
  )
  target_link_libraries(ops_fusion_pass_aicore_llt
    PRIVATE
      $<BUILD_INTERFACE:intf_llt_pub>
    PUBLIC
      ${_common_fusion_pass_dep}
  )
endif()
