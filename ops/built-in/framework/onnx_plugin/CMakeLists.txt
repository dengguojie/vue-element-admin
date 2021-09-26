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
  "${METADEF_DIR}/proto/onnx/ge_onnx.proto"
)

protobuf_generate(opp _proto_cc _proto_h ${_proto_srcs})

# the list of source files
set(_onnx_plugins
  addcmul_plugin.cc
  adaptive_max_pool2d_plugin.cc
  ascend_anti_quant_plugin.cc
  ascend_dequant_plugin.cc
  ascend_dequant_s16_plugin.cc
  ascend_quant_plugin.cc
  ascend_requant_plugin.cc
  ascend_requant_s16_plugin.cc
  batch_normalization_plugin.cc
  batch_multi_class_non_max_suppression_plugin.cc
  concat_plugin.cc
  conv_plugin.cc
  gather_plugin.cc
  gather_nd_plugin.cc
  gemm_plugin.cc
  gru_plugin.cc
  max_pool_plugin.cc
  mean_variance_normalization_plugin.cc
  mod_plugin.cc
  onehot_plugin.cc
  pad_plugin.cc
  reduce_mean_plugin.cc
  reduce_sum_plugin.cc
  reshape_plugin.cc
  shape_plugin.cc
  topk_plugin.cc
  unsqueeze_plugin.cc
  argmax_plugin.cc
  argmin_plugin.cc
  average_pool_plugin.cc
  batch_matmul_plugin.cc
  cast_plugin.cc
  constantofshape_plugin.cc
  conv_transpose_plugin.cc
  cumsum_plugin.cc
  elu_plugin.cc
  flatten_plugin.cc
  leakyrelu_plugin.cc
  logsoftmax_plugin.cc
  lrn_plugin.cc
  matmul_plugin.cc
  mean_plugin.cc
  max_plugin.cc
  min_plugin.cc
  multinomial_plugin.cc
  reduce_max_plugin.cc
  reduce_min_plugin.cc
  reduce_prod_plugin.cc
  resize_plugin.cc
  reverseSequence_plugin.cc
  roiAlign_plugin.cc
  similar_opp_plugins.cc
  size_plugin.cc
  softmax_plugin.cc
  split_plugin.cc
  squeeze_plugin.cc
  sum_plugin.cc
  slice_plugin.cc
  threshold_relu_plugin.cc
  transpose_plugin.cc
  lstm_plugin.cc
  hardmax_plugin.cc
  non_max_suppression_plugin.cc
  roi_extractor_plugin.cc
  depth_to_space_plugin.cc
  affine_grid_plugin.cc
  reduceL2_plugin.cc
  clip_plugin.cc
  celu_plugin.cc
  hard_sigmoid_plugin.cc
  lp_normalization_plugin.cc
  reduce_sum_square_plugin.cc
  gather_elements_plugin.cc
  global_max_pool_plugin.cc
  reduceL1_plugin.cc
  embedding_bag_plugin.cc
  space_to_depth_plugin.cc
  max_roi_pool_plugin.cc
  selu_plugin.cc
  if_plugin.cc
  random_uniform_plugin.cc
  deformable_conv_plugin.cc
  nonzero_plugin.cc
  quant/int8_dequantize_plugin.cc
  quant/Int8Transpose_plugin.cc
  random_normal_like_plugin.cc
  random_uniform_like_plugin.cc
  adaptive_average_pool2d_plugin.cc
  quant/int8conv_plugin.cc
  quant/int8_give_tensorfill_plugin.cc
  quant/int8_giveint_tensorfill_plugin.cc
  quant/int8_fc_plugin.cc
  quant/int8_conv_relu_plugin.cc
  quant/int8_quantize_plugin.cc
  instancenormalization_plugin.cc
  shrink_plugin.cc
  scatter_elements_plugin.cc
  bitshift_plugin.cc
  tfidf_vectorizer_plugin.cc
  custom/yolo_plugin.cc
  custom/yolov3detectionoutputv2_plugin.cc
  custom/yolo_predetection_plugin.cc
  custom/yolov5detectionoutput_plugin.cc  
  eyelike_plugin.cc
  einsum_plugin.cc
  random_normal_plugin.cc
  isinf_plugin.cc
  onnx_common.cc
  reduce_log_sum_plugin.cc
  lppool_plugin.cc
  compress_plugin.cc
  xor_plugin.cc
  upsample_plugin.cc
  global_lppool_plugin.cc
  reduce_log_sum_exp_plugin.cc
  trilu_plugin.cc
  maxunpool_plugin.cc
)

set(_ops_all_onnx_plugin_include
  ${OPS_INCLUDE}
  ${FWKACLLIB_INCLUDE}
  ${METADEF_INCLUDE}
  ${METADEF_INCLUDE}/external
  ${METADEF_INCLUDE}/external/graph
  ${GRAPHENGINE_INCLUDE}
  ${PROTO_BINARY_DIR}/opp
  ${Protobuf_INCLUDE}
  ${OPS_DIR}/built-in/op_proto/inc
)

set(_ops_all_onnx_plugin_link_libs
  -Wl,--no-as-needed
  graph
  register
  alog
  error_manager
  te_fusion
  _caffe_parser
  ascend_protobuf
  -Wl,--as-needed
  c_sec
)


if(ENABLE_TEST STREQUAL "")
  add_library(ops_all_onnx_plugin SHARED
    ${_onnx_plugins}
    ${_proto_h}
  )

  target_include_directories(ops_all_onnx_plugin PRIVATE
    ${_ops_all_onnx_plugin_include}
  )

  target_compile_options(ops_all_onnx_plugin PRIVATE
    "-fno-common"
    "-fno-strict-aliasing"
    -O2
    -Werror
    -Wno-deprecated-declarations
    -Dgoogle=ascend_private
  )

  target_link_libraries(ops_all_onnx_plugin PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:$<BUILD_INTERFACE:slog_headers>>
    ${_ops_all_onnx_plugin_link_libs}
  )

  if(BUILD_OPEN_PROJECT)
    set(ONNX_PLUGIN_PATH "${INSTALL_PATH}/opp/framework/built-in/onnx")
    cann_install(
      TARGET      ops_all_onnx_plugin
      FILES       $<TARGET_FILE:ops_all_onnx_plugin>
      DESTINATION "${ONNX_PLUGIN_PATH}"
    )
  else()
    install(
      TARGETS ops_all_onnx_plugin OPTIONAL
      EXPORT  ops_all_onnx_plugin-targets
      LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
    )
  endif()
elseif(UT_TEST_ALL OR ONNX_PLUGIN_UT)
  add_library(ops_all_onnx_plugin_llt STATIC
    ${_onnx_plugins}
    ${_proto_h}
    )
  target_include_directories(ops_all_onnx_plugin_llt PUBLIC
    ${_ops_all_onnx_plugin_include}
    )
  if(NOT ${CMAKE_BUILD_MODE} STREQUAL "FALSE")
   set(compile_opt_mode ${CMAKE_BUILD_MODE})
  else()
   set(compile_opt_mode -O0)
  endif()
  target_compile_options(ops_all_onnx_plugin_llt PUBLIC
    ${compile_opt_mode}
    -Dgoogle=ascend_private
    )
  target_link_libraries(ops_all_onnx_plugin_llt
    PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    PUBLIC
    ${_ops_all_onnx_plugin_link_libs}
    )
endif()
