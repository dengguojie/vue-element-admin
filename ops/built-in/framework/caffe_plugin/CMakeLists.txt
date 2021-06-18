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
if(NOT ENABLE_TEST)
  set(_proto_srcs
    "${METADEF_DIR}/proto/caffe/caffe.proto"
  )

  protobuf_generate(opp _proto_cc _proto_h ${_proto_srcs})

  set(_caffe_plugins
    ../../op_proto/util/error_util.cc
    arg_max_with_k_plugin.cc
    basic_rnn_cell_plugin.cc
    bias_plugin.cc
    bnll_plugin.cc
    caffe_absval_plugin.cc
    caffe_ascend_anti_quant_plugin.cc
    caffe_ascend_dequant_plugin.cc
    caffe_ascend_dequant_s16_plugin.cc
    caffe_ascend_quant_plugin.cc
    caffe_ascend_requant_plugin.cc
    caffe_ascend_requant_s16_plugin.cc
    caffe_batch_matmul_plugin.cc
    caffe_bninference_plugin.cc
    caffe_condtake_plugin.cc
    caffe_conv2d_plugin.cc
    caffe_deconv_plugin.cc
    caffe_exp_plugin.cc
    caffe_leaky_relu_plugin.cc
    caffe_log_plugin.cc
    caffe_mish_plugin.cc
    caffe_permute_plugin.cc
    caffe_prelu_plugin.cc
    caffe_prior_box_plugin.cc
    caffe_proposal_plugin.cc
    caffe_psroipooling_plugin.cc
    caffe_reduction_plugin.cc
    caffe_tile_with_axis_plugin.cc
    caffe_warp_perspective_plugin.cc
    caffe_yolov2_detection_output_d_plugin.cc
    caffe_yolov3_detection_output_d_plugin.cc
    caffe_yolov3_detection_output_v2_d_plugin.cc
    case_plugin.cc
    concat_plugin.cc
    copy_plugin.cc
    crop_plugin.cc
    eltwise_plugin.cc
    elu_plugin.cc
    flatten_v2_plugin.cc
    fsrdetectionoutput_plugin.cc
    if_plugin.cc
    inner_product_plugin.cc
    lrn_plugin.cc
    lstm_plugin.cc
    matrix_inverse_plugin.cc
    mvn_plugin.cc
    normalize_plugin.cc
    partitioned_call_plugin.cc
    pass_through_plugin.cc
    pooling_plugin.cc
    power_plugin.cc
    relu6_plugin.cc
    reverse_plugin.cc
    rnn_plugin.cc
    roi_align_plugin.cc
    roipooling_plugin.cc
    scale_plugin.cc
    shuffle_channel_plugin.cc
    sigmoid_plugin.cc
    slice_plugin.cc
    softmax_plugin.cc
    spatial_transformer_d_plugin.cc
    spp_plugin.cc
    ssddetectionoutput_plugin.cc
    tanh_plugin.cc
    threshold_plugin.cc
    upsample_plugin.cc
    while_plugin.cc
    yolo_plugin.cc
    correlation_plugin.cc
  )

  add_library(ops_all_caffe_plugin SHARED
    ${_caffe_plugins}
    ${_proto_h}
  )

  target_include_directories(ops_all_caffe_plugin PRIVATE
    ${OPS_INCLUDE}
    ${FWKACLLIB_INCLUDE}
    ${METADEF_INCLUDE}
    ${METADEF_INCLUDE}/external
    ${METADEF_INCLUDE}/external/graph
    ${GRAPHENGINE_INCLUDE}
    ${PROTO_BINARY_DIR}/opp
    ${Protobuf_INCLUDE}
  )

  target_compile_options(ops_all_caffe_plugin PRIVATE
    "-fno-common"
    "-fno-strict-aliasing"
    -O2
    -Werror
    -Wno-deprecated-declarations
    -Dgoogle=ascend_private
  )

  target_link_libraries(ops_all_caffe_plugin PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:$<BUILD_INTERFACE:slog_headers>>
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

  if(BUILD_OPEN_PROJECT)
    set(CAFFE_PLUGIN_PATH "${INSTALL_PATH}/opp/framework/built-in/caffe")
    cann_install(
      TARGET      ops_all_caffe_plugin
      FILES       $<TARGET_FILE:ops_all_caffe_plugin>
      DESTINATION "${CAFFE_PLUGIN_PATH}"
    )
  else()
    install(
      TARGETS ops_all_caffe_plugin OPTIONAL
      EXPORT  ops_all_caffe_plugin-targets
      LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
    )
  endif()
endif()
