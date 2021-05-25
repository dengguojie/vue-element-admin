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
  "${METADEF_DIR}/proto/insert_op.proto"
)

# the output path of generated files
set(_proto_include "${CMAKE_CURRENT_BINARY_DIR}/proto_tmp")
protobuf_generate(opp _proto_cc _proto_h ${_proto_srcs})

set(_op_proto_utils
  util/array_ops_shape_fns.cc
  util/axis_util.cc
  util/candidate_sampling_shape_fns.cc
  util/common_shape_fns.cc
  util/error_util.cc
  util/images_ops_shape_fns.cc
  util/linalg_ops_shape_fns.cc
  util/lookup_ops_shape_fns.cc
  util/nn_shape_fns.cc
  util/ragged_conversion_ops_shape_fns.cc
  util/random_ops_shape_fns.cc
  util/resource_variable_ops_shape_fns.cc
  util/transfer_shape_according_to_format.cc
  util/util.cc
)

set(_op_proto_srcs
  acts_ulq.cc
  acts_ulq_input_grad.cc
  act_ulq_clamp_max_grad.cc
  act_ulq_clamp_min_grad.cc
  aipp.cc
  array_ops.cc
  audio_ops.cc
  batch_ops.cc
  bitwise_ops.cc
  boosted_trees_ops.cc
  candidate_sampling_ops.cc
  condtake_ops.cc
  control_flow_ops.cc
  copy_ops.cc
  ctc_ops.cc
  data_flow_ops.cc
  elewise_calculation_ops.cc
  format_transfer_fractal_z.cc
  fsrdetectionoutput_ops.cc
  functional_ops.cc
  hcom_ops.cc
  hvd_ops.cc
  ifmr.cc
  image_ops.cc
  internal_ops.cc
  linalg_ops.cc
  list_ops.cc
  logging_ops.cc
  lookup_ops.cc
  math_ops.cc
  matrix_calculation_ops.cc
  nn_batch_norm_ops.cc
  nn_calculation_ops.cc
  nn_detect_ops.cc
  nn_norm_ops.cc
  nn_ops.cc
  nn_other_ops.cc
  nn_pooling_ops.cc
  nn_training_ops.cc
  nonlinear_fuc_ops.cc
  no_op.cc
  npu_loss_scale_ops.cc
  outfeed_ops.cc
  pad_ops.cc
  parsing_ops.cc
  psroipooling_ops.cc
  quantize_ops.cc
  ragged_array_ops.cc
  ragged_conversion_ops.cc
  ragged_math_ops.cc
  random_ops.cc
  reduce_ops.cc
  reduction_ops.cc
  resource_variable_ops.cc
  rnn.cc
  roipooling_ops.cc
  rpn_ops.cc
  sdca_ops.cc
  selection_ops.cc
  set_ops.cc
  sparse_ops.cc
  spectral_ops.cc
  split_combination_ops.cc
  ssddetectionoutput_ops.cc
  stateful_random_ops.cc
  stateless_random_ops.cc
  state_ops.cc
  string_ops.cc
  target_crop_and_resize.cc
  transformation_ops.cc
  warp_perspective_ops.cc
  wts_arq.cc
  avg_pool_1d_ops.cc
  correlation.cc
  globalavgpool.cc
)

set(_op_proto_include
  util
  inc
  ${OPS_DIR}
  ${OPS_INCLUDE}
  ${FWKACLLIB_INCLUDE}
  ${METADEF_DIR}
  ${METADEF_INCLUDE}
  ${METADEF_INCLUDE}/external
  ${METADEF_INCLUDE}/external/graph
  ${GRAPHENGINE_INCLUDE}
  ${GRAPHENGINE_INCLUDE}/external
  ${PROTO_BINARY_DIR}/opp
  ${Protobuf_INCLUDE}
)

set(_op_proto_link_libs
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
  add_library(opsproto SHARED
    ${_op_proto_srcs}
    ${_op_proto_utils}
    ${_proto_h}
  )
  target_include_directories(opsproto PRIVATE
    ${_op_proto_include}
  )
  target_compile_options(opsproto PRIVATE
    -O2
    -Dgoogle=ascend_private
    # TODO: Fix warnings
    # -Werror
  )
  target_link_libraries(opsproto PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    ${_op_proto_link_libs}
  )

  if(BUILD_OPEN_PROJECT)
    set(OPS_PROTO_PATH "${INSTALL_PATH}/opp/op_proto/built-in")
    cann_install(
      TARGET      opsproto
      FILES       $<TARGET_FILE:opsproto>
      DESTINATION "${OPS_PROTO_PATH}"
    )

    add_custom_target(copy_op_proto_inc ALL)
    cann_install(
      TARGET      copy_op_proto_inc
      DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/inc
      DESTINATION "${OPS_PROTO_PATH}"
    )
  else()
    install(
      TARGETS opsproto OPTIONAL
      EXPORT  opsproto-targets
      LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
    )
  endif()
elseif(UT_TEST_ALL OR PROTO_UT OR PASS_UT OR PLUGIN_UT OR ONNX_PLUGIN_UT)
  add_library(opsproto_llt STATIC
    ${_op_proto_srcs}
    ${_op_proto_utils}
    ${_proto_h}
  )
  target_include_directories(opsproto_llt PUBLIC
    ${_op_proto_include}
  )
  target_compile_options(opsproto_llt PUBLIC
    -O0
    -Dgoogle=ascend_private
  )
  target_link_libraries(opsproto_llt
    PRIVATE
      $<BUILD_INTERFACE:intf_llt_pub>
    PUBLIC
      ${_op_proto_link_libs}
  )
endif()
