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
  "${METADEF_INCLUDE}/register/proto/onnx/ge_onnx.proto"
)

protobuf_generate(opp _proto_cc _proto_h ${_proto_srcs})

# the list of source files
set(_onnx_plugins
  abs_plugin.cc
  add_plugin.cc
  ascend_anti_quant_plugin.cc
  ascend_dequant_plugin.cc
  ascend_dequant_s16_plugin.cc
  ascend_quant_plugin.cc
  ascend_requant_plugin.cc
  ascend_requant_s16_plugin.cc
  batch_normalization_plugin.cc
  concat_plugin.cc
  conv2d_plugin.cc
  div_plugin.cc
  gather_plugin.cc
  gemm_plugin.cc
  global_average_pool_plugin.cc
  max_pool_plugin.cc
  mul_plugin.cc
  pad_plugin.cc
  reduce_mean_plugin.cc
  reduce_sum_plugin.cc
  relu_plugin.cc
  reshape_plugin.cc
  shape_plugin.cc
  sigmoid_plugin.cc
  sqrt_plugin.cc
  sub_plugin.cc
  tile_plugin.cc
  topk_plugin.cc
  unsqueeze_plugin.cc
)

add_library(ops_all_onnx_plugin SHARED
  ${_onnx_plugins}
  ${_proto_h}
)

target_include_directories(ops_all_onnx_plugin PRIVATE
  ${OPS_INCLUDE}
  ${FWKACLLIB_INCLUDE}
  ${METADEF_INCLUDE}
  ${METADEF_INCLUDE}/external
  ${METADEF_INCLUDE}/external/graph
  ${GRAPHENGINE_INCLUDE}
  ${PROTO_BINARY_DIR}/opp
  ${Protobuf_INCLUDE}
)

target_compile_options(ops_all_onnx_plugin PRIVATE
  -O2
  -Werror
  -Dgoogle=ascend_private
)

target_link_libraries(ops_all_onnx_plugin PRIVATE
  $<BUILD_INTERFACE:intf_pub>
  -Wl,--no-as-needed
    graph
    register
    slog
    error_manager
    te_fusion
    _caffe_parser
    ascend_protobuf
  -Wl,--as-needed
    c_sec
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
