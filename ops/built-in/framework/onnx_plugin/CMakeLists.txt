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
  "${CANN_ROOT}/third_party/metadef/inc/register/proto/onnx/ge_onnx.proto"
)

# the output path of generated files
set(_proto_include "${CMAKE_CURRENT_BINARY_DIR}/proto_tmp")
protobuf_generate_cc(_proto_h _proto_cc ${_proto_include} ${_proto_srcs})

# the list of source files
set(_onnx_plugins
 abs_plugin.cc
 add_plugin.cc
 batch_normalization_plugin.cc
 concat_plugin.cc
 conv2d_plugin.cc
 div_plugin.cc
 gather_plugin.cc
 gemm_plugin.cc
 global_average_pool_plugin.cc
 max_pool_plugin.cc
 mul_plugin.cc
 reduce_mean_plugin.cc
 reduce_sum_plugin.cc
 relu_plugin.cc
 reshape_plugin.cc
 shape_plugin.cc
 sigmoid_plugin.cc
 sqrt_plugin.cc
 sub_plugin.cc
 unsqueeze_plugin.cc
)

add_library(ops_all_onnx_plugin SHARED
  ${_onnx_plugins}
  ${_proto_cc}
)

target_include_directories(ops_all_onnx_plugin PRIVATE
  ${CANN_ROOT}/ops/common/inc
  ${CANN_ROOT}/inc
  ${CANN_ROOT}/inc/external
  ${CANN_ROOT}/third_party/fwkacllib/inc
  ${_proto_include}
  ${Protobuf_INCLUDE}
)

target_link_libraries(ops_all_onnx_plugin PRIVATE
  ${graph}
  ${register}
  ${slog}
  ${error_manager}
  ${te_fusion}
  ${_caffe_parser}
  ${protobuf}
  c_sec
)

set(ONNX_PLUGIN_PATH "${INSTALL_PATH}/opp/framework/built-in/onnx")
cann_install(
  TARGET      ops_all_onnx_plugin
  FILES       $<TARGET_FILE:ops_all_onnx_plugin>
  DESTINATION "${ONNX_PLUGIN_PATH}"
)
