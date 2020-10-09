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

set(PROTO_OUTPATH ${CMAKE_CURRENT_BINARY_DIR}/proto_tmp)

file(GLOB_RECURSE _proto_files "proto/*.proto")
set(_valid_proto_files "")
foreach(_file ${_proto_files})
  if(EXISTS "${_file}")
    list(APPEND _valid_proto_files "${_file}")
  endif() 
endforeach()

if(_valid_proto_files)
  set(_valid_proto_targets "")
  foreach(_file ${_valid_proto_files})
    protobuf_generate_cc("${_file}" "${PROTO_OUTPATH}")
  endforeach()
endif()

# set(CAFFE_PROTO_PATH "${PROTO_OUTPATH}/proto/caffe")
# add_custom_target(generate_caffe_proto_files ALL
#   DEPENDS "${CAFFE_PROTO_PATH}/caffe.pb.h" "${CAFFE_PROTO_PATH}/caffe.pb.cc"
#   COMMENT "Generate caffe proto targets."
# )
# 
# set(TF_PROTO_PATH "${PROTO_OUTPATH}/proto/tensorflow")
# add_custom_target(generate_tf_proto_files ALL
#   DEPENDS "${TF_PROTO_PATH}/attr_value.pb.h" "${TF_PROTO_PATH}/attr_value.pb.cc"
#           "${TF_PROTO_PATH}/node_def.pb.h"   "${TF_PROTO_PATH}/node_def.pb.cc"
#           "${TF_PROTO_PATH}/types.pb.h"      "${TF_PROTO_PATH}/types.pb.cc"
# )
#
# set(ONNX_PROTO_PATH "${PROTO_OUTPATH}/proto/onnx")
# add_custom_target(generate_onnx_proto_files ALL
#   DEPENDS "${ONNX_PROTO_PATH}/ge_onnx.pb.h" "${ONNX_PROTO_PATH}/ge_onnx.pb.cc"
# )

add_subdirectory(framework)

add_subdirectory(fusion_pass)

add_subdirectory(fusion_rules)

add_subdirectory(op_proto)

add_subdirectory(tbe)
