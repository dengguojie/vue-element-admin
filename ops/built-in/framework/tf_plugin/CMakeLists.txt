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
set(_tf_plugins
  # file1
  # file2
)

set(_tf_scope_fusion_passes
  # ../tf_scope_fusion_passes/file1
  # ../tf_scope_fusion_passes/file2
)

if(_tf_plugins)

  add_library(ops_all_plugin SHARED ${_tf_plugin} ${_tf_scope_fusion_passes})

  # add_dependencies(ops_all_plugin generate_tf_proto_tiles)

  target_include_directories(ops_all_plugin PRIVATE
    ${CANN_ROOT}/ops/built-in/common/inc
    ${CANN_ROOT}/inc
    ${CANN_ROOT}/inc/external
    ${JSON_INCLUDE}
    ${Protobuf_INCLUDE}
    ${PROTO_OUTPATH}
  )

  target_link_libraries(ops_all_plugin PRIVATE
    ${graph}
    ${register}
    ${slog}
    ${error_manager}
    ${te_fusion}
    ${_caffe_parser}
    c_sec
    protobuf
  )

  set(TF_PLUGIN_PATH "${INSTALL_PATH}/opp/framework/built-in/tensorflow")
  cann_install(
    TARGET      ops_all_plugin
    FILES       $<TARGET_FILE:ops_all_plugin>
    DESTINATION "${TF_PLUGIN_PATH}"
  )

endif()
