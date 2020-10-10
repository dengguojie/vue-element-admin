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
set(_caffe_plugins
  # file1
  # file2
)

if(_caffe_plugins)

  add_library(ops_all_caffe_plugin SHARED ${_caffe_plugins})

  add_dependencies(ops_all_caffe_plugin generate_caffe_proto_files)

  target_include_directories(ops_all_caffe_plugin PRIVATE
    ${CANN_ROOT}/ops/built-in/common/inc
    ${CANN_ROOT}/inc
    ${CANN_ROOT}/inc/external
    ${JSON_INCLUDE}
    ${Protobuf_INCLUDE}
    ${PROTO_OUTPATH}
  )

  target_link_libraries(ops_all_caffe_plugin PRIVATE
    ${graph}
    ${register}
    ${slog}
    ${error_manager}
    ${te_fusion}
    ${_caffe_parser}
    ${protobuf}
    c_sec
  )

  set(CAFFE_PLUGIN_PATH "${INSTALL_PATH}/opp/framework/built-in/caffe")
  cann_install(
    TARGET      ops_all_caffe_plugin
    FILES       $<TARGET_FILE:ops_all_caffe_plugin>
    DESTINATION "${CAFFE_PLUGIN_PATH}"
  )

endif()
