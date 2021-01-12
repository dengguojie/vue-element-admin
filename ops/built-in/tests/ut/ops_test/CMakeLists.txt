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

file(GLOB PLUGIN_PROTO_TC_FILES **/*proto.cpp)

if(BUILD_OPEN_PROJECT)
  add_executable(ops_cpp_proto_utest
    ${PLUGIN_PROTO_TC_FILES}
    ${OP_TEST_PLUFIN_AND_PROTO_SOURCE}
  )
  target_include_directories(ops_cpp_proto_utest PRIVATE
    ${GTEST_INCLUDE}
    ${OP_TEST_PLUFIN_AND_PROTO_INCLUDE}
    ${METADEF_INCLUDE}/graph
    ${METADEF_INCLUDE}/inc/external/graph
  )
  target_compile_definitions(ops_cpp_proto_utest PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
  )
  target_compile_options(ops_cpp_proto_utest PUBLIC
    -O0
    -Dgoogle=ascend_private
  )
  target_link_libraries(ops_cpp_proto_utest PRIVATE
    -Wl,--whole-archive
      opsproto_llt
    -Wl,--no-whole-archive
  )
  add_custom_command(
    TARGET  ops_cpp_proto_utest POST_BUILD
    COMMAND ops_cpp_proto_utest
    COMMENT "Run ops proto utest"
  )
elseif(OPS_UT_YELLOW)
  set(PROTO_LIST
    "${TOP_DIR}/metadef/proto/insert_op.proto"
  )
  protobuf_generate(ops PROTO_SRCS PROTO_HDRS ${PROTO_LIST})
  file(GLOB_RECURSE src_files "${TOP_DIR}/cann/ops/built-in/op_proto/*.cc")
  add_executable(ut_ops_cpp_proto
    ${TOP_DIR}/cann/ops/built-in/tests/common/utils_plugin_and_op_proto/op_proto_test_util.cpp
    ${TOP_DIR}/cann/ops/built-in/tests/ut/test_plugin_and_op_proto_main.cpp
    ${src_files}
    ${PROTO_SRCS}
    ${PLUGIN_PROTO_TC_FILES}
  )
  target_include_directories(ut_ops_cpp_proto PRIVATE
    ${TOP_DIR}/inc
    ${TOP_DIR}/inc/external
    ${TOP_DIR}/metadef/graph
    ${TOP_DIR}/common
    ${TOP_DIR}/metadef
    ${TOP_DIR}/metadef/inc
    ${TOP_DIR}/metadef/inc/graph
    ${TOP_DIR}/metadef/inc/external
    ${TOP_DIR}/metadef/inc/external/graph
    ${TOP_DIR}/graphengine/inc
    ${TOP_DIR}/graphengine/inc/external
    ${TOP_DIR}/graphengine/inc/external/ge
    ${TOP_DIR}/graphengine/inc/framework
    ${TOP_DIR}/cann/ops/common/inc
    ${TOP_DIR}/cann/ops/built-in/op_proto/inc
    ${TOP_DIR}/cann/ops/built-in/op_proto/util
    ${TOP_DIR}/cann/ops
    ${TOP_DIR}/cann/ops/proto
    ${TOP_DIR}/cann/ops/built-in/tests/stub/framework/domi
    ${TOP_DIR}/cann/ops/built-in/tests/common/src/inc
    ${TOP_DIR}/cann/ops/built-in/tests/common/utils_plugin_and_op_proto
    ${TOP_DIR}/lib_sec/include
    ${TOP_DIR}/third_party/protobuf/include
    ${TOP_DIR}/third_party/json/include
    ${CMAKE_BINARY_DIR}/proto/ops
  )
  target_compile_options(ut_ops_cpp_proto PRIVATE
    -g3
    -O0
  )
  target_compile_definitions(ut_ops_cpp_proto PRIVATE
    CFG_BUILD_DEBUG
    google=ascend_private
  )
  target_link_libraries(ut_ops_cpp_proto PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    -Wl,--no-as-needed
      c_sec
      ascend_protobuf
      graph
      error_manager
      register
      slog
    -Wl,--no-as-needed
    -Wl,--whole-archive
     slog_ops_stub
    -Wl,--no-whole-archive
  )
  run_llt_test(
    TARGET   ut_ops_cpp_proto
    TASK_NUM 1
  )
endif()
