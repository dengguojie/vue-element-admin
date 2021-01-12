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

file(GLOB FUSION_PASS_TC_FILES *.cpp)

if(BUILD_OPEN_PROJECT)
  add_executable(ops_cpp_fusion_pass_aicore_utest
    ${FUSION_PASS_TC_FILES}
    ${OP_TEST_FUSION_PASS_SOURCE}
  )
  target_include_directories(ops_cpp_fusion_pass_aicore_utest PRIVATE
    ${OPS_DIR}/built-in/op_proto/inc
    ${OPS_DIR}/built-in/fusion_pass
    ${OPS_DIR}/../third_party/fusion_engine/inc
    ${OP_TEST_FUSION_PASS_INCLUDE}
    ${GTEST_INCLUDE}
    ${OP_TEST_PLUFIN_AND_PROTO_INCLUDE}
  )
  target_compile_definitions(ops_cpp_fusion_pass_aicore_utest PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
  )
  target_compile_options(ops_cpp_fusion_pass_aicore_utest PUBLIC
    -O0
    -Dgoogle=ascend_private
  )
  target_link_libraries(ops_cpp_fusion_pass_aicore_utest PRIVATE
    -Wl,--whole-archive
      ops_fusion_pass_aicore_llt
      opsproto_llt
    -Wl,--no-whole-archive
    -Wl,--allow-multiple-definition
  )
  add_custom_command(
    TARGET  ops_cpp_fusion_pass_aicore_utest POST_BUILD
    COMMAND ops_cpp_fusion_pass_aicore_utest
    COMMENT "Run ops fusion_pass utest"
  )
elseif(OPS_UT_YELLOW)
  set(PROTO_LIST
    "${TOP_DIR}/metadef/proto/insert_op.proto"
    "${TOP_DIR}/metadef/proto/task.proto"
  )
  protobuf_generate(ops PROTO_SRCS PROTO_HDRS ${PROTO_LIST})
  file(GLOB_RECURSE graph_fusion_srcs "${TOP_DIR}/cann/ops/built-in/fusion_pass/graph_fusion/ai_core/*.cc")
  file(GLOB_RECURSE aipp_ub_fusion_srcs "${TOP_DIR}/cann/ops/built-in/fusion_pass/buffer_fusion/ub_fusion/ai_core/aipp_conv/*.cc")
  file(GLOB_RECURSE common_srcs "${TOP_DIR}/cann/ops/built-in/fusion_pass/common/*.cc")
  file(GLOB_RECURSE op_proto_srcs "${TOP_DIR}/cann/ops/built-in/op_proto/*.cc")
  file(GLOB_RECURSE third_fe_common_srcs "${TOP_DIR}/cann/third_party/fusion_engine/inc/common/*.cc")
  file(GLOB_RECURSE third_fe_utils_srcs "${TOP_DIR}/cann/third_party/fusion_engine/inc/utils/*.cc")
  set(src_files
    ${TOP_DIR}/cann/ops/built-in/tests/common/src/fusion_pass_utils/fusion_pass_test_utils.cpp
    ${graph_fusion_srcs}
    ${aipp_ub_fusion_srcs}
    ${third_fe_common_srcs}
    ${third_fe_utils_srcs}
    ${common_srcs}
    ${op_proto_srcs}
  )
  add_executable(ut_ops_cpp_graph_fusion_aicore
    ${src_files}
    ${PROTO_SRCS}
    ${FUSION_PASS_TC_FILES}
  )
  target_include_directories(ut_ops_cpp_graph_fusion_aicore PRIVATE
    ${TOP_DIR}/inc
    ${TOP_DIR}/inc/external
    ${TOP_DIR}/inc/fusion_engine
    ${TOP_DIR}/metadef/inc/register
    ${TOP_DIR}/graphengine/inc
    ${TOP_DIR}/graphengine/inc/external
    ${TOP_DIR}/graphengine/inc/framework
    ${TOP_DIR}/metadef
    ${TOP_DIR}/metadef/inc
    ${TOP_DIR}/metadef/inc/external
    ${TOP_DIR}/metadef/inc/external/graph
    ${TOP_DIR}/metadef/graph
    ${TOP_DIR}/common
    ${TOP_DIR}/lib_sec/include
    ${TOP_DIR}/third_party/protobuf/include
    ${TOP_DIR}/third_party/json/include
    ${TOP_DIR}/third_party/googltest/include
    ${TOP_DIR}/cann/ops/common/inc
    ${TOP_DIR}/cann/ops/built-in/op_proto/inc
    ${TOP_DIR}/cann/ops/built-in/op_proto
    ${TOP_DIR}/cann/ops/built-in/op_proto/util
    ${TOP_DIR}/cann/ops/common/inc
    ${TOP_DIR}/cann/ops
    ${TOP_DIR}/cann/ops/proto
    ${TOP_DIR}/cann/ops/built-in/fusion_pass
    ${TOP_DIR}/cann/ops/built-in/fusion_pass/common
    ${TOP_DIR}/cann/ops/built-in/tests/stub/fusion_engine
    ${TOP_DIR}/cann/ops/built-in/tests/common/src/inc
    ${TOP_DIR}/cann/third_party/fusion_engine/inc
    ${CMAKE_BINARY_DIR}/proto/ops
  )
  target_compile_options(ut_ops_cpp_graph_fusion_aicore PRIVATE
    -g3
    -O0
  )
  target_compile_definitions(ut_ops_cpp_graph_fusion_aicore PRIVATE
    CFG_BUILD_DEBUG
    google=ascend_private
  )
  target_link_libraries(ut_ops_cpp_graph_fusion_aicore PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    -Wl,--no-as-needed
      c_sec
      ascend_protobuf
      graph
      error_manager
      register
      slog
      platform
    -Wl,--no-as-needed
    -Wl,--whole-archive 
     slog_ops_stub
    -Wl,--no-whole-archive 
  )
  run_llt_test(
    TARGET   ut_ops_cpp_graph_fusion_aicore
    TASK_NUM 1
  )
endif()
